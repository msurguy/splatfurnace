import { quatAxes, type Splats } from './ply';

// Everything on the GPU lives in "grid units": origin at grid.min, one unit = one voxel.

export interface Grid {
  min: [number, number, number];
  voxel: number; // world size of one voxel
  dims: [number, number, number]; // grid points per axis
}

export interface GridOptions {
  resolution: number; // voxels along the longest axis
  opacityThreshold: number;
  outlierPercentile: number; // e.g. 0.5 -> bounds from 0.5..99.5 percentile of centres
  maxCells: number; // device-imposed cap on dims product
}

const PAD = 4; // empty voxels around the object so the exterior flood fill has room

/**
 * Bounds from percentiles of visible splat centres instead of min/max, so a few stray
 * floaters far away can't shrink the effective resolution of the actual object.
 */
export function computeGrid(s: Splats, o: GridOptions): Grid {
  const visible: number[] = [];
  for (let i = 0; i < s.count; i++) if (s.opacities[i] >= o.opacityThreshold) visible.push(i);
  if (!visible.length) throw new Error('No splats above the opacity threshold.');
  const lo = [0, 0, 0], hi = [0, 0, 0];
  const tmp = new Float32Array(visible.length);
  for (let a = 0; a < 3; a++) {
    for (let k = 0; k < visible.length; k++) tmp[k] = s.centers[visible[k] * 3 + a];
    tmp.sort();
    const q = o.outlierPercentile / 100;
    lo[a] = tmp[Math.floor(q * (tmp.length - 1))];
    hi[a] = tmp[Math.ceil((1 - q) * (tmp.length - 1))];
  }
  // Grow by a typical splat footprint so surfels on the bounding surface aren't clipped.
  const big = new Float32Array(visible.length);
  for (let k = 0; k < visible.length; k++) {
    const i = visible[k];
    big[k] = Math.max(s.scales[i * 3], s.scales[i * 3 + 1], s.scales[i * 3 + 2]);
  }
  big.sort();
  const margin = 3 * big[Math.floor(0.95 * (big.length - 1))];
  for (let a = 0; a < 3; a++) { lo[a] -= margin; hi[a] += margin; }

  const extent = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 1e-9);
  let voxel = extent / o.resolution;
  const dimsFor = (v: number) => [0, 1, 2].map((a) => Math.ceil((hi[a] - lo[a]) / v) + 1 + 2 * PAD) as [number, number, number];
  let dims = dimsFor(voxel);
  while (dims[0] * dims[1] * dims[2] > o.maxCells) {
    voxel *= 1.02;
    dims = dimsFor(voxel);
  }
  return { min: [lo[0] - PAD * voxel, lo[1] - PAD * voxel, lo[2] - PAD * voxel], voxel, dims };
}

/** Per splat 5 x vec4: [center(grid), opacity] [axis0, sigma0] [axis1, sigma1] [axis2, sigma2] [rgb, extent]. */
export const SPLAT_VEC4S = 5;

export function packSplats(s: Splats, g: Grid, opacityThreshold: number): { data: Float32Array; count: number; index: Uint32Array } {
  const keep: number[] = [];
  for (let i = 0; i < s.count; i++) if (s.opacities[i] >= opacityThreshold) keep.push(i);
  const data = new Float32Array(keep.length * SPLAT_VEC4S * 4);
  for (let k = 0; k < keep.length; k++) {
    const i = keep[k];
    const o = k * SPLAT_VEC4S * 4;
    data[o] = (s.centers[i * 3] - g.min[0]) / g.voxel;
    data[o + 1] = (s.centers[i * 3 + 1] - g.min[1]) / g.voxel;
    data[o + 2] = (s.centers[i * 3 + 2] - g.min[2]) / g.voxel;
    data[o + 3] = s.opacities[i];
    const axes = quatAxes(s.rotations[i * 4], s.rotations[i * 4 + 1], s.rotations[i * 4 + 2], s.rotations[i * 4 + 3]);
    let ex = 0, ey = 0, ez = 0;
    for (let a = 0; a < 3; a++) {
      const sigma = s.scales[i * 3 + a] / g.voxel;
      data.set([axes[a][0], axes[a][1], axes[a][2], sigma], o + 4 + a * 4);
      ex += (axes[a][0] * sigma) ** 2; ey += (axes[a][1] * sigma) ** 2; ez += (axes[a][2] * sigma) ** 2;
    }
    data[o + 16] = s.colors[i * 3];
    data[o + 17] = s.colors[i * 3 + 1];
    data[o + 18] = s.colors[i * 3 + 2];
    // 3-sigma half-extent along the largest world axis (voxels)
    data[o + 19] = 3 * Math.sqrt(Math.max(ex, ey, ez));
  }
  return { data, count: keep.length, index: Uint32Array.from(keep) };
}

export interface Buckets {
  size: number; // voxels per bucket edge
  dims: [number, number, number];
  starts: Uint32Array; // CSR offsets, length cells+1
  refs: Uint32Array; // packed splat indices
}

/**
 * Uniform CSR buckets over the grid; every splat is referenced from each bucket its 3-sigma
 * axis-aligned box touches. Giant splats (mostly background floaters) are skipped.
 */
export function buildBuckets(packed: Float32Array, count: number, g: Grid, size: number, maxExtent: number): Buckets {
  const dims = g.dims.map((d) => Math.ceil(d / size)) as [number, number, number];
  const cells = dims[0] * dims[1] * dims[2];
  const counts = new Uint32Array(cells + 1);
  const ranges = new Int32Array(count * 6);
  for (let k = 0; k < count; k++) {
    const o = k * SPLAT_VEC4S * 4;
    let skip = false;
    for (let a = 0; a < 3; a++) {
      let e = 0;
      for (let j = 0; j < 3; j++) e += (packed[o + 4 + j * 4 + a] * packed[o + 4 + j * 4 + 3]) ** 2;
      e = 3 * Math.sqrt(e);
      if (e > maxExtent) skip = true;
      ranges[k * 6 + a] = Math.max(0, Math.floor((packed[o + a] - e) / size));
      ranges[k * 6 + 3 + a] = Math.min(dims[a] - 1, Math.floor((packed[o + a] + e) / size));
    }
    if (skip) { ranges[k * 6] = 1; ranges[k * 6 + 3] = 0; continue; }
    forRange(ranges, k, dims, (c) => counts[c]++);
  }
  const starts = new Uint32Array(cells + 1);
  for (let c = 0; c < cells; c++) starts[c + 1] = starts[c] + counts[c];
  const fill = starts.slice(0, cells);
  const refs = new Uint32Array(starts[cells]);
  for (let k = 0; k < count; k++) forRange(ranges, k, dims, (c) => { refs[fill[c]++] = k; });
  return { size, dims, starts, refs };
}

function forRange(r: Int32Array, k: number, dims: number[], fn: (cell: number) => void) {
  for (let z = r[k * 6 + 2]; z <= r[k * 6 + 5]; z++)
    for (let y = r[k * 6 + 1]; y <= r[k * 6 + 4]; y++)
      for (let x = r[k * 6]; x <= r[k * 6 + 3]; x++) fn(x + dims[0] * (y + dims[1] * z));
}
