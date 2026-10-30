import type { Splats } from './ply';
import { buildBuckets, computeGrid, packSplats, type Grid } from './grid';
import { createFieldSampler, type Gpu } from './gpu/gpu';
import { extractSurface } from './gpu/surface';
import { computeNormals, removeIslands, smoothScalar, taubinSmooth, vertexNeighbors, type Mesh } from './mesh';
import { bakeTextures, buildAtlas, simplify, uvCoverage, type Atlased, type BakeResult } from './bake';
import { buildChartAtlas, type UnwrapRunner } from './unwrap';

export interface Settings {
  resolution: number;
  iso: number;
  minSigma: number;
  opacityThreshold: number;
  fillCavities: boolean;
  islandFraction: number; // 0..1
  smoothing: number; // Taubin iterations before snapping
  snap: boolean;
  maxSnap: number; // voxels
  targetTriangles: number;
  textureSize: number;
  normalMap: boolean;
  chartAtlas: boolean; // xatlas charts (shared vertices) instead of one cell per triangle pair
}

export const DEFAULTS: Settings = {
  resolution: 320,
  iso: 0.2,
  minSigma: 0.6,
  // About 1/255, the renderer's own cutoff. Splats only count in proportion to their opacity, so
  // faint ones matter only where they stack up, which is exactly where captures use them as
  // the visible surface (the idol's dark jacket back is made of 0.01-0.05 splats).
  opacityThreshold: 0.004,
  fillCavities: true,
  islandFraction: 0.01,
  smoothing: 3,
  snap: true,
  maxSnap: 3,
  targetTriangles: 30000,
  textureSize: 2048,
  normalMap: true,
  chartAtlas: true,
};

/** Platform services the pipeline needs: where xatlas runs (browser worker or in-thread in Node). */
export interface Env {
  unwrap: UnwrapRunner;
}

export interface Result {
  grid: Grid;
  dense: Mesh; // world units, vertex colours
  final: Atlased; // world units
  bake: BakeResult;
  stats: Record<string, number | string>;
}

const BUCKET = 4; // voxels per spatial bucket edge
const MAX_EXTENT = 48; // voxels; larger splats are treated as background
const BOUNDS_OPACITY = 0.05; // grid bounds come from clearly visible splats, so faint haze can't stretch them

export async function convert(gpu: Gpu, splats: Splats, s: Settings, log: (m: string) => void, env: Env): Promise<Result> {
  const stats: Record<string, number | string> = {};
  const t0 = performance.now();
  let tStage = t0;
  const lap = (name: string) => {
    const now = performance.now();
    stats[`ms_${name}`] = Math.round(now - tStage);
    tStage = now;
  };
  const tick = () => new Promise((r) => setTimeout(r, 0));

  const grid = computeGrid(splats, { resolution: s.resolution, opacityThreshold: Math.max(s.opacityThreshold, BOUNDS_OPACITY), outlierPercentile: 0.2, maxCells: gpu.maxBinding / 4 });
  const packed = packSplats(splats, grid, s.opacityThreshold);
  const buckets = buildBuckets(packed.data, packed.count, grid, BUCKET, MAX_EXTENT);
  stats.splats = packed.count;
  stats.grid = grid.dims.join('x');
  lap('prepare');

  const raw = await extractSurface(gpu, grid, packed.data, packed.count, {
    iso: s.iso, minSigma: s.minSigma, support: 3, maxExtent: MAX_EXTENT, fillCavities: s.fillCavities,
  }, log);
  stats.floodPasses = raw.floodPasses;
  stats.rawTriangles = raw.indices.length / 3;
  lap('surface');

  log('Removing floating islands');
  await tick();
  const isl = removeIslands({ positions: raw.positions, normals: raw.normals, indices: raw.indices }, s.islandFraction);
  const mesh = isl.mesh;
  stats.islandsRemoved = isl.removed;
  stats.components = isl.kept;
  const ring = vertexNeighbors(mesh.positions.length / 3, mesh.indices);
  computeNormals(mesh.positions, mesh.indices, mesh.normals);
  log(`Smoothing ${(mesh.indices.length / 3).toLocaleString()} triangles`);
  await tick();
  taubinSmooth(mesh, ring, s.smoothing, 0.45);
  lap('cleanup');

  const sampler = await createFieldSampler(gpu, packed.data, buckets);
  try {
    if (s.snap) {
      log('Snapping surface to where the splats become opaque');
      const r = await sampler.sample(mesh.positions, mesh.normals, s.maxSnap + 2);
      const n = mesh.positions.length / 3;
      const delta = new Float32Array(n), conf = new Float32Array(n);
      let snapped = 0, sum = 0;
      for (let v = 0; v < n; v++) {
        const off = r.offsets[v];
        if (Number.isFinite(off) && r.coverage[v] > 0.5 && Math.abs(off) <= s.maxSnap) {
          delta[v] = off;
          conf[v] = r.coverage[v];
          snapped++;
          sum += off;
        }
      }
      // Smooth the displacement field itself so per-vertex noise doesn't become bumps.
      const smoothDelta = smoothScalar(delta, conf, ring, 4);
      for (let v = 0; v < n; v++) {
        for (let k = 0; k < 3; k++) mesh.positions[v * 3 + k] += mesh.normals[v * 3 + k] * smoothDelta[v];
      }
      computeNormals(mesh.positions, mesh.indices, mesh.normals);
      taubinSmooth(mesh, ring, 1, 0.3);
      stats.snappedFraction = +(snapped / n).toFixed(3);
      stats.meanSnapVoxels = +(sum / Math.max(1, snapped)).toFixed(3);
      lap('snap');
    }

    log('Ray-sampling vertex colours');
    const vc = await sampler.sample(mesh.positions, mesh.normals, 3);
    mesh.colors = fillColors(vc.colors, vc.coverage, ring);
    lap('vertexColors');

    log(`Simplifying to ${s.targetTriangles.toLocaleString()} triangles`);
    await tick();
    const simp = await simplify(mesh, s.targetTriangles);
    stats.denseTriangles = mesh.indices.length / 3;
    stats.finalTriangles = simp.mesh.indices.length / 3;
    stats.simplifyErrorVoxels = +simp.error.toFixed(3);
    lap('simplify');

    log('Building texture atlas');
    await tick();
    let atlas: Atlased | null = null;
    if (s.chartAtlas) {
      try {
        const charts = await buildChartAtlas(simp.mesh, s.textureSize, env.unwrap);
        stats.atlas = `charts: ${charts.charts}`;
        atlas = charts;
      } catch (err) {
        log(`Chart atlas failed (${(err as Error).message}); using per-triangle cells`);
      }
    }
    if (!atlas) {
      const cells = buildAtlas(simp.mesh, s.textureSize);
      stats.atlas = `triangle cells: ${cells.cell} texels`;
      atlas = cells;
    }
    stats.atlasCoverage = uvCoverage(atlas);
    lap('atlas');
    const range = Math.min(16, Math.max(3, 2 * simp.error + 2));
    stats.bakeRangeVoxels = +range.toFixed(2);
    log('Baking textures');
    const bake = await bakeTextures(atlas, sampler, { size: s.textureSize, range, normalMap: s.normalMap }, log);
    stats.texels = bake.texels;
    stats.texelMissRate = +(bake.misses / Math.max(1, bake.texels)).toFixed(4);
    lap('bake');

    stats.ms_total = Math.round(performance.now() - t0);
    return { grid, dense: toWorld(mesh, grid), final: atlasToWorld(atlas, grid), bake, stats };
  } finally {
    sampler.dispose();
  }
}

/** Vertices whose ray hit nothing borrow colour from their neighbours. */
function fillColors(colors: Float32Array, coverage: Float32Array, ring: ReturnType<typeof vertexNeighbors>) {
  const n = coverage.length;
  const out = colors.slice();
  const known = new Uint8Array(n);
  for (let v = 0; v < n; v++) known[v] = colors[v * 3] >= 0 && coverage[v] > 0.05 ? 1 : 0;
  for (let pass = 0; pass < 64; pass++) {
    let changed = 0;
    for (let v = 0; v < n; v++) {
      if (known[v]) continue;
      let r = 0, g = 0, b = 0, c = 0;
      for (let k = ring.offsets[v]; k < ring.offsets[v + 1]; k++) {
        const u = ring.list[k];
        if (known[u] !== 1) continue;
        r += out[u * 3]; g += out[u * 3 + 1]; b += out[u * 3 + 2]; c++;
      }
      if (c) { out[v * 3] = r / c; out[v * 3 + 1] = g / c; out[v * 3 + 2] = b / c; known[v] = 2; changed++; }
    }
    for (let v = 0; v < n; v++) if (known[v] === 2) known[v] = 1;
    if (!changed) break;
  }
  for (let v = 0; v < n; v++) if (!known[v]) out[v * 3] = out[v * 3 + 1] = out[v * 3 + 2] = 0.5;
  return out;
}

function positionsToWorld(p: Float32Array, g: Grid) {
  const out = new Float32Array(p.length);
  for (let i = 0; i < p.length; i += 3) {
    out[i] = g.min[0] + p[i] * g.voxel;
    out[i + 1] = g.min[1] + p[i + 1] * g.voxel;
    out[i + 2] = g.min[2] + p[i + 2] * g.voxel;
  }
  return out;
}

function toWorld(m: Mesh, g: Grid): Mesh {
  return { ...m, positions: positionsToWorld(m.positions, g) };
}

function atlasToWorld(a: Atlased, g: Grid): Atlased {
  return { ...a, positions: positionsToWorld(a.positions, g) };
}
