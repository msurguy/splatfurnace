// Typed wrapper over the xatlas WebAssembly build (third_party/xatlas-wasm/binding.cpp).
// Works in a worker, the main thread or Node; src/xatlas/worker.ts runs it off the main thread.
import createXAtlas from './xatlas-wasm.mjs';

export interface ChartOptions {
  normalDeviationWeight: number;
  roundnessWeight: number;
  straightnessWeight: number;
  normalSeamWeight: number;
  textureSeamWeight: number;
  maxCost: number;
  maxIterations: number;
  maxChartArea: number;
  maxBoundaryLength: number;
}

export interface PackOptions {
  resolution: number;
  padding: number;
  texelsPerUnit: number;
  bilinear: boolean;
  blockAlign: boolean;
  bruteForce: boolean;
  maxChartSize: number;
  rotateChartsToAxis: boolean;
  rotateCharts: boolean;
}

// xatlas defaults
export const CHART_DEFAULTS: ChartOptions = {
  normalDeviationWeight: 2, roundnessWeight: 0.01, straightnessWeight: 6, normalSeamWeight: 4,
  textureSeamWeight: 0.5, maxCost: 2, maxIterations: 1, maxChartArea: 0, maxBoundaryLength: 0,
};
export const PACK_DEFAULTS: PackOptions = {
  resolution: 0, padding: 0, texelsPerUnit: 0, bilinear: true, blockAlign: false, bruteForce: false,
  maxChartSize: 0, rotateChartsToAxis: true, rotateCharts: true,
};

export interface AtlasInfo { width: number; height: number; atlasCount: number; chartCount: number; texelsPerUnit: number }

export interface AtlasMesh {
  xref: Uint32Array; // input vertex of each output vertex
  uv: Float32Array; // texels, 0..width / 0..height
  atlasIndex: Int32Array; // -1: vertex not in any atlas
  indices: Uint32Array;
  chartCount: number;
}

const ADD_MESH_ERRORS = ['Success', 'Error', 'IndexOutOfRange', 'InvalidFaceVertexCount', 'InvalidIndexCount'];

interface Module {
  HEAPU8: Uint8Array;
  _malloc(n: number): number;
  _free(p: number): void;
  _xa_create(): void;
  _xa_destroy(): void;
  _xa_add_mesh(pos: number, nrm: number, vertexCount: number, idx: number, indexCount: number): number;
  _xa_add_uv_mesh(uv: number, vertexCount: number, idx: number, indexCount: number): number;
  _xa_compute_charts(opts: number): void;
  _xa_pack_charts(opts: number): void;
  _xa_atlas_info(out: number): void;
  _xa_mesh_info(mesh: number, out: number): void;
  _xa_mesh_copy(mesh: number, xref: number, uv: number, atlasIndex: number, idx: number): void;
}

export class XAtlas {
  private m: Module;
  private constructor(m: Module) { this.m = m; }

  /** `wasmUrl` overrides where the .wasm file is fetched from (bundlers rename it). */
  static async load(wasmUrl?: string): Promise<XAtlas> {
    const m = (await createXAtlas(wasmUrl ? { locateFile: () => wasmUrl } : {})) as Module;
    return new XAtlas(m);
  }

  // The heap can grow (and move) during any call into xatlas, so views are made per access.
  private put(a: Float32Array | Uint32Array): number {
    const p = this.m._malloc(a.byteLength);
    this.m.HEAPU8.set(new Uint8Array(a.buffer, a.byteOffset, a.byteLength), p);
    return p;
  }

  private withFloats<T>(values: number[], f: (p: number) => T): T {
    const p = this.put(Float32Array.from(values));
    try { return f(p); } finally { this.m._free(p); }
  }

  create() { this.m._xa_create(); }
  destroy() { this.m._xa_destroy(); }

  addMesh(positions: Float32Array, normals: Float32Array | null, indices: Uint32Array) {
    const pp = this.put(positions), pn = normals ? this.put(normals) : 0, pi = this.put(indices);
    try {
      const e = this.m._xa_add_mesh(pp, pn, positions.length / 3, pi, indices.length);
      if (e) throw new Error(`xatlas AddMesh: ${ADD_MESH_ERRORS[e] ?? e}`);
    } finally {
      this.m._free(pp); if (pn) this.m._free(pn); this.m._free(pi);
    }
  }

  /** A mesh whose UV islands are already charts (packing only). */
  addUvMesh(uvs: Float32Array, indices: Uint32Array) {
    const pu = this.put(uvs), pi = this.put(indices);
    try {
      const e = this.m._xa_add_uv_mesh(pu, uvs.length / 2, pi, indices.length);
      if (e) throw new Error(`xatlas AddUvMesh: ${ADD_MESH_ERRORS[e] ?? e}`);
    } finally {
      this.m._free(pu); this.m._free(pi);
    }
  }

  computeCharts(o: Partial<ChartOptions> = {}) {
    const c = { ...CHART_DEFAULTS, ...o };
    this.withFloats([c.normalDeviationWeight, c.roundnessWeight, c.straightnessWeight, c.normalSeamWeight,
      c.textureSeamWeight, c.maxCost, c.maxIterations, c.maxChartArea, c.maxBoundaryLength], (p) => this.m._xa_compute_charts(p));
  }

  packCharts(o: Partial<PackOptions> = {}) {
    const k = { ...PACK_DEFAULTS, ...o };
    this.withFloats([k.resolution, k.padding, k.texelsPerUnit, +k.bilinear, +k.blockAlign, +k.bruteForce,
      k.maxChartSize, +k.rotateChartsToAxis, +k.rotateCharts], (p) => this.m._xa_pack_charts(p));
  }

  info(): AtlasInfo {
    const p = this.m._malloc(24);
    try {
      this.m._xa_atlas_info(p);
      const u = new Uint32Array(this.m.HEAPU8.buffer, p, 6);
      return { width: u[0], height: u[1], atlasCount: u[2], chartCount: u[3], texelsPerUnit: new Float32Array(this.m.HEAPU8.buffer, p + 20, 1)[0] };
    } finally { this.m._free(p); }
  }

  mesh(index: number): AtlasMesh {
    const pinfo = this.m._malloc(12);
    let vertexCount: number, indexCount: number, chartCount: number;
    try {
      this.m._xa_mesh_info(index, pinfo);
      [vertexCount, indexCount, chartCount] = new Uint32Array(this.m.HEAPU8.buffer, pinfo, 3);
    } finally { this.m._free(pinfo); }
    const px = this.m._malloc(vertexCount * 4), pu = this.m._malloc(vertexCount * 8);
    const pa = this.m._malloc(vertexCount * 4), pi = this.m._malloc(indexCount * 4);
    try {
      this.m._xa_mesh_copy(index, px, pu, pa, pi);
      const b = this.m.HEAPU8.buffer;
      return {
        xref: new Uint32Array(b, px, vertexCount).slice(),
        uv: new Float32Array(b, pu, vertexCount * 2).slice(),
        atlasIndex: new Int32Array(b, pa, vertexCount).slice(),
        indices: new Uint32Array(b, pi, indexCount).slice(),
        chartCount,
      };
    } finally {
      for (const p of [px, pu, pa, pi]) this.m._free(p);
    }
  }
}

export interface UnwrapRequest {
  positions: Float32Array;
  normals: Float32Array | null;
  indices: Uint32Array;
  size: number; // texture size the atlas must fit
  padding: number; // texels between charts
  chunkTriangles?: number; // chart spatial chunks of about this many triangles (default 4000)
  chart?: Partial<ChartOptions>;
}

export interface UnwrapResult {
  xref: Uint32Array; // input vertex of each output vertex
  uv: Float32Array; // 0..1 of the size x size texture
  indices: Uint32Array; // output vertices, same triangle order as the input
  charts: number;
  chunks: number;
  width: number; // atlas size xatlas packed (before scaling to fit)
  height: number;
  scale: number; // uniform UV scale applied to fit (1 = none)
  ms: { charts: number; pack: number };
}

/** Split triangles into spatially coherent chunks (Morton order of centroids). */
function mortonChunks(positions: Float32Array, indices: Uint32Array, perChunk: number): Uint32Array[] {
  const T = indices.length / 3;
  const n = Math.max(1, Math.round(T / perChunk));
  if (n === 1) return [Uint32Array.from({ length: T }, (_, t) => t)];
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < positions.length; i++) { lo[i % 3] = Math.min(lo[i % 3], positions[i]); hi[i % 3] = Math.max(hi[i % 3], positions[i]); }
  const spread = (x: number) => { // 10 bits -> every third bit
    x = (x | (x << 16)) & 0x030000ff; x = (x | (x << 8)) & 0x0300f00f; x = (x | (x << 4)) & 0x030c30c3; return (x | (x << 2)) & 0x09249249;
  };
  const key = new Uint32Array(T);
  for (let t = 0; t < T; t++) {
    let k = 0;
    for (let a = 0; a < 3; a++) {
      const c = (positions[indices[t * 3] * 3 + a] + positions[indices[t * 3 + 1] * 3 + a] + positions[indices[t * 3 + 2] * 3 + a]) / 3;
      k |= spread(Math.min(1023, Math.floor(((c - lo[a]) / Math.max(1e-9, hi[a] - lo[a])) * 1024))) << a;
    }
    key[t] = k;
  }
  const order = Uint32Array.from({ length: T }, (_, t) => t).sort((a, b) => key[a] - key[b]);
  const per = Math.ceil(T / n);
  return Array.from({ length: n }, (_, i) => order.subarray(i * per, Math.min(T, (i + 1) * per)));
}

/**
 * Chart and pack one mesh into a single size x size atlas at the highest texel density that
 * fits.
 *
 * Charting cost grows faster than the triangle count, so the mesh goes in as spatial chunks
 * of a few thousand triangles (separate xatlas meshes, packed together). Chunk borders become
 * chart borders; at 4000 triangles per chunk that adds ~1% vertices for ~40% less time.
 *
 * xatlas's texels-per-unit estimate for a resolution assumes a fixed utilisation, and the
 * atlas typically comes out ~15% too large. A cheap pack at half resolution (and half the
 * padding) measures the real ratio; the full pack at the corrected density lands within a
 * few percent, and any remaining overshoot is removed by scaling UVs uniformly. (Packing into
 * a fixed-size atlas instead spills charts into a second atlas unless the density is backed
 * off by more than this loses.)
 */
export function unwrap(x: XAtlas, r: UnwrapRequest): UnwrapResult {
  const t0 = performance.now();
  const chunks = mortonChunks(r.positions, r.indices, r.chunkTriangles ?? 4000);
  x.create();
  try {
    const globalOf: number[][] = [];
    for (const tris of chunks) {
      const local = new Map<number, number>(), glob: number[] = [];
      const idx = new Uint32Array(tris.length * 3);
      tris.forEach((t, j) => {
        for (let k = 0; k < 3; k++) {
          const v = r.indices[t * 3 + k];
          let l = local.get(v);
          if (l === undefined) { l = glob.length; local.set(v, l); glob.push(v); }
          idx[j * 3 + k] = l;
        }
      });
      const pos = new Float32Array(glob.length * 3), nrm = r.normals ? new Float32Array(glob.length * 3) : null;
      glob.forEach((v, l) => {
        pos.set(r.positions.subarray(v * 3, v * 3 + 3), l * 3);
        nrm?.set(r.normals!.subarray(v * 3, v * 3 + 3), l * 3);
      });
      x.addMesh(pos, nrm, idx);
      globalOf.push(glob);
    }
    x.computeCharts(r.chart);
    const t1 = performance.now();
    const half = Math.max(64, r.size >> 1);
    x.packCharts({ resolution: half, padding: Math.max(1, r.padding >> 1), bilinear: true });
    let info = x.info();
    const tpu = info.texelsPerUnit * (half / Math.max(info.width, info.height)) * (r.size / half) * 0.98;
    x.packCharts({ padding: r.padding, bilinear: true, texelsPerUnit: tpu });
    info = x.info();
    if (info.atlasCount !== 1) throw new Error(`xatlas produced ${info.atlasCount} atlases.`);
    const scale = Math.min(1, r.size / Math.max(info.width, info.height)) / r.size;

    const meshes = chunks.map((_, c) => x.mesh(c));
    const vertexCount = meshes.reduce((n, m) => n + m.xref.length, 0);
    const xref = new Uint32Array(vertexCount), uv = new Float32Array(vertexCount * 2);
    const indices = new Uint32Array(r.indices.length);
    let base = 0;
    meshes.forEach((m, c) => {
      if (m.indices.length !== chunks[c].length * 3) throw new Error('xatlas changed the triangle count.');
      for (let v = 0; v < m.xref.length; v++) {
        xref[base + v] = globalOf[c][m.xref[v]];
        uv[(base + v) * 2] = m.uv[v * 2] * scale;
        uv[(base + v) * 2 + 1] = m.uv[v * 2 + 1] * scale;
      }
      chunks[c].forEach((t, j) => { for (let k = 0; k < 3; k++) indices[t * 3 + k] = base + m.indices[j * 3 + k]; });
      base += m.xref.length;
    });
    return {
      xref, uv, indices, charts: info.chartCount, chunks: chunks.length,
      width: info.width, height: info.height, scale: +(scale * r.size).toFixed(3),
      ms: { charts: Math.round(t1 - t0), pack: Math.round(performance.now() - t1) },
    };
  } finally {
    x.destroy();
  }
}
