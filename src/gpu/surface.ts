// Surface extraction stage: splats -> closed triangle mesh in grid units.
// Design notes and sources: docs/surface.md.
//
// density scatter -> solid/exterior bit masks -> exterior flood fill (cavity sealing)
// -> cell configurations -> count + prefix scan -> emit vertices -> emit triangles
import { buffer, checkShader, readBack, S, type Gpu } from './gpu';
import type { Grid } from '../grid';
import { buildTable, EDGES, MAX_TRIS } from './surface-table';
import {
  CLASSIFY_WGSL, CONFIG_WGSL, COUNT_WGSL, EMIT_TRIS_WGSL, EMIT_VERTS_WGSL, FLOOD_WGSL, SCAN_BLOCKS_WGSL, SCATTER_WGSL,
} from './surface.wgsl';

export interface SurfaceOptions {
  iso: number; // density threshold, in units of summed opacity-weighted Gaussian response
  minSigma: number; // voxels; lower bound applied to every splat sigma for geometry only
  support: number; // sigmas; how far each Gaussian's influence extends
  maxExtent: number; // voxels; splats whose half-extent (col.w) exceeds this are ignored
  fillCavities: boolean; // turn empty space unreachable from outside into solid
}

export interface RawSurface {
  positions: Float32Array; // xyz per vertex, grid units
  normals: Float32Array; // unit, pointing outward (from solid toward empty)
  indices: Uint32Array; // triangles, counter-clockwise seen from outside
  floodPasses: number; // diagnostic: iterations used by cavity filling (0 if disabled)
}

const SCALE = 65536; // density fixed point: 16 fractional bits
const FLOOD_BATCH = 4; // flood dispatches between convergence checks
const MAX_FLOOD_PASSES = 4096;

let last: RawSurface | null = null;
/** The most recent raw surface, for `furnace.checkSurface()`. */
export const lastRawSurface = () => last;

let tableData: Uint32Array | null = null;
function table() {
  if (!tableData) {
    const { counts, tris } = buildTable();
    // [256 counts][256 x 36 edge ids][12 edges as corner | axis << 3]
    tableData = new Uint32Array(256 + 256 * 36 + 12);
    tableData.set(counts);
    for (let c = 0; c < 256; c++) tableData.set(tris.subarray(c * MAX_TRIS * 3, c * MAX_TRIS * 3 + 36), 256 + c * 36);
    EDGES.forEach(([corner, axis], e) => { tableData![256 + 256 * 36 + e] = corner | (axis << 3); });
    if (Math.max(...counts) > 12) throw new Error('surface table: more than 12 triangles in a cell');
  }
  return tableData;
}

export async function extractSurface(
  gpu: Gpu, grid: Grid, splats: Float32Array, splatCount: number, options: SurfaceOptions,
  progress: (msg: string) => void,
): Promise<RawSurface> {
  const { device } = gpu;
  const owned: GPUBuffer[] = [];
  const make = (label: string, size: number, usage: number, data?: ArrayBufferView) => {
    if (size > gpu.maxBinding) {
      throw new Error(`Surface: ${label} needs ${(size / 2 ** 20).toFixed(0)} MB, more than this GPU allows per buffer `
        + `(${(gpu.maxBinding / 2 ** 20).toFixed(0)} MB). Lower Resolution.`);
    }
    const b = buffer(device, size, usage, data);
    owned.push(b);
    return b;
  };
  const maxGroups = device.limits.maxComputeWorkgroupsPerDimension;
  const groups = (n: number) => [Math.min(n, maxGroups), Math.ceil(n / maxGroups)] as const;

  device.pushErrorScope('validation');
  device.pushErrorScope('out-of-memory');
  let scopes = 2;
  const popScopes = async () => {
    const errors: (GPUError | null)[] = [];
    while (scopes > 0) { scopes--; errors.push(await device.popErrorScope()); }
    return errors.find((e) => e);
  };
  try {
    const [dx, dy, dz] = grid.dims;
    const nPoints = dx * dy * dz;
    const bdims = [Math.ceil(dx / 8), Math.ceil(dy / 8), Math.ceil(dz / 8)];
    const nBricks = bdims[0] * bdims[1] * bdims[2];
    const nWords = Math.ceil(nPoints / 4);
    const nBlocks = Math.ceil(nWords / 256);

    const params = new ArrayBuffer(80);
    const u = new Uint32Array(params), f = new Float32Array(params);
    u.set([dx, dy, dz, nPoints, ...bdims, nBricks]);
    f.set([options.iso, options.minSigma, options.support, options.maxExtent], 8);
    u.set([splatCount, nWords, nBlocks], 12);
    f[15] = SCALE;
    u[16] = options.fillCavities ? 1 : 0;

    const uParams = make('parameters', 80, S.UNIFORM | S.COPY_DST, u);
    const bSplats = make('splat buffer', Math.max(16, splatCount * 80), S.STORAGE | S.COPY_DST, splats.subarray(0, splatCount * 20));
    const bDens = make('density grid', nPoints * 4, S.STORAGE);
    const bSolid = make('solid mask', nBricks * 64, S.STORAGE);
    const bOutside = make('exterior mask', nBricks * 64, S.STORAGE);
    const bChanges = make('flood counter', 4, S.STORAGE | S.COPY_SRC | S.COPY_DST);
    const bCfg = make('cell configurations', nWords * 4, S.STORAGE);
    const bBlocks = make('scan blocks', (nBlocks + 1) * 8, S.STORAGE | S.COPY_SRC);
    const bTable = make('case table', table().byteLength, S.STORAGE | S.COPY_DST, table());

    const pipe = async (code: string, label: string) => {
      const module = device.createShaderModule({ code, label });
      await checkShader(module, `Surface ${label}`);
      return device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' }, label });
    };
    const [scatter, classify, flood, config, count, scanBlocks, emitVerts, emitTris] = await Promise.all([
      pipe(SCATTER_WGSL, 'scatter'), pipe(CLASSIFY_WGSL, 'classify'), pipe(FLOOD_WGSL, 'flood'), pipe(CONFIG_WGSL, 'config'),
      pipe(COUNT_WGSL, 'count'), pipe(SCAN_BLOCKS_WGSL, 'scan'), pipe(EMIT_VERTS_WGSL, 'emit vertices'), pipe(EMIT_TRIS_WGSL, 'emit triangles'),
    ]);
    const bind = (p: GPUComputePipeline, bufs: GPUBuffer[]) => device.createBindGroup({
      layout: p.getBindGroupLayout(0),
      entries: bufs.map((b, binding) => ({ binding, resource: { buffer: b } })),
    });
    const run = (enc: GPUCommandEncoder, p: GPUComputePipeline, group: GPUBindGroup, n: number) => {
      const pass = enc.beginComputePass();
      pass.setPipeline(p);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(...groups(n));
      pass.end();
    };

    progress(`Splatting ${splatCount.toLocaleString()} Gaussians into a ${grid.dims.join('×')} density grid`);
    let enc = device.createCommandEncoder();
    run(enc, scatter, bind(scatter, [uParams, bSplats, bDens]), splatCount);
    run(enc, classify, bind(classify, [uParams, bDens, bSolid, bOutside]), Math.ceil(nBricks * 16 / 256));
    device.queue.submit([enc.finish()]);

    let floodPasses = 0;
    if (options.fillCavities) {
      progress('Sealing interior cavities');
      const group = bind(flood, [uParams, bSolid, bOutside, bChanges]);
      for (;;) {
        enc = device.createCommandEncoder();
        enc.clearBuffer(bChanges);
        for (let i = 0; i < FLOOD_BATCH; i++) run(enc, flood, group, nBricks);
        device.queue.submit([enc.finish()]);
        floodPasses += FLOOD_BATCH;
        const changed = new Uint32Array(await readBack(device, bChanges, 4))[0];
        if (!changed) break;
        if (floodPasses >= MAX_FLOOD_PASSES) throw new Error('Surface: cavity flood fill did not converge.');
      }
    }

    progress('Counting surface cells');
    enc = device.createCommandEncoder();
    run(enc, config, bind(config, [uParams, bOutside, bCfg]), nBlocks);
    const countGroup = [uParams, bCfg, bTable, bBlocks];
    run(enc, count, bind(count, countGroup), nBlocks);
    run(enc, scanBlocks, bind(scanBlocks, [uParams, bBlocks]), 1);
    device.queue.submit([enc.finish()]);
    const totals = new Uint32Array(await readBack(device, bBlocks, (nBlocks + 1) * 8)).subarray(nBlocks * 2);
    const [nVerts, nTris] = totals;
    if (!nTris) throw new Error('No surface found: nothing in the grid reaches the density threshold. Lower the density threshold (iso) or the opacity cut-off.');

    progress(`Emitting ${nTris.toLocaleString()} triangles`);
    const bVbase = make('vertex offsets', nPoints * 4, S.STORAGE);
    const bPos = make('vertex positions', nVerts * 12, S.STORAGE | S.COPY_SRC);
    const bNrm = make('vertex normals', nVerts * 12, S.STORAGE | S.COPY_SRC);
    const bIdx = make('triangle indices', nTris * 12, S.STORAGE | S.COPY_SRC);
    enc = device.createCommandEncoder();
    run(enc, emitVerts, bind(emitVerts, [...countGroup, bDens, bVbase, bPos, bNrm]), nBlocks);
    run(enc, emitTris, bind(emitTris, [...countGroup, bVbase, bIdx]), nBlocks);
    device.queue.submit([enc.finish()]);
    const [pos, nrm, idx] = await Promise.all([
      readBack(device, bPos, nVerts * 12), readBack(device, bNrm, nVerts * 12), readBack(device, bIdx, nTris * 12),
    ]);

    const gpuError = await popScopes();
    if (gpuError) throw new Error(`Surface GPU error: ${gpuError.message}`);
    last = { positions: new Float32Array(pos), normals: new Float32Array(nrm), indices: new Uint32Array(idx), floodPasses };
    return last;
  } finally {
    await popScopes().catch(() => undefined);
    for (const b of owned) b.destroy();
  }
}
