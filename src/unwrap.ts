// Chart-based UV atlas: xatlas (MIT; vendored in third_party/xatlas with speed fixes, built
// to WebAssembly by scripts/build-xatlas.sh, run off the main thread in the browser) segments the mesh into charts,
// flattens and packs them. Unlike the per-triangle atlas in bake.ts, vertices inside a chart
// are shared, so the exported mesh needs far fewer vertices.
// Tangents come from meshoptimizer in MikkTSpace-compatible mode, which is what Blender
// recomputes on import, so the baked normal map reads the same in every engine.
import { MeshoptTangents } from 'meshoptimizer';
import type { UnwrapRequest, UnwrapResult } from './xatlas/xatlas';
import type { Mesh } from './mesh';
import type { Atlased } from './bake';

const PADDING = 3; // texels between charts

/** Runs xatlas somewhere: a Web Worker in the browser (xatlas/browser.ts), in-thread in Node (xatlas/node.ts). */
export type UnwrapRunner = (request: UnwrapRequest) => Promise<UnwrapResult>;

export async function buildChartAtlas(mesh: Mesh, size: number, run: UnwrapRunner): Promise<Atlased & { charts: string }> {
  const t0 = performance.now();
  const r = await run({ positions: mesh.positions, normals: mesh.normals, indices: mesh.indices, size, padding: PADDING });
  const ms = Math.round(performance.now() - t0);

  const T = mesh.indices.length / 3;
  const positions = new Float32Array(T * 9), normals = new Float32Array(T * 9), uvs = new Float32Array(T * 6);
  const colors = mesh.colors ? new Float32Array(T * 9) : null;
  for (let q = 0; q < T * 3; q++) {
    const av = r.indices[q], v = r.xref[av];
    positions.set(mesh.positions.subarray(v * 3, v * 3 + 3), q * 3);
    normals.set(mesh.normals.subarray(v * 3, v * 3 + 3), q * 3);
    if (colors) colors.set(mesh.colors!.subarray(v * 3, v * 3 + 3), q * 3);
    uvs[q * 2] = r.uv[av * 2];
    uvs[q * 2 + 1] = r.uv[av * 2 + 1];
  }
  const nv = r.xref.length;
  await MeshoptTangents.ready;
  const vp = new Float32Array(nv * 3), vn = new Float32Array(nv * 3);
  r.xref.forEach((v, i) => { vp.set(mesh.positions.subarray(v * 3, v * 3 + 3), i * 3); vn.set(mesh.normals.subarray(v * 3, v * 3 + 3), i * 3); });
  const tangents = MeshoptTangents.generateTangents(r.indices, vp, 3, vn, 3, r.uv, 2, ['Compatible']);

  const fit = r.scale < 1 ? `, scaled ${r.scale}x to fit` : '';
  return {
    positions, normals, uvs, tangents, colors, triangles: T,
    charts: `${r.charts} charts in ${r.chunks} chunks, ${nv} vertices; xatlas ${ms} ms (charts ${r.ms.charts}, pack ${r.ms.pack}), ${r.width}x${r.height}${fit}`,
  };
}
