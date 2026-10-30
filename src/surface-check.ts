// Topology and quality check of a raw surface (spec §4, `furnace.checkSurface()`).
import type { RawSurface } from './gpu/surface';

export function checkSurface(s: RawSurface) {
  const idx = s.indices, p = s.positions;
  const nv = p.length / 3, nt = idx.length / 3;
  // One key per half-edge: undirected edge * 2 + (1 if it runs from the lower index).
  const keys = new Float64Array(idx.length);
  for (let t = 0; t < nt; t++) {
    for (let e = 0; e < 3; e++) {
      const a = idx[t * 3 + e], b = idx[t * 3 + (e + 1) % 3];
      keys[t * 3 + e] = (Math.min(a, b) * nv + Math.max(a, b)) * 2 + (a < b ? 1 : 0);
    }
  }
  keys.sort();
  let boundaryEdges = 0, nonManifoldEdges = 0, flippedEdges = 0;
  for (let i = 0; i < keys.length;) {
    const edge = Math.floor(keys[i] / 2);
    let n = 0, forward = 0;
    for (; i < keys.length && Math.floor(keys[i] / 2) === edge; i++) { n++; forward += keys[i] % 2; }
    if (n === 1) boundaryEdges++;
    else if (n > 2) nonManifoldEdges++;
    else if (forward !== 1) flippedEdges++;
  }
  let degenerate = 0, volume = 0, normalAgree = 0;
  for (let t = 0; t < nt; t++) {
    const a = idx[t * 3] * 3, b = idx[t * 3 + 1] * 3, c = idx[t * 3 + 2] * 3;
    const ux = p[b] - p[a], uy = p[b + 1] - p[a + 1], uz = p[b + 2] - p[a + 2];
    const vx = p[c] - p[a], vy = p[c + 1] - p[a + 1], vz = p[c + 2] - p[a + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    if (0.5 * Math.hypot(nx, ny, nz) <= 1e-6) degenerate++;
    volume += (p[a] * nx + p[a + 1] * ny + p[a + 2] * nz) / 6;
    // Does the face winding agree with the supplied vertex normals?
    const sn = s.normals;
    if (nx * (sn[a] + sn[b] + sn[c]) + ny * (sn[a + 1] + sn[b + 1] + sn[c + 1]) + nz * (sn[a + 2] + sn[b + 2] + sn[c + 2]) > 0) normalAgree++;
  }
  return {
    boundaryEdges, nonManifoldEdges, flippedEdges, degenerate,
    triangles: nt, vertices: nv,
    degenerateFraction: +(degenerate / Math.max(1, nt)).toFixed(6),
    signedVolume: Math.round(volume), // voxels^3; positive = outward winding
    normalWindingAgreement: +(normalAgree / Math.max(1, nt)).toFixed(4),
  };
}
