// CPU check of the generated marching-cubes table on random solid fields:
// closed, edge-manifold, consistently oriented, outward (positive signed volume).
import { buildTable, CORNER, EDGES, MAX_TRIS } from '../src/gpu/surface-table.ts';

const { counts, tris } = buildTable();
let maxT = 0; for (const c of counts) maxT = Math.max(maxT, c);
console.log('max triangles per cell', maxT, 'config 1 ->', counts[1], 'config 255 ->', counts[255]);

function trial(n: number, density: number, seed: number) {
  let s = seed; const rnd = () => ((s = (s * 1103515245 + 12345) >>> 0) / 2 ** 32);
  const solid = new Uint8Array(n * n * n);
  const id = (i: number, j: number, k: number) => i + n * (j + n * k);
  for (let k = 1; k < n - 1; k++) for (let j = 1; j < n - 1; j++) for (let i = 1; i < n - 1; i++) solid[id(i, j, k)] = rnd() < density ? 1 : 0;
  const vmap = new Map<number, number>(); const pos: number[] = []; const idx: number[] = [];
  const vert = (p: number, axis: number, i: number, j: number, k: number) => {
    const key = p * 3 + axis;
    let v = vmap.get(key);
    if (v === undefined) { v = pos.length / 3; vmap.set(key, v); const o = [i, j, k]; o[axis] += 0.5; pos.push(...o); }
    return v;
  };
  for (let k = 0; k < n - 1; k++) for (let j = 0; j < n - 1; j++) for (let i = 0; i < n - 1; i++) {
    let cfg = 0;
    for (let c = 0; c < 8; c++) { const [x, y, z] = CORNER(c); cfg |= solid[id(i + x, j + y, k + z)] << c; }
    for (let t = 0; t < counts[cfg]; t++) for (let q = 0; q < 3; q++) {
      const [c, axis] = EDGES[tris[(cfg * MAX_TRIS + t) * 3 + q]];
      const [x, y, z] = CORNER(c);
      idx.push(vert(id(i + x, j + y, k + z), axis, i + x, j + y, k + z));
    }
  }
  const half = new Map<string, number>(); let vol = 0;
  for (let t = 0; t < idx.length; t += 3) {
    for (let e = 0; e < 3; e++) { const k = `${idx[t + e]},${idx[t + (e + 1) % 3]}`; half.set(k, (half.get(k) ?? 0) + 1); }
    const [a, b, c] = [0, 1, 2].map((q) => pos.slice(idx[t + q] * 3, idx[t + q] * 3 + 3));
    vol += (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6;
  }
  let boundary = 0, dup = 0;
  for (const [k, c] of half) { if (c > 1) dup++; const [a, b] = k.split(','); if (!half.has(`${b},${a}`)) boundary++; }
  let solids = 0; for (const v of solid) solids += v;
  return { tris: idx.length / 3, boundary, dupHalfEdges: dup, vol: +vol.toFixed(1), solids };
}
for (const [n, d, seed] of [[12, 0.5, 1], [16, 0.3, 2], [16, 0.7, 3], [20, 0.5, 4], [10, 0.1, 5]] as const) console.log(n, d, trial(n, d, seed));
