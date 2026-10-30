// Marching-cubes case table, generated at load time from first principles (no copied table).
//
// For each of the 256 corner configurations we draw the contour on each of the six cube
// faces, link the face segments into closed cycles on the cube surface, and fan-triangulate
// each cycle. Ambiguous faces (diagonal corners solid) always join the solid corners; the
// decision depends only on the face's own corners, so the two cells sharing a face always
// agree and the surface is watertight. See docs/surface.md.

/** Corner c of a cell sits at offset (c & 1, c >> 1 & 1, c >> 2 & 1). */
export const CORNER = (c: number) => [c & 1, (c >> 1) & 1, (c >> 2) & 1] as const;

/** Edge e: [min corner, axis]. Edges 0-3 run along x, 4-7 along y, 8-11 along z. */
export const EDGES: ReadonlyArray<readonly [number, number]> = (() => {
  const out: [number, number][] = [];
  for (let axis = 0; axis < 3; axis++)
    for (let c = 0; c < 8; c++) if (!((c >> axis) & 1)) out.push([c, axis]);
  return out;
})();

const edgeOf = (a: number, b: number) => {
  const lo = Math.min(a, b), axis = Math.log2(a ^ b);
  return EDGES.findIndex(([c, ax]) => c === lo && ax === axis);
};

/** The six faces, each as four corners counter-clockwise seen from outside the cell. */
const FACES: number[][] = (() => {
  const faces: number[][] = [];
  for (let axis = 0; axis < 3; axis++) {
    for (let side = 0; side < 2; side++) {
      const u = (axis + 1) % 3, v = (axis + 2) % 3; // u x v = +axis
      const at = (du: number, dv: number) => (side << axis) | (du << u) | (dv << v);
      const ring = [at(0, 0), at(1, 0), at(1, 1), at(0, 1)]; // CCW seen from +axis
      faces.push(side ? ring : ring.reverse());
    }
  }
  return faces;
})();

/** Two edges lie on a common cell face. */
const shareFace = (e1: number, e2: number) => {
  const corners = (e: number) => [EDGES[e][0], EDGES[e][0] | (1 << EDGES[e][1])];
  return FACES.some((f) => corners(e1).every((c) => f.includes(c)) && corners(e2).every((c) => f.includes(c)));
};

export const MAX_TRIS = 12;

/** Per configuration: triangle count, then up to MAX_TRIS triangles of 3 edge ids. */
export function buildTable(): { counts: Uint32Array; tris: Uint32Array } {
  const counts = new Uint32Array(256);
  const tris = new Uint32Array(256 * MAX_TRIS * 3);
  for (let cfg = 0; cfg < 256; cfg++) {
    const solid = (c: number) => (cfg >> c) & 1;
    // Directed face segments: solid on the left seen from outside. Walking a face CCW,
    // each solid->empty crossing ("exit") joins the next empty->solid crossing ("entry").
    const next = new Int32Array(12).fill(-1);
    for (const f of FACES) {
      const cross: { e: number; exit: boolean }[] = [];
      for (let k = 0; k < 4; k++) {
        const a = f[k], b = f[(k + 1) % 4];
        if (solid(a) !== solid(b)) cross.push({ e: edgeOf(a, b), exit: solid(a) === 1 });
      }
      for (let k = 0; k < cross.length; k++) {
        if (!cross[k].exit) continue;
        const entry = cross[(k + 1) % cross.length];
        next[cross[k].e] = entry.e;
      }
    }
    const seen = new Uint8Array(12);
    const out: number[] = [];
    for (let s = 0; s < 12; s++) {
      if (next[s] < 0 || seen[s]) continue;
      const cycle: number[] = [];
      for (let e = s; !seen[e]; e = next[e]) { seen[e] = 1; cycle.push(e); }
      // Choose a fan apex whose diagonals never join two edges of one face: only such a
      // diagonal could also appear in the neighbouring cell and make the edge non-manifold.
      const n = cycle.length;
      let apex = -1;
      for (let a = 0; a < n && apex < 0; a++) {
        let ok = true;
        for (let k = 2; k < n - 1 && ok; k++) if (shareFace(cycle[a], cycle[(a + k) % n])) ok = false;
        if (ok) apex = a;
      }
      if (apex < 0) throw new Error(`surface table: no safe fan for config ${cfg}`);
      // Cycles run with solid on the left seen from outside the cell, i.e. clockwise seen
      // from the empty side, so emit each fan triangle reversed (CCW from outside the solid).
      for (let k = 1; k < n - 1; k++) out.push(cycle[apex], cycle[(apex + k + 1) % n], cycle[(apex + k) % n]);
    }
    if (out.length / 3 > MAX_TRIS) throw new Error(`surface table: config ${cfg} needs ${out.length / 3} triangles`);
    counts[cfg] = out.length / 3;
    tris.set(out, cfg * MAX_TRIS * 3);
  }
  return { counts, tris };
}
