import { MeshoptSimplifier } from 'meshoptimizer';
import { compact, computeNormals, normalize3, type Mesh } from './mesh';
import type { FieldSampler } from './gpu/gpu';

export interface Simplified {
  mesh: Mesh;
  error: number; // absolute, grid units
}

/** Decimate with meshoptimizer, letting it move vertices and update normals/colours. */
export async function simplify(mesh: Mesh, targetTriangles: number): Promise<Simplified> {
  await MeshoptSimplifier.ready;
  const n = mesh.positions.length / 3;
  if (mesh.indices.length / 3 <= targetTriangles) return { mesh, error: 0 };
  const positions = mesh.positions.slice();
  const attrs = new Float32Array(n * 6);
  for (let v = 0; v < n; v++) {
    attrs.set(mesh.normals.subarray(v * 3, v * 3 + 3), v * 6);
    if (mesh.colors) attrs.set(mesh.colors.subarray(v * 3, v * 3 + 3), v * 6 + 3);
  }
  const indices = mesh.indices.slice();
  const [count, error] = MeshoptSimplifier.simplifyWithUpdate(
    indices, positions, 3, attrs, 6, [0.25, 0.25, 0.25, 0.1, 0.1, 0.1], null,
    targetTriangles * 3, 1e9, ['ErrorAbsolute', 'Prune'],
  );
  const normals = new Float32Array(n * 3);
  const colors = new Float32Array(n * 3);
  for (let v = 0; v < n; v++) {
    normals.set(attrs.subarray(v * 6, v * 6 + 3), v * 3);
    colors.set(attrs.subarray(v * 6 + 3, v * 6 + 6), v * 3);
  }
  normalize3(normals);
  const out = compact({ positions, normals, colors: mesh.colors ? colors : undefined, indices: indices.slice(0, count) });
  // Blend the simplifier's attribute normals with geometric ones: geometric normals keep
  // silhouettes honest, attribute normals keep smooth areas smooth.
  const geo = computeNormals(out.positions, out.indices);
  for (let i = 0; i < geo.length; i++) out.normals[i] = 0.5 * out.normals[i] + 0.5 * geo[i];
  normalize3(out.normals);
  return { mesh: out, error };
}

export interface Atlased {
  positions: Float32Array; // per corner (unindexed), grid units
  normals: Float32Array;
  tangents: Float32Array; // xyzw
  uvs: Float32Array;
  colors: Float32Array | null; // per corner rgb, fallback where rays miss
  triangles: number;
}

const MIN_CELL = 6; // texels; below this the gutters eat the triangle
const PAD = 1.5; // texels between each triangle and its cell edge / the diagonal

/**
 * Per-triangle atlas with area-proportional texel density. Triangles are sorted by area,
 * paired with a similar-sized neighbour into one square cell (one triangle per half), and
 * the cells are shelf-packed. A binary search on the texels-per-unit scale fills the
 * texture. No unwrapping step, can't fail on bad topology, and big flat triangles get
 * as many texels as their surface area deserves.
 */
export function buildAtlas(mesh: Mesh, size: number): Atlased & { cell: number; perRow: number } {
  const T = mesh.indices.length / 3;
  const p = mesh.positions;
  const area = new Float32Array(T);
  for (let t = 0; t < T; t++) area[t] = triangleArea(p, mesh.indices[t * 3], mesh.indices[t * 3 + 1], mesh.indices[t * 3 + 2]);
  const order = Array.from({ length: T }, (_, t) => t).sort((a, b) => area[b] - area[a]);
  const pairs = Math.ceil(T / 2);
  const pairArea = new Float32Array(pairs);
  for (let k = 0; k < pairs; k++) pairArea[k] = area[order[2 * k]]; // sorted: first is the larger

  // A right isosceles half-cell of side s has area s^2/2; reserve gutters on top.
  const cellSize = (k: number, scale: number) => Math.max(MIN_CELL, Math.ceil(scale * Math.sqrt(2 * pairArea[k]) + 2 * PAD + PAD * Math.SQRT2));
  const pack = (scale: number, out?: Int32Array) => {
    let x = 0, y = 0, rowH = 0;
    for (let k = 0; k < pairs; k++) {
      const s = cellSize(k, scale);
      if (x + s > size) { y += rowH; x = 0; rowH = 0; }
      if (s > size || y + s > size) return false;
      if (out) { out[k * 3] = x; out[k * 3 + 1] = y; out[k * 3 + 2] = s; }
      x += s;
      rowH = Math.max(rowH, s);
    }
    return true;
  };
  if (!pack(0)) throw new Error(`Texture ${size}px is too small for ${T} triangles; raise the texture size or lower the triangle count.`);
  let lo = 0, hi = size / Math.sqrt(2 * Math.max(pairArea[0], 1e-12));
  for (let i = 0; i < 40; i++) {
    const mid = (lo + hi) / 2;
    if (pack(mid)) lo = mid; else hi = mid;
  }
  const cells = new Int32Array(pairs * 3);
  pack(lo, cells);
  let cellSum = 0;
  for (let k = 0; k < pairs; k++) cellSum += cells[k * 3 + 2];

  const positions = new Float32Array(T * 9);
  const normals = new Float32Array(T * 9);
  const uvs = new Float32Array(T * 6);
  const tangents = new Float32Array(T * 12);
  const colors = mesh.colors ? new Float32Array(T * 9) : null;
  for (let rank = 0; rank < T; rank++) {
    const t = order[rank];
    const pair = rank >> 1;
    const x0 = cells[pair * 3], y0 = cells[pair * 3 + 1], s = cells[pair * 3 + 2];
    const ids = [mesh.indices[t * 3], mesh.indices[t * 3 + 1], mesh.indices[t * 3 + 2]];
    // put the corner with the largest angle on the right-angle corner of the cell half
    const ang = ids.map((_, k) => angleAt(p, ids[k], ids[(k + 1) % 3], ids[(k + 2) % 3]));
    const r = ang.indexOf(Math.max(...ang));
    const order3 = [r, (r + 1) % 3, (r + 2) % 3];
    const pad = PAD;
    const d = pad * Math.SQRT2;
    const corners = (rank & 1) === 0
      ? [[x0 + pad, y0 + pad], [x0 + s - pad - d, y0 + pad], [x0 + pad, y0 + s - pad - d]]
      : [[x0 + s - pad, y0 + s - pad], [x0 + pad + d, y0 + s - pad], [x0 + s - pad, y0 + pad + d]];
    for (let k = 0; k < 3; k++) {
      const src = order3[k];
      const v = ids[src];
      const dst = t * 3 + src; // keep the original winding
      positions.set(p.subarray(v * 3, v * 3 + 3), dst * 3);
      normals.set(mesh.normals.subarray(v * 3, v * 3 + 3), dst * 3);
      if (colors) colors.set(mesh.colors!.subarray(v * 3, v * 3 + 3), dst * 3);
      uvs[dst * 2] = corners[k][0] / size;
      uvs[dst * 2 + 1] = corners[k][1] / size;
    }
    writeTangents(positions, normals, uvs, tangents, t);
  }
  return { positions, normals, uvs, tangents, colors, triangles: T, cell: +(cellSum / pairs).toFixed(1), perRow: 0 };
}

/** Fraction of the texture covered by triangles in UV space. */
export function uvCoverage(a: Atlased) {
  let sum = 0;
  for (let t = 0; t < a.triangles; t++) {
    const u = a.uvs, i = t * 6;
    sum += Math.abs((u[i + 2] - u[i]) * (u[i + 5] - u[i + 1]) - (u[i + 4] - u[i]) * (u[i + 3] - u[i + 1])) / 2;
  }
  return +sum.toFixed(3);
}

function triangleArea(p: Float32Array, a: number, b: number, c: number) {
  const ux = p[b * 3] - p[a * 3], uy = p[b * 3 + 1] - p[a * 3 + 1], uz = p[b * 3 + 2] - p[a * 3 + 2];
  const vx = p[c * 3] - p[a * 3], vy = p[c * 3 + 1] - p[a * 3 + 1], vz = p[c * 3 + 2] - p[a * 3 + 2];
  return 0.5 * Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx);
}

function angleAt(p: Float32Array, a: number, b: number, c: number) {
  const ux = p[b * 3] - p[a * 3], uy = p[b * 3 + 1] - p[a * 3 + 1], uz = p[b * 3 + 2] - p[a * 3 + 2];
  const vx = p[c * 3] - p[a * 3], vy = p[c * 3 + 1] - p[a * 3 + 1], vz = p[c * 3 + 2] - p[a * 3 + 2];
  const l = Math.hypot(ux, uy, uz) * Math.hypot(vx, vy, vz);
  return l > 0 ? Math.acos(Math.max(-1, Math.min(1, (ux * vx + uy * vy + uz * vz) / l))) : 0;
}

function writeTangents(P: Float32Array, N: Float32Array, UV: Float32Array, out: Float32Array, t: number) {
  const i0 = t * 3, i1 = i0 + 1, i2 = i0 + 2;
  const e1 = [P[i1 * 3] - P[i0 * 3], P[i1 * 3 + 1] - P[i0 * 3 + 1], P[i1 * 3 + 2] - P[i0 * 3 + 2]];
  const e2 = [P[i2 * 3] - P[i0 * 3], P[i2 * 3 + 1] - P[i0 * 3 + 1], P[i2 * 3 + 2] - P[i0 * 3 + 2]];
  const du1 = UV[i1 * 2] - UV[i0 * 2], dv1 = UV[i1 * 2 + 1] - UV[i0 * 2 + 1];
  const du2 = UV[i2 * 2] - UV[i0 * 2], dv2 = UV[i2 * 2 + 1] - UV[i0 * 2 + 1];
  const det = du1 * dv2 - du2 * dv1;
  const r = Math.abs(det) > 1e-20 ? 1 / det : 0;
  const T = [0, 1, 2].map((k) => (e1[k] * dv2 - e2[k] * dv1) * r);
  const B = [0, 1, 2].map((k) => (e2[k] * du1 - e1[k] * du2) * r);
  for (const i of [i0, i1, i2]) {
    const n = [N[i * 3], N[i * 3 + 1], N[i * 3 + 2]];
    // Gram-Schmidt against this corner's normal (what the shader will interpolate)
    const dn = T[0] * n[0] + T[1] * n[1] + T[2] * n[2];
    let tx = T[0] - n[0] * dn, ty = T[1] - n[1] * dn, tz = T[2] - n[2] * dn;
    const l = Math.hypot(tx, ty, tz) || 1;
    tx /= l; ty /= l; tz /= l;
    const cx = n[1] * tz - n[2] * ty, cy = n[2] * tx - n[0] * tz, cz = n[0] * ty - n[1] * tx;
    const w = cx * B[0] + cy * B[1] + cz * B[2] < 0 ? -1 : 1;
    out.set([tx, ty, tz, w], i * 4);
  }
}

export interface BakeOptions {
  size: number;
  range: number; // ray half-length, grid units
  normalMap: boolean;
}

export interface BakeResult {
  color: Uint8ClampedArray; // RGBA, sRGB
  normal: Uint8ClampedArray | null; // RGBA, tangent space
  texels: number;
  misses: number;
}

/** Rasterise every triangle in UV space, ray-sample the splats at each texel, dilate. */
export async function bakeTextures(atlas: Atlased, sampler: FieldSampler, o: BakeOptions, progress: (m: string) => void): Promise<BakeResult> {
  const S = o.size;
  const owner = new Int32Array(S * S).fill(-1);
  const bary: number[] = [];
  const texelIds: number[] = [];
  const { uvs } = atlas;
  for (let t = 0; t < atlas.triangles; t++) {
    const ax = uvs[t * 6] * S, ay = uvs[t * 6 + 1] * S;
    const bx = uvs[t * 6 + 2] * S, by = uvs[t * 6 + 3] * S;
    const cx = uvs[t * 6 + 4] * S, cy = uvs[t * 6 + 5] * S;
    const area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay);
    if (Math.abs(area) < 1e-12) continue;
    // conservative: include texels whose centre is within ~0.75 texel of the triangle
    const grow = 0.75;
    const minX = Math.max(0, Math.floor(Math.min(ax, bx, cx) - grow)), maxX = Math.min(S - 1, Math.ceil(Math.max(ax, bx, cx) + grow));
    const minY = Math.max(0, Math.floor(Math.min(ay, by, cy) - grow)), maxY = Math.min(S - 1, Math.ceil(Math.max(ay, by, cy) + grow));
    const edges = [[ax, ay, bx, by], [bx, by, cx, cy], [cx, cy, ax, ay]].map(([x0, y0, x1, y1]) => Math.hypot(x1 - x0, y1 - y0));
    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        const px = x + 0.5, py = y + 0.5;
        const w0 = ((bx - px) * (cy - py) - (cx - px) * (by - py)) / area;
        const w1 = ((cx - px) * (ay - py) - (ax - px) * (cy - py)) / area;
        const w2 = 1 - w0 - w1;
        // barycentric -> signed distance to each edge in texels
        const d0 = w0 * Math.abs(area) / edges[1], d1 = w1 * Math.abs(area) / edges[2], d2 = w2 * Math.abs(area) / edges[0];
        if (Math.min(d0, d1, d2) < -grow) continue;
        const id = y * S + x;
        if (owner[id] >= 0) continue;
        owner[id] = t;
        // clamp to the triangle so gutter texels sample the edge, not beyond it
        let c0 = Math.max(0, w0), c1 = Math.max(0, w1), c2 = Math.max(0, w2);
        const cs = c0 + c1 + c2;
        c0 /= cs; c1 /= cs; c2 /= cs;
        bary.push(c0, c1, c2);
        texelIds.push(id);
      }
    }
  }
  const count = texelIds.length;
  const pts = new Float32Array(count * 3), nrm = new Float32Array(count * 3);
  const P = atlas.positions, N = atlas.normals;
  for (let i = 0; i < count; i++) {
    const t = owner[texelIds[i]];
    for (let k = 0; k < 3; k++) {
      pts[i * 3 + k] = bary[i * 3] * P[t * 9 + k] + bary[i * 3 + 1] * P[t * 9 + 3 + k] + bary[i * 3 + 2] * P[t * 9 + 6 + k];
      nrm[i * 3 + k] = bary[i * 3] * N[t * 9 + k] + bary[i * 3 + 1] * N[t * 9 + 3 + k] + bary[i * 3 + 2] * N[t * 9 + 6 + k];
    }
  }
  normalize3(nrm);
  progress(`Ray-sampling splats for ${count.toLocaleString()} texels`);
  const r = await sampler.sample(pts, nrm, o.range);

  const color = new Uint8ClampedArray(S * S * 4);
  const normal = o.normalMap ? new Uint8ClampedArray(S * S * 4) : null;
  const filled = new Uint8Array(S * S);
  let misses = 0;
  const T4 = atlas.tangents;
  for (let i = 0; i < count; i++) {
    const id = texelIds[i];
    const hit = r.colors[i * 3] >= 0 && r.coverage[i] >= 0.05;
    if (!hit) {
      misses++;
      if (!atlas.colors) continue;
      const t = owner[id], C = atlas.colors;
      for (let k = 0; k < 3; k++) {
        color[id * 4 + k] = 255 * (bary[i * 3] * C[t * 9 + k] + bary[i * 3 + 1] * C[t * 9 + 3 + k] + bary[i * 3 + 2] * C[t * 9 + 6 + k]);
      }
    } else {
      color[id * 4] = r.colors[i * 3] * 255;
      color[id * 4 + 1] = r.colors[i * 3 + 1] * 255;
      color[id * 4 + 2] = r.colors[i * 3 + 2] * 255;
    }
    color[id * 4 + 3] = 255;
    filled[id] = 1;
    if (normal) {
      const t = owner[id];
      const b = [bary[i * 3], bary[i * 3 + 1], bary[i * 3 + 2]];
      const tan = [0, 1, 2, 3].map((k) => b[0] * T4[t * 12 + k] + b[1] * T4[t * 12 + 4 + k] + b[2] * T4[t * 12 + 8 + k]);
      const n = [nrm[i * 3], nrm[i * 3 + 1], nrm[i * 3 + 2]];
      const dn = tan[0] * n[0] + tan[1] * n[1] + tan[2] * n[2];
      let tx = tan[0] - n[0] * dn, ty = tan[1] - n[1] * dn, tz = tan[2] - n[2] * dn;
      const tl = Math.hypot(tx, ty, tz) || 1;
      tx /= tl; ty /= tl; tz /= tl;
      const w = tan[3] < 0 ? -1 : 1;
      const bx = (n[1] * tz - n[2] * ty) * w, by = (n[2] * tx - n[0] * tz) * w, bz = (n[0] * ty - n[1] * tx) * w;
      const f = hit ? [r.normals[i * 3], r.normals[i * 3 + 1], r.normals[i * 3 + 2]] : n;
      const ts = [f[0] * tx + f[1] * ty + f[2] * tz, f[0] * bx + f[1] * by + f[2] * bz, f[0] * n[0] + f[1] * n[1] + f[2] * n[2]];
      normal[id * 4] = (ts[0] * 0.5 + 0.5) * 255;
      normal[id * 4 + 1] = (ts[1] * 0.5 + 0.5) * 255;
      normal[id * 4 + 2] = (Math.max(0, ts[2]) * 0.5 + 0.5) * 255;
      normal[id * 4 + 3] = 255;
    }
  }
  progress('Filling misses and gutters');
  dilate(color, normal, filled, S, 24);
  return { color, normal, texels: count, misses };
}

/** Grow filled texels outward (misses and gutters), then flood the rest with the mean. */
function dilate(color: Uint8ClampedArray, normal: Uint8ClampedArray | null, filled: Uint8Array, S: number, passes: number) {
  let front = filled.slice();
  for (let pass = 0; pass < passes; pass++) {
    const next = front.slice();
    let changed = 0;
    for (let y = 0; y < S; y++) {
      for (let x = 0; x < S; x++) {
        const id = y * S + x;
        if (front[id]) continue;
        let r = 0, g = 0, b = 0, nx = 0, ny = 0, nz = 0, c = 0;
        for (let dy = -1; dy <= 1; dy++) {
          const yy = y + dy;
          if (yy < 0 || yy >= S) continue;
          for (let dx = -1; dx <= 1; dx++) {
            const xx = x + dx;
            if (xx < 0 || xx >= S) continue;
            const j = yy * S + xx;
            if (!front[j]) continue;
            r += color[j * 4]; g += color[j * 4 + 1]; b += color[j * 4 + 2];
            if (normal) { nx += normal[j * 4]; ny += normal[j * 4 + 1]; nz += normal[j * 4 + 2]; }
            c++;
          }
        }
        if (!c) continue;
        color[id * 4] = r / c; color[id * 4 + 1] = g / c; color[id * 4 + 2] = b / c; color[id * 4 + 3] = 255;
        if (normal) { normal[id * 4] = nx / c; normal[id * 4 + 1] = ny / c; normal[id * 4 + 2] = nz / c; normal[id * 4 + 3] = 255; }
        next[id] = 1;
        changed++;
      }
    }
    front = next;
    if (!changed) break;
  }
  for (let id = 0; id < S * S; id++) {
    if (front[id]) continue;
    color[id * 4] = color[id * 4 + 1] = color[id * 4 + 2] = 128; color[id * 4 + 3] = 255;
    if (normal) { normal[id * 4] = 128; normal[id * 4 + 1] = 128; normal[id * 4 + 2] = 255; normal[id * 4 + 3] = 255; }
  }
}
