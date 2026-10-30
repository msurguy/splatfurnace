// CPU mesh utilities: adjacency, island removal, normals, feature-aware smoothing.

export interface Mesh {
  positions: Float32Array;
  normals: Float32Array;
  indices: Uint32Array;
  colors?: Float32Array; // rgb, display space
}

/** Vertex one-ring as CSR (offsets + neighbour list), built from triangle edges. */
export function vertexNeighbors(vertexCount: number, indices: Uint32Array) {
  const degree = new Uint32Array(vertexCount + 1);
  for (let t = 0; t < indices.length; t += 3)
    for (let e = 0; e < 3; e++) { degree[indices[t + e]] += 2; }
  const offsets = new Uint32Array(vertexCount + 1);
  for (let v = 0; v < vertexCount; v++) offsets[v + 1] = offsets[v] + degree[v];
  const raw = new Uint32Array(offsets[vertexCount]);
  const fill = offsets.slice(0, vertexCount);
  for (let t = 0; t < indices.length; t += 3) {
    for (let e = 0; e < 3; e++) {
      const a = indices[t + e], b = indices[t + (e + 1) % 3], c = indices[t + (e + 2) % 3];
      raw[fill[a]++] = b;
      raw[fill[a]++] = c;
    }
  }
  // dedupe each ring in place
  const outOffsets = new Uint32Array(vertexCount + 1);
  const out = new Uint32Array(raw.length);
  let w = 0;
  for (let v = 0; v < vertexCount; v++) {
    outOffsets[v] = w;
    const ring = raw.subarray(offsets[v], offsets[v + 1]).sort();
    for (let k = 0; k < ring.length; k++) if (k === 0 || ring[k] !== ring[k - 1]) out[w++] = ring[k];
  }
  outOffsets[vertexCount] = w;
  return { offsets: outOffsets, list: out.subarray(0, w) };
}

/** Drop connected components whose triangle count is below `minFraction` of the largest. */
export function removeIslands(mesh: Mesh, minFraction: number): { mesh: Mesh; removed: number; kept: number } {
  const n = mesh.positions.length / 3;
  const parent = new Int32Array(n);
  for (let i = 0; i < n; i++) parent[i] = i;
  const find = (x: number) => {
    while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; }
    return x;
  };
  const idx = mesh.indices;
  for (let t = 0; t < idx.length; t += 3) {
    const a = find(idx[t]), b = find(idx[t + 1]), c = find(idx[t + 2]);
    parent[b] = a;
    parent[find(c)] = a;
  }
  const tris = new Map<number, number>();
  for (let t = 0; t < idx.length; t += 3) {
    const r = find(idx[t]);
    tris.set(r, (tris.get(r) ?? 0) + 1);
  }
  let largest = 0;
  for (const c of tris.values()) largest = Math.max(largest, c);
  const keepRoot = new Set<number>();
  for (const [r, c] of tris) if (c >= largest * minFraction) keepRoot.add(r);
  const keptIdx: number[] = [];
  for (let t = 0; t < idx.length; t += 3) {
    if (keepRoot.has(find(idx[t]))) keptIdx.push(idx[t], idx[t + 1], idx[t + 2]);
  }
  return { mesh: compact({ ...mesh, indices: Uint32Array.from(keptIdx) }), removed: tris.size - keepRoot.size, kept: keepRoot.size };
}

/** Remove unreferenced vertices. */
export function compact(mesh: Mesh): Mesh {
  const n = mesh.positions.length / 3;
  const remap = new Int32Array(n).fill(-1);
  let next = 0;
  const indices = new Uint32Array(mesh.indices.length);
  for (let i = 0; i < mesh.indices.length; i++) {
    const v = mesh.indices[i];
    if (remap[v] < 0) remap[v] = next++;
    indices[i] = remap[v];
  }
  const take = (src: Float32Array | undefined, k: number) => {
    if (!src) return undefined;
    const dst = new Float32Array(next * k);
    for (let v = 0; v < n; v++) if (remap[v] >= 0) for (let c = 0; c < k; c++) dst[remap[v] * k + c] = src[v * k + c];
    return dst;
  };
  return { positions: take(mesh.positions, 3)!, normals: take(mesh.normals, 3)!, colors: take(mesh.colors, 3), indices };
}

/** Area-weighted vertex normals. */
export function computeNormals(positions: Float32Array, indices: Uint32Array, out: Float32Array = new Float32Array(positions.length)): Float32Array {
  out.fill(0);
  const p = positions;
  for (let t = 0; t < indices.length; t += 3) {
    const a = indices[t] * 3, b = indices[t + 1] * 3, c = indices[t + 2] * 3;
    const ux = p[b] - p[a], uy = p[b + 1] - p[a + 1], uz = p[b + 2] - p[a + 2];
    const vx = p[c] - p[a], vy = p[c + 1] - p[a + 1], vz = p[c + 2] - p[a + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    for (const v of [a, b, c]) { out[v] += nx; out[v + 1] += ny; out[v + 2] += nz; }
  }
  normalize3(out);
  return out;
}

export function normalize3(a: Float32Array) {
  for (let i = 0; i < a.length; i += 3) {
    const l = Math.hypot(a[i], a[i + 1], a[i + 2]);
    if (l > 1e-20) { a[i] /= l; a[i + 1] /= l; a[i + 2] /= l; } else { a[i] = 0; a[i + 1] = 1; a[i + 2] = 0; }
  }
}

/**
 * Taubin (lambda|mu) smoothing over the one-ring, with neighbours weighted by normal
 * agreement so creases and thin features are not melted. Does not shrink the mesh.
 */
export function taubinSmooth(mesh: Mesh, ring: ReturnType<typeof vertexNeighbors>, iterations: number, maxStep: number) {
  const n = mesh.positions.length / 3;
  const tmp = new Float32Array(mesh.positions.length);
  const step = (factor: number) => {
    const p = mesh.positions, nn = mesh.normals;
    for (let v = 0; v < n; v++) {
      let sx = 0, sy = 0, sz = 0, sw = 0;
      for (let k = ring.offsets[v]; k < ring.offsets[v + 1]; k++) {
        const u = ring.list[k];
        const agree = nn[v * 3] * nn[u * 3] + nn[v * 3 + 1] * nn[u * 3 + 1] + nn[v * 3 + 2] * nn[u * 3 + 2];
        const w = smoothstep(0.3, 0.8, agree);
        sx += p[u * 3] * w; sy += p[u * 3 + 1] * w; sz += p[u * 3 + 2] * w; sw += w;
      }
      let dx = 0, dy = 0, dz = 0;
      if (sw > 1e-6) {
        dx = (sx / sw - p[v * 3]) * factor; dy = (sy / sw - p[v * 3 + 1]) * factor; dz = (sz / sw - p[v * 3 + 2]) * factor;
        const l = Math.hypot(dx, dy, dz);
        if (l > maxStep) { dx *= maxStep / l; dy *= maxStep / l; dz *= maxStep / l; }
      }
      tmp[v * 3] = p[v * 3] + dx; tmp[v * 3 + 1] = p[v * 3 + 1] + dy; tmp[v * 3 + 2] = p[v * 3 + 2] + dz;
    }
    mesh.positions.set(tmp);
  };
  for (let i = 0; i < iterations; i++) {
    step(0.5);
    step(-0.53);
    computeNormals(mesh.positions, mesh.indices, mesh.normals);
  }
}

/** Confidence-weighted Jacobi smoothing of a per-vertex scalar. */
export function smoothScalar(values: Float32Array, weights: Float32Array, ring: ReturnType<typeof vertexNeighbors>, iterations: number): Float32Array {
  let cur: Float32Array = values.slice();
  const next = new Float32Array(values.length);
  for (let it = 0; it < iterations; it++) {
    for (let v = 0; v < cur.length; v++) {
      let s = cur[v] * (weights[v] + 1e-3), w = weights[v] + 1e-3;
      for (let k = ring.offsets[v]; k < ring.offsets[v + 1]; k++) {
        const u = ring.list[k];
        s += cur[u] * weights[u];
        w += weights[u];
      }
      next[v] = s / w;
    }
    cur = next.slice();
  }
  return cur;
}

export function smoothstep(a: number, b: number, x: number) {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}
