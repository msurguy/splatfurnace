// WGSL for the surface stage. One module per kernel so each 'auto' layout only holds
// the bindings that kernel uses. Grid units throughout; see docs/surface.md.

const PARAMS = /* wgsl */ `
struct Params {
  dims: vec3<u32>, nPoints: u32,
  bdims: vec3<u32>, nBricks: u32, // 8^3 bricks of the bit masks
  iso: f32, minSigma: f32, support: f32, maxExtent: f32,
  splatCount: u32, nWords: u32, nBlocks: u32, scale: f32, // nWords: config words (4 points each)
  fill: u32, p0: u32, p1: u32, p2: u32,
};
@group(0) @binding(0) var<uniform> P: Params;

fn groupIndex(wid: vec3<u32>, nwg: vec3<u32>) -> u32 { return wid.x + wid.y * nwg.x; }
fn pointCoord(l: u32) -> vec3<u32> {
  return vec3<u32>(l % P.dims.x, (l / P.dims.x) % P.dims.y, l / (P.dims.x * P.dims.y));
}
// Bit masks: brick b holds 64 rows (y + 8z) of 8 bits (x), four rows per word.
fn maskWord(x: u32, y: u32, z: u32) -> u32 {
  let b = (x >> 3u) + P.bdims.x * ((y >> 3u) + P.bdims.y * (z >> 3u));
  return b * 16u + (((y & 7u) + 8u * (z & 7u)) >> 2u);
}
fn maskShift(x: u32, y: u32, z: u32) -> u32 { return ((((y & 7u) + 8u * (z & 7u)) & 3u) * 8u) + (x & 7u); }
`;

/** Scatter every splat's Gaussian into the fixed-point density grid. */
export const SCATTER_WGSL = PARAMS + /* wgsl */ `
@group(0) @binding(1) var<storage, read> splats: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dens: array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
  let s = groupIndex(wid, nwg);
  if (s >= P.splatCount) { return; }
  if (splats[s * 5u + 4u].w > P.maxExtent) { return; }
  let head = splats[s * 5u];
  let c = head.xyz;
  let o = head.w;
  // Inverse covariance A = sum_a u u^T / sigma_a^2 and the box half-extent per world axis.
  var A = mat3x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  var e2 = vec3<f32>(0.0);
  for (var a = 0u; a < 3u; a++) {
    let ax = splats[s * 5u + 1u + a];
    let sg = max(ax.w, P.minSigma);
    let u = ax.xyz;
    A += mat3x3<f32>(u * u.x, u * u.y, u * u.z) * (1.0 / (sg * sg));
    e2 += (u * sg) * (u * sg);
  }
  let h = P.support * sqrt(e2);
  let lo = vec3<i32>(max(ceil(c - h), vec3<f32>(0.0)));
  let hi = vec3<i32>(min(floor(c + h), vec3<f32>(P.dims) - 1.0));
  if (any(hi < lo)) { return; }
  let ny = hi.y - lo.y + 1;
  let rows = ny * (hi.z - lo.z + 1);
  let s2 = P.support * P.support;
  let a0 = A[0][0];
  for (var r = i32(lid); r < rows; r += 64) {
    let y = lo.y + r % ny;
    let z = lo.z + r / ny;
    let dy = f32(y) - c.y;
    let dz = f32(z) - c.z;
    // m^2(dx) = a0 dx^2 + 2 b dx + cc; keep the x span where it is <= support^2.
    let b = A[0][1] * dy + A[0][2] * dz;
    let cc = A[1][1] * dy * dy + 2.0 * A[1][2] * dy * dz + A[2][2] * dz * dz;
    let disc = b * b - a0 * (cc - s2);
    if (disc < 0.0) { continue; }
    let sq = sqrt(disc);
    let x0 = max(lo.x, i32(ceil(c.x + (-b - sq) / a0)));
    let x1 = min(hi.x, i32(floor(c.x + (-b + sq) / a0)));
    let row = (u32(y) + P.dims.y * u32(z)) * P.dims.x;
    for (var x = x0; x <= x1; x++) {
      let dx = f32(x) - c.x;
      let m2 = max(0.0, a0 * dx * dx + 2.0 * b * dx + cc);
      let q = u32(o * exp(-0.5 * m2) * P.scale + 0.5);
      if (q > 0u) { atomicAdd(&dens[row + u32(x)], q); }
    }
  }
}
`;

/** Density -> solid bits; seed the exterior (grid faces, or every empty point without sealing). */
export const CLASSIFY_WGSL = PARAMS + /* wgsl */ `
@group(0) @binding(1) var<storage, read> dens: array<u32>;
@group(0) @binding(2) var<storage, read_write> solidMask: array<u32>;
@group(0) @binding(3) var<storage, read_write> outside: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
  let w = groupIndex(wid, nwg) * 256u + lid;
  if (w >= P.nBricks * 16u) { return; }
  let brick = w / 16u;
  let bx = brick % P.bdims.x;
  let by = (brick / P.bdims.x) % P.bdims.y;
  let bz = brick / (P.bdims.x * P.bdims.y);
  let isoQ = P.iso * P.scale;
  var sw = 0u;
  var ow = 0u;
  for (var k = 0u; k < 4u; k++) {
    let r = (w % 16u) * 4u + k;
    let y = by * 8u + (r & 7u);
    let z = bz * 8u + (r >> 3u);
    for (var lx = 0u; lx < 8u; lx++) {
      let x = bx * 8u + lx;
      let bit = 1u << (k * 8u + lx);
      if (x >= P.dims.x || y >= P.dims.y || z >= P.dims.z) { ow |= bit; continue; }
      let face = x == 0u || y == 0u || z == 0u || x == P.dims.x - 1u || y == P.dims.y - 1u || z == P.dims.z - 1u;
      let solid = !face && f32(dens[x + P.dims.x * (y + P.dims.y * z)]) >= isoQ;
      if (solid) { sw |= bit; } else if (face || P.fill == 0u) { ow |= bit; }
    }
  }
  solidMask[w] = sw;
  outside[w] = ow;
}
`;

/** One flood step: each workgroup grows the exterior inside one brick to a fixed point. */
export const FLOOD_WGSL = PARAMS + /* wgsl */ `
@group(0) @binding(1) var<storage, read> solidMask: array<u32>;
@group(0) @binding(2) var<storage, read_write> outside: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> changes: atomic<u32>;

var<workgroup> rowsW: array<u32, 64>;
var<workgroup> changedW: atomic<u32>;
var<workgroup> stopW: u32;

fn rowOf(brick: u32, r: u32) -> u32 {
  return (atomicLoad(&outside[brick * 16u + (r >> 2u)]) >> ((r & 3u) * 8u)) & 0xFFu;
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) r: u32) {
  let brick = groupIndex(wid, nwg);
  if (brick >= P.nBricks) { return; }
  let bx = brick % P.bdims.x;
  let by = (brick / P.bdims.x) % P.bdims.y;
  let bz = brick / (P.bdims.x * P.bdims.y);
  let ly = r & 7u;
  let lz = r >> 3u;
  let shift = (r & 3u) * 8u;
  let open = ~(solidMask[brick * 16u + (r >> 2u)] >> shift) & 0xFFu;
  let o0 = rowOf(brick, r);
  // Exterior arriving from neighbouring bricks (read once; anything missed comes next pass).
  // Off-grid neighbours count as exterior.
  var halo = 0u;
  if (bx == 0u) { halo |= 1u; } else { halo |= (rowOf(brick - 1u, r) >> 7u) & 1u; }
  if (bx + 1u == P.bdims.x) { halo |= 0x80u; } else { halo |= (rowOf(brick + 1u, r) & 1u) << 7u; }
  let sy = P.bdims.x;
  let sz = P.bdims.x * P.bdims.y;
  if (ly == 0u) { if (by == 0u) { halo |= 0xFFu; } else { halo |= rowOf(brick - sy, r + 7u); } }
  if (ly == 7u) { if (by + 1u == P.bdims.y) { halo |= 0xFFu; } else { halo |= rowOf(brick + sy, r - 7u); } }
  if (lz == 0u) { if (bz == 0u) { halo |= 0xFFu; } else { halo |= rowOf(brick - sz, r + 56u); } }
  if (lz == 7u) { if (bz + 1u == P.bdims.z) { halo |= 0xFFu; } else { halo |= rowOf(brick + sz, r - 56u); } }

  var o = o0;
  rowsW[r] = o;
  workgroupBarrier();
  for (var it = 0u; it < 64u; it++) {
    var n = o | halo;
    if (ly > 0u) { n |= rowsW[r - 1u]; }
    if (ly < 7u) { n |= rowsW[r + 1u]; }
    if (lz > 0u) { n |= rowsW[r - 8u]; }
    if (lz < 7u) { n |= rowsW[r + 8u]; }
    n &= open;
    for (var k = 0u; k < 7u; k++) { n |= ((n << 1u) | (n >> 1u)) & open; }
    workgroupBarrier();
    rowsW[r] = n;
    if (n != o) { atomicAdd(&changedW, 1u); }
    o = n;
    workgroupBarrier();
    if (r == 0u) { stopW = select(0u, 1u, atomicExchange(&changedW, 0u) == 0u); }
    if (workgroupUniformLoad(&stopW) == 1u) { break; }
  }
  if (o != o0) {
    atomicOr(&outside[brick * 16u + (r >> 2u)], o << shift);
    atomicAdd(&changes, 1u);
  }
}
`;

/** Corner configuration byte per cell (bit c = corner c solid), 4 cells per word. */
export const CONFIG_WGSL = PARAMS + /* wgsl */ `
@group(0) @binding(1) var<storage, read> outside: array<u32>;
@group(0) @binding(2) var<storage, read_write> cfg: array<u32>;

fn solidAt(x: u32, y: u32, z: u32) -> u32 { return 1u - ((outside[maskWord(x, y, z)] >> maskShift(x, y, z)) & 1u); }

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
  let w = groupIndex(wid, nwg) * 256u + lid;
  if (w >= P.nWords) { return; }
  var word = 0u;
  for (var q = 0u; q < 4u; q++) {
    let l = w * 4u + q;
    if (l >= P.nPoints) { break; }
    let p = pointCoord(l);
    if (any(p + 1u >= P.dims)) { continue; } // no cell; grid faces are empty anyway
    var c = 0u;
    for (var k = 0u; k < 8u; k++) {
      c |= solidAt(p.x + (k & 1u), p.y + ((k >> 1u) & 1u), p.z + (k >> 2u)) << k;
    }
    word |= c << (q * 8u);
  }
  cfg[w] = word;
}
`;

// Shared by the count and emit passes: per-cell counts and a 256-wide workgroup scan.
const COUNTS = /* wgsl */ `
@group(0) @binding(1) var<storage, read> cfg: array<u32>;
@group(0) @binding(2) var<storage, read> table: array<u32>; // [256 counts][256 x 36 edge ids][12 edges]
@group(0) @binding(3) var<storage, read_write> blocks: array<vec2<u32>>;

var<workgroup> scanW: array<vec2<u32>, 256>;

fn cfgAt(l: u32) -> u32 { return (cfg[l >> 2u] >> ((l & 3u) * 8u)) & 0xFFu; }
fn vertCount(c: u32) -> u32 { return ((c ^ (c >> 1u)) & 1u) + ((c ^ (c >> 2u)) & 1u) + ((c ^ (c >> 4u)) & 1u); }
/** (vertices, triangles) owned by the 4 points of config word w. */
fn wordCounts(w: u32) -> vec2<u32> {
  var sum = vec2<u32>(0u);
  if (w >= P.nWords) { return sum; }
  let word = cfg[w];
  for (var q = 0u; q < 4u; q++) {
    let c = (word >> (q * 8u)) & 0xFFu;
    sum += vec2<u32>(vertCount(c), table[c]);
  }
  return sum;
}
/** Exclusive scan of v across the workgroup; all invocations must call it. */
fn scanExclusive(lid: u32, v: vec2<u32>) -> vec2<u32> {
  scanW[lid] = v;
  for (var d = 1u; d < 256u; d <<= 1u) {
    workgroupBarrier();
    var add = vec2<u32>(0u);
    if (lid >= d) { add = scanW[lid - d]; }
    workgroupBarrier();
    scanW[lid] += add;
  }
  workgroupBarrier();
  return scanW[lid] - v;
}
`;

export const COUNT_WGSL = PARAMS + COUNTS + /* wgsl */ `
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
  let g = groupIndex(wid, nwg);
  if (g >= P.nBlocks) { return; }
  let v = wordCounts(g * 256u + lid);
  let ex = scanExclusive(lid, v);
  if (lid == 255u) { blocks[g] = ex + v; }
}
`;

export const SCAN_BLOCKS_WGSL = PARAMS + /* wgsl */ `
@group(0) @binding(1) var<storage, read_write> blocks: array<vec2<u32>>;
var<workgroup> scanW: array<vec2<u32>, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) lid: u32) {
  let chunk = (P.nBlocks + 255u) / 256u;
  let start = lid * chunk;
  let end = min(start + chunk, P.nBlocks);
  var sum = vec2<u32>(0u);
  for (var i = start; i < end; i++) { sum += blocks[i]; }
  scanW[lid] = sum;
  for (var d = 1u; d < 256u; d <<= 1u) {
    workgroupBarrier();
    var add = vec2<u32>(0u);
    if (lid >= d) { add = scanW[lid - d]; }
    workgroupBarrier();
    scanW[lid] += add;
  }
  workgroupBarrier();
  var run = scanW[lid] - sum;
  for (var i = start; i < end; i++) {
    let b = blocks[i];
    blocks[i] = run;
    run += b;
  }
  if (lid == 255u) { blocks[P.nBlocks] = scanW[255]; }
}
`;

export const EMIT_VERTS_WGSL = PARAMS + COUNTS + /* wgsl */ `
@group(0) @binding(4) var<storage, read> dens: array<u32>;
@group(0) @binding(5) var<storage, read_write> vbase: array<u32>;
@group(0) @binding(6) var<storage, read_write> positions: array<f32>;
@group(0) @binding(7) var<storage, read_write> normals: array<f32>;

fn D(p: vec3<u32>) -> f32 { return f32(dens[p.x + P.dims.x * (p.y + P.dims.y * p.z)]) / P.scale; }
fn grad(p: vec3<u32>) -> vec3<f32> {
  var g = vec3<f32>(0.0);
  for (var a = 0u; a < 3u; a++) {
    var lo = p;
    var hi = p;
    if (p[a] > 0u) { lo[a] -= 1u; }
    if (p[a] + 1u < P.dims[a]) { hi[a] += 1u; }
    g[a] = (D(hi) - D(lo)) / f32(max(1u, hi[a] - lo[a]));
  }
  return g;
}

/**
 * Where the field crosses iso between grid points p (s = 0) and n (s = 1), on the
 * Catmull-Rom cubic through the four samples along that grid line (the tricubic
 * Catmull-Rom interpolant restricted to the line). Linear interpolation of this steep,
 * non-linear field misplaces vertices by a sub-voxel-phase-dependent amount, which shows
 * as terracing on gently tilted surfaces. Solved by bisection, clamped off the grid points.
 */
fn crossing(d0: f32, d1: f32, d2: f32, d3: f32, solidP: bool) -> f32 {
  // f(s) - iso with f(0) = d1, f(1) = d2; orient so that f(lo) < iso <= f(hi).
  let c0 = d1 - P.iso;
  let c1 = 0.5 * (d2 - d0);
  let c2 = d0 - 2.5 * d1 + 2.0 * d2 - 0.5 * d3;
  let c3 = 0.5 * (d3 - d0) + 1.5 * (d1 - d2);
  if ((d1 >= P.iso) != solidP || (d2 >= P.iso) == solidP) {
    // Sample signs disagree with the classification (sealed cavity or forced-empty grid face).
    return 0.5;
  }
  var lo = 0.0;
  var hi = 1.0;
  for (var i = 0; i < 20; i++) {
    let m = 0.5 * (lo + hi);
    let f = c0 + m * (c1 + m * (c2 + m * c3));
    if ((f >= 0.0) == solidP) { lo = m; } else { hi = m; }
  }
  return clamp(0.5 * (lo + hi), 0.001, 0.999);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
  let g = groupIndex(wid, nwg);
  if (g >= P.nBlocks) { return; }
  let w = g * 256u + lid;
  let v = wordCounts(w);
  var next = blocks[g].x + scanExclusive(lid, v).x;
  if (v.x == 0u) { return; }
  for (var q = 0u; q < 4u; q++) {
    let l = w * 4u + q;
    let c = cfgAt(l);
    if (vertCount(c) == 0u) { continue; }
    vbase[l] = next;
    let p = pointCoord(l);
    let solidP = (c & 1u) == 1u;
    for (var a = 0u; a < 3u; a++) {
      if (((c ^ (c >> (1u << a))) & 1u) == 0u) { continue; }
      var n = p;
      n[a] += 1u;
      var pm = p;
      var nn = n;
      if (p[a] > 0u) { pm[a] -= 1u; }
      if (nn[a] + 1u < P.dims[a]) { nn[a] += 1u; }
      let s = crossing(D(pm), D(p), D(n), D(nn), solidP);
      let x = vec3<f32>(p) + f32(s) * vec3<f32>(n - p);
      var nrm = -mix(grad(p), grad(n), s);
      let len = length(nrm);
      if (len > 1e-12) { nrm /= len; } else if (solidP) { nrm = vec3<f32>(n - p); } else { nrm = -vec3<f32>(n - p); }
      positions[next * 3u] = x.x; positions[next * 3u + 1u] = x.y; positions[next * 3u + 2u] = x.z;
      normals[next * 3u] = nrm.x; normals[next * 3u + 1u] = nrm.y; normals[next * 3u + 2u] = nrm.z;
      next++;
    }
  }
}
`;

export const EMIT_TRIS_WGSL = PARAMS + COUNTS + /* wgsl */ `
@group(0) @binding(4) var<storage, read> vbase: array<u32>;
@group(0) @binding(5) var<storage, read_write> indices: array<u32>;

const TRIS = 256u;
const EDGE_INFO = 256u + 256u * 36u; // per edge: corner | axis << 3

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
  let g = groupIndex(wid, nwg);
  if (g >= P.nBlocks) { return; }
  let w = g * 256u + lid;
  let v = wordCounts(w);
  var next = blocks[g].y + scanExclusive(lid, v).y;
  if (v.y == 0u) { return; }
  let sy = P.dims.x;
  let sz = P.dims.x * P.dims.y;
  for (var q = 0u; q < 4u; q++) {
    let l = w * 4u + q;
    let c = cfgAt(l);
    let nt = table[c];
    for (var t = 0u; t < nt; t++) {
      for (var k = 0u; k < 3u; k++) {
        let info = table[EDGE_INFO + table[TRIS + c * 36u + t * 3u + k]];
        let corner = info & 7u;
        let axis = info >> 3u;
        let owner = l + (corner & 1u) + ((corner >> 1u) & 1u) * sy + (corner >> 2u) * sz;
        // The owner's vertices are stored in axis order; skip its crossings on lower axes.
        let oc = cfgAt(owner);
        var rank = 0u;
        for (var b = 0u; b < axis; b++) { rank += (oc ^ (oc >> (1u << b))) & 1u; }
        indices[next * 3u + k] = vbase[owner] + rank;
      }
      next++;
    }
  }
}
`;
