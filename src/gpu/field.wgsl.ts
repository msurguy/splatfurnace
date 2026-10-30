// Splat ray evaluation. For each query (point p, normal n, range r) a ray starts at
// p + n*r and travels along -n for 2r. Every Gaussian it passes is evaluated at its
// maximum response along the ray (closed form), the hits are depth-sorted and
// alpha-composited front to back - i.e. what a splat renderer shows when looking at
// the surface head-on. Hidden interior splats never leak into the colour.
export const MAX_HITS = 32;

export const FIELD_WGSL = /* wgsl */ `
struct Params {
  buckets: vec4<u32>, // bucket dims xyz, query count
  tune: vec4<f32>,    // bucket size (voxels), min alpha, unused, unused
}

struct Splat {
  c: vec4<f32>,
  a0: vec4<f32>,
  a1: vec4<f32>,
  a2: vec4<f32>,
  col: vec4<f32>,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> splats: array<Splat>;
@group(0) @binding(2) var<storage, read> starts: array<u32>;
@group(0) @binding(3) var<storage, read> refs: array<u32>;
@group(0) @binding(4) var<storage, read> queries: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> results: array<vec4<f32>>;

const MAXH: u32 = ${MAX_HITS}u;

fn rows(s: Splat) -> mat3x3<f32> {
  // columns of this matrix are the rows of M (axis / sigma); use transpose to apply M
  let sg = max(vec3<f32>(s.a0.w, s.a1.w, s.a2.w), vec3<f32>(1e-3));
  return mat3x3<f32>(s.a0.xyz / sg.x, s.a1.xyz / sg.y, s.a2.xyz / sg.z);
}

@compute @workgroup_size(64)
fn sampleSplats(@builtin(global_invocation_id) gid: vec3<u32>) {
  let qi = gid.x + gid.y * 65535u * 64u;
  if (qi >= P.buckets.w) { return; }
  let q0 = queries[qi * 2u];
  let q1 = queries[qi * 2u + 1u];
  let range = q0.w;
  let n = normalize(q1.xyz);
  let origin = q0.xyz + n * range;
  let dir = -n;
  let tmax = 2.0 * range;

  var ht: array<f32, MAXH>;
  var ha: array<f32, MAXH>;
  var hid: array<u32, MAXH>;
  var count = 0u;

  let bs = P.tune.x;
  let bdims = vec3<i32>(P.buckets.xyz);
  var lastBucket = 0xffffffffu;
  let stepLen = bs * 0.4;
  let steps = u32(ceil(tmax / stepLen)) + 1u;
  for (var st = 0u; st <= steps; st++) {
    let x = origin + dir * min(f32(st) * stepLen, tmax);
    let b = vec3<i32>(floor(x / bs));
    if (any(b < vec3<i32>(0)) || any(b >= bdims)) { continue; }
    let bi = u32(b.x + bdims.x * (b.y + bdims.y * b.z));
    if (bi == lastBucket) { continue; }
    lastBucket = bi;
    for (var r = starts[bi]; r < starts[bi + 1u]; r++) {
      let id = refs[r];
      let s = splats[id];
      let M = transpose(rows(s));
      let u = M * (origin - s.c.xyz);
      let v = M * dir;
      let vv = max(dot(v, v), 1e-12);
      let uv = dot(u, v);
      let t = -uv / vv;
      if (t < -0.5 || t > tmax + 0.5) { continue; }
      let m2 = dot(u, u) - uv * uv / vv;
      if (m2 > 9.0) { continue; }
      let a = min(s.c.w * exp(-0.5 * m2), 0.999);
      if (a < P.tune.y) { continue; }
      // dedupe: a splat can live in several buckets along the ray
      var seen = false;
      for (var k = 0u; k < count; k++) { if (hid[k] == id) { seen = true; break; } }
      if (seen) { continue; }
      // sorted insert, dropping the farthest hit when full
      if (count == MAXH) {
        if (t >= ht[MAXH - 1u]) { continue; }
        count -= 1u;
      }
      var k = count;
      while (k > 0u && ht[k - 1u] > t) {
        ht[k] = ht[k - 1u]; ha[k] = ha[k - 1u]; hid[k] = hid[k - 1u];
        k -= 1u;
      }
      ht[k] = t; ha[k] = a; hid[k] = id;
      count += 1u;
    }
  }

  var T = 1.0;
  var color = vec3<f32>(0.0);
  var weight = 0.0;
  var tHalf = -1.0;
  var tBest = 0.0;
  var wBest = 0.0;
  for (var k = 0u; k < count; k++) {
    let w = T * ha[k];
    color += w * splats[hid[k]].col.xyz;
    weight += w;
    if (w > wBest) { wBest = w; tBest = ht[k]; }
    if (tHalf < 0.0 && T * (1.0 - ha[k]) <= 0.5) {
      // T crosses 0.5 inside this splat; place the crossing by its share of alpha
      let need = (T - 0.5) / max(T * ha[k], 1e-6);
      let sigmaT = 1.0 / max(length(transpose(rows(splats[hid[k]])) * dir), 1e-6);
      tHalf = ht[k] + (need - 0.5) * 2.0 * sigmaT;
    }
    T *= 1.0 - ha[k];
    if (T < 0.004) { break; }
  }

  // Density gradient at the visible surface gives a smooth, coherent normal.
  let tn = select(tBest, tHalf, tHalf >= 0.0);
  let xh = origin + dir * tn;
  var grad = vec3<f32>(0.0);
  for (var k = 0u; k < count; k++) {
    let s = splats[hid[k]];
    let R = rows(s);
    let l = transpose(R) * (xh - s.c.xyz);
    let g = s.c.w * exp(-0.5 * dot(l, l));
    grad -= g * (R * l);
  }
  var normal = n;
  if (dot(grad, grad) > 1e-12) { normal = -normalize(grad); }
  if (dot(normal, n) < 0.0) { normal = n; } // never let a back-side gradient flip the shading

  let coverage = 1.0 - T;
  var rgb = vec3<f32>(-1.0);
  if (weight > 1e-6) { rgb = color / weight; }
  var offset = 1e9;
  if (tHalf >= 0.0) { offset = range - tHalf; }
  results[qi * 2u] = vec4<f32>(rgb, coverage);
  results[qi * 2u + 1u] = vec4<f32>(normal, offset);
}
`;
