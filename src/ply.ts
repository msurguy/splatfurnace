// Gaussian-splat PLY loader (standard 3DGS layout: x,y,z, f_dc_*, opacity, scale_*, rot_*).
// Decodes activations: opacity = sigmoid, scale = exp, rotation normalised (w,x,y,z),
// colour = 0.5 + SH_C0 * f_dc (display / sRGB-like space, as splat renderers show it).

export interface Splats {
  count: number;
  centers: Float32Array; // xyz
  scales: Float32Array; // linear sigma per local axis
  rotations: Float32Array; // normalised quaternion w,x,y,z
  opacities: Float32Array; // 0..1
  colors: Float32Array; // rgb 0..1, display space
}

const SH_C0 = 0.28209479177387814;

const TYPE_SIZES: Record<string, number> = {
  char: 1, int8: 1, uchar: 1, uint8: 1,
  short: 2, int16: 2, ushort: 2, uint16: 2,
  int: 4, int32: 4, uint: 4, uint32: 4,
  float: 4, float32: 4, double: 8, float64: 8,
};

type Reader = (view: DataView, offset: number, little: boolean) => number;

const READERS: Record<string, Reader> = {
  char: (v, o) => v.getInt8(o), int8: (v, o) => v.getInt8(o),
  uchar: (v, o) => v.getUint8(o), uint8: (v, o) => v.getUint8(o),
  short: (v, o, l) => v.getInt16(o, l), int16: (v, o, l) => v.getInt16(o, l),
  ushort: (v, o, l) => v.getUint16(o, l), uint16: (v, o, l) => v.getUint16(o, l),
  int: (v, o, l) => v.getInt32(o, l), int32: (v, o, l) => v.getInt32(o, l),
  uint: (v, o, l) => v.getUint32(o, l), uint32: (v, o, l) => v.getUint32(o, l),
  float: (v, o, l) => v.getFloat32(o, l), float32: (v, o, l) => v.getFloat32(o, l),
  double: (v, o, l) => v.getFloat64(o, l), float64: (v, o, l) => v.getFloat64(o, l),
};

export function parseSplatPly(buffer: ArrayBuffer): Splats {
  const bytes = new Uint8Array(buffer);
  const headerEnd = findHeaderEnd(bytes);
  const header = new TextDecoder().decode(bytes.subarray(0, headerEnd));
  const lines = header.split(/\r?\n/);
  if (lines[0].trim() !== 'ply') throw new Error('Not a PLY file.');

  let format = '';
  let count = 0;
  let inVertex = false;
  const props: { name: string; type: string; offset: number }[] = [];
  let stride = 0;
  for (const line of lines) {
    const parts = line.trim().split(/\s+/);
    if (parts[0] === 'format') format = parts[1];
    else if (parts[0] === 'element') {
      inVertex = parts[1] === 'vertex';
      if (inVertex) count = parseInt(parts[2], 10);
      else if (count > 0 && props.length) break; // ignore elements after vertex
    } else if (parts[0] === 'property' && inVertex) {
      if (parts[1] === 'list') throw new Error('List properties on vertices are not supported.');
      const size = TYPE_SIZES[parts[1]];
      if (!size) throw new Error(`Unknown PLY type ${parts[1]}`);
      props.push({ name: parts[2], type: parts[1], offset: stride });
      stride += size;
    }
  }
  if (format !== 'binary_little_endian' && format !== 'binary_big_endian') {
    throw new Error(`PLY format "${format}" is not supported; expected a binary Gaussian splat PLY.`);
  }
  const byName = new Map(props.map((p) => [p.name, p]));
  for (const required of ['x', 'y', 'z', 'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']) {
    if (!byName.has(required)) {
      throw new Error(`This PLY has no "${required}" property, so it is not a Gaussian splat file (it may be a regular mesh or point cloud).`);
    }
  }
  const hasDc = byName.has('f_dc_0');
  const hasRgb = byName.has('red');
  const little = format === 'binary_little_endian';
  const view = new DataView(buffer, headerEnd);
  if (view.byteLength < count * stride) throw new Error('PLY file is truncated.');

  const field = (name: string) => {
    const p = byName.get(name)!;
    return { offset: p.offset, read: READERS[p.type] };
  };
  const f = {
    x: field('x'), y: field('y'), z: field('z'), op: field('opacity'),
    s0: field('scale_0'), s1: field('scale_1'), s2: field('scale_2'),
    r0: field('rot_0'), r1: field('rot_1'), r2: field('rot_2'), r3: field('rot_3'),
  };
  const dc = hasDc ? [field('f_dc_0'), field('f_dc_1'), field('f_dc_2')] : null;
  const rgb = !hasDc && hasRgb ? [field('red'), field('green'), field('blue')] : null;

  const out: Splats = {
    count,
    centers: new Float32Array(count * 3),
    scales: new Float32Array(count * 3),
    rotations: new Float32Array(count * 4),
    opacities: new Float32Array(count),
    colors: new Float32Array(count * 3),
  };
  for (let i = 0; i < count; i++) {
    const base = i * stride;
    const r = (a: { offset: number; read: Reader }) => a.read(view, base + a.offset, little);
    out.centers[i * 3] = r(f.x);
    out.centers[i * 3 + 1] = r(f.y);
    out.centers[i * 3 + 2] = r(f.z);
    out.opacities[i] = 1 / (1 + Math.exp(-r(f.op)));
    out.scales[i * 3] = Math.exp(r(f.s0));
    out.scales[i * 3 + 1] = Math.exp(r(f.s1));
    out.scales[i * 3 + 2] = Math.exp(r(f.s2));
    let qw = r(f.r0), qx = r(f.r1), qy = r(f.r2), qz = r(f.r3);
    const len = Math.hypot(qw, qx, qy, qz) || 1;
    qw /= len; qx /= len; qy /= len; qz /= len;
    out.rotations.set([qw, qx, qy, qz], i * 4);
    for (let c = 0; c < 3; c++) {
      let v = 0.5;
      if (dc) v = 0.5 + SH_C0 * r(dc[c]);
      else if (rgb) v = r(rgb[c]) / 255;
      out.colors[i * 3 + c] = Math.min(1, Math.max(0, v));
    }
  }
  return out;
}

function findHeaderEnd(bytes: Uint8Array): number {
  const marker = 'end_header';
  const limit = Math.min(bytes.length, 64 * 1024);
  outer: for (let i = 0; i < limit - marker.length; i++) {
    for (let j = 0; j < marker.length; j++) if (bytes[i + j] !== marker.charCodeAt(j)) continue outer;
    let end = i + marker.length;
    if (bytes[end] === 0x0d) end++;
    if (bytes[end] === 0x0a) end++;
    return end;
  }
  throw new Error('PLY header has no end_header.');
}

/** Rotation matrix columns (local axes in world space) of a w,x,y,z quaternion. */
export function quatAxes(w: number, x: number, y: number, z: number): [number[], number[], number[]] {
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)],
    [2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)],
    [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)],
  ];
}
