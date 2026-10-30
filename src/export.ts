// GLB writer: builds the glTF directly from the mesh data and baked texels with glTF-Transform,
// so textures are encoded once (WebP) and geometry can be welded, quantized and meshopt-compressed.
// Targets Blender 5.2+, three.js and Babylon.js; with `compress` off the file also opens in Blender 4.2+.
import { Document, WebIO, type mat4, type Texture } from '@gltf-transform/core';
import { EXTMeshoptCompression, EXTTextureWebP, KHRMaterialsUnlit, KHRMeshQuantization } from '@gltf-transform/extensions';
import { meshopt, quantize, reorder, weld } from '@gltf-transform/functions';
import { MeshoptEncoder } from 'meshoptimizer';

export interface ExportImage {
  rgba: Uint8Array | Uint8ClampedArray; // size x size, row 0 = v 0 (glTF: top of the image)
  size: number;
}

export interface ExportInput {
  name: string;
  matrix: number[]; // column-major node transform
  positions: Float32Array;
  normals?: Float32Array;
  uvs?: Float32Array;
  tangents?: Float32Array; // xyzw
  colors?: Float32Array; // linear rgb per vertex
  indices?: Uint32Array;
  colorTexture?: ExportImage;
  normalTexture?: ExportImage;
  unlit: boolean;
  compress: boolean; // EXT_meshopt_compression (needs a meshopt decoder; Blender 5.2+)
  quality?: { color?: number; normal?: number }; // WebP quality 0..1
  normalSize?: number; // downsample the normal map to this size (power-of-two fraction of its size)
}

/** Export settings exposed to the UI and the s2m scripting hook. */
export type ExportOptions = Pick<ExportInput, 'unlit' | 'compress' | 'quality'> & { halfNormalMap?: boolean };

// WebP qualities chosen by measurement; see README "Export". Lower colour quality darkens
// near-black texels enough to shrink silhouettes; the normal map loses ~10 deg per texel to
// chroma subsampling at any lossy setting, but lit renders stay within ~41 dB of the original.
const COLOR_QUALITY = 0.9;
const NORMAL_QUALITY = 0.9;
// Bits per component. UVs keep 16 bits: atlas cells are ~16 texels, so 12 bits (0.5 texel
// at 2048^2) would let samples cross chart gutters.
const BITS = { quantizePosition: 14, quantizeNormal: 8, quantizeTexcoord: 16, quantizeColor: 8 };

export async function encodeWebp(img: ExportImage, quality: number): Promise<Uint8Array> {
  const opaque = new Uint8ClampedArray(img.rgba);
  for (let i = 3; i < opaque.length; i += 4) opaque[i] = 255; // alpha unused; keeps WebP from storing it
  const canvas = new OffscreenCanvas(img.size, img.size);
  canvas.getContext('2d')!.putImageData(new ImageData(opaque, img.size, img.size), 0, 0);
  const blob = await canvas.convertToBlob({ type: 'image/webp', quality });
  if (blob.type !== 'image/webp') throw new Error('This browser cannot encode WebP.');
  return stripWebpMetadata(new Uint8Array(await blob.arrayBuffer()));
}

/**
 * Box-filter a tangent-space normal map down by an integer factor, averaging the decoded
 * vectors and renormalising. Atlas gutters are dilated before export, so averaging across a
 * chart edge only mixes in copies of the edge texels.
 */
export function downsampleNormals(img: ExportImage, size: number): ExportImage {
  const f = img.size / size;
  if (f <= 1 || !Number.isInteger(f)) return img;
  const out = new Uint8ClampedArray(size * size * 4);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let nx = 0, ny = 0, nz = 0;
      for (let j = 0; j < f; j++) {
        for (let i = 0; i < f; i++) {
          const s = ((y * f + j) * img.size + x * f + i) * 4;
          nx += img.rgba[s] / 127.5 - 1; ny += img.rgba[s + 1] / 127.5 - 1; nz += img.rgba[s + 2] / 127.5 - 1;
        }
      }
      const l = Math.hypot(nx, ny, nz) || 1, d = (y * size + x) * 4;
      out[d] = (nx / l + 1) * 127.5; out[d + 1] = (ny / l + 1) * 127.5; out[d + 2] = (nz / l + 1) * 127.5; out[d + 3] = 255;
    }
  }
  return { rgba: out, size };
}

/**
 * Drop ICC/EXIF/XMP chunks. Browsers embed an sRGB ICC profile, but glTF textures must not
 * carry colour-space information (a colour-managed decoder could alter normal-map values).
 */
function stripWebpMetadata(webp: Uint8Array): Uint8Array {
  const view = new DataView(webp.buffer, webp.byteOffset, webp.byteLength);
  const chunks: { id: string; start: number; end: number }[] = [];
  for (let o = 12; o + 8 <= webp.length;) {
    const size = view.getUint32(o + 4, true);
    const end = Math.min(webp.length, o + 8 + size + (size & 1));
    chunks.push({ id: String.fromCharCode(...webp.subarray(o, o + 4)), start: o, end });
    o = end;
  }
  const meta = new Set(['ICCP', 'EXIF', 'XMP ']);
  if (!chunks.some((c) => meta.has(c.id))) return webp;
  const image = chunks.filter((c) => c.id === 'VP8 ' || c.id === 'VP8L');
  // Without alpha or animation the simple format (one image chunk) suffices; otherwise keep
  // VP8X and clear its ICC/EXIF/XMP flags.
  const simple = image.length === 1 && chunks.every((c) => c.id === 'VP8X' || meta.has(c.id) || c === image[0]);
  const kept = chunks.filter((c) => !meta.has(c.id) && (!simple || c.id !== 'VP8X'));
  const out = new Uint8Array(12 + kept.reduce((n, c) => n + c.end - c.start, 0));
  out.set(webp.subarray(0, 12));
  let o = 12;
  for (const c of kept) {
    out.set(webp.subarray(c.start, c.end), o);
    if (c.id === 'VP8X') out[o + 8] &= ~(0x20 | 0x08 | 0x04); // ICC, EXIF, XMP flags
    o += c.end - c.start;
  }
  new DataView(out.buffer).setUint32(4, out.length - 8, true);
  return out;
}

export async function writeGlb(input: ExportInput): Promise<Uint8Array> {
  const doc = new Document();
  const buffer = doc.createBuffer();
  const prim = doc.createPrimitive();
  const attr = (name: string, array: Float32Array, type: 'VEC2' | 'VEC3' | 'VEC4') =>
    prim.setAttribute(name, doc.createAccessor().setType(type).setArray(array).setBuffer(buffer));
  attr('POSITION', input.positions, 'VEC3');
  // Unlit materials never read normals or tangents, so they are left out.
  if (input.normals && !input.unlit) attr('NORMAL', input.normals, 'VEC3');
  if (input.tangents && !input.unlit && input.normalTexture) attr('TANGENT', input.tangents, 'VEC4');
  if (input.uvs) attr('TEXCOORD_0', input.uvs, 'VEC2');
  if (input.colors) attr('COLOR_0', input.colors, 'VEC3');
  if (input.indices) prim.setIndices(doc.createAccessor().setType('SCALAR').setArray(input.indices).setBuffer(buffer));

  const material = doc.createMaterial(input.name).setRoughnessFactor(0.9).setMetallicFactor(0);
  const texture = async (img: ExportImage, quality: number, label: string): Promise<Texture> => {
    doc.createExtension(EXTTextureWebP).setRequired(true);
    return doc.createTexture(label).setImage(await encodeWebp(img, quality)).setMimeType('image/webp');
  };
  if (input.colorTexture) material.setBaseColorTexture(await texture(input.colorTexture, input.quality?.color ?? COLOR_QUALITY, 'baseColor'));
  if (input.normalTexture && !input.unlit) {
    const img = input.normalSize ? downsampleNormals(input.normalTexture, input.normalSize) : input.normalTexture;
    material.setNormalTexture(await texture(img, input.quality?.normal ?? NORMAL_QUALITY, 'normal'));
  }
  if (input.unlit) material.setExtension('KHR_materials_unlit', doc.createExtension(KHRMaterialsUnlit).createUnlit());
  prim.setMaterial(material);

  const mesh = doc.createMesh(input.name).addPrimitive(prim);
  const node = doc.createNode(input.name).setMesh(mesh).setMatrix(input.matrix as mat4);
  doc.createScene().addChild(node);

  await MeshoptEncoder.ready;
  await doc.transform(weld());
  if (input.compress) {
    await doc.transform(meshopt({ encoder: MeshoptEncoder, level: 'high', ...BITS }));
  } else {
    await doc.transform(reorder({ encoder: MeshoptEncoder, target: 'size' }), quantize(BITS));
  }
  const io = new WebIO()
    .registerExtensions([EXTMeshoptCompression, EXTTextureWebP, KHRMaterialsUnlit, KHRMeshQuantization])
    .registerDependencies({ 'meshopt.encoder': MeshoptEncoder });
  return io.writeBinary(doc);
}
