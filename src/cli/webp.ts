// WebP encoding with sharp (libwebp, the same encoder browsers use), in place of OffscreenCanvas.
import sharp from 'sharp';
import { stripWebpMetadata, type WebpEncoder } from '../export';

export const encodeWebpNode: WebpEncoder = async (img, quality) => {
  const raw = Buffer.from(img.rgba.buffer, img.rgba.byteOffset, img.rgba.byteLength);
  const webp = await sharp(raw, { raw: { width: img.size, height: img.size, channels: 4 } })
    .removeAlpha() // alpha is unused; keeps WebP from storing it
    .webp({ quality: Math.round(quality * 100) })
    .toBuffer();
  return stripWebpMetadata(new Uint8Array(webp.buffer, webp.byteOffset, webp.byteLength));
};
