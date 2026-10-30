// Runs xatlas off the main thread. One request at a time; the module is loaded once.
import { XAtlas, unwrap, type UnwrapRequest } from './xatlas';

let atlas: Promise<XAtlas> | null = null;

self.onmessage = async (e: MessageEvent<{ id: number; request: UnwrapRequest }>) => {
  const { id, request } = e.data;
  try {
    atlas ??= XAtlas.load();
    const result = unwrap(await atlas, request);
    postMessage({ id, result }, { transfer: [result.xref.buffer, result.uv.buffer, result.indices.buffer] });
  } catch (err) {
    atlas = null; // a failed load (or a trapped module) is retried on the next request
    postMessage({ id, error: err instanceof Error ? err.message : String(err) });
  }
};
