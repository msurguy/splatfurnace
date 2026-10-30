// Node UnwrapRunner: xatlas in-thread. The CLI converts one file at a time, so a worker
// would only add copies. The module is loaded once and reloaded after a failure.
import { fileURLToPath } from 'node:url';
import { XAtlas, unwrap, type UnwrapRequest, type UnwrapResult } from './xatlas';

let atlas: Promise<XAtlas> | null = null;

export async function runInThread(request: UnwrapRequest): Promise<UnwrapResult> {
  try {
    atlas ??= XAtlas.load(fileURLToPath(new URL('./xatlas-wasm.wasm', import.meta.url)));
    return unwrap(await atlas, request);
  } catch (err) {
    atlas = null; // a failed load (or a trapped module) is retried on the next request
    throw new Error(`xatlas: ${err instanceof Error ? err.message : String(err)}`);
  }
}
