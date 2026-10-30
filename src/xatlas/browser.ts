// Browser UnwrapRunner: xatlas in a Web Worker (worker.ts), started on first use.
import XAtlasWorker from './worker?worker';
import type { UnwrapRequest, UnwrapResult } from './xatlas';

let worker: Worker | null = null;
let nextId = 0;
export function runWorker(request: UnwrapRequest): Promise<UnwrapResult> {
  worker ??= new XAtlasWorker();
  const w = worker, id = nextId++;
  return new Promise((resolve, reject) => {
    const done = (e: MessageEvent<{ id: number; result?: UnwrapResult; error?: string }>) => {
      if (e.data.id !== id) return;
      w.removeEventListener('message', done);
      w.removeEventListener('error', failed);
      if (e.data.result) resolve(e.data.result);
      else reject(new Error(`xatlas: ${e.data.error}`));
    };
    const failed = (e: ErrorEvent) => {
      w.removeEventListener('message', done);
      w.removeEventListener('error', failed);
      if (worker === w) worker = null; // start a fresh worker next time
      w.terminate();
      reject(new Error(`xatlas worker failed: ${e.message}`));
    };
    w.addEventListener('message', done);
    w.addEventListener('error', failed);
    // Copies, not transfers: the caller keeps using the mesh.
    w.postMessage({ id, request });
  });
}
