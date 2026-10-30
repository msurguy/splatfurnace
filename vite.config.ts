import { defineConfig, type Plugin } from 'vite';
import tailwindcss from '@tailwindcss/vite';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

// Dev-only convenience: lists Gaussian-splat PLYs in ~/Downloads as one-click samples.
function localSamples(): Plugin {
  const dir = path.join(os.homedir(), 'Downloads');
  const isSplat = (file: string) => {
    const fd = fs.openSync(file, 'r');
    const head = Buffer.alloc(4096);
    fs.readSync(fd, head, 0, 4096, 0);
    fs.closeSync(fd);
    const text = head.toString('latin1');
    return text.startsWith('ply') && text.includes('scale_0') && text.includes('rot_0');
  };
  return {
    name: 'local-samples',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use('/__samples', (req, res) => {
        const name = decodeURIComponent((req.url ?? '/').slice(1));
        if (!name) {
          let list: string[] = [];
          try {
            list = fs.readdirSync(dir).filter((f) => f.toLowerCase().endsWith('.ply') && isSplat(path.join(dir, f)));
          } catch { /* no Downloads folder */ }
          res.setHeader('content-type', 'application/json');
          res.end(JSON.stringify(list));
          return;
        }
        const file = path.join(dir, path.basename(name));
        if (!file.toLowerCase().endsWith('.ply') || !fs.existsSync(file)) {
          res.statusCode = 404;
          res.end();
          return;
        }
        res.setHeader('content-type', 'application/octet-stream');
        fs.createReadStream(file).pipe(res);
      });
    },
  };
}

export default defineConfig({
  plugins: [tailwindcss(), localSamples()],
  server: { port: 5199 },
  build: { target: 'es2022' },
});
