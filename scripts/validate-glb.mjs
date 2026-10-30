// Validate exported GLBs with the Khronos glTF validator: node scripts/validate-glb.mjs file.glb ...
import fs from 'node:fs';
import validator from 'gltf-validator';
for (const f of process.argv.slice(2)) {
  const r = await validator.validateBytes(new Uint8Array(fs.readFileSync(f)), { maxIssues: 20 });
  const i = r.issues;
  console.log(f.split('/').pop(), `errors ${i.numErrors} warnings ${i.numWarnings} infos ${i.numInfos}`,
    i.messages.filter((m) => m.severity <= 1).map((m) => `${m.code}: ${m.message}`).slice(0, 4).join(' | '));
}
