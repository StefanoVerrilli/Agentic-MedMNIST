import {readFile,readdir,mkdir,writeFile} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {readRun} from '../src/dashboard.mjs';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const source=path.resolve(process.argv[2] ?? path.join(root,'../runs/pathmnist_20261008T090954174041Z/seed_42'));
const output=path.resolve(process.argv[3] ?? path.join(root,'public/run.json'));
const files={};
const warnings=[];
async function collect(dir) {
  let entries;
  try {entries=await readdir(path.join(source,dir),{withFileTypes:true});}catch(e){if(e.code==='ENOENT'){warnings.push(`Optional evidence directory missing: ${dir}`);return;}throw e;}
  for (const entry of entries) {
    const relative=path.posix.join(dir,entry.name);
    if (entry.isDirectory()) await collect(relative);
    else if (/\.(json|md)$/.test(entry.name)) {const text=await readFile(path.join(source,relative),'utf8');files[relative]=relative.endsWith('.md') ? text : JSON.parse(text);}
  }
}
await collect('artefacts'); await collect('blobs/reasoning');
for (const file of await readdir(source)) if (/^report_summary.*\.md$/.test(file)) files[file]=await readFile(path.join(source,file),'utf8');
let dossier={};
try {dossier=JSON.parse(await readFile(path.join(source,'dossier.json'),'utf8'));}catch(e){if(e.code==='ENOENT')warnings.push('Optional dossier.json missing.');else throw e;}
const events=(await readFile(path.join(source,'decision_log.jsonl'),'utf8')).split(/\r?\n/).filter(line=>line.trim()).map(line=>JSON.parse(line));
const archive={format:'agent-run-explorer-v1',name:`${path.basename(path.dirname(source))} / ${path.basename(source)}`,events,dossier:{run_id:dossier.run_id,status:dossier.status,artefact_registry:dossier.artefact_registry},files,warnings};
const checked=readRun(archive);
await mkdir(path.dirname(output),{recursive:true});await writeFile(output,JSON.stringify(archive));
console.log(`Prepared ${checked.events.length} events, ${checked.decisions.length} decisions, ${Object.keys(files).length} internal evidence files → ${output}`);
