import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {buildDashboard,readRun} from '../src/dashboard.mjs';
const archive=JSON.parse(await readFile(new URL('../public/run.json',import.meta.url),'utf8'));
const model=buildDashboard(archive);
test('14 tasks account for all 28 decisions exactly once in their editorial briefs',()=>{
  assert.equal(model.tasks.length,14);
  const sequences=model.tasks.flatMap(t=>t.pages.flatMap(p=>p.decisionSequences));
  assert.equal(sequences.length,28);assert.equal(new Set(sequences).size,28);
  assert.deepEqual([...sequences].sort((a,b)=>a-b),readRun(archive).decisions.map(e=>e.sequence).sort((a,b)=>a-b));
  for(const task of model.tasks)for(const page of task.pages){assert.ok(page.sources.length);assert.ok(page.goal&&page.choice&&page.outcome);}
});
test('statistics agree with the selected checkpoint and the held-out evaluation',()=>{
  assert.deepEqual(model.stats,{decisions:28,trials:8,validation:0.995902,test:0.917827,macroF1:0.890046,review:2206});
  const search=model.tasks.find(t=>t.id==='model_search');
  const overview=search.pages.find(p=>p.view==='statistics');
  assert.equal(overview.table.rows.length,8);
  assert.equal(overview.table.rows[6][0],'7 ★');
  assert.equal(overview.table.rows[5][4],'99.680%');
  assert.equal(overview.table.rows[6][2],'99.590%');
  const classes=model.tasks.find(t=>t.id==='evaluation').pages.find(p=>p.table?.columns.includes('Tissue class'));
  assert.equal(classes.table.rows.length,9);
  assert.deepEqual(classes.table.rows.find(row=>row[0]==='cancer-associated stroma'),['cancer-associated stroma','92.8%','52.0%','0.667','421']);
});
test('tables are separate bounded screens and raw evidence is never used as description',()=>{
  for(const task of model.tasks)for(const page of task.pages){
    if(page.view==='summary'){assert.equal(page.table,undefined);assert.equal(page.chart,undefined);}
    if(page.view==='statistics'){assert.ok(page.table);assert.ok(page.table.rows.length<=9);assert.equal(page.chart,undefined);}
    const text=[page.title,page.goal,page.choice,page.outcome,page.note??''].join(' ');
    assert.doesNotMatch(text,/\.jsonl|\.json|\.yaml|\.ckpt|sha256|artefacts\//i);
  }
});
test('failed proposal, recovery and report correction remain part of the story',()=>{
  const search=model.tasks.find(t=>t.id==='model_search');
  assert.match(search.pages.at(-1).choice,/ninth proposal failed validation/);
  assert.match(search.pages.at(-1).outcome,/not a ninth experiment/);
  const reporting=model.tasks.find(t=>t.id==='reporting');
  assert.match(reporting.pages[0].outcome,/28 lymphocyte false positives/);
  assert.match(model.tasks.find(t=>t.id==='orchestrator').pages.map(p=>p.outcome).join(' '),/both false/);
});
test('missing run data gives an error rather than invented statistics',()=>{
  assert.throws(()=>readRun({}),/could not be read/);
  assert.throws(()=>readRun({...archive,files:{}}),/incomplete/);
});
