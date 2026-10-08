import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {buildCalls,phasePositions,summaries} from '../src/calls.mjs';
import {buildDashboard} from '../src/dashboard.mjs';
const archive=JSON.parse(await readFile(new URL('../public/run.json',import.meta.url),'utf8'));
const graph=buildCalls(archive);
test('56 invocations match the original stage, reviewer and worker events',()=>{
  assert.equal(graph.calls.length,56);
  assert.equal(graph.calls.filter(c=>c.operation==='Review task').length,12);
  assert.equal(graph.calls.filter(c=>c.target==='worker').length,32);
  assert.equal(graph.calls.filter(c=>c.source==='orchestrator'&&c.target!=='reviewer').length,12);
  const expected=archive.events.filter(e=>e.event==='stage_started'||e.event==='local_job_process_started'||e.event==='llm_decision_started'&&e.stage.startsWith('review.')).map(e=>e.sequence);
  assert.deepEqual(graph.calls.map(c=>c.sequence),expected);
  assert.deepEqual(graph.calls.map(c=>c.number),Array.from({length:56},(_,i)=>i+1));
});
test('repeated passes are aggregated once, retaining all chronological identifiers',()=>{
  assert.equal(graph.connections.length,15);
  assert.equal(new Set(graph.connections.map(c=>c.id)).size,15);
  const ids=graph.connections.flatMap(c=>c.calls.map(v=>v.id));
  assert.equal(ids.length,56);assert.equal(new Set(ids).size,56);
  assert.equal(graph.connections.find(c=>c.source==='orchestrator'&&c.target==='reporting').calls.length,2);
  assert.equal(graph.connections.find(c=>c.target==='reviewer').calls.length,12);
  assert.equal(graph.connections.find(c=>c.source==='model_search'&&c.target==='worker').calls.length,24);
});
test('directions match orchestrator dispatch and stage-owned subprocess work',()=>{
  for(const c of graph.calls){assert.ok(c.source==='orchestrator'||c.target==='worker');assert.ok(phasePositions[c.source]);assert.ok(phasePositions[c.target]);}
  assert.deepEqual(graph.calls.filter(c=>c.source==='evaluation'&&c.target==='worker').map(c=>c.operation),['Run predictions','Run predictions']);
  assert.equal(graph.calls.filter(c=>c.source==='abstention'&&c.target==='worker').length,6);
  assert.equal(graph.calls.filter(c=>c.source==='reporting').length,0);
});
test('each of the 14 existing agents has a stable rectangle and editorial hover summary',()=>{
  const tasks=buildDashboard(archive).tasks;
  assert.equal(Object.keys(phasePositions).length,14);
  for(const t of tasks){assert.ok(phasePositions[t.id]);assert.equal(summaries[t.id].length,3);assert.doesNotMatch(summaries[t.id].join(' '),/\.json|artefacts\/|sha256/);}
});
