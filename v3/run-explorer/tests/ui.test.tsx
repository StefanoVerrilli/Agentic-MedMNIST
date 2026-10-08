import React from 'react';
import {readFileSync} from 'node:fs';
import {afterEach,beforeEach,describe,it,expect,vi} from 'vitest';
import {render,screen,fireEvent,act,cleanup} from '@testing-library/react';
import {buildDashboard} from '../src/dashboard.mjs';
vi.mock('recharts',()=>({ResponsiveContainer:({children}:any)=><div>{children}</div>,AreaChart:()=>null,Area:()=>null,XAxis:()=>null,YAxis:()=>null,Tooltip:()=>null,CartesianGrid:()=>null}));
vi.mock('@xyflow/react',()=>({ReactFlowProvider:({children}:any)=><>{children}</>,ReactFlow:({nodes,edges,nodeTypes,children}:any)=><div aria-label="Call diagram">{nodes.map((n:any)=><div key={n.id}>{React.createElement(nodeTypes.agent,{id:n.id,data:n.data})}</div>)}{edges.map((e:any)=><button key={e.id} aria-label={`Connection: ${e.id}`} onClick={()=>e.data.inspect(e.id)}>{e.data.label}</button>)}{children}</div>,Background:()=>null,Controls:()=>null,Handle:()=>null,BaseEdge:()=>null,EdgeLabelRenderer:({children}:any)=><>{children}</>,Position:{Top:'top',Left:'left',Bottom:'bottom',Right:'right'},MarkerType:{ArrowClosed:'closed'},useReactFlow:()=>({fitView:vi.fn()})}));
import {App} from '../src/main';
const archive=JSON.parse(readFileSync('public/run.json','utf8'));
const model=buildDashboard(archive);
beforeEach(()=>{window.history.replaceState(null,'','/');vi.stubGlobal('fetch',vi.fn(async()=>({ok:true,json:async()=>structuredClone(archive)})));});
afterEach(()=>{cleanup();vi.useRealTimers();vi.unstubAllGlobals();});
async function open(){render(<App/>);await screen.findByRole('button',{name:'Data ingestion'});}
describe('dashboard navigation',()=>{
  it('starts at ingestion and removes replay, imports and raw data surfaces',async()=>{
    await open();expect(screen.getByRole('button',{name:'Data ingestion'}).getAttribute('aria-pressed')).toBe('true');
    expect((screen.getByRole('button',{name:'Previous step'}) as HTMLButtonElement).disabled).toBe(true);
    for(const name of ['Start replay','Import archive','Evidence','Prompt'])expect(screen.queryByRole('button',{name})).toBeNull();
    expect(document.querySelector('pre')).toBeNull();expect(screen.getByText('91.78%')).toBeTruthy();
  });
  it('next and previous steps work without automatic advancement',async()=>{
    await open();fireEvent.click(screen.getByRole('button',{name:'Next step'}));expect(screen.getByRole('table')).toBeTruthy();
    vi.useFakeTimers();await act(async()=>vi.advanceTimersByTime(180000));expect(screen.getByRole('table')).toBeTruthy();
    fireEvent.click(screen.getByRole('button',{name:'Previous step'}));expect(screen.queryByRole('table')).toBeNull();
  });
  it('keyboard arrows stop at boundaries and selecting an agent resets its step',async()=>{
    await open();const length=model.tasks[0].pages.length;
    for(let i=0;i<length+3;i++)fireEvent.keyDown(window,{key:'ArrowRight'});
    expect((screen.getByRole('button',{name:'Next step'}) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(screen.getByRole('button',{name:'Evaluator'}));expect(screen.getByRole('heading',{name:'Strong validation, lower separate-center accuracy'})).toBeTruthy();
    expect((screen.getByRole('button',{name:'Previous step'}) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.keyDown(window,{key:'ArrowLeft'});expect(screen.getByRole('heading',{name:'Strong validation, lower separate-center accuracy'})).toBeTruthy();
  });
  it('renders every task step without exposing filenames or dropping result tables',async()=>{
    await open();
    for(const task of model.tasks){fireEvent.click(screen.getByRole('button',{name:task.name}));for(let i=0;i<task.pages.length;i++){
      const page=task.pages[i];expect(screen.getByRole('heading',{name:page.title})).toBeTruthy();
      if(page.view==='statistics')expect(screen.getByRole('table').querySelectorAll('tbody tr').length).toBe(page.table.rows.length);
      expect(document.body.textContent).not.toMatch(/\.json|\.yaml|artefacts\/|sha256/);
      if(i<task.pages.length-1)fireEvent.click(screen.getByRole('button',{name:'Next step'}));
    }}
  });
  it('shows a readable loading failure',async()=>{
    vi.stubGlobal('fetch',vi.fn(async()=>({ok:false})));render(<App/>);expect(await screen.findByRole('alert')).toBeTruthy();
  });
  it('links to the complete call graph and preserves dashboard selection on return',async()=>{
    await open();fireEvent.click(screen.getByRole('button',{name:'Evaluator'}));fireEvent.click(screen.getByRole('button',{name:'Next step'}));
    fireEvent.click(screen.getByRole('button',{name:'Agent call graph'}));
    expect(window.location.hash).toBe('#/graph');expect(screen.getByRole('heading',{name:'Agent call graph'})).toBeTruthy();
    expect(screen.getAllByRole('button',{name:/^Summary:/}).length).toBe(14);
    expect(screen.getByText('COMPLETE GRAPH')).toBeTruthy();
    fireEvent.click(screen.getByRole('button',{name:'Back to dashboard'}));expect(screen.getByRole('table')).toBeTruthy();
    expect(screen.getByRole('button',{name:'Evaluator'}).getAttribute('aria-pressed')).toBe('true');
  });
  it('graph arrows navigate all calls and overview resets to the full graph',async()=>{
    await open();fireEvent.click(screen.getByRole('button',{name:'Agent call graph'}));
    expect((screen.getByRole('button',{name:'Previous call'}) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(screen.getByRole('button',{name:'Next call'}));expect(screen.getByText('Call 1 of 56')).toBeTruthy();
    for(let i=0;i<60;i++)fireEvent.keyDown(window,{key:'ArrowRight'});
    expect(screen.getByText('Call 56 of 56')).toBeTruthy();expect((screen.getByRole('button',{name:'Next call'}) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.keyDown(window,{key:'ArrowLeft'});expect(screen.getByText('Call 55 of 56')).toBeTruthy();
    fireEvent.click(screen.getByRole('button',{name:'Overview'}));expect(screen.getByText('COMPLETE GRAPH')).toBeTruthy();
  });
  it('hover and focus summaries describe decisions and open the corresponding task',async()=>{
    await open();fireEvent.click(screen.getByRole('button',{name:'Agent call graph'}));
    const node=screen.getByRole('button',{name:'Summary: Autonomous search'});fireEvent.mouseEnter(node);
    expect(screen.getByRole('dialog',{name:'Autonomous search summary'})).toBeTruthy();
    expect(screen.getByText('Compared five transformers and three ResNet18 variants.')).toBeTruthy();
    fireEvent.click(screen.getByRole('button',{name:'Explore this task'}));
    expect(screen.getByRole('button',{name:'Autonomous search'}).getAttribute('aria-pressed')).toBe('true');
    fireEvent.click(screen.getByRole('button',{name:'Agent call graph'}));fireEvent.focus(screen.getByRole('button',{name:'Summary: Independent reviewer'}));
    expect(screen.getByRole('dialog',{name:'Independent reviewer summary'})).toBeTruthy();
    fireEvent.keyDown(window,{key:'Escape'});expect(screen.queryByRole('dialog')).toBeNull();
  });
  it('aggregated connections paginate repeated passes and select the exact call',async()=>{
    await open();fireEvent.click(screen.getByRole('button',{name:'Agent call graph'}));
    fireEvent.click(screen.getByRole('button',{name:'Connection: orchestrator:reviewer'}));
    const panel=screen.getByRole('complementary',{name:'Connection passes'});expect(panel.querySelectorAll('.connection-calls button').length).toBe(4);
    fireEvent.click(screen.getByRole('button',{name:'Next connection page'}));expect(panel.textContent).toContain('2 / 3');
    fireEvent.click(panel.querySelector('.connection-calls button')!);expect(screen.queryByRole('complementary',{name:'Connection passes'})).toBeNull();
    expect(screen.getByText(/Call \d+ of 56/)).toBeTruthy();
  });
  it('supports a direct graph URL',async()=>{
    window.history.replaceState(null,'','/#/graph');render(<App/>);
    expect(await screen.findByRole('heading',{name:'Agent call graph'})).toBeTruthy();
  });
});
