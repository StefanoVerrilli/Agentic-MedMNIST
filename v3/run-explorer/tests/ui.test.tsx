import React from 'react';
import {readFileSync} from 'node:fs';
import {afterEach,beforeEach,describe,it,expect,vi} from 'vitest';
import {render,screen,fireEvent,act,cleanup} from '@testing-library/react';
import {buildDashboard} from '../src/dashboard.mjs';
vi.mock('recharts',()=>({ResponsiveContainer:({children}:any)=><div>{children}</div>,AreaChart:()=>null,Area:()=>null,XAxis:()=>null,YAxis:()=>null,Tooltip:()=>null,CartesianGrid:()=>null}));
import {App} from '../src/main';
const archive=JSON.parse(readFileSync('public/run.json','utf8'));
const model=buildDashboard(archive);
beforeEach(()=>vi.stubGlobal('fetch',vi.fn(async()=>({ok:true,json:async()=>structuredClone(archive)}))));
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
});
