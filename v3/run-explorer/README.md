# PathMNIST Run Dashboard

A React dashboard dedicated to `pathmnist_20261008T090954174041Z`, seed 42. The interface and editorial summaries are in English.

## Run locally

Requires Node.js 20.19+ or 22.12+.

```powershell
cd "D:\Python Repository\Agentic-MedMNIST\v3\run-explorer"
npm.cmd install
npm.cmd run dev
```

Open the local address printed by Vite. Use `npm` instead of `npm.cmd` on Linux/macOS.

Choose one of the 14 agent/task boxes on the left. Use **Previous / Next** or the keyboard arrow keys to navigate its brief, statistics and results. Changing task resets to its first step. The completed run’s metrics are available immediately; nothing plays automatically.

The OLED layout uses a black background and colored task cards, with descriptions on the right. At 1366×768 and above, all content is divided into bounded screens rather than scrollable panels. Smaller windows scale the desktop layout; they are not the primary reading target. Tables and the training curve use separate steps to preserve readability. Reduced-motion preferences disable decorative transitions.

The dashboard explains objectives, choices, observed outcomes and limits. It includes all 28 recorded decisions across its editorial briefs, the eight trials, the failed ninth proposal and recovery, and the reporting corrections. It does not display raw files, prompts or JSON, and it does not infer additional internal reasoning. Imported runs and the general-purpose replay explorer have been removed.

## Agent call graph

Click **Agent call graph** to open `#/graph`; **Back to dashboard** restores the previous task and step. A graph node’s **Explore this task** button instead opens that task at its first step. Direct graph links work on the same local server.

There is one rectangle per agent/role, with all 56 invocations grouped into 15 directional connections. A connection label such as `#02 · ×12` means its first chronological call was number 2 and it was used 12 times. Click the label for a paginated list of passes, with original UTC timestamps and task context. Stage dispatches, independent reviews and subprocess operations have distinct connection colors.

Use **Previous call / Next call** or keyboard arrows to highlight the next invocation. **Overview** clears the selection; **Fit graph** and zoom controls restore or inspect the whole diagram. Nothing advances automatically. Hover or focus an agent for three short points about its choices and contribution; click pins the summary, and Escape closes it.

The numbering includes 12 stage starts, 12 reviewer invocations and 32 worker executions. Internal LLM decisions, retries of an LLM request, heartbeats and return values are not counted as extra agent invocations. Edges reflect orchestrator dispatch and stage-owned worker calls, rather than suggesting direct calls between consecutive task agents. Summaries focus on work and results; relevant limitations remain in the task descriptions.

## Verify

```powershell
npm.cmd test
npm.cmd run test:ui
npm.cmd run build
```

Data tests compare the dashboard’s statistics and decision coverage against the original run. UI tests exercise every screen, agent selection, keyboard boundaries and the absence of automatic progression and raw evidence surfaces. Actual layout verification requires a browser; jsdom does not measure browser geometry.

`npm.cmd run prepare-run` regenerates the bundled internal evidence from the original run without changing it. The preparation utility is retained for maintaining this specific dashboard; there is no run import control in the product. `src/dashboard.mjs` owns the source-backed editorial model. Internal source references support checks but are not rendered. No Python server or LLM session is required.
