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

## Verify

```powershell
npm.cmd test
npm.cmd run test:ui
npm.cmd run build
```

Data tests compare the dashboard’s statistics and decision coverage against the original run. UI tests exercise every screen, agent selection, keyboard boundaries and the absence of automatic progression and raw evidence surfaces. Actual layout verification requires a browser; jsdom does not measure browser geometry.

`npm.cmd run prepare-run` regenerates the bundled internal evidence from the original run without changing it. The preparation utility is retained for maintaining this specific dashboard; there is no run import control in the product. `src/dashboard.mjs` owns the source-backed editorial model. Internal source references support checks but are not rendered. No Python server or LLM session is required.
