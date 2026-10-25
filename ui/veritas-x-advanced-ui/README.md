# VERITAS-X Advanced UI

A dark, forensic-intelligence style React dashboard frontend for the VERITAS-X research prototype. It includes a command center, claim investigation workspace, evidence docket, recorded benchmark view, and read-only frozen configuration screen.

> **Important:** Demo mode shows explicitly labeled illustrative fixtures. It does not run NLI inference. Live API mode requires a thin adapter over the existing Python `VerificationEngine`; this package cannot assume the real engine's response schema because the original `D:\VERITAS-X` source tree was not present while this frontend package was created.

## Run locally

Requires Node.js and npm.

```powershell
cd D:\VERITAS-X\ui\veritas-x-advanced-ui
npm install
Copy-Item .env.example .env
npm run dev
```

Open `http://127.0.0.1:5173`.

If you extracted this package somewhere else, change the `cd` path accordingly. Do not copy this directory over your existing `src/`, `data/`, or `scripts/` folders.

## Switch to live API mode

1. Implement the backend adapter described in [`backend_contract.md`](./backend_contract.md) inside the original VERITAS-X repository.
2. Start the Python API on `http://127.0.0.1:8000` and expose `POST /api/verify`.
3. In `.env`, set `VITE_DEMO_MODE=false` and set `VITE_API_BASE_URL` if your API uses another origin.
4. Restart `npm run dev` after changing `.env`.

The Live API toggle in the UI is also available to test a connection without restarting; requests will fail visibly until the adapter is ready.

## Features

- Responsive left navigation and mobile layout.
- Command center with the audited Phase 3.2.5 test metrics.
- Claim-intake form and four illustrative fixture cases.
- Verdict probabilities, stance-disparity status, evidence cards, and abstention messaging.
- Evidence docket populated by local session records only.
- Model performance view with frozen, recorded metrics.
- Frozen configuration view for transparency.
- No evaluation, tuning, or calibration commands invoked by the UI.

## Frozen evaluation figures displayed

- Accuracy: 59.38%
- Macro-F1: 43.34%
- Macro-precision: 46.91%
- Macro-recall: 49.07%
- Calibrated ECE: 10.60%
- Selective coverage: 94.53%
- Selective accuracy: 62.40%
- Conflicting-class recall: 10.53% (2 of 19)

The benchmark UI displays class-wise F1 values: true 53.10%, false 74.31%, conflicting 15.38%, insufficient evidence 30.59%.

## Build

```powershell
npm run build
```

This command produces a static frontend build in `dist/`. This UI has not been wired to the actual local engine in this package; see `backend_contract.md` before treating Live API mode as production-ready.
