# VERITAS-X Frontend ↔ Backend Contract

The current UI can run in explicit **Demo** mode without calling the model. Demo fixture output is illustrative only and must not be presented as a real model prediction.

For Live API mode, the frontend sends:

```http
POST /api/verify
Content-Type: application/json
```

```json
{
  "claim": "A factual claim to verify.",
  "evidence": []
}
```

`evidence` is an optional list of user-provided passages represented as strings. The final Python adapter may ignore it or support it, but must not silently claim it was used if unsupported.

## Expected response schema

```json
{
  "claim_id": "VX-123456",
  "claim": "A factual claim to verify.",
  "verdict": "VERIFIED_TRUE",
  "confidence": 0.72,
  "probabilities": {
    "VERIFIED_TRUE": 0.72,
    "VERIFIED_FALSE": 0.12,
    "CONFLICTING_OR_CHERRYPICKED": 0.06,
    "INSUFFICIENT_EVIDENCE": 0.10
  },
  "has_cross_passage_disparity": false,
  "abstained": false,
  "reason": "Optional concise explanation supported by the returned evidence.",
  "evidence": [
    {
      "id": "E-01",
      "source": "Source title or local corpus identifier",
      "text": "Retrieved evidence passage.",
      "stance": "support",
      "score": 0.82
    }
  ],
  "device": "cuda"
}
```

Allowed verdict labels:
- `VERIFIED_TRUE`
- `VERIFIED_FALSE`
- `CONFLICTING_OR_CHERRYPICKED`
- `INSUFFICIENT_EVIDENCE`

Evidence stance values can be `support`, `refute`, or `neutral` (or be mapped from the engine's entailment/contradiction/neutral classes).

## Important adapter work required

This project package does **not** include the user's existing Python source tree, so the actual `VerificationEngine` result object has not been inspected here. Before live mode is used, create a thin API adapter in the existing VERITAS-X repository that:

1. Instantiates the existing engine with the frozen Phase 3.2.5 aggregator, temperature calibrator and abstainer values.
2. Uses the real local evidence retriever; does not invent sources or imply live web search.
3. Converts the engine's actual result object into the response schema above.
4. Returns the actual calibrated four-class probabilities (if exposed) and abstention decision.
5. Handles errors and CORS appropriately for `http://127.0.0.1:5173`.
6. Does not mutate calibration, thresholds, the held-out artifacts, or model parameters.

Until this adapter exists and is tested against the actual engine, use Demo mode for interface walkthroughs only.
