"""
Live Demonstration Verification Runner for VERITAS-X.
Tests 5 distinct non-held-out scenarios against the live running API server:
1. Supporting Evidence Claim (retrieval from diagnostic corpus)
2. Contradicting Evidence Claim (direct refutation)
3. Ambiguous / Insufficient Evidence Claim (corpus has no data)
4. Multi-Evidence Claim (cross-passage disparity between pro and con)
5. Unusual Query with No Lexical Match (tests BM25 zero-score filtering)
"""
import os
import sys
import json
import time
import urllib.request

DEMO_CASES = [
    {
        "category": "1. Supporting Evidence Claim",
        "claim": "India landed Chandrayaan-3 near the south pole of the Moon on August 23, 2023.",
        "evidence": []  # Retrieved from diagnostic corpus
    },
    {
        "category": "2. Contradicting Evidence Claim",
        "claim": "The Moon produces its own light through sustained thermonuclear reactions in its core.",
        "evidence": [
            "The Moon is a cold rocky planetary satellite that reflects sunlight; it lacks sufficient mass for thermonuclear fusion."
        ]
    },
    {
        "category": "3. Ambiguous / Insufficient Evidence Claim",
        "claim": "Extraterrestrial spacecraft landed in the central city plaza yesterday evening.",
        "evidence": []
    },
    {
        "category": "4. Multi-Evidence Claim (Cross-Passage Disparity)",
        "claim": "The municipal bridge was completely closed to vehicle traffic on Monday.",
        "evidence": [
            "The municipal bulletin stated the bridge was closed to all traffic on Monday for emergency repairs.",
            "The transport authority stated that the bridge remained fully open to commercial and private vehicles on Monday."
        ]
    },
    {
        "category": "5. Unusual Query with No Lexical Match",
        "claim": "zzqxkj9999000nonexistentterm xyzwvvv arbitrary random claim statement 8888",
        "evidence": []
    }
]


def run_demo():
    port = os.getenv("VERITAS_API_PORT", "8011")
    target_url = f"http://127.0.0.1:{port}/api/verify"
    print("=" * 80)
    print("LIVE END-TO-END VERIFICATION RUN ACROSS 5 DEMONSTRATION CASES")
    print(f"Server Target: {target_url}")
    print("=" * 80)

    results = []
    for case in DEMO_CASES:
        print(f"\n--- [{case['category']}] ---")
        print("Claim:", case["claim"])
        payload = json.dumps({"claim": case["claim"], "evidence": case["evidence"]}).encode("utf-8")
        req = urllib.request.Request(
            target_url,
            data=payload,
            headers={"Content-Type": "application/json", "Origin": "http://127.0.0.1:5173"},
            method="POST"
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                latency_ms = round((time.time() - t0) * 1000, 1)
                data = json.loads(resp.read().decode("utf-8"))
                data["latency_ms"] = latency_ms
                data["http_status"] = resp.status
                results.append({"case": case, "result": data})
                print(f"HTTP Status  : {resp.status} (Latency: {latency_ms} ms)")
                print(f"Claim ID     : {data.get('claim_id')}")
                print(f"Verdict      : {data.get('verdict')}")
                print(f"Confidence   : {data.get('confidence')}")
                print(f"Abstained    : {data.get('abstained')}")
                print(f"Disparity    : {data.get('has_cross_passage_disparity')}")
                print(f"Reason       : {data.get('reason')}")
                print(f"Probabilities: {data.get('probabilities')}")
                print(f"Evidence Used: {len(data.get('evidence', []))} passage(s)")
                for ev in data.get("evidence", []):
                    text_snippet = ev.get("text", "")[:80].replace("\n", " ")
                    print(f"   [{ev.get('id')}] stance={ev.get('stance')} | score={ev.get('score')} | E={ev.get('entailment_prob')} C={ev.get('contradiction_prob')} N={ev.get('neutral_prob')}")
                    print(f"        source={ev.get('source')} | text=\"{text_snippet}...\"")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()

    os.makedirs("tmp", exist_ok=True)
    with open("tmp/live_demo_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\n[+] Saved complete demonstration telemetry to tmp/live_demo_results.json")


if __name__ == "__main__":
    run_demo()
