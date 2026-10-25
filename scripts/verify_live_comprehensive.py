import urllib.request
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

def query_claim(claim, mode="web", evidence=None):
    payload = {"claim": claim, "mode": mode}
    if evidence:
        payload["evidence"] = evidence
    req = urllib.request.Request(
        "http://127.0.0.1:8011/api/verify",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data
    except Exception as e:
        return {"error": str(e)}

def print_summary(name, claim, res):
    print(f"\n{'='*70}")
    print(f"CASE: {name}")
    print(f"Claim: '{claim}'")
    if "error" in res:
        print(f"ERROR: {res['error']}")
        return
    print(f"Verdict: {res.get('verdict')}")
    print(f"Confidence: {res.get('confidence')}")
    print(f"Abstained: {res.get('abstained')}")
    print(f"Evidence counts: {res.get('evidence_counts')}")
    print(f"Reason: {res.get('reason')}")
    print("Passages:")
    for idx, p in enumerate(res.get("evidence", [])[:5]):
        text = (p.get("text") or "").replace("\n", " ")[:120]
        print(f"  [{idx}] stance={p.get('stance')} elig={p.get('is_decision_eligible')} cov={p.get('claim_coverage')} src={p.get('title')}")
        print(f"       text: {text}...")

if __name__ == "__main__":
    test_cases = [
        ("Claim A (Incognito Tracking)", "Using incognito mode prevents websites from tracking a user's activity.", "web", None),
        ("Claim B (Incognito IP)", "Private browsing mode automatically hides a user's IP address from the websites they visit.", "web", None),
        ("Straightforward True (Water composition)", "Water is composed of hydrogen and oxygen atoms.", "local", None),
        ("Straightforward False (Moon green cheese)", "The Moon is made of green cheese.", "local", [
            "The Moon is an astronomical body that orbits Earth and is composed of silicate rock and metallic iron.",
            "Lunar sample analysis confirms the Moon consists of basalt, anorthosite, and breccia, not dairy or cheese."
        ]),
        ("Insufficient Evidence (Obscure claim)", "In 1943, an unrecorded traveler dropped an unmarked copper button into an undisclosed lake.", "local", None),
    ]

    for name, claim, mode, evidence in test_cases:
        res = query_claim(claim, mode=mode, evidence=evidence)
        print_summary(name, claim, res)
