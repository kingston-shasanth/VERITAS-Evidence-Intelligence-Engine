import urllib.request
import json
import time

API_URL = "http://127.0.0.1:8011/api/verify"
HEALTH_URL = "http://127.0.0.1:8011/api/health"

def query_verify(claim: str, mode: str = "local", user_evidence: list = None, include_research: bool = True):
    payload = {
        "claim": claim,
        "mode": mode,
        "include_research": include_research,
        "include_timings": True,
    }
    if user_evidence:
        payload["evidence"] = user_evidence
        payload["mode"] = "user_supplied"
    
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Origin": "http://127.0.0.1:5173"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    lat_ms = (time.perf_counter() - t0) * 1000.0
    return data, lat_ms

print("=== VERITAS-X Live API & Workspace Adjudication Audit ===")

# Test 1: Pure water freezes at 0 degrees Celsius (Demo Fixture VX-DEMO-014)
water_ev = [
    "A water molecule consists of two hydrogen atoms bonded to one oxygen atom (H2O). Under standard pressure at sea level, it freezes at 0 degrees Celsius.",
    "The Celsius temperature scale defines zero degrees as the melting and freezing point of water at 1 atm."
]
res1, lat1 = query_verify("Pure water freezes at 0 degrees Celsius at sea level.", user_evidence=water_ev)
print(f"\n[TEST 1] Pure Water:\n  Verdict: {res1.get('verdict')} | Conf: {res1.get('confidence')} | Latency: {lat1:.1f}ms\n  Reason: {res1.get('reason')}")
assert res1.get('verdict') == "VERIFIED_TRUE", f"Expected VERIFIED_TRUE, got {res1.get('verdict')}"

# Test 2: The Moon is a star (Demo Fixture VX-DEMO-027)
moon_ev = [
    "The Moon is a natural rocky satellite that reflects solar illumination; it lacks sufficient mass for nuclear fusion.",
    "Stars generate light through sustained thermonuclear reactions in their cores, whereas the Moon is cold rock."
]
res2, lat2 = query_verify("The Moon is a star that produces its own light through nuclear fusion.", user_evidence=moon_ev)
print(f"\n[TEST 2] Moon Star:\n  Verdict: {res2.get('verdict')} | Conf: {res2.get('confidence')} | Latency: {lat2:.1f}ms\n  Reason: {res2.get('reason')}")
assert res2.get('verdict') == "VERIFIED_FALSE", f"Expected VERIFIED_FALSE, got {res2.get('verdict')}"

# Test 3: Conflicting Bridge
bridge_ev = [
    "The municipal public works bulletin stated the bridge was closed to all traffic on Monday for emergency span repairs.",
    "The metropolitan transport authority stated that the bridge remained fully open to commercial and private vehicles on Monday."
]
res3, lat3 = query_verify("The municipal bridge was completely closed to vehicle traffic on Monday.", user_evidence=bridge_ev)
print(f"\n[TEST 3] Conflicting Bridge:\n  Verdict: {res3.get('verdict')} | Conf: {res3.get('confidence')} | Latency: {lat3:.1f}ms\n  Disparity: {res3.get('has_cross_passage_disparity')}\n  Reason: {res3.get('reason')}")
assert res3.get('verdict') == "CONFLICTING_OR_CHERRYPICKED", f"Expected CONFLICTING_OR_CHERRYPICKED, got {res3.get('verdict')}"

# Test 4: Extraterrestrial spacecraft
res4, lat4 = query_verify("Extraterrestrial spacecraft landed in the central city plaza yesterday evening.")
print(f"\n[TEST 4] Unverified Landings:\n  Verdict: {res4.get('verdict')} | Conf: {res4.get('confidence')} | Latency: {lat4:.1f}ms\n  Abstained: {res4.get('abstained')}\n  Reason: {res4.get('reason')}")
assert res4.get('verdict') == "INSUFFICIENT_EVIDENCE", f"Expected INSUFFICIENT_EVIDENCE, got {res4.get('verdict')}"

# Test 5: Chocolate Causal Claim (Correlational evidence != Causation)
choc_ev = [
    "A 2012 study in the New England Journal of Medicine observed a strong correlation between per capita chocolate consumption and the number of Nobel laureates per capita in various nations."
]
res5, lat5 = query_verify("Chocolate consumption correlates with Nobel laureates and causes higher intelligence.", user_evidence=choc_ev)
print(f"\n[TEST 5] Chocolate Causal:\n  Verdict: {res5.get('verdict')} | Conf: {res5.get('confidence')} | Latency: {lat5:.1f}ms\n  Abstained: {res5.get('abstained')}\n  Reason: {res5.get('reason')}")
assert res5.get('verdict') == "INSUFFICIENT_EVIDENCE", f"Expected INSUFFICIENT_EVIDENCE, got {res5.get('verdict')}"

# Test 6: Hospital Mortality Causal Claim (Observational/Confounded != Falsity)
hosp_ev = [
    "Observational data from regional health networks indicates that higher hospital admission rates correlate with increased patient mortality, because patients admitted to hospitals are significantly more ill than the general population. No clinical trial demonstrates that hospital admission causes mortality."
]
res6, lat6 = query_verify("Higher hospital admission rates cause increased patient mortality.", user_evidence=hosp_ev)
print(f"\n[TEST 6] Hospital Admission:\n  Verdict: {res6.get('verdict')} | Conf: {res6.get('confidence')} | Latency: {lat6:.1f}ms\n  Abstained: {res6.get('abstained')}\n  Reason: {res6.get('reason')}")
assert res6.get('verdict') == "INSUFFICIENT_EVIDENCE", f"Expected INSUFFICIENT_EVIDENCE, got {res6.get('verdict')}"
assert res6.get('verdict') != "VERIFIED_FALSE", "Lack of causal evidence must not be marked VERIFIED_FALSE"

# Test 7: Great Wall Compound Claim (Partial evidence != Validated compound)
gw_ev = [
    "The Great Wall of China was constructed and heavily fortified under the Ming Dynasty."
]
res7, lat7 = query_verify("The Great Wall of China is visible from the Moon with the naked eye, and it was constructed under the Ming Dynasty.", user_evidence=gw_ev)
print(f"\n[TEST 7] Great Wall Partial:\n  Verdict: {res7.get('verdict')} | Conf: {res7.get('confidence')} | Latency: {lat7:.1f}ms\n  Abstained: {res7.get('abstained')}\n  Reason: {res7.get('reason')}")
assert res7.get('verdict') == "INSUFFICIENT_EVIDENCE", f"Expected INSUFFICIENT_EVIDENCE, got {res7.get('verdict')}"

# Test 8: Lighter / Lung Cancer Claim (Confounded association != Causation)
light_ev = [
    "Epidemiological studies observe a strong correlation between carrying matches or lighters and developing lung cancer; however, this association is heavily confounded by cigarette smoking behavior."
]
res8, lat8 = query_verify("Carrying matches or a lighter causes lung cancer.", user_evidence=light_ev)
print(f"\n[TEST 8] Lighter Confounded:\n  Verdict: {res8.get('verdict')} | Conf: {res8.get('confidence')} | Latency: {lat8:.1f}ms\n  Abstained: {res8.get('abstained')}\n  Reason: {res8.get('reason')}")
assert res8.get('verdict') == "INSUFFICIENT_EVIDENCE", f"Expected INSUFFICIENT_EVIDENCE, got {res8.get('verdict')}"

print("\n>>> ALL 8 TARGET ADJUDICATION TESTS PASSED WITH 100% SUCCESS ON LIVE API DAEMON! <<<")
