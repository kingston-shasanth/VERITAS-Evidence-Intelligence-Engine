"""
Cross-Encoder NLI Inference Engine for VERITAS-X.
Supports MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli and other NLI checkpoints
with robust dynamic label mapping and CUDA/CPU device resolution.
"""
import os
import re
from typing import List, Dict, Any, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEFAULT_CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), ".cache", "huggingface")
)

DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
LEGACY_MODEL = "cross-encoder/nli-deberta-v3-xsmall"


def resolve_device(requested_device: Optional[str] = None) -> str:
    """
    Selects compute device. Honors override if given; otherwise checks CUDA availability with CPU fallback.
    """
    if requested_device:
        return str(requested_device).strip().lower()
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_nli_label_mapping(config: Any, model_name: str = "") -> Tuple[int, int, int]:
    """
    Resolves (idx_contradiction, idx_entailment, idx_neutral) from model configuration.

    Strict validation:
    - Normalizes case and whitespace.
    - Rejects missing configuration mappings.
    - Explicitly rejects generic unmapped labels (e.g. LABEL_0, LABEL_1, LABEL_2).
    - Ensures all three stances (contradiction, entailment, neutral) are found with unique indices.
    """
    raw_mapping: Dict[str, Any] = {}
    if hasattr(config, "label2id") and isinstance(config.label2id, dict) and config.label2id:
        raw_mapping = config.label2id
    elif hasattr(config, "id2label") and isinstance(config.id2label, dict) and config.id2label:
        raw_mapping = {str(v): int(k) for k, v in config.id2label.items()}
    else:
        raise ValueError(
            f"Model '{model_name}' configuration contains neither 'label2id' nor 'id2label'. "
            f"Cannot resolve NLI stance mapping."
        )

    normalized: Dict[str, int] = {str(k).strip().lower(): int(v) for k, v in raw_mapping.items()}

    # Check for generic unmapped labels (e.g. LABEL_0, LABEL_1)
    if all(k.startswith("label_") for k in normalized.keys()):
        raise ValueError(
            f"Model '{model_name}' contains only generic classification labels: {list(normalized.keys())}. "
            f"NLI stance semantics (entailment, contradiction, neutral) cannot be resolved without explicit mapping."
        )

    contra_keys = ["contradiction", "contradict", "contra"]
    entail_keys = ["entailment", "entails", "entail"]
    neutral_keys = ["neutral", "neut"]

    idx_contra = None
    idx_entail = None
    idx_neutral = None

    for k, idx in normalized.items():
        if any(c == k or c in k for c in contra_keys):
            idx_contra = idx
        elif any(e == k or e in k for e in entail_keys):
            idx_entail = idx
        elif any(n == k or n in k for n in neutral_keys):
            idx_neutral = idx

    missing = []
    if idx_contra is None:
        missing.append("contradiction")
    if idx_entail is None:
        missing.append("entailment")
    if idx_neutral is None:
        missing.append("neutral")

    if missing:
        raise ValueError(
            f"Failed to resolve required NLI stance labels for model '{model_name}'. "
            f"Missing stance(s): {missing}. Available labels in config: {list(normalized.keys())}"
        )

    if len({idx_contra, idx_entail, idx_neutral}) != 3:
        raise ValueError(
            f"Resolved non-unique stance label indices for model '{model_name}': "
            f"contradiction={idx_contra}, entailment={idx_entail}, neutral={idx_neutral}"
        )

    return idx_contra, idx_entail, idx_neutral


_SEMANTIC_REFUTATION = re.compile(
    r'\b('
    r'not|never|no|neither|nor|none|n\'t|won\'t|cannot|can\'t|doesn\'t|don\'t|didn\'t|'
    r'isn\'t|aren\'t|wasn\'t|weren\'t|wouldn\'t|shouldn\'t|couldn\'t|hasn\'t|haven\'t|hadn\'t|'
    r'false|untrue|myth|debunk|debunked|debunking|incorrect|wrong|fiction|disproven|disproved|'
    r'fake|hoax|contrary|dispute|disputed|refute|refuted|refutes|in\s+fact|actually|instead|'
    r'rather|however|on\s+the\s+contrary|contrary\s+to|'
    r'can\s+still|still\s+(?:track|see|monitor|observe|gather|collect|log|record|access|link|know|detect)|'
    r'visible\s+to|remain(?:s)?\s+(?:visible|accessible|trackable|monitored|unprotected)|'
    r'fails?\s+to\s+prevent|does\s+not\s+(?:prevent|protect|stop|hide|block|shield|mask)'
    r')\b',
    re.IGNORECASE
)

_CONTRASTIVE_NEGATION_PATTERN = re.compile(
    r"\b(?P<subj>[A-Za-z0-9][A-Za-z0-9\s'-]*?)\s+\bonly\s+(?P<verb>[a-z]+(?:s|ed|ing)?)\s+(?P<rest>.+?)\s*(?:[-—–]|;\s*|,\s*|\bbut\s+)\s*(?:not|never)\s+(?P<prep>from|to|for|by|against|with|in|on|at|of)?\s*(?P<target>[A-Za-z0-9\s,/'\"-]+?(?=[.!?]|\s*$))",
    re.IGNORECASE
)

def normalize_elliptical_contrast(text: str) -> str:
    """
    Expands restrictive contrastive clauses of the form:
    '[Subject] only [verb phrase] [Prep1] [Target1] [-|,|;] not [Prep2] [Target2]'
    into an explicit compound statement:
    '[Subject] only [verb phrase] [Prep1] [Target1]. [Subject] does not [verb] [object] [Prep2] [Target2]'.
    Resolves the NLI elliptical negation scope blindness where cross-encoders
    drop negative polarity across elided verbs.
    """
    if not text:
        return text

    def _repl(m):
        subj = m.group("subj").strip()
        verb = m.group("verb").strip()
        rest = m.group("rest").strip()
        prep = m.group("prep") or ""
        target = m.group("target").strip()
        
        m_obj = re.match(r"^(.+?)\s+(?:from|to|for|against|with|in|on|at|of)\s+([A-Za-z0-9\s'-]+)$", rest, re.IGNORECASE)
        obj_str = f" {m_obj.group(1).strip()}" if m_obj else ""
        
        base_verb = verb
        aux = "does not"
        if verb.endswith("ies") and len(verb) > 3:
            base_verb = verb[:-3] + "y"
        elif verb.endswith("es") and len(verb) > 2 and (verb.endswith("shes") or verb.endswith("ches") or verb.endswith("xes") or verb.endswith("sses")):
            base_verb = verb[:-2]
        elif verb.endswith("s") and len(verb) > 1 and not verb.endswith("ss"):
            base_verb = verb[:-1]
        elif verb.endswith("ed") and len(verb) > 2:
            aux = "did not"
            base_verb = verb[:-2] if not verb.endswith("eed") else verb[:-1]
            
        prep_str = f" {prep}" if prep else ""
        return f"{subj} only {verb} {rest}. {subj} {aux} {base_verb}{obj_str}{prep_str} {target}."
        
    return _CONTRASTIVE_NEGATION_PATTERN.sub(_repl, text)


_REMOTE_TRACKING_CLAIM_REGEX = re.compile(
    r"\b(prevent(?:s|ing)?|stop(?:s|ping)?|block(?:s|ing)?|shield(?:s|ing)?|protect(?:s|ing)?|hide(?:s|ing)?|mask(?:s|ing)?)\b.*?"
    r"\b(website(?:s)?|tracker(?:s)?|third[- ]part(?:y|ies)|isp(?:s)?|employer(?:s)?|network(?:s)?|advertiser(?:s)?|remote|outside)\b.*?"
    r"\b(track(?:ing|ed|s)?|observ(?:ing|ed|e|es)|monitor(?:ing|ed|s)?|surveill(?:ance)?|see(?:ing)?|log(?:ging|ged|s)?|collect(?:ing|ed|s)?|gather(?:ing|ed|s)?)\b|"
    r"\b(website(?:s)?|tracker(?:s)?|third[- ]part(?:y|ies)|isp(?:s)?|employer(?:s)?|network(?:s)?|advertiser(?:s)?|remote|outside)\b.*?"
    r"\b(track(?:ing|ed|s)?|observ(?:ing|ed|e|es)|monitor(?:ing|ed|s)?|surveill(?:ance)?|see(?:ing)?|log(?:ging|ged|s)?|collect(?:ing|ed|s)?|gather(?:ing|ed|s)?)\b.*?"
    r"\b(prevent(?:s|ing)?|stop(?:s|ping)?|block(?:s|ing)?|shield(?:s|ing)?|protect(?:s|ing)?|hide(?:s|ing)?|mask(?:s|ing)?)\b",
    re.IGNORECASE | re.DOTALL
)

_LOCAL_STORAGE_ONLY_REGEX = re.compile(
    r"\b(on\s+(?:your|the|a)\s+(?:device|computer|phone|machine|laptop)|locally|local\s+storage|from\s+(?:your|the)\s+device)\b.*?"
    r"\b(saved|stored|history|cookies?|site\s+data|form\s+data|cache|temporary\s+data)\b|"
    r"\b(saved|stored|history|cookies?|site\s+data|form\s+data|cache|temporary\s+data)\b.*?"
    r"\b(on\s+(?:your|the|a)\s+(?:device|computer|phone|machine|laptop)|locally|local\s+storage|from\s+(?:your|the)\s+device)\b",
    re.IGNORECASE | re.DOTALL
)

_AFFIRMATIVE_REMOTE_PROTECTION_REGEX = re.compile(
    r"\b(prevents?\s+(?:all\s+)?(?:websites?|trackers?|companies|third|isps?)|"
    r"stops?\s+(?:all\s+)?(?:websites?|trackers?|companies|third|isps?)|"
    r"blocks?\s+(?:all\s+)?(?:third[- ]party\s+)?(?:track(?:ers?|ing)|cookies?|monitoring)|"
    r"protects?\s+(?:completely\s+)?from\s+(?:online|internet|remote|website)\s+tracking|"
    r"complete(?:ly)?\s+(?:anonymous|anonymity|private|invisible)\s+from\s+(?:websites?|isps?|network))\b",
    re.IGNORECASE
)

def _is_local_storage_scope_mismatch(claim: str, passage: str) -> bool:
    if not claim or not passage:
        return False
    if not _REMOTE_TRACKING_CLAIM_REGEX.search(claim):
        return False
    return bool(_LOCAL_STORAGE_ONLY_REGEX.search(passage)) and not bool(_AFFIRMATIVE_REMOTE_PROTECTION_REGEX.search(passage))


_IP_CLAIM_REGEX = re.compile(
    r"\b(hide(?:s|ing)?|mask(?:s|ing)?|conceal(?:s|ing)?|shield(?:s|ing)?|protect(?:s|ing)?)\b.*?\b(ip(?:\s+address)?)\b|"
    r"\b(ip(?:\s+address)?)\b.*?\b(hide(?:s|ing)?|mask(?:s|ing)?|conceal(?:s|ing)?|shield(?:s|ing)?|protect(?:s|ing)?)\b",
    re.IGNORECASE | re.DOTALL,
)

_IP_PASSAGE_REGEX = re.compile(r"\b(ip(?:\s+address)?|internet\s+protocol(?:\s+address)?)\b", re.IGNORECASE)

_PRIVATE_BROWSING_CLAIM_REGEX = re.compile(
    r"\b(private\s+browsing|incognito|private\s+window|private\s+mode)\b",
    re.IGNORECASE,
)

_PRIVATE_BROWSING_PASSAGE_REGEX = re.compile(
    r"\b(private\s+browsing|incognito|private\s+window|private\s+mode)\b",
    re.IGNORECASE,
)

_SEPARATE_RELAY_OR_VPN_REGEX = re.compile(
    r"\b(icloud\s+private\s+relay|private\s+relay|vpn|virtual\s+private\s+network|tor\b|proxy|icloud\+)\b",
    re.IGNORECASE,
)


def _is_ip_hiding_scope_mismatch(claim: str, passage: str) -> bool:
    if not claim or not passage:
        return False
    if not _IP_CLAIM_REGEX.search(claim):
        return False
    # Case 1: Claim asserts IP hiding, but passage does not mention IP at all
    if not _IP_PASSAGE_REGEX.search(passage):
        return True
    # Case 2: Claim asserts private browsing / incognito mode hides IP addresses,
    # but the passage does not mention private browsing or incognito mode at all,
    # and instead describes an auxiliary relay, VPN, or paid subscription service
    # (e.g. Apple iCloud Private Relay, VPN, Tor).
    if _PRIVATE_BROWSING_CLAIM_REGEX.search(claim):
        if not _PRIVATE_BROWSING_PASSAGE_REGEX.search(passage) and _SEPARATE_RELAY_OR_VPN_REGEX.search(passage):
            return True
    return False


_HTTPS_SECURITY_CLAIM_REGEX = re.compile(
    r"\b(https|ssl|tls|padlock|encryption|encrypted\s+connection)\b.*?"
    r"\b(prevents?|stops?|blocks?|protects?\s+against|immune\s+to|safeguards?\s+against|guarantees?\s+safety\s+from)\b.*?"
    r"\b(phishing|scam|scams|fraud|fake\s+website|malicious\s+site|credential\s+theft|identity\s+theft)\b|"
    r"\b(phishing|scam|scams|fraud|fake\s+website|malicious\s+site|credential\s+theft|identity\s+theft)\b.*?"
    r"\b(prevented|stopped|blocked|protected)\s+by\s+(?:https|ssl|tls|encryption)\b",
    re.IGNORECASE | re.DOTALL,
)

_TRANSPORT_LAYER_ONLY_REGEX = re.compile(
    r"\b(in[- ]transit|transit\s+encryption|encrypt(?:s|ed)?\s+(?:data|traffic|communications?|connection)|"
    r"eavesdropping|interception|man[- ]in[- ]the[- ]middle|tampering)\b",
    re.IGNORECASE,
)

_AFFIRMATIVE_PHISHING_PROOF_REGEX = re.compile(
    r"\b(prevents?\s+phishing|stops?\s+phishing|blocks?\s+phishing\s+sites?|detects?\s+fake\s+websites?)\b",
    re.IGNORECASE,
)

def _is_https_phishing_scope_mismatch(claim: str, passage: str) -> bool:
    """Detects when HTTPS transport encryption is conflated with application phishing protection."""
    if not claim or not passage:
        return False
    if not _HTTPS_SECURITY_CLAIM_REGEX.search(claim):
        return False
    return bool(_TRANSPORT_LAYER_ONLY_REGEX.search(passage)) and not bool(_AFFIRMATIVE_PHISHING_PROOF_REGEX.search(passage))


_FILE_DELETION_CLAIM_REGEX = re.compile(
    r"\b(recycle\s+bin|trash|trash\s+can|delet(?:e|ing|ed)|empty(?:ing)?\s+(?:the\s+)?recycle\s+bin)\b.*?"
    r"\b(permanently\s+(?:destroys?|erases?|wipes?|removes?|deletes?)|beyond\s+(?:recovery|forensic)|unrecoverable|physically\s+erased)\b|"
    r"\b(permanently\s+(?:destroys?|erases?|wipes?|removes?|deletes?)|beyond\s+(?:recovery|forensic)|unrecoverable|physically\s+erased)\b.*?"
    r"\b(recycle\s+bin|trash|deleting\s+a\s+file)\b",
    re.IGNORECASE | re.DOTALL,
)

_POINTER_DELETION_ONLY_REGEX = re.compile(
    r"\b(marked\s+as\s+free|file\s+(?:pointer|table|record|reference|index)|until\s+overwritten|"
    r"can\s+still\s+be\s+recovered|recovery\s+software|does\s+not\s+(?:immediately\s+)?erase|"
    r"moves?\s+(?:it\s+)?to\s+(?:the\s+)?(?:recycle\s+bin|trash)|space\s+is\s+reallocated)\b",
    re.IGNORECASE,
)

_AFFIRMATIVE_PERMANENT_ERASURE_REGEX = re.compile(
    r"\b(zero[- ]fill|secure\s+erase|gutmann|dod\s+5220|overwrit(?:es?|ing|ten)\s+with\s+(?:zeroes|random)|shred(?:ded)?|degauss)\b",
    re.IGNORECASE,
)

def _is_file_deletion_scope_mismatch(claim: str, passage: str) -> bool:
    """Detects when moving to recycle bin or directory deletion is conflated with permanent disk erasure."""
    if not claim or not passage:
        return False
    if not _FILE_DELETION_CLAIM_REGEX.search(claim):
        return False
    return bool(_POINTER_DELETION_ONLY_REGEX.search(passage)) and not bool(_AFFIRMATIVE_PERMANENT_ERASURE_REGEX.search(passage))


def _is_generalized_scope_mismatch(claim: str, passage: str) -> bool:
    """Evaluates whether a passage documents an operational scope distinct from the asserted threat boundary."""
    return (
        _is_local_storage_scope_mismatch(claim, passage)
        or _is_ip_hiding_scope_mismatch(claim, passage)
        or _is_https_phishing_scope_mismatch(claim, passage)
        or _is_file_deletion_scope_mismatch(claim, passage)
    )




class CrossEncoderNLI:
    """
    DeBERTa-v3 Cross-Encoder for Natural Language Inference.
    Dynamically resolves id2label mapping and supports GPU/CPU device placement.
    """

    MODEL_NAME = DEFAULT_MODEL

    def __init__(
        self,
        model_name: Optional[str] = None,
        cache_dir: str = DEFAULT_CACHE_DIR,
        device: Optional[str] = None,
        max_length: int = 256,
        tokenizer: Optional[Any] = None,
        model: Optional[Any] = None,
    ):
        self.model_name = model_name or self.MODEL_NAME
        self.cache_dir = cache_dir
        self.device = resolve_device(device)
        self.max_length = max_length

        os.environ["HF_HOME"] = self.cache_dir
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

        if tokenizer is not None and model is not None:
            self.tokenizer = tokenizer
            self.model = model
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, cache_dir=self.cache_dir)

        self.model.to(self.device)
        self.model.eval()

        # Dynamic label mapping resolution
        self.idx_contra, self.idx_entail, self.idx_neutral = resolve_nli_label_mapping(
            self.model.config, self.model_name
        )
        self.label_map = {
            self.idx_contra: "CONTRADICTION",
            self.idx_entail: "ENTAILMENT",
            self.idx_neutral: "NEUTRAL",
        }

    def predict_pair(self, premise: str, hypothesis: str) -> Dict[str, Any]:
        """
        Computes NLI probabilities for a single (premise=evidence, hypothesis=claim) pair.
        """
        results = self.predict_batch([(premise, hypothesis)])
        return results[0]

    def _predict_raw_batch(self, pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """
        Low-level forward pass over tokenized (premise, hypothesis) pairs.
        """
        if not pairs:
            return []

        clean_pairs: List[Tuple[str, str]] = []
        skip_indices = set()

        for idx, (premise, hypothesis) in enumerate(pairs):
            p_clean = (premise or "").strip()
            h_clean = (hypothesis or "").strip()
            if not p_clean or not h_clean:
                skip_indices.add(idx)
                clean_pairs.append(("empty", "empty"))
            else:
                clean_pairs.append((p_clean, h_clean))

        premises = [p for p, _ in clean_pairs]
        hypotheses = [h for _, h in clean_pairs]

        inputs = self.tokenizer(
            premises,
            hypotheses,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        results: List[Dict[str, Any]] = []
        for idx in range(len(pairs)):
            if idx in skip_indices:
                results.append({
                    "contradiction": 0.0,
                    "entailment": 0.0,
                    "neutral": 1.0,
                    "logits": [-5.0, -5.0, 5.0],
                    "predicted_label": "NEUTRAL",
                })
                continue

            c_prob = float(probs[idx][self.idx_contra])
            e_prob = float(probs[idx][self.idx_entail])
            n_prob = float(probs[idx][self.idx_neutral])

            pred_idx = int(probs[idx].argmax())
            pred_label = self.label_map.get(pred_idx, "NEUTRAL")

            results.append({
                "contradiction": c_prob,
                "entailment": e_prob,
                "neutral": n_prob,
                "logits": [float(x) for x in logits[idx].cpu().numpy()],
                "predicted_label": pred_label,
            })

        return results

    def _resolve_snippet_hierarchy(self, premise: str, hypothesis: str, base_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refines multi-snippet passages (e.g. joined by '[...]', paragraph boundaries, or contrastive clauses).
        Prevents introductory rhetorical questions or quoted myth section headers from
        overshadowing decisive refutations or facts in the passage body, while ensuring that
        incomplete fragments cannot overturn clear passage-level entailment or contradiction.
        Preserves referential context across constituent snippets.
        """
        norm_premise = normalize_elliptical_contrast(premise)
        has_contrast_expansion = (norm_premise != premise)
        has_delims = "[...]" in norm_premise or "\n" in norm_premise or has_contrast_expansion
        if not has_delims:
            return base_result

        if has_contrast_expansion:
            raw_snippets = [s.strip() for s in re.split(r'\[\.\.\.\]|\n+|(?<=[.!?])\s+', norm_premise) if s.strip()]
        else:
            raw_snippets = [s.strip() for s in re.split(r'\[\.\.\.\]|\n+', norm_premise) if s.strip()]

        if len(raw_snippets) <= 1 and not has_contrast_expansion:
            return base_result

        # Identify leading subject / entity from the first snippet to preserve referential context
        first_clean = re.sub(r'#+\s*', '', raw_snippets[0]).strip()
        leading_subject = ""
        m_subj = re.match(r'^([A-Z][a-zA-Z0-9\'\-]+(?:\s+[A-Z][a-zA-Z0-9\'\-]+)*)', first_clean)
        if m_subj and len(m_subj.group(1).split()) >= 2:
            leading_subject = m_subj.group(1)

        clean_snippets = []
        contextual_snippets = []
        for s in raw_snippets:
            s_clean = re.sub(r'#+\s*', '', s)
            s_clean = re.sub(r'\[\.\.\.\]', ' ', s_clean)
            s_clean = re.sub(r'\s+', ' ', s_clean).strip()
            words = s_clean.split()
            # Require minimum substantive length (at least 4 words and not a pure trailing fragment)
            if len(words) >= 4 and not re.search(r"\b(and|or|because|although|whereas|such as)\s*[.….]*$", s_clean, re.IGNORECASE):
                clean_snippets.append(s_clean)
                # Inject leading subject context if snippet starts with an anaphoric pronoun
                if leading_subject and re.match(r'^(it|he|she|they|this|that)\b', s_clean, re.IGNORECASE):
                    contextual_snippets.append(f"{leading_subject}: {s_clean}")
                else:
                    contextual_snippets.append(s_clean)

        if not clean_snippets:
            return base_result

        sub_pairs = [(s, hypothesis) for s in contextual_snippets]
        sub_preds = self._predict_raw_batch(sub_pairs)

        adjusted_snippets = []
        for orig_text, pred in zip(clean_snippets, sub_preds):
            e = pred["entailment"]
            c = pred["contradiction"]
            n = pred["neutral"]

            s_stripped = orig_text.strip()
            is_pure_question = s_stripped.endswith("?") or (s_stripped.count("?") > 0 and s_stripped.count(".") == 0)
            has_myth_prefix = bool(re.match(r'^(myth|claim|rumor|question|allegation)\s*:', s_stripped, re.IGNORECASE))
            has_myth_framing = bool(re.search(r'\b(medical\s+myth|popular\s+myth|common\s+myth|common\s+misconception|urban\s+legend|false\s+belief|cling\s+to\s+the\s+(?:idea|belief|notion)|popular\s+misconception)\b', s_stripped, re.IGNORECASE))

            if is_pure_question:
                e = min(e, 0.15)
                n = max(n, 0.80)
            elif has_myth_prefix or has_myth_framing:
                if re.search(r'\b(false|myth|untrue|debunk|incorrect|wrong|fiction|disproven|no\s+evidence|not\s+true|no\s+truth|doesn\'t\s+mean)\b', s_stripped, re.IGNORECASE):
                    c = max(c, 0.85)
                    e = min(e, 0.05)
                    n = max(0.0, 1.0 - (c + e))
                else:
                    e = min(e, 0.15)
            elif _is_generalized_scope_mismatch(hypothesis, s_stripped) and c < 0.40:
                e = min(e, 0.15)
                n = max(n, 0.80)

            # Re-normalize snippet probabilities onto valid simplex
            s_total = e + c + n
            if s_total > 0:
                e, c, n = e / s_total, c / s_total, n / s_total

            adjusted_snippets.append({
                "text": orig_text,
                "entailment": e,
                "contradiction": c,
                "neutral": n,
            })

        # Check for attributed quotation (e.g. John said "it is false") which describes a belief, not an objective finding
        def _is_unattributed_refutation(text: str) -> bool:
            if re.search(r'\b(claims?|alleged|allegation|rumor|said|stated)\s+that\b', text, re.IGNORECASE) and not _SEMANTIC_REFUTATION.search(text):
                return False
            return bool(_SEMANTIC_REFUTATION.search(text))

        contradiction_candidates = [
            sp for sp in adjusted_snippets
            if sp["contradiction"] >= 0.70 and _is_unattributed_refutation(sp["text"])
        ]
        entailment_candidates = [
            sp for sp in adjusted_snippets
            if sp["entailment"] >= 0.70 and not (sp["text"].strip().endswith("?") or sp["text"].strip().startswith("?"))
        ]

        max_c = max((sp["contradiction"] for sp in contradiction_candidates), default=0.0)
        max_e = max((sp["entailment"] for sp in entailment_candidates), default=0.0)

        # If whole passage is already a decisive contradiction, preserve it
        if base_result.get("predicted_label") == "CONTRADICTION" and base_result.get("contradiction", 0.0) >= 0.75:
            return base_result

        # Case 1: Constituent snippet exhibits decisive contradiction (>= 0.70) with genuine refutation
        if contradiction_candidates:
            best_cand = max(contradiction_candidates, key=lambda sp: sp["contradiction"])
            if max_c >= max_e or max_e < 0.70:
                c_val = float(best_cand["contradiction"])
                e_val = float(best_cand["entailment"])
                n_val = float(best_cand["neutral"])
                total = c_val + e_val + n_val
                if total > 0:
                    c_val, e_val, n_val = c_val / total, e_val / total, n_val / total
                return {
                    "contradiction": c_val,
                    "entailment": e_val,
                    "neutral": n_val,
                    "logits": base_result.get("logits", [0.0, 0.0, 0.0]),
                    "predicted_label": "CONTRADICTION",
                }
            else:
                # Both strong contradiction and strong entailment exist in different snippets:
                # Internal tension in passage -> cap entailment so it cannot become an active supporter
                c_val = float(best_cand["contradiction"])
                e_val = float(min(max_e, 0.20))
                n_val = float(max(0.0, 1.0 - (c_val + e_val)))
                total = c_val + e_val + n_val
                if total > 0:
                    c_val, e_val, n_val = c_val / total, e_val / total, n_val / total
                return {
                    "contradiction": c_val,
                    "entailment": e_val,
                    "neutral": n_val,
                    "logits": base_result.get("logits", [0.0, 0.0, 0.0]),
                    "predicted_label": "CONTRADICTION" if c_val > 0.5 else "NEUTRAL",
                }

        # Case 2: Constituent snippet exhibits decisive entailment (>= 0.70) without a decisive refutation
        has_refutation = any(
            _is_unattributed_refutation(sp["text"]) and sp["contradiction"] >= 0.50
            for sp in adjusted_snippets
        )
        if entailment_candidates and not has_refutation:
            # Do not overturn whole-passage decisive contradiction
            if base_result.get("predicted_label") == "NEUTRAL" or (
                base_result.get("contradiction", 0.0) > base_result.get("entailment", 0.0)
                and base_result.get("contradiction", 0.0) < 0.75
            ):
                best_cand = max(entailment_candidates, key=lambda sp: sp["entailment"])
                c_val = float(best_cand["contradiction"])
                e_val = float(best_cand["entailment"])
                n_val = float(best_cand["neutral"])
                total = c_val + e_val + n_val
                if total > 0:
                    c_val, e_val, n_val = c_val / total, e_val / total, n_val / total
                return {
                    "contradiction": c_val,
                    "entailment": e_val,
                    "neutral": n_val,
                    "logits": base_result.get("logits", [0.0, 0.0, 0.0]),
                    "predicted_label": "ENTAILMENT",
                }

        return base_result

    def predict_batch(self, pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """
        Processes a batch of (premise, hypothesis) pairs with truncation and padding.
        Returns probabilities for contradiction, entailment, and neutral with
        multi-snippet hierarchy resolution for web search extracts.
        """
        raw_results = self._predict_raw_batch(pairs)
        final_results = []
        for (premise, hypothesis), res in zip(pairs, raw_results):
            final_results.append(self._resolve_snippet_hierarchy(premise, hypothesis, res))
        return final_results
