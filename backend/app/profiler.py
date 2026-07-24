"""Profiler — raw intake context -> structured Profile.

Live: Gemini reasons out roles / trajectory / the ask.
Fallback: a heuristic that pulls the same fields with regex + keyword cues, so the
route works with no key. Either way the output is roles + trajectory + seeking —
reasoning about the person, never bare topic tags.
"""
from __future__ import annotations

import json
import re
import uuid

from .config import settings
from .schemas import Profile

_PROMPT = """You are Kindred's Profiler. Read the person's intake context and return a JSON
object with EXACTLY these keys:
  "name": their name if stated else "".
  "roles": array of 1-4 short role labels, e.g. ["ex-quant", "founder", "infra engineer"].
  "trajectory": their arc as "A -> B" (e.g. "finance -> agent infra").
  "seeking": the ONE concrete thing they are looking for (e.g. "technical cofounder").
  "domain": their topical area in 1-3 words.
  "tags": array of up to 5 short topic tags.
  "summary": one plain sentence describing them.
Return ONLY the JSON object, no prose. Intake context:
---
{context}
---"""


def _new_id(prefix: str = "p") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# --------------------------------------------------------------------------- #
#  Gemini path
# --------------------------------------------------------------------------- #
def _gemini_profile(context: str) -> dict | None:  # pragma: no cover - network
    try:
        import google.generativeai as genai

        genai.configure(api_key=settings.gemini_api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp = model.generate_content(_PROMPT.format(context=context))
        raw = re.sub(r"^```(?:json)?|```$", "", resp.text.strip(), flags=re.MULTILINE).strip()
        return json.loads(raw)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
#  Heuristic fallback
# --------------------------------------------------------------------------- #
_SEEK_RE = re.compile(
    r"(?:seeking|looking for|want(?:ing)? to (?:find|meet)|need|hoping to find|"
    r"in search of|to find)\s+(?:a |an |my )?(.+?)(?:[.;\n]|$)",
    re.IGNORECASE,
)
_BUILD_RE = re.compile(
    r"(?:building|working on|building out|shipping|developing|founder of|currently)\s+"
    r"(?:a |an |the )?(.+?)(?:[.;\n]|$)",
    re.IGNORECASE,
)
_ARROW_RE = re.compile(
    r"([A-Za-z][\w /&-]+?)\s*(?:->|→|to|into|then)\s+([A-Za-z][\w /&-]+)"
)
# origin cues: "ex-quant", "from a derivatives desk", "left investment banking"
_ORIGIN_RE = re.compile(
    r"(?:ex-|former(?:ly)?\s+|left\s+(?:a |an |the )?|"
    r"from\s+(?:a |an |the )?|spent\s+[\w ]+?\s+(?:on|at|in)\s+(?:a |an |the )?)"
    r"([a-z][\w -]*?(?:\s[\w-]+){0,2})\b",
    re.IGNORECASE,
)

_DOMAINS = {
    "agent infra": ["agent", "llm", "inference", "orchestration", "infra", "infrastructure"],
    "fintech": ["finance", "fintech", "trading", "payments", "banking", "quant"],
    "bio": ["bio", "biotech", "genomics", "protein", "medicine", "health", "clinical"],
    "climate": ["climate", "carbon", "energy", "grid", "solar", "battery"],
    "devtools": ["devtools", "developer", "compiler", "database", "sdk", "api"],
    "robotics": ["robot", "robotics", "hardware", "drone", "actuator"],
    "consumer": ["consumer", "social", "creator", "marketplace", "app"],
    "security": ["security", "crypto", "auth", "privacy", "threat"],
}
_STOP = {
    "a", "an", "the", "and", "or", "to", "of", "for", "with", "in", "on", "my", "i",
    "from", "that", "this", "into", "now", "still", "just", "been", "have", "then",
    "when", "than", "they", "them", "here", "there", "about", "over", "after", "years",
    "year", "who", "what", "some", "very", "really", "looking", "building", "want",
}


def _pick_domain(text: str) -> str:
    low = text.lower()
    best, score = "", 0
    for dom, kws in _DOMAINS.items():
        s = sum(low.count(k) for k in kws)
        if s > score:
            best, score = dom, s
    return best or "general"


def _keywords(text: str, n: int = 5) -> list[str]:
    words = [w for w in re.findall(r"[a-zA-Z][a-zA-Z-]{3,}", text.lower()) if w not in _STOP]
    seen, out = set(), []
    for w in words:
        if w not in seen:
            seen.add(w)
            out.append(w)
        if len(out) >= n:
            break
    return out


def _first(pattern: re.Pattern, text: str) -> str:
    m = pattern.search(text)
    return m.group(1).strip().rstrip(".") if m else ""


_PREP = {"from", "of", "a", "an", "the", "and", "to", "at", "in", "on", "with", "now", "then"}


def _clean_phrase(s: str) -> str:
    """Trim a captured phrase at the first preposition/article boundary."""
    words = s.split()
    kept: list[str] = []
    for w in words:
        if w.lower() in _PREP:
            break
        kept.append(w)
    return " ".join(kept).strip()


def derive_roles(trajectory: str, seeking: str, domain: str) -> list[str]:
    """Best-effort role labels from an arc + ask. Used when roles aren't given."""
    roles: list[str] = []
    parts = re.split(r"->|→", trajectory)
    if len(parts) == 2:
        left, right = parts[0].strip(), parts[1].strip()
        if left:
            roles.append(f"ex-{left}")
        if right:
            roles.append(right)
    elif domain:
        roles.append(domain)
    low = f"{seeking} {trajectory}".lower()
    if "cofounder" in low or "founder" in low or "building" in low:
        roles.append("founder")
    # de-dup, keep order
    seen, out = set(), []
    for r in roles:
        if r and r.lower() not in seen:
            seen.add(r.lower())
            out.append(r)
    return out[:4]


def _heuristic_profile(context: str) -> dict:
    seeking = _first(_SEEK_RE, context)
    build = _first(_BUILD_RE, context)
    domain = _pick_domain(context)
    origin = _clean_phrase(_first(_ORIGIN_RE, context))
    if origin:                       # "ex-quant ... now building X" -> "quant -> <domain/build>"
        dest = domain if domain != "general" else (build[:40] if build else "new direction")
        trajectory = f"{origin} -> {dest}"
    elif (arrow := _ARROW_RE.search(context)):
        trajectory = f"{arrow.group(1).strip()} -> {arrow.group(2).strip()}"
    else:
        trajectory = f"{domain} -> {build[:40]}" if build else domain
    first_sentence = re.split(r"(?<=[.!?])\s+", context.strip())[0]
    return {
        "name": "",
        "roles": derive_roles(trajectory, seeking, domain),
        "trajectory": trajectory,
        "seeking": seeking,
        "domain": domain,
        "tags": _keywords(context),
        "summary": first_sentence[:160],
    }


# --------------------------------------------------------------------------- #
#  Public
# --------------------------------------------------------------------------- #
def build_profile(context: str, name: str | None = None, id: str | None = None) -> Profile:
    data = _gemini_profile(context) if settings.gemini_enabled else None
    source = "gemini" if data else "heuristic"
    if data is None:
        data = _heuristic_profile(context)

    trajectory = (data.get("trajectory") or "").strip()
    seeking = (data.get("seeking") or "").strip()
    domain = (data.get("domain") or "general").strip()
    roles = [str(r).strip() for r in (data.get("roles") or []) if str(r).strip()][:4]
    if not roles:
        roles = derive_roles(trajectory, seeking, domain)

    resolved_name = (name or data.get("name") or "").strip() or "You"
    return Profile(
        id=id or _new_id(),
        name=resolved_name,
        roles=roles,
        trajectory=trajectory,
        seeking=seeking,
        tags=[str(t).strip() for t in (data.get("tags") or []) if str(t).strip()][:5],
        domain=domain,
        summary=(data.get("summary") or "").strip(),
        source=source,
    )


def profiler_mode() -> str:
    return "gemini" if settings.gemini_enabled else "heuristic"
