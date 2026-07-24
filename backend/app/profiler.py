"""Profiler — raw intake context -> structured Profile, plus match reasoning.

Live: Gemini reasons out roles / trajectory / the ask, and writes the "why these
two" lines the frontend shows on node click.
Fallback: a heuristic that pulls the same fields with regex + keyword cues, and
template reasons, so the route works with no key. Either way the output is roles
+ trajectory + seeking — reasoning about the person, never bare topic tags.

Both live paths go through `config.gemini_call`, which returns None on any
failure (missing SDK, bad key, 429, timeout, malformed JSON). None always means
"use the deterministic path", so a Gemini problem can never fail a request.
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Any, Iterable, Mapping, Optional, Sequence

from . import config
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

# One call covers the whole result set — never one call per node.
_REASON_PROMPT = """You are Kindred's matchmaker. Someone is looking at their match graph and
clicked through the people below. For EACH candidate, write why this specific
pair is worth a conversation.

Rules:
- 1 to 3 reasons per candidate. Each is a short phrase under 90 characters.
- Be specific and human. Name the shared arc, the concrete overlap, or the
  trade: what one of them has that the other is explicitly asking for.
- The strongest reason is complementarity ("you want X, they do X"), then a
  shared trajectory, then a shared topic. Lead with the strongest.
- Write to the user as "you"; call the candidate by their first name.
- Use ONLY the facts given. Never invent employers, tools or history.
- No trailing periods. Do not restate the score.

Return ONLY a JSON object mapping each candidate id to its array of reasons,
e.g. {"p_maya": ["you want a cofounder; Maya is hiring one", "both left trading desks for agent infra"]}

THE USER:
{user}

CANDIDATES:
{candidates}"""

_MAX_REASON_CANDIDATES = 15   # bounds prompt size; the rest keep their templates
_MAX_REASON_LEN = 140


def _new_id(prefix: str = "p") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _fill(template: str, **slots: str) -> str:
    """Substitute {slots} literally. `str.format` is unusable here: both the
    intake context and the candidate JSON routinely contain braces."""
    out = template
    for key, value in slots.items():
        out = out.replace("{" + key + "}", value)
    return out


def _json_object(raw: str) -> Optional[dict]:
    """Parse a JSON object out of a model response, code fences and all."""
    if not raw:
        return None
    text = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(text)
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except Exception:
            return None
    return data if isinstance(data, dict) else None


# --------------------------------------------------------------------------- #
#  Gemini path
# --------------------------------------------------------------------------- #
def _generate_config(max_tokens: int) -> dict:
    cfg: dict = {
        "response_mime_type": "application/json",
        "temperature": 0.4,
        "max_output_tokens": max_tokens,
    }
    # 2.5 models think by default and we want a fast, cheap structured answer.
    # Older models reject the knob, so only send it where it exists.
    if "gemini-2.5" in config.settings.gemini_model:
        cfg["thinking_config"] = {"thinking_budget": 0}
    return cfg


def _generate(prompt: str, *, op: str, max_tokens: int) -> Optional[dict]:
    """One guarded generate_content call that must return a JSON object."""
    resp = config.gemini_call(
        lambda client: client.models.generate_content(
            model=config.settings.gemini_model,
            contents=prompt,
            config=_generate_config(max_tokens),
        ),
        kind="generate",
        op=op,
    )
    if resp is None:
        return None
    try:
        return _json_object(getattr(resp, "text", "") or "")
    except Exception:
        return None


def _gemini_profile(context: str) -> Optional[dict]:
    return _generate(_fill(_PROMPT, context=context), op="profile", max_tokens=600)


def _clean_reasons(raw: Any) -> list[str]:
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[str] = []
    for item in raw:
        text = str(item).strip().strip("-•").strip()
        if not text:
            continue
        out.append(text[:_MAX_REASON_LEN].rstrip(" ."))
        if len(out) == 3:
            break
    return out


def gemini_reasons(user: Profile, candidates: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    """Gemini-written match reasons for a batch of candidates.

    ONE API call for the whole batch (free-tier quota is per request, not per
    token). Returns {} when Gemini is unavailable or the answer is unusable —
    callers keep their template reasons for any id not returned here.
    """
    if not candidates or not config.settings.gemini_enabled:
        return {}
    known = {str(c.get("id")) for c in candidates if c.get("id")}
    if not known:
        return {}
    payload = {
        "name": user.name,
        "roles": user.roles,
        "trajectory": user.trajectory,
        "seeking": user.seeking,
        "domain": user.domain,
        "tags": user.tags,
        "summary": user.summary,
    }
    prompt = _fill(
        _REASON_PROMPT,
        user=json.dumps(payload, ensure_ascii=False),
        candidates=json.dumps([dict(c) for c in candidates], ensure_ascii=False),
    )
    data = _generate(prompt, op="reasons", max_tokens=2048)
    if not data:
        return {}
    out: dict[str, list[str]] = {}
    for key, value in data.items():
        key = str(key)
        if key not in known:
            continue
        reasons = _clean_reasons(value)
        if reasons:
            out[key] = reasons
    return out


def _candidate_payload(match: Any, meta: Mapping[str, Any]) -> dict:
    return {
        "id": match.id,
        "name": getattr(match, "name", "") or meta.get("name", ""),
        "roles": list(meta.get("roles") or [])[:4],
        "trajectory": meta.get("trajectory", ""),
        "seeking": meta.get("seeking", ""),
        "domain": meta.get("domain", ""),
        "tags": list(meta.get("tags") or [])[:5],
        "summary": meta.get("summary", ""),
    }


def apply_gemini_reasons(
    user: Profile, matches: Iterable[Any], metas: Mapping[str, Mapping[str, Any]]
) -> None:
    """Upgrade `match.reasons` in place with Gemini-written lines.

    Call it once, after ranking, with the candidate metadata:

        apply_gemini_reasons(user, matches, {n.id: n.meta for n in neighbours})

    A no-op without a key, and a no-op on any failure — the template reasons
    already on each match stay exactly as they are.
    """
    matches = list(matches)
    if not matches or not config.settings.gemini_enabled:
        return
    try:
        payload = [
            _candidate_payload(m, metas.get(m.id) or {})
            for m in matches[:_MAX_REASON_CANDIDATES]
        ]
        written = gemini_reasons(user, payload)
        for match in matches:
            better = written.get(match.id)
            if better:
                match.reasons = better
    except Exception:  # pragma: no cover - defensive; gemini_reasons already guards
        return


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
def _strings(value: Any, limit: int) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    return [str(v).strip() for v in value if str(v).strip()][:limit]


def build_profile(context: str, name: str | None = None, id: str | None = None) -> Profile:
    """Intake context -> Profile. Gemini when configured, heuristic otherwise.

    The heuristic result is always computed (it is pure regex, essentially free)
    and used as the base layer, so a Gemini answer that omits or empties a field
    still yields a complete profile instead of a hollow one.
    """
    data = _heuristic_profile(context)
    live = _gemini_profile(context) if config.settings.gemini_enabled else None
    source = "heuristic"
    if live:
        source = "gemini"
        for key in ("name", "trajectory", "seeking", "domain", "summary"):
            value = live.get(key)
            if isinstance(value, str) and value.strip():
                data[key] = value.strip()
        for key, limit in (("roles", 4), ("tags", 5)):
            values = _strings(live.get(key), limit)
            if values:
                data[key] = values

    trajectory = (data.get("trajectory") or "").strip()
    seeking = (data.get("seeking") or "").strip()
    domain = (data.get("domain") or "general").strip()
    roles = _strings(data.get("roles"), 4)
    if not roles:
        roles = derive_roles(trajectory, seeking, domain)

    resolved_name = (name or data.get("name") or "").strip() or "You"
    return Profile(
        id=id or _new_id(),
        name=resolved_name,
        roles=roles,
        trajectory=trajectory,
        seeking=seeking,
        tags=_strings(data.get("tags"), 5),
        domain=domain,
        summary=(data.get("summary") or "").strip(),
        source=source,
    )


def profiler_mode() -> str:
    """What /health reports — the mode actually in force right now."""
    return "gemini" if config.gemini_live() else "heuristic"
