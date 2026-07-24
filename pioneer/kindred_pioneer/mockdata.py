"""Mock cohort + labeled pairs — stands in until Raj's loop produces real landings.

The generative story is the product thesis, written down as coefficients: what
makes an intro land is mostly **ask/offer fit** and **shared trajectory**, with
shared topic contributing only a little. Cosine similarity over profile text
mostly reads the topic term, which is why the baseline is beatable — but the
topic term is genuinely positive, so the baseline is a real signal, not a
strawman.

Nothing in here is visible to the model at inference time: the scorer only ever
sees `Person` fields, never `p_true` or the coefficients below.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from .schema import DIRECTIONAL_ASKS, RECIPROCAL_ASKS, STAGES, LabeledPair, Person

DOMAIN_INTERESTS: dict[str, list[str]] = {
    "fintech": ["payments rails", "underwriting", "compliance automation", "ledger design"],
    "devtools": ["build systems", "observability", "agent infrastructure", "CI pipelines"],
    "climate": ["grid software", "carbon accounting", "battery analytics", "permitting data"],
    "bio": ["protein design", "lab automation", "clinical data", "assay tooling"],
    "robotics": ["manipulation", "fleet software", "sim-to-real", "teleoperation"],
    "consumer social": ["creator tools", "community graphs", "short video", "messaging"],
    "healthcare ops": ["prior authorization", "scheduling", "revenue cycle", "care routing"],
    "security": ["detection engineering", "identity", "supply chain scanning", "red teaming"],
    "education": ["assessment", "tutoring systems", "credentialing", "curriculum tools"],
    "logistics": ["route optimization", "freight matching", "warehouse robotics", "customs data"],
}
DOMAINS = list(DOMAIN_INTERESTS)

PRIOR_DOMAINS = [
    "finance", "big tech", "academia", "consulting",
    "government", "medicine", "defense", "media",
]

CITIES = ["SF", "NYC", "London", "Berlin", "Toronto", "Bangalore", "remote"]

FIRST_NAMES = [
    "Ada", "Bo", "Cass", "Dev", "Ep", "Fern", "Gil", "Hana", "Ira", "Jun",
    "Kai", "Lux", "Mira", "Nils", "Oona", "Pax", "Quin", "Rune", "Sol", "Tam",
    "Uma", "Vee", "Wren", "Xin", "Yara", "Zed", "Alix", "Bram", "Cleo", "Dara",
    "Elio", "Faye", "Gwen", "Hugo", "Isla", "Joss", "Kir", "Lena", "Moss", "Nia",
    "Orin", "Piet", "Rhea", "Sena", "Tove", "Ulla", "Vida", "Wim", "Yann", "Zara",
    "Amir", "Bijan", "Cira", "Doran", "Esen", "Frey", "Gita", "Hollis", "Ines", "Juno",
    "Kesh", "Lior", "Mako", "Neve", "Odin", "Priya", "Rami", "Sana", "Tycho", "Vail",
]

# Ordered so the model's stage gap has a consistent sign.
_STAGE_ASK_WEIGHTS = {
    "exploring": {"cofounder": 0.55, "accountability partner": 0.30, "peer group": 0.25},
    "building": {"cofounder": 0.30, "accountability partner": 0.22, "peer group": 0.30},
    "scaling": {"cofounder": 0.08, "accountability partner": 0.10, "peer group": 0.35},
}

# The generative weights. Ask/offer fit and trajectory dominate; topic is weak
# but genuinely positive. BIAS is calibrated so the overall landing rate is ~40%
# — the 41% the rest of the repo quotes for generation 1 — with the best decile
# of pairs landing ~96% and the worst ~8%. That spread is what leaves the loop
# somewhere to climb to (84%) without putting it at the ceiling.
W_RECIPROCAL = 2.6
W_DIRECTIONAL = 3.6
W_MUTUAL_BONUS = 1.8
W_SAME_ARC = 2.4
W_TOPIC = 1.2
W_SAME_DOMAIN = 0.8
W_SAME_CITY = 1.0
W_STAGE_GAP = -1.0
W_SENIORITY_GAP = -0.9
BIAS = -1.3
NOISE_SD = 0.30


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def make_person(rng: random.Random, idx: int, prefix: str = "p") -> Person:
    domain = rng.choice(DOMAINS)
    prior = rng.choice(PRIOR_DOMAINS)
    stage = rng.choices(STAGES, weights=[0.35, 0.45, 0.20])[0]
    seniority = max(1, int(rng.gauss({"exploring": 5, "building": 8, "scaling": 12}[stage], 3)))

    interests = rng.sample(DOMAIN_INTERESTS[domain], k=rng.randint(2, 3))
    if rng.random() < 0.35:  # cross-pollination: people rarely sit in one topic
        other = rng.choice([d for d in DOMAINS if d != domain])
        interests.append(rng.choice(DOMAIN_INTERESTS[other]))

    seeking: list[str] = []
    for ask, prob in _STAGE_ASK_WEIGHTS[stage].items():
        if rng.random() < prob:
            seeking.append(ask)
    seeking += rng.sample(sorted(DIRECTIONAL_ASKS), k=rng.randint(1, 2))

    # More senior, later-stage people can give away more than they need.
    n_offer = 1 + int(seniority > 6) + int(stage == "scaling")
    available = sorted(DIRECTIONAL_ASKS - set(seeking)) or sorted(DIRECTIONAL_ASKS)
    offering = rng.sample(available, k=min(n_offer, len(available)))

    return Person(
        id=f"{prefix}{idx:03d}",
        name=f"{FIRST_NAMES[idx % len(FIRST_NAMES)]} {chr(65 + idx % 26)}.",
        domain=domain,
        prior_domain=prior,
        stage=stage,
        seniority=seniority,
        city=rng.choice(CITIES),
        interests=sorted(set(interests)),
        seeking=sorted(set(seeking)),
        offering=sorted(set(offering)),
    )


def landing_probability(a: Person, b: Person) -> float:
    """The latent truth the loop is trying to learn. Never exposed as a feature."""
    seek_a, seek_b = set(a.seeking), set(b.seeking)
    offer_a, offer_b = set(a.offering), set(b.offering)

    reciprocal = len(seek_a & seek_b & RECIPROCAL_ASKS) / max(1, len(RECIPROCAL_ASKS))
    dir_ab = len(seek_a & offer_b) / max(1, len(seek_a & DIRECTIONAL_ASKS))
    dir_ba = len(seek_b & offer_a) / max(1, len(seek_b & DIRECTIONAL_ASKS))
    mutual = 1.0 if (dir_ab > 0 and dir_ba > 0) else 0.0

    same_arc = 1.0 if a.prior_domain == b.prior_domain else 0.0
    topic = _jaccard(a.interests, b.interests)
    same_domain = 1.0 if a.domain == b.domain else 0.0
    same_city = 1.0 if a.city == b.city and a.city != "remote" else 0.0
    stage_gap = abs(a.stage_index - b.stage_index) / 2.0
    seniority_gap = min(abs(a.seniority - b.seniority), 15) / 15.0

    logit = (
        BIAS
        + W_RECIPROCAL * reciprocal
        + W_DIRECTIONAL * (dir_ab + dir_ba) / 2.0
        + W_MUTUAL_BONUS * mutual
        + W_SAME_ARC * same_arc
        + W_TOPIC * topic
        + W_SAME_DOMAIN * same_domain
        + W_SAME_CITY * same_city
        + W_STAGE_GAP * stage_gap
        + W_SENIORITY_GAP * seniority_gap
    )
    return _sigmoid(logit)


def make_cohort(n_people: int, seed: int, prefix: str = "p") -> list[Person]:
    rng = random.Random(seed)
    return [make_person(rng, i, prefix) for i in range(n_people)]


def make_pairs(
    people: list[Person],
    n_pairs: int,
    seed: int,
) -> list[LabeledPair]:
    """Sample distinct unordered pairs and label them from the latent model."""
    rng = random.Random(seed + 7919)
    n = len(people)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if n_pairs > len(all_pairs):
        raise ValueError(f"cannot draw {n_pairs} distinct pairs from {n} people")
    rng.shuffle(all_pairs)

    pairs: list[LabeledPair] = []
    for i, j in all_pairs[:n_pairs]:
        a, b = people[i], people[j]
        p = landing_probability(a, b)
        # Observation noise: the same two people don't always click the same way.
        p_obs = min(0.98, max(0.02, _sigmoid(math.log(p / (1 - p)) + rng.gauss(0, NOISE_SD))))
        pairs.append(LabeledPair(a=a, b=b, label=int(rng.random() < p_obs), p_true=p))
    return pairs


def build_datasets(
    seed: int = 0,
    n_people: int = 64,
    n_pairs: int = 200,
    n_people_cold: int = 34,
    n_pairs_cold: int = 100,
) -> tuple[list[LabeledPair], list[LabeledPair]]:
    """Main cohort plus a disjoint cold-start cohort of people never seen in training."""
    main = make_pairs(make_cohort(n_people, seed, "p"), n_pairs, seed)
    cold = make_pairs(make_cohort(n_people_cold, seed + 1000, "c"), n_pairs_cold, seed + 1000)
    return main, cold


def write_jsonl(pairs: list[LabeledPair], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[LabeledPair]:
    with path.open("r", encoding="utf-8") as fh:
        return [LabeledPair.from_dict(json.loads(line)) for line in fh if line.strip()]
