# Sponsors

Who signs up for what, so credit/key approval isn't a bottleneck mid-event. One
sign-up per tool, no overlap, so Tool Use in the demo stays unambiguous.

| Tool | Purpose in Kindred | Owner | Account tier | Used in |
|---|---|---|---|---|
| **Actian** | Vector store for nearest-neighbour profile retrieval | P1 | Pro | Backend — `POST /graph` |
| **Gemini** (DeepMind) | Profile extraction and reasoning | P1 | Pro | Backend — `POST /profile` |
| **Pioneer / Fastino** | Fine-tuned match scorer | P2 | Pro | `score_pair()`, F1 vs baseline — P2 owns the whole fine-tuning workstream |
| **BAND** | Agent-communication / intro-thread API | Raj | 3× Max | The Introducer + the village viz that showcases BAND |
| **Guild** (optional) | Weight/version tracking | Raj | 3× Max | Skip under time pressure — only if time allows |

## Sign up early

**Actian, Gemini, and Pioneer are on the critical path** — register accounts and grab
keys/credits in the **first 30 minutes** of the hackathon. Approval on these can be
slow, and the backend and scorer workstreams block on them.

**BAND and Guild are lower urgency.** They sit at the end of the pipeline (intro
thread, optional versioning), so sign-up can wait — Guild can be skipped entirely if
time runs short.
