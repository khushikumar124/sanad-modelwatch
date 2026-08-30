# Contract intelligence: obligations, coverage, contradictions, review

Four features, each scoped honestly to what it can actually claim:

## Obligation & deadline extraction (`sanad/features/obligations.py`)

An LLM call (same pattern as `summarizer.py`) extracts who owes what to
whom, by when, categorized as one of payment / notice / renewal /
termination / deliverable / confidentiality / liability / other.
"Who owes what by when" needs language understanding a rule engine can't
do -- unlike risk_flagger.py's clause-pattern matching, this isn't
reducible to regex.

**Grounding, not trust.** Every obligation carries a `source_quote`, and
is checked against the document's own text before being trusted: exact
substring match first, falling back to a content-word-overlap check
(stopwords excluded, 0.85 threshold) when that fails. A real test
surfaced why the fallback matters: a genuine, correct extraction's quote
was one word off from the source ("by local" vs. "by the local"), which
an exact-only check flagged as unverified. Ungrounded obligations are
kept, not dropped -- shown with a `grounded: false` flag so a reader
sees what the model claimed without being asked to trust it blindly.

**Deduplication.** Real data showed phi3:3.8b repeating the same
obligation 2-6 times per response with slightly reworded text; exact
(party, obligation) duplicates (after normalization) are now collapsed.

**Known limitation, measured not hidden**: extraction recall varies
significantly by document. On one real rental contract, a single call
extracted only 1 obligation; on another (`regression.pdf`, 18 clauses),
the same model extracted 22 (6 after dedup collapsed repeats, ~16
distinct). This is a small-CPU-model schema-constrained-decoding
limitation, not a parsing bug -- consistent with this project's other
documented small-model limitations (see `DEMO.md`).

**Latency**: array-of-objects schema with an enum field is measurably
slower to decode than a flat-object schema (summarizer.py's). Observed
9-fewer-than-a-minute up to several minutes on a real document; the
timeout was raised to 420s (`EXTRACTION_TIMEOUT_SECONDS`) after a real
run hit the previous 180s default and surfaced as an unhandled 500 --
see "Bugs found" below.

## Missing/weak clause detection (`sanad/features/coverage.py`)

Rule-based (like `risk_flagger.py`), checking for termination, notice,
confidentiality, IP ownership, dispute resolution, liability, payment,
renewal, and governing-law language. Every result is `found` or
`not_found` -- **never** `missing`. A regex scan can prove a pattern
didn't match; it cannot prove a real lawyer reading the whole document
wouldn't find equivalent language phrased differently. `not_found` means
exactly what it says.

A real-document test caught a real bug before shipping: the notice-period
patterns required plural units ("months") and missed "one month" --
fixed to accept singular and plural.

## Contradiction detection (`sanad/features/contradictions.py`)

Deliberately narrow: parses a duration (days/weeks/months/years, digits
or number-words) out of each obligation's deadline within the `notice`
or `renewal` categories, and flags when a category has more than one
distinct value after normalizing to days. This is not general semantic
contradiction detection -- `termination` is excluded on purpose, since
different notice lengths for cause vs. no-cause termination are normal,
not a drafting error.

## Review synthesis (`sanad/features/review.py`)

No new LLM call, no new judgment -- purely re-ranks/re-phrases findings
the three modules above already computed: top issues (high-severity
risks + contradictions + high-importance coverage gaps), negotiable
clauses, templated negotiation questions (keyed by risk rule ID), and
clarification prompts for coverage gaps. Every item traces back to a
source report; nothing is synthesized from whole cloth.

The `/review` endpoint runs obligation extraction internally (needed for
contradiction detection) and returns those obligations in its response,
specifically so the frontend never needs a second, duplicate extraction
call -- see "Bugs found" below.

## Bugs found and fixed while building this

- **Timeout crashed as an unhandled 500.** `OllamaClient.generate()` only
  caught `ConnectionError`/`HTTPError`; a slow array-schema extraction
  hitting the 180s default surfaced as a raw `requests.exceptions.ReadTimeout`
  all the way to the API layer instead of the clean 503 every other
  LLM-unavailable path produces. Fixed: `generate()` now takes an
  explicit `timeout` parameter and catches `Timeout` too.
- **Duplicate LLM calls from the frontend.** The Review tab fetched
  `/review` and `/obligations` in parallel, unaware `/review` already
  runs the same (slow) obligation extraction internally -- doubling the
  slowest call on every page load. Fixed by having `/review` return the
  obligations it already extracted.
- **Grounding check too strict.** See obligations.py's own section above.
- **Notice-period regex missed singular units.** See coverage.py's
  section above.

None of these were caught by unit tests alone -- all four surfaced only
when the feature was run against a real document with a real model,
which is why `sanad/tests/test_coverage.py` includes a real-PDF test
alongside the controlled-input ones.
