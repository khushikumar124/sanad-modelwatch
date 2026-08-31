"""Flag clauses that are unusual or work against the weaker party.

This is the proactive counterpart to the chatbot: the chatbot answers what
you thought to ask, this surfaces what you didn't know to ask about. That
matters for contracts specifically, because the risk in a rental agreement
or offer letter is rarely in the clause you were looking for.

Detection is rule-based rather than LLM-judged, deliberately:

* it is deterministic, so it can be tested against known inputs the same
  way the drift detectors are, instead of being spot-checked;
* it is instant, where asking a local model to assess ~20 clauses one at a
  time takes minutes;
* "this clause matches a documented unfavourable pattern, here is the
  text" is a defensible claim. "The model thought this looked risky" is
  not, and would inherit every hallucination problem the chatbot's
  grounding rules exist to prevent.

The rules encode patterns common in Indian rental, employment and
freelance agreements. They are intentionally conservative: a rule fires on
explicit contract language, not on inference. Every finding quotes the
clause it came from so the reader can judge it themselves.

Limitations are real and documented in the README: this cannot catch a
harmful term phrased in wording no rule anticipates, and a fired rule is
not legal advice -- an unusual clause may be perfectly reasonable in
context.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Pattern

from sanad.ingestion.chunking import Chunk

logger = logging.getLogger(__name__)

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


@dataclass(frozen=True)
class RiskFactor:
    """One decomposable sub-condition a rule's overall severity implicitly
    depends on -- e.g. a non-compete clause is a different kind of risk
    depending on whether it states a duration and a geographic scope.
    Detected the same way the rule itself is (a regex against the clause's
    own text, quoted when it matches), so a factor is either a real,
    checkable fact about the clause or explicitly absent -- never a
    fabricated weight or score."""
    key: str
    label: str
    pattern: Pattern[str]


@dataclass(frozen=True)
class RiskRule:
    rule_id: str
    label: str
    severity: str
    #: Why this matters, in plain language, for a non-lawyer.
    explanation: str
    #: Who the term tends to disadvantage.
    affects: str
    patterns: tuple[Pattern[str], ...]
    #: Optional decomposable factors shown alongside the finding -- see
    #: RiskFactor. Empty for rules where the single matched pattern is
    #: already the whole story (most rules); populated for rules whose
    #: real-world severity varies with details worth surfacing separately
    #: rather than folding into one opaque label.
    factor_checks: tuple[RiskFactor, ...] = ()


def _p(*expressions: str) -> tuple[Pattern[str], ...]:
    return tuple(re.compile(e, re.IGNORECASE | re.DOTALL) for e in expressions)


def _f(*factors: tuple[str, str, str]) -> tuple[RiskFactor, ...]:
    return tuple(RiskFactor(key=k, label=label, pattern=re.compile(p, re.IGNORECASE | re.DOTALL)) for k, label, p in factors)


#: The spec's own worked example: a non-compete clause isn't one flat risk
#: -- an unlimited-duration, worldwide restriction with no compensation is
#: a materially different problem than a 6-month, named-competitor one, even
#: though both clauses fire the same rule above. Each factor is checked
#: against the same clause text the rule matched on and reported as
#: present/absent with its own quote, rather than being collapsed into a
#: single number.
NON_COMPETE_FACTORS = _f(
    ("duration_specified", "States how long the restriction lasts",
     r"for\s+(a\s+period\s+of\s+)?\(?\d+\)?\s*\(?\w*\)?\s*(day|month|year)s?"),
    ("geographic_scope_specified", "States a geographic area the restriction covers",
     r"(within\s+a\s+radius\s+of|world[-\s]?wide|anywhere\s+in|geographic(al)?\s+(area|territory|scope)|territory\s+of)"),
    ("consideration_mentioned", "States a payment made specifically in exchange for the restriction",
     r"(consideration|compensation|payment)\s+(paid|payable|of\s+rs\.?)[^.]{0,60}(non[-\s]?compete|restriction)"),
)


RISK_RULES: list[RiskRule] = [
    RiskRule(
        rule_id="unilateral_termination",
        label="Other party can terminate at will",
        severity="high",
        explanation=(
            "One side can end the agreement whenever it likes, while the other is held to "
            "the full term or a notice period. That asymmetry leaves you without the "
            "stability the contract appears to give you."
        ),
        affects="the party who cannot terminate freely",
        patterns=_p(
            r"(lessor|owner|landlord|company|employer)[^.]{0,120}?right to terminate[^.]{0,120}?at any (point|time)",
            r"terminate[^.]{0,60}?with or without notice[^.]{0,60}?at any time",
        ),
    ),
    RiskRule(
        rule_id="termination_without_notice",
        label="Termination without notice or reason",
        severity="high",
        explanation=(
            "Your engagement can be ended with no notice period and no reason given. You "
            "would have no warning and no stated grounds to contest."
        ),
        affects="the employee or contractor",
        patterns=_p(
            r"terminated without notice and without assigning any reason",
            r"terminate[^.]{0,80}?without (any )?(prior )?notice",
        ),
    ),
    RiskRule(
        rule_id="penalty_multiplier",
        label="Penalty charged at a multiple of the normal amount",
        severity="high",
        explanation=(
            "Breaching this term costs you a multiple of the usual payment, not the actual "
            "loss caused. Check the multiplier and what triggers it."
        ),
        affects="the tenant or paying party",
        patterns=_p(
            r"(two|three|double|twice|thrice)\s*(\(\d\)\s*)?times?\s+the\s+(rent|lease|amount|fee)",
            r"damages calculated at\s+\w+\s+times",
        ),
    ),
    RiskRule(
        rule_id="unilateral_change_of_terms",
        label="Other party can change the terms later",
        severity="high",
        explanation=(
            "The terms you are agreeing to can be altered by the other side after signing. "
            "What you sign today is not necessarily what binds you later."
        ),
        affects="the party who cannot change terms",
        patterns=_p(
            r"terms and conditions[^.]{0,80}?subject to (future )?change by the (lessor|owner|company|employer)",
            r"(company|employer|lessor)[^.]{0,60}?reserves the right to (modify|amend|change)[^.]{0,60}?(terms|policy)",
        ),
    ),
    RiskRule(
        rule_id="non_compete",
        label="Restriction on working elsewhere",
        severity="high",
        explanation=(
            "You are restricted from competing, or from working with similar businesses, "
            "possibly after this agreement ends. Check how long it lasts and how broadly "
            "'competing' is defined."
        ),
        affects="the employee or contractor",
        patterns=_p(
            r"non[-\s]?compete",
            r"not to[^.]{0,140}?(compete|competing)[^.]{0,80}?business",
            r"shall not[^.]{0,100}?engage[^.]{0,80}?(similar|competing) business",
        ),
        factor_checks=NON_COMPETE_FACTORS,
    ),
    RiskRule(
        rule_id="deposit_forfeiture",
        label="Deposit may be forfeited",
        severity="high",
        explanation=(
            "Your deposit can be kept rather than returned in certain situations. Check "
            "exactly which situations, and whether they are within your control."
        ),
        affects="the tenant or paying party",
        patterns=_p(
            r"(deposit|advance)[^.]{0,120}?(shall|may|will) be forfeit",
            r"forfeit[^.]{0,80}?(security )?deposit",
        ),
    ),
    RiskRule(
        rule_id="ip_assignment",
        label="You give up ownership of what you create",
        severity="medium",
        explanation=(
            "Everything you produce under this agreement belongs to the other side, often "
            "including rights that would otherwise stay yours. Check whether work you "
            "created beforehand is swept in too."
        ),
        affects="the consultant, freelancer or employee",
        patterns=_p(
            r"work for hire",
            r"(vest|vests|shall vest) (solely |exclusively )?with the company",
            r"irrevocably (and perpetually )?assigns",
        ),
    ),
    RiskRule(
        rule_id="lock_in_period",
        label="Lock-in period",
        severity="medium",
        explanation=(
            "You are committed for a minimum duration and cannot exit early without a "
            "penalty, even if your circumstances change."
        ),
        affects="the tenant or employee",
        patterns=_p(
            r"lock[-\s]?in period",
            r"shall not (vacate|leave|resign)[^.]{0,80}?before[^.]{0,60}?(months|years)",
        ),
    ),
    RiskRule(
        rule_id="auto_renewal",
        label="Renews automatically",
        severity="medium",
        explanation=(
            "The agreement continues on its own unless you actively stop it. Missing the "
            "cancellation window means you are bound for another term."
        ),
        affects="both parties, but usually whoever forgets",
        patterns=_p(
            r"automatically (convert|converts|renew|renews|be renewed)",
            r"shall (stand )?renewed automatically",
        ),
    ),
    RiskRule(
        rule_id="broad_indemnity",
        label="Broad indemnity obligation",
        severity="medium",
        explanation=(
            "You agree to cover the other side's losses, legal costs and damages. This can "
            "extend well beyond your own fault, so check what triggers it."
        ),
        affects="the indemnifying party",
        patterns=_p(
            r"indemnify,? (and )?(defend,? )?(and )?hold harmless",
            r"shall[^.]{0,60}?indemnify[^.]{0,100}?(losses|damages|claims)",
        ),
    ),
    RiskRule(
        rule_id="liquidated_damages",
        label="Pre-agreed damages on breach",
        severity="medium",
        explanation=(
            "A fixed sum is payable if you break this term, regardless of the actual loss "
            "suffered. You cannot argue the real damage was smaller."
        ),
        affects="the breaching party",
        patterns=_p(r"liquidated damages"),
    ),
    RiskRule(
        rule_id="maintenance_on_weaker_party",
        label="Repairs and upkeep fall on you",
        severity="low",
        explanation=(
            "Day-to-day repairs are your responsibility and expense. Usually reasonable, "
            "but worth knowing before you sign, and check where 'minor' ends."
        ),
        affects="the tenant",
        patterns=_p(
            r"minor repairs[^.]{0,140}?(shall be )?(carried out|borne|responsibility)[^.]{0,60}?tenant",
            r"day[-\s]?to[-\s]?day[^.]{0,80}?repairs[^.]{0,80}?tenant",
        ),
    ),
    RiskRule(
        rule_id="assignment_restriction",
        label="You cannot transfer or sublet",
        severity="low",
        explanation=(
            "You cannot pass your rights to anyone else, including subletting, without "
            "written permission. Normal, but it removes an exit route."
        ),
        affects="the tenant or contractor",
        patterns=_p(
            r"shall not sublet",
            r"not to sublet, assign or transfer",
        ),
    ),
]


@dataclass(frozen=True)
class FactorResult:
    key: str
    label: str
    present: bool
    #: The matched text when present, else None -- never fabricated.
    quote: str | None

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "label": self.label, "present": self.present, "quote": self.quote}


@dataclass
class RiskFinding:
    rule_id: str
    label: str
    severity: str
    explanation: str
    affects: str
    clause_heading: str | None
    excerpt: str
    chunk_index: int
    #: Decomposable factors for this specific finding (see RiskFactor) --
    #: empty for rules with no factor_checks defined. Never a computed
    #: score: each entry is a real present/absent fact about the clause.
    factors: list[FactorResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "label": self.label,
            "severity": self.severity,
            "explanation": self.explanation,
            "affects": self.affects,
            "clause_heading": self.clause_heading,
            "excerpt": self.excerpt,
            "chunk_index": self.chunk_index,
            "factors": [f.to_dict() for f in self.factors],
        }


@dataclass
class RiskReport:
    findings: list[RiskFinding] = field(default_factory=list)
    clauses_scanned: int = 0

    @property
    def counts(self) -> dict[str, int]:
        out = {"high": 0, "medium": 0, "low": 0}
        for f in self.findings:
            out[f.severity] = out.get(f.severity, 0) + 1
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "findings": [f.to_dict() for f in self.findings],
            "clauses_scanned": self.clauses_scanned,
            "counts": self.counts,
        }


def _excerpt_around(text: str, match: re.Match[str], width: int = 260) -> str:
    start = max(0, match.start() - width // 3)
    end = min(len(text), match.end() + width)
    snippet = " ".join(text[start:end].split())
    return ("…" if start > 0 else "") + snippet + ("…" if end < len(text) else "")


def flag_risks(chunks: list[Chunk], rules: list[RiskRule] | None = None) -> RiskReport:
    """Scan already-chunked contract text for known unfavourable patterns.

    Operates on the same clause-aware chunks the chatbot retrieves over, so
    a finding always points at a real clause rather than an arbitrary
    window of characters.
    """
    rules = rules if rules is not None else RISK_RULES
    findings: list[RiskFinding] = []

    for chunk in chunks:
        for rule in rules:
            for pattern in rule.patterns:
                match = pattern.search(chunk.text)
                if match is None:
                    continue
                factors = []
                for fc in rule.factor_checks:
                    fmatch = fc.pattern.search(chunk.text)
                    factors.append(FactorResult(
                        key=fc.key, label=fc.label,
                        present=fmatch is not None,
                        quote=" ".join(fmatch.group(0).split()) if fmatch else None,
                    ))
                findings.append(
                    RiskFinding(
                        rule_id=rule.rule_id,
                        label=rule.label,
                        severity=rule.severity,
                        explanation=rule.explanation,
                        affects=rule.affects,
                        clause_heading=chunk.heading,
                        excerpt=_excerpt_around(chunk.text, match),
                        chunk_index=chunk.index,
                        factors=factors,
                    )
                )
                break  # one finding per rule per clause, not per phrasing

    findings.sort(key=lambda f: (SEVERITY_ORDER.get(f.severity, 9), f.chunk_index))
    logger.info(
        "risk scan complete",
        extra={"clauses": len(chunks), "findings": len(findings)},
    )
    return RiskReport(findings=findings, clauses_scanned=len(chunks))
