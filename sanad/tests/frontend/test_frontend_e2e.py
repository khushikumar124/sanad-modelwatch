"""Browser-driven tests of sanad/frontend/index.html against the real
running app (live_server fixture in conftest.py) -- the same kind of
verification that was previously only done manually (upload, risk scan,
document reader, chat, sidebar search, mobile breakpoint) during UI
work, now automated so a future change can't silently break them.

Deliberately scoped to features that don't need a real LLM to produce
meaningful output: the risk scan is pure rule-based, the document reader
just renders extracted text, and chat uses a FakeLLMClient (see
conftest.py) since only the frontend's rendering of *a* grounded answer
is under test here, not answer quality (that's sanad/tests/test_chatbot.py
and sanad/evaluation/'s job).

NOT covered here yet: Overview, Summary, and Review rendering their
*correct* content. All three run a background LLM job and expect the
model to return a specific multi-field JSON shape (15 Overview fields,
obligations, coverage) -- a FakeLLMClient faking all three shapes
accurately is real additional work, not just more of what's already
here. Left as a follow-up rather than done badly. The one Overview/
Review behavior tested below (test_revisiting_overview_before_its_job_
finishes_does_not_start_a_second_one) doesn't need a correctly-shaped
response, since it's only checking how many jobs get *started*, not
what they return -- see the regression this test guards against.
"""
from __future__ import annotations

from pathlib import Path

import pytest

RENTAL_DOC = str(Path(__file__).resolve().parents[2] / "sample_docs" / "rental" / "rental_agreement_sample_1.pdf")
FREELANCE_DOC = str(Path(__file__).resolve().parents[2] / "sample_docs" / "freelance" / "freelance_agreement_sample1.pdf")


def _upload(page, base_url, file_path: str):
    filename = Path(file_path).name
    page.goto(base_url)
    page.set_input_files("#fileInput", file_path)
    # "#workspace becomes visible" isn't specific enough -- it's already
    # true if any earlier document exists and was auto-selected on page
    # load, so it can resolve before *this* upload's extract/chunk/embed
    # round-trip actually finishes. #docTitle only shows this filename
    # once selectDoc() runs for *this* upload's returned doc_id.
    page.wait_for_selector(f"#docTitle:text-is('{filename}')", timeout=20000)


@pytest.fixture(scope="module", autouse=True)
def uploaded_rental_doc(live_server, browser):
    """Uploads once per module via a throwaway page/context, so the
    per-test `page` fixture below always finds at least the rental
    document already in the sidebar -- mirrors how a real session
    accumulates documents rather than starting empty every test."""
    context = browser.new_context()
    page = context.new_page()
    _upload(page, live_server, RENTAL_DOC)
    context.close()


def test_uploading_a_document_shows_it_in_the_sidebar_and_opens_the_workspace(page, live_server):
    _upload(page, live_server, FREELANCE_DOC)
    assert page.locator(".doc-item", has_text="freelance_agreement_sample1.pdf").count() == 1
    assert page.locator("#docTitle").inner_text() == "freelance_agreement_sample1.pdf"


def test_risk_scan_renders_real_findings_with_the_donut_and_heatmap(page, live_server):
    page.goto(live_server)
    page.click(".doc-item >> text=rental_agreement_sample_1.pdf")
    page.click('.nav-item[data-pane="risks"]')

    # Rule-based, no LLM involved -- should render promptly and for real.
    page.wait_for_selector("#risksBody .risk-overview", timeout=10000)
    assert page.locator("#risksBody .donut-num").count() == 1
    assert page.locator("#risksBody .heatmap-cell").count() > 0
    # At least one real finding with a quoted excerpt from the actual document.
    assert page.locator("#risksBody .risk-quote").count() > 0


def test_clicking_view_in_document_jumps_to_and_highlights_the_clause(page, live_server):
    page.goto(live_server)
    page.click(".doc-item >> text=rental_agreement_sample_1.pdf")
    page.click('.nav-item[data-pane="risks"]')
    page.wait_for_selector("#risksBody .risk", timeout=10000)

    page.click("#risksBody .risk >> text=View in document →")
    # The tab switch itself (nav-item.active) is synchronous and fires
    # before loadDocument()'s async fetch resolves -- wait for the flash
    # class directly, the thing this test actually cares about, not an
    # earlier signal that races ahead of it.
    page.wait_for_selector(".clause.clause-flash", timeout=5000)


def test_document_reader_renders_real_extracted_clause_text(page, live_server):
    page.goto(live_server)
    page.click(".doc-item >> text=rental_agreement_sample_1.pdf")
    page.click('.nav-item[data-pane="document"]')
    page.wait_for_selector("#documentBody .clause", timeout=10000)
    assert page.locator("#documentBody .clause-text").count() > 0
    # Real contract text, not a placeholder.
    assert "rent" in page.locator("#documentBody").inner_text().lower()


def test_sidebar_search_filters_the_document_list(page, live_server):
    page.goto(live_server)
    # Unlike .click(), .count() doesn't auto-wait for the element to
    # appear -- it just reads however many matches exist at that instant,
    # which can be zero if the /api/documents fetch hasn't resolved yet.
    page.wait_for_selector(".doc-item", timeout=5000)
    total_docs = page.locator(".doc-item").count()
    assert total_docs >= 2  # rental + freelance from earlier tests

    page.fill("#docSearch", "freelance")
    assert page.locator(".doc-item").count() == 1
    assert page.locator(".doc-item", has_text="freelance").count() == 1

    page.fill("#docSearch", "")
    assert page.locator(".doc-item").count() == total_docs


def test_asking_a_question_renders_a_grounded_answer_bubble_with_citation(page, live_server):
    page.goto(live_server)
    page.click(".doc-item >> text=rental_agreement_sample_1.pdf")
    page.click('.nav-item[data-pane="ask"]')
    assert "rental_agreement_sample_1.pdf" in page.locator("#chatContext").inner_text()

    page.fill("#chatInput", "What law governs this agreement?")
    page.click("#askBtn")

    # ".turn.bot .bubble" also matches the transient "thinking…" placeholder
    # bubble appended before the fake LLM call resolves -- wait for the
    # citation disclosure instead, which only renders with the real answer.
    page.wait_for_selector(".turn.bot .cite", timeout=10000)
    bot_bubble = page.locator(".turn.bot .bubble").last
    assert "India" in bot_bubble.inner_text()


def test_revisiting_overview_before_its_job_finishes_does_not_start_a_second_one(page, live_server):
    """Regression test for a real bug found live: /overview/job starts a
    brand-new background LLM job on every call with no server-side
    dedupe, so switching away from the Overview tab and back (or just
    clicking it twice) before the first job finished used to start a
    *second* job for the same document -- whichever finished second
    silently overwrote whichever rendered first, which looked exactly
    like "the overview flashed and then vanished". The fix
    (overviewInFlight in frontend/index.html) doesn't need a correctly
    shaped LLM response to verify -- it only needs to be right about how
    many jobs get *started*, not what they return.
    """
    page.goto(live_server)
    page.click(".doc-item >> text=rental_agreement_sample_1.pdf")

    overview_job_requests = []
    page.on("request", lambda req: overview_job_requests.append(req.url) if "/overview/job" in req.url else None)

    page.click('.nav-item[data-pane="overview"]')
    page.click('.nav-item[data-pane="risks"]')
    page.click('.nav-item[data-pane="overview"]')
    page.click('.nav-item[data-pane="risks"]')
    page.click('.nav-item[data-pane="overview"]')
    page.wait_for_timeout(500)  # let any (incorrect) extra request actually get sent before asserting

    assert len(overview_job_requests) == 1


def test_mobile_viewport_hides_sidebar_and_collapses_nav_to_horizontal_strip(page, live_server):
    # Select the document at desktop size first -- the sidebar (and the
    # doc-item this needs to click) is display:none on mobile by design,
    # so a document has to be chosen before shrinking the viewport.
    page.goto(live_server)
    page.click(".doc-item >> text=rental_agreement_sample_1.pdf")
    page.wait_for_selector("#docTitle:text-is('rental_agreement_sample_1.pdf')")

    page.set_viewport_size({"width": 375, "height": 812})

    assert not page.locator(".sidebar").is_visible()
    nav_box = page.locator(".section-nav").bounding_box()
    # A collapsed horizontal strip is short; the desktop vertical rail is tall.
    assert nav_box["height"] < 100
