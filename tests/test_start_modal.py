"""
Tests for the start dialog (#cifModal) served by templates/index.html:
accessibility markup, inline error handling and deferred "New CIF" reset.
"""
import re


def _function_body(html, signature):
    """Return the source of the top-level JS function starting at `signature`."""
    start = html.index(signature)
    # Functions in the inline script are indented by 8 spaces; the next
    # sibling declaration ends the body.
    match = re.search(r"\n        (?:async )?function |\n        // ", html[start + len(signature):])
    end = start + len(signature) + (match.start() if match else len(html))
    return html[start:end]


def test_start_modal_accessibility_markup(client):
    """The start dialog exposes dialog semantics and an error region."""
    html = client.get("/").text

    assert 'role="dialog"' in html
    assert 'aria-modal="true"' in html
    assert 'aria-labelledby="cifModalTitle"' in html
    assert 'id="cifModalTitle"' in html
    assert 'id="modalError"' in html
    assert 'aria-label="About ML potentials"' in html
    assert 'id="fileStatusArea" class="file-status-area" aria-live="polite"' in html


def test_start_modal_lifecycle_functions(client):
    """Open/close helpers and the deferred New CIF reset are defined and used."""
    html = client.get("/").text

    assert "function openCifModal(" in html
    assert "function closeCifModal(" in html
    assert "function ensurePendingReset(" in html
    assert "openCifModal({ dismissible: false });" in html

    new_cif = _function_body(html, "function resetAndLoadNewCIF()")
    assert "pendingNewCifReset = true;" in new_cif
    assert "setTimeout" not in new_cif

    for signature in ("async function analyzeSampleFile(", "async function analyzeUploadedFile("):
        body = _function_body(html, signature)
        assert "ensurePendingReset();" in body


def test_start_modal_functions_do_not_use_alert(client):
    """Errors in the start dialog are shown inline, not via alert()."""
    html = client.get("/").text

    for signature in (
        "function validateAtomCount(",
        "async function createSupercell()",
        "async function analyzeSampleFile(",
    ):
        body = _function_body(html, signature)
        assert "alert(" not in body, signature
