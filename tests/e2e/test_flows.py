"""Real-browser user-flow tests (Playwright) against the fake EMT backend.

Known frontend bugs are documented with ``xfail(strict=True, reason="Fn: ...")``
until fixed; the PR that fixes a bug removes the matching marker.
"""
import io
import json
import re
import zipfile

import pytest
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import expect

from tests.e2e.helpers import (SAMPLE_CU, SAMPLE_NI, Hold, analyze, atom_count,
                               cif_atom_rows, cif_symbols, labels, load_sample,
                               new_cif_dialog, open_app, pick_sample_and_confirm,
                               run_manual_op, start_auto_sweep, start_manual_op)

pytestmark = pytest.mark.e2e

BUTTON_PROCESSING = re.compile("button-processing")


@pytest.fixture
def app(page, live_server):
    open_app(page, live_server)
    return page


def read_download(download) -> str:
    with open(download.path(), encoding="utf-8") as fh:
        return fh.read()


def download_via_save_button(page, timeout=8_000) -> str:
    try:
        with page.expect_download(timeout=timeout) as info:
            page.click("#downloadCifBtn")
    except PlaywrightTimeoutError:
        pytest.fail(f"no download started; alerts={page.log.dialogs}; "
                    f"failed requests={page.log.failed_requests}", pytrace=False)
    return read_download(info.value)


# ---------------------------------------------------------------------------
# Loading, supercell and edits
# ---------------------------------------------------------------------------

def test_load_sample_supercell_edit_and_reset(app):
    page = app
    load_sample(page, SAMPLE_CU, supercell=(2, 1, 1))
    assert atom_count(page) == 8
    expect(page.locator("#main-crystal-info")).to_contain_text("Cu8")

    # Substitute Cu0 -> Ni (Manual)
    run_manual_op(page, "Cu0", "Ni")
    assert atom_count(page) == 8
    expect(page.locator("#main-crystal-info")).to_contain_text("Cu7 Ni1")

    # Delete Cu1 (Manual)
    run_manual_op(page, "Cu1", "DELETE")
    assert atom_count(page) == 7

    # Insert a Cu atom (Insert New tab, Auto: most stable void)
    page.click("#tab-insert")
    expect(page.locator("#insert-workflow")).to_be_visible()
    page.wait_for_function(
        "document.querySelector('#insert-element').options.length > 1")
    page.select_option("#insert-element", "Cu")
    expect(page.locator("#execute-btn")).to_be_enabled()
    page.click("#execute-btn")
    page.wait_for_function("window.currentLabels.length === 8", timeout=60_000)
    expect(page.locator("#execute-btn")).not_to_have_class(BUTTON_PROCESSING)

    # Reset returns to the original supercell
    page.click("#reset-operations-btn")
    page.wait_for_function("window.currentLabels.length === 8 && window.atomicOperations.length === 0")
    expect(page.locator("#main-crystal-info")).to_contain_text("Cu8")
    assert all(l.startswith("Cu") for l in labels(page))


# ---------------------------------------------------------------------------
# Relaxation, SSE progress, details modal
# ---------------------------------------------------------------------------

def test_relax_shows_progress_results_and_details_modal(app):
    page = app
    load_sample(page, SAMPLE_CU, supercell=(2, 1, 1))
    # The relax opens the SSE progress stream for its session.
    with page.expect_request("**/api/relax-progress/**"):
        analyze(page)

    expect(page.locator("#mlip-results")).to_be_visible()
    expect(page.locator("#energy-summary")).to_contain_text(re.compile(r"Energy:\s*-?\d"))
    expect(page.locator("#relax-charts")).to_be_visible()
    expect(page.locator("#calculation-time")).to_be_visible()

    # Energy detail drawer
    page.click("#energy-detail-btn")
    expect(page.locator("#energy-detail")).to_be_visible()

    # Details modal
    page.click("#detailAnalysisBtn")
    expect(page.locator("#analysisModal")).to_be_visible()
    expect(page.locator("#summary-energy")).not_to_have_text("-")
    expect(page.locator("#summary-convergence")).not_to_have_text("-")
    page.keyboard.press("Escape")
    expect(page.locator("#analysisModal")).to_be_hidden()


def test_relax_with_elastic_shows_elastic_panel(app):
    page = app
    load_sample(page, SAMPLE_CU, supercell=(1, 1, 1))
    page.click("label[for=elastic-toggle]")
    analyze(page)
    expect(page.locator("#elastic-item")).to_be_visible(timeout=60_000)
    expect(page.locator("#elastic-summary")).to_contain_text("GPa")


# ---------------------------------------------------------------------------
# 💾 download of the current CIF in three states
# ---------------------------------------------------------------------------

def test_download_current_cif_unrelaxed(app):
    page = app
    load_sample(page, SAMPLE_CU, supercell=(2, 1, 1))
    text = download_via_save_button(page)
    assert cif_symbols(text) == ["Cu"] * 8


def test_download_current_cif_after_relax(app):
    page = app
    load_sample(page, SAMPLE_CU, supercell=(2, 1, 1))
    analyze(page)
    text = download_via_save_button(page)
    assert cif_symbols(text) == ["Cu"] * 8
    # Relaxed Cu cell (EMT) is slightly larger than the sample's 3.577 A
    a = float(re.search(r"_cell_length_a\s+([\d.]+)", text).group(1))
    assert a == pytest.approx(2 * 3.5897, abs=0.01)


def test_download_current_cif_after_relax_and_edit(app):
    page = app
    load_sample(page, SAMPLE_CU, supercell=(2, 1, 1))
    analyze(page)
    run_manual_op(page, "Cu0", "Ni")
    text = download_via_save_button(page)
    assert cif_symbols(text) == ["Cu"] * 7 + ["Ni"]


def test_export_zip_contains_relaxed_cif(app):
    page = app
    load_sample(page, SAMPLE_CU, supercell=(2, 1, 1))
    analyze(page)
    with page.expect_download() as info:
        page.click("#downloadAnalysisBtn")
    with zipfile.ZipFile(io.BytesIO(open(info.value.path(), "rb").read())) as zf:
        names = zf.namelist()
        assert "structure_P1.cif" in names
        cif = zf.read("structure_P1.cif").decode()
    assert cif_symbols(cif) == ["Cu"] * 8


# ---------------------------------------------------------------------------
# New CIF resets everything
# ---------------------------------------------------------------------------

def _visible(page, selector) -> bool:
    return page.evaluate(
        "s => { const e = document.querySelector(s); return !!e && "
        "getComputedStyle(e).display !== 'none' && e.offsetParent !== null; }", selector)


def test_new_cif_hides_elastic_drawers_and_report(app):
    page = app
    load_sample(page, SAMPLE_CU, supercell=(2, 1, 1))
    page.click("label[for=elastic-toggle]")
    analyze(page)
    expect(page.locator("#elastic-item")).to_be_visible(timeout=60_000)
    page.click("#energy-detail-btn")
    page.click("#elastic-detail-btn")
    expect(page.locator("#energy-detail")).to_be_visible()
    expect(page.locator("#elastic-detail")).to_be_visible()
    # A completed Auto sweep (some candidates failed) leaves #calculation-report
    # behind. The fake backend never fails candidates, so render the report the
    # way the sweep code does.
    page.evaluate("showCalculationReport({successRate: 50, successfulCalculations: 1, "
                  "totalAtoms: 2}, [], [{atom: 'Cu1', finalError: 'boom'}], "
                  "{atom: 'Cu0', energy: -1.0}, 'substitution')")
    expect(page.locator("#calculation-report")).to_be_visible()

    new_cif_dialog(page)
    pick_sample_and_confirm(page, SAMPLE_NI)

    stale = [sel for sel in ("#elastic-item", "#elastic-detail", "#energy-detail",
                             "#calculation-report") if _visible(page, sel)]
    assert not stale, f"stale panels still visible after New CIF: {stale}"
    expect(page.locator("#analysisActions")).to_be_hidden()


def test_new_cif_hides_structure_status_banner(app):
    page = app
    load_sample(page, SAMPLE_CU, supercell=(2, 1, 1))
    analyze(page)
    run_manual_op(page, "Cu0", "DELETE")
    expect(page.locator("#structure-status")).to_be_visible()  # edit invalidated the analysis

    new_cif_dialog(page)
    pick_sample_and_confirm(page, SAMPLE_NI)
    assert not _visible(page, "#structure-status"),         "'Structure modified - Analysis required' banner still shown for the new structure"
    expect(page.locator("#analysisActions")).to_be_hidden()


def test_new_cif_resets_core_state(app):
    page = app
    load_sample(page, SAMPLE_CU, supercell=(2, 1, 1))
    analyze(page)
    new_cif_dialog(page)
    pick_sample_and_confirm(page, SAMPLE_NI)
    assert page.evaluate("window.currentLabels.every(l => l.startsWith('Ni'))")
    assert page.evaluate("window.atomicOperations.length") == 0
    expect(page.locator("#analysisActions")).to_be_hidden()
    expect(page.locator("#analyze-btn")).not_to_contain_text("Calculated")
    expect(page.locator("#calculation-time")).to_be_hidden()
    expect(page.locator("#relax-charts")).to_be_hidden()
    expect(page.locator("#energy-summary")).not_to_contain_text(re.compile(r"-?\d+\.\d+ eV"))


# ---------------------------------------------------------------------------
# Auto sweep: completion, Cancel, Reset during the sweep
# ---------------------------------------------------------------------------

def test_auto_sweep_substitution_completes(app):
    page = app
    load_sample(page, SAMPLE_CU, supercell=(2, 1, 1))
    run_manual_op(page, "Cu0", "Ni")
    # With one Ni present the Cu sites fall into several inequivalent
    # classes, so the sweep really evaluates candidate energies.
    start_auto_sweep(page, "Cu", "Ni")
    page.wait_for_function("window.atomicOperations.length === 2", timeout=60_000)
    expect(page.locator("#execute-btn")).not_to_have_class(BUTTON_PROCESSING)
    assert sorted(l[:2] for l in labels(page)).count("Ni") == 2


def test_cancel_during_auto_sweep(app):
    page = app
    load_sample(page, SAMPLE_CU, supercell=(2, 1, 1))
    run_manual_op(page, "Cu0", "Ni")
    hold = Hold(page, "**/api/evaluate-candidate-energies")
    start_auto_sweep(page, "Cu", "Ni")
    expect(page.locator("#cancel-btn")).to_be_visible()
    hold.wait_held(page)
    page.click("#cancel-btn")
    expect(page.locator("#execute-status")).to_be_hidden()
    expect(page.locator("#execute-btn")).not_to_have_class(BUTTON_PROCESSING)
    hold.release()
    page.wait_for_timeout(500)
    # Cancelled: nothing applied.
    assert page.evaluate("window.atomicOperations.length") == 1
    assert sorted(l[:2] for l in labels(page)).count("Ni") == 1
    expect(page.locator("#execute-status")).to_be_hidden()
    # The app is usable again: a fresh sweep succeeds.
    start_auto_sweep(page, "Cu", "Ni")
    page.wait_for_function("window.atomicOperations.length === 2", timeout=60_000)


def test_reset_during_auto_sweep(app):
    page = app
    load_sample(page, SAMPLE_CU, supercell=(2, 1, 1))
    run_manual_op(page, "Cu0", "Ni")  # makes Reset available
    hold = Hold(page, "**/api/evaluate-candidate-energies")
    start_auto_sweep(page, "Cu", "Ni")
    expect(page.locator("#cancel-btn")).to_be_visible()
    hold.wait_held(page)
    page.click("#reset-operations-btn")
    page.wait_for_function("window.atomicOperations.length === 0")
    expect(page.locator("#main-crystal-info")).to_contain_text("Cu8")
    hold.release()
    page.wait_for_timeout(800)
    expect(page.locator("#execute-status")).to_be_hidden()
    expect(page.locator("#cancel-btn")).to_be_hidden()
    expect(page.locator("#execute-btn")).not_to_have_class(BUTTON_PROCESSING)
    assert page.evaluate("window.atomicOperations.length") == 0
    assert all(l.startswith("Cu") for l in labels(page))


# ---------------------------------------------------------------------------
# Upload errors
# ---------------------------------------------------------------------------

def test_invalid_upload_shows_error_and_valid_upload_works(app, tmp_path):
    page = app
    page.log.allow_errors = True  # the 400 and console.error are the point
    bad = tmp_path / "broken.cif"
    bad.write_text("this is not a CIF file\n")
    page.set_input_files("#uploadFile", str(bad))
    expect(page.locator("#fileStatusArea")).to_contain_text(re.compile("error|invalid|fail|could not|unable", re.I))
    expect(page.locator("#confirmButton")).to_be_disabled()

    # A valid file afterwards loads fine.
    good = tmp_path / "Cu.cif"
    good.write_text(open("sample_cif/Metals/Cu.cif", encoding="utf-8").read())
    page.set_input_files("#uploadFile", str(good))
    expect(page.locator("#confirmButton")).to_be_enabled()
    page.click("#confirmButton")
    expect(page.locator("#cifModal")).to_be_hidden()
    wait = page.wait_for_function("window.currentLabels && window.currentLabels.length === 4")
    assert wait


def test_failed_upload_clears_file_input_so_same_file_can_be_reselected(app, tmp_path):
    page = app
    page.log.allow_errors = True
    bad = tmp_path / "fixme.cif"
    bad.write_text("this is not a CIF file\n")
    page.set_input_files("#uploadFile", str(bad))
    expect(page.locator("#fileStatusArea")).to_contain_text(re.compile("error|invalid|fail|could not|unable", re.I))
    # Re-choosing the same path only fires `change` if the input was cleared.
    expect(page.locator("#uploadFile")).to_have_value("")


# ---------------------------------------------------------------------------
# Relax edge cases (mocked responses / held requests)
# ---------------------------------------------------------------------------

def test_relax_response_with_null_energy_is_handled(app):
    page = app
    load_sample(page, SAMPLE_CU, supercell=(1, 1, 1))

    def mangle(route):
        response = route.fetch()
        body = response.json()
        body["final_prediction"]["energy_eV_per_atom"] = None
        body["final_prediction"]["total_energy_eV"] = None
        route.fulfill(response=response, json=body)

    page.route("**/api/relax", mangle)
    page.click("#analyze-btn", force=True)
    # Whatever the UI chooses to show, it must not blow up with a TypeError
    # and the rest of the results (actions row) must still be displayed.
    expect(page.locator("#analyze-btn")).to_be_enabled(timeout=30_000)
    page.wait_for_timeout(500)
    assert not page.log.console_errors, page.log.console_errors
    expect(page.locator("#analysisActions")).to_be_visible()


def test_new_cif_mid_relax_does_not_pollute_new_state(app):
    page = app
    load_sample(page, SAMPLE_CU, supercell=(1, 1, 1))
    stale_event = json.dumps({
        "step": 7, "max_steps": 100, "max_force_eV_per_A": 0.5, "fmax": 0.01,
        "history": [{"step": 1, "energy_eV": -0.1, "max_force_eV_per_A": 0.9},
                    {"step": 2, "energy_eV": -0.2, "max_force_eV_per_A": 0.7}],
    })
    page.route("**/api/relax-progress/**", lambda route: route.fulfill(
        status=200, content_type="text/event-stream",
        headers={"Cache-Control": "no-cache"}, body=f"data: {stale_event}\n\n"))
    hold = Hold(page, "**/api/relax")
    page.click("#analyze-btn", force=True)
    hold.wait_held(page)
    expect(page.locator("#status-text")).to_contain_text("Step 7 / 100")

    # Abandon the relax: load a different structure.
    new_cif_dialog(page)
    pick_sample_and_confirm(page, SAMPLE_NI)
    page.wait_for_timeout(1500)  # long enough for several SSE (re)connects

    expect(page.locator("#relax-charts")).to_be_hidden()
    expect(page.locator("#mlip-status")).to_be_hidden()
    hold.release()
    page.wait_for_timeout(800)
    expect(page.locator("#analysisActions")).to_be_hidden()
    expect(page.locator("#analyze-btn")).not_to_contain_text("Calculated")
    expect(page.locator("#relax-charts")).to_be_hidden()


# ---------------------------------------------------------------------------
# Reset races
# ---------------------------------------------------------------------------

def test_reset_while_manual_edit_in_flight(app):
    page = app
    load_sample(page, SAMPLE_CU, supercell=(2, 1, 1))
    run_manual_op(page, "Cu0", "Ni")  # makes Reset available
    hold = Hold(page, "**/api/apply-atomic-operations")
    start_manual_op(page, "Cu1", "DELETE")
    hold.wait_held(page)
    page.click("#reset-operations-btn")
    page.wait_for_function("window.atomicOperations.length === 0")
    expect(page.locator("#main-crystal-info")).to_contain_text("Cu8")
    hold.release()
    page.wait_for_timeout(1000)
    expect(page.locator("#main-crystal-info")).to_contain_text("Cu8")
    assert atom_count(page) == 8
    assert all(l.startswith("Cu") for l in labels(page))
    assert page.evaluate("window.atomicOperations.length") == 0


def test_backend_reset_failure_keeps_ui_in_sync(app):
    page = app
    page.log.allow_errors = True  # the injected 500 and the app's console.error
    load_sample(page, SAMPLE_CU, supercell=(2, 1, 1))
    run_manual_op(page, "Cu0", "DELETE")
    assert atom_count(page) == 7
    page.route("**/api/reset-session-structure", lambda route: route.fulfill(
        status=500, json={"detail": "boom"}))
    page.click("#reset-operations-btn")
    page.wait_for_timeout(1000)
    # Server still holds the 7-atom structure; the UI must not pretend otherwise.
    assert atom_count(page) == 7


# ---------------------------------------------------------------------------
# Server atom limit, 429 handling, partial Auto sweeps, small fixes
# ---------------------------------------------------------------------------

def test_supercell_above_server_atom_limit_is_rejected_in_dialog(app):
    page = app
    page.select_option("#sampleFileSelect", SAMPLE_CU)
    expect(page.locator("#confirmButton")).to_be_enabled()
    # 4 atoms x 6 x 6 x 6 = 864, above the server's MAX_RELAX_ATOMS (400)
    for axis in "abc":
        page.fill(f"#supercell-{axis}", "6")
    page.click("#confirmButton")
    expect(page.locator("#modalError")).to_contain_text("server limit")
    expect(page.locator("#cifModal")).to_be_visible()
    server_max = page.evaluate("fetch('/api/models').then(r => r.json()).then(d => d.max_relax_atoms)")
    assert page.evaluate("document.querySelector('#modal-max-atoms-limit').max") == str(server_max)


def test_auto_sweep_retries_after_429(app):
    page = app
    page.log.allow_errors = True  # the injected 429 is the point
    load_sample(page, SAMPLE_CU, supercell=(2, 1, 1))
    run_manual_op(page, "Cu0", "Ni")
    calls = []

    def limited_once(route):
        calls.append(1)
        if len(calls) == 1:
            route.fulfill(status=429, headers={"Retry-After": "1"},
                          json={"detail": "Rate limit exceeded"})
        else:
            route.continue_()

    page.route("**/api/evaluate-candidate-energies", limited_once)
    start_auto_sweep(page, "Cu", "Ni")
    page.wait_for_function("window.atomicOperations.length === 2", timeout=60_000)
    assert len(calls) >= 2
    assert not [d for d in page.log.dialogs if "failed" in d[1]]


def test_auto_sweep_with_failed_candidates_asks_before_applying(app):
    page = app
    load_sample(page, SAMPLE_CU, supercell=(2, 1, 1))
    run_manual_op(page, "Cu0", "Ni")

    def fail_first(route):
        response = route.fetch()
        body = response.json()
        body["results"][0]["error"] = "boom"
        body["results"][0]["energy"] = None
        route.fulfill(response=response, json=body)

    page.route("**/api/evaluate-candidate-energies", fail_first)
    start_auto_sweep(page, "Cu", "Ni")
    # The page fixture accepts every confirm(); the question itself must be asked.
    page.wait_for_function("window.atomicOperations.length === 2", timeout=60_000)
    asked = [m for kind, m in page.log.dialogs
             if kind == "confirm" and "candidate calculations failed" in m]
    assert asked, page.log.dialogs
    expect(page.locator("#calculation-report")).to_be_visible()


def test_insertable_elements_reload_after_failed_load(app):
    page = app
    page.log.allow_errors = True
    load_sample(page, SAMPLE_CU, supercell=(2, 1, 1))
    state = {"n": 0}

    def fail_once(route):
        state["n"] += 1
        if state["n"] == 1:
            route.fulfill(status=500, json={"detail": "boom"})
        else:
            route.continue_()

    page.route("**/api/get-insertable-elements", fail_once)
    page.click("#tab-insert")
    expect(page.locator("#insert-element")).to_contain_text("Failed to load elements")
    page.click("#tab-modify")
    page.click("#tab-insert")
    page.wait_for_function("document.querySelector('#insert-element').options.length > 1")


def test_fmax_input_is_clamped_to_server_minimum(app):
    page = app
    load_sample(page, SAMPLE_CU, supercell=(1, 1, 1))
    page.evaluate("document.querySelector('#fmax-input').value = '0.001'")
    with page.expect_request("**/api/relax") as info:
        analyze(page)
    assert json.loads(info.value.post_data)["fmax"] == 0.01
    assert page.evaluate("document.querySelector('#fmax-input').value") == "0.01"


def test_details_charts_handle_single_point_and_empty_data(app):
    page = app
    load_sample(page, SAMPLE_CU, supercell=(1, 1, 1))
    html = page.evaluate("""() => {
        drawEnergyChart({trajectory_data: {energies: [-3.5], force_magnitudes: [[0.1, 0.2]]}});
        drawForceChart({trajectory_data: {energies: [-3.5], force_magnitudes: [[0.1, 0.2]]}});
        return document.getElementById('energy-chart-stats').innerHTML +
               document.getElementById('force-chart-stats').innerHTML;
    }""")
    assert "NaN" not in html and "-3.5" in html
    # An empty data set must clear the previous statistics
    page.evaluate("drawEnergyChart({trajectory_data: {energies: []}}); "
                  "drawForceChart({trajectory_data: {force_magnitudes: []}});")
    assert page.evaluate("document.getElementById('energy-chart-stats').innerHTML") == ""
    assert page.evaluate("document.getElementById('force-chart-stats').innerHTML") == ""


def test_relax_null_energy_shows_na(app):
    page = app
    load_sample(page, SAMPLE_CU, supercell=(1, 1, 1))

    def mangle(route):
        response = route.fetch()
        body = response.json()
        body["final_prediction"]["energy_eV_per_atom"] = None
        body["final_prediction"]["total_energy_eV"] = None
        route.fulfill(response=response, json=body)

    page.route("**/api/relax", mangle)
    page.click("#analyze-btn", force=True)
    expect(page.locator("#analysisActions")).to_be_visible(timeout=60_000)
    expect(page.locator("#energy-summary")).to_contain_text("N/A")


def test_exports_tolerate_null_energy(app):
    page = app
    load_sample(page, SAMPLE_CU, supercell=(1, 1, 1))

    def mangle(route):
        response = route.fetch()
        body = response.json()
        body["final_prediction"]["energy_eV_per_atom"] = None
        body["final_prediction"]["total_energy_eV"] = None
        route.fulfill(response=response, json=body)

    page.route("**/api/relax", mangle)
    page.click("#analyze-btn", force=True)
    expect(page.locator("#analysisActions")).to_be_visible(timeout=60_000)
    text = page.evaluate("generateEnergyMagneticText()")
    assert "N/A" in text and "NaN" not in text
    # Headless Chromium denies clipboard writes; only a TypeError would matter.
    page.log.allow_errors = True
    page.evaluate("copyToSpreadsheet()")
    page.wait_for_timeout(500)
    assert not page.log.page_errors
    assert not [e for e in page.log.console_errors if "TypeError" in e]


def test_download_does_not_pass_off_unrelaxed_cif_when_relaxed_fetch_fails(app):
    page = app
    page.log.allow_errors = True
    load_sample(page, SAMPLE_CU, supercell=(2, 1, 1))
    analyze(page)
    page.route("**/api/generate-relaxed-structure-cif", lambda route: route.fulfill(
        status=500, json={"detail": "boom"}))
    try:
        with page.expect_download(timeout=3_000):
            page.click("#downloadCifBtn")
        downloaded = True
    except PlaywrightTimeoutError:
        downloaded = False
    assert not downloaded


def test_failed_reset_does_not_cancel_running_auto_sweep(app):
    page = app
    page.log.allow_errors = True
    load_sample(page, SAMPLE_CU, supercell=(2, 1, 1))
    run_manual_op(page, "Cu0", "Ni")
    hold = Hold(page, "**/api/evaluate-candidate-energies")
    start_auto_sweep(page, "Cu", "Ni")
    hold.wait_held(page)
    page.route("**/api/reset-session-structure", lambda route: route.fulfill(
        status=500, json={"detail": "boom"}))
    page.click("#reset-operations-btn")
    page.wait_for_timeout(500)
    expect(page.locator("#cancel-btn")).to_be_visible()
    hold.release()
    page.wait_for_function("window.atomicOperations.length === 2", timeout=60_000)


def test_new_cif_closes_relax_event_source_immediately(page, live_server):
    page.add_init_script("""
        (() => {
            const Orig = window.EventSource;
            window.__sources = [];
            window.EventSource = function (...a) {
                const es = new Orig(...a);
                window.__sources.push(es);
                return es;
            };
        })();""")
    open_app(page, live_server)
    load_sample(page, SAMPLE_CU, supercell=(1, 1, 1))
    hold = Hold(page, "**/api/relax")
    page.click("#analyze-btn", force=True)
    hold.wait_held(page)
    assert page.evaluate("window.__sources.some(s => s.readyState !== 2)")
    new_cif_dialog(page)
    pick_sample_and_confirm(page, SAMPLE_NI)
    assert page.evaluate("window.__sources.every(s => s.readyState === 2)")
    hold.release()
