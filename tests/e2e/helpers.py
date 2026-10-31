"""Page-object style helpers shared by the E2E tests."""
import re
import threading
import time

from playwright.sync_api import Page, expect

SAMPLE_CU = "Metals/Cu.cif"
SAMPLE_NI = "Metals/Ni.cif"


def open_app(page: Page, base_url: str):
    page.goto(base_url)
    expect(page.locator("#cifModal")).to_be_visible()
    # Wait for the model list to load and the fake model to be selected.
    expect(page.locator("#mlip-model-select")).to_have_value("fake-emt")
    page.wait_for_function("document.querySelector('#sampleFileSelect').options.length > 1")


def pick_sample_and_confirm(page: Page, sample: str, supercell=(1, 1, 1)):
    """Choose a sample CIF in the (already open) start dialog and Confirm."""
    page.select_option("#sampleFileSelect", sample)
    expect(page.locator("#confirmButton")).to_be_enabled()
    for axis, n in zip("abc", supercell):
        page.fill(f"#supercell-{axis}", str(n))
    page.click("#confirmButton")
    expect(page.locator("#cifModal")).to_be_hidden()
    expect(page.locator("#execute-btn")).to_be_visible()
    expect(page.locator("#analyze-btn")).to_be_enabled()
    wait_viewer(page)


def load_sample(page: Page, sample: str = SAMPLE_CU, supercell=(1, 1, 1)):
    pick_sample_and_confirm(page, sample, supercell)


def wait_viewer(page: Page):
    page.wait_for_function(
        "window.viewer3D && window.currentLabels && window.currentLabels.length > 0")


def labels(page: Page):
    return page.evaluate("window.currentLabels.slice()")


def atom_count(page: Page) -> int:
    return page.evaluate("window.currentLabels.length")


def run_manual_op(page: Page, atom: str, action: str):
    """Manual-mode substitute/delete of one atom; ``action`` is DELETE or an element."""
    n_ops = page.evaluate("window.atomicOperations.length")
    n_atoms = atom_count(page)
    start_manual_op(page, atom, action)
    page.wait_for_function(f"window.atomicOperations.length === {n_ops + 1}")
    if action == "DELETE":
        page.wait_for_function(f"window.currentLabels.length === {n_atoms - 1}")
    expect(page.locator("#execute-btn")).not_to_have_class(re.compile("button-processing"))


def start_manual_op(page: Page, atom: str, action: str):
    """Select a manual operation and click Execute without waiting for it."""
    page.check("#manual-mode")
    page.select_option("#element-dropdown", atom)
    page.select_option("#substitute-element", action)
    expect(page.locator("#execute-btn")).to_be_enabled()
    page.click("#execute-btn")


def start_auto_sweep(page: Page, element: str, action: str):
    """Select an Auto-mode operation and click Execute without waiting for it."""
    page.check("#auto-mode")
    page.select_option("#element-type-dropdown", element)
    page.select_option("#substitute-element", action)
    expect(page.locator("#execute-btn")).to_be_enabled()
    page.click("#execute-btn")


def analyze(page: Page):
    """Click Analyze and wait for the relaxation result."""
    # force: the button pulses (needs-analysis animation) so is never "stable"
    page.click("#analyze-btn", force=True)
    expect(page.locator("#analysisActions")).to_be_visible(timeout=60_000)
    expect(page.locator("#analyze-btn")).to_be_enabled()
    expect(page.locator("#analyze-btn")).to_contain_text("Calculated")


def formula_text(page: Page) -> str:
    """Formula as displayed in the Crystal Information panel."""
    return page.locator("#main-crystal-info").inner_text()


def new_cif_dialog(page: Page):
    """Click New CIF (the confirm() is auto-accepted by the page fixture)."""
    page.click("#load-new-cif-btn")
    expect(page.locator("#cifModal")).to_be_visible()


def cif_atom_rows(text: str):
    """Rows of a CIF's atom_site loop: [label, symbol, x, y, z, ...]."""
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 5 and re.fullmatch(r"[A-Z][a-z]?\d*", parts[0]):
            try:
                [float(x) for x in parts[2:5]]
            except ValueError:
                continue
            rows.append(parts)
    return rows


def cif_symbols(text: str):
    return sorted(r[0] for r in cif_atom_rows(text))


class Hold:
    """Hold back matching requests until ``release()`` is called.

    Lets a test keep a request "in flight" deterministically (the fake backend
    answers in milliseconds). Held requests are forwarded unchanged on release.
    """

    def __init__(self, page: Page, pattern: str):
        self._lock = threading.Lock()
        self._held = []
        self._released = False
        self.seen = 0
        page.route(pattern, self._handler)

    def _handler(self, route):
        with self._lock:
            self.seen += 1
            if self._released:
                route.continue_()
            else:
                self._held.append(route)

    def release(self):
        with self._lock:
            self._released = True
            held, self._held = self._held, []
        for route in held:
            try:
                route.continue_()
            except Exception:
                pass  # request already aborted by the page

    def wait_held(self, page: Page, timeout_ms: int = 10_000):
        deadline = time.time() + timeout_ms / 1000
        while time.time() < deadline:
            with self._lock:
                if self._held:
                    return
            page.wait_for_timeout(50)
        raise AssertionError("request was never held back")
