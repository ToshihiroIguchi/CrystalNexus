"""
Tests for 3D viewer atom click & hover interaction in Auto and Manual modes.
Verifies that the root page serves templates/index.html with:
1. Auto-to-manual switching logic upon atom click.
2. Defect protection guards against processing states and non-modify tab clicks.
3. Updated hover interactions enabling pointer cursor in Auto & Manual modes when idle.
4. Updated user-facing tooltips and user guide documentation.
"""


def test_root_serves_atom_click_switching_logic(client):
    """Verify templates/index.html served by / contains auto-to-manual switching logic on 3D atom click."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.text

    # Verify execution state guard is defined
    assert "function isOperationProcessing()" in html
    assert "button-processing" in html
    assert "currentAbortController" in html

    # Verify setClickable callback includes tab and processing guards
    assert "!isModifyActive || isOperationProcessing()" in html

    # Verify auto-to-manual switching code
    assert "if (manualMode && !manualMode.checked)" in html
    assert "manualMode.checked = true;" in html
    assert "if (autoMode) autoMode.checked = false;" in html
    assert "manualMode.dispatchEvent(new Event('change'));" in html

    # Verify atom dropdown assignment & dispatch
    assert "atomDropdown.value = label;" in html
    assert "atomDropdown.dispatchEvent(new Event('change'));" in html


def test_root_serves_hover_pointer_in_both_modes(client):
    """Verify hover interaction sets pointer cursor in Modify tab whenever not processing."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.text

    assert "isModifyActive && !isOperationProcessing()" in html
    assert "container.style.cursor = 'pointer';" in html
    assert "container.style.cursor = 'default';" in html


def test_root_serves_updated_auto_mode_tooltip(client):
    """Verify auto-mode radio label has updated title explaining 3D click to switch to manual."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.text

    expected_title = (
        'title="Select element type - automatically finds the most energy-stable '
        'atom of that type (or click any atom in 3D to switch to Manual)"'
    )
    assert expected_title in html


def test_user_guide_contains_atom_click_switching_note():
    """Verify static/user_guide.html mentions 3D atom clicking switches to Manual Mode."""
    from pathlib import Path
    guide_path = Path(__file__).resolve().parent.parent / "static" / "user_guide.html"
    assert guide_path.exists()
    content = guide_path.read_text(encoding="utf-8")

    assert "even while in Auto Mode" in content or "even while in Auto mode" in content or "Auto Mode" in content
    assert "switches to Manual Mode" in content or "switches to Manual mode" in content
