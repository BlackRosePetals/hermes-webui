"""Opening the model picker must not auto-focus the search input.

Why: the picker is a CHOOSER. Users open it to pick a model, not to search. On
phones the auto-focus pops the on-screen keyboard the instant the chip is
tapped, so the model list sits behind the IME and the user has to dismiss the
keyboard before they can tap a row. Upstream suppressed this for the Settings
picker on coarse pointers only; the local policy is uniform: NO device
auto-focuses.

What replaces it: the dropdown container receives focus, so arrow-key /
Enter / Escape navigation still works without any text field being focused
(and therefore without any IME). Focus is still restored for a user who is
already typing in the search box when a re-render happens.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UI_JS = (REPO / "static" / "ui.js").read_text(encoding="utf-8")


def _function_body(src: str, name: str) -> str:
    match = re.search(rf"function\s+{re.escape(name)}\s*\([^)]*\)\s*\{{", src)
    assert match, f"{name} not found"
    start = match.end()
    depth = 1
    pos = start
    while pos < len(src) and depth:
        if src[pos] == "{":
            depth += 1
        elif src[pos] == "}":
            depth -= 1
        pos += 1
    assert depth == 0, f"{name} body did not terminate"
    return src[start : pos - 1]


def test_renderer_never_auto_focuses_the_search_input():
    body = _function_body(UI_JS, "renderModelDropdown")
    # Single source of truth: the policy is hard-coded in the renderer, so no
    # call site can re-enable the keyboard pop by passing an option.
    assert "const _autoFocusSearch=false;" in body
    assert "opts.autoFocusSearch" not in body
    # The one surviving focus call is the typing-preservation path, not a
    # first-open autofocus.
    assert "if(_autoFocusSearch||_hadFocus) _si.focus();" in body


def test_no_call_site_passes_the_autofocus_option():
    # The option is gone from the API; a caller still passing it is dead plumbing
    # that would silently stop working if the renderer ever read it again.
    assert "autoFocusSearch:" not in UI_JS
    assert "autoFocusSearch!==false" not in UI_JS


def test_dropdown_container_is_focusable_and_receives_focus():
    body = _function_body(UI_JS, "renderModelDropdown")
    # A plain <div> is not focusable — focus() on it is a silent no-op, which
    # would leave arrow-key navigation dead once the search box stops grabbing
    # focus. tabindex="-1" makes it programmatically focusable only.
    assert "dd.setAttribute('tabindex','-1')" in body

    open_body = _function_body(UI_JS, "toggleModelDropdown")
    assert "dd.focus({preventScroll:true})" in open_body
    # Focus must land AFTER the phone path reparents the menu onto <body>
    # (_positionModelDropdown), or the reparent detaches the focused node.
    assert open_body.index("_positionModelDropdown();") < open_body.index("dd.focus({preventScroll:true})")
    # And it must never target the search input on open.
    assert "querySelector('.model-search-input')" not in open_body
    assert ".model-search-input').focus" not in open_body


def test_settings_picker_also_focuses_container_not_input():
    body = _function_body(UI_JS, "openSettingsModelDropdown")
    assert "querySelector('.model-search-input')" not in body
    assert "dd.focus({preventScroll:true})" in body


def test_keyboard_navigation_survives_container_focus():
    body = _function_body(UI_JS, "renderModelDropdown")
    # Shared handler, wired to BOTH the search input (user tapped it) and the
    # container (opened without autofocus).
    assert "const _modelPickerKey=(e)=>" in body
    assert "_si.addEventListener('keydown',e=>{_modelPickerKey(e);});" in body
    assert "dd._modelPickerKey=_modelPickerKey;" in body
    # Bound once and dispatched through the refreshed slot: dd survives every
    # re-render (only its children are replaced), so re-binding would stack
    # duplicate listeners while a first-render closure would point at the
    # detached search input.
    assert "if(!dd._modelPickerKeyBound){" in body
    # Only handle keys when the CONTAINER itself holds focus, so events bubbling
    # from the inputs are never handled twice.
    assert "if(e.target!==dd) return;" in body


def test_typing_focus_is_preserved_across_rerenders():
    body = _function_body(UI_JS, "renderModelDropdown")
    # A late live-model fetch / preferences refresh replaces the input the user
    # is typing in. _hadFocus alone can't see it (it compares against the NEW
    # input after the old one detached), so a pre-teardown snapshot exists.
    assert "_searchHadFocusBefore" in body
    assert "_ddHadFocusBefore" in body
    # The snapshot must be taken in the render entry, before the children are
    # torn down (the real statement, not the comment that mentions it).
    snapshot_idx = body.index("const _searchHadFocusBefore=")
    teardown_idx = body.index("    dd.innerHTML='';")
    assert snapshot_idx < teardown_idx, "snapshot must precede the child teardown"
