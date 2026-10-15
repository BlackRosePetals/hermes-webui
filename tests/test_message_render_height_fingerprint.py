"""Review-fix coverage for the #8228 supersede: full-surface render-height
fingerprint reuse, cross-message tool-result digests, and the binary-search
viewport anchor.

Each reuse scenario drives the REAL functions extracted from static/ui.js
through two complete _syncMessageVirtualHeightCache passes — the production
call sequence: the first pass seals the tool state, a measured height is
recorded, then S.messages is replaced wholesale (the loadSession(force:true)
refresh shape) and the second pass runs the actual reuse decision. Every
field the #8228 review called out gets its own mutation case:

  * reasoning_content-only change            -> must NOT reuse
  * a joined role:'tool' result change       -> must NOT reuse (digest)
  * a middle-only tool result edit           -> must NOT reuse (full hash)
  * _statusCard object change                -> must NOT reuse
  * _anchor_activity_scene change            -> must NOT reuse
  * partial-tool argument change             -> must NOT reuse
  * equal-length mid-content change          -> must NOT reuse
  * same-count tool-argument change          -> must NOT reuse

plus the positive case (byte-equivalent fresh objects reuse) and two
fail-closed pins (missing fingerprint helper, missing tool-digest helper).

Sensitivity note: on a tree without the fingerprint machinery (the pre-PR
baseline) the NOT-reuse cases pass trivially (identity never matches there,
so nothing is reused), while the byte-equivalent positive case fails. That
asymmetry is the test bite the reviewer asked us to state accurately — the
positive case is what red-flags a regression to identity-only matching.

The binary-search anchor gets its real-browser proof in
tests/browser_virtual_transcript_anchor.py; the mocked fast-path test from the
first round is replaced here by a rect-driven one that checks the search
against the reference predicate and the fallback instrumentation.
"""
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.resolve()
UI_JS_PATH = REPO_ROOT / "static" / "ui.js"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node not on PATH")


def _run_node(source: str) -> str:
    with tempfile.NamedTemporaryFile(
        "w", suffix=".cjs", encoding="utf-8", dir=REPO_ROOT, delete=False
    ) as script:
        script.write(source)
        script_path = Path(script.name)
    try:
        result = subprocess.run(
            [NODE, str(script_path)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
    finally:
        script_path.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return result.stdout.strip()


def _extract_func_script(js: str) -> str:
    return f"""
const src = {js!r};
function extractFunc(name) {{
  const re = new RegExp('function\\\\s+' + name + '\\\\s*\\\\(');
  const start = src.search(re);
  if (start < 0) throw new Error(name + ' not found');
  let i = src.indexOf('{{', start);
  let depth = 1; i++;
  while (depth > 0 && i < src.length) {{
    if (src[i] === '{{') depth++;
    else if (src[i] === '}}') depth--;
    i++;
  }}
  return src.slice(start, i);
}}
"""


# The helpers the matcher and sync functions need. Evaluated inside try/catch so
# the sensitivity run against the pre-PR ui.js (where these functions do not
# exist yet) degrades to the matcher's fail-closed path instead of crashing.
_FULL_HELPERS = """\
try {
  eval(extractFunc('_messageRenderHeightFingerprint'));
  eval(extractFunc('_addMessageRenderHash'));
  eval(extractFunc('_addBoundedHash'));
  eval(extractFunc('_hashObjectInto'));
  eval(extractFunc('_messageToolRenderDigest'));
  eval(extractFunc('_messageToolRenderIndex'));
} catch (e) {}
"""

# Deliberately WITHOUT the tool-digest helpers — pins the second fail-closed layer.
_FINGERPRINT_ONLY_HELPERS = """\
try {
  eval(extractFunc('_messageRenderHeightFingerprint'));
  eval(extractFunc('_addMessageRenderHash'));
  eval(extractFunc('_addBoundedHash'));
  eval(extractFunc('_hashObjectInto'));
} catch (e) {}
"""

_TWO_PASS_PROLOGUE = """\
const MESSAGE_VIRTUAL_DEFAULT_ROW_HEIGHT = 140;
let _messageVirtualHeightCache = [];
let _messageVirtualHeightCacheEntries = [];
let _messageVirtualHeightCacheLen = 0;
let _messageVirtualHeightCacheSrc = null;
let _messageVirtualEstimatedRowHeight = 140;
let _messageVirtualWindowKey = '';
function _clearMessageVirtualHeightCache() {
  _messageVirtualHeightCache = [];
  _messageVirtualHeightCacheEntries = [];
  _messageVirtualHeightCacheLen = 0;
  _messageVirtualHeightCacheSrc = null;
  _messageVirtualEstimatedRowHeight = MESSAGE_VIRTUAL_DEFAULT_ROW_HEIGHT;
  _messageVirtualWindowKey = '';
}
const window = {};
"""

_EVAL_MATCHER_BLOCK = """\
eval(extractFunc('_messageVirtualHeightEntryMatches'));
eval(extractFunc('_messageVirtualHeightPrefixEntryMatches'));
eval(extractFunc('_syncMessageVirtualHeightCache'));
"""

_TWO_PASS_BODY = """\
const round1Messages = __ROUND1__;
const round2Messages = __ROUND2__;
let S = {messages: round1Messages};
_syncMessageVirtualHeightCache([{rawIdx: 0, m: round1Messages[0]}]);
// Record a measured height for the first pass's entry.
_messageVirtualHeightCache = [300];
// Refresh: S.messages is replaced wholesale (the loadSession(force:true)
// shape) and the real second sync runs the reuse decision against the state
// sealed by the first pass.
S = {messages: round2Messages};
_syncMessageVirtualHeightCache([{rawIdx: 0, m: round2Messages[0]}]);
console.log(JSON.stringify({cache: _messageVirtualHeightCache}));
"""


def _run_two_pass(round1_js: str, round2_js: str, helpers: str = _FULL_HELPERS) -> dict:
    js = UI_JS_PATH.read_text(encoding="utf-8")
    source = (
        _extract_func_script(js)
        + _TWO_PASS_PROLOGUE
        + helpers
        + _EVAL_MATCHER_BLOCK
        + _TWO_PASS_BODY.replace("__ROUND1__", round1_js).replace("__ROUND2__", round2_js)
    )
    return json.loads(_run_node(source))


def _assert_reused(metrics: dict) -> None:
    assert metrics["cache"] == [300], "equivalent refresh must reuse the measured height"


def _assert_not_reused(metrics: dict) -> None:
    assert metrics["cache"][0] is None, (
        "a render-affecting change must NOT reuse the stale row height"
    )


# ── the positive cases: byte-equivalent fresh objects keep their height ─────

def test_reuse_for_byte_equivalent_fresh_objects():
    metrics = _run_two_pass(
        "[{id: 'm1', role: 'assistant', content: 'hello world'}]",
        "[{id: 'm1', role: 'assistant', content: 'hello world'}]",
    )
    _assert_reused(metrics)


def test_reuse_when_only_object_identity_changes_with_tool_chain():
    """The full loadSession(force:true) shape incl. a tool chain: fresh objects,
    byte-equivalent fields, unchanged tool result -> reuse."""
    body = (
        "[{id: 'a1', role: 'assistant', content: 'run it',"
        " tool_calls: [{id: 'tc1', type: 'function', function: {name: 'shell', arguments: '{}'}}]},"
        " {id: 't1', role: 'tool', tool_call_id: 'tc1', content: 'identical result'}]"
    )
    metrics = _run_two_pass(body, body)
    _assert_reused(metrics)


# ── message-surface field cases (review finding 1) ──────────────────────────

def test_no_reuse_on_equal_length_middle_content_change():
    """Equal-length content with a different MIDDLE must not reuse — a
    length+edges signature aliased this onto one height."""
    metrics = _run_two_pass(
        "[{id: 'm1', role: 'assistant', content: 'A'.repeat(200) + 'MIDDLE' + 'A'.repeat(200)}]",
        "[{id: 'm1', role: 'assistant', content: 'A'.repeat(200) + 'MUTATE' + 'A'.repeat(200)}]",
    )
    _assert_not_reused(metrics)


def test_no_reuse_on_reasoning_content_only_change():
    """reasoning_content is the field the row renderer reads FIRST; changing
    only it changed the rendered height (74.6px -> 300px in the review) while
    the old fingerprint kept a cached height."""
    metrics = _run_two_pass(
        "[{id: 'm1', role: 'assistant', content: 'answer', reasoning_content: 'brief thought'}]",
        "[{id: 'm1', role: 'assistant', content: 'answer',"
        " reasoning_content: 'brief thought, now dramatically longer so the row grows tall'}]",
    )
    _assert_not_reused(metrics)


def test_no_reuse_on_status_card_object_change():
    """_statusCard used to String() to '[object Object]', aliasing every card
    state onto one fingerprint."""
    metrics = _run_two_pass(
        "[{id: 'm1', role: 'assistant', content: '',"
        " _statusCard: {state: 'running', step: 1, label: 'fetching'}}]",
        "[{id: 'm1', role: 'assistant', content: '',"
        " _statusCard: {state: 'done', step: 2, label: 'fetching'}}]",
    )
    _assert_not_reused(metrics)


def test_no_reuse_on_activity_scene_change():
    """The activity-scene payload renders inside the row (worklog steps) and
    was not in the old field list at all."""
    metrics = _run_two_pass(
        "[{id: 'm1', role: 'assistant', content: '',"
        " _anchor_activity_scene: {version: 'activity_scene_v1', steps: [{id: 's1', text: 'one'}]}}]",
        "[{id: 'm1', role: 'assistant', content: '',"
        " _anchor_activity_scene: {version: 'activity_scene_v1',"
        " steps: [{id: 's1', text: 'one'}, {id: 's2', text: 'two'}]}}]",
    )
    _assert_not_reused(metrics)


def test_no_reuse_on_partial_tool_args_change():
    """Partial-tool argument payloads drive the live tool card; the old list
    only hashed id/name/snippet."""
    metrics = _run_two_pass(
        "[{id: 'm1', role: 'assistant', content: '',"
        " _partial_tool_calls: [{id: 'tc1', name: 'shell', args: {cmd: 'ls', flags: '-la'}}]}]",
        "[{id: 'm1', role: 'assistant', content: '',"
        " _partial_tool_calls: [{id: 'tc1', name: 'shell', args: {cmd: 'ls', flags: '-lh'}}]}]",
    )
    _assert_not_reused(metrics)


def test_no_reuse_on_tool_argument_change_constant_count():
    """Same tool-call COUNT, changed arguments: tool render state is a
    render-affecting field."""
    metrics = _run_two_pass(
        "[{id: 'm1', role: 'assistant', content: 'same text',"
        " tool_calls: [{id: 'tc1', name: 'shell', type: 'function',"
        " function: {name: 'shell', arguments: '{\"cmd\":\"ls\"}'}}]}]",
        "[{id: 'm1', role: 'assistant', content: 'same text',"
        " tool_calls: [{id: 'tc1', name: 'shell', type: 'function',"
        " function: {name: 'shell', arguments: '{\"cmd\":\"cd\"}'}}]}]",
    )
    _assert_not_reused(metrics)


# ── cross-message tool results (review finding 1, the digest) ───────────────

def test_no_reuse_on_tool_result_change():
    """The review's killer case: ONLY a role:'tool' result changed between
    refreshes. The result renders inside the assistant row but lives on a
    separate message joined by tool_call_id — the message fingerprint cannot
    see it, the tool digest must."""
    metrics = _run_two_pass(
        "[{id: 'a1', role: 'assistant', content: 'run it',"
        " tool_calls: [{id: 'tc1', type: 'function', function: {name: 'shell', arguments: '{}'}}]},"
        " {id: 't1', role: 'tool', tool_call_id: 'tc1', content: 'result-A'}]",
        "[{id: 'a1', role: 'assistant', content: 'run it',"
        " tool_calls: [{id: 'tc1', type: 'function', function: {name: 'shell', arguments: '{}'}}]},"
        " {id: 't1', role: 'tool', tool_call_id: 'tc1', content: 'result-B'}]",
    )
    _assert_not_reused(metrics)


def test_no_reuse_on_middle_only_tool_result_change():
    """A same-length middle-only edit of a large tool result must not alias —
    the digest hashes bodies in full, never clipped."""
    metrics = _run_two_pass(
        "[{id: 'a1', role: 'assistant', content: 'run it',"
        " tool_calls: [{id: 'tc1', type: 'function', function: {name: 'shell', arguments: '{}'}}]},"
        " {id: 't1', role: 'tool', tool_call_id: 'tc1',"
        " content: 'X'.repeat(3000) + 'MID' + 'X'.repeat(3000)}]",
        "[{id: 'a1', role: 'assistant', content: 'run it',"
        " tool_calls: [{id: 'tc1', type: 'function', function: {name: 'shell', arguments: '{}'}}]},"
        " {id: 't1', role: 'tool', tool_call_id: 'tc1',"
        " content: 'X'.repeat(3000) + 'MUT' + 'X'.repeat(3000)}]",
    )
    _assert_not_reused(metrics)


def test_no_reuse_on_settled_tool_call_metadata_change():
    """The settled S.toolCalls item renders into the row (snippet/args/done);
    a change there must invalidate too."""
    js = UI_JS_PATH.read_text(encoding="utf-8")
    body = _TWO_PASS_PROLOGUE + _FULL_HELPERS + _EVAL_MATCHER_BLOCK + """
const assistantRound1 = {id: 'a1', role: 'assistant', content: 'run it',
  tool_calls: [{id: 'tc1', type: 'function', function: {name: 'shell', arguments: '{}'}}]};
const assistantRound2 = {id: 'a1', role: 'assistant', content: 'run it',
  tool_calls: [{id: 'tc1', type: 'function', function: {name: 'shell', arguments: '{}'}}]};
const toolMsg = {id: 't1', role: 'tool', tool_call_id: 'tc1', content: 'stable result'};
let S = {messages: [assistantRound1, toolMsg], toolCalls: [{tid: 'tc1', name: 'shell', done: true, snippet: 'out-A'}]};
_syncMessageVirtualHeightCache([{rawIdx: 0, m: assistantRound1}]);
_messageVirtualHeightCache = [300];
S = {messages: [assistantRound2, toolMsg], toolCalls: [{tid: 'tc1', name: 'shell', done: true, snippet: 'out-B'}]};
_syncMessageVirtualHeightCache([{rawIdx: 0, m: assistantRound2}]);
console.log(JSON.stringify({cache: _messageVirtualHeightCache}));
"""
    metrics = json.loads(_run_node(_extract_func_script(js) + body))
    _assert_not_reused(metrics)


# ── fail-closed pins ────────────────────────────────────────────────────────

def test_matcher_fails_closed_when_fingerprint_helper_absent():
    """Extracted alone (upstream extractFunc pattern), the matcher must fail
    closed to identity-only matching instead of throwing or carrying a second,
    drift-prone field list."""
    js = UI_JS_PATH.read_text(encoding="utf-8")
    source = _extract_func_script(js) + """
const window = {};
eval(extractFunc('_messageVirtualHeightEntryMatches'));
// Deliberately NOT evaluating any fingerprint helper.
const first = {id: 'm1', role: 'assistant', content: 'hello world'};
const second = {id: 'm1', role: 'assistant', content: 'hello world'};
const result = _messageVirtualHeightEntryMatches({rawIdx: 0, m: first}, {rawIdx: 0, m: second});
console.log(JSON.stringify({result}));
"""
    metrics = json.loads(_run_node(source))
    assert metrics["result"] is False, "missing fingerprint helper must fail closed"


def test_matcher_fails_closed_when_tool_digest_helper_absent():
    """Byte-equivalent objects, full fingerprint helpers, but no tool-digest
    helper: the cross-message half of the check cannot run, so the matcher
    must fail closed rather than green-light a possibly-stale reuse."""
    js = UI_JS_PATH.read_text(encoding="utf-8")
    source = (
        _extract_func_script(js)
        + "const window = {};\n"
        + _FINGERPRINT_ONLY_HELPERS
        + "eval(extractFunc('_messageVirtualHeightEntryMatches'));\n"
        + """
const first = {id: 'm1', role: 'assistant', content: 'hello world'};
const second = {id: 'm1', role: 'assistant', content: 'hello world'};
const result = _messageVirtualHeightEntryMatches({rawIdx: 0, m: first}, {rawIdx: 0, m: second});
console.log(JSON.stringify({result}));
"""
    )
    metrics = json.loads(_run_node(source))
    assert metrics["result"] is False, "missing tool-digest helper must fail closed"


# ── the binary-search anchor (mocked sanity; the real proof is the browser) ─

_ANCHOR_NODE_BODY = r"""
const window = {};
function makeRows(rects){
  return rects.map((rect, i) => ({
    dataset: {msgIdx: String(i * 3), sessionMsgIdx: String(500 + i * 3), messageAnchorKey: 'k' + i},
    getBoundingClientRect(){ return rect; },
  }));
}
const container = {
  _rows: [],
  scrollTop: 0,
  scrollHeight: 5000,
  getBoundingClientRect(){ return {top: 0, bottom: 720}; },
  querySelector(sel){ return null; },
  querySelectorAll(sel){ return sel === '[data-msg-idx]' ? container._rows : []; },
};
function $(id){ return id === 'messages' ? container : null; }
function _messageSessionIndexForRawIdx(rawIdx){ return rawIdx + 1000; }
eval(extractFunc('_captureMessageViewportAnchor'));

function referenceHit(rects){
  for (let i = 0; i < rects.length; i++){
    if (rects[i].bottom > 0 + 1) return i;
  }
  return -1;
}
const scenarios = [
  [{top: -120, bottom: -40}, {top: -40, bottom: 40}, {top: 40, bottom: 160}, {top: 160, bottom: 340}],
  [{top: -400, bottom: -200}, {top: -200, bottom: -100}, {top: -100, bottom: 20}, {top: 20, bottom: 300}],
  [{top: -800, bottom: -700}, {top: -700, bottom: -650}, {top: -650, bottom: -640}, {top: -640, bottom: 100}],
  [{top: 5, bottom: 50}, {top: 50, bottom: 90}],
  [{top: -50, bottom: 0.5}, {top: 0.5, bottom: 10}],
];
const fastResults = scenarios.map(rects => {
  container._rows = makeRows(rects);
  const anchor = _captureMessageViewportAnchor();
  const refIdx = referenceHit(rects);
  return {
    fastIdx: anchor ? anchor.rawIdx : null,
    refIdx: refIdx >= 0 ? refIdx * 3 : null,
    sessionIdx: anchor ? anchor.sessionIdx : null,
  };
});
// Fallback comparison at scenario 1 — the per-row scan must find the same row
// and the fallback counter must bite.
window.__vmFastAnchor = false;
window.__vmAnchorFallbackRowReads = 0;
container._rows = makeRows(scenarios[1]);
const slowAnchor = _captureMessageViewportAnchor();
const slowReads = window.__vmAnchorFallbackRowReads;
const slowRefIndex = referenceHit(scenarios[1]);
// Empty row set: falls through and returns null without throwing.
window.__vmFastAnchor = true;
container._rows = [];
const emptyAnchor = _captureMessageViewportAnchor();
console.log(JSON.stringify({
  fastResults,
  slowIdx: slowAnchor ? slowAnchor.rawIdx : null,
  slowRefIdx: slowRefIndex >= 0 ? slowRefIndex * 3 : null,
  slowReads,
  emptyOk: emptyAnchor === null,
}));
"""


def _run_anchor_node() -> dict:
    js = UI_JS_PATH.read_text(encoding="utf-8")
    source = _extract_func_script(js) + _ANCHOR_NODE_BODY
    return json.loads(_run_node(source))


def test_binary_search_anchor_matches_reference_predicate():
    """The fast anchor must return the same row as the first-row-with-bottom-
    past-the-top-edge predicate the per-row scan implements, across covering
    positions (row straddling the edge, row fully above, edge cases)."""
    out = _run_anchor_node()
    for case in out["fastResults"]:
        assert case["fastIdx"] == case["refIdx"], (
            f"fast anchor picked row {case['fastIdx']} but the reference predicate "
            f"picks {case['refIdx']}: {case}"
        )
        assert case["sessionIdx"] is not None, "fast anchor must carry a session index"


def test_binary_search_anchor_falls_back_safely():
    """Fast path off -> the per-row scan returns the same anchor and the
    __vmAnchorFallbackRowReads instrumentation bites. An empty row set must
    fall through to null without throwing."""
    out = _run_anchor_node()
    assert out["slowIdx"] == out["slowRefIdx"], (
        "fallback scan must find the same first-covering row as the reference predicate"
    )
    assert out["slowReads"] >= 1, "fallback loop must be instrumented"
    assert out["emptyOk"] is True, "empty row set must return null, not throw"
