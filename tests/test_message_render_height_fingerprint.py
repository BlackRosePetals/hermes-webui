"""Review-fix coverage for the #7695 supersede: render-height fingerprint reuse
and the instrumented anchor fallback.

Pins the three reviewer cases — byte-equivalent fresh objects keep their measured
height, an equal-length middle-only content change must NOT reuse, a constant
tool-count state change must NOT reuse — plus the matcher's fail-closed behavior
when extracted without its shared fingerprint helper, and the fast anchor path
returning without a single fallback row read (with the counter biting when the
correctness gate fails and the per-row loop actually runs).
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

# The matcher's identity-change path needs the shared fingerprint helper and its
# module-level deps. On trees where the helper does not exist (sensitivity run
# against the pre-fix ui.js), extraction throws and the evals are skipped so the
# matcher still runs standalone — test 1 then acts as a pure no-regression pin.
_HELPER_EVAL_BLOCK = """\
try {
  eval(extractFunc('_messageRenderHeightFingerprint'));
  eval(extractFunc('_addMessageRenderHash'));
  eval(extractFunc('msgContent'));
  eval(extractFunc('_messageHasReasoningPayload'));
  eval(extractFunc('_addBoundedHash'));
  eval(extractFunc('_hashObjectInto'));
} catch (e) {}
"""


def test_height_reuse_for_byte_equivalent_fresh_objects():
    """Reviewer case (a): two fresh objects with byte-equivalent fields reuse the
    measured row height — through the matcher AND through _syncMessageVirtualHeightCache."""
    js = UI_JS_PATH.read_text(encoding="utf-8")
    source = _extract_func_script(js) + """
const window = {};
""" + _HELPER_EVAL_BLOCK + """
eval(extractFunc('_messageVirtualHeightEntryMatches'));
eval(extractFunc('_messageVirtualHeightPrefixEntryMatches'));
eval(extractFunc('_syncMessageVirtualHeightCache'));
const MESSAGE_VIRTUAL_DEFAULT_ROW_HEIGHT = 140;
let _messageVirtualHeightCache = [180];
let _messageVirtualHeightCacheEntries = [];
let _messageVirtualHeightCacheLen = 1;
let _messageVirtualHeightCacheSrc = null;
let _messageVirtualEstimatedRowHeight = 140;
let _messageVirtualWindowKey = 'stale';
function _clearMessageVirtualHeightCache() {
  _messageVirtualHeightCache = [];
  _messageVirtualHeightCacheEntries = [];
  _messageVirtualHeightCacheLen = 0;
  _messageVirtualHeightCacheSrc = null;
  _messageVirtualEstimatedRowHeight = MESSAGE_VIRTUAL_DEFAULT_ROW_HEIGHT;
  _messageVirtualWindowKey = '';
}
const first = {
  id: 'm1', role: 'assistant', content: 'hello world',
  tool_calls: [{id: 'tc1', name: 'shell', type: 'function', function: {name: 'shell', arguments: '{"cmd":"ls"}'}}],
  reasoning: 'thinking about it',
};
// Fresh object, byte-for-byte equivalent fields: the loadSession(force:true)
// refresh shape — identity never matches, every render field does.
const second = {
  id: 'm1', role: 'assistant', content: 'hello world',
  tool_calls: [{id: 'tc1', name: 'shell', type: 'function', function: {name: 'shell', arguments: '{"cmd":"ls"}'}}],
  reasoning: 'thinking about it',
};
let S = {messages: [first]};
_messageVirtualHeightCacheEntries = [{rawIdx: 0, m: first}];
_messageVirtualHeightCacheSrc = S.messages;
S = {messages: [second]};
const matched = _messageVirtualHeightEntryMatches({rawIdx: 0, m: first}, {rawIdx: 0, m: second});
_syncMessageVirtualHeightCache([{rawIdx: 0, m: second}]);
console.log(JSON.stringify({matched, cache: _messageVirtualHeightCache}));
"""
    metrics = json.loads(_run_node(source))
    assert metrics["matched"] is True, "byte-equivalent fresh objects must reuse"
    assert metrics["cache"] == [180], "measured height must survive the refresh"


def test_no_height_reuse_on_equal_length_middle_change():
    """Reviewer case (b): equal-length content with a different MIDDLE must not
    reuse — the old length+edges signature aliased this onto one height."""
    js = UI_JS_PATH.read_text(encoding="utf-8")
    source = _extract_func_script(js) + """
const window = {};
""" + _HELPER_EVAL_BLOCK + """
eval(extractFunc('_messageVirtualHeightEntryMatches'));
const pad = 'A'.repeat(200);
const first = {id: 'm1', role: 'assistant', content: pad + 'MIDDLE' + pad};
// Same total length, same head and tail, different middle — byte-length and
// edges are identical, only the full fingerprint can tell them apart.
const second = {id: 'm1', role: 'assistant', content: pad + 'MUTATE' + pad};
const matched = _messageVirtualHeightEntryMatches({rawIdx: 0, m: first}, {rawIdx: 0, m: second});
console.log(JSON.stringify({matched}));
"""
    metrics = json.loads(_run_node(source))
    assert metrics["matched"] is False, "equal-length middle change must NOT reuse"


def test_no_height_reuse_on_tool_state_change_constant_count():
    """Reviewer case (c): same tool-call COUNT but changed arguments must not
    reuse — tool render state is a render-affecting field."""
    js = UI_JS_PATH.read_text(encoding="utf-8")
    source = _extract_func_script(js) + """
const window = {};
""" + _HELPER_EVAL_BLOCK + """
eval(extractFunc('_messageVirtualHeightEntryMatches'));
const first = {id: 'm1', role: 'assistant', content: 'same text', tool_calls: [{id: 'tc1', name: 'shell', type: 'function', function: {name: 'shell', arguments: '{"cmd":"ls"}'}}]};
// Tool count unchanged, arguments changed: the old signature (string length +
// edges + tool count) reused this height.
const second = {id: 'm1', role: 'assistant', content: 'same text', tool_calls: [{id: 'tc1', name: 'shell', type: 'function', function: {name: 'shell', arguments: '{"cmd":"pwd"}'}}]};
const matched = _messageVirtualHeightEntryMatches({rawIdx: 0, m: first}, {rawIdx: 0, m: second});
console.log(JSON.stringify({matched}));
"""
    metrics = json.loads(_run_node(source))
    assert metrics["matched"] is False, "tool render-state change must NOT reuse"


def test_matcher_fails_closed_when_fingerprint_helper_absent():
    """Extracted alone (upstream extractFunc pattern), the matcher must fail
    closed to identity-only matching instead of throwing or carrying a second,
    drift-prone field list."""
    js = UI_JS_PATH.read_text(encoding="utf-8")
    source = _extract_func_script(js) + """
const window = {};
eval(extractFunc('_messageVirtualHeightEntryMatches'));
// Deliberately NOT evaluating _messageRenderHeightFingerprint / _addMessageRenderHash.
const first = {id: 'm1', role: 'assistant', content: 'hello world'};
const second = {id: 'm1', role: 'assistant', content: 'hello world'};
const result = _messageVirtualHeightEntryMatches({rawIdx: 0, m: first}, {rawIdx: 0, m: second});
console.log(JSON.stringify({result}));
"""
    metrics = json.loads(_run_node(source))
    assert metrics["result"] is False, "missing fingerprint helper must fail closed"


def test_fast_anchor_returns_without_fallback_reads():
    """The fast anchor path must return with a finite sessionIdx and ZERO
    fallback row reads when the cache is complete and the correctness gate
    passes — and the __vmAnchorFallbackRowReads counter must bite (>=1) when
    the gate fails and the per-row loop actually runs."""
    js = UI_JS_PATH.read_text(encoding="utf-8")
    source = _extract_func_script(js) + """
const window = {};
let gateFail = false;
const row0 = {
  dataset: {msgIdx: '0', sessionMsgIdx: '7', messageAnchorKey: 'k0'},
  getBoundingClientRect(){ return gateFail ? {top: 200, bottom: 300} : {top: -10, bottom: 90}; },
};
const container = {
  scrollTop: 50,
  scrollHeight: 5000,
  getBoundingClientRect(){ return {top: 0, bottom: 720}; },
  querySelector(sel){
    if (sel === '[data-msg-idx="0"]') return row0;
    if (sel === '[data-virtual-spacer="before"]') return null;
    return null;
  },
  querySelectorAll(sel){ return sel === '[data-msg-idx]' ? [row0] : []; },
};
function $(id){ return id === 'messages' ? container : null; }
function _getVisibleMessagesWithIdx(){ return [{rawIdx: 0, m: {id: 'm1', role: 'assistant', content: 'hello'}}]; }
function _syncMessageVirtualHeightCache(){}
let _messageVirtualHeightCache = [100];
function _messageVirtualDefaultHeightForRole(){ return 140; }
function _messageVirtualRoleForEntry(){ return 'assistant'; }
function _messageSessionIndexForRawIdx(){ return 7; }
eval(extractFunc('_captureMessageViewportAnchor'));
// Scenario 1: cache complete and the gate passes — fast path returns the
// anchor without a single fallback row read.
const s1 = _captureMessageViewportAnchor();
const s1FallbackReads = window.__vmAnchorFallbackRowReads === undefined ? 0 : window.__vmAnchorFallbackRowReads;
// Scenario 2: break the gate (row rect no longer covers the container top) —
// the fast path must fall through to the per-row loop, counter bites.
window.__vmAnchorFallbackRowReads = 0;
gateFail = true;
const s2 = _captureMessageViewportAnchor();
const s2FallbackReads = window.__vmAnchorFallbackRowReads === undefined ? 0 : window.__vmAnchorFallbackRowReads;
console.log(JSON.stringify({s1, s1FallbackReads, s2, s2FallbackReads}));
"""
    metrics = json.loads(_run_node(source))
    assert metrics["s1"] is not None, "fast path must return an anchor"
    assert metrics["s1"]["sessionIdx"] == 7, "fast path must return a finite sessionIdx"
    assert metrics["s1FallbackReads"] == 0, "complete cache must not read any fallback row"
    assert metrics["s2"] is not None, "gate failure must still produce the anchor via fallback"
    assert metrics["s2FallbackReads"] >= 1, "fallback loop must be instrumented"
