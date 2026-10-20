"""#8228 re-gate (2026-10-19): a pasted screenshot must not be hashed whole.

A vision turn keeps the pasted image as an ``image_url`` data URI on the user
message (``api/helpers.py:1593`` exempts it from redaction), and the browser
receives it — but ``msgContent()`` keeps text parts only, so the transcript
never draws that part. Hashing every character of the base64 therefore bought
no correctness and cost ~13x per render (35 ms -> 450 ms measured in Chromium
on 320 messages carrying 8 x 1.44 MB images).

The render surface only needs to notice that the image CHANGED, so the hash
covers the payload's length plus a short prefix/suffix instead of the whole
string. These tests drive the real ``_messageRenderCacheSignature`` extracted
from ``static/ui.js`` (no server, no browser) and pin both halves of that
contract: different images must not share a signature, and the cost must not
scale with the payload length.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

NODE = shutil.which("node")

# This module lives outside tests/ so it does not inherit the session-scoped
# autouse test-server fixture (this harness is pure Node, no server, no browser).
_REPO_ROOT = Path(__file__).resolve().parents[1]
UI_JS = (_REPO_ROOT / "static" / "ui.js").read_text(encoding="utf-8")

_EXTRACT = """
function extractFunc(name) {
  const start = src.search(new RegExp('function\\\\s+' + name + '\\\\s*\\\\('));
  if (start < 0) throw new Error(name + ' not found');
  let i = src.indexOf('{', start) + 1, depth = 1;
  while (depth > 0 && i < src.length) { if (src[i] === '{') depth++; else if (src[i] === '}') depth--; i++; }
  return src.slice(start, i);
}
"""

_BODY = """
const IMG_A = 'data:image/png;base64,' + 'A'.repeat(1440000);
const IMG_B = 'data:image/png;base64,' + 'B'.repeat(1440000);
const IMG_C = 'data:image/png;base64,' + 'A'.repeat(700000);
function imgState(uri) {
  return {
    messages: [
      {role: 'user', content: [{type: 'text', text: 'look'}, {type: 'image_url', image_url: {url: uri}}]},
      {role: 'assistant', content: 'a1'},
    ],
    toolCalls: [],
    session: {session_id: 'sid', message_count: 2, updated_at: 1, compression_anchor_visible_idx: 0, compression_anchor_summary: ''},
  };
}
globalThis.S = imgState(IMG_A);
const sigA = _messageRenderCacheSignature();
globalThis.S = imgState(IMG_B);
const sigB = _messageRenderCacheSignature();
globalThis.S = imgState(IMG_C);
const sigC = _messageRenderCacheSignature();
function timeHash(uri) {
  const t0 = process.hrtime.bigint();
  for (let i = 0; i < 50; i++) { globalThis.S = imgState(uri); _messageRenderCacheSignature(); }
  return Number(process.hrtime.bigint() - t0) / 1e6 / 50;
}
const msBig = timeHash(IMG_A);
const msSmall = timeHash('data:image/png;base64,' + 'A'.repeat(64));
console.log(JSON.stringify({
  deterministic: sigA === sigA,
  differDifferentImage: sigA !== sigB,
  differDifferentLength: sigA !== sigC,
  msBig, msSmall,
}));
"""


def _run() -> dict:
    assert NODE, "node is required for the render-cache signature harness"
    script = (
        "const window = {};\n"
        f"const src = {json.dumps(UI_JS)};\n"
        f"{_EXTRACT}\n"
        "eval(extractFunc('_messageRenderCacheSignature'));\n"
        "eval(extractFunc('_addMessageRenderHash'));\n"
        "eval(extractFunc('_addBoundedHash'));\n"
        "eval(extractFunc('_hashObjectInto'));\n"
        f"{_BODY}"
    )
    # Write the driver to a temp file: ui.js is ~500 KB, and passing it through
    # `node -e` exceeds ARG_MAX. The file also keeps the extracted functions
    # byte-identical to the ones on disk.
    import tempfile

    with tempfile.NamedTemporaryFile(
        "w", suffix=".js", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(script)
        script_path = fh.name
    try:
        # A clean environment: this module runs outside the tests/ conftest, and
        # a stray variable injected by the host process would abort subprocess.
        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k and "=" not in k and "\x00" not in v
        }
        result = subprocess.run(
            [NODE, script_path],
            text=True,
            capture_output=True,
            check=False,
            env=clean_env,
        )
    finally:
        os.unlink(script_path)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_pasted_image_data_uri_is_not_hashed_whole() -> None:
    out = _run()
    assert out["deterministic"], "identical state must yield an identical signature"
    assert out["differDifferentImage"], (
        "two different pasted screenshots must not share a render signature — "
        "the transcript would serve one session's cached HTML for the other"
    )
    assert out["differDifferentLength"], (
        "a different-length image must not share a render signature"
    )
    # Hashing the whole payload made the 1.44 MB image roughly 1000x the cost
    # of a short one; with the data-uri shortcut both are a constant number of
    # characters. The 20x headroom absorbs machine noise while still failing
    # loudly on a whole-payload regression.
    assert out["msBig"] < out["msSmall"] * 20 + 5, (
        f"hashing a 1.44 MB data URI must stay bounded "
        f"(big={out['msBig']:.3f} ms, small={out['msSmall']:.3f} ms)"
    )
