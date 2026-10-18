#pragma once

static const char * k_page_html = R"html(<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>tyger live</title>
<style>
:root { --bg:#f7f7f5; --fg:#1d1d1b; --mut:#6b6b66; --card:#fff; --line:#deded8; --acc:#2f6fdb; --hold:#c9861a; --speak:#2e9d57; --bc:#8a52c9; }
@media (prefers-color-scheme: dark) { :root { --bg:#151514; --fg:#ececea; --mut:#9a9a94; --card:#1f1f1d; --line:#33332f; --acc:#6d9cf0; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.45 system-ui, sans-serif; }
main { max-width:760px; margin:0 auto; padding:24px 16px; }
h1 { font-size:20px; margin:0 0 16px; }
.row { display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin-bottom:14px; }
button, select { font:inherit; padding:7px 14px; border-radius:7px; border:1px solid var(--line); background:var(--card); color:var(--fg); cursor:pointer; }
button.on { background:var(--acc); border-color:var(--acc); color:#fff; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px; margin-bottom:14px; }
#state { font-size:28px; font-weight:600; letter-spacing:.02em; }
.bars { display:grid; grid-template-columns:90px 1fr 48px; gap:6px 10px; align-items:center; margin-top:10px; font-size:13px; color:var(--mut); }
.bar { height:8px; background:var(--line); border-radius:4px; overflow:hidden; }
.bar i { display:block; height:100%; width:0; background:var(--acc); }
#partial { color:var(--mut); min-height:1.4em; }
#log { max-height:340px; overflow-y:auto; font-size:14px; }
#log div { padding:5px 0; border-bottom:1px solid var(--line); }
#log b { display:inline-block; min-width:110px; }
.mut { color:var(--mut); font-size:13px; }
</style>
</head>
<body>
<main>
<h1>tyger live</h1>
<div class="row">
<button id="go">Start mic</button>
<select id="enc"><option value="exact">exact encoder</option><option value="fast">fast encoder</option></select>
<button id="spk" title="tell the model the agent is talking, for barge in">Agent speaking: off</button>
<button id="rst">Reset turn</button>
<span id="st" class="mut">idle</span>
</div>
<div class="card">
<div id="state">idle</div>
<div class="bars">
<span>p_end</span><div class="bar"><i id="b_end"></i></div><span id="v_end">0</span>
<span>yield</span><div class="bar"><i id="b_yield"></i></div><span id="v_yield">0</span>
<span>backchannel</span><div class="bar"><i id="b_bc"></i></div><span id="v_bc">0</span>
<span>hazard</span><div class="bar"><i id="b_hz"></i></div><span id="v_hz">0</span>
</div>
</div>
<div class="card"><div class="mut">partial</div><div id="partial"></div></div>
<div class="card"><div id="log"></div></div>
</main>
<script>
const $ = (id) => document.getElementById(id);
let ws = null, ctx = null, node = null, src = null, speaking = false;
const colors = { HOLD:'var(--hold)', SPEAK:'var(--speak)', BACKCHANNEL:'var(--bc)', LISTEN:'var(--acc)' };
function bar(k, v) { $('b_' + k).style.width = (100 * Math.max(0, Math.min(1, v))) + '%'; $('v_' + k).textContent = v.toFixed(2); }
function log(kind, text) {
  const d = document.createElement('div');
  const b = document.createElement('b'); b.textContent = kind; d.appendChild(b);
  d.appendChild(document.createTextNode(text));
  $('log').prepend(d);
}
function onEvent(e) {
  if (e.kind === 'FRAME') {
    $('state').textContent = e.state; $('state').style.color = colors[e.state] || 'var(--fg)';
    bar('end', e.p_end); bar('yield', e.yield_prob); bar('bc', e.bc_prob); bar('hz', e.hazard);
  } else if (e.kind === 'PARTIAL') {
    $('partial').textContent = e.text;
  } else if (e.kind === 'ENDPOINT') {
    log('endpoint', e.t.toFixed(2) + ' s  ' + (e.transcript || '(no asr)') + '  [' + e.by + ', ' + e.turn_s.toFixed(1) + ' s]');
    $('partial').textContent = '';
  } else if (e.kind === 'BARGE') {
    log('barge in', e.t.toFixed(2) + ' s  yield ' + e.yield_prob.toFixed(2) + ' after ' + e.onset_s.toFixed(2) + ' s');
  } else if (e.kind === 'BACKCHANNEL') {
    log('backchannel', e.t.toFixed(2) + ' s  ' + (e.partial || ''));
  }
}
const worklet = `class P extends AudioWorkletProcessor {
  constructor() { super(); this.buf = new Int16Array(1600); this.n = 0; this.ratio = sampleRate / 16000; this.acc = 0; }
  process(inp) {
    const ch = inp[0] && inp[0][0]; if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      this.acc += 1; if (this.acc < this.ratio) continue; this.acc -= this.ratio;
      const v = Math.max(-1, Math.min(1, ch[i])); this.buf[this.n++] = v * 32767;
      if (this.n === this.buf.length) { this.port.postMessage(this.buf.buffer.slice(0)); this.n = 0; }
    }
    return true;
  }
}
registerProcessor('p', P);`;
async function start() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: false, autoGainControl: false } });
  try { ctx = new AudioContext({ sampleRate: 16000 }); } catch (_) { ctx = new AudioContext(); }
  await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([worklet], { type: 'application/javascript' })));
  src = ctx.createMediaStreamSource(stream);
  node = new AudioWorkletNode(ctx, 'p');
  src.connect(node);
  const url = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws?format=s16&channels=1&frames=1&encoder=' + $('enc').value;
  ws = new WebSocket(url);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => { $('st').textContent = 'connected'; };
  ws.onclose = () => { $('st').textContent = 'closed'; stop(); };
  ws.onmessage = (m) => {
    const j = JSON.parse(m.data);
    if (j.type === 'events') j.events.forEach(onEvent);
    else if (j.type === 'error') log('error', j.message);
  };
  node.port.onmessage = (m) => { if (ws && ws.readyState === 1) ws.send(m.data); };
  $('go').textContent = 'Stop'; $('go').classList.add('on');
}
function stop() {
  if (src) { src.mediaStream.getTracks().forEach((t) => t.stop()); src = null; }
  if (ctx) { ctx.close(); ctx = null; }
  if (ws) { const w = ws; ws = null; w.close(); }
  $('go').textContent = 'Start mic'; $('go').classList.remove('on');
}
$('go').onclick = () => { if (ws) stop(); else start().catch((e) => { log('error', String(e)); stop(); }); };
$('spk').onclick = () => {
  speaking = !speaking;
  $('spk').textContent = 'Agent speaking: ' + (speaking ? 'on' : 'off'); $('spk').classList.toggle('on', speaking);
  if (ws) ws.send(JSON.stringify({ type: 'mode', mode: speaking ? 'speaking' : 'listen' }));
};
$('rst').onclick = () => { if (ws) ws.send(JSON.stringify({ type: 'reset' })); };
</script>
</body>
</html>
)html";
