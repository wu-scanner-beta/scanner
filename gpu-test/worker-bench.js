/* 背景執行緒模式測速：和 App 的 ai-worker.js 一樣在 Worker 裡用 importScripts 載入 ONNX Runtime */
const RUNS = 5, N = 320 * 320;
let loaded = null;
function load() {
  if (!loaded) {
    self.importScripts(new URL('ort.webgpu.min.js', self.location.href).href);
    ort.env.wasm.wasmPaths = new URL('./', self.location.href).href;
    ort.env.wasm.numThreads = 1;
    loaded = true;
  }
}
function toMask(out) {
  let lo = Infinity, hi = -Infinity; for (let i = 0; i < N; i++) { lo = Math.min(lo, out[i]); hi = Math.max(hi, out[i]); }
  const m = new Uint8Array(N), k = hi > lo ? 255 / (hi - lo) : 0;
  for (let i = 0; i < N; i++) m[i] = Math.round((out[i] - lo) * k);
  return m;
}
self.onmessage = async (e) => {
  const { type, ep, input, model } = e.data;
  const post = (o) => self.postMessage(o);
  try {
    if (type === 'check') {
      let gpu = false, name = '';
      try {
        if (self.navigator && self.navigator.gpu) {
          const ad = await self.navigator.gpu.requestAdapter();
          if (ad) { gpu = true; const inf = ad.info || {}; name = [inf.vendor, inf.architecture].filter(Boolean).join(' '); }
        }
      } catch (err) { name = '錯誤：' + err.message; }
      post({ type: 'check', gpu, name, hasNav: !!(self.navigator && self.navigator.gpu) });
      return;
    }
    load();
    const t0 = performance.now();
    const s = await ort.InferenceSession.create(model, { executionProviders: [ep], graphOptimizationLevel: 'all' });
    const loadMs = performance.now() - t0;
    const times = []; let first = null;
    for (let r = 0; r < RUNS; r++) {
      const a = performance.now();
      const o = await s.run({ [s.inputNames[0]]: new ort.Tensor('float32', input.slice(), [1, 3, 320, 320]) });
      const data = await o[s.outputNames[0]].getData();
      times.push(performance.now() - a);
      if (r === 0) first = data;
      post({ type: 'log', msg: `背景 ${ep} 第 ${r + 1} 次：${Math.round(times[r])} ms` });
    }
    try { await s.release(); } catch (_) {}
    const rest = times.slice(1);
    const mask = toMask(first);
    post({ type: 'done', load: loadMs, first: times[0], avg: rest.reduce((p, q) => p + q, 0) / rest.length, mask }, [mask.buffer]);
  } catch (err) {
    post({ type: 'error', msg: (err && err.message) || String(err) });
  }
};
