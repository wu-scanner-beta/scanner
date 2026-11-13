/* 6.2 多核心測速：每種執行緒數開一個新的 Worker（numThreads 只能在第一次建立前設定） */
const HERE = new URL('./', self.location.href).href, AI = new URL('../lib/ai/', self.location.href).href;
self.onmessage = async (e) => {
  const { threads, u2, td, tw, th } = e.data;
  const post = (o) => self.postMessage(o);
  try {
    self.importScripts(HERE + 'ort.webgpu.min.js');
    ort.env.wasm.wasmPaths = HERE; // 要在這個資料夾裡（Service Worker 管得到），執行緒才會有隔離標頭
    ort.env.wasm.numThreads = threads;
    const opt = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
    const res = { threads };
    const go = async (name, model, data, shape, runs) => {
      let t0 = performance.now();
      const s = await ort.InferenceSession.create(AI + model, opt);
      res[name + 'Load'] = performance.now() - t0;
      const times = []; let out = null;
      for (let r = 0; r < runs; r++) {
        t0 = performance.now();
        const o = await s.run({ [s.inputNames[0]]: new ort.Tensor('float32', data.slice(), shape) });
        const d = o[s.outputNames[0]].data;
        times.push(performance.now() - t0);
        if (r === 0) out = Float32Array.from(d);
        post({ type: 'log', msg: `${threads} 條 ${name} 第 ${r + 1} 次 ${Math.round(times[r])} ms` });
      }
      res[name + 'First'] = times[0];
      const rest = times.slice(1); res[name + 'Avg'] = rest.reduce((a, b) => a + b, 0) / rest.length;
      res[name + 'Out'] = out;
      try { await s.release(); } catch (_) {}
    };
    await go('u2', 'u2netp.onnx', u2, [1, 3, 320, 320], 5);
    await go('td', 'teed.onnx', td, [1, 3, th, tw], 4);
    post({ type: 'done', res });
  } catch (err) { post({ type: 'error', msg: (err && err.message) || String(err) }); }
};
