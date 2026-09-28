// Classificador de óculos (grau / escuros) em ONNX num Web Worker.
// Modelo: models/oculos.onnx, exportado por server/export_onnx.py (glasses-detector, ShuffleNet v2 × 2).
// Mensagens: {type:"init", model}          → {type:"ready", delegate:"wasm"} | {type:"error"}
//            {type:"frame", id, bitmap}    → {type:"result", id, scores:{grau, escuros}|null}
// O bitmap já chega recortado e em 256×256 (app.js, cropSquare); os pixels vão crus, a
// normalização está dentro do modelo.
import * as ort from "./vendor/ort/ort.wasm.min.mjs";

ort.env.wasm.wasmPaths = new URL("./vendor/ort/", import.meta.url).href;
ort.env.wasm.numThreads = 1;  // várias threads exigem cross-origin isolation (COOP/COEP)

const S = 256;
let sess = null, ctx = null;

async function init({ model }) {
  sess = await ort.InferenceSession.create(model, { executionProviders: ["wasm"] });
  ctx = new OffscreenCanvas(S, S).getContext("2d", { willReadFrequently: true });
  postMessage({ type: "ready", delegate: "wasm" });
}

async function classify(id, bitmap) {
  let scores = null;
  try {
    ctx.clearRect(0, 0, S, S);
    ctx.drawImage(bitmap, 0, 0, S, S);
    const rgba = ctx.getImageData(0, 0, S, S).data;
    const rgb = new Uint8Array(S * S * 3);
    for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
      rgb[j] = rgba[i]; rgb[j + 1] = rgba[i + 1]; rgb[j + 2] = rgba[i + 2];
    }
    const out = await sess.run({ pixels: new ort.Tensor("uint8", rgb, [1, S, S, 3]) });
    const p = out.probs.data;
    scores = { grau: p[0], escuros: p[1] };
  } catch (e) {
    console.error("óculos:", e);
  } finally {
    bitmap.close();
  }
  postMessage({ type: "result", id, scores });
}

onmessage = e => {
  const m = e.data;
  if (m.type === "init") init(m).catch(err => { console.error(err); postMessage({ type: "error" }); });
  else if (m.type === "frame") classify(m.id, m.bitmap);
};
