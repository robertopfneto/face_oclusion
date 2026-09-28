// Segmentador em um Web Worker: roda em paralelo aos landmarks, sem travar a tela.
// Mensagens: {type:"init", wasm, models, delegate} → {type:"ready", delegate} | {type:"error"}
//            {type:"frame", id, bitmap}             → {type:"result", id, mask|null}
import * as vision from "./vendor/vision_bundle.mjs";

// Em worker de módulo não há importScripts; o MediaPipe usa self.import para carregar o
// loader do wasm. O loader é um script clássico (define ModuleFactory global), então roda
// com eval indireto no escopo global.
self.import = async url => { (0, eval)(await (await fetch(url)).text()); };

let seg = null, lastTs = 0;

async function create(fileset, url, delegate) {
  return vision.ImageSegmenter.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: url, delegate },
    runningMode: "VIDEO",
    outputCategoryMask: true,
    outputConfidenceMasks: false,
  });
}

async function init({ wasm, models, delegate }) {
  const fileset = await vision.FilesetResolver.forVisionTasks(wasm);
  const delegates = delegate === "CPU" ? ["CPU"] : ["GPU", "CPU"];
  for (const url of models) {
    for (const d of delegates) {
      try {
        seg = await create(fileset, url, d);
        return postMessage({ type: "ready", delegate: d });
      } catch (e) {
        console.warn("segmentador falhou em", url, d, e);
      }
    }
  }
  postMessage({ type: "error" });
}

function segment(id, bitmap) {
  let mask = null;
  // timestamps precisam ser crescentes no modo VIDEO
  lastTs = Math.max(lastTs + 1, performance.now());
  try {
    seg.segmentForVideo(bitmap, lastTs, r => {
      const m = r.categoryMask;
      if (m) mask = { data: m.getAsUint8Array().slice(), width: m.width, height: m.height };
    });
  } finally {
    bitmap.close();
  }
  postMessage({ type: "result", id, mask }, mask ? [mask.data.buffer] : []);
}

onmessage = e => {
  const m = e.data;
  if (m.type === "init") init(m).catch(err => { console.error(err); postMessage({ type: "error" }); });
  else if (m.type === "frame") segment(m.id, m.bitmap);
};
