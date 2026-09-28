// UI: webcam ao vivo / captura / foto enviada → MediaPipe → occlusion.js → canvas + JSON
import * as vision from "./vendor/vision_bundle.mjs";
import {
  buildRegions, sampleClasses, analyzeClasses, toView, jsonToHtml, glassesState, NUM_POINTS, CLS,
  accessoryRules, binaryState, capPoints, summarizeReport, RULES, GLASSES, DEFAULTS,
  panelHtml, panelSummaryHtml, confidence, detectedTags, tagsHtml,
} from "./occlusion.js";

const here = p => new URL(p, import.meta.url).href;
const ASSETS = {
  wasm: here("vendor/wasm"),
  face: here("models/face_landmarker.task"),
  oculos: here("models/oculos.onnx"),
  seg: [
    here("models/selfie_multiclass_256x256.tflite"),
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
  ],
};
const MAX_SIDE = 1280;
// GPU por padrão, com volta para CPU. ?cpu força CPU (usado nos testes em headless).
const DELEGATES = new URLSearchParams(location.search).has("cpu") ? ["CPU"] : ["GPU", "CPU"];
const OK = "#3ddc97", BAD = "#ff4d5e";
const CAP_OFF = "#ffd479", CAP_ON = "#ff9f1c";  // faixa do boné: ponto lido / ponto que contou como boné

const $ = id => document.getElementById(id);
const canvas = $("canvas"), ctx = canvas.getContext("2d");
const jsonEl = $("json"), toggleBtn = $("toggle"), refBtn = $("ref"), shot = $("shot"), fileInput = $("file");
const liveBtn = $("live"), captureBtn = $("capture"), video = $("video");
const notice = $("notice"), fpsEl = $("fps");
const tagsEl = $("tags");
const panel = $("panel"), panelHead = $("panelHead"), panelBody = $("panelBody"), panelSum = $("panelSum");
// limiares mostrados como marca nas barras do painel
const THR = { chapeu_ou_bone: RULES.capOn, mascara: RULES.maskOn, oculos: GLASSES.photoOn, regiao: DEFAULTS.regionOn };

const R = buildRegions(vision.FaceLandmarker);
let face = null, seg = null;
let photo = null, lm = null, cls = null, result = null, runId = 0, blink = null;
let showPoints = true, fullJson = false;
// mode: "idle" | "live" (webcam quadro a quadro) | "photo" (imagem parada)
let mode = "idle", runningMode = "IMAGE", mirror = false;
let stream = null, rafId = 0, lastVideoTime = -1;
let segBusy = false, frames = 0, masks = 0, fpsTimer = 0, faceDelegate = "", lastJson = "", lastPanel = "", lastSum = "", lastTags = "";
const ALL_FACE = new Uint8Array(NUM_POINTS).fill(CLS.FACE);
// Acessórios (boné, óculos, máscara): servidor Python opcional (server/app.py). ?api=URL/ aponta outro host.
const API = new URLSearchParams(location.search).get("api") || here("api/");
const ACC_EVERY_MS = 1000;  // ao vivo: no máximo uma consulta por segundo
let accOn = false, acc = null, accBusy = false, accLast = 0, accMs = 0;
// Óculos no próprio navegador (oculos-worker.js); tem prioridade sobre o "oculos" do servidor.
const GLASSES_EVERY_MS = 150;
let oc = null, glasses = null, glassesBusy = false, glassesLast = 0, glassesN = 0;
// Boné e máscara por regras sobre a máscara do segmentador (occlusion.js, accessoryRules).
let rules = null;
// Modo de teste (?teste): grava amostras por cenário e baixa um relatório JSON.
const TEST = new URLSearchParams(location.search).has("teste");
const REC_MS = 5000, REC_EVERY_MS = 200;
let samples = [], recUntil = 0, recLast = 0, recCenario = "", lastFps = 0;
let baseline = loadBaseline();

/* ---------- carregamento ---------- */
async function init() {
  const segReady = startSegWorker();
  const apiReady = checkApi();
  const glassesReady = startGlassesWorker();
  try {
    const fileset = await vision.FilesetResolver.forVisionTasks(ASSETS.wasm);
    face = await createFace(fileset);
  } catch (e) {
    return fail("Não consegui carregar o detector de rosto",
      "Abra a pasta por um servidor (GitHub Pages, Vercel ou npx serve), não direto do arquivo.", e);
  }

  seg = await segReady;
  await apiReady;
  oc = await glassesReady;
  if (!seg) {
    notice.textContent = "Segmentador indisponível: detectando só olhos fechados. Coloque selfie_multiclass_256x256.tflite na pasta models/.";
    notice.hidden = false;
  }

  $("spinner").hidden = true;
  $("guide").hidden = false;
  $("emptyTitle").textContent = "Tire uma foto do rosto";
  $("emptyText").textContent = "Use a webcam ao vivo, capture um quadro ou envie uma foto. A análise roda só no seu aparelho.";
  shot.classList.remove("busy");
  fileInput.disabled = false;
  const canCam = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  liveBtn.disabled = !canCam;
  if (!canCam) liveBtn.title = "Webcam exige HTTPS ou localhost";
}

// Os modelos alternam entre IMAGE (foto) e VIDEO (webcam, com rastreamento entre quadros).
async function setRunningMode(m) {
  if (runningMode === m) return;
  runningMode = m;
  await face.setOptions({ runningMode: m });
}

const timeout = (p, ms) => Promise.race([p, new Promise((_, rej) => setTimeout(() => rej(new Error("timeout")), ms))]);

async function createFace(fileset) {
  let err;
  for (const delegate of DELEGATES) {
    try {
      const f = await timeout(vision.FaceLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: ASSETS.face, delegate },
        runningMode: "IMAGE",
        numFaces: 1,
        outputFaceBlendshapes: true,  // eyeBlinkLeft/Right: olho fechado sem confundir com olho pequeno
      }), 20000);
      faceDelegate = delegate;
      return f;
    } catch (e) {
      console.warn("detector de rosto falhou com", delegate, e);
      err = e;
    }
  }
  throw err;
}

// Cliente de Web Worker de módulo: init → "ready" | "error"; depois pedidos com id → resposta com o mesmo id.
// Resolve com { info, request(msg, transfer) → Promise<resposta|null> } ou null se o worker não subir.
function startWorker(file, init, name) {
  return new Promise(resolve => {
    let w;
    try { w = new Worker(here(file), { type: "module" }); } catch (e) { console.warn(e); return resolve(null); }
    const pending = new Map();
    let nextId = 0;
    const flush = () => { for (const r of pending.values()) r(null); pending.clear(); };
    const give = v => { clearTimeout(timer); resolve(v); };
    const timer = setTimeout(() => { w.terminate(); give(null); }, 60000);
    w.onerror = e => { console.error(`worker ${name}:`, e.message || e); flush(); give(null); };
    w.onmessage = e => {
      const m = e.data;
      if (m.type === "ready") {
        give({
          info: m,
          request(msg, transfer) {
            const id = nextId++;
            return new Promise(r => { pending.set(id, r); w.postMessage({ ...msg, id }, transfer); });
          },
        });
      } else if (m.type === "error") {
        w.terminate();
        give(null);
      } else if (pending.has(m.id)) {
        pending.get(m.id)(m);
        pending.delete(m.id);
      }
    };
    w.postMessage({ type: "init", ...init });
  });
}

// Segmentador (seg-worker.js): { delegate, segment(src) → Promise<mask|null> } ou null.
async function startSegWorker() {
  const w = await startWorker("seg-worker.js", { wasm: ASSETS.wasm, models: ASSETS.seg, delegate: DELEGATES[0] }, "do segmentador");
  if (!w) return null;
  return {
    delegate: w.info.delegate,
    // createImageBitmap copia o quadro na hora da chamada: a máscara corresponde aos landmarks deste instante
    async segment(src) {
      let bitmap;
      try { bitmap = await createImageBitmap(src); } catch { return null; }
      const r = await w.request({ type: "frame", bitmap }, [bitmap]);
      return r && r.mask;
    },
  };
}

// Óculos (oculos-worker.js, ONNX): { classify(src, lm) → Promise<{grau, escuros}|null> } ou null.
async function startGlassesWorker() {
  const w = await startWorker("oculos-worker.js", { model: ASSETS.oculos }, "de óculos");
  if (!w) return null;
  return {
    async classify(src, frameLm) {
      const W = src.videoWidth || src.width, H = src.videoHeight || src.height;
      const [x, y, side] = cropSquare(frameLm, W, H);
      let bitmap;
      try {
        bitmap = await createImageBitmap(src, x, y, side, side, { resizeWidth: 256, resizeHeight: 256, resizeQuality: "high" });
      } catch { return null; }
      const r = await w.request({ type: "frame", bitmap }, [bitmap]);
      return r && r.scores;
    },
  };
}

// Recorte quadrado do rosto em pixels [x, y, lado]; igual a server/oculos.py (caixa_quadrada, MARGEM).
const GLASSES_MARGIN = 0.15;
function cropSquare(frameLm, W, H) {
  let x1 = 1, y1 = 1, x2 = 0, y2 = 0;
  for (const p of frameLm) { x1 = Math.min(x1, p.x); y1 = Math.min(y1, p.y); x2 = Math.max(x2, p.x); y2 = Math.max(y2, p.y); }
  const side = Math.max((x2 - x1) * W, (y2 - y1) * H) * (1 + 2 * GLASSES_MARGIN);
  const cx = (x1 + x2) / 2 * W, cy = (y1 + y2) / 2 * H;
  return [Math.round(cx - side / 2), Math.round(cy - side / 2), Math.round(side)];
}

function fail(title, text, err) {
  console.error(err);
  $("spinner").hidden = true;
  $("emptyTitle").textContent = title;
  $("emptyText").textContent = text;
  const el = $("emptyErr");
  el.textContent = String((err && (err.message || err)) || "");
  el.hidden = !el.textContent;
}

/* ---------- acessórios (servidor) ---------- */
async function checkApi() {
  try {
    const r = await fetch(API + "health");
    accOn = r.ok && (await r.json()).ok === true;
  } catch { accOn = false; }
}

// Envia o quadro (até 640px) + 5 pontos para o alinhamento do FaRL + caixa do rosto.
// O quadro é copiado de forma síncrona, então corresponde a frameLm.
async function fetchAccessories(src, frameLm) {
  const w0 = src.videoWidth || src.width, h0 = src.videoHeight || src.height;
  const k = Math.min(1, 640 / Math.max(w0, h0));
  const c = document.createElement("canvas");
  c.width = Math.round(w0 * k); c.height = Math.round(h0 * k);
  c.getContext("2d").drawImage(src, 0, 0, c.width, c.height);
  const W = c.width, H = c.height;
  const P = i => [frameLm[i].x * W, frameLm[i].y * H];
  const byX = (a, b) => a[0] - b[0];
  // íris (468, 473), ponta do nariz (1), cantos da boca (61, 291); ordem da esquerda p/ direita da imagem
  const points = [...[P(468), P(473)].sort(byX), P(1), ...[P(61), P(291)].sort(byX)];
  let x1 = 1, y1 = 1, x2 = 0, y2 = 0;
  for (const p of frameLm) { x1 = Math.min(x1, p.x); y1 = Math.min(y1, p.y); x2 = Math.max(x2, p.x); y2 = Math.max(y2, p.y); }
  const blob = await new Promise(r => c.toBlob(r, "image/jpeg", 0.9));
  const fd = new FormData();
  fd.append("image", blob, "frame.jpg");
  fd.append("points", JSON.stringify(points));
  fd.append("box", JSON.stringify([x1 * W, y1 * H, x2 * W, y2 * H]));
  const t = performance.now();
  const r = await fetch(API + "acessorios", { method: "POST", body: fd });
  if (!r.ok) throw new Error("servidor respondeu " + r.status);
  const out = await r.json();
  accMs = Math.round(performance.now() - t);
  return out;
}

function liveAccessories(frameLm) {
  accBusy = true;
  accLast = performance.now();
  fetchAccessories(video, frameLm)
    .then(r => { if (mode === "live") acc = r; })
    .catch(e => console.warn("acessórios:", e))
    .finally(() => { accBusy = false; });
}

/* ---------- foto ---------- */
fileInput.addEventListener("change", async () => {
  const file = fileInput.files && fileInput.files[0];
  fileInput.value = "";
  if (!file || !face) return;
  shot.classList.add("busy");
  try {
    const img = await loadImage(file);
    stopLive();
    await setRunningMode("IMAGE");
    photo = resize(img);
    mirror = false;
    mode = "photo";
    await run();
  } catch (e) {
    console.error(e);
    notice.textContent = "Não consegui ler essa imagem. Tente tirar outra foto.";
    notice.hidden = false;
  } finally {
    shot.classList.remove("busy");
  }
});

async function loadImage(file) {
  if ("createImageBitmap" in window) {
    try { return await createImageBitmap(file, { imageOrientation: "from-image" }); } catch {}
  }
  const url = URL.createObjectURL(file);
  try {
    const img = new Image();
    img.src = url;
    await img.decode();
    return img;
  } finally {
    URL.revokeObjectURL(url);
  }
}

function resize(img) {
  const w = img.videoWidth || img.width, h = img.videoHeight || img.height;
  const s = Math.min(1, MAX_SIDE / Math.max(w, h));
  const c = document.createElement("canvas");
  c.width = Math.round(w * s); c.height = Math.round(h * s);
  c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
  return c;
}

async function run() {
  const id = ++runId;
  const res = face.detect(photo);
  blink = blinkOf(res);
  lm = (res.faceLandmarks && res.faceLandmarks[0]) || null;
  const frameLm = lm;
  acc = null;
  glasses = null;
  rules = null;
  const glReq = oc && frameLm ? oc.classify(photo, frameLm) : null;
  const accReq = accOn && frameLm ? fetchAccessories(photo, frameLm).catch(e => { console.warn("acessórios:", e); return null; }) : null;
  const mask = frameLm && seg ? await seg.segment(photo) : null;
  if (id !== runId || mode !== "photo") return;  // chegou outra foto ou o ao vivo começou
  cls = frameLm && mask ? sampleClasses(frameLm, mask, R) : null;
  result = analyzeClasses(frameLm, cls, R, baseline, DEFAULTS, { blink });
  rules = ruleStates(accessoryRules(frameLm, cls, R, baseline), null);
  render(photo);
  if (glReq) {
    const g = await glReq;
    if (id !== runId || mode !== "photo") return;
    glasses = g && glassesState(g);
    showJson();
  }
  if (accReq) {
    const r = await accReq;
    if (id === runId && mode === "photo") { acc = r || { erro: true }; showJson(); }
  }
}

/* ---------- webcam ---------- */
async function startLive() {
  liveBtn.disabled = true;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    await setRunningMode("VIDEO");
    mode = "live";
    mirror = true;
    lastVideoTime = -1;
    cls = null;
    acc = null;
    glasses = null;
    rules = null;
    frames = masks = glassesN = 0;
    fpsTimer = setInterval(showFps, 1000);
    notice.hidden = true;
    liveBtn.textContent = "Parar";
    liveBtn.setAttribute("aria-pressed", "true");
    captureBtn.hidden = false;
    rafId = requestAnimationFrame(loop);
  } catch (e) {
    console.error(e);
    stopLive();
    notice.textContent = e && e.name === "NotAllowedError"
      ? "Permissão da câmera negada. Libere o acesso no navegador."
      : "Não consegui abrir a webcam: " + ((e && e.message) || e);
    notice.hidden = false;
  } finally {
    liveBtn.disabled = false;
  }
}

function stopLive() {
  cancelAnimationFrame(rafId);
  clearInterval(fpsTimer);
  fpsEl.hidden = true;
  // o último quadro exibido vira a foto atual (Landmarks/JSON continuam coerentes)
  if (mode === "live" && video.readyState >= 2) photo = resize(video);
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = null;
  video.srcObject = null;
  if (mode === "live") mode = photo ? "photo" : "idle";
  liveBtn.textContent = "Ao vivo";
  liveBtn.setAttribute("aria-pressed", "false");
  captureBtn.hidden = true;
}

function loop() {
  if (mode !== "live") return;
  rafId = requestAnimationFrame(loop);
  if (video.readyState < 2 || video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;
  const res = face.detectForVideo(video, performance.now());
  blink = blinkOf(res);
  lm = (res.faceLandmarks && res.faceLandmarks[0]) || null;
  // o segmentador é mais lento: roda em paralelo e as classes da última máscara pronta
  // são aplicadas aos landmarks atuais (olho fechado continua em tempo real)
  if (lm && seg && !segBusy) requestMask(lm);
  if (lm && accOn && !accBusy && performance.now() - accLast > ACC_EVERY_MS) liveAccessories(lm);
  if (lm && oc && !glassesBusy && performance.now() - glassesLast > GLASSES_EVERY_MS) liveGlasses(lm);
  result = analyzeClasses(lm, cls || (seg ? ALL_FACE : null), R, baseline, DEFAULTS, { blink });
  frames++;
  render(video);
  recordTick();
}

function requestMask(frameLm) {
  segBusy = true;
  seg.segment(video).then(mask => {
    if (mode === "live" && mask) {
      cls = sampleClasses(frameLm, mask, R);
      rules = ruleStates(accessoryRules(frameLm, cls, R, baseline), rules);
      masks++;
    }
  }).finally(() => { segBusy = false; });
}

// eyeBlink do MediaPipe por olho (Left/Right = olho da pessoa, conferido em server/: igual a EYE)
function blinkOf(res) {
  const c = res.faceBlendshapes && res.faceBlendshapes[0];
  if (!c) return null;
  const get = name => { const x = c.categories.find(q => q.categoryName === name); return x ? x.score : null; };
  return { olho_esquerdo: get("eyeBlinkLeft"), olho_direito: get("eyeBlinkRight") };
}

// scores das regras → { chapeu_ou_bone, mascara } com { presente, score, classes }; prev = estado anterior (ao vivo)
function ruleStates(r, prev) {
  if (!r) return null;
  const st = (k, thr) => ({ ...binaryState(r[k].score, thr, prev && prev[k]), classes: r[k].classes });
  return { chapeu_ou_bone: st("chapeu_ou_bone", RULES.capOn), mascara: st("mascara", RULES.maskOn) };
}

function liveGlasses(frameLm) {
  glassesBusy = true;
  glassesLast = performance.now();
  oc.classify(video, frameLm)
    .then(g => { if (mode === "live" && g) { glasses = glassesState(g, glasses); glassesN++; } })
    .finally(() => { glassesBusy = false; });
}

function showFps() {
  fpsEl.textContent = `${frames} fps · segmentação ${masks}/s · ${faceDelegate}/${seg ? seg.delegate || "?" : "—"}`
    + (oc ? ` · óculos ${glassesN}/s` : "")
    + (accOn ? ` · acessórios ${accMs}ms` : "");
  fpsEl.hidden = false;
  lastFps = frames;
  frames = masks = glassesN = 0;
}

// Congela o quadro atual e analisa como foto (modo IMAGE, resultado estável).
async function capture() {
  if (mode !== "live") return;
  const frame = resize(video);
  stopLive();
  await setRunningMode("IMAGE");
  photo = frame;
  mode = "photo";
  mirror = true;
  await run();
}

liveBtn.addEventListener("click", () => (mode === "live" ? stopLive() : startLive()));
captureBtn.addEventListener("click", capture);
document.addEventListener("visibilitychange", () => { if (document.hidden && mode === "live") stopLive(); });

/* ---------- referência ---------- */
function loadBaseline() {
  try {
    const a = JSON.parse(localStorage.getItem("oclusao_baseline") || "null");
    if (Array.isArray(a) && a.length === NUM_POINTS) return Float32Array.from(a);
  } catch {}
  return new Float32Array(NUM_POINTS);
}
refBtn.addEventListener("click", () => {
  if (!result || !result.nonSkin) return;
  baseline = Float32Array.from(result.nonSkin);
  try { localStorage.setItem("oclusao_baseline", JSON.stringify([...baseline])); } catch {}
  // ao vivo, o próximo quadro já usa a nova referência
  if (mode === "photo") {
    result = analyzeClasses(lm, cls, R, baseline, DEFAULTS, { blink });
    rules = ruleStates(accessoryRules(lm, cls, R, baseline), null);
    render(photo);
  }
  refBtn.textContent = "Referência salva";
  refBtn.classList.add("done");
});

/* ---------- desenho ---------- */
function render(src) {
  $("empty").hidden = true;
  canvas.hidden = false;
  panel.hidden = false;
  const w = src.videoWidth || src.width, h = src.videoHeight || src.height;
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  // câmera frontal é espelhada só na exibição; landmarks e "esquerdo/direito" seguem a imagem real
  ctx.save();
  if (mirror) { ctx.translate(w, 0); ctx.scale(-1, 1); }
  ctx.drawImage(src, 0, 0, w, h);

  const hasFace = !!(result.red && lm);
  toggleBtn.disabled = !hasFace;
  refBtn.disabled = !hasFace || !seg;
  if (!hasFace) {
    notice.textContent = "Nenhum rosto encontrado. Tente de frente e mais perto.";
    notice.hidden = false;
  } else if (seg) {
    notice.hidden = true;
  }

  if (hasFace && showPoints) {
    const W = canvas.width, H = canvas.height;
    const r = Math.max(1.5, Math.max(W, H) / 400);
    const ok = new Path2D(), bad = new Path2D();
    for (let i = 0; i < lm.length; i++) {
      const p = result.red[i] ? bad : ok;
      const x = lm[i].x * W, y = lm[i].y * H;
      p.moveTo(x + r, y);
      p.arc(x, y, r, 0, Math.PI * 2);
    }
    ctx.fillStyle = OK; ctx.fill(ok);
    ctx.fillStyle = BAD; ctx.fill(bad);
    // faixa acima da testa onde o boné é procurado (quadrados); laranja = contou como boné
    if (cls && cls.length > lm.length) {
      const on = new Path2D(), off = new Path2D();
      capPoints(lm).forEach((q, k) => {
        const c = cls[lm.length + k];
        (c === CLS.ACC || c === CLS.CLOTHES ? on : off).rect(q.x * W - r, q.y * H - r, 2 * r, 2 * r);
      });
      ctx.fillStyle = CAP_OFF; ctx.fill(off);
      ctx.fillStyle = CAP_ON; ctx.fill(on);
    }
  }
  ctx.restore();
  showJson();
}

// JSON (compacto ou completo, conforme o toque) e painel (sempre a partir da forma completa)
function showJson() {
  const st = result.state;
  const withAcc = (accOn || oc || seg) && st.rosto_detectado;
  const view = toView(st, R.keys, fullJson);
  if (withAcc) view.acessorios = accView(fullJson);
  const html = jsonToHtml(view);
  if (html !== lastJson) jsonEl.innerHTML = lastJson = html;
  const acc = withAcc ? accView(true) : null;
  const body = panelHtml(st, R.keys, acc, THR), sum = panelSummaryHtml(st, acc);
  if (body !== lastPanel) panelBody.innerHTML = lastPanel = body;
  if (sum !== lastSum) panelSum.innerHTML = lastSum = sum;
  // etiquetas grandes na tela: o que foi detectado
  const tags = tagsHtml(detectedTags(acc));
  if (tags !== lastTags) { tagsEl.innerHTML = lastTags = tags; tagsEl.hidden = !tags; }
}

// recolher/expandir; no celular começa recolhido. A preferência fica no navegador.
function setPanelOpen(open) {
  panel.classList.toggle("collapsed", !open);
  panelHead.setAttribute("aria-expanded", String(open));
}
let panelPref = null;
try { panelPref = localStorage.getItem("oclusao_painel"); } catch {}
setPanelOpen(panelPref ? panelPref === "aberto" : matchMedia("(min-width: 760px)").matches);
panelHead.addEventListener("click", () => {
  const open = panel.classList.contains("collapsed");
  setPanelOpen(open);
  try { localStorage.setItem("oclusao_painel", open ? "aberto" : "fechado"); } catch {}
});

// compacto: { chapeu_ou_bone: false, oculos: "grau" | "escuros" | false, mascara: false, servidor: { chapeu_ou_bone, mascara } }
// completo: cada item vira { presente, score } (óculos: { presente, tipo, grau, escuros })
// chapeu_ou_bone e mascara: regras sobre o segmentador, no navegador. oculos: ONNX no navegador (ou servidor, sem o worker).
// servidor: o que os modelos PyTorch dizem, para comparar (só com server/app.py no ar).
function accView(full = fullJson) {
  const r2 = x => Math.round(x * 100) / 100;
  const conf = (score, thr) => { const c = confidence(score, thr); return c === null ? null : r2(c); };
  const bin = (x, thr) => !x ? "analisando" : x.fora ? "fora do quadro"
    : full ? { presente: x.presente, score: r2(x.score), confianca: conf(x.score, thr) } : x.presente;
  const g = oc ? glasses : acc && !acc.erro && acc.oculos;
  const o = {};
  if (seg) o.chapeu_ou_bone = bin(rules && rules.chapeu_ou_bone, RULES.capOn);
  if (oc || accOn) {
    const top = g && Math.max(g.grau, g.escuros);
    o.oculos = !g ? "analisando"
      : full ? {
        presente: g.presente, tipo: g.tipo, grau: r2(g.grau), escuros: r2(g.escuros), confianca: conf(top, GLASSES.photoOn),
      }
      : g.tipo || false;
  }
  if (seg) o.mascara = bin(rules && rules.mascara, RULES.maskOn);
  if (accOn) {
    o.servidor = !acc ? "analisando" : acc.erro ? "erro"
      : { chapeu_ou_bone: bin(acc.chapeu_ou_bone, 0.5), mascara: bin(acc.mascara, 0.5) };
  }
  return o;
}

/* ---------- modo de teste: gravação por cenário e relatório ---------- */
const recBtn = $("rec"), dlBtn = $("dl"), cenarioSel = $("cenario");
if (TEST) $("testBar").hidden = false;

function takeSample() {
  if (!result) return;
  const st = result.state;
  const r3 = x => (typeof x === "number" ? Math.round(x * 1000) / 1000 : x);
  const s = { t: Date.now(), cenario: recCenario, modo: mode, rosto: !!st.rosto_detectado };
  if (mode === "live") s.fps = lastFps;
  if (st.rosto_detectado) {
    s.regioes = Object.fromEntries(R.keys.map(k => [k, { ...st[k] }]));
    if (glasses) {
      s.oculos = { grau: r3(glasses.grau), escuros: r3(glasses.escuros), tipo: glasses.tipo,
        confianca: r3(confidence(Math.max(glasses.grau, glasses.escuros), GLASSES.photoOn)),
        bruto: { grau: r3(glasses.bruto.grau), escuros: r3(glasses.bruto.escuros) } };
    }
    if (rules) {
      for (const k of ["chapeu_ou_bone", "mascara"]) {
        s[k] = { score: r3(rules[k].score), presente: rules[k].presente, classes: rules[k].classes,
          confianca: r3(confidence(rules[k].score, k === "mascara" ? RULES.maskOn : RULES.capOn)) };
      }
    }
    if (acc && !acc.erro) {
      s.servidor = {
        chapeu_ou_bone: acc.chapeu_ou_bone.score, chapeu_ou_bone_presente: acc.chapeu_ou_bone.presente,
        mascara: acc.mascara.score, mascara_presente: acc.mascara.presente,
        grau: acc.oculos && acc.oculos.grau, escuros: acc.oculos && acc.oculos.escuros, ms: acc.ms,
      };
    }
  }
  samples.push(s);
  dlBtn.disabled = false;
  dlBtn.textContent = `Baixar relatório (${samples.length})`;
}

// chamado a cada quadro do ao vivo
function recordTick() {
  if (!recUntil) return;
  const now = performance.now();
  if (now >= recUntil) return stopRec();
  recBtn.textContent = `Gravando… ${Math.ceil((recUntil - now) / 1000)} s`;
  if (now - recLast >= REC_EVERY_MS) { recLast = now; takeSample(); }
}

function stopRec() {
  recUntil = 0;
  recBtn.disabled = false;
  recBtn.textContent = "Gravar 5 s";
}

recBtn.addEventListener("click", () => {
  recCenario = cenarioSel.value;
  if (mode === "live") {
    recUntil = performance.now() + REC_MS;
    recLast = 0;
    recBtn.disabled = true;
  } else if (mode === "photo") {
    takeSample();  // foto: uma amostra
  } else {
    notice.textContent = "Abra o ao vivo ou envie uma foto antes de gravar.";
    notice.hidden = false;
  }
});

dlBtn.addEventListener("click", () => {
  const report = {
    versao: 1,
    criado: new Date().toISOString(),
    navegador: navigator.userAgent,
    tela: `${innerWidth}x${innerHeight}`,
    delegates: { rosto: faceDelegate, segmentador: seg ? seg.delegate : null, oculos: oc ? "wasm" : null, servidor: accOn },
    config: { RULES, GLASSES, DEFAULTS, GLASSES_MARGIN, referencia: baseline.some(Boolean) },
    resumo: summarizeReport(samples),
    amostras: samples,
  };
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(report, null, 1)], { type: "application/json" }));
  a.download = `oclusao-relatorio-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
});

toggleBtn.addEventListener("click", () => {
  showPoints = !showPoints;
  toggleBtn.setAttribute("aria-pressed", String(showPoints));
  if (mode === "photo") render(photo);
});
jsonEl.addEventListener("click", () => {
  fullJson = !fullJson;
  if (result) showJson();
});

/* ---------- gancho para testes automatizados ---------- */
const ready = init();
window.__oclusao = {
  ready,
  get state() { return result && result.state; },
  get red() { return result && result.red && Array.from(result.red); },
  get segmenter() { return !!seg; },
  // troca o segmentador por um falso síncrono { segment(img, cb) } (usado só nos testes ponta a ponta)
  setSegmenter(fake) {
    seg = {
      delegate: "fake",
      async segment(img) {
        let mask = null;
        fake.segment(img, r => {
          const m = r.categoryMask;
          if (m) mask = { data: m.getAsUint8Array().slice(), width: m.width, height: m.height };
        });
        return mask;
      },
    };
  },
  get landmarks() { return lm; },
  get mode() { return mode; },
  get acessorios() { return acc; },
  get oculos() { return glasses; },
  get regras() { return rules; },
};
