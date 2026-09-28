// Lógica pura de oclusão facial. Sem DOM e sem MediaPipe: recebe landmarks
// normalizados (0..1) e, opcionalmente, a máscara de classes do segmentador.
// Tudo aqui é coberto por tests/occlusion.test.mjs.

/** Classes do modelo selfie_multiclass_256x256 */
export const CLS = { BG: 0, HAIR: 1, BODY: 2, FACE: 3, CLOTHES: 4, ACC: 5 };
export const MOTIVO = { 0: "fundo", 1: "cabelo", 2: "mao_ou_pele", 4: "roupa", 5: "acessorio" };

export const DEFAULTS = {
  regionOn: 0.40,   // fração de pontos fora da pele (acima da referência) para marcar a região
  pointOn: 0.5,     // ponto isolado, fora das regiões
  earClosed: 0.10,  // razão de abertura (EAR): abaixo disso = fechado pela geometria
  earOpen: 0.20,    // EAR acima disso = aberto. Olho pequeno/semiaberto costuma ficar entre 0,15 e 0,25
  blinkOpen: 0.35,  // blendshape eyeBlink do MediaPipe: abaixo disso = aberto
  blinkClosed: 0.70,// eyeBlink acima disso = fechado
  eyeRing: 0.6,     // olhos: amostra um anel ao redor do olho (lente e armação)
  contourPull: 0.15 // contorno: amostra um pouco para dentro do rosto
};

export const NUM_POINTS = 478;
export const NOSE_TIP = 1;

// Pontos para a razão de abertura. Esquerdo/direito do ponto de vista da pessoa.
export const EYE = {
  olho_esquerdo: { up: 386, down: 374, a: 362, b: 263 },
  olho_direito:  { up: 159, down: 145, a: 33,  b: 133 },
};

const NOSE = [1, 2, 4, 5, 6, 19, 45, 48, 64, 94, 98, 168, 195, 197, 275, 278, 294, 327];
const FOREHEAD = [9, 10, 67, 69, 104, 108, 109, 151, 297, 299, 333, 337, 338];

const uniq = conns => [...new Set(conns.flatMap(c => [c.start, c.end]))];

/**
 * Monta as regiões a partir das listas de conexões do FaceLandmarker.
 * @param {object} F classe FaceLandmarker (ou objeto com as mesmas constantes)
 */
export function buildRegions(F) {
  const regions = {
    olho_esquerdo: uniq(F.FACE_LANDMARKS_LEFT_EYE),
    olho_direito:  uniq(F.FACE_LANDMARKS_RIGHT_EYE),
    sobrancelhas:  [...new Set([...uniq(F.FACE_LANDMARKS_LEFT_EYEBROW), ...uniq(F.FACE_LANDMARKS_RIGHT_EYEBROW)])],
    nariz:         NOSE,
    boca:          uniq(F.FACE_LANDMARKS_LIPS),
    testa:         FOREHEAD,
    contorno:      uniq(F.FACE_LANDMARKS_FACE_OVAL),
  };
  const keys = Object.keys(regions);
  const pointRegion = new Array(NUM_POINTS).fill(null);
  for (const k of keys) for (const i of regions[k]) if (pointRegion[i] === null) pointRegion[i] = k;
  return { regions, keys, pointRegion };
}

export function eyeAspectRatio(lm, eye) {
  const d = (a, b) => Math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y);
  return d(eye.up, eye.down) / Math.max(1e-6, d(eye.a, eye.b));
}

export const clamp01 = x => Math.max(0, Math.min(1, x));

/**
 * Quanto o olho parece fechado (0..1). Com o blendshape eyeBlink, as duas evidências precisam
 * concordar (mínimo): olho pequeno tem EAR baixo mas piscada baixa; semiaberto tem piscada média.
 */
export function eyeClosedScore(ear, cfg = DEFAULTS, blink = null) {
  const geo = clamp01((cfg.earOpen - ear) / (cfg.earOpen - cfg.earClosed));
  if (typeof blink !== "number") return geo;
  return Math.min(geo, clamp01((blink - cfg.blinkOpen) / (cfg.blinkClosed - cfg.blinkOpen)));
}

/**
 * Confiança numa decisão por limiar (0..1): distância do score ao limiar, relativa ao espaço
 * até o extremo. 0 = em cima do limiar (tanto faz sim ou não), 1 = no extremo (0 ou 1).
 */
export function confidence(score, thr) {
  if (typeof score !== "number" || typeof thr !== "number") return null;
  return clamp01(score > thr ? (score - thr) / (1 - thr) : (thr - score) / thr);
}
export const confLevel = c => (c === null ? null : c >= 0.6 ? "alta" : c >= 0.3 ? "média" : "baixa");

/** Coordenada onde cada landmark lê a máscara (anel nos olhos, contorno puxado para dentro). */
export function samplePoints(lm, R, cfg = DEFAULTS) {
  const center = {};
  for (const k of Object.keys(EYE)) {
    const idx = R.regions[k];
    center[k] = {
      x: idx.reduce((s, i) => s + lm[i].x, 0) / idx.length,
      y: idx.reduce((s, i) => s + lm[i].y, 0) / idx.length,
    };
  }
  const nose = lm[NOSE_TIP];
  const out = new Array(lm.length);
  for (let i = 0; i < lm.length; i++) {
    let { x, y } = lm[i];
    const k = R.pointRegion[i];
    if (k === "contorno") {
      x += (nose.x - x) * cfg.contourPull; y += (nose.y - y) * cfg.contourPull;
    } else if (center[k]) {
      x += (x - center[k].x) * cfg.eyeRing; y += (y - center[k].y) * cfg.eyeRing;
    }
    out[i] = { x, y };
  }
  return out;
}

/** Classe da máscara em uma coordenada normalizada. mask = { data, width, height } */
export function classAt(mask, x, y) {
  const px = Math.min(mask.width - 1, Math.max(0, Math.round(x * mask.width)));
  const py = Math.min(mask.height - 1, Math.max(0, Math.round(y * mask.height)));
  return mask.data[py * mask.width + px];
}

function topVote(votes) {
  let best = null, n = 0;
  for (const c in votes) if (votes[c] > n) { n = votes[c]; best = c; }
  return best === null ? null : (MOTIVO[best] || "outro");
}

/**
 * Analisa um rosto.
 * @param {Array<{x:number,y:number}>|null} lm landmarks normalizados (ou null se não há rosto)
 * @param {{data:Uint8Array,width:number,height:number}|null} mask máscara de classes (null = sem segmentador)
 * @param {ReturnType<typeof buildRegions>} R
 * @param {Float32Array} baseline fração "normal" de não-pele por ponto (referência)
 * @returns {{state: object, nonSkin: Float32Array|null, red: Uint8Array|null}}
 */
export function analyze(lm, mask, R, baseline = new Float32Array(NUM_POINTS), cfg = DEFAULTS) {
  const ok = lm && lm.length >= NUM_POINTS;
  return analyzeClasses(lm, ok && mask ? sampleClasses(lm, mask, R, cfg) : null, R, baseline, cfg);
}

/**
 * Classe da máscara lida por cada landmark, seguida das classes da faixa do boné (capPoints).
 * Uint8Array com lm.length + pontos da faixa; analyzeClasses só usa os primeiros lm.length.
 */
export function sampleClasses(lm, mask, R, cfg = DEFAULTS) {
  const pts = samplePoints(lm, R, cfg);
  const cap = sampleCapClasses(lm, mask);
  const cls = new Uint8Array(lm.length + cap.length);
  for (let i = 0; i < lm.length; i++) cls[i] = classAt(mask, pts[i].x, pts[i].y);
  cls.set(cap, lm.length);
  return cls;
}

/**
 * Como analyze, mas recebe as classes já amostradas por ponto (null = sem segmentador).
 * No modo ao vivo as classes vêm de um quadro anterior e são aplicadas aos landmarks atuais
 * pelo índice do ponto; o olho fechado (EAR) usa sempre os landmarks atuais.
 * extra.blink: { olho_esquerdo, olho_direito } com o eyeBlink do MediaPipe (opcional).
 */
export function analyzeClasses(lm, cls, R, baseline = new Float32Array(NUM_POINTS), cfg = DEFAULTS, extra = {}) {
  if (!lm || lm.length < NUM_POINTS) {
    return { state: { rosto_detectado: false }, nonSkin: null, red: null };
  }
  const n = lm.length;
  const nonSkin = new Float32Array(n);
  const hasSeg = !!cls;
  if (cls) for (let i = 0; i < n; i++) nonSkin[i] = cls[i] === CLS.FACE ? 0 : 1;
  else cls = new Uint8Array(n).fill(CLS.FACE);

  const state = { rosto_detectado: true };
  if (!hasSeg) state.segmentacao = false;

  for (const k of R.keys) {
    const idx = R.regions[k];
    let sum = 0; const votes = {};
    for (const i of idx) {
      const v = Math.max(0, nonSkin[i] - (baseline[i] || 0));
      sum += v;
      if (v > 0.5) votes[cls[i]] = (votes[cls[i]] || 0) + 1;
    }
    let score = sum / idx.length;
    let motivo = topVote(votes);
    let eye = null;
    if (EYE[k]) {
      const ear = eyeAspectRatio(lm, EYE[k]);
      const blink = extra.blink ? extra.blink[k] : null;
      const closed = eyeClosedScore(ear, cfg, blink);
      if (closed > score) { score = closed; motivo = "fechado"; }
      eye = { abertura: Math.round(ear * 1000) / 1000, piscada: typeof blink === "number" ? Math.round(blink * 100) / 100 : null };
    }
    const ocluso = score > cfg.regionOn;
    state[k] = {
      ocluso, motivo: ocluso ? motivo : null, score: Math.round(score * 100) / 100,
      confianca: Math.round(confidence(score, cfg.regionOn) * 100) / 100,
      ...eye,
    };
  }

  const red = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const k = R.pointRegion[i];
    red[i] = k ? (state[k].ocluso ? 1 : 0)
               : ((nonSkin[i] - (baseline[i] || 0)) > cfg.pointOn ? 1 : 0);
  }
  return { state, nonSkin, red };
}

/** Objeto exibido: compacto (booleanos) ou completo. Sem rosto → regiões null. */
export function toView(state, keys, full) {
  const o = { rosto_detectado: !!(state && state.rosto_detectado) };
  if (state && state.segmentacao === false) o.segmentacao = false;
  for (const k of keys) {
    if (!o.rosto_detectado) o[k] = null;
    else o[k] = full ? state[k] : state[k].ocluso;
  }
  return o;
}

const esc = s => s.replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

/** JSON formatado com spans de cor; objetos de região ficam em uma linha. */
export function jsonToHtml(obj) {
  const txt = JSON.stringify(obj, null, 2)
    .replace(/\{\n\s+("(?:ocluso|presente)"[\s\S]*?)\n\s+\}/g, m => m.replace(/\n\s+/g, " "));
  return esc(txt)
    .replace(/"([^"]+)":/g, '<span class="k">"$1"</span>:')
    .replace(/: true/g, ': <span class="t">true</span>')
    .replace(/: false/g, ': <span class="f">false</span>')
    .replace(/: null/g, ': <span class="n">null</span>')
    .replace(/: "([^"]*)"/g, ': <span class="s">"$1"</span>')
    // em rosto_detectado e segmentacao, true é bom (verde) e false é ruim (vermelho)
    .replace(/("(?:rosto_detectado|segmentacao)"<\/span>: <span class=")([tf])"/g,
      (_, pre, c) => `${pre}${c === "t" ? "f" : "t"}"`);
}

/* ---------- óculos (glasses-detector, ver server/oculos.py) ---------- */

export const GLASSES_TYPES = ["grau", "escuros"];
export const GLASSES = {
  photoOn: 0.65, // foto: limiar simples
  on: 0.75,      // ao vivo: liga acima disso...
  off: 0.5,      // ...e só desliga abaixo disso (histerese, para não piscar)
  alpha: 0.35,  // ao vivo: peso da leitura nova na média móvel (menor = uma leitura isolada pesa menos)
};

/**
 * Scores do classificador → estado dos óculos.
 * Sem prev (foto): tipo = o maior score acima de photoOn.
 * Com prev (ao vivo): média móvel exponencial dos scores + histerese no tipo.
 * @param {{grau:number, escuros:number}} scores
 * @param {{grau:number, escuros:number, tipo:string|null}|null} prev estado anterior (ao vivo)
 * @returns {{presente:boolean, tipo:"grau"|"escuros"|null, grau:number, escuros:number}}
 */
export function glassesState(scores, prev = null, cfg = GLASSES) {
  const s = {};
  for (const t of GLASSES_TYPES) s[t] = prev ? prev[t] + cfg.alpha * (scores[t] - prev[t]) : scores[t];
  const best = lim => GLASSES_TYPES.filter(t => s[t] > lim).sort((a, b) => s[b] - s[a])[0] || null;
  let tipo;
  if (!prev) tipo = best(cfg.photoOn);
  else {
    const cand = best(cfg.on);
    const keep = prev.tipo && s[prev.tipo] > cfg.off ? prev.tipo : null;
    tipo = cand && (!keep || s[cand] > s[keep]) ? cand : keep;
  }
  return {
    presente: tipo !== null, tipo, grau: s.grau, escuros: s.escuros,
    bruto: { grau: scores.grau, escuros: scores.escuros },
  };
}

/* ---------- boné e máscara por regras sobre o segmentador ---------- */

export const RULES = {
  capRows: [0.10, 0.20, 0.30], // faixa acima da testa: distância do ponto 10, em fração da altura do rosto
  capCols: 7,                  // pontos por linha
  capSpread: 0.8,              // largura da faixa, em fração da largura do rosto
  capOn: 0.35,                 // fração da faixa em acessório/roupa para marcar boné
  maskOn: 0.40,                // fração da metade de baixo do rosto em roupa/acessório para marcar máscara
  minValid: 0.5,               // boné: mínimo da faixa dentro da imagem; abaixo disso "fora do quadro"
  hyst: 0.1,                   // ao vivo: liga em limiar + hyst, desliga em limiar - hyst
  alpha: 0.5,                  // ao vivo: peso da leitura nova na média móvel
};
export const OUTSIDE = 255; // classe de um ponto de amostragem fora da imagem
const COVER = new Set([CLS.ACC, CLS.CLOTHES]);
const LOWER_NOSE = [2, 94, 97, 98, 326, 327];
const CLASS_NAME = { 0: "fundo", 1: "cabelo", 2: "mao_ou_pele", 3: "rosto", 4: "roupa", 5: "acessorio", 255: "fora" };

/** Pontos da faixa acima da testa (onde fica a copa do boné). Acompanham a inclinação da cabeça. */
export function capPoints(lm, cfg = RULES) {
  const top = lm[10], chin = lm[152], l = lm[234], r = lm[454];
  const ux = top.x - chin.x, uy = top.y - chin.y, ax = r.x - l.x, ay = r.y - l.y;
  const pts = [];
  for (const k of cfg.capRows) for (let c = 0; c < cfg.capCols; c++) {
    const t = (c / (cfg.capCols - 1) - 0.5) * cfg.capSpread;
    pts.push({ x: top.x + ux * k + ax * t, y: top.y + uy * k + ay * t });
  }
  return pts;
}

/** Landmarks da metade de baixo do rosto (onde fica a máscara): lábios, base do nariz e contorno abaixo do nariz. */
export function lowerFace(lm, R) {
  const top = lm[10], chin = lm[152], nose = lm[NOSE_TIP];
  const ux = top.x - chin.x, uy = top.y - chin.y;
  const below = i => (lm[i].x - nose.x) * ux + (lm[i].y - nose.y) * uy < 0;
  return [...new Set([...R.regions.boca, ...LOWER_NOSE, ...R.regions.contorno.filter(below)])];
}

function histogram(classes) {
  const h = {};
  for (const c of classes) { const k = CLASS_NAME[c] || "outro"; h[k] = (h[k] || 0) + 1; }
  return h;
}

/**
 * Boné e máscara a partir das classes amostradas (sampleClasses: 478 landmarks + faixa do boné no fim).
 * Só acessório e roupa contam: cabelo (franja) não é boné, pele do corpo (mão) não é máscara.
 * Na máscara, pontos que a referência diz serem "normalmente fora da pele" não contam.
 * @returns {{chapeu_ou_bone:{score:number|null, classes:object}, mascara:{score:number, classes:object}}|null}
 */
export function accessoryRules(lm, cls, R, baseline = new Float32Array(NUM_POINTS), cfg = RULES) {
  if (!lm || !cls || cls.length <= lm.length) return null;
  const cap = Array.from(cls.subarray(lm.length));
  const valid = cap.filter(c => c !== OUTSIDE);
  const chapeu_ou_bone = {
    score: valid.length < cfg.minValid * cap.length ? null : valid.filter(c => COVER.has(c)).length / valid.length,
    classes: histogram(cap),
  };
  const idx = lowerFace(lm, R);
  const low = idx.map(i => cls[i]);
  const mascara = {
    score: idx.filter(i => COVER.has(cls[i]) && !((baseline[i] || 0) >= 0.5)).length / idx.length,
    classes: histogram(low),
  };
  return { chapeu_ou_bone, mascara };
}

/**
 * Score → { presente, score }. Sem prev (foto): limiar simples.
 * Com prev (ao vivo): média móvel + histerese. score null → { presente:false, score:null, fora:true }.
 */
export function binaryState(score, thr, prev = null, cfg = RULES) {
  if (score === null) return { presente: false, score: null, fora: true };
  const live = prev && prev.score !== null && prev.score !== undefined;
  const s = live ? prev.score + cfg.alpha * (score - prev.score) : score;
  const presente = !live ? s > thr : prev.presente ? s > thr - cfg.hyst : s > thr + cfg.hyst;
  return { presente, score: s };
}

/** Classes amostradas nos pontos da faixa do boné (OUTSIDE se fora da imagem). */
export function sampleCapClasses(lm, mask, cfg = RULES) {
  return capPoints(lm, cfg).map(p => (p.x < 0 || p.x > 1 || p.y < 0 || p.y > 1) ? OUTSIDE : classAt(mask, p.x, p.y));
}

/* ---------- relatório de teste ---------- */

const pick = (o, path) => path.split(".").reduce((v, k) => (v == null ? undefined : v[k]), o);
export const REPORT_METRICS = [
  "fps", "chapeu_ou_bone.score", "mascara.score", "oculos.grau", "oculos.escuros",
  "chapeu_ou_bone.confianca", "mascara.confianca", "oculos.confianca",
  "oculos.bruto.grau", "oculos.bruto.escuros",
  "regioes.olho_esquerdo.abertura", "regioes.olho_esquerdo.piscada", "regioes.olho_esquerdo.score",
  "regioes.olho_direito.abertura", "regioes.olho_direito.piscada", "regioes.olho_direito.score",
  "servidor.chapeu_ou_bone", "servidor.mascara", "servidor.grau", "servidor.escuros",
];
export const REPORT_FLAGS = ["regioes.olho_esquerdo.ocluso", "regioes.olho_direito.ocluso", "chapeu_ou_bone.presente", "mascara.presente", "oculos.tipo", "servidor.chapeu_ou_bone_presente", "servidor.mascara_presente"];

/** Resumo por cenário: n, média/mín/máx de cada métrica numérica e contagem de cada valor dos sinalizadores. */
export function summarizeReport(samples) {
  const by = {};
  for (const s of samples) (by[s.cenario] ||= []).push(s);
  const out = {};
  for (const [cen, list] of Object.entries(by)) {
    const o = { n: list.length };
    for (const m of REPORT_METRICS) {
      const v = list.map(s => pick(s, m)).filter(x => typeof x === "number");
      if (!v.length) continue;
      const r2 = x => Math.round(x * 100) / 100;
      o[m] = { media: r2(v.reduce((a, b) => a + b, 0) / v.length), min: r2(Math.min(...v)), max: r2(Math.max(...v)) };
    }
    for (const f of REPORT_FLAGS) {
      const v = list.map(s => pick(s, f)).filter(x => x !== undefined);
      if (!v.length) continue;
      o[f] = {};
      for (const x of v) o[f][String(x)] = (o[f][String(x)] || 0) + 1;
    }
    out[cen] = o;
  }
  return out;
}

/* ---------- painel lateral (HTML a partir do estado; sem DOM) ---------- */

export const LABELS = {
  olho_esquerdo: "Olho esquerdo", olho_direito: "Olho direito", sobrancelhas: "Sobrancelhas",
  nariz: "Nariz", boca: "Boca", testa: "Testa", contorno: "Contorno",
  chapeu_ou_bone: "Chapéu ou boné", oculos: "Óculos", mascara: "Máscara",
};
const SHORT = { chapeu_ou_bone: "chapéu", oculos: "óculos", mascara: "máscara" };
const MOTIVO_LABEL = {
  acessorio: "acessório", roupa: "roupa", cabelo: "cabelo", mao_ou_pele: "mão ou pele",
  fundo: "fundo", fechado: "olho fechado", outro: "outro",
};
const ACC_KEYS = ["chapeu_ou_bone", "oculos", "mascara"];

const pct = x => Math.round(Math.max(0, Math.min(1, x)) * 100);
function meter(score, thr, on) {
  if (typeof score !== "number") return "";
  return `<span class="meter ${on ? "on" : "off"}"><i style="width:${pct(score)}%"></i>`
    + (typeof thr === "number" ? `<b style="left:${pct(thr)}%"></b>` : "") + "</span>";
}
function row(label, chip, kind, sub = "", bar = "", conf = null) {
  const c = typeof conf === "number"
    ? `<span class="conf ${confLevel(conf) === "média" ? "media" : confLevel(conf)}" title="confiança ${confLevel(conf)}">${pct(conf)}%</span>`
    : "<span></span>";
  return `<div class="row"><span class="lbl">${esc(label)}${sub ? ` <small>${esc(sub)}</small>` : ""}</span>`
    + `${c}<span class="chip ${kind}">${esc(chip)}</span>${bar}</div>`;
}

/**
 * Um acessório (forma completa de accView) → { chip, kind, score, sub }.
 * kind: "on" (detectado), "off" (não), "na" (analisando / fora do quadro / erro).
 */
export function accessoryChip(key, v) {
  if (v === undefined) return null;
  if (typeof v === "string" || v === null) return { chip: v || "—", kind: "na" };
  if (key === "oculos") {
    const score = Math.max(v.grau, v.escuros);
    return { chip: v.tipo || "não", kind: v.presente ? "on" : "off", score, conf: v.confianca,
      sub: `grau ${v.grau.toFixed(2)} · escuros ${v.escuros.toFixed(2)}` };
  }
  if (v.score === null || v.score === undefined) return { chip: v.presente ? "sim" : "não", kind: v.presente ? "on" : "off" };
  return { chip: v.presente ? "sim" : "não", kind: v.presente ? "on" : "off", score: v.score, conf: v.confianca };
}

/**
 * Corpo do painel. state: result.state; acc: accView completo (ou null); thr: limiares por item.
 */
export function panelHtml(state, keys, acc, thr = {}) {
  if (!state || !state.rosto_detectado) return `<p class="empty-row">Nenhum rosto encontrado</p>`;
  let h = "";
  if (acc && typeof acc === "object") {
    h += `<h3 class="sec">Acessórios</h3>`;
    for (const k of ACC_KEYS) {
      const c = accessoryChip(k, acc[k]);
      if (c) h += row(LABELS[k], c.chip, c.kind, c.sub, meter(c.score, thr[k], c.kind === "on"), c.conf);
    }
    if (acc.servidor !== undefined) {
      h += `<h3 class="sec">Servidor (PyTorch)</h3>`;
      if (typeof acc.servidor !== "object") h += row("Status", acc.servidor, "na");
      else for (const k of ["chapeu_ou_bone", "mascara"]) {
        const c = accessoryChip(k, acc.servidor[k]);
        if (c) h += row(LABELS[k], c.chip, c.kind, "", meter(c.score, 0.5, c.kind === "on"), c.conf);
      }
    }
  } else if (typeof acc === "string") {
    h += `<h3 class="sec">Acessórios</h3>` + row("Status", acc, "na");
  }
  h += `<h3 class="sec">Regiões do rosto</h3>`;
  if (state.segmentacao === false) h += `<p class="note">sem segmentador: só olho fechado</p>`;
  for (const k of keys) {
    const r = state[k];
    const why = r.ocluso && r.motivo ? MOTIVO_LABEL[r.motivo] || r.motivo : "";
    // olhos: abertura (EAR) e piscada, para calibrar olho pequeno / semiaberto
    const eye = typeof r.abertura === "number"
      ? `abertura ${r.abertura.toFixed(2)}` + (typeof r.piscada === "number" ? ` · piscada ${r.piscada.toFixed(2)}` : "")
      : "";
    h += row(LABELS[k] || k, r.ocluso ? "coberto" : "livre", r.ocluso ? "on" : "off",
      [why, eye].filter(Boolean).join(" · "), meter(r.score, thr.regiao, r.ocluso), r.confianca);
  }
  return h;
}

/** Resumo de uma linha (cabeçalho do painel, útil recolhido no celular). */
export function panelSummaryHtml(state, acc) {
  if (!state || !state.rosto_detectado) return `<span class="chip na">sem rosto</span>`;
  const covered = Object.values(state).filter(v => v && v.ocluso).length;
  let h = "";
  if (acc && typeof acc === "object") {
    for (const k of ACC_KEYS) {
      const c = accessoryChip(k, acc[k]);
      if (!c) continue;
      const txt = (k === "oculos" && c.kind === "on" ? c.chip : SHORT[k]) + (confLevel(c.conf ?? null) === "baixa" ? "?" : "");
      h += `<span class="chip ${c.kind}">${esc(txt)}</span>`;
    }
  }
  return h + `<span class="chip ${covered ? "on" : "off"}">${covered ? `${covered} coberta${covered > 1 ? "s" : ""}` : "rosto livre"}</span>`;
}

/** Etiquetas para a tela: só os acessórios detectados (forma completa de accView). */
export function detectedTags(acc) {
  if (!acc || typeof acc !== "object") return [];
  const tags = [];
  const c = acc.chapeu_ou_bone, g = acc.oculos, m = acc.mascara;
  if (c && c.presente) tags.push({ key: "chapeu_ou_bone", text: "Chapéu ou boné", confianca: c.confianca ?? null });
  if (g && g.presente) tags.push({ key: "oculos", text: g.tipo === "escuros" ? "Óculos escuros" : "Óculos de grau", confianca: g.confianca ?? null });
  if (m && m.presente) tags.push({ key: "mascara", text: "Máscara", confianca: m.confianca ?? null });
  return tags;
}

export function tagsHtml(tags) {
  return tags.map(t => `<span class="tag">${esc(t.text)}`
    + (typeof t.confianca === "number" ? ` <small>${pct(t.confianca)}%</small>` : "") + "</span>").join("");
}
