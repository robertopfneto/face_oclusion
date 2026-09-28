// Gera tests/fixtures/golden.json: entradas fixas + o que occlusion.js responde para elas.
// O port Python (oclusao_py/regras.py) é testado contra este arquivo (tests/python/test_paridade.py).
// Rode de novo sempre que mudar occlusion.js:  node tests/golden.mjs
import { readFileSync, writeFileSync } from "node:fs";
import { FaceLandmarker } from "../vendor/vision_bundle.mjs";
import * as O from "../occlusion.js";

const LM = JSON.parse(readFileSync(new URL("./fixtures/astronaut_landmarks.json", import.meta.url)));
const R = O.buildRegions(FaceLandmarker);
const clone = lm => lm.map(p => ({ x: p.x, y: p.y }));
const S = 128; // máscaras sintéticas pequenas (a resolução não importa: a leitura é normalizada)

function makeMask(paint = () => null) {
  const data = new Uint8Array(S * S).fill(O.CLS.FACE);
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    const c = paint((x + 0.5) / S, (y + 0.5) / S);
    if (c !== null) data[y * S + x] = c;
  }
  return { data, width: S, height: S };
}
const center = idx => ({
  x: idx.reduce((s, i) => s + LM[i].x, 0) / idx.length,
  y: idx.reduce((s, i) => s + LM[i].y, 0) / idx.length,
});
const eyeL = center(R.regions.olho_esquerdo), eyeR = center(R.regions.olho_direito);
const eyeW = Math.hypot(LM[33].x - LM[133].x, LM[33].y - LM[133].y);
const near = (x, y, c, r) => Math.hypot(x - c.x, y - c.y) < r;
const TOP = LM[10], NOSE = LM[1];

const MASKS = {
  pele: makeMask(),
  oculos: makeMask((x, y) => (near(x, y, eyeL, eyeW * 1.1) || near(x, y, eyeR, eyeW * 1.1) ? O.CLS.ACC : null)),
  bone: makeMask((x, y) => (y < TOP.y - 0.01 ? O.CLS.ACC : null)),
  cabelo_acima: makeMask((x, y) => (y < TOP.y - 0.01 ? O.CLS.HAIR : null)),
  mascara: makeMask((x, y) => (y > NOSE.y + 0.01 ? O.CLS.CLOTHES : null)),
  mao: makeMask((x, y) => (y > NOSE.y + 0.01 ? O.CLS.BODY : null)),
  franja_testa: makeMask((x, y) => (y < TOP.y + 0.04 ? O.CLS.HAIR : null)),
  // empate proposital de votos na boca (roupa x acessório) para conferir a ordem do topVote
  empate: makeMask((x, y) => (y > NOSE.y + 0.01 ? (x < NOSE.x ? O.CLS.CLOTHES : O.CLS.ACC) : null)),
};

// conjuntos de landmarks
const closedL = clone(LM);
{ const e = O.EYE.olho_esquerdo, mid = (closedL[e.up].y + closedL[e.down].y) / 2;
  closedL[e.up].y = mid; closedL[e.down].y = mid + 0.0005; closedL[e.down].x = closedL[e.up].x; }
const dy = TOP.y - 0.005;
const shifted = LM.map(p => ({ x: p.x, y: p.y - dy }));            // faixa do boné fora da imagem
const ang = 15 * Math.PI / 180, cos = Math.cos(ang), sin = Math.sin(ang);
const tilted = LM.map(p => ({                                     // cabeça inclinada 15° em torno do nariz
  x: NOSE.x + (p.x - NOSE.x) * cos - (p.y - NOSE.y) * sin,
  y: NOSE.y + (p.x - NOSE.x) * sin + (p.y - NOSE.y) * cos,
}));
const LMS = { original: LM, olho_esq_fechado: closedL, deslocado: shifted, inclinado: tilted };

const ones = new Float32Array(O.NUM_POINTS).fill(1);
const half = new Float32Array(O.NUM_POINTS); for (let i = 0; i < O.NUM_POINTS; i += 2) half[i] = 1;
const BASELINES = { zero: new Float32Array(O.NUM_POINTS), um: ones, alternado: half };
const BLINKS = { nenhum: null, aberto: { olho_esquerdo: 0.1, olho_direito: 0.05 }, fechado_esq: { olho_esquerdo: 0.95, olho_direito: 0.05 } };

const b64 = u8 => Buffer.from(u8.buffer, u8.byteOffset, u8.byteLength).toString("base64");
const arr = a => Array.from(a);

const casos = [];
const combos = [
  ["original", "pele", "zero", "nenhum"], ["original", "oculos", "zero", "nenhum"], ["original", "bone", "zero", "nenhum"],
  ["original", "cabelo_acima", "zero", "nenhum"], ["original", "mascara", "zero", "nenhum"], ["original", "mao", "zero", "nenhum"],
  ["original", "franja_testa", "zero", "nenhum"], ["original", "empate", "zero", "nenhum"],
  ["original", "mascara", "um", "nenhum"], ["original", "oculos", "alternado", "aberto"],
  ["olho_esq_fechado", "pele", "zero", "nenhum"], ["olho_esq_fechado", "pele", "zero", "aberto"],
  ["olho_esq_fechado", "pele", "zero", "fechado_esq"], ["olho_esq_fechado", null, "zero", "fechado_esq"],
  ["deslocado", "bone", "zero", "nenhum"], ["inclinado", "bone", "zero", "nenhum"], ["inclinado", "mascara", "zero", "aberto"],
  ["original", null, "zero", "nenhum"],
];
for (const [lmName, maskName, blName, blinkName] of combos) {
  const lm = LMS[lmName], mask = maskName ? MASKS[maskName] : null, bl = BASELINES[blName], blink = BLINKS[blinkName];
  const cls = mask ? O.sampleClasses(lm, mask, R) : null;
  const res = O.analyzeClasses(lm, cls, R, bl, O.DEFAULTS, blink ? { blink } : {});
  const rules = O.accessoryRules(lm, cls, R, bl);
  casos.push({
    landmarks: lmName, mascara: maskName, referencia: blName, piscada: blinkName,
    esperado: {
      pontos_amostra: O.samplePoints(lm, R).map(p => [p.x, p.y]),
      classes: cls ? arr(cls) : null,
      estado: res.state, nao_pele: arr(res.nonSkin), vermelho: arr(res.red),
      regras: rules,
      estados_regras: rules && {
        chapeu_ou_bone: O.binaryState(rules.chapeu_ou_bone.score, O.RULES.capOn),
        mascara: O.binaryState(rules.mascara.score, O.RULES.maskOn),
      },
      faixa_bone: O.capPoints(lm).map(p => [p.x, p.y]),
      metade_baixo: O.lowerFace(lm, R),
    },
  });
}

// sequências do ao vivo (média móvel + histerese)
function chain(fn, inputs) {
  let prev = null; const out = [];
  for (const x of inputs) { prev = fn(x, prev); out.push(prev); }
  return out;
}
const seqBin = [0, 0.9, 0.9, 0.2, 0, 0, 0.5, 0.55, 0.3, null, 0.8];
const seqGl = [
  { grau: 0, escuros: 0 }, { grau: 0.9, escuros: 0 }, { grau: 0.95, escuros: 0.1 }, { grau: 0.99, escuros: 0 },
  { grau: 0.3, escuros: 0.99 }, { grau: 0.1, escuros: 0.99 }, { grau: 0.1, escuros: 0.2 }, { grau: 0, escuros: 0 },
  { grau: 0.8, escuros: 0.8 },
];

const golden = {
  gerado_por: "tests/golden.mjs a partir de occlusion.js",
  config: { DEFAULTS: O.DEFAULTS, RULES: O.RULES, GLASSES: O.GLASSES, NUM_POINTS: O.NUM_POINTS, OUTSIDE: O.OUTSIDE },
  regioes: { chaves: R.keys, regioes: R.regions, ponto_regiao: R.pointRegion },
  landmarks: Object.fromEntries(Object.entries(LMS).map(([k, v]) => [k, v.map(p => [p.x, p.y])])),
  mascaras: Object.fromEntries(Object.entries(MASKS).map(([k, m]) => [k, { largura: m.width, altura: m.height, dados_b64: b64(m.data) }])),
  referencias: Object.fromEntries(Object.entries(BASELINES).map(([k, v]) => [k, arr(v)])),
  piscadas: BLINKS,
  casos,
  olho_fechado: [0.05, 0.1, 0.13, 0.15, 0.2, 0.3].flatMap(ear => [null, 0.1, 0.35, 0.45, 0.7, 0.95]
    .map(bl => ({ ear, piscada: bl, score: O.eyeClosedScore(ear, O.DEFAULTS, bl) }))),
  confianca: [[0, 0.4], [0.2, 0.4], [0.4, 0.4], [0.7, 0.4], [1, 0.4], [0.9, 0.65], [0.3, 0.35], [null, 0.4]]
    .map(([s, t]) => ({ score: s, limiar: t, confianca: O.confidence(s, t), nivel: O.confLevel(O.confidence(s, t)) })),
  ao_vivo_binario: { entradas: seqBin, limiar: 0.4, saidas: chain((x, p) => O.binaryState(x, 0.4, p), seqBin) },
  etiquetas: [
    null,
    { chapeu_ou_bone: "analisando", oculos: "analisando", mascara: "fora do quadro" },
    { chapeu_ou_bone: { presente: false, score: 0.1, confianca: 0.71 }, oculos: { presente: false, tipo: null, grau: 0.1, escuros: 0, confianca: 0.85 }, mascara: { presente: false, score: 0, confianca: 1 } },
    { chapeu_ou_bone: { presente: true, score: 0.8, confianca: 0.69 }, oculos: { presente: true, tipo: "grau", grau: 0.9, escuros: 0.1, confianca: 0.71 }, mascara: { presente: true, score: 0.9, confianca: 0.83 } },
    { oculos: { presente: true, tipo: "escuros", grau: 0, escuros: 0.99, confianca: 0.97 } },
  ].map(acc => ({ acessorios: acc, etiquetas: O.detectedTags(acc) })),
  oculos_foto: seqGl.map(s => ({ entrada: s, saida: O.glassesState(s) })),
  oculos_ao_vivo: { entradas: seqGl, saidas: chain((x, p) => O.glassesState(x, p), seqGl) },
};

const out = new URL("./fixtures/golden.json", import.meta.url);
writeFileSync(out, JSON.stringify(golden));
console.log(`golden.json: ${casos.length} casos, ${(JSON.stringify(golden).length / 1024).toFixed(0)} KB`);
