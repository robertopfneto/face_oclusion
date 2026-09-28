// Testes unitários da lógica de oclusão (node --test).
// Usa landmarks reais extraídos do MediaPipe numa foto de teste (tests/fixtures).
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { FaceLandmarker } from "../vendor/vision_bundle.mjs";
import {
  CLS, EYE, DEFAULTS, NUM_POINTS, NOSE_TIP,
  buildRegions, eyeAspectRatio, eyeClosedScore, samplePoints, classAt, analyze, toView, jsonToHtml, glassesState,
  capPoints, lowerFace, accessoryRules, analyzeClasses, binaryState, sampleClasses, summarizeReport, RULES, OUTSIDE,
} from "../occlusion.js";

const LM = JSON.parse(readFileSync(new URL("./fixtures/astronaut_landmarks.json", import.meta.url)));
const R = buildRegions(FaceLandmarker);
const clone = lm => lm.map(p => ({ ...p }));

/** Máscara 256×256 toda "pele do rosto", com pintura opcional por predicado. */
function makeMask(paint = () => null, W = 256, H = 256) {
  const data = new Uint8Array(W * H).fill(CLS.FACE);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const c = paint((x + 0.5) / W, (y + 0.5) / H);
    if (c !== null) data[y * W + x] = c;
  }
  return { data, width: W, height: H };
}
const center = idx => ({
  x: idx.reduce((s, i) => s + LM[i].x, 0) / idx.length,
  y: idx.reduce((s, i) => s + LM[i].y, 0) / idx.length,
});
const eyeL = center(R.regions.olho_esquerdo), eyeR = center(R.regions.olho_direito);
const eyeW = Math.hypot(LM[33].x - LM[133].x, LM[33].y - LM[133].y);
const near = (p, c, r) => Math.hypot(p.x - c.x, p.y - c.y) < r;
const glasses = (x, y) => (near({ x, y }, eyeL, eyeW * 1.1) || near({ x, y }, eyeR, eyeW * 1.1)) ? CLS.ACC : null;
const flags = s => Object.fromEntries(R.keys.map(k => [k, s[k].ocluso]));

test("fixture tem 478 landmarks normalizados", () => {
  assert.equal(LM.length, NUM_POINTS);
  for (const p of LM) { assert.ok(p.x > 0 && p.x < 1); assert.ok(p.y > 0 && p.y < 1); }
});

test("buildRegions monta as 7 regiões com os tamanhos do MediaPipe", () => {
  assert.deepEqual(R.keys, ["olho_esquerdo", "olho_direito", "sobrancelhas", "nariz", "boca", "testa", "contorno"]);
  assert.equal(R.regions.olho_esquerdo.length, 16);
  assert.equal(R.regions.olho_direito.length, 16);
  assert.equal(R.regions.boca.length, 40);
  assert.equal(R.regions.contorno.length, 36);
  assert.equal(new Set(R.regions.sobrancelhas).size, R.regions.sobrancelhas.length);
  for (const k of R.keys) for (const i of R.regions[k]) assert.ok(i >= 0 && i < NUM_POINTS);
  assert.equal(R.pointRegion.length, NUM_POINTS);
});

test("olho esquerdo é o da pessoa (aparece à direita na imagem)", () => {
  assert.ok(eyeL.x > eyeR.x, "olho_esquerdo deveria ter x maior que olho_direito");
  assert.ok(R.regions.olho_esquerdo.includes(EYE.olho_esquerdo.up));
  assert.ok(R.regions.olho_direito.includes(EYE.olho_direito.up));
});

test("olhos abertos da foto real não passam do limiar de fechado", () => {
  for (const k of Object.keys(EYE)) {
    const ear = eyeAspectRatio(LM, EYE[k]);
    assert.ok(ear > DEFAULTS.earClosed, `${k}: EAR ${ear.toFixed(3)} deveria indicar olho aberto`);
    assert.ok(eyeClosedScore(ear) < DEFAULTS.regionOn, `${k}: score ${eyeClosedScore(ear)}`);
  }
});

test("eyeClosedScore é 0 no olho aberto, 1 no fechado e limitado a [0,1]", () => {
  assert.equal(eyeClosedScore(DEFAULTS.earOpen), 0);
  assert.equal(eyeClosedScore(0.5), 0);
  assert.equal(eyeClosedScore(DEFAULTS.earClosed), 1);
  assert.equal(eyeClosedScore(0), 1);
});

test("sem rosto → rosto_detectado false e sem pontos", () => {
  const r = analyze(null, null, R);
  assert.deepEqual(r.state, { rosto_detectado: false });
  assert.equal(r.red, null);
  const v = toView(r.state, R.keys, false);
  assert.equal(v.rosto_detectado, false);
  for (const k of R.keys) assert.equal(v[k], null);
});

test("rosto livre, sem segmentador → nada ocluso e segmentacao false", () => {
  const r = analyze(LM, null, R);
  assert.equal(r.state.segmentacao, false);
  assert.ok(Object.values(flags(r.state)).every(v => v === false));
  assert.equal(r.red.reduce((a, b) => a + b, 0), 0);
});

test("rosto livre com máscara toda pele → nada ocluso", () => {
  const r = analyze(LM, makeMask(), R);
  assert.equal(r.state.segmentacao, undefined);
  assert.ok(Object.values(flags(r.state)).every(v => v === false));
});

test("olho fechado (geometria) marca só aquele olho, motivo 'fechado'", () => {
  const lm = clone(LM);
  const e = EYE.olho_esquerdo;
  const mid = (lm[e.up].y + lm[e.down].y) / 2;
  lm[e.up].y = mid; lm[e.down].y = mid + 0.0005; lm[e.down].x = lm[e.up].x;
  const r = analyze(lm, null, R);
  const { abertura, piscada, ...decisao } = r.state.olho_esquerdo;
  assert.deepEqual(decisao, { ocluso: true, motivo: "fechado", score: 1, confianca: 1 });
  assert.ok(abertura < DEFAULTS.earClosed);
  assert.equal(piscada, null);
  assert.equal(r.state.olho_direito.ocluso, false);
  for (const i of R.regions.olho_esquerdo) assert.equal(r.red[i], 1);
  for (const i of R.regions.olho_direito) assert.equal(r.red[i], 0);
});

test("óculos (acessório ao redor dos olhos) → olhos oclusos, motivo 'acessorio'", () => {
  const r = analyze(LM, makeMask(glasses), R);
  for (const k of ["olho_esquerdo", "olho_direito"]) {
    assert.equal(r.state[k].ocluso, true, k);
    assert.equal(r.state[k].motivo, "acessorio", k);
  }
  assert.equal(r.state.boca.ocluso, false);
  assert.equal(r.state.testa.ocluso, false);
});

test("máscara de tecido abaixo do nariz → boca oclusa, motivo 'roupa'", () => {
  const nose = LM[NOSE_TIP];
  const r = analyze(LM, makeMask((x, y) => (y > nose.y ? CLS.CLOTHES : null)), R);
  assert.equal(r.state.boca.ocluso, true);
  assert.equal(r.state.boca.motivo, "roupa");
  assert.equal(r.state.olho_esquerdo.ocluso, false);
  assert.equal(r.state.olho_direito.ocluso, false);
});

test("mão sobre a boca (pele do corpo) → motivo 'mao_ou_pele'", () => {
  const nose = LM[NOSE_TIP];
  const r = analyze(LM, makeMask((x, y) => (y > nose.y ? CLS.BODY : null)), R);
  assert.equal(r.state.boca.ocluso, true);
  assert.equal(r.state.boca.motivo, "mao_ou_pele");
});

test("boné/cabelo cobrindo a testa → testa oclusa, motivo 'cabelo'", () => {
  const browTop = Math.min(...R.regions.sobrancelhas.map(i => LM[i].y));
  const r = analyze(LM, makeMask((x, y) => (y < browTop - 0.01 ? CLS.HAIR : null)), R);
  assert.equal(r.state.testa.ocluso, true);
  assert.equal(r.state.testa.motivo, "cabelo");
  assert.equal(r.state.boca.ocluso, false);
});

test("referência remove falso positivo de região naturalmente fora da pele", () => {
  // simula um segmentador que não rotula os lábios como pele
  const lips = R.regions.boca.map(i => LM[i]);
  const minX = Math.min(...lips.map(p => p.x)) - 0.01, maxX = Math.max(...lips.map(p => p.x)) + 0.01;
  const minY = Math.min(...lips.map(p => p.y)) - 0.01, maxY = Math.max(...lips.map(p => p.y)) + 0.01;
  const lipsMask = makeMask((x, y) => (x > minX && x < maxX && y > minY && y < maxY ? CLS.BG : null));

  const semRef = analyze(LM, lipsMask, R);
  assert.equal(semRef.state.boca.ocluso, true, "sem referência deveria ser falso positivo");

  const baseline = Float32Array.from(semRef.nonSkin);
  const comRef = analyze(LM, lipsMask, R, baseline);
  assert.equal(comRef.state.boca.ocluso, false, "com referência não deveria marcar");
  assert.equal(comRef.state.boca.score, 0);

  // a referência não esconde uma oclusão nova (óculos)
  const both = makeMask((x, y) => glasses(x, y) ?? (x > minX && x < maxX && y > minY && y < maxY ? CLS.BG : null));
  const r = analyze(LM, both, R, baseline);
  assert.equal(r.state.olho_esquerdo.ocluso, true);
  assert.equal(r.state.boca.ocluso, false);
});

test("ponto fora das regiões fica vermelho quando sai da pele", () => {
  const free = [...Array(NUM_POINTS).keys()].find(i => R.pointRegion[i] === null);
  assert.ok(free !== undefined, "deveria haver pontos fora das regiões (bochechas)");
  const p = LM[free];
  const r = analyze(LM, makeMask((x, y) => (near({ x, y }, p, 0.01) ? CLS.ACC : null)), R);
  assert.equal(r.red[free], 1);
  assert.equal(Object.values(flags(r.state)).some(Boolean), false);
});

test("samplePoints: olhos amostram fora do olho e contorno para dentro", () => {
  const s = samplePoints(LM, R);
  const i = R.regions.olho_esquerdo[0];
  assert.ok(Math.hypot(s[i].x - eyeL.x, s[i].y - eyeL.y) > Math.hypot(LM[i].x - eyeL.x, LM[i].y - eyeL.y));
  const c = R.regions.contorno.find(j => R.pointRegion[j] === "contorno");
  const n = LM[NOSE_TIP];
  assert.ok(Math.hypot(s[c].x - n.x, s[c].y - n.y) < Math.hypot(LM[c].x - n.x, LM[c].y - n.y));
});

test("classAt limita coordenadas fora da imagem", () => {
  const m = makeMask((x, y) => (x < 0.5 ? CLS.HAIR : CLS.ACC), 4, 4);
  assert.equal(classAt(m, -1, -1), CLS.HAIR);
  assert.equal(classAt(m, 2, 2), CLS.ACC);
});

test("toView: compacto só com booleanos, completo com score e motivo", () => {
  const r = analyze(LM, makeMask(glasses), R);
  const c = toView(r.state, R.keys, false);
  assert.equal(c.rosto_detectado, true);
  assert.equal(c.olho_esquerdo, true);
  assert.equal(c.boca, false);
  const f = toView(r.state, R.keys, true);
  assert.deepEqual(Object.keys(f.olho_esquerdo), ["ocluso", "motivo", "score", "confianca", "abertura", "piscada"]);
  assert.deepEqual(Object.keys(f.boca), ["ocluso", "motivo", "score", "confianca"]);
  assert.equal(toView(analyze(LM, null, R).state, R.keys, false).segmentacao, false);
});

test("jsonToHtml colore valores, escapa HTML e deixa cada região em uma linha", () => {
  const r = analyze(LM, makeMask(glasses), R);
  const html = jsonToHtml(toView(r.state, R.keys, true));
  assert.match(html, /<span class="k">"olho_esquerdo"<\/span>: \{ <span class="k">"ocluso"<\/span>: <span class="t">true<\/span>, .*"acessorio".* \}/);
  assert.match(html, /<span class="f">false<\/span>/);
  assert.equal(jsonToHtml({ a: "<b>" }).includes("<b>"), false);
  assert.match(html, /"rosto_detectado"<\/span>: <span class="f">true/);
  assert.match(jsonToHtml({ rosto_detectado: false }), /"rosto_detectado"<\/span>: <span class="t">false/);
  assert.match(jsonToHtml({ rosto_detectado: true, segmentacao: false }), /"segmentacao"<\/span>: <span class="t">false/);
  const plain = html.replace(/<[^>]+>/g, "").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&");
  assert.deepEqual(JSON.parse(plain), toView(r.state, R.keys, true));
});

test("óculos na foto: maior score acima do limiar; nada acima → sem óculos", () => {
  assert.equal(glassesState({ grau: 0.9, escuros: 0.2 }).tipo, "grau");
  assert.equal(glassesState({ grau: 0.7, escuros: 0.95 }).tipo, "escuros");
  const nada = glassesState({ grau: 0.3, escuros: 0.1 });
  assert.equal(nada.presente, false);
  assert.equal(nada.tipo, null);
});

test("óculos ao vivo: média móvel e histerese evitam piscar", () => {
  const A = { ...GLASSES_CFG, alpha: 0.5, on: 0.6, off: 0.4 }; // contas abaixo usam estes valores
  const glassesState = (sc, prev) => glassesStateFn(sc, prev, A);
  let g = glassesState({ grau: 0, escuros: 0 });
  // um quadro isolado alto não liga (média 0,45 < on 0,6)
  g = glassesState({ grau: 0.9, escuros: 0 }, g);
  assert.equal(g.tipo, null);
  // sustentado, liga
  g = glassesState({ grau: 0.9, escuros: 0 }, g);
  assert.equal(g.tipo, "grau");
  // uma leitura baixa não desliga (média 0,46 > off 0,4)
  g = glassesState({ grau: 0.2, escuros: 0 }, g);
  assert.equal(g.tipo, "grau");
  // baixa sustentada, desliga
  g = glassesState({ grau: 0.1, escuros: 0 }, g);
  g = glassesState({ grau: 0.1, escuros: 0 }, g);
  assert.equal(g.tipo, null);
});

test("óculos ao vivo: troca de tipo só quando o outro supera o atual", () => {
  const A = { ...GLASSES_CFG, alpha: 0.5, on: 0.6, off: 0.4 }; // contas abaixo usam estes valores
  const glassesState = (sc, prev) => glassesStateFn(sc, prev, A);
  let g = { grau: 0.8, escuros: 0.1, tipo: "grau" };
  g = glassesState({ grau: 0.7, escuros: 0.75 }, g);   // escuros 0,43 < on
  assert.equal(g.tipo, "grau");
  g = glassesState({ grau: 0.3, escuros: 0.99 }, g);   // escuros 0,71 > grau 0,6
  assert.equal(g.tipo, "escuros");
});

/* ---------- boné e máscara (regras) ---------- */
const TOP = LM[10], NOSE = LM[1];
const aboveHead = paint => (x, y) => (y < TOP.y - 0.01 ? paint : null);
const belowNose = paint => (x, y) => (y > NOSE.y + 0.01 ? paint : null);
const rules = (mask, baseline) => accessoryRules(LM, sampleClasses(LM, mask, R), R, baseline);

test("capPoints: 21 pontos acima da testa, dentro da imagem na foto de teste", () => {
  const pts = capPoints(LM);
  assert.equal(pts.length, RULES.capRows.length * RULES.capCols);
  for (const p of pts) { assert.ok(p.y < TOP.y, JSON.stringify(p)); assert.ok(p.x > 0 && p.x < 1 && p.y > 0); }
});

test("lowerFace: lábios, base do nariz e contorno abaixo do nariz; nada da testa", () => {
  const idx = lowerFace(LM, R);
  for (const i of R.regions.boca) assert.ok(idx.includes(i));
  assert.ok(idx.includes(152), "queixo");
  assert.ok(!idx.includes(10), "testa");
});

test("boné: faixa acima da testa em acessório → presente; em cabelo (franja/cabelo) → não", () => {
  const cap = rules(makeMask(aboveHead(CLS.ACC)));
  assert.ok(cap.chapeu_ou_bone.score > 0.9, cap.chapeu_ou_bone.score);
  assert.equal(binaryState(cap.chapeu_ou_bone.score, RULES.capOn).presente, true);
  const hair = rules(makeMask(aboveHead(CLS.HAIR)));
  assert.equal(hair.chapeu_ou_bone.score, 0);
  assert.equal(hair.chapeu_ou_bone.classes.cabelo, RULES.capRows.length * RULES.capCols);
});

test("máscara: roupa abaixo do nariz → presente; mão (pele do corpo) → não", () => {
  const m = rules(makeMask(belowNose(CLS.CLOTHES)));
  assert.ok(m.mascara.score > RULES.maskOn, m.mascara.score);
  assert.equal(binaryState(m.mascara.score, RULES.maskOn).presente, true);
  const hand = rules(makeMask(belowNose(CLS.BODY)));
  assert.equal(hand.mascara.score, 0);
  assert.ok(hand.mascara.classes.mao_ou_pele > 0);
});

test("máscara: pontos que a referência diz serem normalmente fora da pele não contam", () => {
  const baseline = new Float32Array(NUM_POINTS).fill(1);
  assert.equal(rules(makeMask(belowNose(CLS.CLOTHES)), baseline).mascara.score, 0);
});

test("rosto livre (tudo pele) → nem boné nem máscara", () => {
  const r = rules(makeMask());
  assert.equal(r.chapeu_ou_bone.score, 0);
  assert.equal(r.mascara.score, 0);
});

test("boné: faixa fora da imagem → score null (fora do quadro)", () => {
  const dy = TOP.y - 0.005;
  const lm = LM.map(p => ({ x: p.x, y: p.y - dy }));
  const cls = sampleClasses(lm, makeMask(), R);
  assert.ok(Array.from(cls.subarray(lm.length)).every(c => c === OUTSIDE));
  const r = accessoryRules(lm, cls, R);
  assert.equal(r.chapeu_ou_bone.score, null);
  assert.deepEqual(binaryState(r.chapeu_ou_bone.score, RULES.capOn), { presente: false, score: null, fora: true });
});

test("binaryState ao vivo: média móvel e histerese", () => {
  let s = binaryState(0, 0.4);
  s = binaryState(0.9, 0.4, s);            // média 0,45 < 0,5 (liga em limiar + 0,1)
  assert.equal(s.presente, false);
  s = binaryState(0.9, 0.4, s);            // 0,675 → liga
  assert.equal(s.presente, true);
  s = binaryState(0.2, 0.4, s);            // 0,44 > 0,3 → continua
  assert.equal(s.presente, true);
  s = binaryState(0, 0.4, s); s = binaryState(0, 0.4, s);
  assert.equal(s.presente, false);
});

test("sampleClasses não altera a análise das regiões (classes extras ficam no fim)", () => {
  const mask = makeMask(glasses);
  const cls = sampleClasses(LM, mask, R);
  assert.equal(cls.length, NUM_POINTS + RULES.capRows.length * RULES.capCols);
  assert.deepEqual(analyzeClasses(LM, cls, R).state, analyzeClasses(LM, cls.slice(0, NUM_POINTS), R).state);
});

test("summarizeReport: agrupa por cenário, média/mín/máx e contagem de sinalizadores", () => {
  const r = summarizeReport([
    { cenario: "chapeu_ou_bone", chapeu_ou_bone: { score: 0.8, presente: true } },
    { cenario: "chapeu_ou_bone", chapeu_ou_bone: { score: 0.4, presente: false } },
    { cenario: "sem", chapeu_ou_bone: { score: 0, presente: false }, oculos: { grau: 0.1, escuros: 0, tipo: null } },
  ]);
  assert.equal(r.chapeu_ou_bone.n, 2);
  assert.deepEqual(r.chapeu_ou_bone["chapeu_ou_bone.score"], { media: 0.6, min: 0.4, max: 0.8 });
  assert.deepEqual(r.chapeu_ou_bone["chapeu_ou_bone.presente"], { true: 1, false: 1 });
  assert.deepEqual(r.sem["oculos.tipo"], { null: 1 });
});

/* ---------- painel ---------- */
import { panelHtml, panelSummaryHtml, accessoryChip } from "../occlusion.js";

test("painel: sem rosto mostra aviso; com rosto lista acessórios e as 7 regiões", () => {
  assert.match(panelHtml({ rosto_detectado: false }, R.keys, null), /Nenhum rosto/);
  const { state } = analyze(LM, makeMask(glasses), R);
  const acc = { chapeu_ou_bone: { presente: false, score: 0.1 }, oculos: { presente: true, tipo: "grau", grau: 0.9, escuros: 0.02 }, mascara: "analisando" };
  const h = panelHtml(state, R.keys, acc, { chapeu_ou_bone: 0.35, regiao: 0.4 });
  for (const k of R.keys) assert.ok(h.includes(LABELS_OF(k)), k);
  assert.match(h, /Chapéu ou boné.*chip off">não/s);
  assert.match(h, /Óculos.*chip on">grau/s);
  assert.match(h, /Máscara.*chip na">analisando/s);
  assert.match(h, /Olho esquerdo.*chip on">coberto.*acessório/s);
  assert.equal((h.match(/class="row"/g) || []).length, 3 + R.keys.length);
});
const LABELS_OF = k => ({ olho_esquerdo: "Olho esquerdo", olho_direito: "Olho direito", sobrancelhas: "Sobrancelhas", nariz: "Nariz", boca: "Boca", testa: "Testa", contorno: "Contorno" })[k];

test("painel: servidor aparece para comparar; fora do quadro e erros viram chip neutro", () => {
  const { state } = analyze(LM, makeMask(), R);
  const h = panelHtml(state, R.keys, { chapeu_ou_bone: "fora do quadro", servidor: { chapeu_ou_bone: { presente: true, score: 0.9 }, mascara: { presente: false, score: 0.1 } } });
  assert.match(h, /Servidor \(PyTorch\)/);
  assert.match(h, /chip na">fora do quadro/);
  assert.deepEqual(accessoryChip("mascara", undefined), null);
});

test("resumo do painel: chips dos acessórios e contagem de regiões cobertas", () => {
  const { state } = analyze(LM, makeMask(glasses), R);
  const s = panelSummaryHtml(state, { chapeu_ou_bone: { presente: false, score: 0 }, oculos: { presente: true, tipo: "escuros", grau: 0, escuros: 1 }, mascara: { presente: true, score: 0.8 } });
  assert.match(s, /chip off">chapéu/);
  assert.match(s, /chip on">escuros/);
  assert.match(s, /chip on">máscara/);
  const n = R.keys.filter(k => state[k].ocluso).length;
  assert.ok(n > 1);
  assert.ok(s.includes(`chip on">${n} cobertas`), s);
  assert.match(panelSummaryHtml({ rosto_detectado: false }, null), /sem rosto/);
});

/* ---------- olho fechado com blendshape e confiança ---------- */
import { confidence, confLevel } from "../occlusion.js";

test("olho pequeno ou semiaberto (EAR baixo) não vira fechado se a piscada é baixa", () => {
  const ear = 0.13;                                  // pela geometria já seria quase fechado
  assert.ok(eyeClosedScore(ear) > DEFAULTS.regionOn);
  assert.equal(eyeClosedScore(ear, DEFAULTS, 0.2), 0); // olho pequeno: piscada baixa
  assert.ok(eyeClosedScore(ear, DEFAULTS, 0.45) < DEFAULTS.regionOn); // semiaberto
  assert.equal(eyeClosedScore(0.05, DEFAULTS, 0.9), 1);               // fechado de verdade
  assert.equal(eyeClosedScore(0.35, DEFAULTS, 0.9), 0);               // piscada alta mas olho aberto
});

test("analyzeClasses usa a piscada por olho (extra.blink) e expõe abertura e piscada", () => {
  const lm = clone(LM);
  const e = EYE.olho_esquerdo;
  const mid = (lm[e.up].y + lm[e.down].y) / 2;
  lm[e.up].y = mid; lm[e.down].y = mid + 0.0005; lm[e.down].x = lm[e.up].x;     // olho esquerdo geometricamente fechado
  const aberto = analyzeClasses(lm, null, R, undefined, DEFAULTS, { blink: { olho_esquerdo: 0.1, olho_direito: 0.05 } });
  assert.equal(aberto.state.olho_esquerdo.ocluso, false, "piscada baixa manda");
  assert.equal(aberto.state.olho_esquerdo.piscada, 0.1);
  const fechado = analyzeClasses(lm, null, R, undefined, DEFAULTS, { blink: { olho_esquerdo: 0.95, olho_direito: 0.05 } });
  assert.equal(fechado.state.olho_esquerdo.motivo, "fechado");
  assert.equal(fechado.state.olho_direito.ocluso, false);
});

test("confiança: 0 no limiar, 1 nos extremos, simétrica para sim e não", () => {
  assert.equal(confidence(0.4, 0.4), 0);
  assert.equal(confidence(1, 0.4), 1);
  assert.equal(confidence(0, 0.4), 1);
  assert.ok(Math.abs(confidence(0.7, 0.4) - 0.5) < 1e-9);
  assert.ok(Math.abs(confidence(0.2, 0.4) - 0.5) < 1e-9);
  assert.equal(confidence(null, 0.4), null);
  assert.equal(confLevel(0.8), "alta");
  assert.equal(confLevel(0.4), "média");
  assert.equal(confLevel(0.1), "baixa");
});

test("painel mostra confiança (% e nível) e abertura/piscada nos olhos", () => {
  const { state } = analyzeClasses(LM, null, R, undefined, DEFAULTS, { blink: { olho_esquerdo: 0.1, olho_direito: 0.12 } });
  const h = panelHtml(state, R.keys, { mascara: { presente: false, score: 0.38, confianca: 0.05 } }, { regiao: 0.4 });
  assert.match(h, /Olho esquerdo <small>abertura 0\.\d\d · piscada 0\.10<\/small>/);
  assert.match(h, /class="conf alta"[^>]*>100%/);       // região livre, score 0 → confiança máxima
  assert.match(h, /Máscara.*class="conf baixa"[^>]*>5%/s); // perto do limiar → baixa
  assert.match(panelSummaryHtml(state, { mascara: { presente: false, score: 0.38, confianca: 0.05 } }), /máscara\?/);
});

/* ---------- óculos: limiares ---------- */
import { glassesState as glassesStateFn, GLASSES as GLASSES_CFG } from "../occlusion.js";

test("óculos: limiar da foto mais alto; score médio não acusa", () => {
  assert.equal(glassesStateFn({ grau: 0.6, escuros: 0 }).tipo, null, "0,6 < 0,65");
  assert.equal(glassesStateFn({ grau: 0.9, escuros: 0 }).tipo, "grau");
  assert.deepEqual(glassesStateFn({ grau: 0.9, escuros: 0.1 }).bruto, { grau: 0.9, escuros: 0.1 });
  assert.ok(GLASSES_CFG.on > GLASSES_CFG.photoOn && GLASSES_CFG.off < GLASSES_CFG.photoOn);
});

/* ---------- etiquetas na tela ---------- */
import { detectedTags, tagsHtml } from "../occlusion.js";

test("etiquetas: só o que foi detectado, com o tipo dos óculos e a confiança", () => {
  assert.deepEqual(detectedTags(null), []);
  assert.deepEqual(detectedTags({ chapeu_ou_bone: "analisando", oculos: { presente: false }, mascara: { presente: false } }), []);
  const t = detectedTags({
    chapeu_ou_bone: { presente: true, score: 0.8, confianca: 0.7 },
    oculos: { presente: true, tipo: "escuros", confianca: 0.9 },
    mascara: { presente: true, score: 0.9, confianca: 0.83 },
  });
  assert.deepEqual(t.map(x => x.text), ["Chapéu ou boné", "Óculos escuros", "Máscara"]);
  assert.equal(detectedTags({ oculos: { presente: true, tipo: "grau" } })[0].text, "Óculos de grau");
  assert.equal(tagsHtml(t.slice(0, 1)), '<span class="tag">Chapéu ou boné <small>70%</small></span>');
});
