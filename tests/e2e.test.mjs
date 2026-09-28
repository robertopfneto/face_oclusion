// Testes ponta a ponta: app real + MediaPipe real num Chromium headless (viewport de celular).
// Pré-requisito: servidor na pasta do projeto (npm run serve) em http://localhost:8765/
import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdirSync } from "node:fs";
import { openApp, upload, BASE } from "./browser.mjs";

const here = f => new URL(f, import.meta.url).pathname;
const OUT = here("./out/");
mkdirSync(OUT, { recursive: true });

// foto com "óculos escuros" desenhados sobre os olhos (para ver se o rosto continua sendo achado)
execFileSync("python3", ["-c", `
from PIL import Image, ImageDraw
im = Image.open("${here("./astronaut.jpg")}").convert("RGB"); d = ImageDraw.Draw(im)
for cx, cy in ((203, 101), (246, 103)): d.ellipse([cx-17, cy-12, cx+17, cy+12], fill=(12,12,14))
d.line([220,102,229,102], fill=(12,12,14), width=4)
im.save("${OUT}sunglasses.jpg")
`]);

let browser, page, logs;
before(async () => { ({ browser, page, logs } = await openApp()); });
after(async () => { await browser?.close(); });

const counts = () => page.evaluate(() => {
  const c = document.getElementById("canvas");
  const d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
  let ok = 0, bad = 0;
  for (let i = 0; i < d.length; i += 4) {
    if (d[i] === 61 && d[i + 1] === 220 && d[i + 2] === 151) ok++;
    else if (d[i] === 255 && d[i + 1] === 77 && d[i + 2] === 94) bad++;
  }
  return { ok, bad };
});
const text = sel => page.$eval(sel, el => el.textContent);
const jsonObj = async () => JSON.parse(await text("#json"));

// Segmentador falso: pinta "acessório" num círculo em volta de cada olho (simula óculos).
const installFakeGlasses = () => page.evaluate(() => {
  window.__oclusao.setSegmenter({
    segment(_img, cb) {
      const lm = window.__oclusao.landmarks, W = 256, H = 256;
      const data = new Uint8Array(W * H).fill(3);
      const ctr = ids => ({ x: ids.reduce((s, i) => s + lm[i].x, 0) / ids.length, y: ids.reduce((s, i) => s + lm[i].y, 0) / ids.length });
      const eyes = [ctr([33, 133, 159, 145]), ctr([362, 263, 386, 374])];
      const r = Math.hypot(lm[33].x - lm[133].x, lm[33].y - lm[133].y) * 1.1;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++)
        if (eyes.some(e => Math.hypot((x + .5) / W - e.x, (y + .5) / H - e.y) < r)) data[y * W + x] = 5;
      cb({ categoryMask: { getAsUint8Array: () => data, width: W, height: H } });
    },
  });
});

test("carrega os modelos locais sem erros de página", async () => {
  assert.equal(await page.$eval("#spinner", el => el.hidden), true);
  assert.equal(await text("#emptyTitle"), "Tire uma foto do rosto");
  assert.equal(await page.$eval("#file", el => el.disabled), false);
  assert.equal(logs.filter(l => l.startsWith("[pageerror]")).length, 0, logs.join("\n"));
  // sem o arquivo do segmentador (e sem internet para o Google), o app avisa e segue
  const seg = await page.evaluate(() => window.__oclusao.segmenter);
  if (!seg) assert.match(await text("#notice"), /Segmentador indisponível/);
});

test("foto com rosto livre: JSON todo false e pontos verdes", async () => {
  await upload(page, here("./astronaut.jpg"));
  const j = await jsonObj();
  assert.equal(j.rosto_detectado, true);
  for (const k of ["olho_esquerdo", "olho_direito", "sobrancelhas", "nariz", "boca", "testa", "contorno"]) assert.equal(j[k], false, k);
  const c = await counts();
  assert.ok(c.ok > 1000, `pontos verdes: ${c.ok}`);
  assert.equal(c.bad, 0);
  await page.screenshot({ path: OUT + "1_livre.png" });
});

test("botão Landmarks oculta e mostra os pontos", async () => {
  await page.click("#toggle");
  assert.equal(await page.$eval("#toggle", b => b.getAttribute("aria-pressed")), "false");
  assert.equal((await counts()).ok, 0);
  await page.click("#toggle");
  assert.ok((await counts()).ok > 1000);
});

test("tocar no JSON alterna para a visão completa e volta", async () => {
  await page.click("#json");
  const full = await jsonObj();
  assert.deepEqual(Object.keys(full.boca), ["ocluso", "motivo", "score"]);
  await page.click("#json");
  assert.equal(typeof (await jsonObj()).boca, "boolean");
});

test("foto sem rosto: rosto_detectado false, regiões null e aviso", async () => {
  await upload(page, here("./coffee.jpg"));
  const j = await jsonObj();
  assert.equal(j.rosto_detectado, false);
  assert.equal(j.boca, null);
  assert.match(await text("#notice"), /Nenhum rosto/);
  assert.equal(await page.$eval("#toggle", b => b.disabled), true);
});

test("óculos escuros desenhados: o rosto continua sendo detectado", async () => {
  await upload(page, OUT + "sunglasses.jpg");
  assert.equal((await jsonObj()).rosto_detectado, true);
});

test("classificador de óculos (ONNX no navegador): escuros desenhados → \"escuros\"", async () => {
  // a foto anterior (sunglasses.jpg) já passou pelo classificador
  assert.equal((await jsonObj()).acessorios.oculos, "escuros");
  const g = await page.evaluate(() => window.__oclusao.oculos);
  assert.ok(g.escuros > 0.9 && g.grau < 0.1, JSON.stringify(g));
});

test("classificador de óculos: rosto sem óculos → false", async () => {
  await upload(page, here("./astronaut.jpg"));
  assert.equal((await jsonObj()).acessorios.oculos, false);
});

test("com segmentação indicando óculos: olhos true e pontos vermelhos", async () => {
  await installFakeGlasses();
  await upload(page, here("./astronaut.jpg"));
  const j = await jsonObj();
  assert.equal(j.olho_esquerdo, true);
  assert.equal(j.olho_direito, true);
  assert.equal(j.boca, false);
  assert.equal(j.segmentacao, undefined);
  const c = await counts();
  assert.ok(c.bad > 100, `pontos vermelhos: ${c.bad}`);
  assert.ok(c.ok > 500, `pontos verdes: ${c.ok}`);
  // cada região true aparece em vermelho; rosto_detectado true fica verde
  const red = await page.$$eval("#json .t", els => els.length);
  assert.equal(red, Object.entries(j).filter(([k, v]) => k !== "rosto_detectado" && v === true).length);
  for (const k of ["nariz", "testa", "contorno"]) assert.equal(j[k], false, k);
  await page.click("#json");
  const full = await jsonObj();
  assert.equal(full.olho_esquerdo.motivo, "acessorio");
  await page.click("#json");
  await page.screenshot({ path: OUT + "2_oculos.png" });
});

test("Usar como referência: salva, recalcula e persiste no navegador", async () => {
  assert.equal(await page.$eval("#ref", b => b.disabled), false);
  await page.click("#ref");
  assert.equal(await text("#ref"), "Referência salva");
  const j = await jsonObj();
  assert.equal(j.olho_esquerdo, false, "óculos da referência viram o normal");
  const saved = await page.evaluate(() => JSON.parse(localStorage.getItem("oclusao_baseline")).length);
  assert.equal(saved, 478);
  await page.evaluate(() => localStorage.removeItem("oclusao_baseline"));
});

test("layout de celular: sem rolagem lateral e botões dentro da tela", async () => {
  const m = await page.evaluate(() => ({
    sw: document.documentElement.scrollWidth, vw: innerWidth, vh: innerHeight,
    btns: [...document.querySelectorAll(".bar > *:not([hidden])")].map(b => b.getBoundingClientRect().toJSON()),
    cv: document.getElementById("canvas").getBoundingClientRect().toJSON(),
  }));
  assert.ok(m.sw <= m.vw);
  for (const b of m.btns) { assert.ok(b.left >= 0 && b.right <= m.vw && b.bottom <= m.vh, JSON.stringify(b)); assert.ok(b.height >= 44, `alvo de toque ${b.height}px`); }
  assert.ok(m.cv.width <= m.vw && m.cv.height <= m.vh);
});

test("aberto como arquivo (file://) explica que precisa de servidor", async () => {
  const p = await browser.newPage();
  await p.goto("file://" + here("../index.html"));
  await p.waitForFunction(() => document.getElementById("emptyTitle").textContent !== "Carregando modelos");
  assert.equal(await p.$eval("#emptyTitle", e => e.textContent), "Abra por um servidor");
  await p.close();
});

test("sem erros de página durante toda a sessão", () => {
  assert.equal(logs.filter(l => l.startsWith("[pageerror]")).length, 0, logs.join("\n"));
  assert.ok(BASE);
});
