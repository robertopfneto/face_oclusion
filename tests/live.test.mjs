// Modo ao vivo com a câmera falsa do Chromium (padrão sintético, sem rosto).
// Pré-requisito: servidor na pasta do projeto (npm run serve) em http://localhost:8765/
import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { openApp } from "./browser.mjs";

let browser, page, logs;
before(async () => {
  ({ browser, page, logs } = await openApp({
    args: ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"],
  }));
});
after(async () => { await browser?.close(); });

const mode = () => page.evaluate(() => window.__oclusao.mode);
const hidden = sel => page.$eval(sel, el => el.hidden);

test("Ao vivo: abre a webcam, analisa quadros e mostra Capturar", async () => {
  assert.equal(await hidden("#capture"), true);
  await page.click("#live");
  await page.waitForFunction(() => window.__oclusao.mode === "live" && window.__oclusao.state, { timeout: 30000 });
  assert.equal(await hidden("#capture"), false);
  assert.equal(await page.$eval("#live", b => b.textContent), "Parar");
  const j = JSON.parse(await page.$eval("#json", el => el.textContent));
  assert.equal(typeof j.rosto_detectado, "boolean");
});

test("Capturar: congela o quadro, para a webcam e vira foto", async () => {
  await page.click("#capture");
  await page.waitForFunction(() => window.__oclusao.mode === "photo");
  assert.equal(await hidden("#capture"), true);
  assert.equal(await page.$eval("#live", b => b.textContent), "Ao vivo");
  assert.equal(await page.evaluate(() => document.getElementById("video").srcObject), null);
});

test("Ao vivo de novo e Parar: volta para o modo foto", async () => {
  await page.click("#live");
  await page.waitForFunction(() => window.__oclusao.mode === "live");
  await page.click("#live");
  assert.equal(await mode(), "photo");
});

test("sem erros de página no modo ao vivo", () => {
  assert.equal(logs.filter(l => l.startsWith("[pageerror]")).length, 0, logs.join("\n"));
});
