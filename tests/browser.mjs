// Utilitário comum aos testes de navegador: abre o app num Chromium headless.
import chromium from "@sparticuz/chromium";
import pp from "puppeteer-core";

// ?cpu: o delegate GPU trava no Chromium headless (GPU emulada em software)
export const BASE = process.env.APP_URL || "http://localhost:8765/?cpu";

export async function openApp({ mobile = true, args = [] } = {}) {
  const browser = await pp.launch({ executablePath: await chromium.executablePath(), args: [...chromium.args, ...args], headless: true });
  const page = await browser.newPage();
  if (mobile) await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
  const logs = [];
  page.on("console", m => logs.push(`[${m.type()}] ${m.text()}`));
  page.on("pageerror", e => logs.push(`[pageerror] ${e.message}`));
  await page.goto(BASE, { waitUntil: "load" });
  await page.waitForFunction(() => window.__oclusao, { timeout: 30000 });
  await page.evaluate(() => window.__oclusao.ready);
  return { browser, page, logs };
}

export async function upload(page, file) {
  const input = await page.$("#file");
  await input.uploadFile(file);
  await page.waitForFunction(() => !document.getElementById("canvas").hidden && window.__oclusao.state, { timeout: 30000 });
  await page.waitForFunction(() => !document.getElementById("shot").classList.contains("busy"));
}
