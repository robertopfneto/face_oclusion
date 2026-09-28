// Gera fixtures de landmarks reais a partir das fotos de teste.
import { writeFileSync } from "node:fs";
import { openApp, upload } from "./browser.mjs";
const { browser, page, logs } = await openApp();
console.log("segmentador:", await page.evaluate(() => window.__oclusao.segmenter));
for (const name of ["astronaut", "coffee"]) {
  await upload(page, new URL(`./${name}.jpg`, import.meta.url).pathname);
  const lm = await page.evaluate(() => window.__oclusao.landmarks && window.__oclusao.landmarks.map(p => ({ x: p.x, y: p.y })));
  console.log(name, lm ? lm.length : null, JSON.stringify(await page.evaluate(() => window.__oclusao.state)));
  if (lm) writeFileSync(new URL(`./fixtures/${name}_landmarks.json`, import.meta.url), JSON.stringify(lm));
}
console.log(logs.filter(l => !l.includes("INFO")).join("\n"));
await browser.close();
