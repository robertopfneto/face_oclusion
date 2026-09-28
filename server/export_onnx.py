"""Exporta os classificadores de óculos para models/oculos.onnx (usado no navegador) e confere a paridade.

    server/.venv/bin/python server/export_onnx.py

Confere três coisas nas imagens de server/out/ (geradas por smoke.py) e na foto do astronauta:
  1. Combinado (PyTorch) == GlassesClassifier.predict original  → a normalização embutida está certa
  2. ONNX (onnxruntime) == Combinado                            → a exportação está certa
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from glasses_detector import GlassesClassifier
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import oculos  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "models/oculos.onnx"
TOL = 1e-4

model = oculos.Combinado().eval()
torch.onnx.export(
    model, oculos.pixels(Image.new("RGB", (256, 256))), OUT,
    input_names=["pixels"], output_names=["probs"],
    dynamic_axes={"pixels": {0: "n"}, "probs": {0: "n"}},
    opset_version=17, dynamo=False,
)
print(f"{OUT.relative_to(ROOT)}: {OUT.stat().st_size / 1e6:.1f} MB")

# recortes de teste: astronauta (+ variações de smoke.py, se existirem) usando os landmarks do fixture
lm = json.load(open(ROOT / "tests/fixtures/astronaut_landmarks.json"))
imgs = [ROOT / "tests/astronaut.jpg", *sorted((ROOT / "server/out").glob("*.jpg"))]
orig = {t: GlassesClassifier(size="medium", kind=k, device="cpu") for t, k in oculos.KINDS.items()}
sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])

pior = 0.0
for p in imgs:
    img = Image.open(p).convert("RGB")
    W, H = img.size
    xs, ys = [q["x"] * W for q in lm], [q["y"] * H for q in lm]
    crop = oculos.recorte(img, [min(xs), min(ys), max(xs), max(ys)])
    x = oculos.pixels(crop)
    with torch.inference_mode():
        pt = model(x)[0].numpy()
    ref = np.array([orig[t].predict(crop, format="proba") for t in oculos.TIPOS])
    onx = sess.run(None, {"pixels": x.numpy()})[0][0]
    d1, d2 = np.abs(pt - ref).max(), np.abs(onx - pt).max()
    pior = max(pior, d1, d2)
    print(f"{p.name:16s} grau={onx[0]:.2f} escuros={onx[1]:.2f}   |pt-original|={d1:.1e} |onnx-pt|={d2:.1e}")

print("paridade OK" if pior < TOL else f"PARIDADE FALHOU: diferença {pior:.1e} > {TOL}")
sys.exit(0 if pior < TOL else 1)
