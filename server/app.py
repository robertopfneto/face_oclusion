"""Protótipo: classificador de acessórios (boné, óculos, máscara) em PyTorch.

- boné: facer / FaRL treinado no CelebA (atributo Wearing_Hat)
- óculos: glasses-detector (grau e escuros, ver oculos.py); o navegador roda o mesmo modelo em ONNX
- máscara: prithivMLmods/Face-Mask-Detection (SigLIP2, Apache 2.0)

O navegador já tem os landmarks do MediaPipe, então manda a imagem + 5 pontos
(para o alinhamento do FaRL) + a caixa do rosto (para o recorte da máscara).
Também serve o app estático, para ficar tudo na mesma origem.

    server/.venv/bin/uvicorn server.app:app --port 8000     (na raiz do projeto)
"""
import io
import json
import mimetypes
import threading
import time
from pathlib import Path

import sys

import facer
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification

sys.path.insert(0, str(Path(__file__).parent))
import oculos  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEVICE = "cpu"
MASK_MODEL = "prithivMLmods/Face-Mask-Detection"
LIMIAR = {"chapeu_ou_bone": 0.5, "mascara": 0.5}
MARGEM_MASCARA = 0.15  # recorte da máscara: caixa dos landmarks + 15% de cada lado

mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("text/javascript", ".mjs")

torch.set_grad_enabled(False)


def _face_attr():
    # facer chama model.cuda() fixo e torch.load sem map_location; sem CUDA os dois quebram.
    # Ajusta os dois só durante a criação do modelo.
    cuda, load = torch.nn.Module.cuda, torch.load
    if DEVICE == "cpu":
        torch.nn.Module.cuda = lambda self, device=None: self
        torch.load = lambda f, map_location=None, **kw: load(f, map_location=map_location or "cpu", **kw)
    try:
        return facer.face_attr("farl/celeba/224", device=DEVICE)
    finally:
        torch.nn.Module.cuda, torch.load = cuda, load


attr_model = _face_attr()
ATTR_IDX = {"chapeu_ou_bone": attr_model.labels.index("Wearing_Hat")}
oculos_model = oculos.Combinado(DEVICE).eval()
mask_proc = AutoImageProcessor.from_pretrained(MASK_MODEL)
mask_model = SiglipForImageClassification.from_pretrained(MASK_MODEL).eval()
MASK_IDX = next(i for i, name in mask_model.config.id2label.items() if "Not" not in name)
lock = threading.Lock()  # uma inferência por vez; em CPU paralelizar só disputa núcleos

app = FastAPI(title="Oclusão: acessórios")
# permite usar a API a partir do app servido em outra porta (?api=http://host:8000/api/)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def resultado(k, p):
    return {"presente": bool(p > LIMIAR[k]), "score": round(float(p), 2)}


@app.get("/api/health")
def health():
    return {"ok": True, "modelos": ["farl/celeba/224", "glasses-detector/medium", MASK_MODEL]}


@app.post("/api/acessorios")
def acessorios(image: UploadFile = File(...), points: str = Form(...), box: str = Form(...)):
    """points: 5 pares [x,y] em pixels (olho, olho, nariz, boca, boca; da esquerda p/ direita da imagem).
    box: [x1, y1, x2, y2] em pixels, a caixa dos landmarks."""
    try:
        img = Image.open(io.BytesIO(image.file.read())).convert("RGB")
        pts = np.asarray(json.loads(points), dtype=np.float32).reshape(5, 2)
        x1, y1, x2, y2 = (float(v) for v in json.loads(box))
    except Exception as e:
        raise HTTPException(400, f"entrada inválida: {e}")

    t0 = time.perf_counter()
    arr = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).unsqueeze(0)  # 1×3×H×W uint8
    mx, my = (x2 - x1) * MARGEM_MASCARA, (y2 - y1) * MARGEM_MASCARA
    crop = img.crop((max(0, x1 - mx), max(0, y1 - my), min(img.width, x2 + mx), min(img.height, y2 + my)))

    with lock, torch.inference_mode():
        data = {"points": torch.from_numpy(pts).unsqueeze(0), "image_ids": torch.tensor([0])}
        attrs = attr_model(arr, data)["attrs"][0]
        logits = mask_model(**mask_proc(images=crop, return_tensors="pt")).logits[0]
        p_mask = torch.softmax(logits, -1)[MASK_IDX]
        p_oculos = oculos_model(oculos.pixels(oculos.recorte(img, [x1, y1, x2, y2])))[0]

    out = {k: resultado(k, attrs[i]) for k, i in ATTR_IDX.items()}
    out["oculos"] = oculos.resultado(dict(zip(oculos.TIPOS, p_oculos.tolist())))
    out["mascara"] = resultado("mascara", p_mask)
    out["ms"] = round((time.perf_counter() - t0) * 1000)
    return out


# app estático (mesma origem que a API)
for name in ("index.html", "app.js", "occlusion.js", "seg-worker.js", "oculos-worker.js"):
    app.add_api_route("/" + ("" if name == "index.html" else name),
                      (lambda p=ROOT / name: FileResponse(p)), methods=["GET"], include_in_schema=False)
for d in ("vendor", "models", "tests"):
    app.mount(f"/{d}", StaticFiles(directory=ROOT / d), name=d)
