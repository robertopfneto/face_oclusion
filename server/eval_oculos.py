"""Avalia o classificador de óculos em fotos reais e sugere recorte e limiares.

    server/.venv/bin/python server/eval_oculos.py [pasta] [--farl]

Estrutura da pasta (padrão tests/fixtures/oculos/), uma subpasta por caso:
    grau/      óculos de grau          → esperado "grau"
    escuros/   óculos escuros          → esperado "escuros"
    qualquer outra (sem/, testa/, franja/, olho_fechado/, ...) → esperado sem óculos

Para cada variação de recorte, mostra acerto do tipo, precisão/recall de cada classe no limiar
atual e o limiar que maximiza o F1. Com --farl compara com o atributo Eyeglasses do FaRL
(só presença, sem tipo). Os landmarks vêm do mesmo face_landmarker.task do app.
"""
import sys
from pathlib import Path

import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks.python import BaseOptions, vision
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).parent))
import oculos  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
args = [a for a in sys.argv[1:] if not a.startswith("--")]
PASTA = Path(args[0]) if args else ROOT / "tests/fixtures/oculos"
FARL = "--farl" in sys.argv
EXT = {".jpg", ".jpeg", ".png", ".webp"}
OLHOS = [33, 133, 159, 145, 362, 263, 386, 374, 70, 300]  # cantos, pálpebras e sobrancelhas


def esperado(pasta: str):
    return pasta if pasta in oculos.TIPOS else None


def caixa(pts, idx=None):
    sel = pts if idx is None else pts[idx]
    return [sel[:, 0].min(), sel[:, 1].min(), sel[:, 0].max(), sel[:, 1].max()]


VARIANTES = {  # nome → (pontos, margem) do recorte quadrado
    "rosto_15": (None, 0.15),   # o padrão atual (app.js e oculos.py)
    "rosto_30": (None, 0.30),
    "olhos":    (OLHOS, 0.35),
}


def main():
    fotos = sorted(p for p in PASTA.rglob("*") if p.suffix.lower() in EXT)
    if not fotos:
        sys.exit(f"nenhuma foto em {PASTA}. Crie subpastas grau/, escuros/, sem/ ... (ver docstring)")

    model = oculos.Combinado().eval()
    farl = None
    if FARL:
        import app  # carrega o FaRL do servidor (e os outros modelos)
        farl = app.attr_model
        farl_idx = farl.labels.index("Eyeglasses")

    opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ROOT / "models/face_landmarker.task")), num_faces=1)
    linhas, sem_rosto = [], []
    with vision.FaceLandmarker.create_from_options(opts) as lmk, torch.inference_mode():
        for p in fotos:
            img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
            r = lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(img).copy()))
            if not r.face_landmarks:
                sem_rosto.append(p)
                continue
            pts = np.array([[q.x * img.width, q.y * img.height] for q in r.face_landmarks[0]])
            linha = {"foto": p, "esperado": esperado(p.parent.name)}
            for nome, (idx, margem) in VARIANTES.items():
                crop = oculos.recorte(img, caixa(pts, idx), margem)
                linha[nome] = dict(zip(oculos.TIPOS, model(oculos.pixels(crop))[0].tolist()))
            if farl is not None:
                P = lambda i: pts[i]
                cinco = np.array(sorted([P(468), P(473)], key=lambda q: q[0]) + [P(1)]
                                 + sorted([P(61), P(291)], key=lambda q: q[0]), dtype=np.float32)
                arr = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).unsqueeze(0)
                data = {"points": torch.from_numpy(cinco).unsqueeze(0), "image_ids": torch.tensor([0])}
                linha["farl"] = float(farl(arr, data)["attrs"][0][farl_idx])
            linhas.append(linha)

    n = len(linhas)
    casos = {}
    for l in linhas:
        casos[l["foto"].parent.name] = casos.get(l["foto"].parent.name, 0) + 1
    print(f"{n} fotos com rosto ({', '.join(f'{k}: {v}' for k, v in sorted(casos.items()))})")
    if sem_rosto:
        print(f"{len(sem_rosto)} sem rosto detectado (ignoradas): {', '.join(p.name for p in sem_rosto)}")

    for nome in VARIANTES:
        tipo = lambda s: oculos.resultado(s)["tipo"]
        acerto = sum(tipo(l[nome]) == l["esperado"] for l in linhas) / n
        print(f"\n== {nome}: acerto do tipo {acerto:.0%} (limiar {oculos.LIMIAR})")
        for t in oculos.TIPOS:
            y = np.array([l["esperado"] == t for l in linhas])
            s = np.array([l[nome][t] for l in linhas])
            print(f"   {t:8s} {prf(y, s > oculos.LIMIAR)}   melhor limiar: {melhor_limiar(y, s)}")
        erros = [l for l in linhas if tipo(l[nome]) != l["esperado"]]
        for l in erros[:8]:
            print(f"   erro: {l['foto'].parent.name}/{l['foto'].name} → {tipo(l[nome])} "
                  f"(grau={l[nome]['grau']:.2f} escuros={l[nome]['escuros']:.2f})")
        if len(erros) > 8:
            print(f"   ... e mais {len(erros) - 8}")

    if farl is not None:
        y = np.array([l["esperado"] is not None for l in linhas])
        s = np.array([l["farl"] for l in linhas])
        pres = np.array([max(l["rosto_15"].values()) for l in linhas])
        print("\n== presença de óculos (qualquer tipo)")
        print(f"   FaRL Eyeglasses      {prf(y, s > 0.5)}   melhor limiar: {melhor_limiar(y, s)}")
        print(f"   glasses-detector     {prf(y, pres > oculos.LIMIAR)}   melhor limiar: {melhor_limiar(y, pres)}")


def prf(y, pred):
    tp, fp, fn = int((y & pred).sum()), int((~y & pred).sum()), int((y & ~pred).sum())
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if p + r else float("nan")
    return f"precisão={p:.2f} recall={r:.2f} F1={f:.2f} (vp={tp} fp={fp} fn={fn})"


def melhor_limiar(y, s):
    if y.all() or not y.any():
        return "precisa de exemplos com e sem"
    f1 = lambda t: (lambda tp, fp, fn: 2 * tp / (2 * tp + fp + fn) if tp else 0)(
        (y & (s > t)).sum(), (~y & (s > t)).sum(), (y & ~(s > t)).sum())
    t = max(np.arange(0.05, 0.96, 0.05), key=f1)
    return f"{t:.2f} (F1={f1(t):.2f})"


if __name__ == "__main__":
    main()
