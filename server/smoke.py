"""Teste de fumaça da API de acessórios: foto do astronauta + versões com acessórios desenhados.

    server/.venv/bin/python server/smoke.py [http://localhost:8000/api/]

Os acessórios desenhados são grosseiros (formas chapadas): servem para ver se a
rota funciona e se os scores reagem, não para medir a qualidade dos modelos.
"""
import io
import json
import sys
import urllib.request
import uuid
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
API = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/api/"
img = Image.open(ROOT / "tests/astronaut.jpg").convert("RGB")
W, H = img.size
lm = [(p["x"] * W, p["y"] * H) for p in json.load(open(ROOT / "tests/fixtures/astronaut_landmarks.json"))]
P = lambda i: lm[i]
points = sorted([P(468), P(473)]) + [P(1)] + sorted([P(61), P(291)])
xs, ys = [p[0] for p in lm], [p[1] for p in lm]
box = [min(xs), min(ys), max(xs), max(ys)]
OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
        152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


def oculos(im):
    d = ImageDraw.Draw(im)
    for c in (P(468), P(473)):
        d.ellipse([c[0] - 17, c[1] - 12, c[0] + 17, c[1] + 12], fill=(12, 12, 14))
    d.line([P(468), P(473)], fill=(12, 12, 14), width=4)


def grau(im):
    # armação fina, lente transparente
    d = ImageDraw.Draw(im)
    for c in (P(468), P(473)):
        d.rounded_rectangle([c[0] - 19, c[1] - 12, c[0] + 19, c[1] + 12], radius=6, outline=(25, 20, 18), width=3)
    (a, b) = sorted([P(468), P(473)])
    d.line([(a[0] + 19, a[1] - 3), (b[0] - 19, b[1] - 3)], fill=(25, 20, 18), width=3)
    for c, s in ((a, -1), (b, 1)):
        d.line([(c[0] + s * 19, c[1] - 6), (c[0] + s * 40, c[1] - 8)], fill=(25, 20, 18), width=3)


def mascara(im):
    # metade de baixo do oval do rosto, a partir da altura do nariz
    y0 = P(195)[1]
    poly = [(x, max(y, y0)) for x, y in (P(i) for i in OVAL)]
    ImageDraw.Draw(im).polygon(poly, fill=(170, 200, 225))


def chapeu_ou_bone(im):
    # copa acima da testa + aba sobre a testa
    top, (xl, _), (xr, _) = P(10), P(234), P(454)
    h = box[3] - box[1]
    d = ImageDraw.Draw(im)
    d.chord([xl - 8, top[1] - 0.55 * h, xr + 8, top[1] + 0.25 * h], 180, 360, fill=(160, 25, 30))
    d.rectangle([xl - 8, top[1] - 0.05 * h, xr + 30, top[1] + 0.06 * h], fill=(130, 20, 25))


def post(im):
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    b = uuid.uuid4().hex
    parts = [(f'--{b}\r\nContent-Disposition: form-data; name="image"; filename="f.jpg"\r\n'
              "Content-Type: image/jpeg\r\n\r\n").encode() + buf.getvalue() + b"\r\n"]
    for k, v in (("points", points), ("box", box)):
        parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{json.dumps(v)}\r\n'.encode())
    body = b"".join(parts) + f"--{b}--\r\n".encode()
    req = urllib.request.Request(API + "acessorios", body, {"Content-Type": f"multipart/form-data; boundary={b}"})
    return json.load(urllib.request.urlopen(req, timeout=60))


out = ROOT / "server/out"
out.mkdir(exist_ok=True)
for nome, fn in (("livre", None), ("oculos", oculos), ("grau", grau), ("mascara", mascara), ("chapeu_ou_bone", chapeu_ou_bone)):
    im = img.copy()
    if fn:
        fn(im)
    im.save(out / f"{nome}.jpg")
    r = post(im)
    o = r["oculos"]
    print(f"{nome:8s}", "  ".join(f"{k}={r[k]['score']:.2f}{'*' if r[k]['presente'] else ' '}" for k in ("chapeu_ou_bone", "mascara")),
          f" oculos={o['tipo'] or '-':7s} (grau={o['grau']:.2f} escuros={o['escuros']:.2f})  {r['ms']}ms")
