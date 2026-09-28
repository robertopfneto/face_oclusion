"""Óculos com o glasses-detector (github.com/mantasu/glasses-detector, MIT).

Dois classificadores ShuffleNet v2 (5 MB cada): "eyeglasses" (grau) e "sunglasses" (escuros).
Entrada: recorte quadrado do rosto, redimensionado para 256×256; normalização ImageNet; saída logit → sigmoid.

O mesmo recorte é feito no navegador (app.js, cropSquare) para a versão ONNX dar o mesmo resultado.
"""
import numpy as np
import torch
from glasses_detector import GlassesClassifier
from PIL import Image

MARGEM = 0.15   # recorte: lado = maior lado da caixa dos landmarks × (1 + 2 × MARGEM)
LIMIAR = 0.5
TAMANHO = 256
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
TIPOS = ("grau", "escuros")
KINDS = {"grau": "eyeglasses", "escuros": "sunglasses"}


def caixa_quadrada(box, margem=MARGEM):
    """[x1, y1, x2, y2] → caixa quadrada centrada, com margem. Pode sair da imagem (fora vira preto)."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    lado = max(x2 - x1, y2 - y1) * (1 + 2 * margem)
    return cx - lado / 2, cy - lado / 2, cx + lado / 2, cy + lado / 2


def recorte(img: Image.Image, box, margem=MARGEM) -> Image.Image:
    return img.crop(tuple(round(v) for v in caixa_quadrada(box, margem))).resize((TAMANHO, TAMANHO), Image.BICUBIC)


def resultado(scores: dict) -> dict:
    """{"grau": p, "escuros": p} → { presente, tipo, grau, escuros }. tipo = o maior acima do limiar."""
    acima = [t for t in TIPOS if scores[t] > LIMIAR]
    tipo = max(acima, key=scores.get) if acima else None
    return {"presente": tipo is not None, "tipo": tipo, **{t: round(float(scores[t]), 2) for t in TIPOS}}


class Combinado(torch.nn.Module):
    """Os dois classificadores num só grafo: uint8 [N, 256, 256, 3] (RGB) → probabilidades [N, 2] (grau, escuros).
    A normalização fica dentro do modelo, para o navegador só mandar os pixels."""

    def __init__(self, device="cpu"):
        super().__init__()
        self.nets = torch.nn.ModuleList(
            GlassesClassifier(size="medium", kind=KINDS[t], device=device).model.eval() for t in TIPOS)
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1) * 255)

    def forward(self, x):
        x = (x.permute(0, 3, 1, 2).float() - self.mean) / self.std
        return torch.cat([net(x) for net in self.nets], dim=1).sigmoid()


def pixels(crop: Image.Image) -> torch.Tensor:
    """PIL → uint8 [1, 256, 256, 3], a entrada de Combinado."""
    return torch.from_numpy(np.asarray(crop.convert("RGB")).copy()).unsqueeze(0)
