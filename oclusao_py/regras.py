"""Lógica de oclusão facial: port 1:1 de occlusion.js (mesmos nomes, mesma ordem, mesmos limiares).

Sem MediaPipe e sem imagem: recebe landmarks normalizados (0..1) e, opcionalmente, a máscara de
classes do segmentador. A paridade com o JS é conferida por tests/python/test_paridade.py contra
tests/fixtures/golden.json (gerado de occlusion.js por tests/golden.mjs).

Formatos:
  lm      sequência de 478 pontos: objetos com .x/.y (MediaPipe) ou pares (x, y)
  mascara ndarray uint8 (altura, largura) ou (altura, largura, 1) com as classes CLS
  R       resultado de build_regions()
  cls     ndarray/lista com a classe lida por ponto (478) + a faixa do boné (21) no fim

Cuidados de paridade (não "simplifique"):
  - js_round imita Math.round do JS; o round() do Python arredonda .5 para o par.
  - top_vote percorre as classes em ordem numérica crescente, como o for...in do JS.
  - somas e médias seguem a mesma ordem dos índices do JS (o resultado bate até o último bit).
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

# ---------------------------------------------------------------- constantes

CLS = {"BG": 0, "HAIR": 1, "BODY": 2, "FACE": 3, "CLOTHES": 4, "ACC": 5}
MOTIVO = {0: "fundo", 1: "cabelo", 2: "mao_ou_pele", 4: "roupa", 5: "acessorio"}

DEFAULTS = {
    "regionOn": 0.40,     # fração de pontos fora da pele (acima da referência) para marcar a região
    "pointOn": 0.5,       # ponto isolado, fora das regiões
    "earClosed": 0.10,    # razão de abertura (EAR): abaixo disso = fechado pela geometria
    "earOpen": 0.20,      # EAR acima disso = aberto
    "blinkOpen": 0.35,    # blendshape eyeBlink: abaixo disso = aberto
    "blinkClosed": 0.70,  # eyeBlink acima disso = fechado
    "eyeRing": 0.6,       # olhos: amostra um anel ao redor do olho (lente e armação)
    "contourPull": 0.15,  # contorno: amostra um pouco para dentro do rosto
}

NUM_POINTS = 478
NOSE_TIP = 1

# Pontos para a razão de abertura. Esquerdo/direito do ponto de vista da pessoa.
EYE = {
    "olho_esquerdo": {"up": 386, "down": 374, "a": 362, "b": 263},
    "olho_direito": {"up": 159, "down": 145, "a": 33, "b": 133},
}

NOSE = [1, 2, 4, 5, 6, 19, 45, 48, 64, 94, 98, 168, 195, 197, 275, 278, 294, 327]
FOREHEAD = [9, 10, 67, 69, 104, 108, 109, 151, 297, 299, 333, 337, 338]

GLASSES_TYPES = ["grau", "escuros"]
GLASSES = {
    "photoOn": 0.65,  # foto: limiar simples
    "on": 0.75,       # ao vivo: liga acima disso...
    "off": 0.5,       # ...e só desliga abaixo disso (histerese)
    "alpha": 0.35,    # ao vivo: peso da leitura nova na média móvel
}

RULES = {
    "capRows": [0.10, 0.20, 0.30],  # faixa acima da testa, em fração da altura do rosto
    "capCols": 7,
    "capSpread": 0.8,               # largura da faixa, em fração da largura do rosto
    "capOn": 0.35,                  # chapéu ou boné
    "maskOn": 0.40,                 # máscara
    "minValid": 0.5,                # boné: mínimo da faixa dentro da imagem
    "hyst": 0.1,                    # ao vivo: liga em limiar + hyst, desliga em limiar - hyst
    "alpha": 0.5,                   # ao vivo: peso da leitura nova na média móvel
}
OUTSIDE = 255
COVER = {CLS["ACC"], CLS["CLOTHES"]}
LOWER_NOSE = [2, 94, 97, 98, 326, 327]
CLASS_NAME = {0: "fundo", 1: "cabelo", 2: "mao_ou_pele", 3: "rosto", 4: "roupa", 5: "acessorio", 255: "fora"}


# ---------------------------------------------------------------- utilitários

def js_round(x: float) -> float:
    """Math.round do JS (meio arredonda para cima)."""
    return math.floor(x + 0.5)


def r2(x: float) -> float:
    return js_round(x * 100) / 100


def r3(x: float) -> float:
    return js_round(x * 1000) / 1000


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _xy(lm) -> list[tuple[float, float]]:
    """Landmarks do MediaPipe (.x/.y) ou pares → lista de (x, y)."""
    if lm is None:
        return None
    return [(p.x, p.y) if hasattr(p, "x") else (float(p[0]), float(p[1])) for p in lm]


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _uniq(conns) -> list[int]:
    return list(dict.fromkeys(i for c in conns for i in (c.start, c.end)))


# ---------------------------------------------------------------- regiões e geometria

def build_regions(C=None) -> dict:
    """Regiões a partir das conexões do FaceLandmarker (mediapipe.tasks.python.vision.FaceLandmarksConnections)."""
    if C is None:
        from mediapipe.tasks.python.vision import FaceLandmarksConnections as C
    regions = {
        "olho_esquerdo": _uniq(C.FACE_LANDMARKS_LEFT_EYE),
        "olho_direito": _uniq(C.FACE_LANDMARKS_RIGHT_EYE),
        "sobrancelhas": list(dict.fromkeys(_uniq(C.FACE_LANDMARKS_LEFT_EYEBROW) + _uniq(C.FACE_LANDMARKS_RIGHT_EYEBROW))),
        "nariz": NOSE,
        "boca": _uniq(C.FACE_LANDMARKS_LIPS),
        "testa": FOREHEAD,
        "contorno": _uniq(C.FACE_LANDMARKS_FACE_OVAL),
    }
    keys = list(regions)
    point_region: list[str | None] = [None] * NUM_POINTS
    for k in keys:
        for i in regions[k]:
            if point_region[i] is None:
                point_region[i] = k
    return {"regions": regions, "keys": keys, "pointRegion": point_region}


def eye_aspect_ratio(lm, eye) -> float:
    p = _xy(lm)
    d = lambda a, b: math.hypot(p[a][0] - p[b][0], p[a][1] - p[b][1])
    return d(eye["up"], eye["down"]) / max(1e-6, d(eye["a"], eye["b"]))


def eye_closed_score(ear: float, cfg=DEFAULTS, blink: float | None = None) -> float:
    """Olho fechado (0..1). Com a piscada do MediaPipe, as duas evidências precisam concordar (mínimo)."""
    geo = clamp01((cfg["earOpen"] - ear) / (cfg["earOpen"] - cfg["earClosed"]))
    if not _num(blink):
        return geo
    return min(geo, clamp01((blink - cfg["blinkOpen"]) / (cfg["blinkClosed"] - cfg["blinkOpen"])))


def confidence(score, thr):
    """Distância do score ao limiar (0 = em cima do limiar, 1 = no extremo)."""
    if not _num(score) or not _num(thr):
        return None
    return clamp01((score - thr) / (1 - thr) if score > thr else (thr - score) / thr)


def conf_level(c):
    return None if c is None else "alta" if c >= 0.6 else "média" if c >= 0.3 else "baixa"


def sample_points(lm, R, cfg=DEFAULTS) -> list[tuple[float, float]]:
    """Onde cada landmark lê a máscara: anel nos olhos, contorno puxado para dentro."""
    p = _xy(lm)
    center = {}
    for k in EYE:
        idx = R["regions"][k]
        sx = sy = 0.0
        for i in idx:
            sx += p[i][0]
        for i in idx:
            sy += p[i][1]
        center[k] = (sx / len(idx), sy / len(idx))
    nx, ny = p[NOSE_TIP]
    out = []
    for i, (x, y) in enumerate(p):
        k = R["pointRegion"][i] if i < len(R["pointRegion"]) else None
        if k == "contorno":
            x += (nx - x) * cfg["contourPull"]
            y += (ny - y) * cfg["contourPull"]
        elif k in center:
            cx, cy = center[k]
            x += (x - cx) * cfg["eyeRing"]
            y += (y - cy) * cfg["eyeRing"]
        out.append((x, y))
    return out


def _mask2d(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask)
    return m[..., 0] if m.ndim == 3 else m


def class_at(mask: np.ndarray, x: float, y: float) -> int:
    m = _mask2d(mask)
    h, w = m.shape
    px = min(w - 1, max(0, js_round(x * w)))
    py = min(h - 1, max(0, js_round(y * h)))
    return int(m[py, px])


def _top_vote(votes: dict[int, int]):
    best, n = None, 0
    for c in sorted(votes):  # o for...in do JS percorre chaves numéricas em ordem crescente
        if votes[c] > n:
            n, best = votes[c], c
    return None if best is None else MOTIVO.get(best, "outro")


# ---------------------------------------------------------------- análise das regiões

def cap_points(lm, cfg=RULES) -> list[tuple[float, float]]:
    """Faixa acima da testa (copa do boné). Acompanha a inclinação da cabeça."""
    p = _xy(lm)
    top, chin, l, r = p[10], p[152], p[234], p[454]
    ux, uy, ax, ay = top[0] - chin[0], top[1] - chin[1], r[0] - l[0], r[1] - l[1]
    pts = []
    for k in cfg["capRows"]:
        for c in range(cfg["capCols"]):
            t = (c / (cfg["capCols"] - 1) - 0.5) * cfg["capSpread"]
            pts.append((top[0] + ux * k + ax * t, top[1] + uy * k + ay * t))
    return pts


def sample_cap_classes(lm, mask, cfg=RULES) -> list[int]:
    return [OUTSIDE if (x < 0 or x > 1 or y < 0 or y > 1) else class_at(mask, x, y) for x, y in cap_points(lm, cfg)]


def sample_classes(lm, mask, R, cfg=DEFAULTS) -> np.ndarray:
    """Classe lida por landmark (478), seguida das classes da faixa do boné."""
    pts = sample_points(lm, R, cfg)
    cap = sample_cap_classes(lm, mask)
    return np.array([class_at(mask, x, y) for x, y in pts] + cap, dtype=np.uint8)


def analyze_classes(lm, cls, R, baseline=None, cfg=DEFAULTS, blink: dict | None = None) -> dict:
    """Estado de cada região. cls=None → sem segmentador (só olho fechado).
    blink: {"olho_esquerdo": x, "olho_direito": y} com o eyeBlink do MediaPipe (opcional).
    Retorna {"state", "nonSkin", "red"} como o JS."""
    if lm is None or len(lm) < NUM_POINTS:
        return {"state": {"rosto_detectado": False}, "nonSkin": None, "red": None}
    p = _xy(lm)
    n = len(p)
    base = np.zeros(NUM_POINTS, dtype=np.float32) if baseline is None else np.asarray(baseline, dtype=np.float32)
    bl = lambda i: float(base[i]) if i < len(base) else 0.0
    has_seg = cls is not None
    cls = list(cls[:n]) if has_seg else [CLS["FACE"]] * n
    non_skin = [0.0 if (not has_seg or cls[i] == CLS["FACE"]) else 1.0 for i in range(n)]

    state: dict[str, Any] = {"rosto_detectado": True}
    if not has_seg:
        state["segmentacao"] = False

    for k in R["keys"]:
        idx = R["regions"][k]
        total, votes = 0.0, {}
        for i in idx:
            v = max(0.0, non_skin[i] - bl(i))
            total += v
            if v > 0.5:
                votes[int(cls[i])] = votes.get(int(cls[i]), 0) + 1
        score = total / len(idx)
        motivo = _top_vote(votes)
        eye = {}
        if k in EYE:
            ear = eye_aspect_ratio(p, EYE[k])
            b = blink.get(k) if blink else None
            closed = eye_closed_score(ear, cfg, b)
            if closed > score:
                score, motivo = closed, "fechado"
            eye = {"abertura": r3(ear), "piscada": r2(b) if _num(b) else None}
        ocluso = score > cfg["regionOn"]
        state[k] = {
            "ocluso": ocluso, "motivo": motivo if ocluso else None, "score": r2(score),
            "confianca": r2(confidence(score, cfg["regionOn"])), **eye,
        }

    red = []
    for i in range(n):
        k = R["pointRegion"][i]
        red.append((1 if state[k]["ocluso"] else 0) if k else (1 if (non_skin[i] - bl(i)) > cfg["pointOn"] else 0))
    return {"state": state, "nonSkin": np.array(non_skin, dtype=np.float32), "red": np.array(red, dtype=np.uint8)}


def analyze(lm, mask, R, baseline=None, cfg=DEFAULTS, blink=None) -> dict:
    ok = lm is not None and len(lm) >= NUM_POINTS
    return analyze_classes(lm, sample_classes(lm, mask, R, cfg) if ok and mask is not None else None, R, baseline, cfg, blink)


def to_view(state, keys, full: bool) -> dict:
    """JSON exibido: compacto (booleanos) ou completo. Sem rosto → regiões None."""
    o = {"rosto_detectado": bool(state and state.get("rosto_detectado"))}
    if state and state.get("segmentacao") is False:
        o["segmentacao"] = False
    for k in keys:
        o[k] = None if not o["rosto_detectado"] else (state[k] if full else state[k]["ocluso"])
    return o


# ---------------------------------------------------------------- óculos

def glasses_state(scores: dict, prev: dict | None = None, cfg=GLASSES) -> dict:
    """Scores do classificador → estado. Sem prev (foto): limiar simples. Com prev (ao vivo): média móvel + histerese."""
    s = {t: (prev[t] + cfg["alpha"] * (scores[t] - prev[t]) if prev else scores[t]) for t in GLASSES_TYPES}

    def best(lim):
        acima = [t for t in GLASSES_TYPES if s[t] > lim]
        return sorted(acima, key=lambda t: -s[t])[0] if acima else None  # sorted é estável, como o sort do JS

    if not prev:
        tipo = best(cfg["photoOn"])
    else:
        cand = best(cfg["on"])
        keep = prev["tipo"] if prev.get("tipo") and s[prev["tipo"]] > cfg["off"] else None
        tipo = cand if cand and (not keep or s[cand] > s[keep]) else keep
    return {"presente": tipo is not None, "tipo": tipo, "grau": s["grau"], "escuros": s["escuros"],
            "bruto": {"grau": scores["grau"], "escuros": scores["escuros"]}}


# ---------------------------------------------------------------- chapéu/boné e máscara

def lower_face(lm, R) -> list[int]:
    """Lábios, base do nariz e contorno abaixo do nariz (onde fica a máscara)."""
    p = _xy(lm)
    top, chin, nose = p[10], p[152], p[NOSE_TIP]
    ux, uy = top[0] - chin[0], top[1] - chin[1]
    below = lambda i: (p[i][0] - nose[0]) * ux + (p[i][1] - nose[1]) * uy < 0
    return list(dict.fromkeys(R["regions"]["boca"] + LOWER_NOSE + [i for i in R["regions"]["contorno"] if below(i)]))


def _histogram(classes) -> dict[str, int]:
    h: dict[str, int] = {}
    for c in classes:
        k = CLASS_NAME.get(int(c), "outro")
        h[k] = h.get(k, 0) + 1
    return h


def accessory_rules(lm, cls, R, baseline=None, cfg=RULES) -> dict | None:
    """Chapéu/boné (faixa acima da testa) e máscara (metade de baixo) em acessório/roupa."""
    if lm is None or cls is None or len(cls) <= len(lm):
        return None
    n = len(lm)
    base = np.zeros(NUM_POINTS, dtype=np.float32) if baseline is None else np.asarray(baseline, dtype=np.float32)
    cap = [int(c) for c in cls[n:]]
    valid = [c for c in cap if c != OUTSIDE]
    chapeu = {
        "score": None if len(valid) < cfg["minValid"] * len(cap) else sum(1 for c in valid if c in COVER) / len(valid),
        "classes": _histogram(cap),
    }
    idx = lower_face(lm, R)
    mascara = {
        "score": sum(1 for i in idx if int(cls[i]) in COVER and not (float(base[i]) >= 0.5)) / len(idx),
        "classes": _histogram(int(cls[i]) for i in idx),
    }
    return {"chapeu_ou_bone": chapeu, "mascara": mascara}


def binary_state(score, thr: float, prev: dict | None = None, cfg=RULES) -> dict:
    """Score → {presente, score}. Sem prev: limiar simples. Com prev: média móvel + histerese.
    score None → {presente: False, score: None, fora: True} (faixa do boné fora da imagem)."""
    if score is None:
        return {"presente": False, "score": None, "fora": True}
    live = bool(prev) and prev.get("score") is not None
    s = prev["score"] + cfg["alpha"] * (score - prev["score"]) if live else score
    if not live:
        presente = s > thr
    elif prev["presente"]:
        presente = s > thr - cfg["hyst"]
    else:
        presente = s > thr + cfg["hyst"]
    return {"presente": presente, "score": s}


# ---------------------------------------------------------------- etiquetas

def detected_tags(acessorios) -> list[dict]:
    """Só os acessórios detectados, com o texto que o app mostra na tela.
    acessorios: a forma completa ({"chapeu_ou_bone": {...}, "oculos": {...}, "mascara": {...}}).
    Itens ainda sem decisão ("analisando", "fora do quadro") não viram etiqueta."""
    if not isinstance(acessorios, dict):
        return []
    presente = lambda v: isinstance(v, dict) and bool(v.get("presente"))
    c, g, m = acessorios.get("chapeu_ou_bone"), acessorios.get("oculos"), acessorios.get("mascara")
    tags = []
    if presente(c):
        tags.append({"key": "chapeu_ou_bone", "text": "Chapéu ou boné", "confianca": c.get("confianca")})
    if presente(g):
        tags.append({"key": "oculos", "text": "Óculos escuros" if g.get("tipo") == "escuros" else "Óculos de grau",
                     "confianca": g.get("confianca")})
    if presente(m):
        tags.append({"key": "mascara", "text": "Máscara", "confianca": m.get("confianca")})
    return tags
