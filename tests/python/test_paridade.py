"""Paridade do port Python (oclusao_py/regras.py) com occlusion.js.

    server/.venv/bin/python -m pytest tests/python -q

As respostas esperadas vêm de tests/fixtures/golden.json, gerado de occlusion.js por
`node tests/golden.mjs`. Mudou occlusion.js? Regere o golden e rode este teste.
Classes, booleanos e textos precisam ser iguais; números, iguais até 1e-12.
"""
import base64
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from oclusao_py import regras as P  # noqa: E402

G = json.loads((ROOT / "tests/fixtures/golden.json").read_text())
TOL = 1e-12


def same(a, b, path="$"):
    """Igualdade profunda: números com tolerância, o resto exato."""
    if isinstance(a, (np.ndarray, tuple)):
        a = a.tolist() if isinstance(a, np.ndarray) else list(a)
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None or isinstance(a, str):
        assert a == b, f"{path}: python={a!r} js={b!r}"
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        assert math.isclose(a, b, rel_tol=0, abs_tol=TOL), f"{path}: python={a!r} js={b!r}"
    elif isinstance(a, dict):
        assert set(a) == set(b), f"{path}: chaves python={sorted(a)} js={sorted(b)}"
        for k in a:
            same(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        assert len(a) == len(b), f"{path}: tamanho python={len(a)} js={len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f"{path}[{i}]")
    else:
        raise AssertionError(f"{path}: tipo inesperado {type(a)} / {type(b)}")


R = P.build_regions()
LMS = {k: [tuple(p) for p in v] for k, v in G["landmarks"].items()}
MASKS = {k: np.frombuffer(base64.b64decode(m["dados_b64"]), dtype=np.uint8).reshape(m["altura"], m["largura"])
         for k, m in G["mascaras"].items()}
BASES = {k: np.array(v, dtype=np.float32) for k, v in G["referencias"].items()}


def test_config_igual_ao_js():
    same(P.DEFAULTS, G["config"]["DEFAULTS"])
    same(P.RULES, G["config"]["RULES"])
    same(P.GLASSES, G["config"]["GLASSES"])
    assert P.NUM_POINTS == G["config"]["NUM_POINTS"] and P.OUTSIDE == G["config"]["OUTSIDE"]


def test_regioes_mesma_ordem_do_js():
    assert R["keys"] == G["regioes"]["chaves"]
    assert R["regions"] == G["regioes"]["regioes"]
    assert R["pointRegion"] == G["regioes"]["ponto_regiao"]


CASOS = G["casos"]


@pytest.mark.parametrize("caso", CASOS, ids=[f"{c['landmarks']}-{c['mascara']}-{c['referencia']}-{c['piscada']}" for c in CASOS])
def test_caso(caso):
    lm, e = LMS[caso["landmarks"]], caso["esperado"]
    mask = MASKS[caso["mascara"]] if caso["mascara"] else None
    base, blink = BASES[caso["referencia"]], G["piscadas"][caso["piscada"]]

    same(P.sample_points(lm, R), e["pontos_amostra"], "pontos_amostra")
    same(P.cap_points(lm), e["faixa_bone"], "faixa_bone")
    assert P.lower_face(lm, R) == e["metade_baixo"]

    cls = P.sample_classes(lm, mask, R) if mask is not None else None
    same(cls, e["classes"], "classes")

    res = P.analyze_classes(lm, cls, R, base, P.DEFAULTS, blink)
    same(res["state"], e["estado"], "estado")
    same(res["nonSkin"], e["nao_pele"], "nao_pele")
    same(res["red"], e["vermelho"], "vermelho")

    rules = P.accessory_rules(lm, cls, R, base)
    same(rules, e["regras"], "regras")
    if rules:
        same({"chapeu_ou_bone": P.binary_state(rules["chapeu_ou_bone"]["score"], P.RULES["capOn"]),
              "mascara": P.binary_state(rules["mascara"]["score"], P.RULES["maskOn"])},
             e["estados_regras"], "estados_regras")


def test_mascara_3d_do_mediapipe_python():
    """O ImageSegmenter do Python devolve (altura, largura, 1); tem que dar o mesmo que (altura, largura)."""
    lm, m = LMS["original"], MASKS["oculos"]
    assert (P.sample_classes(lm, m[..., None], R) == P.sample_classes(lm, m, R)).all()


def test_olho_fechado():
    for c in G["olho_fechado"]:
        same(P.eye_closed_score(c["ear"], P.DEFAULTS, c["piscada"]), c["score"], f"ear={c['ear']} piscada={c['piscada']}")


def test_confianca():
    for c in G["confianca"]:
        conf = P.confidence(c["score"], c["limiar"])
        same(conf, c["confianca"], f"score={c['score']}")
        assert P.conf_level(conf) == c["nivel"]


def test_ao_vivo_binario():
    g, prev = G["ao_vivo_binario"], None
    for i, (x, esperado) in enumerate(zip(g["entradas"], g["saidas"])):
        prev = P.binary_state(x, g["limiar"], prev)
        same(prev, esperado, f"passo {i}")


def test_oculos():
    for i, c in enumerate(G["oculos_foto"]):
        same(P.glasses_state(c["entrada"]), c["saida"], f"foto {i}")
    g, prev = G["oculos_ao_vivo"], None
    for i, (x, esperado) in enumerate(zip(g["entradas"], g["saidas"])):
        prev = P.glasses_state(x, prev)
        same(prev, esperado, f"ao vivo {i}")


def test_js_round_igual_ao_math_round():
    assert P.js_round(2.5) == 3 and round(2.5) == 2  # o motivo do helper
    assert P.js_round(-2.5) == -2 and P.js_round(0.49999) == 0


def test_top_vote_empate_segue_a_ordem_do_js():
    """Empate de votos: o JS (for...in) percorre as classes em ordem numérica, então a menor vence,
    mesmo inserida depois. Conferido no Node: {5: 3, 4: 3} → chaves "4,5"."""
    assert P._top_vote({5: 3, 4: 3}) == "roupa"       # 4 = roupa, antes de 5 = acessório
    assert P._top_vote({2: 1, 1: 1}) == "cabelo"
    assert P._top_vote({}) is None


def test_etiquetas():
    for i, c in enumerate(G["etiquetas"]):
        same(P.detected_tags(c["acessorios"]), c["etiquetas"], f"etiquetas {i}")
