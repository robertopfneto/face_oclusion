"""Pipeline Python com os modelos de verdade (MediaPipe + ONNX), sem navegador. ~3 s.

    server/.venv/bin/python -m pytest tests/python -q

As imagens de tests/fixtures/acessorios são a foto do astronauta com acessórios desenhados
(server/smoke.py): servem para ver que tudo está ligado, não para medir qualidade.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from oclusao_py.modelos import Modelos  # noqa: E402
from oclusao_py.pipeline import analisar, criar_referencia, etiquetas, to_compacto  # noqa: E402
from oclusao_py.sessao import SessaoAoVivo  # noqa: E402

FIX = ROOT / "tests/fixtures"
rgb = lambda p: np.asarray(Image.open(p).convert("RGB"))


@pytest.fixture(scope="module")
def modelos():
    with Modelos() as m:
        yield m


def test_landmarks_iguais_aos_do_navegador(modelos):
    """Mesmo modelo, mesma foto: os 478 pontos batem com os que o app gerou (tests/fixtures)."""
    lm, blink = modelos.landmarks(rgb(ROOT / "tests/astronaut.jpg"))
    js = json.loads((FIX / "astronaut_landmarks.json").read_text())
    dif = np.abs(np.array([[p.x, p.y] for p in lm]) - np.array([[p["x"], p["y"]] for p in js]))
    assert len(lm) == 478
    assert dif.max() * 512 < 0.5, f"diferença máxima {dif.max() * 512:.2f} px"
    assert blink["olho_esquerdo"] < 0.2 and blink["olho_direito"] < 0.2, "olhos abertos"


def test_sem_rosto(modelos):
    out = analisar(rgb(ROOT / "tests/coffee.jpg"), modelos)
    assert out["rosto_detectado"] is False and out["boca"] is None and "acessorios" not in out


@pytest.mark.parametrize("foto,esperado", [
    ("astronaut.jpg", {"chapeu_ou_bone": False, "oculos": False, "mascara": False}),
    ("acessorios/oculos_escuros.jpg", {"chapeu_ou_bone": False, "oculos": "escuros", "mascara": False}),
    ("acessorios/oculos_grau.jpg", {"chapeu_ou_bone": False, "oculos": "grau", "mascara": False}),
    ("acessorios/mascara.jpg", {"chapeu_ou_bone": False, "oculos": False, "mascara": True}),
    ("acessorios/chapeu_ou_bone.jpg", {"chapeu_ou_bone": True, "oculos": False, "mascara": False}),
])
def test_acessorios(modelos, foto, esperado):
    p = ROOT / "tests" / foto if foto == "astronaut.jpg" else FIX / foto
    out = analisar(rgb(p), modelos)
    assert out["rosto_detectado"] is True
    assert to_compacto(out)["acessorios"] == esperado, json.dumps(out["acessorios"], ensure_ascii=False)
    for k, v in out["acessorios"].items():
        assert 0 <= v["confianca"] <= 1, k


def test_forma_completa_tem_as_chaves_do_app(modelos):
    out = analisar(rgb(ROOT / "tests/astronaut.jpg"), modelos, detalhes=True)
    assert set(out["olho_esquerdo"]) == {"ocluso", "motivo", "score", "confianca", "abertura", "piscada"}
    assert set(out["boca"]) == {"ocluso", "motivo", "score", "confianca"}
    assert set(out["acessorios"]["oculos"]) == {"presente", "tipo", "grau", "escuros", "confianca"}
    assert len(out["_detalhes"]["landmarks"]) == 478 and len(out["_detalhes"]["classes"]) == 478 + 21


def test_sessao_ao_vivo_estavel_e_timestamps_crescentes():
    quadro = rgb(FIX / "acessorios/oculos_escuros.jpg")
    with SessaoAoVivo(segmentar_a_cada=2) as s:
        saidas = [s.processar(quadro, ts_ms=1000) for _ in range(6)]  # mesmo ts: a sessão corrige
    assert all(o["rosto_detectado"] for o in saidas)
    # 1º quadro sem estado anterior usa o limiar de foto (como o app); depois, média móvel estável
    tipos = [o["acessorios"]["oculos"]["tipo"] for o in saidas]
    assert tipos == ["escuros"] * 6, tipos
    # segmentar_a_cada=2: o 2º quadro reaproveita a máscara do 1º, sem voltar para "analisando"
    assert all(isinstance(o["acessorios"]["mascara"], dict) for o in saidas)
    assert saidas[-1]["acessorios"]["chapeu_ou_bone"]["presente"] is False


def test_referencia_remove_falso_positivo(modelos):
    """Como o botão "Usar como referência": a franja do astronauta faz a testa parecer coberta;
    com a referência tirada da própria foto (rosto "livre"), ela deixa de contar."""
    foto = rgb(ROOT / "tests/astronaut.jpg")
    assert to_compacto(analisar(foto, modelos))["testa"] is True
    ref = criar_referencia(foto, modelos)
    assert ref.shape == (478,) and set(np.unique(ref)) <= {0.0, 1.0}
    compacto = to_compacto(analisar(foto, modelos, referencia=ref))
    assert not any(compacto[k] for k in ("olho_esquerdo", "olho_direito", "sobrancelhas", "nariz", "boca", "testa", "contorno"))


def test_referencia_sem_rosto_explica_o_erro(modelos):
    with pytest.raises(ValueError, match="nenhum rosto"):
        criar_referencia(rgb(ROOT / "tests/coffee.jpg"), modelos)


def test_etiquetas(modelos):
    assert [t["text"] for t in etiquetas(analisar(rgb(FIX / "acessorios/oculos_escuros.jpg"), modelos))] == ["Óculos escuros"]
    assert etiquetas(analisar(rgb(ROOT / "tests/astronaut.jpg"), modelos)) == []
    assert etiquetas(analisar(rgb(ROOT / "tests/coffee.jpg"), modelos)) == []


def test_sessao_usar_como_referencia():
    foto = rgb(ROOT / "tests/astronaut.jpg")
    with SessaoAoVivo() as s:
        assert s.processar(foto)["testa"]["ocluso"] is True
        s.usar_como_referencia(foto)
        assert s.processar(foto)["testa"]["ocluso"] is False
