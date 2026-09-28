"""Pipeline: uma imagem → o JSON que o app mostra.

O FLUXO (o mesmo do app; cada etapa é uma função abaixo, nesta ordem):

    imagem RGB
      │
      ├─ 1. achar_rosto ............... FaceLandmarker: 478 pontos + piscada de cada olho
      │                                 (sem rosto → para aqui: {"rosto_detectado": false})
      ├─ 2. ler_classes ............... segmentador: classe de cada pixel (fundo, cabelo, pele,
      │                                 rosto, roupa, acessório); cada ponto lê a classe embaixo dele
      ├─ 3. decidir_regioes ........... olhos, boca, testa... cobertos? por quê? (+ olho fechado)
      ├─ 4. decidir_chapeu_e_mascara .. faixa acima da testa / metade de baixo em roupa ou acessório
      ├─ 5. decidir_oculos ............ classificador ONNX no recorte do rosto: grau / escuros
      │
      └─ 6. montar_saida .............. JSON completo (to_compacto() dá a forma curta)

As decisões em si (limiares, contas) ficam em regras.py, igual a occlusion.js no app.

Uso:
    with Modelos() as m:
        saida = analisar(rgb, m)                      # rgb: numpy uint8 (altura, largura, 3)
        ref = criar_referencia(rgb_rosto_livre, m)    # opcional: o "Usar como referência" do app
        saida = analisar(rgb, m, referencia=ref)

Linha de comando:
    python -m oclusao_py.pipeline foto.jpg [outra.jpg ...] [--compacto] [--referencia rosto_livre.jpg]

Para vídeo (vários quadros da mesma câmera), use sessao.SessaoAoVivo.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass

import numpy as np
from PIL import Image

from . import regras
from .modelos import Modelos

REGIOES = regras.build_regions()   # pontos de cada região (olhos, boca, ...), montado uma vez

# Recorte do rosto para o classificador de óculos: o mesmo do app (cropSquare em app.js).
MARGEM_RECORTE = 0.15   # lado do quadrado = maior lado do rosto × (1 + 2 × margem)
TAMANHO_RECORTE = 256   # o modelo de óculos recebe 256×256


@dataclass
class Rosto:
    """O que a etapa 1 encontrou."""
    pontos: list[tuple[float, float]]   # 478 landmarks (x, y), de 0 a 1 em relação à imagem
    piscada: dict | None                # {"olho_esquerdo": 0..1, "olho_direito": 0..1} (olho da pessoa)


# ================================================================== o fluxo

def analisar(rgb: np.ndarray, modelos: Modelos, referencia=None, detalhes: bool = False) -> dict:
    """Uma foto → JSON completo.
    referencia: resultado de criar_referencia() (opcional; remove falsos positivos de lábios/sobrancelhas).
    detalhes:   inclui pontos, classes lidas e scores brutos em "_detalhes" (para depurar)."""
    rosto = achar_rosto(rgb, modelos)                                         # 1
    if rosto is None:
        return saida_sem_rosto(modelos)

    classes = ler_classes(rgb, rosto, modelos)                                # 2
    regioes = decidir_regioes(rosto, classes, referencia)                     # 3
    chapeu_e_mascara = decidir_chapeu_e_mascara(rosto, classes, referencia)   # 4
    oculos = decidir_oculos(rgb, rosto, modelos)                              # 5

    saida = montar_saida(regioes, chapeu_e_mascara, oculos, modelos)          # 6
    if detalhes:
        saida["_detalhes"] = _detalhes(rosto, classes, regioes, chapeu_e_mascara, oculos)
    return saida


def criar_referencia(rgb: np.ndarray, modelos: Modelos) -> np.ndarray:
    """O botão "Usar como referência" do app: numa foto do rosto LIVRE (sem óculos, boné ou
    máscara, de frente), anota quais pontos já caem fora da pele normalmente (lábios,
    sobrancelhas...). Passada para analisar(), esses pontos deixam de contar como cobertos.
    Devolve 478 valores (0 = ponto na pele, 1 = fora da pele)."""
    rosto = achar_rosto(rgb, modelos)
    if rosto is None:
        raise ValueError("nenhum rosto na foto de referência")
    classes = ler_classes(rgb, rosto, modelos)
    if classes is None:
        raise ValueError("a referência precisa do segmentador (Modelos(segmentador=True))")
    return decidir_regioes(rosto, classes, referencia=None)["nonSkin"]


def etiquetas(saida: dict) -> list[dict]:
    """Os acessórios detectados, com o texto que o app mostra na tela:
    [{"key": "oculos", "text": "Óculos de grau", "confianca": 0.71}, ...]"""
    return regras.detected_tags(saida.get("acessorios"))


# ================================================================== as etapas

def achar_rosto(rgb: np.ndarray, modelos: Modelos, ts_ms: int | None = None) -> Rosto | None:
    """Etapa 1. Os 478 pontos do rosto e a piscada de cada olho; None se não houver rosto."""
    landmarks, piscada = modelos.landmarks(rgb, ts_ms)
    if landmarks is None:
        return None
    return Rosto(pontos=regras._xy(landmarks), piscada=piscada)


def ler_classes(rgb: np.ndarray, rosto: Rosto, modelos: Modelos, ts_ms: int | None = None) -> np.ndarray | None:
    """Etapa 2. Classe do segmentador embaixo de cada ponto: 478 landmarks + 21 pontos da faixa
    acima da testa (onde se procura o boné). None se não houver segmentador."""
    mascara = modelos.mascara(rgb, ts_ms)
    if mascara is None:
        return None
    return regras.sample_classes(rosto.pontos, mascara, REGIOES)


def decidir_regioes(rosto: Rosto, classes: np.ndarray | None, referencia=None) -> dict:
    """Etapa 3. Cada região coberta ou não, com motivo, score e confiança.
    Nos olhos entra também "fechado" (piscada E abertura da pálpebra precisam concordar).
    Devolve {"state": {...}, "nonSkin": [...478], "red": [...478]} (os nomes do JS)."""
    return regras.analyze_classes(rosto.pontos, classes, REGIOES, referencia, regras.DEFAULTS, rosto.piscada)


def decidir_chapeu_e_mascara(rosto: Rosto, classes: np.ndarray | None, referencia=None, anterior: dict | None = None) -> dict | None:
    """Etapa 4. Chapéu/boné (faixa acima da testa) e máscara (metade de baixo do rosto):
    fração de pontos em roupa ou acessório, comparada com o limiar.
    anterior: o resultado do quadro anterior, no ao vivo (média móvel + histerese).
    None se não houver segmentador."""
    scores = regras.accessory_rules(rosto.pontos, classes, REGIOES, referencia)
    if scores is None:
        return None

    def decidir(item: str, limiar: float) -> dict:
        estado = regras.binary_state(scores[item]["score"], limiar, anterior and anterior[item])
        return {**estado, "classes": scores[item]["classes"]}   # classes: o que foi lido (para depurar)

    return {
        "chapeu_ou_bone": decidir("chapeu_ou_bone", regras.RULES["capOn"]),
        "mascara": decidir("mascara", regras.RULES["maskOn"]),
    }


def decidir_oculos(rgb: np.ndarray, rosto: Rosto, modelos: Modelos, anterior: dict | None = None) -> dict | None:
    """Etapa 5. Recorta o rosto, passa no classificador (scores de "grau" e "escuros") e decide.
    anterior: o resultado do quadro anterior, no ao vivo (média móvel + histerese).
    None se não houver o modelo de óculos."""
    scores = modelos.scores_oculos(recorte_do_rosto(rgb, rosto))
    if scores is None:
        return None
    return regras.glasses_state(scores, anterior)


def montar_saida(regioes: dict, chapeu_e_mascara: dict | None, oculos: dict | None, modelos: Modelos) -> dict:
    """Etapa 6. O JSON completo, com as mesmas chaves do app."""
    saida = regras.to_view(regioes["state"], REGIOES["keys"], full=True)

    tem_segmentador = modelos.segmentador is not None
    tem_oculos = modelos.oculos is not None
    if not (tem_segmentador or tem_oculos):
        return saida

    acessorios = {}
    if tem_segmentador:
        acessorios["chapeu_ou_bone"] = _item_por_regra(chapeu_e_mascara and chapeu_e_mascara["chapeu_ou_bone"], regras.RULES["capOn"])
    if tem_oculos:
        acessorios["oculos"] = _item_oculos(oculos)
    if tem_segmentador:
        acessorios["mascara"] = _item_por_regra(chapeu_e_mascara and chapeu_e_mascara["mascara"], regras.RULES["maskOn"])
    saida["acessorios"] = acessorios
    return saida


def saida_sem_rosto(modelos: Modelos) -> dict:
    """Nenhum rosto: regiões None e sem acessórios (igual ao app)."""
    return regras.to_view({"rosto_detectado": False}, REGIOES["keys"], full=True)


# ================================================================== detalhes

def recorte_do_rosto(rgb: np.ndarray, rosto: Rosto) -> np.ndarray:
    """Quadrado em volta do rosto (caixa dos pontos + margem), redimensionado para 256×256.
    O que sai da imagem vira preto (como no app)."""
    altura, largura = rgb.shape[:2]
    xs = [x for x, _ in rosto.pontos]
    ys = [y for _, y in rosto.pontos]
    lado = max((max(xs) - min(xs)) * largura, (max(ys) - min(ys)) * altura) * (1 + 2 * MARGEM_RECORTE)
    cx = (min(xs) + max(xs)) / 2 * largura
    cy = (min(ys) + max(ys)) / 2 * altura
    caixa = tuple(round(v) for v in (cx - lado / 2, cy - lado / 2, cx + lado / 2, cy + lado / 2))
    return np.asarray(Image.fromarray(rgb).crop(caixa).resize((TAMANHO_RECORTE, TAMANHO_RECORTE), Image.BICUBIC))


def _item_por_regra(estado: dict | None, limiar: float):
    """Chapéu/boné ou máscara no formato do app: {presente, score, confianca},
    "analisando" (ainda sem resultado) ou "fora do quadro" (faixa do boné fora da imagem)."""
    if not estado:
        return "analisando"
    if estado.get("fora"):
        return "fora do quadro"
    return {
        "presente": estado["presente"],
        "score": regras.r2(estado["score"]),
        "confianca": regras.r2(regras.confidence(estado["score"], limiar)),
    }


def _item_oculos(estado: dict | None):
    """Óculos no formato do app: {presente, tipo, grau, escuros, confianca} ou "analisando"."""
    if not estado:
        return "analisando"
    maior = max(estado["grau"], estado["escuros"])
    return {
        "presente": estado["presente"],
        "tipo": estado["tipo"],
        "grau": regras.r2(estado["grau"]),
        "escuros": regras.r2(estado["escuros"]),
        "confianca": regras.r2(regras.confidence(maior, regras.GLASSES["photoOn"])),
    }


def _detalhes(rosto: Rosto, classes, regioes: dict, chapeu_e_mascara, oculos) -> dict:
    return {
        "landmarks": [list(p) for p in rosto.pontos],
        "piscada": rosto.piscada,
        "vermelho": regioes["red"].tolist(),
        "classes": None if classes is None else classes.tolist(),
        "classes_chapeu": chapeu_e_mascara and chapeu_e_mascara["chapeu_ou_bone"]["classes"],
        "classes_mascara": chapeu_e_mascara and chapeu_e_mascara["mascara"]["classes"],
        "oculos_bruto": oculos and oculos["bruto"],
    }


def to_compacto(saida: dict) -> dict:
    """Forma curta do app: regiões e acessórios como booleanos (óculos: "grau" | "escuros" | False)."""
    curta = {k: (v["ocluso"] if isinstance(v, dict) and "ocluso" in v else v)
             for k, v in saida.items() if k != "_detalhes"}
    if isinstance(saida.get("acessorios"), dict):
        curta["acessorios"] = {
            k: v if not isinstance(v, dict) else (v["tipo"] or False) if k == "oculos" else v["presente"]
            for k, v in saida["acessorios"].items()
        }
    return curta


# ================================================================== linha de comando

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    compacto = "--compacto" in argv
    arquivo_ref = None
    if "--referencia" in argv:
        i = argv.index("--referencia")
        arquivo_ref = argv[i + 1]
        del argv[i:i + 2]
    fotos = [a for a in argv if not a.startswith("--")]
    if not fotos:
        sys.exit("uso: python -m oclusao_py.pipeline foto.jpg [...] [--compacto] [--referencia rosto_livre.jpg]")

    ler = lambda caminho: np.asarray(Image.open(caminho).convert("RGB"))
    with Modelos() as modelos:
        referencia = criar_referencia(ler(arquivo_ref), modelos) if arquivo_ref else None
        for foto in fotos:
            saida = analisar(ler(foto), modelos, referencia)
            linha = {"foto": foto, **(to_compacto(saida) if compacto else saida),
                     "etiquetas": [t["text"] for t in etiquetas(saida)]}
            print(json.dumps(linha, ensure_ascii=False))


if __name__ == "__main__":
    main()
