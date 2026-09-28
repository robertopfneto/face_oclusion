"""Modo ao vivo: vários quadros da MESMA câmera, como o botão "Ao vivo" do app.

A diferença para pipeline.analisar() é a memória entre quadros:
  - média móvel: um quadro ruim isolado não muda o resultado;
  - histerese: para ligar e para desligar é preciso passar do limiar com folga (não pisca);
  - o segmentador e os óculos, que são mais pesados, podem rodar a cada N quadros.

    with SessaoAoVivo(segmentar_a_cada=2) as sessao:     # uma sessão por câmera/cliente
        sessao.usar_como_referencia(quadro_rosto_livre)   # opcional
        for quadro in quadros:                            # numpy uint8 (altura, largura, 3), RGB
            saida = sessao.processar(quadro)              # mesmo JSON de pipeline.analisar

Nunca compartilhe uma sessão entre câmeras: o MediaPipe em modo vídeo rastreia o rosto
de um quadro para o outro, e a memória acima também é de um rosto só.
"""
from __future__ import annotations

import time

import numpy as np

from . import regras
from .modelos import Modelos
from .pipeline import (achar_rosto, decidir_chapeu_e_mascara, decidir_oculos, decidir_regioes,
                       ler_classes, montar_saida, saida_sem_rosto)

# Antes da primeira máscara ficar pronta, todo ponto conta como "pele do rosto" (nada coberto).
TUDO_PELE = np.full(regras.NUM_POINTS, regras.CLS["FACE"], dtype=np.uint8)


class SessaoAoVivo:
    def __init__(self, referencia=None, segmentar_a_cada: int = 1, oculos_a_cada: int = 1,
                 modelos: Modelos | None = None):
        self.modelos = modelos or Modelos(modo="VIDEO")
        if self.modelos.modo != "VIDEO":
            raise ValueError("SessaoAoVivo precisa de Modelos(modo='VIDEO')")
        self.referencia = referencia
        self.segmentar_a_cada = max(1, segmentar_a_cada)
        self.oculos_a_cada = max(1, oculos_a_cada)

        # memória entre quadros
        self.quadro = 0                 # quantos quadros já passaram
        self.ultimo_ts = -1             # último timestamp enviado ao MediaPipe (ms)
        self.classes = None             # classes da última máscara pronta
        self.chapeu_e_mascara = None    # último resultado (a média móvel parte dele)
        self.oculos = None              # idem

    # ============================================================== o fluxo de um quadro

    def processar(self, rgb: np.ndarray, ts_ms: int | None = None) -> dict:
        """Um quadro → JSON completo. O fluxo (o mesmo do ao vivo do app):

          1. achar o rosto ........................... todo quadro
          2. ler classes + chapéu/máscara ............ a cada `segmentar_a_cada` quadros
          3. decidir as regiões ...................... todo quadro, com as classes mais recentes
                                                       e a piscada DESTE quadro (olho fechado)
          4. óculos .................................. a cada `oculos_a_cada` quadros
          5. montar o JSON
        """
        ts = self._proximo_timestamp(ts_ms)
        quadro = self._contar_quadro()

        rosto = achar_rosto(rgb, self.modelos, ts)                                   # 1
        if rosto is None:
            return saida_sem_rosto(self.modelos)

        if self._hora_de(self.segmentar_a_cada, quadro):                             # 2
            self.classes = ler_classes(rgb, rosto, self.modelos, ts)
            # as classes foram lidas com os pontos DESTE quadro; nos quadros seguintes elas
            # valem para os pontos novos pelo índice (o rosto quase não se move entre quadros)
            self.chapeu_e_mascara = decidir_chapeu_e_mascara(
                rosto, self.classes, self.referencia, anterior=self.chapeu_e_mascara)

        regioes = decidir_regioes(rosto, self._classes_mais_recentes(), self.referencia)   # 3

        if self._hora_de(self.oculos_a_cada, quadro):                                # 4
            self.oculos = decidir_oculos(rgb, rosto, self.modelos, anterior=self.oculos)

        return montar_saida(regioes, self.chapeu_e_mascara, self.oculos, self.modelos)   # 5

    def usar_como_referencia(self, rgb: np.ndarray, ts_ms: int | None = None) -> np.ndarray:
        """O botão "Usar como referência" do app, com o quadro atual (rosto LIVRE e de frente).
        Vale para os próximos quadros; devolve a referência para você guardar, se quiser."""
        ts = self._proximo_timestamp(ts_ms)
        rosto = achar_rosto(rgb, self.modelos, ts)
        if rosto is None:
            raise ValueError("nenhum rosto no quadro de referência")
        classes = ler_classes(rgb, rosto, self.modelos, ts)
        if classes is None:
            raise ValueError("a referência precisa do segmentador")
        self.referencia = decidir_regioes(rosto, classes, referencia=None)["nonSkin"]
        return self.referencia

    # ============================================================== apoio

    def _proximo_timestamp(self, ts_ms: int | None) -> int:
        """O MediaPipe em modo vídeo exige timestamps em ms, inteiros e sempre crescentes.
        Sem ts_ms, usa o relógio; um ts repetido ou menor vira o anterior + 1."""
        agora = int(time.monotonic() * 1000) if ts_ms is None else int(ts_ms)
        self.ultimo_ts = max(self.ultimo_ts + 1, agora)
        return self.ultimo_ts

    def _contar_quadro(self) -> int:
        atual, self.quadro = self.quadro, self.quadro + 1
        return atual

    @staticmethod
    def _hora_de(a_cada: int, quadro: int) -> bool:
        return quadro % a_cada == 0

    def _classes_mais_recentes(self):
        """Classes da última máscara; antes da primeira, tudo pele. Sem segmentador, None."""
        if self.modelos.segmentador is None:
            return None
        return self.classes if self.classes is not None else TUDO_PELE

    def close(self):
        self.modelos.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

