"""Carrega os três modelos, os mesmos arquivos que o app usa no navegador (pasta models/).

- FaceLandmarker (face_landmarker.task): 478 pontos + blendshapes (piscada)
- ImageSegmenter (selfie_multiclass_256x256.tflite): classe de cada pixel
- Óculos (oculos.onnx): glasses-detector exportado por server/export_onnx.py, via onnxruntime

modo "IMAGE" para fotos soltas; "VIDEO" para uma sequência de quadros da mesma câmera
(o MediaPipe rastreia o rosto entre quadros; exige timestamps crescentes em ms).
Uma instância em modo VIDEO serve a UMA câmera: não compartilhe entre fluxos.
"""
from __future__ import annotations

from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODELS = Path(__file__).resolve().parent.parent / "models"


class Modelos:
    def __init__(self, modo: str = "IMAGE", pasta: Path | str = MODELS, segmentador: bool = True, oculos: bool = True):
        pasta = Path(pasta)
        rm = getattr(vision.RunningMode, modo)
        self.modo = modo
        self.landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(pasta / "face_landmarker.task")),
            running_mode=rm, num_faces=1, output_face_blendshapes=True))
        self.segmentador = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=str(pasta / "selfie_multiclass_256x256.tflite")),
            running_mode=rm, output_category_mask=True, output_confidence_masks=False)) if segmentador else None
        self.oculos = None
        if oculos:
            import onnxruntime as ort
            self.oculos = ort.InferenceSession(str(pasta / "oculos.onnx"), providers=["CPUExecutionProvider"])

    # ------------------------------------------------------------ inferência

    @staticmethod
    def _img(rgb: np.ndarray) -> mp.Image:
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb, dtype=np.uint8))

    def landmarks(self, rgb: np.ndarray, ts_ms: int | None = None):
        """→ (landmarks | None, piscada {olho_esquerdo, olho_direito} | None)."""
        img = self._img(rgb)
        r = self.landmarker.detect_for_video(img, ts_ms) if self.modo == "VIDEO" else self.landmarker.detect(img)
        if not r.face_landmarks:
            return None, None
        blink = None
        if r.face_blendshapes:
            b = {c.category_name: c.score for c in r.face_blendshapes[0]}
            # Left/Right = olho da pessoa (conferido: igual a regras.EYE)
            blink = {"olho_esquerdo": b.get("eyeBlinkLeft"), "olho_direito": b.get("eyeBlinkRight")}
        return r.face_landmarks[0], blink

    def mascara(self, rgb: np.ndarray, ts_ms: int | None = None) -> np.ndarray | None:
        """Máscara de classes (altura, largura) uint8, ou None sem segmentador."""
        if self.segmentador is None:
            return None
        img = self._img(rgb)
        r = self.segmentador.segment_for_video(img, ts_ms) if self.modo == "VIDEO" else self.segmentador.segment(img)
        mascara = r.category_mask.numpy_view()        # (altura, largura, 1), na resolução da imagem
        if mascara.ndim == 3:
            mascara = mascara[..., 0]
        return mascara.copy()                         # copia: a memória original é do MediaPipe

    def scores_oculos(self, recorte_256: np.ndarray) -> dict | None:
        """recorte RGB uint8 256×256 → {grau, escuros} (probabilidades), ou None sem o modelo."""
        if self.oculos is None:
            return None
        p = self.oculos.run(None, {"pixels": np.ascontiguousarray(recorte_256[None], dtype=np.uint8)})[0][0]
        return {"grau": float(p[0]), "escuros": float(p[1])}

    # ------------------------------------------------------------ ciclo de vida

    def close(self):
        for m in (self.landmarker, self.segmentador):
            if m is not None:
                m.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
