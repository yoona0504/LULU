# emotions.py
# -*- coding: utf-8 -*-
"""
카메라 프레임에서 7개 감정(angry, disgust, fear, happy, sad, surprise, neutral)
확률 분포를 반환합니다.
- FER(ResNet + MTCNN) 사용 가능하면 해당 모델을 사용
- FER이 없거나 런타임 오류면 휴리스틱(규칙 기반)으로 안전 폴백
"""

import cv2
import numpy as np

# ======================================================
# 7개 감정 레이블 (프론트/서버 공통 계약)
# ======================================================
EMOTIONS: list[str] = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


# ======================================================
# 공용 유틸리티 함수
# ------------------------------------------------------
# - _enhance_light : CLAHE로 조명/대비 보정
# - _is_blurry     : 라플라시안 분산으로 블러 여부 판단
# - _norm_vec      : 벡터 정규화(합=1)
# - _to_dict       : vec(길이 7) → dict 변환
# ======================================================
def _enhance_light(bgr: np.ndarray) -> np.ndarray:
    """어두운 얼굴 대비를 살리기 위한 CLAHE 조명 보정."""
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(y)
    return cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2BGR)

def _is_blurry(bgr: np.ndarray, th: float = 80.0) -> bool:
    """라플라시안 분산으로 블러(흐림) 판정."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()) < th

def _norm_vec(vec: np.ndarray) -> np.ndarray:
    """벡터 정규화(합=1)."""
    s = float(vec.sum())
    if s > 0:
        return vec / s
    return np.ones_like(vec, dtype=np.float32) / float(len(vec))

def _to_dict(vec: np.ndarray) -> dict[str, float]:
    """vec(길이 7)을 감정 딕셔너리로 변환."""
    return {k: float(v) for k, v in zip(EMOTIONS, vec)}


# ======================================================
# 휴리스틱 베이스라인 (7클래스 출력)
# ------------------------------------------------------
# - 얼굴이 없거나 FER이 사용 불가할 때 사용
# - mean(밝기), std(대비) 기반 간단 규칙
# - 항상 7개 감정 확률 분포 반환
# ======================================================
class HeuristicEmotion:
    """
    입력: 얼굴 BGR 크롭 이미지
    출력: 7클래스 확률 딕셔너리
    """
    def predict(self, face_bgr: np.ndarray) -> dict[str, float]:
        if face_bgr is None or face_bgr.size == 0:
            # 기본 priors (중립 비중↑)
            priors = np.array([0.05, 0.03, 0.05, 0.10, 0.10, 0.07, 0.60], dtype=np.float32)
            return _to_dict(_norm_vec(priors))

        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        mean = float(np.mean(gray))
        std = float(np.std(gray))

        # 간단한 규칙 기반 분포 (데모/폴백용)
        if mean > 150 and std > 60:
            base = np.array([0.03, 0.02, 0.03, 0.65, 0.08, 0.07, 0.12], dtype=np.float32)
        elif mean < 90 and std < 40:
            base = np.array([0.08, 0.03, 0.05, 0.07, 0.55, 0.05, 0.17], dtype=np.float32)
        else:
            base = np.array([0.07, 0.03, 0.05, 0.20, 0.15, 0.05, 0.45], dtype=np.float32)

        return _to_dict(_norm_vec(base))


# ======================================================
# FER 래퍼 (7클래스 그대로 출력)
# ------------------------------------------------------
# - fer 라이브러리 기반 (ResNet + MTCNN)
# - 기능:
#   * CLAHE 조명 보정
#   * 얼굴 크기/블러 가드
#   * 신뢰도(conf_th) 가드
#   * EMA(지수 이동 평균) 스무딩
#   * 프레임 스킵 (2프레임 중 1번만 추론 → FPS↑)
# - 출력: 7클래스 확률 딕셔너리
# ======================================================
try:
    from fer import FER  # pip install fer mtcnn

    class FerEmotion:
        def __init__(self, alpha: float = 0.50, conf_th: float = 0.40, min_rel: float = 0.12) -> None:
            """
            alpha   : EMA 계수 (클수록 과거값 비중↑ → 더 부드럽지만 반응 느림)
            conf_th : FER 7클래스 중 최대 확률 임계값 (낮으면 priors와 혼합)
            min_rel : 얼굴 최소 길이 비율 (프레임 짧은 변 대비)
            """
            self.detector = FER(mtcnn=True)
            self.alpha = float(alpha)
            self.conf_th = float(conf_th)
            self.min_rel = float(min_rel)

            self._ema: np.ndarray | None = None   # EMA 상태
            self._skip = 0                        # 프레임 스킵 카운터

            # neutral 중심 priors (노이즈/미검출 시 혼합용)
            self._priors = np.array([0.05, 0.03, 0.05, 0.10, 0.10, 0.07, 0.60], dtype=np.float32)

        # --- 내부 유틸 ---
        def _ema7(self, new7: np.ndarray) -> np.ndarray:
            """EMA(지수 이동 평균)로 부드럽게 스무딩"""
            v = new7.astype(np.float32)
            if self._ema is None:
                self._ema = v
            else:
                self._ema = self.alpha * self._ema + (1.0 - self.alpha) * v
            return _norm_vec(self._ema)

        def _small_or_blur(self, frame_bgr: np.ndarray, box: tuple[int, int, int, int], min_need: float) -> bool:
            """얼굴이 작거나 블러면 True."""
            bx, by, bw, bh = box
            if min(bw, bh) < min_need:
                return True
            h, w = frame_bgr.shape[:2]
            x1, y1 = max(int(bx), 0), max(int(by), 0)
            x2, y2 = min(int(bx + bw), w), min(int(by + bh), h)
            face_bgr = frame_bgr[y1:y2, x1:x2]
            if face_bgr.size == 0:
                return True
            return _is_blurry(face_bgr, th=80.0)

        # --- 메인 추론 ---
        def predict(self, frame_bgr: np.ndarray) -> dict[str, float]:
            priors = self._priors

            # ----- 프레임 스킵 -----
            self._skip = (self._skip + 1) % 2   # 2로 두면 절반만 추론
            if self._skip != 0 and self._ema is not None:
                # 스킵 프레임은 EMA 결과만 반환
                return _to_dict(self._ema)

            # 입력 None → priors
            if frame_bgr is None or frame_bgr.size == 0:
                sm = self._ema7(_norm_vec(priors))
                return _to_dict(sm)

            # 조명 보정 + RGB 변환 (FER는 RGB 입력)
            frame_bgr = _enhance_light(frame_bgr)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # FER 추론 (감정 검출)
            results = self.detector.detect_emotions(frame_rgb)
            if not results:
                # 얼굴 미검출 시 → 이전 EMA 또는 priors
                if self._ema is not None:
                    return _to_dict(self._ema)
                sm = self._ema7(_norm_vec(priors))
                return _to_dict(sm)

            # 가장 큰 얼굴 선택
            h, w = frame_rgb.shape[:2]
            results.sort(
                key=lambda r: max(0, r.get("box", (0, 0, 0, 0))[2]) * max(0, r.get("box", (0, 0, 0, 0))[3]),
                reverse=True
            )
            best = results[0]
            emotions7: dict[str, float] = best.get("emotions", {}) or {}
            bx, by, bw, bh = best.get("box", (0, 0, 0, 0))

            # dict → vec7 (고정 순서)
            vec7 = np.array([float(emotions7.get(k, 0.0)) for k in EMOTIONS], dtype=np.float32)
            vec7 = _norm_vec(vec7)

            # 작은 얼굴/블러 가드 또는 신뢰도↓ 가드
            min_need = self.min_rel * float(min(w, h))
            top7 = float(np.max(vec7)) if vec7.size else 0.0
            guard = self._small_or_blur(frame_bgr, (bx, by, bw, bh), min_need) or (top7 < self.conf_th)
            if guard:
                vec7 = _norm_vec(0.6 * vec7 + 0.4 * priors)

            # EMA 스무딩 후 반환
            sm = self._ema7(vec7)
            return _to_dict(sm)

except Exception:
    # fer/mtcnn 미설치 또는 런타임 오류 시 안전한 폴백
    FerEmotion = None
