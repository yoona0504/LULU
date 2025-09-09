import cv2
import numpy as np

EMOTIONS7 = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

class HeuristicEmotion:
    def predict(self, face_bgr: np.ndarray) -> dict:
        """
        얼굴 이미지를 받아서 단순히 밝기/대비로 happy, neutral, sad 확률을 추정합니다.
        """
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        mean = float(np.mean(gray))
        std = float(np.std(gray))  # 대비(표준편차)

        # 매우 단순한 규칙 (원하는 대로 조정 가능)
        if mean > 150 and std > 60:
            probs = {"happy": 0.7, "neutral": 0.2, "sad": 0.1}
        elif mean < 90 and std < 40:
            probs = {"happy": 0.1, "neutral": 0.3, "sad": 0.6}
        else:
            probs = {"happy": 0.3, "neutral": 0.5, "sad": 0.2}

        return probs


# ===== FER 기반 래퍼 (정확도 개선 버전) =====
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


def _normalize_dict(d: dict, keys: list[str]) -> dict:
    """딕셔너리를 주어진 키 순서로 정규화(합=1)."""
    arr = np.array([float(d.get(k, 0.0)) for k in keys], dtype=np.float32)
    s = float(arr.sum())
    if s <= 0:
        arr = np.ones(len(keys), dtype=np.float32) / len(keys)
    else:
        arr /= s
    return {k: float(v) for k, v in zip(keys, arr)}


try:
    from fer import FER  # pip install fer mtcnn

    class FerEmotion:
        """
        fer 라이브러리의 7클래스 결과를 우리 3클래스(happy/neutral/sad)로 매핑하고,
        조명보정/블러필터/작은얼굴제거/EMA 스무딩으로 안정화합니다.
        """

        def __init__(self, alpha: float = 0.45, conf_th: float = 0.60, min_rel: float = 0.20):
            """
            alpha: EMA 스무딩 계수(0~1, 클수록 최근값 반영 큼)
            conf_th: FER의 최고 확률 신뢰도 임계값(낮으면 중립 쪽으로)
            min_rel: 얼굴 최소 크기 비율(프레임 짧은 변 대비)
            """
            # 얼굴 검출을 MTCNN으로 → 일반 Haar보다 안정적
            self.detector = FER(mtcnn=True)
            self.alpha = float(alpha)
            self.conf_th = float(conf_th)
            self.min_rel = float(min_rel)
            self._ema = None  # np.array([happy, neutral, sad]) 길이 3 고정

        def _ema_smooth(self, new_vec: np.ndarray) -> np.ndarray:
            if self._ema is None:
                self._ema = new_vec.astype(np.float32)
            else:
                self._ema = self.alpha * new_vec.astype(np.float32) + (1 - self.alpha) * self._ema
            # 정규화 보정
            s = float(self._ema.sum())
            if s > 0:
                self._ema = self._ema / s
            return self._ema

        def _map7to3(self, fer_probs: dict) -> dict:
            """
            FER의 7클래스 -> 3클래스 매핑.
            - happy := happy
            - sad   := sad + angry + fear + disgust
            - neutral := neutral + 0.5 * surprise   (놀람은 중립/부정 중간 성격 → 반반 가중)
            """
            happy = float(fer_probs.get("happy", 0.0))
            neutral = float(fer_probs.get("neutral", 0.0)) + 0.5 * float(fer_probs.get("surprise", 0.0))
            sad = (
                float(fer_probs.get("sad", 0.0))
                + float(fer_probs.get("angry", 0.0))
                + float(fer_probs.get("fear", 0.0))
                + float(fer_probs.get("disgust", 0.0))
                + 0.5 * float(fer_probs.get("surprise", 0.0))
            )
            out = {"happy": happy, "neutral": neutral, "sad": sad}
            return _normalize_dict(out, EMOTIONS)

        def predict(self, frame_bgr: np.ndarray) -> dict:
            """
            전체 프레임(BGR)을 입력받아 내부에서 얼굴 검출 + 감정 분석.
            반환: {"happy": p, "neutral": p, "sad": p} (합=1)
            """
            if frame_bgr is None or frame_bgr.size == 0:
                # 비정상 프레임은 중립
                return {"happy": 0.0, "neutral": 1.0, "sad": 0.0}

            # 1) 조명 보정
            frame_bgr = _enhance_light(frame_bgr)

            # 2) FER는 RGB 기준 → 변환
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # 3) 감지 + 감정
            results = self.detector.detect_emotions(frame_rgb)
            if not results:
                # 얼굴 없음 → 이전 EMA 유지 or 중립
                if self._ema is not None:
                    vec = self._ema.copy()
                    return {k: float(v) for k, v in zip(EMOTIONS, vec)}
                return {"happy": 0.0, "neutral": 1.0, "sad": 0.0}

            # 가장 큰 얼굴 사용
            h, w = frame_rgb.shape[:2]
            def face_area(res):
                # fer 결과의 'box' = (x, y, w, h)
                bx, by, bw, bh = res.get("box", (0, 0, 0, 0))
                return max(0, bw) * max(0, bh)

            results.sort(key=face_area, reverse=True)
            best = results[0]
            emotions7 = best.get("emotions", {}) or {}

            # 4) 작은 얼굴/블러 필터
            bx, by, bw, bh = best.get("box", (0, 0, 0, 0))
            min_need = self.min_rel * float(min(w, h))
            if min(bw, bh) < min_need:
                # 너무 작음 → 신뢰 낮음 처리
                mapped = self._map7to3(emotions7)
                vec = np.array([mapped[e] for e in EMOTIONS], dtype=np.float32)
                vec = 0.3 * vec + 0.7 * np.array([0.0, 1.0, 0.0], dtype=np.float32)  # 중립 쪽으로 당김
                vec = vec / float(vec.sum())
                return {k: float(v) for k, v in zip(EMOTIONS, self._ema_smooth(vec))}

            # 얼굴 크롭해서 블러 체크
            x1, y1 = max(int(bx), 0), max(int(by), 0)
            x2, y2 = min(int(bx + bw), w), min(int(by + bh), h)
            face_bgr = frame_bgr[y1:y2, x1:x2]
            if face_bgr.size > 0 and _is_blurry(face_bgr, th=80.0):
                # 흐림 → 신뢰 낮음 처리
                mapped = self._map7to3(emotions7)
                vec = np.array([mapped[e] for e in EMOTIONS], dtype=np.float32)
                vec = 0.3 * vec + 0.7 * np.array([0.0, 1.0, 0.0], dtype=np.float32)  # 중립 쪽으로 당김
                vec = vec / float(vec.sum())
                return {k: float(v) for k, v in zip(EMOTIONS, self._ema_smooth(vec))}

            # 5) 7→3 매핑 후 신뢰도/EMA 스무딩
            mapped = self._map7to3(emotions7)
            # FER 최고 확률(원본 7클래스 기준) 확인해서 낮으면 중립 쪽으로 살짝 당김
            top7 = max(emotions7.values()) if emotions7 else 0.0
            vec = np.array([mapped[e] for e in EMOTIONS], dtype=np.float32)
            if float(top7) < self.conf_th:
                vec = 0.6 * vec + 0.4 * np.array([0.0, 1.0, 0.0], dtype=np.float32)  # 중립 혼합
            vec = vec / float(vec.sum())

            smoothed = self._ema_smooth(vec)
            return {k: float(v) for k, v in zip(EMOTIONS, smoothed)}

except Exception:
    # fer/mtcnn 미설치 또는 런타임 오류 시 안전한 폴백
    FerEmotion = None
