from flask import request  # ← 추가
import os                 # ← (OpenAI 키 감지용, 없어도 됨)
import time
import cv2
from collections import deque
from flask import Flask, render_template, Response, jsonify
from emotions import HeuristicEmotion, FerEmotion  # 기존 그대로

app = Flask(__name__)

# ======================================================
# 7감정 보정 유틸리티
# ------------------------------------------------------
# - FER이 특정 감정을 반환하지 않거나 값이 비정상일 때를 대비
# - 항상 7개 감정 키를 채워서 반환하도록 보정
# - 값 범위를 [0,1]로 클램프, 합계가 0이면 neutral=1.0
# ======================================================
ALL7 = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

def _complete7(d: dict) -> dict:
    out = {}
    for k in ALL7:
        v = float(d.get(k, 0.0)) if d else 0.0
        if v < 0: v = 0.0
        if v > 1: v = 1.0
        out[k] = v
    if sum(out.values()) == 0.0:
        out["neutral"] = 1.0
    return out


# ======================================================
# 카메라 초기화
# ======================================================
cap = cv2.VideoCapture(0)  # 외장 카메라면 1 또는 2로 변경
if not cap.isOpened():
    raise RuntimeError("웹캠을 열 수 없습니다. 카메라 연결을 확인하세요.")


# ======================================================
# 감정 엔진 선택
# ------------------------------------------------------
# - FerEmotion: FER 라이브러리 기반 (ResNet + MTCNN)
# - HeuristicEmotion: 규칙 기반 (폴백용)
# - 파라미터를 완화(alpha=0.50, conf_th=0.40, min_rel=0.12)
#   → 탐지 민감도↑, 반응 속도↑
# ======================================================
if FerEmotion is not None:
    predictor = FerEmotion(alpha=0.50, conf_th=0.40, min_rel=0.12)
    ENGINE_NAME = "fer"
else:
    predictor = HeuristicEmotion()
    ENGINE_NAME = "heuristic"


# ======================================================
# 얼굴 탐지기 (휴리스틱 전용)
# ======================================================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ======================================================
# 공유 상태 (대시보드용)
# ------------------------------------------------------
# - last_emotions : 최근 감정 확률 (7키 보정됨)
# - last_top      : 가장 높은 감정 {label, score}
# - fps_ma        : 최근 FPS 이동평균
# - emotion_hist  : 최근 기록 히스토리 (여러 페이지 통계/차트용)
# ======================================================
last_emotions = {"neutral": 1.0}
last_top = {"label": "neutral", "score": 1.0}
fps_ma = deque(maxlen=30)
last_time = time.time()

HIST_MAX = 200
emotion_hist = deque(maxlen=HIST_MAX)


# ======================================================
# 유틸 함수
# ======================================================
def _now_ms(): 
    return int(time.time() * 1000)

def _push_emotion_sample(probs: dict):
    """히스토리에 샘플 추가 (7키 보정 후)"""
    if not probs:
        return
    probs7 = _complete7(probs)
    emotion_hist.append({"t": _now_ms(), "probs": probs7, "engine": ENGINE_NAME})

def _update_state(probs):
    """최근 상태 업데이트 (감정, top, FPS 이동평균)"""
    global last_emotions, last_top, last_time
    probs7 = _complete7(probs if probs else {"neutral": 1.0})
    last_emotions = probs7

    # top 감정 계산
    top_label = max(probs7, key=probs7.get)
    last_top = {"label": top_label, "score": float(probs7[top_label])}

    # FPS 계산 (이동평균)
    now = time.time()
    dt = now - last_time
    last_time = now
    if dt > 0:
        fps_ma.append(1.0 / dt)

def _current_fps():
    """최근 FPS 이동평균 반환"""
    if not fps_ma:
        return 0.0
    return sum(fps_ma) / len(fps_ma)


# ======================================================
# 비디오 프레임 생성 제너레이터
# ------------------------------------------------------
# - 카메라 프레임 읽기
# - FER 또는 휴리스틱으로 감정 추론
# - 상태/히스토리 업데이트
# - JPEG 인코딩 후 스트리밍
# ======================================================
def gen_frames():
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)  # 좌우반전(셀피 느낌)

        # 감정 추론
        if hasattr(predictor, "detector"):
            # FER류: 프레임 전체
            probs = predictor.predict(frame)
            probs = _complete7(probs)  # 안전 보정
        else:
            # 휴리스틱: 얼굴이 있으면 첫 얼굴로 처리
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face = frame[y:y + h, x:x + w]
                probs = predictor.predict(face)
                probs = _complete7(probs)
                # 디버그용 시각화
                top_label = max(probs, key=probs.get)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame, top_label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
                )
            else:
                probs = {"neutral": 1.0}

        # 상태 업데이트 + 히스토리 저장
        _update_state(probs)
        _push_emotion_sample(probs)

        # 스트림 인코딩
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


# ======================================================
# 페이지 라우트 (멀티페이지 구성)
# ======================================================
@app.route("/")
def home():
    return render_template("home.html", engine=ENGINE_NAME)

@app.route("/face")
def face_page():
    return render_template("face.html", engine=ENGINE_NAME)

@app.route("/emotion")
def emotion_page():
    return render_template("emotion.html", engine=ENGINE_NAME)

@app.route("/persona")
def persona_page():
    return render_template("persona.html", engine=ENGINE_NAME)

@app.route("/chat")
def chat_page():
    return render_template("chat.html", engine=ENGINE_NAME)

@app.route("/offline")
def offline_page():
    return render_template("offline.html", engine=ENGINE_NAME)


# ======================================================
# API 엔드포인트
# ------------------------------------------------------
# - /video_feed : 카메라 MJPEG 스트림
# - /api/ping   : 서버 헬스체크
# - /api/emotion: 최근 감정 1개 + fps (7키 보정 적용)
# - /api/last   : 최근 1개 기록 반환
# - /api/history: 최근 N개 기록 반환
# - /api/summary: 평균 통계 요약 반환
# ======================================================
@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/ping")
def ping():
    return jsonify({"ok": True})

@app.route("/api/emotion")
def api_emotion():
    """최근 감정 상태 1개 + fps"""
    probs7 = _complete7(last_emotions)
    top_label = max(probs7, key=probs7.get)
    top_score = float(probs7[top_label])

    resp = jsonify({
        "engine": ENGINE_NAME,
        "fps": round(_current_fps(), 2),
        "emotions": probs7,
        "top": {"label": top_label, "score": top_score},
    })
    # 참고 로그
    print("[ENGINE]", ENGINE_NAME, "->", predictor.__class__.__name__)
    return resp

@app.route("/api/last")
def api_last():
    """최근 샘플 1개"""
    if len(emotion_hist) == 0:
        return jsonify(ok=True, data=None)
    return jsonify(ok=True, data=emotion_hist[-1])

@app.route("/api/history")
def api_history():
    """최근 N개"""
    return jsonify(ok=True, data=list(emotion_hist))

@app.route("/api/summary")
def api_summary():
    """최근 평균/우세 감정 등 간단 통계"""
    if len(emotion_hist) == 0:
        return jsonify(ok=True, total=0, top=None, means={})
    keys = list(next(iter(emotion_hist))["probs"].keys())
    sums = {k: 0.0 for k in keys}
    for rec in emotion_hist:
        for k, v in rec["probs"].items():
            sums[k] += float(v)
    n = len(emotion_hist)
    means = {k: (sums[k] / n) for k in keys}
    top = max(means, key=means.get)
    return jsonify(ok=True, total=n, top=top, means=means)

def _fallback_reply(user_msg: str, emotion=None) -> str:
    emo = ""
    if isinstance(emotion, dict) and emotion.get("top"):
        emo = f"(현재 우세 감정: {emotion['top']}) "
    return f"{emo}{user_msg}\n\n— 메시지 잘 받았어요! 로컬 폴백 응답입니다."

@app.post("/api/chat")
def api_chat():
    data = request.get_json(silent=True) or {}
    user_msg = (data.get("message") or "").strip()
    emotion = (data.get("context") or {}).get("emotion")
    if not user_msg:
        return jsonify({"ok": False, "error": "empty_message"}), 400

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            sys_prompt = "You are a friendly robot pet. Reply concisely in Korean."
            if emotion:
                sys_prompt += f"\n사용자 감정 컨텍스트: {emotion}"
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role":"system","content":sys_prompt},
                    {"role":"user","content":user_msg},
                ],
                temperature=0.7,
            )
            reply = resp.choices[0].message.content.strip()
            return jsonify({"ok": True, "reply": reply})
        except Exception:
            pass  # 실패 시 폴백으로

    return jsonify({"ok": True, "reply": _fallback_reply(user_msg, emotion)})


# ======================================================
# 실행부
# ======================================================
if __name__ == "__main__":
    # 외부 접근 필요 없으면 host 제거해도 됩니다.
    app.run(host="0.0.0.0", port=5000, debug=True)
