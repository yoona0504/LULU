import time
import cv2
from collections import deque
from flask import Flask, render_template, Response, jsonify
from emotions import HeuristicEmotion, FerEmotion  # 기존 그대로

app = Flask(__name__)

# -----------------------
# 카메라 초기화
# -----------------------
cap = cv2.VideoCapture(0)  # 외장 카메라면 1 또는 2로 변경
if not cap.isOpened():
    raise RuntimeError("웹캠을 열 수 없습니다. 카메라 연결을 확인하세요.")

# -----------------------
# 감정 엔진 선택
# -----------------------
if FerEmotion is not None:
    predictor = FerEmotion()
    ENGINE_NAME = "fer"
else:
    predictor = HeuristicEmotion()
    ENGINE_NAME = "heuristic"

# 얼굴 탐지기 (휴리스틱일 때 얼굴 크롭 사용)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------
# 공유 상태(대시보드용)
# -----------------------
last_emotions = {"neutral": 1.0}
last_top = {"label": "neutral", "score": 1.0}
fps_ma = deque(maxlen=30)  # 최근 30프레임 이동평균
last_time = time.time()

def _update_state(probs):
    """감정 확률과 top, FPS 이동평균을 업데이트"""
    global last_emotions, last_top, last_time
    last_emotions = probs if probs else {"neutral": 1.0}

    # top 계산
    top_label = max(last_emotions, key=last_emotions.get)
    last_top = {"label": top_label, "score": float(last_emotions[top_label])}

    # FPS 계산(이동평균)
    now = time.time()
    dt = now - last_time
    last_time = now
    if dt > 0:
        fps_ma.append(1.0 / dt)

def _current_fps():
    if not fps_ma:
        return 0.0
    return sum(fps_ma) / len(fps_ma)

# -----------------------
# 비디오 프레임 생성
# -----------------------
def gen_frames():
    while True:
        ok, frame = cap.read()
        if not ok:
            # 카메라 끊김 방지: 빈 루프가 아닌 약간의 대기 후 계속 시도
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)  # 좌우반전(셀피 느낌)

        # 감정 추론
        if hasattr(predictor, "detector"):
            # FER류: 프레임 전체
            probs = predictor.predict(frame)
        else:
            # 휴리스틱: 얼굴이 있으면 첫 얼굴로 처리
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face = frame[y:y+h, x:x+w]
                probs = predictor.predict(face)
                # 시각화: 박스/라벨
                top_label = max(probs, key=probs.get)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame, top_label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
                )
            else:
                probs = {"neutral": 1.0}

        # 대시보드용 상태 업데이트
        _update_state(probs)

        # 스트림 인코딩
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )

# -----------------------
# 라우트
# -----------------------
@app.route("/")
def index():
    # templates/index.html 에 파일이 있어야 합니다.
    return render_template("dashboard.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/ping")
def ping():
    return jsonify({"ok": True})

@app.route("/api/emotion")
def api_emotion():
    """대시보드 폴링용 JSON"""
    return jsonify({
        "engine": ENGINE_NAME,
        "fps": round(_current_fps(), 2),
        "emotions": last_emotions,
        "top": last_top,
    })

# -----------------------
# 실행
# -----------------------
if __name__ == "__main__":
    # 외부 접근 필요 없으면 host 제거해도 됩니다.
    app.run(host="0.0.0.0", port=5000, debug=True)
