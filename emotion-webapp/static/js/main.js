const METER_KEYS = ["happy", "neutral", "sad"];

function $(id) { return document.getElementById(id); }

function log(msg) {
  const el = $("log");
  el.textContent = `[${new Date().toLocaleTimeString()}] ${msg}\n` + el.textContent;
}

async function ping() {
  try {
    const res = await fetch("/api/ping");
    const data = await res.json();
    $("status").textContent = data.ok ? "서버 연결됨" : "오프라인";
  } catch {
    $("status").textContent = "오프라인";
  }
}

function buildMeters() {
  const wrap = $("meters");
  wrap.innerHTML = "";
  METER_KEYS.forEach((k) => {
    const row = document.createElement("div");
    row.className = "row";
    row.innerHTML = `
      <div class="label">${k}</div>
      <div class="track"><div class="fill" id="fill-${k}"></div></div>
      <div class="val" id="val-${k}">0%</div>
    `;
    wrap.appendChild(row);
  });
}

let lastTs = performance.now();
function updateFps() {
  const now = performance.now();
  const dt = now - lastTs;
  lastTs = now;
  const fps = Math.max(1, Math.min(60, Math.round(1000 / dt)));
  $("fps").textContent = `${fps}`;
}

async function pollEmotion() {
  try {
    const res = await fetch("/api/emotion", { cache: "no-store" });
    const data = await res.json();

    const probs = data.probs || {};
    const label = data.label || "-";
    $("topLabel").textContent = label;

    // 엔진 힌트
    $("engine").textContent = probs.angry !== undefined ? "FER(딥러닝)" : "Heuristic";

    // happy/neutral/sad만 우선 표시 (FER일 경우 없는 키는 0%)
    METER_KEYS.forEach((k) => {
      const v = Math.round(100 * (probs[k] || 0));
      const fill = $(`fill-${k}`);
      const val = $(`val-${k}`);
      if (fill) fill.style.width = `${v}%`;
      if (val) val.textContent = `${v}%`;
    });

    updateFps();
  } catch (e) {
    log("poll 실패: " + e.message);
  } finally {
    requestAnimationFrame(pollEmotion);
  }
}

buildMeters();
ping();
pollEmotion();
