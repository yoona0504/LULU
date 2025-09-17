// ========= DOM refs =========
const sizeRange = document.getElementById('sizeRange');
const aspectSelect = document.getElementById('aspectSelect');
const html = document.documentElement;
const camBox = document.getElementById('camBox');

const emotionList = document.getElementById('emotionList');
const topEmotionEl = document.getElementById('topEmotion');
const topScoreEl = document.getElementById('topScore');
const fpsEl = document.getElementById('fps');
const engineEl = document.getElementById('engineName');
const logEl = document.getElementById('log');
const clearBtn = document.getElementById('clearLog');

clearBtn?.addEventListener('click', () => (logEl.textContent = ''));

// ========= 카메라 크기/종횡비 =========
function applySize(px) {
  html.style.setProperty('--cam-max-width', `${px}px`);
  camBox.style.maxWidth = `var(--cam-max-width)`;
}
function applyAspect(ratio) {
  html.style.setProperty('--cam-aspect', ratio);
  camBox.style.aspectRatio = `var(--cam-aspect)`;
}

applySize(520);
applyAspect('4/3');

sizeRange?.addEventListener('input', () => applySize(Number(sizeRange.value)));
aspectSelect?.addEventListener('change', () => applyAspect(aspectSelect.value));

// ========= 감정 막대 =========
const EMOTION_KEYS = ["angry","disgust","fear","happy","sad","surprise","neutral"];

function ensureBars() {
  if (emotionList.children.length) return;
  EMOTION_KEYS.forEach((k) => {
    const row = document.createElement('div');
    row.className = 'row';
    row.innerHTML = `
      <div class="label text-capitalize">${k}</div>
      <div class="track"><div class="fill" id="bar-${k}"></div></div>
      <div class="val" id="val-${k}">0%</div>
    `;
    emotionList.appendChild(row);
  });
}
ensureBars();

// ========= 폴링 =========
const POLL_MS = 500;

async function poll() {
  try {
    const res = await fetch('/api/emotion', { cache: 'no-store' });
    const ct = res.headers.get('content-type') || '';
    if (!ct.includes('application/json')) {
      const text = await res.text();
      pushLog(`poll 실패: JSON 아님 (content-type=${ct}). 예: ${text.slice(0,80)}...`);
      return;
    }
    const data = await res.json();
    updateUI(data);
  } catch (e) {
    pushLog('poll 실패: ' + (e?.message || e));
  }
}
setInterval(poll, POLL_MS);

// ========= UI 업데이트 =========
function updateUI(data) {
  if (!data) return;

  if (data.engine) engineEl.textContent = `엔진: ${data.engine}`;
  if (typeof data.fps === 'number') fpsEl.textContent = `FPS: ${data.fps.toFixed(1)}`;

  const probs = data.emotions || data.probs || {};
  EMOTION_KEYS.forEach((k) => {
    const v = Math.max(0, Math.min(1, Number(probs[k] ?? 0)));
    const pct = Math.round(v * 100);
    const bar = document.getElementById(`bar-${k}`);
    const val = document.getElementById(`val-${k}`);
    if (bar) bar.style.width = pct + '%';
    if (val) val.textContent = pct + '%';
  });

  if (data.top && data.top.label) {
    topEmotionEl.textContent = data.top.label;
    topScoreEl.textContent =
      typeof data.top.score === 'number' ? Math.round(data.top.score * 100) + '%' : '-';
  } else {
    // top이 없을 때 최대값 계산
    let bestK = '-', bestV = 0;
    for (const [k, v] of Object.entries(probs)) {
      const n = Number(v) || 0;
      if (n > bestV) { bestV = n; bestK = k; }
    }
    topEmotionEl.textContent = bestK;
    topScoreEl.textContent = Math.round(bestV * 100) + '%';
  }
}

// ========= 로그 =========
function pushLog(msg) {
  const t = new Date().toLocaleTimeString();
  logEl.textContent = `[${t}] ${msg}\n` + logEl.textContent;
}
