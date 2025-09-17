// ========= DOM refs  =========
const sizeRange    = document.getElementById('sizeRange')    || document.getElementById('camScale');
const aspectSelect = document.getElementById('aspectSelect') || document.getElementById('aspect');
const html   = document.documentElement;
const camBox = document.getElementById('camBox');

const metersEl     = document.getElementById('meters')       || document.getElementById('emotionList');
const topEmotionEl = document.getElementById('topEmotion');
const topScoreEl   = document.getElementById('topScore')     || document.getElementById('confidence');
const fpsEl        = document.getElementById('fps');
const engineEl     = document.getElementById('engineName')   || document.getElementById('engine');
const logEl        = document.getElementById('log');
const clearBtn     = document.getElementById('clearLog');

clearBtn?.addEventListener('click', () => {
  if (!logEl) return;
  if (logEl.tagName === 'TEXTAREA' || logEl.tagName === 'INPUT') logEl.value = '';
  else logEl.textContent = '';
});

// ========= 카메라 크기/종횡비 (변수명/속성 모두 갱신: 캐시/스타일 차이 대비) =========
function applySize(px) {
  const v = `${px}px`;
  html.style.setProperty('--cam-max', v);  // ← 이 변수로 통일
  if (camBox) { camBox.style.width = v; }  // ← fallback (확실하게 적용)
}
function applyAspect(ratio) {
  if (camBox) camBox.style.aspectRatio = ratio;
}

// 초기값
applySize(520);
applyAspect('4/3');

// 컨트롤 이벤트
sizeRange?.addEventListener('input', () => applySize(Number(sizeRange.value)));
aspectSelect?.addEventListener('change', () => applyAspect(aspectSelect.value));

// ========= 감정 막대 (face.css의 .meter-row / .track / .fill / .pct 구조에 맞춤) =========
const EMOTION_KEYS = ["angry","disgust","fear","happy","sad","surprise","neutral"];

function ensureBars() {
  if (!metersEl || metersEl.children.length) return;
  EMOTION_KEYS.forEach((k) => {
    const row = document.createElement('div');
    row.className = 'meter-row';
    row.innerHTML = `
      <div class="label text-capitalize">${k}</div>
      <div class="track"><div class="fill" id="fill-${k}"></div></div>
      <div class="pct" id="pct-${k}">0%</div>
    `;
    metersEl.appendChild(row);
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

  if (engineEl && data.engine) engineEl.textContent = engineEl.id === 'engineName' ? `엔진: ${data.engine}` : data.engine;

  if (fpsEl && typeof data.fps === 'number')
    fpsEl.textContent = (fpsEl.id === 'fps' && engineEl?.id === 'engineName') ? `FPS: ${data.fps.toFixed(1)}` : data.fps.toFixed(1);

  const probs = data.emotions || data.probs || {};
  EMOTION_KEYS.forEach((k) => {
    const v = Math.max(0, Math.min(1, Number(probs[k] ?? 0)));
    const pct = Math.round(v * 100);
    const fill = document.getElementById(`fill-${k}`);
    const pctEl = document.getElementById(`pct-${k}`);
    if (fill)  fill.style.width = pct + '%';
    if (pctEl) pctEl.textContent = pct + '%';
  });

  if (data.top && data.top.label) {
    if (topEmotionEl) topEmotionEl.textContent = data.top.label;
    if (topScoreEl)   topScoreEl.textContent =
      typeof data.top.score === 'number' ? Math.round(data.top.score * 100) + '%' : '-';
  } else {
    // top이 없을 때 최대값 계산
    let bestK = '-', bestV = 0;
    for (const [k, v] of Object.entries(probs)) {
      const n = Number(v) || 0;
      if (n > bestV) { bestV = n; bestK = k; }
    }
    if (topEmotionEl) topEmotionEl.textContent = bestK;
    if (topScoreEl)   topScoreEl.textContent = Math.round(bestV * 100) + '%';
  }
}

// ========= 로그 =========
function pushLog(msg) {
  if (!logEl) return;
  const t = new Date().toLocaleTimeString();
  const line = `[${t}] ${msg}\n`;
  if (logEl.tagName === 'TEXTAREA' || logEl.tagName === 'INPUT') logEl.value = line + (logEl.value || '');
  else logEl.textContent = line + (logEl.textContent || '');
}
