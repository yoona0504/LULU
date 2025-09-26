/* static/js/face.js */
(() => {
  'use strict';

  // ========= 중복 실행 방지 (IIFE 안에서만 return 허용) =========
  if (window.__faceinit) { console.debug("face.js: duplicate load skipped"); return; }
  window.__faceinit = true;

  // ========= DOM refs =========
  const html         = document.documentElement;
  const grid         = document.querySelector('.face-grid');
  const camBox       = document.getElementById('camBox');

  // 컨트롤 (네 템플릿 id와 호환)
  const sizeRange    = document.getElementById('camScale')    || document.getElementById('sizeRange');
  const aspectSelect = document.getElementById('aspect')      || document.getElementById('aspectSelect');
  const resetBtn     = document.getElementById('camReset');

  // 표시 요소 (네 템플릿 id 우선 → 과거 id 폴백)
  const metersEl     = document.getElementById('meters')      || document.getElementById('emotionList');
  const topEmotionEl = document.getElementById('topEmotion')  || document.getElementById('primary');
  const topScoreEl   = document.getElementById('confidence')  || document.getElementById('topScore');
  const fpsEl        = document.getElementById('fps');
  const engineEl     = document.getElementById('engine')      || document.getElementById('engineName');
  const logEl        = document.getElementById('log');
  const clearBtn     = document.getElementById('clearLog');

  clearBtn?.addEventListener('click', () => {
    if (!logEl) return;
    if (logEl.tagName === 'TEXTAREA' || logEl.tagName === 'INPUT') logEl.value = '';
    else logEl.textContent = '';
  });

  // ========= 카메라 크기/종횡비 =========
  function applySize(px) {
    const v = `${px}px`;
    html.style.setProperty('--cam-max', v);           // 전역 변수
    if (camBox) camBox.style.width = v;               // 확실한 적용
    if (grid)   grid.style.gridTemplateColumns = `minmax(280px, ${v}) 1fr`;
  }
  function applyAspect(ratio) {
    if (camBox) camBox.style.aspectRatio = ratio;
  }

  // 초기값 적용 (첫 렌더 후 1프레임)
  requestAnimationFrame(() => {
    applySize(Number((sizeRange && sizeRange.value) || 560));
    applyAspect((aspectSelect && aspectSelect.value) || '4/3');
  });

  sizeRange?.addEventListener('input', () => applySize(Number(sizeRange.value)));
  aspectSelect?.addEventListener('change', () => applyAspect(aspectSelect.value));
  resetBtn?.addEventListener('click', () => {
    if (sizeRange) sizeRange.value = 560;
    if (aspectSelect) aspectSelect.value = '4/3';
    applySize(560);
    applyAspect('4/3');
  });

  // ========= 감정 막대 =========
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

  // ========= 데이터 가져오기 (face_state → emotion 폴백) =========
  async function fetchState() {
    // 1) 가벼운 상태 엔드포인트
    try {
      const r = await fetch('/api/face_state', { cache: 'no-store' });
      if (r.ok && (r.headers.get('content-type') || '').includes('application/json')) {
        const j = await r.json();  // { ok,label,score,probs }
        if (j && typeof j === 'object') {
          return {
            engine: engineEl?.textContent || undefined, // face_state에는 engine/fps 없음
            fps: undefined,
            emotions: j.probs || {},
            top: (j.label != null) ? { label: j.label, score: j.score } : null
          };
        }
      }
    } catch (_) { /* 무시: 폴백 시도 */ }

    // 2) 상세 상태 엔드포인트
    const r2 = await fetch('/api/emotion', { cache: 'no-store' });
    const ct = r2.headers.get('content-type') || '';
    if (!ct.includes('application/json')) {
      const text = await r2.text();
      throw new Error(`/api/emotion JSON 아님: ${ct} 예: ${text.slice(0,80)}...`);
    }
    return await r2.json(); // {engine,fps,emotions,top}
  }

  // ========= 폴링 =========
  const POLL_MS = 1000;
  let timer = null;

  async function poll() {
    try {
      const data = await fetchState();
      updateUI(data);
    } catch (e) {
      pushLog('poll 실패: ' + (e?.message || e));
    }
  }

  function start(){ if (!timer) timer = setInterval(poll, POLL_MS); }
  function stop(){ if (timer) { clearInterval(timer); timer = null; } }
  document.addEventListener('visibilitychange', () => document.hidden ? stop() : start());
  poll();  // 즉시 1회
  start();

  // ========= UI 업데이트 =========
  function updateUI(data) {
    if (!data) return;

    // 엔진/FPS
    if (engineEl && data.engine) engineEl.textContent = data.engine;
    if (fpsEl && typeof data.fps === 'number') fpsEl.textContent = data.fps.toFixed(1);

    // 확률 (emotions/probs 아무 키나 받아서 보정)
    const raw = data.emotions || data.probs || {};
    const probs = {};
    let sum = 0;
    EMOTION_KEYS.forEach(k => {
      const v = Math.max(0, Math.min(1, Number(raw[k] ?? 0)));
      probs[k] = v; sum += v;
    });
    if (sum > 0) { EMOTION_KEYS.forEach(k => probs[k] = probs[k] / sum); } // 정규화

    // 막대 갱신
    EMOTION_KEYS.forEach((k) => {
      const pct = Math.round((probs[k] || 0) * 100);
      const fill = document.getElementById(`fill-${k}`);
      const pctEl = document.getElementById(`pct-${k}`);
      if (fill)  fill.style.width = pct + '%';
      if (pctEl) pctEl.textContent = pct + '%';
    });

    // 주 감정/신뢰도
    if (data.top && data.top.label != null) {
      if (topEmotionEl) topEmotionEl.textContent = data.top.label;
      if (topScoreEl)   topScoreEl.textContent   = 
        (typeof data.top.score === 'number') ? Math.round(data.top.score * 100) + '%' : '-';
    } else {
      let bestK = '-', bestV = 0;
      for (const [k, v] of Object.entries(probs)) {
        const n = Number(v) || 0;
        if (n > bestV) { bestV = n; bestK = k; }
      }
      if (topEmotionEl) topEmotionEl.textContent = bestK;
      if (topScoreEl)   topScoreEl.textContent   = Math.round(bestV * 100) + '%';
    }
  }

  // ========= 로그 =========
  function pushLog(msg) {
    if (!logEl) return;
    const t = new Date().toLocaleTimeString();
    const line = `[${t}] ${msg}\n`;
    if (logEl.tagName === 'TEXTAREA' || logEl.tagName === 'INPUT') {
      logEl.value = line + (logEl.value || '');
    } else {
      logEl.textContent = line + (logEl.textContent || '');
    }
  }
})();
