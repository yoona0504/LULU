let keys = null;

function renderMeters(probs){
  if(!probs) return;
  if(!keys){
    keys = Object.keys(probs);
    document.getElementById("meters").innerHTML =
      keys.map(k=>`<div class="rowbar">
        <div class="k">${k}</div>
        <div class="track"><div class="fill" id="f-${k}"></div></div>
        <div class="v" id="v-${k}" style="min-width:48px; text-align:right;">0%</div>
      </div>`).join("");
  }
  keys.forEach(k=>{
    const p = Math.round((probs[k]||0)*100);
    const f = document.getElementById(`f-${k}`);
    const v = document.getElementById(`v-${k}`);
    if(f) f.style.width = p + "%";
    if(v) v.textContent = p + "%";
  });
}

async function poll(){
  try{
    const last = await (await fetch("/api/last")).json();
    renderMeters(last.data?.probs || null);

    const sum = await (await fetch("/api/summary")).json();
    if(sum.ok){
      const top = sum.top ? `우세 감정: <b>${sum.top}</b>` : "우세 감정: -";
      const means = sum.means
        ? Object.entries(sum.means).map(([k,v])=>`${k}: ${(v*100).toFixed(0)}%`).join(" · ")
        : "";
      document.getElementById("summary").innerHTML =
        `${top}<br/><span class="muted">${means}</span>`;
    }
  }catch(e){
    // 필요 시 콘솔 로깅(페이지에 노이즈 싫으면 UI 로그는 생략)
    console.warn("poll error:", e);
  }
}

setInterval(poll, 1500);
poll();
