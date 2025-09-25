const logEl = document.getElementById("chatLog");
const msgEl = document.getElementById("msg");
const sendEl = document.getElementById("send");
const useEmotionEl = document.getElementById("useEmotion");
const ctxInfoEl = document.getElementById("ctxInfo");

let emotionCtx = null;

function el(tag, cls, html){ const e=document.createElement(tag); if(cls) e.className=cls; if(html!==undefined) e.innerHTML=html; return e; }
function appendMessage(who, text){
  const row = el("div", `row ${who}`);
  row.appendChild(el("div", `msg ${who}`, text.replace(/\n/g,"<br/>")));
  logEl.appendChild(row); logEl.scrollTop = logEl.scrollHeight;
}
function setTyping(on){
  if(on){ appendMessage("bot","<span class='typing'>입력 중…</span>"); }
  else{ logEl.querySelector(".typing")?.closest(".row.bot")?.remove(); }
}
async function loadEmotionCtx(){
  try{
    const sum = await (await fetch("/api/summary",{cache:"no-store"})).json();
    if(sum?.means){
      const entries = Object.entries(sum.means).sort((a,b)=>b[1]-a[1]);
      const top = entries[0]?.[0] || "-";
      emotionCtx = { top, means: sum.means };
      ctxInfoEl.textContent = `컨텍스트: top=${top}`;
    }else{ emotionCtx=null; ctxInfoEl.textContent="컨텍스트: 없음"; }
  }catch{ emotionCtx=null; ctxInfoEl.textContent="컨텍스트: 로드 실패"; }
}
async function send(){
  const text = msgEl.value.trim(); if(!text) return;
  appendMessage("me", text); msgEl.value=""; setTyping(true);
  try{
    const body = { message: text };
    if(useEmotionEl.checked && emotionCtx) body.context = { emotion: emotionCtx };
    const res = await fetch("/api/chat",{ method:"POST", headers:{ "Content-Type":"application/json" }, body: JSON.stringify(body) });
    const data = await res.json(); setTyping(false);
    appendMessage("bot", data?.reply || "응답 형식을 이해하지 못했어요.");
  }catch(e){ setTyping(false); appendMessage("bot","네트워크 오류가 발생했어요."); }
}
sendEl.addEventListener("click", send);
msgEl.addEventListener("keydown", e=>{ if(e.key==="Enter" && !e.shiftKey){ e.preventDefault(); send(); }});
loadEmotionCtx();