async function tick(){
  try {
    const s = await (await fetch("/api/summary")).json();
    document.getElementById("avg").textContent =
      JSON.stringify(s.means || {}, null, 2);
  } catch (e) {
    console.warn("tick error:", e);
  }
}
setInterval(tick, 2000);
tick();