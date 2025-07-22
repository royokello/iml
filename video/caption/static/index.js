// ---------- DOM helpers ----------
const firstImg = document.getElementById("first");
const midImg   = document.getElementById("mid");
const tagsBox  = document.getElementById("tags");
const statusEl = document.getElementById("status");
const statsUl  = document.getElementById("tagStats");

let currentIdx = 0;

// ---------- API wrappers ----------
async function fetchLoop(idx) {
  const res = await fetch(`/api/loop?idx=${idx}`);
  return res.json();
}
async function saveCaption() {
  const body = JSON.stringify({ idx: currentIdx, tags: tagsBox.value });
  await fetch("/api/save", { method:"POST", headers:{ "Content-Type":"application/json" }, body });
  statusEl.textContent = "💾 saved";
  refreshStats();
}
async function nav(dir) {
  const res = await fetch(`/api/nav?dir=${dir}`);
  const data = await res.json();
  renderLoop(data);
}
async function refreshStats() {
  const stats = await (await fetch("/api/stats")).json();
  statsUl.innerHTML = Object.entries(stats)
    .sort((a,b)=>b[1]-a[1])
    .map(([tag,c])=>`<li data-tag="${tag}">${tag} (${c})</li>`).join("");
}

// ---------- UI update ----------
function renderLoop(data){
    currentIdx      = data.idx;
    gifPrev.src  = `/frames/${data.gif}`;
    tagsBox.value   = data.tags || "";
    statusEl.textContent = `Loop ${data.idx}`;
}

// ---------- button hooks ----------
document.getElementById("prevBtn").onclick = ()=>nav("prev");
document.getElementById("nextBtn").onclick = ()=>nav("next");
document.getElementById("randBtn").onclick = ()=>nav("random");
document.getElementById("saveBtn").onclick = saveCaption;

// Save on Ctrl+S or Cmd+S
document.addEventListener("keydown", e=>{
  if((e.ctrlKey||e.metaKey) && e.key==="s"){ e.preventDefault(); saveCaption(); }
});
// Save on Enter inside textarea with Ctrl
tagsBox.addEventListener("keydown", e=>{
  if(e.key==="Enter" && e.ctrlKey){ e.preventDefault(); saveCaption(); }
});

statsUl.addEventListener("click", e => {
  const li  = e.target.closest("li[data-tag]");
  if (!li) return;

  const tag = li.dataset.tag;
  const current = tagsBox.value
      .split(",")
      .map(t => t.trim())
      .filter(Boolean);

  if (!current.includes(tag)) {
    current.push(tag);
    tagsBox.value = current.join(", ");
  }
  tagsBox.focus();
});

// ---------- init ----------
window.addEventListener("load", async ()=>{
  renderLoop(await fetchLoop(0));
  refreshStats();
});
