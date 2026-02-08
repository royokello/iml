const meta = document.getElementById("meta");
const images = document.getElementById("images");
const empty = document.getElementById("empty");
const state = { idx: 0, total: 0 };

function setMeta(payload) {
  if (!payload || payload.total === 0) {
    meta.textContent = "No groups";
    return;
  }
  meta.textContent = `Group ${payload.idx + 1} / ${payload.total} (${payload.count} images)`;
}

function renderImages(payload) {
  images.replaceChildren();
  if (!payload || payload.total === 0) {
    empty.hidden = false;
    return;
  }
  empty.hidden = true;
  for (const item of payload.images) {
    const fig = document.createElement("figure");
    const img = document.createElement("img");
    img.src = item.url;
    img.className = "photo";
    img.loading = "lazy";
    const cap = document.createElement("figcaption");
    cap.textContent = item.label;
    fig.appendChild(img);
    fig.appendChild(cap);
    images.appendChild(fig);
  }
}

async function loadGroup(idx) {
  const resp = await fetch(`/api/group?idx=${idx}`);
  const payload = await resp.json();
  state.idx = payload.idx || 0;
  state.total = payload.total || 0;
  setMeta(payload);
  renderImages(payload);
}

document.getElementById("prevBtn").onclick = () => loadGroup(state.idx - 1);
document.getElementById("nextBtn").onclick = () => loadGroup(state.idx + 1);
document.addEventListener("keydown", (e) => {
  if (e.key === "ArrowLeft") loadGroup(state.idx - 1);
  if (e.key === "ArrowRight") loadGroup(state.idx + 1);
});

const params = new URLSearchParams(window.location.search);
const start = parseInt(params.get("idx") || "0", 10);
loadGroup(Number.isFinite(start) ? start : 0);
