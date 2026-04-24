const form = document.getElementById("load-form");
const statusNode = document.getElementById("status");
const metaNode = document.getElementById("meta");
const saveResultsNode = document.getElementById("save-results");
const galleryNode = document.getElementById("gallery");
const selectionSummaryNode = document.getElementById("selection-summary");
const saveButton = document.getElementById("save-button");
const selectAllButton = document.getElementById("select-all-button");
const clearSelectionButton = document.getElementById("clear-selection-button");

let previewItems = [];

function setStatus(message, tone = "") {
  statusNode.textContent = message;
  statusNode.className = tone ? `status ${tone}` : "status";
}

function selectedFrames() {
  return Array.from(document.querySelectorAll(".pick-box:checked")).map((input) => Number(input.value));
}

function updateSelectionSummary() {
  selectionSummaryNode.textContent = `${selectedFrames().length} selected`;
  for (const card of document.querySelectorAll(".card")) {
    const checkbox = card.querySelector(".pick-box");
    card.classList.toggle("selected", checkbox.checked);
  }
}

function renderMeta(state) {
  if (!state.video) {
    metaNode.textContent = "";
    return;
  }

  const fps = Number(state.video.fps || 0).toFixed(3);
  const duration = Number(state.video.duration_seconds || 0).toFixed(3);
  const ranges = Array.isArray(state.video.selected_ranges) ? state.video.selected_ranges : [];
  const rangeText = ranges.length
    ? `<br /><strong>${ranges.length}</strong> range(s): ${ranges.map((range) => range.label).join(", ")}`
    : "";
  metaNode.innerHTML = `
    <strong>${state.items.length}</strong> preview marks,
    <strong>${state.video.total_frames}</strong> frames,
    <strong>${fps}</strong> fps,
    <strong>${state.video.width}x${state.video.height}</strong> source,
    <strong>${duration}s</strong> duration.
    ${rangeText}
  `;
}

function renderGallery(items) {
  previewItems = items;
  if (!items.length) {
    galleryNode.innerHTML = `<div class="empty">No preview frames were generated for this video.</div>`;
    updateSelectionSummary();
    return;
  }

  galleryNode.innerHTML = items.map((item, index) => `
    <article class="card">
      <div class="thumb-wrap">
        <img src="${item.preview_url}" alt="Preview ${index + 1}" loading="lazy" />
      </div>
      <div class="card-body">
        <div class="card-top">
          <strong>${item.preview_name.replace(/\.[^.]+$/, "")}</strong>
          <label class="pick">
            <input class="pick-box" type="checkbox" value="${item.mark_index}" />
            Pick
          </label>
        </div>
        <p class="card-meta">
          Mark frame ${item.mark_index}<br />
          ${item.timestamp}${item.range_label ? `<br />${item.range_label}` : ""}
        </p>
      </div>
    </article>
  `).join("");

  for (const checkbox of document.querySelectorAll(".pick-box")) {
    checkbox.addEventListener("change", updateSelectionSummary);
  }
  updateSelectionSummary();
}

function loadPayload() {
  return {
    video_path: document.getElementById("video-path").value.trim(),
    output_dir: document.getElementById("output-dir").value.trim(),
    time_ranges: document.getElementById("time-ranges").value.trim(),
    preview_resolution: document.getElementById("preview-resolution").value.trim(),
    interval_seconds: document.getElementById("interval-seconds").value.trim(),
    buffer: document.getElementById("buffer").value.trim(),
    filename_width: document.getElementById("filename-width").value.trim(),
  };
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Request failed.");
  }
  return data;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setStatus("Generating preview set...", "");
  saveResultsNode.textContent = "";

  try {
    const state = await postJson("/api/load", loadPayload());
    renderMeta(state);
    renderGallery(state.items);
    setStatus(`Loaded ${state.items.length} preview marks.`, "success");
  } catch (error) {
    renderMeta({});
    renderGallery([]);
    setStatus(error.message, "error");
  }
});

saveButton.addEventListener("click", async () => {
  const frames = selectedFrames();
  if (!previewItems.length) {
    setStatus("Load previews before saving.", "error");
    return;
  }
  if (!frames.length) {
    setStatus("Select at least one preview before saving.", "error");
    return;
  }

  setStatus(`Saving ${frames.length} selected mark(s)...`, "");
  saveResultsNode.textContent = "";

  try {
    const result = await postJson("/api/save", {
      selected_frames: frames,
      output_dir: document.getElementById("output-dir").value.trim(),
      buffer: document.getElementById("buffer").value.trim(),
      filename_width: document.getElementById("filename-width").value.trim(),
    });

    const sample = result.results.slice(0, 6).map((item) => {
      return `${item.filename} from ${item.mark_timestamp} -> ${item.winner_timestamp}`;
    });
    const suffix = result.results.length > 6 ? `\n...and ${result.results.length - 6} more.` : "";
    saveResultsNode.textContent = `Saved ${result.saved_count} image(s) to ${result.output_dir}.\n${sample.join("\n")}${suffix}`;
    setStatus(`Saved ${result.saved_count} native-res image(s).`, "success");
  } catch (error) {
    setStatus(error.message, "error");
  }
});

selectAllButton.addEventListener("click", () => {
  for (const checkbox of document.querySelectorAll(".pick-box")) {
    checkbox.checked = true;
  }
  updateSelectionSummary();
});

clearSelectionButton.addEventListener("click", () => {
  for (const checkbox of document.querySelectorAll(".pick-box")) {
    checkbox.checked = false;
  }
  updateSelectionSummary();
});

renderGallery([]);
