// ── State ─────────────────────────────────────────────────────────────
const state = {
  version: '4b',
  modelVersions: [],
  loras: [],
  activeLoras: [],   // [{name, weight}]
  refs: [],          // [{path, name}]
  jobId: null,
  polling: false,
  busy: false,
};

// ── DOM refs ──────────────────────────────────────────────────────────
const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

const dom = {
  versionRadios:  $$('input[name="version"]'),
  modeRadios:     $$('input[name="mode"]'),
  teSelect:       $('#te-select'),
  denoiserSelect: $('#denoiser-select'),
  promptInput:    $('#prompt-input'),
  negPromptInput: $('#neg-prompt-input'),
  widthInput:     $('#width-input'),
  heightInput:    $('#height-input'),
  seedInput:      $('#seed-input'),
  loraList:       $('#lora-list'),
  loraEmpty:      $('#lora-empty'),
  loraAddBtn:     $('#lora-add-btn'),
  refList:        $('#ref-list'),
  refEmpty:       $('#ref-empty'),
  refAddBtn:      $('#ref-add-btn'),
  refResInput:    $('#ref-res-input'),
  genBtn:         $('#gen-btn'),
  genStatus:      $('#gen-status'),
  resultImage:    $('#result-image'),
  resultPlaceholder: $('#result-placeholder'),
  logOutput:      $('#log-output'),
  historyList:    $('#history-list'),
  historyEmpty:   $('#history-empty'),
};

// ── API helpers ───────────────────────────────────────────────────────
function apiUrl(path) {
  return path;
}

async function apiGet(path) {
  const r = await fetch(apiUrl(path));
  if (!r.ok) throw new Error(`GET ${path} → ${r.status}`);
  return r.json();
}

async function apiPost(path, body) {
  const r = await fetch(apiUrl(path), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`POST ${path} → ${r.status}`);
  return r.json();
}

// ── Load models ───────────────────────────────────────────────────────
async function loadModels() {
  try {
    const data = await apiGet('/api/models');
    state.modelVersions = data.versions;
    // If current version not available, pick first
    if (!data.versions.includes(state.version) && data.versions.length > 0) {
      state.version = data.versions[0];
      dom.versionRadios.forEach((r) => { r.checked = r.value === state.version; });
    }
    refreshModelDependents();
  } catch (e) {
    console.error(e);
  }
}

// Refresh things that depend on model selection
function refreshModelDependents() {
  loadTextEncoderMethods();
  loadDenoiserMethods();
  loadLoras();
  loadHistory();
}

// ── Text encoder methods ──────────────────────────────────────────────
async function loadTextEncoderMethods() {
  const version = state.version;
  if (!version) return;
  try {
    const data = await apiGet(`/api/text-encoder-methods?version=${encodeURIComponent(version)}`);
    const sel = dom.teSelect;
    const current = sel.value;
    sel.innerHTML = '<option value="">None</option>';
    data.methods.forEach((m) => {
      const opt = document.createElement('option');
      opt.value = m;
      opt.textContent = m;
      sel.appendChild(opt);
    });
    if (current && data.methods.includes(current)) sel.value = current;
  } catch (e) {
    console.error(e);
  }
}

// ── Denoiser methods ──────────────────────────────────────────────────
async function loadDenoiserMethods() {
  const version = state.version;
  const variant = getMode();
  if (!version) return;
  try {
    const data = await apiGet(
      `/api/denoiser-methods?version=${encodeURIComponent(version)}&variant=${encodeURIComponent(variant)}`
    );
    const sel = dom.denoiserSelect;
    const current = sel.value;
    sel.innerHTML = '<option value="">None</option>';
    data.methods.forEach((m) => {
      const opt = document.createElement('option');
      opt.value = m;
      opt.textContent = m;
      sel.appendChild(opt);
    });
    if (current && data.methods.includes(current)) sel.value = current;
  } catch (e) {
    console.error(e);
  }
}

function getMode() {
  let mode = 'distill';
  dom.modeRadios.forEach((r) => { if (r.checked) mode = r.value; });
  return mode;
}

// ── Load LoRAs ────────────────────────────────────────────────────────
async function loadLoras() {
  const version = state.version;
  if (!version) {
    dom.loraAddBtn.disabled = true;
    dom.loraEmpty.textContent = 'Select a model first';
    return;
  }
  try {
    const data = await apiGet(`/api/loras?version=${encodeURIComponent(version)}`);
    state.loras = data.loras;
    dom.loraAddBtn.disabled = data.loras.length === 0;
    dom.loraEmpty.textContent =
      data.loras.length === 0 ? 'No LoRAs available for this model' : '';
    // Re-render existing LoRA rows with updated select options
    renderLoraRows();
  } catch (e) {
    console.error(e);
  }
}

// ── Render LoRA rows ──────────────────────────────────────────────────
function renderLoraRows() {
  dom.loraList.innerHTML = '';
  state.activeLoras.forEach((entry, i) => {
    const row = document.createElement('div');
    row.className = 'lora-row';

    const sel = document.createElement('select');
    state.loras.forEach((name) => {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      if (name === entry.name) opt.selected = true;
      sel.appendChild(opt);
    });
    sel.addEventListener('change', () => {
      state.activeLoras[i].name = sel.value;
    });

    const wt = document.createElement('input');
    wt.type = 'number';
    wt.value = entry.weight;
    wt.step = 0.05;
    wt.min = 0;
    wt.max = 2;
    wt.addEventListener('change', () => {
      state.activeLoras[i].weight = parseFloat(wt.value) || 1.0;
    });

    const rm = document.createElement('button');
    rm.className = 'btn-danger';
    rm.textContent = '\u2715';
    rm.addEventListener('click', () => {
      state.activeLoras.splice(i, 1);
      renderLoraRows();
    });

    row.appendChild(sel);
    row.appendChild(wt);
    row.appendChild(rm);
    dom.loraList.appendChild(row);
  });
  dom.loraEmpty.style.display = state.activeLoras.length === 0 ? '' : 'none';
}

dom.loraAddBtn.addEventListener('click', () => {
  if (state.loras.length === 0) return;
  const name = state.loras[0];
  if (!state.activeLoras.find((e) => e.name === name)) {
    state.activeLoras.push({ name, weight: 1.0 });
    renderLoraRows();
  }
});

// ── References ────────────────────────────────────────────────────────
dom.refAddBtn.addEventListener('click', () => {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = 'image/png,image/jpeg,image/webp';
  input.multiple = true;
  input.addEventListener('change', () => {
    Array.from(input.files).forEach((file) => {
      state.refs.push({ path: file.name, file });
    });
    // Can't send files via JSON API — we'll store the filename and
    // require the user to reference full paths. For now, prompt for path.
    renderRefGrid();
  });
  input.click();
});

// A simpler approach: prompt for image path
function addRefByPath() {
  const path = prompt('Enter full path to reference image:');
  if (!path || !path.trim()) return;
  state.refs.push({ path: path.trim(), file: null });
  renderRefGrid();
}

// Override the file input approach with a path prompt for now
dom.refAddBtn.addEventListener('click', addRefByPath);

function renderRefGrid() {
  dom.refList.innerHTML = '';
  state.refs.forEach((ref, i) => {
    const thumb = document.createElement('div');
    thumb.className = 'ref-thumb';

    const img = document.createElement('img');
    // Try to load as URL or file path — the browser can load file://
    // from local paths on the same filesystem.
    img.src = ref.file
      ? URL.createObjectURL(ref.file)
      : ref.path;
    img.alt = ref.path.split(/[/\\]/).pop();

    const rm = document.createElement('button');
    rm.className = 'ref-remove';
    rm.textContent = '\u2715';
    rm.addEventListener('click', (e) => {
      e.stopPropagation();
      state.refs.splice(i, 1);
      renderRefGrid();
    });

    thumb.appendChild(img);
    thumb.appendChild(rm);
    dom.refList.appendChild(thumb);
  });
  dom.refEmpty.style.display = state.refs.length === 0 ? '' : 'none';
}

// ── History ───────────────────────────────────────────────────────────
async function loadHistory() {
  const version = state.version;
  if (!version) return;
  try {
    const data = await apiGet(`/api/history?version=${encodeURIComponent(version)}`);
    dom.historyList.innerHTML = '';
    data.images.forEach((img) => {
      const item = document.createElement('div');
      item.className = 'history-item';
      const imageEl = document.createElement('img');
      imageEl.src = `/api/image?path=${encodeURIComponent(img.path)}`;
      imageEl.alt = img.name;
      imageEl.title = img.name;
      item.appendChild(imageEl);
      item.addEventListener('click', () => {
        showResult(`/api/image?path=${encodeURIComponent(img.path)}`);
      });
      dom.historyList.appendChild(item);
    });
    dom.historyEmpty.style.display = data.images.length === 0 ? '' : 'none';
  } catch (e) {
    console.error(e);
  }
}

// ── Result display ────────────────────────────────────────────────────
function showResult(url) {
  dom.resultPlaceholder.classList.add('hidden');
  dom.resultImage.classList.remove('hidden');
  dom.resultImage.src = url;
}

function clearResult() {
  dom.resultPlaceholder.classList.remove('hidden');
  dom.resultImage.classList.add('hidden');
  dom.resultImage.src = '';
}

// ── Generate ──────────────────────────────────────────────────────────
async function startGeneration() {
  if (state.busy) return;
  state.busy = true;
  dom.genBtn.disabled = true;
  dom.genStatus.textContent = 'Starting...';
  dom.logOutput.textContent = '';
  clearResult();

  const version = state.version;

  const body = {
    version,
    prompt: dom.promptInput.value.trim() || 'A cat holding a sign that says hello world',
    negative_prompt: dom.negPromptInput.value.trim() || null,
    width: parseInt(dom.widthInput.value) || 512,
    height: parseInt(dom.heightInput.value) || 512,
    seed: parseInt(dom.seedInput.value) || 0,
    base: getMode() === 'base',
    loras: state.activeLoras.map((e) => ({ name: e.name, weight: e.weight })),
    ref_images: state.refs.map((r) => r.path),
    ref_res: parseInt(dom.refResInput.value) || 512,
    text_quant: dom.teSelect.value || null,
    denoiser_quant: dom.denoiserSelect.value || null,
  };

  try {
    const { job_id } = await apiPost('/api/generate', body);
    state.jobId = job_id;
    state.polling = true;
    dom.genStatus.textContent = 'Generating...';
    pollJob(job_id);
  } catch (e) {
    dom.genStatus.textContent = `Error: ${e.message}`;
    state.busy = false;
    dom.genBtn.disabled = false;
  }
}

async function pollJob(jobId) {
  try {
    const data = await apiGet(`/api/status/${jobId}`);
    // Append log
    if (data.log) {
      dom.logOutput.textContent = data.log;
      dom.logOutput.parentElement.scrollTop = dom.logOutput.parentElement.scrollHeight;
    }

    if (data.status === 'done') {
      dom.genStatus.textContent = 'Done';
      dom.genBtn.disabled = false;
      state.busy = false;
      state.polling = false;
      if (data.output) {
        const url = `/api/image?path=${encodeURIComponent(data.output)}`;
        showResult(url);
      }
      loadHistory();
    } else if (data.status === 'error') {
      dom.genStatus.textContent = `Error: ${data.error || 'unknown'}`;
      dom.genBtn.disabled = false;
      state.busy = false;
      state.polling = false;
    } else if (data.status === 'running' || data.status === 'queued') {
      setTimeout(() => pollJob(jobId), 1500);
    } else {
      dom.genStatus.textContent = `Unexpected status: ${data.status}`;
      dom.genBtn.disabled = false;
      state.busy = false;
      state.polling = false;
    }
  } catch (e) {
    dom.genStatus.textContent = `Poll error: ${e.message}`;
    dom.genBtn.disabled = false;
    state.busy = false;
    state.polling = false;
  }
}

// ── Event wiring ──────────────────────────────────────────────────────
dom.versionRadios.forEach((r) => {
  r.addEventListener('change', () => {
    if (r.checked) {
      state.version = r.value;
      refreshModelDependents();
    }
  });
});

dom.modeRadios.forEach((r) => {
  r.addEventListener('change', () => {
    loadDenoiserMethods();
  });
});

// ── Size presets ──────────────────────────────────────────────────────
document.querySelectorAll('.preset-btn').forEach((btn) => {
  btn.addEventListener('click', () => {
    dom.widthInput.value = btn.dataset.w;
    dom.heightInput.value = btn.dataset.h;
  });
});

dom.genBtn.addEventListener('click', startGeneration);

// Auto-load on page load
document.addEventListener('DOMContentLoaded', () => {
  loadModels();
});
