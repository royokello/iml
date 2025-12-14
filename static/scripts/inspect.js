(() => {
  const form = document.getElementById("inspect-form");
  const pathInput = document.getElementById("model-path");
  const statusEl = document.getElementById("status");
  const metadataEl = document.getElementById("metadata");
  const tensorRowsEl = document.getElementById("tensor-rows");
  const tensorSummaryEl = document.getElementById("tensor-summary");

  const numberFmt = new Intl.NumberFormat("en-US");

  const setStatus = (message, variant = "") => {
    statusEl.textContent = message || "";
    statusEl.className = `status ${variant}`.trim();
  };

  const encodePath = (path) => encodeURI(path);

  const renderMetadata = (meta = {}) => {
    const sizeMB =
      meta.file_size_bytes != null ? `${(meta.file_size_bytes / 1024 / 1024).toFixed(2)} MB` : "-";

    const rows = [
      ["Source", meta.source || "-"],
      ["Size (MB)", sizeMB],
      ["Format", meta.format || "-"],
      ["Modified", meta.modified_at || "-"],
      ["Tensors", meta.tensor_count != null ? numberFmt.format(meta.tensor_count) : "-"],
      ["Total elements", meta.total_elements != null ? numberFmt.format(meta.total_elements) : "-"],
      [
        "Dtypes",
        meta.dtypes ? Object.entries(meta.dtypes).map(([k, v]) => `${k}: ${v}`).join(", ") : "-",
      ],
    ];

    const extraMeta = meta._metadata || {};
    Object.keys(extraMeta).forEach((key) => {
      const raw = extraMeta[key];
      const value = typeof raw === "object" ? JSON.stringify(raw) : String(raw);
      rows.push([key, value]);
    });

    metadataEl.innerHTML = "";
    rows.forEach(([label, value]) => {
      const tr = document.createElement("tr");
      const keyTd = document.createElement("td");
      keyTd.textContent = label;
      const valueTd = document.createElement("td");
      valueTd.textContent = value;
      tr.append(keyTd, valueTd);
      metadataEl.appendChild(tr);
    });
  };

  const renderTensors = (tensors = []) => {
    tensorRowsEl.innerHTML = "";
    tensors.forEach((t) => {
      const tr = document.createElement("tr");
      const nameTd = document.createElement("td");
      nameTd.textContent = t.name;
      const shapeTd = document.createElement("td");
      shapeTd.textContent = Array.isArray(t.shape) ? t.shape.join(" x ") : String(t.shape);
      const dtypeTd = document.createElement("td");
      dtypeTd.textContent = t.dtype;
      tr.append(nameTd, shapeTd, dtypeTd);
      tensorRowsEl.appendChild(tr);
    });

    tensorSummaryEl.textContent = tensors.length
      ? `${numberFmt.format(tensors.length)} tensors listed`
      : "No tensors loaded yet.";
  };

  const fetchJSON = async (url, options) => {
    const res = await fetch(url, options);
    let data = null;
    try {
      data = await res.json();
    } catch (_err) {
      // ignore parse errors; handled below
    }

    if (!res.ok) {
      const message =
        (data && data.error) ||
        `Request failed (${res.status}${res.statusText ? ` ${res.statusText}` : ""})`;
      throw new Error(message);
    }
    return data;
  };

  const loadModel = async (path) => {
    if (!path) {
      setStatus("Enter a filepath to inspect.", "error");
      return;
    }
    setStatus("Loading model...", "info");

    try {
      const data = await fetchJSON(`/inspect/${encodePath(path)}`);
      renderMetadata(data.metadata);
      renderTensors(data.tensors);
      setStatus("Model loaded.", "success");
    } catch (err) {
      setStatus(err.message || "Failed to load model.", "error");
      renderMetadata({});
      renderTensors([]);
    }
  };

  form?.addEventListener("submit", (event) => {
    event.preventDefault();
    loadModel(pathInput.value.trim());
  });

  // Allow pressing Enter in the input to submit immediately.
  pathInput?.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      loadModel(pathInput.value.trim());
    }
  });
})();
