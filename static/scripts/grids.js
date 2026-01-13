(() => {
  const gallery = document.querySelector("[data-grid-gallery]");
  if (!gallery) {
    return;
  }

  let grids = [];
  try {
    grids = JSON.parse(gallery.dataset.gridGallery || "[]");
  } catch (_err) {
    grids = [];
  }

  if (!Array.isArray(grids) || grids.length === 0) {
    return;
  }

  const runDir = gallery.dataset.gridRun || "";
  const imgEl = document.getElementById("grid-image");
  const countEl = document.getElementById("gallery-count");
  const pathEl = document.getElementById("grid-path");
  const downloadEl = document.getElementById("gallery-download");
  const prevBtn = document.getElementById("gallery-prev");
  const nextBtn = document.getElementById("gallery-next");

  let index = 0;

  const getUrl = (item) =>
    `/api/grid-image?run=${encodeURIComponent(runDir)}&file=${encodeURIComponent(item.filename)}`;

  const render = () => {
    const item = grids[index];
    if (!item) {
      return;
    }

    const url = getUrl(item);
    if (imgEl) {
      imgEl.src = url;
      imgEl.alt = `Grid ${item.grid_index || index + 1} preview`;
    }
    if (countEl) {
      countEl.textContent = `${index + 1} / ${grids.length}`;
    }
    if (pathEl) {
      pathEl.textContent = item.output_path || "";
    }
    if (downloadEl) {
      downloadEl.href = url;
    }
  };

  const goPrev = () => {
    index = (index - 1 + grids.length) % grids.length;
    render();
  };

  const goNext = () => {
    index = (index + 1) % grids.length;
    render();
  };

  prevBtn?.addEventListener("click", goPrev);
  nextBtn?.addEventListener("click", goNext);

  document.addEventListener("keydown", (event) => {
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      goPrev();
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      goNext();
    }
  });

  render();
})();
