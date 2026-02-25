(function () {
const img = document.getElementById('img');
const selectionLayer = document.getElementById('selection-layer');
const classSelect = document.getElementById('classSelect');
const boxesList = document.getElementById('boxesList');

const prevBtn = document.getElementById('prevBtn');
const nextBtn = document.getElementById('nextBtn');
const navGapInput = document.getElementById('navGapInput');
const randomBtn = document.getElementById('randomBtn');
const prevLabelledBtn = document.getElementById('prevLabelledBtn');
const nextLabelledBtn = document.getElementById('nextLabelledBtn');
const saveBtn = document.getElementById('saveBtn');
const deleteSelBtn = document.getElementById('deleteSelBtn');

const imgPos = document.getElementById('imgPos');
const labelledCount = document.getElementById('labelledCount');
const totalLabels = document.getElementById('totalLabels');
const classStatsEl = document.getElementById('classStats');

const ASPECT_BY_CLASS = { 0: 1/1, 1: 3/4, 2: 4/3, 3: 1/2 }; // width / height
function currentAspect() {
    const id = Number(classSelect.value);
    return ASPECT_BY_CLASS.hasOwnProperty(id) ? ASPECT_BY_CLASS[id] : 1;
}

function ensureDefaultBox() {
    if (sel) return;
    const rect = containerRect();
    const imgRect = img.getBoundingClientRect();
    if (!imgRect.width || !imgRect.height) return;

    const ar = currentAspect();             // width / height
    const h = Math.floor(imgRect.height * 0.5);
    const w = Math.max(1, Math.floor(h * ar));

    const left = (imgRect.left - rect.left) + Math.max(0, Math.round((imgRect.width  - w) / 2));
    const top  = (imgRect.top  - rect.top)  + Math.max(0, Math.round((imgRect.height - h) / 2));

    sel = { left, top, width: w, height: h, boxId: null };
    renderSelection();
}
classSelect.addEventListener('change', ensureDefaultBox);


let totalImages = window.__IML__.total_images || 0;
let currentIndex = Math.max(0, window.__IML__.current_index || 0);
let currentLabels = []; // {id,class,x1,y1,x,y} in image pixels

// selection state (display pixels in the container coordinate system)
let sel = null; // {left, top, width, height, boxId|null}
let dragMode = null; // 'move' | 'resize-<dir>' | 'create'
let dragOrigin = null; // {x,y, start}

function classNameFromId(id) {
    const opt = classSelect.querySelector(`option[value="${id}"]`);
    return opt ? opt.textContent : `class ${id}`;
}

function scale() {
    if (!img.naturalWidth || !img.clientWidth) return {sx:1, sy:1};
    return { sx: img.naturalWidth / img.clientWidth, sy: img.naturalHeight / img.clientHeight };
}

function clearSelection() {
    sel = null;
    selectionLayer.innerHTML = '';
    [...boxesList.children].forEach(li => li.classList.remove('active'));
}

function clamp(val, min, max) { return Math.max(min, Math.min(max, val)); }

function renderSelection() {
    selectionLayer.innerHTML = '';
    if (!sel) return;
    const box = document.createElement('div');
    box.className = 'crop-box selected';
    box.style.left = sel.left + 'px';
    box.style.top = sel.top + 'px';
    box.style.width = sel.width + 'px';
    box.style.height = sel.height + 'px';

    const grid = document.createElement('div');
    grid.className = 'grid';
    const v1 = document.createElement('div'); v1.className = 'v'; v1.style.left = (sel.width/3) + 'px';
    const v2 = document.createElement('div'); v2.className = 'v'; v2.style.left = (2*sel.width/3) + 'px';
    const h1 = document.createElement('div'); h1.className = 'h'; h1.style.top = (sel.height/3) + 'px';
    const h2 = document.createElement('div'); h2.className = 'h'; h2.style.top = (2*sel.height/3) + 'px';
    grid.appendChild(v1); grid.appendChild(v2); grid.appendChild(h1); grid.appendChild(h2);
    box.appendChild(grid);

    const handles = ['nw','n','ne','e','se','s','sw','w'];
    handles.forEach(dir => {
    const h = document.createElement('div');
    h.className = 'handle ' + dir;
    h.addEventListener('mousedown', (e) => startResize(e, dir));
    box.appendChild(h);
    });

    box.addEventListener('mousedown', (e) => startMove(e));

    selectionLayer.appendChild(box);
}

function containerRect() { return document.getElementById('image-container').getBoundingClientRect(); }

function eventToContainer(e) {
    const r = containerRect();
    return { x: clamp(e.clientX - r.left, 0, r.width), y: clamp(e.clientY - r.top, 0, r.height), w: r.width, h: r.height };
}

function startCreate(e) {
    const p = eventToContainer(e);
    sel = { left: p.x, top: p.y, width: 1, height: 1, boxId: null };
    dragMode = 'create';
    dragOrigin = { x: p.x, y: p.y };
    renderSelection();
    window.addEventListener('mousemove', onDrag);
    window.addEventListener('mouseup', endDrag);
}

function startMove(e) {
    e.preventDefault();
    const p = eventToContainer(e);
    dragMode = 'move';
    dragOrigin = { x: p.x, y: p.y, start: { left: sel.left, top: sel.top } };
    window.addEventListener('mousemove', onDrag);
    window.addEventListener('mouseup', endDrag);
}

function startResize(e, dir) {
    e.stopPropagation(); e.preventDefault();
    const p = eventToContainer(e);
    dragMode = 'resize-' + dir;
    dragOrigin = { x: p.x, y: p.y, start: { ...sel } };
    window.addEventListener('mousemove', onDrag);
    window.addEventListener('mouseup', endDrag);
}

function onDrag(e) {
    const p = eventToContainer(e);
    if (!sel || !dragMode) return;

    const ar = currentAspect(); // width / height

    if (dragMode === 'move') {
        const dx = p.x - dragOrigin.x, dy = p.y - dragOrigin.y;
        const nl = clamp(dragOrigin.start.left + dx, 0, p.w - sel.width);
        const nt = clamp(dragOrigin.start.top  + dy, 0, p.h - sel.height);
        sel.left = nl; sel.top = nt;

    } else if (dragMode === 'create') {
        // Anchor is the drag start; shape grows with locked aspect ratio
        let left  = Math.min(dragOrigin.x, p.x);
        let top   = Math.min(dragOrigin.y, p.y);
        let width = Math.abs(p.x - dragOrigin.x);
        let height= Math.abs(p.y - dragOrigin.y);

        if (height === 0 && width === 0) { renderSelection(); return; }

        if (width / Math.max(1, height) > ar) {
        // too wide -> fit width to height
        width = Math.round(height * ar);
        if (p.x < dragOrigin.x) left = dragOrigin.x - width;
        } else {
        // too tall -> fit height to width
        height = Math.round(width / ar);
        if (p.y < dragOrigin.y) top = dragOrigin.y - height;
        }

        left  = clamp(left,  0, p.w - 1);
        top   = clamp(top,   0, p.h - 1);
        width = clamp(width, 1, p.w - left);
        height= clamp(height,1, p.h - top);

        sel.left = left; sel.top = top; sel.width = width; sel.height = height;

    } else if (dragMode.startsWith('resize-')) {
        const dir = dragMode.split('-')[1];

        let {left, top, width, height} = dragOrigin.start;
        const r = containerRect();
        const right  = left + width;
        const bottom = top + height;

        // Free resize first
        if (dir.includes('w')) left = clamp(p.x, 0, right - 1);
        if (dir.includes('e')) width = clamp((dir.includes('w') ? right : p.x) - left, 1, r.width - left);
        if (dir.includes('n')) top = clamp(p.y, 0, bottom - 1);
        if (dir.includes('s')) height = clamp((dir.includes('n') ? bottom : p.y) - top, 1, r.height - top);

        // Enforce aspect ratio by pinning the opposite edges as an anchor
        const anchorX = dir.includes('w') ? (left + width) : left;
        const anchorY = dir.includes('n') ? (top + height) : top;

        if (width / height > ar) {
        // too wide -> reduce width
        width = Math.round(height * ar);
        if (dir.includes('w')) left = anchorX - width;
        // if dragging east, left stays as-is
        } else {
        // too tall -> reduce height
        height = Math.round(width / ar);
        if (dir.includes('n')) top = anchorY - height;
        // if dragging south, top stays as-is
        }

        // Clamp to container
        left  = clamp(left, 0, r.width  - width);
        top   = clamp(top,  0, r.height - height);

        sel.left = left; sel.top = top; sel.width = width; sel.height = height;
    }

    renderSelection();
}

function endDrag() {
    dragMode = null; dragOrigin = null;
    window.removeEventListener('mousemove', onDrag);
    window.removeEventListener('mouseup', endDrag);
}

function selectionToImageCoords() {
    const {sx, sy} = scale();
    const rect = containerRect();
    const imgRect = img.getBoundingClientRect();
    // selection is relative to container; translate to the actual drawn image box inside container
    const offsetLeft = imgRect.left - rect.left;
    const offsetTop = imgRect.top - rect.top;
    const x1d = clamp(sel.left - offsetLeft, 0, imgRect.width);
    const y1d = clamp(sel.top - offsetTop, 0, imgRect.height);
    const x2d = clamp(x1d + sel.width, 0, imgRect.width);
    const y2d = clamp(y1d + sel.height, 0, imgRect.height);
    return { x1: Math.round(x1d * sx), y1: Math.round(y1d * sy), x: Math.round(x2d * sx), y: Math.round(y2d * sy) };
}

function imageToSelectionCoords(b) {
    const {sx, sy} = scale();
    const rect = containerRect();
    const imgRect = img.getBoundingClientRect();
    const offsetLeft = imgRect.left - rect.left;
    const offsetTop = imgRect.top - rect.top;
    const left = Math.round(b.x1 / sx) + offsetLeft;
    const top = Math.round(b.y1 / sy) + offsetTop;
    const width = Math.round((b.x - b.x1) / sx);
    const height = Math.round((b.y - b.y1) / sy);
    return { left, top, width, height };
}

function renderBoxesList() {
    boxesList.innerHTML = '';
    currentLabels.forEach((b, i) => {
    const li = document.createElement('li');
    li.dataset.id = i;
    const title = document.createElement('span');
    title.textContent = `#${i+1} ${classNameFromId(b.class)}`;
    const dims = document.createElement('span');
    dims.textContent = `${b.x - b.x1}×${b.y - b.y1}`;
    li.appendChild(title); li.appendChild(dims);
    li.addEventListener('click', () => {
        [...boxesList.children].forEach(el => el.classList.remove('active'));
        li.classList.add('active');
        sel = { ...imageToSelectionCoords(b), boxId: b.id };
        renderSelection();
        classSelect.value = String(b.class);
    });
    boxesList.appendChild(li);
    });
}

function updateStatsPanel(payload) {
    if (typeof payload?.current_index === 'number') currentIndex = payload.current_index;
    imgPos.textContent = `${currentIndex + 1} / ${totalImages}`;
    if (typeof payload?.labelled_images === 'number') labelledCount.textContent = payload.labelled_images;
    if (typeof payload?.total_labels === 'number') totalLabels.textContent = payload.total_labels;
    const stats = payload?.class_stats || {};
    classStatsEl.innerHTML = '';
    const keys = Object.keys(stats).sort((a,b)=>Number(a)-Number(b));
    keys.forEach(k => {
    const li = document.createElement('li');
    li.textContent = `${classNameFromId(k)}: ${stats[k]}`;
    classStatsEl.appendChild(li);
    });
}

function fetchStats() { fetch('/stats').then(r=>r.json()).then(updateStatsPanel); }

function getNavGap() {
    const raw = Number.parseInt(navGapInput?.value ?? '0', 10);
    const gap = Number.isFinite(raw) && raw >= 0 ? raw : 0;
    if (navGapInput) navGapInput.value = String(gap);
    return gap;
}

function loadImage(index) {
    currentIndex = index;
    img.src = `/image/${index}?t=${Date.now()}`;
    img.onload = () => {
    clearSelection();
    fetch(`/labels/${currentIndex}`).then(r=>r.json()).then(({labels})=>{ currentLabels = labels; renderBoxesList(); });
    fetchStats();
    };
}

// Controls
prevBtn.onclick = () => navigate('prev');
nextBtn.onclick = () => navigate('next');
randomBtn.onclick = () => navigate('random');
prevLabelledBtn.onclick = () => navigate('prev_labelled');
nextLabelledBtn.onclick = () => navigate('next_labelled');

function navigate(action) {
    const payload = { action };
    if (action === 'prev' || action === 'next') payload.gap = getNavGap();
    fetch('/navigate', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) })
    .then(r=>r.json()).then(resp => {
        if (typeof resp.index === 'number') loadImage(resp.index);
    });
}

saveBtn.onclick = () => {
    if (!sel) return;
    const coords = selectionToImageCoords();
    const cls = Number(classSelect.value);
    if (sel.boxId != null) {
    fetch('/label/update', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ index: currentIndex, box_id: sel.boxId, class: cls, ...coords }) })
        .then(r=>r.json()).then(()=>{ loadImage(currentIndex); });
    } else {
    fetch('/label', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ index: currentIndex, class: cls, ...coords }) })
        .then(r=>r.json()).then(()=>{ loadImage(currentIndex); });
    }
};

deleteSelBtn.onclick = () => {
    if (!sel || sel.boxId == null) return;
    fetch('/label/delete', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ index: currentIndex, box_id: sel.boxId }) })
    .then(r=>r.json()).then(()=>{ clearSelection(); loadImage(currentIndex); });
};

// Create selection by dragging on empty space
selectionLayer.addEventListener('mousedown', (e) => {
    // If clicking on existing selection, handlers will take over; we only create when clicking on empty layer area
    if (e.target === selectionLayer) startCreate(e);
});
// Also allow creating by clicking directly on the image where there is no selection overlay
img.addEventListener('mousedown', (e) => {
    const topEl = document.elementFromPoint(e.clientX, e.clientY);
    if (topEl === img) startCreate(e);
});

// Initial render
totalImages = window.__IML__.total_images;
updateStatsPanel({ current_index: window.__IML__.current_index, labelled_images: window.__IML__.labelled_count, total_labels: window.__IML__.total_labels, class_stats: window.__IML__.class_stats });
if (totalImages > 0) loadImage(Math.max(0, window.__IML__.current_index));
})();
