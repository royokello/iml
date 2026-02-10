const total_images = window.templateData.total_images;
let total_labels = window.templateData.total_labels;
let label_stats = window.templateData.label_stats;
let label_reasons = window.templateData.label_reasons || {};
const label_options = window.templateData.label_options || [];

const totalLabelsCounter = document.querySelector('.counters');

let img_1 = "";
let img_2 = "";

const img1Element = document.getElementById('image-1');
const img2Element = document.getElementById('image-2');

const leftCountElement = document.getElementById('left-count');
const rightCountElement = document.getElementById('right-count');
const reasonCountElements = {};
label_options.forEach((opt) => {
  reasonCountElements[opt] = document.getElementById(`reason-${opt}`);
});


getRandomPair();

function getRandomPair() {
  fetch('/random')
    .then(r => {
      if (!r.ok) throw new Error('Network response was not ok');
      return r.json();
    })
    .then(d => {
      img_1 = d.img_1;
      img_2 = d.img_2;
      updateImages(img_1, img_2);
    })
    .catch(e => {
      console.error('Error fetching random pair:', e);
      alert('Error loading random pair. Please ensure there are at least 2 images.');
    });
}

function updateImages(img1Id, img2Id) {
  img1Element.src = `/image/${encodeURIComponent(img1Id)}`;
  img2Element.src = `/image/${encodeURIComponent(img2Id)}`;
}

function switchImages() {
  if (img_1 === "" || img_2 === "") return;
  fetch('/switch', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ img_1, img_2 })
  })
    .then(r => r.json())
    .then(d => {
      img_1 = d.img_1;
      img_2 = d.img_2;
      updateImages(d.img_1, d.img_2);
    })
    .catch(e => console.error('Error switching images:', e));
}

function addLabel(choice) {
  if (!img_1 || !img_2) {
    alert('Error: Cannot add label with invalid images. Please get a new random pair.');
    return;
  }
  const selectedReason = getSelectedReason();
  if (!selectedReason) {
    alert('Please select a reason before saving a label.');
    return;
  }
  fetch('/label', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ img_1, img_2, choice, label: selectedReason })
  })
    .then(r => {
      if (!r.ok) return r.json().then(e => { throw new Error(`Server error (${r.status}): ${e.error || 'Unknown error'}`); });
      return r.json();
    })
    .then(d => {
      if (d.success) {
        total_labels++;
        label_stats[choice]++;
        label_reasons[selectedReason] = (label_reasons[selectedReason] || 0) + 1;
        updateLabelStats();
        updateReasonStats();
        getRandomPair();
      } else if (d.error) {
        alert(`Error: ${d.error}`);
      }
    })
    .catch(e => {
      console.error('Error adding label:', e);
      alert(`Failed to add label: ${e.message}`);
    });
}

function updateLabelStats() {
  leftCountElement.textContent = label_stats.left;
  rightCountElement.textContent = label_stats.right;
  totalLabelsCounter.textContent = `Total Images: ${total_images} | Total Labels: ${total_labels}`;
}

function updateReasonStats() {
  Object.keys(reasonCountElements).forEach((key) => {
    if (reasonCountElements[key]) {
      reasonCountElements[key].textContent = label_reasons[key] || 0;
    }
  });
}

function getSelectedReason() {
  const selected = document.querySelector('input[name="rank_reason"]:checked');
  return selected ? selected.value : '';
}

function randomizeImage(imageNumber) {
  if (imageNumber !== 1 && imageNumber !== 2) return;
  fetch('/random_single')
    .then(r => {
      if (!r.ok) throw new Error('Network response was not ok');
      return r.json();
    })
    .then(d => {
      if (imageNumber === 1) {
        img_1 = d.img;
      } else {
        img_2 = d.img;
      }
      updateImages(img_1, img_2);
    })
    .catch(e => {
      console.error(`Error randomizing image ${imageNumber}:`, e);
      alert('Error loading random image. Please ensure there are enough images.');
    });
}
