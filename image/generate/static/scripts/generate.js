(function () {
  'use strict';

  // ─── State ────────────────────────────────────────────
  var state = {
    model: 'flux-4b',
    mode: 'text',
    variant: 'distill',
    fluxSchema: null,
    ideogramSchema: null,
    refImages: [],        // { serverPath, dataUrl }
    polling: null,
    loaded: false,
  };

  var jsonValues = {};

  // ─── DOM refs ──────────────────────────────────────────
  var $ = function (id) { return document.getElementById(id); };
  var el = {
    modelSelect: $('model-select'),
    variantSection: $('variant-section'),
    variantDistill: $('variant-distill'),
    variantBase: $('variant-base'),
    textQuant: $('text-quant'),
    denoiserQuant: $('denoiser-quant'),
    width: $('width'),
    height: $('height'),
    ratio: $('ratio'),
    steps: $('steps'),
    guidance: $('guidance'),
    seed: $('seed'),
    refSection: $('ref-section'),
    refSize: $('ref-size'),
    addRef: $('add-ref'),
    refPicker: $('ref-picker'),
    refList: $('ref-list'),
    modeText: $('mode-text'),
    modeJson: $('mode-json'),
    formText: $('form-text'),
    formJson: $('form-json'),
    textPrompt: $('text-prompt'),
    promptDisplay: $('prompt-display'),
    jsonFields: $('json-fields'),
    runBtn: $('run-btn'),
    progressArea: $('progress-area'),
    progressFill: $('progress-fill'),
    progressText: $('progress-text'),
    resultArea: $('result-area'),
    resultGallery: $('result-gallery'),
    errorArea: $('error-area'),
    historyList: $('history-list'),
    offloadingRadios: function () { return document.querySelectorAll('input[name="offloading"]'); },
    offloadingValue: function () {
      var r = document.querySelector('input[name="offloading"]:checked');
      return r ? r.value : 'off';
    },
  };

  // ─── Helpers ───────────────────────────────────────────
  function isFlux() {
    return state.model && state.model.indexOf('flux') === 0;
  }

  function getModelType() {
    return isFlux() ? 'flux' : 'ideogram';
  }

  function getFluxVersion() {
    var parts = state.model.split('-');
    return parts.length > 1 ? parts[1] : '4b';
  }

  // ─── Init ──────────────────────────────────────────────
  function init() {
    fetch('/api/config')
      .then(function (resp) { return resp.json(); })
      .then(function (cfg) {
        state.fluxSchema = cfg.flux_schema;
        state.ideogramSchema = cfg.ideogram_schema;
        state.loaded = true;
        renderJsonForm(state.fluxSchema);
        bindEvents();
        loadQuantMethods();
        loadHistory();
      });
  }

  // ─── Quant method loader ───────────────────────────────
  function loadQuantMethods() {
    var params = new URLSearchParams();
    params.set('model', state.model);
    if (isFlux()) {
      params.set('variant', state.variant);
    }

    fetch('/api/quant-methods?' + params.toString())
      .then(function (resp) { return resp.json(); })
      .then(function (data) {
        populateQuantSelectors(data.text_encoder || [], data.denoiser || []);
      })
      .catch(function () {
        populateQuantSelectors([], []);
      });
  }

  function populateQuantSelectors(teMethods, denMethods) {
    var i;

    el.textQuant.innerHTML = '';
    if (teMethods.length === 0) {
      var opt = document.createElement('option');
      opt.value = '';
      opt.textContent = '(none available)';
      el.textQuant.appendChild(opt);
    } else {
      for (i = 0; i < teMethods.length; i++) {
        var opt = document.createElement('option');
        opt.value = teMethods[i];
        opt.textContent = teMethods[i];
        el.textQuant.appendChild(opt);
      }
    }

    el.denoiserQuant.innerHTML = '';
    if (denMethods.length === 0) {
      var opt2 = document.createElement('option');
      opt2.value = '';
      opt2.textContent = '(none available)';
      el.denoiserQuant.appendChild(opt2);
    } else {
      for (i = 0; i < denMethods.length; i++) {
        var opt2 = document.createElement('option');
        opt2.value = denMethods[i];
        opt2.textContent = denMethods[i];
        el.denoiserQuant.appendChild(opt2);
      }
    }
  }

  // ─── JSON form rendering ───────────────────────────────
  function renderJsonForm(schema) {
    el.jsonFields.innerHTML = '';
    jsonValues = {};
    if (!schema || !schema.fields) return;

    schema.fields.forEach(function (field) {
      var group = document.createElement('div');
      group.className = 'json-field-group';

      var label = document.createElement('label');
      label.className = 'field-label';
      label.textContent = field.label;
      group.appendChild(label);

      var input;
      if (field.type === 'textarea') {
        input = document.createElement('textarea');
        input.className = 'json-input';
        input.rows = 3;
      } else {
        input = document.createElement('input');
        input.type = 'text';
        input.className = 'json-input';
      }
      input.dataset.key = field.key;

      input.addEventListener('input', function () {
        jsonValues[field.key] = this.value;
        updatePromptDisplay();
      });

      jsonValues[field.key] = '';
      group.appendChild(input);

      // Static suggestion chips
      if (field.suggestions && field.suggestions.length > 0) {
        var chips = document.createElement('div');
        chips.className = 'suggestion-chips';
        field.suggestions.forEach(function (s) {
          var chip = document.createElement('span');
          chip.className = 'suggestion-chip';
          chip.textContent = s;
          chip.addEventListener('click', function () {
            input.value = s;
            jsonValues[field.key] = s;
            updatePromptDisplay();
          });
          chips.appendChild(chip);
        });
        group.appendChild(chips);
      }

      el.jsonFields.appendChild(group);
    });
  }

  // ─── Prompt display ────────────────────────────────────
  function updatePromptDisplay() {
    if (state.mode === 'text') {
      el.promptDisplay.value = el.textPrompt.value;
    } else {
      var schema = getModelType() === 'flux' ? state.fluxSchema : state.ideogramSchema;
      var obj = buildJsonObject(schema);
      el.promptDisplay.value = JSON.stringify(obj, null, 2) || '{}';
    }
  }

  function buildJsonObject(schema) {
    var obj = {};
    if (!schema || !schema.fields) return obj;
    schema.fields.forEach(function (field) {
      var keys = field.key.split('.');
      var current = obj;
      for (var i = 0; i < keys.length; i++) {
        if (i === keys.length - 1) {
          var val = jsonValues[field.key];
          if (val) current[keys[i]] = val;
        } else {
          if (!current[keys[i]]) current[keys[i]] = {};
          current = current[keys[i]];
        }
      }
    });
    return obj;
  }

  function getPromptString() {
    if (state.mode === 'text') {
      return el.textPrompt.value;
    }
    var schema = getModelType() === 'flux' ? state.fluxSchema : state.ideogramSchema;
    return buildJsonObject(schema);
  }

  // ─── Events ────────────────────────────────────────────
  function bindEvents() {
    el.modelSelect.addEventListener('change', function () {
      state.model = el.modelSelect.value;
      onModelChange();
    });

    el.variantDistill.addEventListener('click', function () { setVariant('distill'); });
    el.variantBase.addEventListener('click', function () { setVariant('base'); });

    el.modeText.addEventListener('click', function () { setMode('text'); });
    el.modeJson.addEventListener('click', function () { setMode('json'); });

    el.ratio.addEventListener('change', updateWidthFromRatio);
    el.height.addEventListener('input', updateWidthFromRatio);
    el.width.addEventListener('input', function () {
      el.ratio.value = '';
    });

    el.textPrompt.addEventListener('input', updatePromptDisplay);

    el.addRef.addEventListener('click', function () { el.refPicker.click(); });
    el.refPicker.addEventListener('change', function (e) {
      var files = Array.from(e.target.files || []);
      files.forEach(function (file) { uploadRefImage(file); });
      el.refPicker.value = '';
    });

    el.runBtn.addEventListener('click', submitJob);

    el.historyList.addEventListener('click', function (e) {
      var btn = e.target.closest('.btn-regen');
      if (btn) handleRegen(btn.dataset.entryId);
    });


  }

  function onModelChange() {
    var isFluxModel = isFlux();
    var schema = isFluxModel ? state.fluxSchema : state.ideogramSchema;
    renderJsonForm(schema);
    updatePromptDisplay();

    el.variantSection.style.display = isFluxModel ? 'block' : 'none';
    el.refSection.style.display = isFluxModel ? 'block' : 'none';

    if (isFluxModel) {
      setVariant(state.variant);
    } else {
      el.guidance.value = 7.0;
      el.steps.value = 50;
    }

    if (state.loaded) loadQuantMethods();
  }

  function setVariant(variant) {
    state.variant = variant;
    el.variantDistill.classList.toggle('active', variant === 'distill');
    el.variantBase.classList.toggle('active', variant === 'base');
    if (variant === 'distill') {
      el.steps.value = 4;
      el.guidance.value = 1.0;
    } else {
      el.steps.value = 50;
      el.guidance.value = 4.0;
    }
    loadQuantMethods();
  }

  function setMode(mode) {
    state.mode = mode;
    el.modeText.classList.toggle('active', mode === 'text');
    el.modeJson.classList.toggle('active', mode === 'json');
    el.formText.style.display = mode === 'text' ? 'block' : 'none';
    el.formJson.style.display = mode === 'json' ? 'block' : 'none';
    updatePromptDisplay();
  }

  // ─── Aspect ratio ──────────────────────────────────────
  function updateWidthFromRatio() {
    var ratio = parseFloat(el.ratio.value);
    if (!ratio || isNaN(ratio)) return;
    var h = parseInt(el.height.value) || 1024;
    var w = Math.round(h * ratio / 64) * 64;
    el.width.value = Math.max(64, Math.min(4096, w));
  }

  // ─── Reference images ──────────────────────────────────
  function uploadRefImage(file) {
    var reader = new FileReader();
    reader.onload = function (loadEvent) {
      var dataUrl = loadEvent.target.result;

      var formData = new FormData();
      formData.append('file', file);

      fetch('/api/upload-ref', { method: 'POST', body: formData })
        .then(function (resp) { return resp.json(); })
        .then(function (data) {
          if (data.error) {
            console.error('Upload failed:', data.error);
            return;
          }
          state.refImages.push({ serverPath: data.path, dataUrl: dataUrl });
          renderRefs();
        })
        .catch(function (err) { console.error('Upload error:', err); });
    };
    reader.readAsDataURL(file);
  }

  function removeRefImage(index) {
    state.refImages.splice(index, 1);
    renderRefs();
  }

  function renderRefs() {
    el.refList.innerHTML = '';
    state.refImages.forEach(function (ref, i) {
      var thumb = document.createElement('div');
      thumb.className = 'ref-thumb';

      var img = document.createElement('img');
      img.alt = 'ref-' + i;
      img.src = ref.dataUrl;
      thumb.appendChild(img);

      var btn = document.createElement('button');
      btn.className = 'ref-remove';
      btn.textContent = '\u00d7';
      btn.addEventListener('click', function () { removeRefImage(i); });
      thumb.appendChild(btn);

      el.refList.appendChild(thumb);
    });
  }

  // ─── Job submission ────────────────────────────────────
  function submitJob() {
    hideError();
    hideResult();
    el.runBtn.disabled = true;

    var payload = {
      model: state.model,
      mode: state.mode,
      base: state.variant === 'base',
      prompt: getPromptString(),
      width: parseInt(el.width.value) || 1024,
      height: parseInt(el.height.value) || 1024,
      steps: parseInt(el.steps.value) || 50,
      guidance_scale: parseFloat(el.guidance.value) || 4.0,
      seed: el.seed.value ? parseInt(el.seed.value) : null,
      text_quant_method: el.textQuant.value || null,
      denoiser_quant_method: el.denoiserQuant.value || null,
      reference_size: parseInt(el.refSize.value) || 512,
      reference_images: state.refImages.map(function (r) { return r.serverPath; }),
      offloading: el.offloadingValue() === 'on',
    };

    showProgress('Submitting...', 10);

    fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
      .then(function (resp) { return resp.json(); })
      .then(function (data) {
        if (data.error) {
          showError(data.error);
          el.runBtn.disabled = false;
          hideProgress();
          return;
        }
        pollJob(data.job_id);
      })
      .catch(function (err) {
        showError('Failed to submit job: ' + err.message);
        el.runBtn.disabled = false;
        hideProgress();
      });
  }

  function pollJob(jobId) {
    showProgress('Queued...', 15);
    state.polling = setInterval(function () {
      fetch('/api/generate/status/' + jobId)
        .then(function (resp) { return resp.json(); })
        .then(function (data) {
          if (data.status === 'queued') {
            showProgress('Queued...', 20);
          } else if (data.status === 'running') {
            showProgress('Generating...', 50);
          } else if (data.status === 'done') {
            clearInterval(state.polling);
            state.polling = null;
            el.runBtn.disabled = false;
            hideProgress();
            if (data.paths && data.paths.length > 0) {
              showResult(data.paths);
            }
            loadHistory();
          } else if (data.status === 'failed') {
            clearInterval(state.polling);
            state.polling = null;
            el.runBtn.disabled = false;
            hideProgress();
            showError(data.error || 'Generation failed');
          }
        })
        .catch(function (err) {
          clearInterval(state.polling);
          state.polling = null;
          el.runBtn.disabled = false;
          hideProgress();
          showError('Status polling failed: ' + err.message);
        });
    }, 1500);
  }

  // ─── Progress / Result / Error ─────────────────────────
  function showProgress(text, pct) {
    el.progressArea.style.display = 'block';
    el.progressText.textContent = text;
    el.progressFill.style.width = pct + '%';
  }

  function hideProgress() {
    el.progressArea.style.display = 'none';
  }

  function showResult(paths) {
    el.resultArea.style.display = 'block';
    el.resultGallery.innerHTML = '';
    if (paths && paths.length > 0) {
      var parts = paths[0].split('/');
      var fname = parts[parts.length - 1];
      var img = document.createElement('img');
      img.src = '/api/generate/image/' + fname;
      img.alt = 'Generated image';
      img.loading = 'lazy';
      el.resultGallery.appendChild(img);
      el.resultArea.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }

  function hideResult() {
    el.resultArea.style.display = 'none';
    el.resultGallery.innerHTML = '';
  }

  function showError(msg) {
    el.errorArea.style.display = 'block';
    el.errorArea.textContent = msg;
  }

  function hideError() {
    el.errorArea.style.display = 'none';
    el.errorArea.textContent = '';
  }

  // ─── History ───────────────────────────────────────────
  function loadHistory() {
    fetch('/api/history')
      .then(function (resp) { return resp.json(); })
      .then(function (data) { renderHistory(data.entries || []); })
      .catch(function () {});
  }

  function renderHistory(entries) {
    if (!entries || entries.length === 0) {
      el.historyList.innerHTML = '<p class="muted">No history yet.</p>';
      return;
    }
    el.historyList.innerHTML = '';
    entries.slice().reverse().forEach(function (entry) {
      var card = document.createElement('div');
      card.className = 'history-card';

      var outputPath = entry.outputs && entry.outputs[0];
      if (outputPath) {
        var parts = outputPath.split('/');
        var fname = parts[parts.length - 1];
        var img = document.createElement('img');
        img.src = '/api/generate/image/' + fname;
        img.alt = 'History image';
        img.loading = 'lazy';
        img.addEventListener('click', function () {
          showResult(entry.outputs);
        });
        card.appendChild(img);
      }

      var info = document.createElement('div');
      info.className = 'history-card-info';
      var badge = document.createElement('span');
      badge.className = 'model-badge';
      badge.textContent = entry.model || '?';
      info.appendChild(badge);
      info.appendChild(document.createTextNode(' ' + (entry.timestamp || '')));

      var regenBtn = document.createElement('button');
      regenBtn.className = 'btn-regen';
      regenBtn.dataset.entryId = entry.id;
      regenBtn.textContent = '\u21bb';
      regenBtn.title = 'Regenerate with these settings';

      card.appendChild(info);
      card.appendChild(regenBtn);
      el.historyList.appendChild(card);
    });
  }

  function handleRegen(entryId) {
    fetch('/api/history/' + entryId + '/regen')
      .then(function (resp) { return resp.json(); })
      .then(function (entry) {
        if (entry.error) {
          showError(entry.error);
          return;
        }

        // Restore model
        if (entry.model) {
          var modelType = entry.model;
          var fluxVersion = entry.flux_version || '4b';
          var newModel = modelType === 'ideogram' ? 'ideogram' : 'flux-' + fluxVersion;
          el.modelSelect.value = newModel;
          state.model = newModel;
          onModelChange();
        }

        // Restore variant (Flux only)
        if (isFlux()) {
          var baseFlag = entry.flux_base === true;
          var variant = baseFlag ? 'base' : 'distill';
          setVariant(variant);
        }

        // Restore mode
        var mode = entry.mode || 'text';
        setMode(mode);

        // Restore prompt
        if (mode === 'text') {
          el.textPrompt.value = typeof entry.prompt === 'string' ? entry.prompt : JSON.stringify(entry.prompt || '');
        } else if (typeof entry.prompt === 'object') {
          restoreJsonValues(entry.prompt);
        } else if (entry.prompt) {
          try { restoreJsonValues(JSON.parse(entry.prompt)); } catch (e) {}
        }

        // Restore settings
        var s = entry.settings || {};
        if (s.width) el.width.value = s.width;
        if (s.height) el.height.value = s.height;
        if (s.steps) el.steps.value = s.steps;
        if (s.guidance_scale) el.guidance.value = s.guidance_scale;
        if (s.seed) el.seed.value = s.seed;
        if (s.text_quant_method) el.textQuant.value = s.text_quant_method;
        if (s.denoiser_quant_method) el.denoiserQuant.value = s.denoiser_quant_method;
        if (s.reference_size) el.refSize.value = s.reference_size;
        if (s.offloading != null) {
          var off = document.querySelector('input[name="offloading"][value="' + (s.offloading ? 'on' : 'off') + '"]');
          if (off) off.checked = true;
        }

        // Restore ref images
        state.refImages = [];
        (entry.reference_images || []).forEach(function (path) {
          state.refImages.push({ serverPath: path, dataUrl: '' });
        });
        renderRefs();
        updatePromptDisplay();
      })
      .catch(function (err) {
        showError('Failed to load settings: ' + err.message);
      });
  }

  function restoreJsonValues(obj) {
    var schema = getModelType() === 'flux' ? state.fluxSchema : state.ideogramSchema;
    if (!schema || !schema.fields) return;

    schema.fields.forEach(function (field) {
      var keys = field.key.split('.');
      var current = obj;
      for (var i = 0; i < keys.length; i++) {
        if (current && typeof current === 'object' && keys[i] in current) {
          current = current[keys[i]];
        } else {
          current = '';
          break;
        }
      }
      var val = typeof current === 'string' ? current : '';
      jsonValues[field.key] = val;
      var input = el.jsonFields.querySelector('[data-key="' + field.key + '"]');
      if (input) input.value = val;
    });
  }

  // ─── Start ─────────────────────────────────────────────
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
