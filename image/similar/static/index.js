        /* ============================================================
         *  STATE
         * ============================================================ */
        const state = {
            scanId: null,           // current scan session ID
            groups: [],             // array of group data from API
            selectedPaths: [],      // paths selected in the current group view
            currentGroupIdx: -1,    // which group's images are selected
            pollingInterval: null,  // setInterval handle for scan polling
        };

        /* ============================================================
         *  DOM REFS
         * ============================================================ */
        const $ = (id) => document.getElementById(id);
        const el = {
            dirPath:       $('dir-path'),
            scanBtn:       $('scan-btn'),
            recursive:     $('recursive'),
            threshold:     $('threshold-slider'),
            minGroupSize:  $('min-group-size'),
            filterBtn:     $('filter-btn'),
            hashSize:      $('hash-size'),
            hashSizeVal:   $('hash-size-value'),
            statsBar:      $('stats-bar'),
            scanProgress:  $('scan-progress'),
            progressBar:   $('progress-bar'),
            progressText:  $('progress-text'),
            groupsCont:    $('groups-container'),
            actionBar:     $('action-bar'),
            saveSelBtn:    $('save-selected-btn'),
            outputDir:     $('output-dir'),
            tooltip:       $('tooltip'),
            cardTpl:       $('group-card-template'),
            thumbTpl:      $('group-thumb-template'),
        };

        let currentSid = null;

        function getHashMode() {
            const r = document.querySelector('input[name="hash-mode"]:checked');
            return r ? r.value : 'grey';
        }

        /* ============================================================
         *  scanDirectory
         * ============================================================ */
        function scanDirectory() {
            const path = el.dirPath.value.trim();
            if (!path) {
                updateStatsBar('Please enter a directory path.');
                return;
            }

            // Clear previous results
            el.groupsCont.innerHTML = '';
            el.filterBtn.disabled = true;

            updateStatsBar('Scanning...');
            showProgress(true, 0, 0);

            fetch('/api/scan', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({path: path, recursive: el.recursive.checked, hash_size: parseInt(el.hashSize.value), mode: getHashMode()})
            })
            .then(r => r.json())
            .then(data => {
                if (data.error) {
                    updateStatsBar('Error: ' + data.error);
                    showProgress(false, 0, 0);
                    return;
                }
                if (data.sid) {
                    currentSid = data.sid;
                    pollScanStatus(data.sid);
                }
            })
            .catch(err => {
                updateStatsBar('Network error: ' + err.message);
                showProgress(false, 0, 0);
            });
        }

        /* ============================================================
         *  pollScanStatus
         * ============================================================ */
        function pollScanStatus(sid) {
            const poll = () => {
                fetch('/api/scan-status?sid=' + encodeURIComponent(sid))
                    .then(r => r.json())
                    .then(data => {
                        if (data.error) {
                            updateStatsBar('Error: ' + data.error);
                            showProgress(false, 0, 0);
                            return;
                        }

                        if (data.scanning) {
                            // Still scanning — update progress
                            showProgress(true, data.progress || 0, data.total || 0, data.phase);
                            setTimeout(poll, 500);
                        } else {
                            // Done
                            if (data.error) {
                                updateStatsBar(data.error);
                                showProgress(false, 0, 0);
                                return;
                            }
                            showProgress(false, 0, 0);
                            updateStatsBar('Scanned ' + data.total + ' images. Adjust threshold and click Filter.');
                            el.filterBtn.disabled = false;
                            // Auto-filter once scan is done
                            filterGroups();
                        }
                    })
                    .catch(err => {
                        updateStatsBar('Network error: ' + err.message);
                        showProgress(false, 0, 0);
                    });
            };
            poll();
        }

        /* ============================================================
         *  STUB: filterGroups
         * ============================================================ */
        function filterGroups() {
            if (!currentSid) return;

            const threshold = parseInt(el.threshold.value);
            const minGroupSize = parseInt(el.minGroupSize.value) || 2;

            el.filterBtn.disabled = true;
            updateStatsBar('Filtering...');

            fetch('/api/filter', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    sid: currentSid,
                    hash_size: parseInt(el.hashSize.value),
                    mode: getHashMode(),
                    threshold: threshold,
                    min_group_size: minGroupSize
                })
            })
            .then(r => r.json())
            .then(data => {
                el.filterBtn.disabled = false;
                if (data.error) {
                    updateStatsBar('Error: ' + data.error);
                    return;
                }
                renderGroups(data);
            })
            .catch(err => {
                el.filterBtn.disabled = false;
                updateStatsBar('Network error: ' + err.message);
            });
        }

        /* ============================================================
         *  STUB: renderGroups
         * ============================================================ */
        function renderGroups(data) {
            el.groupsCont.innerHTML = '';

            const groups = data.groups || [];
            if (groups.length === 0) {
                updateStatsBar('No groups found. Try lowering the threshold or min group size.');
                el.actionBar.style.display = 'none';
                return;
            }

            let totalClustered = 0;
            groups.forEach((group, groupIndex) => {
                const card = document.importNode(el.cardTpl.content, true);
                const images = group.images || [];

                card.querySelector('.group-idx').textContent = groupIndex + 1;
                card.querySelector('.group-count').textContent = images.length;
                totalClustered += images.length;

                const container = card.querySelector('.group-images');
                images.forEach(image => {
                    const thumb = document.importNode(el.thumbTpl.content, true);
                    const thumbDiv = thumb.querySelector('.group-thumb');
                    const imgEl = thumb.querySelector('img');

                    thumbDiv.dataset.path = image.path;
                    imgEl.src = '/api/thumb?sid=' + encodeURIComponent(currentSid) + '&path=' + encodeURIComponent(image.path);
                    imgEl.loading = 'lazy';
                    // Add loading spinner
                    const spinner = document.createElement('div');
                    spinner.className = 'thumb-spinner';
                    thumbDiv.appendChild(spinner);
                    imgEl.addEventListener('load', function() {
                        spinner.style.display = 'none';
                    });
                    imgEl.addEventListener('error', function() {
                        spinner.style.display = 'none';
                    });

                    thumbDiv.addEventListener('mouseover', (e) => {
                        const info = image.rel_path + ' · ' + image.width + 'x' + image.height;
                        showTooltip(e, info);
                    });
                    thumbDiv.addEventListener('mouseout', () => {
                        el.tooltip.style.display = 'none';
                    });

                    container.appendChild(thumb);
                });

                el.groupsCont.appendChild(card);
            });

            const totalImages = data.total_images || 0;
            const pct = totalImages > 0 ? ((totalClustered / totalImages) * 100).toFixed(1) : 0;
            updateStatsBar('Filter groups · ' + groups.length + ' groups · ' + totalClustered + ' clustered images (' + pct + '% of total)');

            el.actionBar.style.display = groups.length > 0 ? 'flex' : 'none';
        }

        /* ============================================================
         *  saveGroup
         * ============================================================ */
        function saveGroup(groupIdx, paths) {
            let outputDir = el.outputDir.value.trim();
            if (!outputDir) {
                outputDir = prompt('Enter output directory path:', '');
                if (!outputDir) return; // cancelled
                el.outputDir.value = outputDir;
            }

            if (!currentSid) {
                updateStatsBar('Error: No active scan session.');
                return;
            }

            updateStatsBar('Saving ' + paths.length + ' images...');

            fetch('/api/export', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    sid: currentSid,
                    paths: paths,
                    output_dir: outputDir
                })
            })
            .then(r => r.json())
            .then(data => {
                if (data.error) {
                    updateStatsBar('Error: ' + data.error);
                    return;
                }
                const captionNote = data.captions > 0 ? ' (+ ' + data.captions + ' captions)' : '';
                updateStatsBar('Saved ' + data.copied + ' images' + captionNote + ' to ' + data.output);
                if (data.errors && data.errors.length > 0) {
                    console.warn('Export errors:', data.errors);
                }
            })
            .catch(err => {
                updateStatsBar('Network error: ' + err.message);
            });
        }

        /* ============================================================
         *  STUB: updateStatsBar
         * ============================================================ */
        function updateStatsBar(text) {
            el.statsBar.textContent = text;
        }

        /* ============================================================
         *  STUB: showProgress
         * ============================================================ */
        function showProgress(visible, current, total, phase) {
            if (visible) {
                el.scanProgress.style.display = 'block';
                el.progressBar.max = total > 0 ? total : 1;
                el.progressBar.value = current;
                const label = phase === 'quality' ? 'Scoring quality' : 'Hashing image';
                el.progressText.textContent = `${label} ${current} / ${total}...`;
            } else {
                el.scanProgress.style.display = 'none';
            }
        }

        /* ============================================================
         *  STUB: showTooltip / hideTooltip
         * ============================================================ */
        function showTooltip(event, info) {
            const tt = el.tooltip;
            tt.textContent = info;
            tt.style.display = 'block';
            // Position relative to cursor
            let x = event.clientX + 12;
            let y = event.clientY + 12;
            // Keep within viewport
            const rect = tt.getBoundingClientRect();
            if (x + rect.width > window.innerWidth) x = event.clientX - rect.width - 12;
            if (y + rect.height > window.innerHeight) y = event.clientY - rect.height - 12;
            tt.style.left = x + 'px';
            tt.style.top = y + 'px';
        }

        function hideTooltip() {
            el.tooltip.style.display = 'none';
        }

        /* ============================================================
         *  EVENT LISTENERS
         * ============================================================ */

        // Scan button
        el.scanBtn.addEventListener('click', scanDirectory);

        // Enter key on directory path
        el.dirPath.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') scanDirectory();
        });

        // Filter button
        el.filterBtn.addEventListener('click', filterGroups);

        // Threshold change triggers debounced filter
        let filterDebounce = null;
        el.threshold.addEventListener('change', function () {
            clearTimeout(filterDebounce);
            filterDebounce = setTimeout(filterGroups, 400);
        });

        // Min group size change triggers debounced filter
        el.minGroupSize.addEventListener('change', function () {
            clearTimeout(filterDebounce);
            filterDebounce = setTimeout(filterGroups, 400);
        });

        // Hash size slider — update label + cap threshold at 64
        el.hashSize.addEventListener('input', function () {
            el.hashSizeVal.textContent = this.value;
            el.threshold.max = 64;
            if (parseInt(el.threshold.value) > 64) {
                el.threshold.value = 64;
            }
        });
        el.hashSize.addEventListener('change', function () {
            clearTimeout(filterDebounce);
            filterDebounce = setTimeout(filterGroups, 400);
        });

        // Hash mode toggle triggers re-filter
        document.querySelectorAll('input[name="hash-mode"]').forEach(r => {
            r.addEventListener('change', function () {
                clearTimeout(filterDebounce);
                filterDebounce = setTimeout(filterGroups, 400);
            });
        });

        // Tooltip events on groups container (delegated)
        el.groupsCont.addEventListener('mouseover', (e) => {
            const thumb = e.target.closest('.group-thumb');
            if (thumb) {
                const img = thumb.querySelector('img');
                const path = thumb.dataset.path || img && img.src || '';
                showTooltip(e, path);
            }
        });
        el.groupsCont.addEventListener('mouseout', (e) => {
            if (e.target.closest('.group-thumb')) hideTooltip();
        });

        // Thumbnail click — toggle .selected (delegated)
        el.groupsCont.addEventListener('click', (e) => {
            const thumb = e.target.closest('.group-thumb');
            if (!thumb) return;
            thumb.classList.toggle('selected');
            // Update selected paths from current group
            const groupCard = thumb.closest('.group-card');
            state.currentGroupIdx = Array.from(el.groupsCont.querySelectorAll('.group-card')).indexOf(groupCard);
            const thumbs = groupCard.querySelectorAll('.group-thumb');
            const paths = [];
            thumbs.forEach((t) => {
                if (t.classList.contains('selected')) {
                    paths.push(t.dataset.path);
                }
            });
            state.selectedPaths = paths;
            // Enable/disable save selected button
            el.saveSelBtn.disabled = paths.length === 0;
        });

        // Save Group button (delegated)
        el.groupsCont.addEventListener('click', (e) => {
            const btn = e.target.closest('.btn-save-group');
            if (!btn) return;
            const groupCard = btn.closest('.group-card');
            const idx = Array.from(el.groupsCont.querySelectorAll('.group-card')).indexOf(groupCard);
            const thumbs = groupCard.querySelectorAll('.group-thumb');
            const paths = Array.from(thumbs).map(t => t.dataset.path);
            saveGroup(idx, paths);
        });

        // Save Selected Group button
        el.saveSelBtn.addEventListener('click', () => {
            if (state.selectedPaths.length > 0 && state.currentGroupIdx >= 0) {
                saveGroup(state.currentGroupIdx, state.selectedPaths);
            }
        });

        console.log('Image Similarity Finder — shell loaded. Stubs ready for Tasks 6, 7, 9.');
