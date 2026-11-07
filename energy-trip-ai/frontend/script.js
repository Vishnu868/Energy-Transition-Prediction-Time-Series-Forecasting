/**
 * EnergyTripAI — Frontend JavaScript
 * 
 * Handles:
 * - File upload page interactions
 * - API communication
 * - Dashboard rendering (charts, tables, KPIs)
 */

const API_BASE = "http://localhost:8000/api";

// ============================================================
// SHARED UTILITIES
// ============================================================

/**
 * Show a status message in the upload page.
 * @param {string} msg - Message text
 * @param {string} type - 'info' | 'error'
 */
function showStatus(msg, type = 'info') {
    const el = document.getElementById('uploadStatus');
    if (!el) return;
    el.textContent = msg;
    el.className = `status-msg ${type}`;
    el.classList.remove('hidden');
}

/**
 * Format a number to display (e.g., 2345.6 -> "2,345.6")
 */
function fmt(num, decimals = 1) {
    return Number(num).toLocaleString('en-US', { maximumFractionDigits: decimals });
}

// ============================================================
// UPLOAD PAGE LOGIC
// ============================================================

(function initUploadPage() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const fileInfo = document.getElementById('fileInfo');
    const analyzeBtn = document.getElementById('analyzeBtn');

    if (!dropZone) return;  // Not on upload page

    let uploadedSessionId = null;
    let uploadedPath = null;

    // Click to open file picker
    dropZone.addEventListener('click', () => fileInput.click());

    // Drag & drop support
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file) handleFileSelect(file);
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files[0]) handleFileSelect(fileInput.files[0]);
    });

    /**
     * Upload the selected CSV to the backend.
     * @param {File} file
     */
    async function handleFileSelect(file) {
        if (!file.name.endsWith('.csv')) {
            showStatus('Please select a CSV file.', 'error');
            return;
        }

        showStatus(`Uploading ${file.name}...`, 'info');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await fetch(`${API_BASE}/upload`, {
                method: 'POST',
                body: formData
            });

            if (!res.ok) {
                const err = await res.json();
                showStatus(`Upload failed: ${err.detail}`, 'error');
                return;
            }

            const data = await res.json();
            uploadedSessionId = data.session_id;
            uploadedPath = data.saved_path;

            // Show success info
            fileInfo.textContent = `✅ ${file.name} uploaded — ~${data.estimated_rows.toLocaleString()} rows detected`;
            fileInfo.classList.remove('hidden');
            analyzeBtn.disabled = false;
            showStatus('File uploaded successfully. Configure settings and click Analyze.', 'info');

        } catch (err) {
            showStatus(`Network error: ${err.message}`, 'error');
        }
    }

    /**
     * On "Analyze" click: POST predict request, redirect to dashboard.
     */
    analyzeBtn.addEventListener('click', async () => {
        if (!uploadedSessionId) return;

        const requestBody = {
            session_id: uploadedSessionId,
            breaker_rating_amps: parseFloat(document.getElementById('breakerRating').value),
            voltage: parseFloat(document.getElementById('voltage').value),
            location_type: document.getElementById('locationType').value,
            resample_freq: document.getElementById('resampleFreq').value,
            forecast_steps: parseInt(document.getElementById('forecastSteps').value),
        };

        showStatus('Running AI analysis... please wait.', 'info');
        analyzeBtn.disabled = true;

        try {
            const res = await fetch(`${API_BASE}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody)
            });

            if (!res.ok) {
                const err = await res.json();
                showStatus(`Analysis failed: ${err.detail}`, 'error');
                analyzeBtn.disabled = false;
                return;
            }

            const data = await res.json();

            // Store results in sessionStorage for dashboard page
            sessionStorage.setItem('analysisResult', JSON.stringify(data));
            sessionStorage.setItem('requestConfig', JSON.stringify(requestBody));

            // Redirect to dashboard
            window.location.href = 'dashboard.html';

        } catch (err) {
            showStatus(`Request error: ${err.message}`, 'error');
            analyzeBtn.disabled = false;
        }
    });
})();


// ============================================================
// DASHBOARD PAGE LOGIC
// ============================================================

(function initDashboard() {
    const dashContent = document.getElementById('dashboardContent');
    const loadingScreen = document.getElementById('loadingScreen');

    if (!dashContent) return;  // Not on dashboard page

    // Retrieve stored analysis result
    const rawResult = sessionStorage.getItem('analysisResult');
    const rawConfig = sessionStorage.getItem('requestConfig');

    if (!rawResult) {
        // No data: redirect back to upload
        document.getElementById('loadingScreen').innerHTML = `
            <p>No analysis data found. <a href="index.html" style="color: #388bfd">Go back to upload</a>.</p>
        `;
        return;
    }

    const result = JSON.parse(rawResult);
    const config = rawConfig ? JSON.parse(rawConfig) : {};

    // Hide loading, show dashboard
    loadingScreen.style.display = 'none';
    dashContent.style.display = 'block';

    // ---- Render all sections ----
    renderKPIs(result, config);
    renderPowerChart(result);
    renderProbChart(result);
    renderLabelChart(result);
    renderFaultTable(result);
    renderShapChart(result);
    renderDataInfo(result, config);

})();


/**
 * Render KPI cards at the top of the dashboard.
 */
function renderKPIs(result, config) {
    const cs = result.current_state;
    const clf = result.classification;
    const stability = result.stability_score;

    // Location badge
    const badge = document.getElementById('locationBadge');
    if (badge) {
        const icons = { home:'🏠', office:'🏢', hospital:'🏥', factory:'🏭', school:'🏫' };
        const loc = config.location_type || 'home';
        badge.textContent = `${icons[loc] || '📍'} ${loc.charAt(0).toUpperCase() + loc.slice(1)}`;
    }

    // Status
    const statusEl = document.getElementById('statusBadge');
    if (statusEl) {
        statusEl.textContent = cs.status;
        const cls = cs.status === 'Normal' ? 'status-normal' : cs.status === 'Warning' ? 'status-warning' : 'status-trip';
        statusEl.className = `kpi-value ${cls}`;
    }

    // Power
    setText('kpiPower', `${fmt(cs.total_power_w)} W`);
    setText('kpiCurrent', `${fmt(cs.current_a, 2)} A`);

    // Load %
    const loadEl = document.getElementById('kpiLoad');
    if (loadEl) {
        loadEl.textContent = `${fmt(cs.load_percentage)}%`;
        loadEl.className = `kpi-value ${cs.load_percentage > 100 ? 'status-trip' : cs.load_percentage > 80 ? 'status-warning' : 'status-normal'}`;
    }

    // Trip probability
    const tripPct = (clf.trip_probability * 100).toFixed(1);
    const tripEl = document.getElementById('kpiTripProb');
    if (tripEl) {
        tripEl.textContent = `${tripPct}%`;
        tripEl.className = `kpi-value ${clf.trip_probability > 0.7 ? 'status-trip' : clf.trip_probability > 0.3 ? 'status-warning' : 'status-normal'}`;
    }

    // Stability score
    const stabEl = document.getElementById('kpiStability');
    const stabBar = document.getElementById('stabilityBar');
    if (stabEl) {
        stabEl.textContent = `${stability}/100`;
        stabEl.className = `kpi-value ${stability < 40 ? 'status-trip' : stability < 70 ? 'status-warning' : 'status-normal'}`;
    }
    if (stabBar) {
        stabBar.style.width = `${stability}%`;
        stabBar.style.background = stability > 70 ? '#3fb950' : stability > 40 ? '#d29922' : '#f85149';
    }
}


/**
 * Render main power timeline + forecast chart.
 */
function renderPowerChart(result) {
    const ctx = document.getElementById('powerChart');
    if (!ctx) return;

    const cd = result.chart_data;
    const cs = result.current_state;

    // Historical data
    const histLabels = cd.timestamps;
    const histValues = cd.power_values;

    // Forecast data (appended after history)
    const forecastLabels = cd.forecast_timestamps;
    const forecastValues = cd.forecast_values;

    // Combine for x-axis
    const allLabels = [...histLabels, ...forecastLabels];
    
    // Historical series (nulls for forecast positions)
    const historicalSeries = [...histValues, ...new Array(forecastLabels.length).fill(null)];
    
    // Forecast series (nulls for historical positions)
    const forecastSeries = [...new Array(histLabels.length).fill(null), ...forecastValues];

    // Threshold lines (constant arrays)
    const tripLine = new Array(allLabels.length).fill(cs.trip_threshold_w);
    const warnLine = new Array(allLabels.length).fill(cs.warning_threshold_w);

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: allLabels,
            datasets: [
                {
                    label: 'Historical Power (W)',
                    data: historicalSeries,
                    borderColor: '#388bfd',
                    backgroundColor: 'rgba(56,139,253,0.08)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 0,
                    borderWidth: 2
                },
                {
                    label: 'Forecast (W)',
                    data: forecastSeries,
                    borderColor: '#bc8cff',
                    backgroundColor: 'rgba(188,140,255,0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 3,
                    borderWidth: 2,
                    borderDash: [5, 3]
                },
                {
                    label: 'Trip Threshold',
                    data: tripLine,
                    borderColor: '#f85149',
                    borderWidth: 1.5,
                    borderDash: [6, 3],
                    pointRadius: 0,
                    fill: false,
                    tension: 0
                },
                {
                    label: 'Warning (80%)',
                    data: warnLine,
                    borderColor: '#d29922',
                    borderWidth: 1.5,
                    borderDash: [4, 4],
                    pointRadius: 0,
                    fill: false,
                    tension: 0
                }
            ]
        },
        options: {
            responsive: true,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { 
                    labels: { color: '#e6edf3', boxWidth: 14, font: { size: 11 } }
                },
                tooltip: {
                    backgroundColor: '#1c2128',
                    borderColor: '#30363d',
                    borderWidth: 1,
                    titleColor: '#e6edf3',
                    bodyColor: '#8b949e',
                    callbacks: {
                        label: (ctx) => ` ${ctx.dataset.label}: ${ctx.parsed.y !== null ? ctx.parsed.y.toFixed(1) + ' W' : '—'}`
                    }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#8b949e', maxTicksLimit: 8, font: { size: 10 } },
                    grid: { color: '#21262d' }
                },
                y: {
                    ticks: { color: '#8b949e', font: { size: 10 } },
                    grid: { color: '#21262d' },
                    title: { display: true, text: 'Power (W)', color: '#8b949e' }
                }
            }
        }
    });
}


/**
 * Render class probability doughnut/bar chart.
 */
function renderProbChart(result) {
    const ctx = document.getElementById('probChart');
    if (!ctx) return;

    const probs = result.classification.probabilities;
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Normal', 'Warning', 'Trip'],
            datasets: [{
                label: 'Probability',
                data: [probs.normal * 100, probs.warning * 100, probs.trip * 100],
                backgroundColor: ['rgba(63,185,80,0.7)', 'rgba(210,153,34,0.7)', 'rgba(248,81,73,0.7)'],
                borderColor: ['#3fb950', '#d29922', '#f85149'],
                borderWidth: 1,
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#1c2128',
                    callbacks: {
                        label: (ctx) => ` ${ctx.parsed.y.toFixed(1)}%`
                    }
                }
            },
            scales: {
                y: {
                    min: 0, max: 100,
                    ticks: { color: '#8b949e', callback: v => v + '%' },
                    grid: { color: '#21262d' },
                    title: { display: true, text: 'Probability (%)', color: '#8b949e' }
                },
                x: { ticks: { color: '#8b949e' }, grid: { display: false } }
            }
        }
    });
}


/**
 * Render label distribution pie chart.
 */
function renderLabelChart(result) {
    const ctx = document.getElementById('labelChart');
    if (!ctx) return;

    const ls = result.label_stats;
    
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Normal', 'Warning', 'Trip'],
            datasets: [{
                data: [ls.normal_count, ls.warning_count, ls.trip_count],
                backgroundColor: ['rgba(63,185,80,0.8)', 'rgba(210,153,34,0.8)', 'rgba(248,81,73,0.8)'],
                borderColor: ['#3fb950', '#d29922', '#f85149'],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            cutout: '65%',
            plugins: {
                legend: { 
                    position: 'right',
                    labels: { color: '#e6edf3', font: { size: 11 }, padding: 12 }
                },
                tooltip: {
                    backgroundColor: '#1c2128',
                    callbacks: {
                        label: (ctx) => ` ${ctx.label}: ${ctx.parsed.toLocaleString()} samples`
                    }
                }
            }
        }
    });
}


/**
 * Render the appliance fault ranking table.
 */
function renderFaultTable(result) {
    const container = document.getElementById('faultTable');
    if (!container) return;

    const ranking = result.fault_ranking;
    
    if (!ranking || ranking.length === 0) {
        container.innerHTML = '<p style="color:#8b949e;padding:16px;">No appliance data available for fault analysis.</p>';
        return;
    }

    const badgeClass = {
        CRITICAL: 'badge-critical',
        HIGH: 'badge-high',
        MODERATE: 'badge-moderate',
        LOW: 'badge-low'
    };

    const html = ranking.map((item, i) => `
        <div class="fault-row">
            <div class="fault-name">
                ${i === 0 ? '🔴' : i === 1 ? '🟡' : '⚪'} 
                #${i+1} ${item.appliance}
            </div>
            <div class="fault-power">${fmt(item.avg_power_w)} W avg</div>
            <div class="fault-contrib">${fmt(item.contribution_pct)}%</div>
            <div>
                <span class="fault-badge ${badgeClass[item.status] || 'badge-low'}">${item.status}</span>
            </div>
            <div style="min-width:80px;">
                <div class="fault-bar-outer">
                    <div class="fault-bar-inner" style="width:${Math.min(item.risk_score * 100, 100)}%"></div>
                </div>
                <div style="font-size:0.75rem;color:#8b949e;margin-top:3px;">Risk: ${(item.risk_score*100).toFixed(0)}%</div>
            </div>
        </div>
    `).join('');
    
    container.innerHTML = html;
}


/**
 * Render SHAP feature importance horizontal bar chart.
 */
function renderShapChart(result) {
    const ctx = document.getElementById('shapChart');
    if (!ctx) return;

    const shap = result.shap_explanation;
    
    if (!shap || shap.length === 0) {
        ctx.parentElement.innerHTML += '<p style="color:#8b949e;margin-top:8px;">SHAP analysis not available (model may not be trained).</p>';
        return;
    }

    const labels = shap.map(s => s.feature);
    const values = shap.map(s => s.shap_value);
    const maxVal = Math.max(...values);

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'SHAP Importance (Trip Class)',
                data: values,
                backgroundColor: values.map(v => `rgba(188,140,255,${0.3 + 0.7 * v / (maxVal + 1e-6)})`),
                borderColor: '#bc8cff',
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',  // Horizontal bars
            responsive: true,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#1c2128',
                    callbacks: {
                        label: (ctx) => ` Importance: ${ctx.parsed.x.toFixed(6)}`
                    }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#8b949e', font: { size: 10 } },
                    grid: { color: '#21262d' },
                    title: { display: true, text: 'Mean |SHAP Value|', color: '#8b949e' }
                },
                y: {
                    ticks: { color: '#8b949e', font: { size: 10 } },
                    grid: { display: false }
                }
            }
        }
    });
}


/**
 * Render data info footer card.
 */
function renderDataInfo(result, config) {
    const card = document.getElementById('dataInfoCard');
    if (!card) return;

    const di = result.data_info;
    const ls = result.label_stats;

    card.innerHTML = `
        <div class="info-item">Total Samples: <span>${di.total_rows.toLocaleString()}</span></div>
        <div class="info-item">Appliances: <span>${di.appliances_detected.join(', ')}</span></div>
        <div class="info-item">Start: <span>${new Date(di.time_range.start).toLocaleString()}</span></div>
        <div class="info-item">End: <span>${new Date(di.time_range.end).toLocaleString()}</span></div>
        <div class="info-item">Breaker: <span>${config.breaker_rating_amps}A @ ${config.voltage}V</span></div>
        <div class="info-item">Trip Threshold: <span>${fmt(config.breaker_rating_amps * config.voltage)} W</span></div>
        <div class="info-item">Normal: <span>${ls.normal_pct}%</span></div>
        <div class="info-item">Warning: <span>${ls.warning_pct}%</span></div>
        <div class="info-item">Trip Events: <span>${ls.trip_pct}%</span></div>
    `;
}


/** Helper to set text content safely */
function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
}