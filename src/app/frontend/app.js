'use strict';

// ── Tooltip (event-delegated, single DOM node) ────────────────────────────────
const _tip = (() => {
  const el = document.createElement('div');
  el.className = 'tooltip-popup';
  document.body.appendChild(el);

  function show(target, text) {
    if (!text) return;
    el.textContent = text;
    el.classList.add('visible');
    const r = target.getBoundingClientRect();
    const w = el.offsetWidth;
    const left = r.left + window.scrollX + r.width / 2 - w / 2;
    el.style.left = Math.max(8, Math.min(left, window.innerWidth - w - 8)) + 'px';
    el.style.top  = (r.bottom + window.scrollY + 8) + 'px';
  }

  function hide() { el.classList.remove('visible'); }

  document.addEventListener('mouseover', e => {
    const t = e.target.closest('[data-tooltip]');
    if (t) show(t, t.dataset.tooltip);
  });
  document.addEventListener('mouseout', e => {
    const t = e.target.closest('[data-tooltip]');
    if (t) hide();
  });

  return { show, hide };
})();

// ── State ─────────────────────────────────────────────────────────────────────
const state = {
  catalog:         [],   // [{ticker, name, news_count, date_from, date_to}]
  selectedTickers: [],   // currently active filter
  portfolio:       [],   // [{ticker, name, net_sentiment_7d, alert, ...}]
  activeDays:      7,    // period window (7 | 14 | 30)
  detailTicker:    null,
  sentimentChart:  null,
  volumeChart:     null,
};

// ── API ───────────────────────────────────────────────────────────────────────
async function apiFetch(path) {
  const r = await fetch(path);
  if (!r.ok) {
    const msg = await r.text().catch(() => r.statusText);
    throw new Error(`${r.status} ${msg}`);
  }
  return r.json();
}

const fetchCatalog   = ()             => apiFetch('/catalog');
const fetchPortfolio = (tickers)      => apiFetch(
  tickers.length
    ? `/portfolio?tickers=${tickers.join(',')}&days=${state.activeDays}`
    : `/portfolio?days=${state.activeDays}`
);
const fetchCompany   = (ticker)       => apiFetch(`/company/${ticker}`);
const fetchBriefing  = (ticker) => apiFetch(`/company/${ticker}/briefing`);

// ── Visual helpers ────────────────────────────────────────────────────────────
function netColor(net) {
  if (net > 0.03)  return '#16A34A';
  if (net < -0.03) return '#DC2626';
  return '#9CA3AF';
}

function netClass(net) {
  if (net > 0.03)  return 'val-pos';
  if (net < -0.03) return 'val-neg';
  return 'val-neu';
}

function sentChipHTML(label) {
  const map = { positive: ['sent-pos','POS'], negative: ['sent-neg','NEG'], neutral: ['sent-neu','NEU'] };
  const [cls, txt] = map[label] || map.neutral;
  return `<span class="sent-chip ${cls}">${txt}</span>`;
}

function gaugeHTML(net) {
  const pct   = Math.min(Math.abs(net) * 50, 50);
  const left  = net >= 0 ? 50 : (50 - pct);
  const color = net > 0.03 ? '#22C55E' : net < -0.03 ? '#EF4444' : '#9CA3AF';
  return `<div class="gauge">
    <div class="gauge-center"></div>
    <div class="gauge-fill" style="left:${left}%;width:${Math.max(pct, 1)}%;background:${color}"></div>
  </div>`;
}

function fmtNet(net) {
  return (net >= 0 ? '+' : '') + net.toFixed(3);
}

// ── Render: stats row ─────────────────────────────────────────────────────────
function renderStats(portfolio) {
  // null → empty selection; update period label, show dashes
  if (portfolio === null) {
    document.getElementById('stat-companies').textContent = '0';
    const alertEl = document.getElementById('stat-alerts');
    alertEl.textContent = '—'; alertEl.className = 'stat-value val-neu';
    const sentEl = document.getElementById('stat-sentiment');
    sentEl.textContent = '—'; sentEl.className = 'stat-value val-neu';
    const lblEl = document.getElementById('stat-sentiment-label');
    if (lblEl) lblEl.textContent = `Sentimiento medio ${state.activeDays}d`;
    return;
  }

  const n      = portfolio.length;
  const alerts = portfolio.filter(r => r.alert).length;
  const avg    = n ? portfolio.reduce((s, r) => s + r.net_sentiment_7d, 0) / n : 0;

  document.getElementById('stat-companies').textContent = n;

  const alertEl = document.getElementById('stat-alerts');
  alertEl.textContent = alerts;
  alertEl.className   = 'stat-value ' + (alerts > 0 ? 'val-alert' : 'val-ok');

  const sentEl = document.getElementById('stat-sentiment');
  sentEl.textContent = fmtNet(avg);
  sentEl.className   = 'stat-value ' + netClass(avg);

  const lblEl = document.getElementById('stat-sentiment-label');
  if (lblEl) lblEl.textContent = `Sentimiento medio ${state.activeDays}d`;
}

// ── Render: filter chips ──────────────────────────────────────────────────────
function renderChips() {
  const group = document.getElementById('chip-group');
  group.innerHTML = state.catalog.map(c => {
    const active = state.selectedTickers.includes(c.ticker) ? 'chip-active' : '';
    return `<button class="chip ${active}" data-ticker="${c.ticker}">${c.ticker}</button>`;
  }).join('');
  group.querySelectorAll('.chip').forEach(btn =>
    btn.addEventListener('click', () => toggleTicker(btn.dataset.ticker))
  );
}

function toggleTicker(ticker) {
  const idx = state.selectedTickers.indexOf(ticker);
  if (idx >= 0) state.selectedTickers.splice(idx, 1);
  else          state.selectedTickers.push(ticker);
  renderChips();
  loadPortfolio();
}

// ── Render: portfolio grid ────────────────────────────────────────────────────
function renderPortfolio() {
  const grid    = document.getElementById('company-grid');
  const loading = document.getElementById('grid-loading');
  if (loading) loading.classList.add('hidden');

  // Guard: grid is critical — nothing to paint without it
  if (!grid) {
    console.warn('renderPortfolio: #company-grid not found in DOM');
    return;
  }

  // Alerts strip (non-critical: grid still paints if strip is missing)
  const strip   = document.getElementById('alerts-strip');
  if (!strip) console.warn('renderPortfolio: #alerts-strip not found in DOM');
  const alerted = state.portfolio.filter(i => i.alert);
  if (strip) {
    if (alerted.length === 0) {
      strip.style.display = 'none';
      strip.innerHTML = '';
    } else {
      const label = alerted.length === 1 ? '1 empresa en alerta' : `${alerted.length} empresas en alerta`;
      const chips = alerted.map(i =>
        `<button class="alert-chip" onclick="showCompany('${escapeHtml(i.ticker)}')">${escapeHtml(i.ticker)}</button>`
      ).join('');
      strip.innerHTML = `<span class="alerts-strip-label">⚠ ${label}</span>${chips}`;
      strip.style.display = 'flex';
    }
  }

  if (!state.portfolio.length) {
    grid.innerHTML = '<div class="empty-state">No hay empresas seleccionadas.</div>';
    return;
  }

  const sorted = [...state.portfolio].sort((a, b) => {
    if (a.alert !== b.alert) return a.alert ? -1 : 1;
    return a.net_sentiment_7d - b.net_sentiment_7d;
  });

  grid.innerHTML = sorted.map(item => cardHTML(item)).join('');
  grid.querySelectorAll('.company-card').forEach(card =>
    card.addEventListener('click', () => showCompany(card.dataset.ticker))
  );
}

function cardHTML(item) {
  const net     = item.net_sentiment_7d;
  const isAlert = item.alert;
  const tipAttr = isAlert && item.alert_reason ? ` data-tooltip="${escapeHtml(item.alert_reason)}"` : '';
  const badge   = isAlert
    ? `<span class="badge badge-alert"${tipAttr}>⚠ Alerta</span>`
    : `<span class="badge badge-ok">✓ OK</span>`;

  return `<div class="company-card ${isAlert ? 'card-alert' : ''}" data-ticker="${item.ticker}">
    <div class="card-header">
      <div>
        <div class="card-company-name">${escapeHtml(item.name)}</div>
        <div class="card-ticker">${item.ticker}</div>
      </div>
      ${badge}
    </div>
    <div class="card-metrics">
      <div>
        <div class="metric-label">Sentimiento neto ${state.activeDays}d</div>
        <div class="metric-value" style="color:${netColor(net)}">${fmtNet(net)}</div>
        ${gaugeHTML(net)}
      </div>
      <div>
        <div class="metric-label">Noticias ${state.activeDays}d</div>
        <div class="metric-value">${item.news_count_7d}</div>
      </div>
    </div>
  </div>`;
}

// ── Navigation ────────────────────────────────────────────────────────────────
function showPortfolioView() {
  document.getElementById('view-portfolio').classList.remove('hidden');
  document.getElementById('view-detail').classList.add('hidden');
  state.detailTicker = null;
}

async function showCompany(ticker) {
  document.getElementById('view-portfolio').classList.add('hidden');
  document.getElementById('view-detail').classList.remove('hidden');
  state.detailTicker = ticker;

  // reset to first tab (also restore visibility of conditionally hidden tabs)
  document.querySelectorAll('.tab-btn').forEach(b => { b.classList.remove('active'); b.classList.remove('hidden'); });
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.add('hidden'));
  document.querySelector('[data-tab="sentiment"]').classList.add('active');
  document.getElementById('tab-sentiment').classList.remove('hidden');

  // loading placeholder
  document.getElementById('detail-header').innerHTML = `
    <div class="loading-state"><div class="spinner"></div><span>Cargando ${ticker}…</span></div>`;

  // clear stale chart canvases
  destroyCharts();

  try {
    const data = await fetchCompany(ticker);
    renderDetailHeader(ticker, data);
    renderSentimentChart(data.sentiment_series);
    renderVolumeChart(data.sentiment_series);
    renderNews(data.recent_news);
    renderEvents(data.events);
    setupBriefingTab(ticker);
  } catch (err) {
    document.getElementById('detail-header').innerHTML =
      `<div class="error-state">Error cargando ${ticker}: ${escapeHtml(err.message)}</div>`;
  }
}

function renderDetailHeader(ticker, data) {
  const pItem  = state.portfolio.find(p => p.ticker === ticker);
  const net    = pItem?.net_sentiment_7d ?? 0;
  const badge  = pItem?.alert
    ? `<span class="badge badge-alert" style="margin-top:6px">⚠ Alerta activa</span>`
    : `<span class="badge badge-ok"    style="margin-top:6px">✓ Sin alertas</span>`;

  document.getElementById('detail-header').innerHTML = `
    <div class="detail-title-row">
      <div>
        <div class="detail-company-name">${escapeHtml(data.name)}</div>
        <div class="detail-ticker">${ticker}</div>
        ${badge}
      </div>
      <div style="text-align:right">
        <div class="metric-label" id="detail-sentiment-period-label" style="text-align:right;margin-bottom:6px">
          Sentimiento neto ${state.activeDays}d
        </div>
        <div class="detail-net" id="detail-net" style="color:${netColor(net)}">${fmtNet(net)}</div>
      </div>
    </div>`;
}

function updateDetailSentimentFromPortfolio(ticker) {
  const pItem = state.portfolio.find(p => p.ticker === ticker);
  if (!pItem) return;
  const netEl = document.getElementById('detail-net');
  const lblEl = document.getElementById('detail-sentiment-period-label');
  if (netEl) {
    netEl.textContent = fmtNet(pItem.net_sentiment_7d);
    netEl.style.color = netColor(pItem.net_sentiment_7d);
  }
  if (lblEl) lblEl.textContent = `Sentimiento neto ${state.activeDays}d`;
}

// ── Charts ────────────────────────────────────────────────────────────────────
function destroyCharts() {
  if (state.sentimentChart) { state.sentimentChart.destroy(); state.sentimentChart = null; }
  if (state.volumeChart)    { state.volumeChart.destroy();    state.volumeChart    = null; }
}

const CHART_DEFAULTS = {
  font:         { family: "'Inter', system-ui, sans-serif", size: 11 },
  tickColor:    '#9CA3AF',
  gridColor:    'rgba(0,0,0,0.05)',
  tooltipBg:    '#FFFFFF',
  tooltipBorder:'#E5E7EB',
  tooltipTitle: '#111827',
  tooltipBody:  '#6B7280',
};

function baseTooltip() {
  return {
    backgroundColor: CHART_DEFAULTS.tooltipBg,
    borderColor:     CHART_DEFAULTS.tooltipBorder,
    borderWidth:     1,
    titleColor:      CHART_DEFAULTS.tooltipTitle,
    bodyColor:       CHART_DEFAULTS.tooltipBody,
    titleFont:       { ...CHART_DEFAULTS.font, weight: '600', size: 12 },
    bodyFont:        { ...CHART_DEFAULTS.font },
    padding:         10,
    cornerRadius:    8,
    displayColors:   false,
  };
}

function baseScales(yMin, yMax) {
  return {
    y: {
      min:  yMin,
      max:  yMax,
      grid: { color: CHART_DEFAULTS.gridColor },
      ticks: { font: CHART_DEFAULTS.font, color: CHART_DEFAULTS.tickColor },
      border: { display: false },
    },
    x: {
      grid: { display: false },
      ticks: {
        font:          CHART_DEFAULTS.font,
        color:         CHART_DEFAULTS.tickColor,
        maxRotation:   0,
        autoSkip:      true,
        maxTicksLimit: 10,
      },
      border: { display: false },
    },
  };
}

function renderSentimentChart(series) {
  const ctx   = document.getElementById('chart-sentiment').getContext('2d');
  const dates = series.map(s => s.date);
  const nets  = series.map(s => s.net_sentiment);

  state.sentimentChart = new Chart(ctx, {
    data: {
      labels:   dates,
      datasets: [
        {
          type:            'bar',
          label:           'Sentimiento neto',
          data:            nets,
          backgroundColor: nets.map(v =>
            v > 0.1  ? 'rgba(34,197,94,0.65)'  :
            v < -0.1 ? 'rgba(239,68,68,0.65)'  :
                       'rgba(156,163,175,0.5)'),
          borderRadius:    4,
          borderSkipped:   false,
        },
        {
          type:        'line',
          label:       'Umbral alerta (−0.3)',
          data:        Array(dates.length).fill(-0.3),
          borderColor: 'rgba(239,68,68,0.45)',
          borderDash:  [5, 4],
          borderWidth: 1.5,
          pointRadius: 0,
          fill:        false,
        },
      ],
    },
    options: {
      responsive:          true,
      maintainAspectRatio: false,
      interaction:         { intersect: false, mode: 'index' },
      scales:              baseScales(-1, 1),
      plugins: {
        legend: {
          display: true,
          labels:  { font: CHART_DEFAULTS.font, color: '#6B7280', boxWidth: 10, padding: 16 },
        },
        tooltip: {
          ...baseTooltip(),
          callbacks: {
            label: ctx => ctx.dataset.type === 'line'
              ? `Umbral: ${ctx.parsed.y}`
              : `Net: ${ctx.parsed.y.toFixed(3)}`,
          },
        },
      },
    },
  });
}

function renderVolumeChart(series) {
  const ctx    = document.getElementById('chart-volume').getContext('2d');
  const dates  = series.map(s => s.date);
  const counts = series.map(s => s.n_articles);

  state.volumeChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels:   dates,
      datasets: [{
        label:           'Artículos relevantes',
        data:            counts,
        backgroundColor: 'rgba(99,91,255,0.18)',
        borderColor:     'rgba(99,91,255,0.5)',
        borderWidth:     1,
        borderRadius:    3,
      }],
    },
    options: {
      responsive:          true,
      maintainAspectRatio: false,
      scales:              baseScales(0, null),
      plugins: {
        legend:  {
          display: true,
          labels:  { font: CHART_DEFAULTS.font, color: '#6B7280', boxWidth: 10, padding: 16 },
        },
        tooltip: baseTooltip(),
      },
    },
  });
}

// ── Render: news ──────────────────────────────────────────────────────────────
function renderNews(news) {
  const pane = document.getElementById('tab-news');
  if (!news.length) {
    pane.innerHTML = '<div class="empty-state">Sin noticias recientes.</div>';
    return;
  }

  const items = news.map(n => {
    const lbl  = n.sentiment_label || 'neutral';
    const scr  = (n.sentiment_score || 0.5).toFixed(2);
    const link = n.url
      ? `<a href="${escapeHtml(n.url)}" target="_blank" rel="noopener" class="news-link">${escapeHtml(n.headline)}</a>`
      : `<span>${escapeHtml(n.headline)}</span>`;
    return `<div class="news-item">
      <div class="news-meta">
        <span class="news-date">${n.date}</span>
        ${sentChipHTML(lbl)}
        <span class="news-score">${scr}</span>
      </div>
      <div class="news-title">${link}</div>
    </div>`;
  }).join('');

  pane.innerHTML = `<div class="news-list">
    <div class="news-count">${news.length} artículos recientes (is_relevant = 1)</div>
    ${items}
  </div>`;
}

// ── Render: events ────────────────────────────────────────────────────────────
const EVENT_LABEL = {
  earnings:   'Resultados',
  guidance:   'Previsiones',
  m_and_a:    'Fusiones y compras',
  leadership: 'Cambios directivos',
  legal:      'Legal',
  product:    'Producto',
  analyst:    'Opinión de analistas',
  macro:      'Contexto de mercado',
  other:      'Otros',
};

const EVENT_STYLE = {
  legal:      { bg: '#FEF2F2', border: '#FECACA', text: '#991B1B' },
  earnings:   { bg: '#FFFBEB', border: '#FDE68A', text: '#92400E' },
  guidance:   { bg: '#FFFBEB', border: '#FDE68A', text: '#92400E' },
  m_and_a:    { bg: '#EFF6FF', border: '#BFDBFE', text: '#1E40AF' },
  leadership: { bg: '#F5F3FF', border: '#DDD6FE', text: '#5B21B6' },
  product:    { bg: '#F0FDF4', border: '#BBF7D0', text: '#15803D' },
  analyst:    { bg: '#EFF6FF', border: '#BFDBFE', text: '#1E40AF' },
  macro:      { bg: '#F9FAFB', border: '#E5E7EB', text: '#6B7280' },
  other:      { bg: '#F9FAFB', border: '#E5E7EB', text: '#6B7280' },
};

function eventItemHTML(ev) {
  const s   = EVENT_STYLE[ev.event_type] || EVENT_STYLE.other;
  const dot = ev.is_material ? '● material' : '○ no material';
  const cls = ev.is_material ? 'mat-yes' : 'mat-no';
  return `<div class="event-item">
    <div class="event-header">
      <span class="event-date">${ev.datetime.slice(0, 10)}</span>
      <span class="event-type-badge"
            style="background:${s.bg};border-color:${s.border};color:${s.text}">
        ${EVENT_LABEL[ev.event_type] || ev.event_type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
      </span>
      <span class="event-material ${cls}">${dot}</span>
    </div>
    <div class="event-description">${escapeHtml(ev.description)}</div>
  </div>`;
}

function renderEvents(events) {
  const pane   = document.getElementById('tab-events');
  const tabBtn = document.querySelector('[data-tab="events"]');

  // Guard: if the DOM element is missing (e.g. stale cache), bail cleanly
  if (!pane) {
    console.warn('renderEvents: #tab-events not found in DOM');
    return;
  }

  if (!events.length) {
    if (tabBtn) tabBtn.classList.add('hidden');
    pane.innerHTML = '';
    return;
  }
  if (tabBtn) tabBtn.classList.remove('hidden');

  // Alert state: reuse the flag already in state.portfolio
  const pItem    = state.portfolio.find(p => p.ticker === state.detailTicker);
  const hasAlert = pItem?.alert ?? false;

  // Group by event_type
  const groups = {};
  for (const ev of events) {
    if (!groups[ev.event_type]) groups[ev.event_type] = [];
    groups[ev.event_type].push(ev);
  }

  // Within each group: material first, then datetime DESC
  for (const type in groups) {
    groups[type].sort((a, b) => {
      if (a.is_material !== b.is_material) return a.is_material ? -1 : 1;
      return b.datetime.localeCompare(a.datetime);
    });
  }

  // Group order: legal first, then by size DESC, 'other' last
  const types = Object.keys(groups).sort((a, b) => {
    if (a === 'legal' && b !== 'legal') return -1;
    if (b === 'legal' && a !== 'legal') return  1;
    if (a === 'other' && b !== 'other') return  1;
    if (b === 'other' && a !== 'other') return -1;
    return groups[b].length - groups[a].length;
  });

  const totalMaterial = events.filter(e => e.is_material).length;
  let html = `<div class="events-summary">${events.length} eventos · ${totalMaterial} materiales</div>`;

  for (const type of types) {
    const items  = groups[type];
    const label  = EVENT_LABEL[type] || type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    const isOpen = hasAlert && type === 'legal';
    const bodyId = `ev-group-${type}`;

    html += `
      <div class="ev-group">
        <button class="ev-group-header" aria-expanded="${isOpen}" data-body="${bodyId}">
          <span class="ev-group-arrow">${isOpen ? '▾' : '▸'}</span>
          <span class="ev-group-label">${label}</span>
          <span class="ev-group-count">${items.length}</span>
        </button>
        <div class="ev-group-body${isOpen ? '' : ' hidden'}" id="${bodyId}">
          ${items.map(eventItemHTML).join('')}
        </div>
      </div>`;
  }

  pane.innerHTML = html;

  // Wire collapse toggles
  pane.querySelectorAll('.ev-group-header').forEach(btn => {
    btn.addEventListener('click', () => {
      const bodyEl = document.getElementById(btn.dataset.body);
      const open   = btn.getAttribute('aria-expanded') === 'true';
      btn.setAttribute('aria-expanded', String(!open));
      btn.querySelector('.ev-group-arrow').textContent = open ? '▸' : '▾';
      bodyEl.classList.toggle('hidden', open);
    });
  });
}

// ── Render: briefing tab ──────────────────────────────────────────────────────
function _applyBriefingSentimentBadges(bodyEl) {
  bodyEl.querySelectorAll('table').forEach(table => {
    const headerCells = [...table.querySelectorAll('thead th, tr:first-child th')];
    const sentIdx = headerCells.findIndex(th =>
      /sentim/i.test(th.textContent)
    );
    if (sentIdx < 0) return;
    table.querySelectorAll('tbody tr').forEach(tr => {
      const td = tr.querySelectorAll('td')[sentIdx];
      if (!td) return;
      const raw = td.textContent.trim().toLowerCase();
      const cls = /neg/.test(raw) ? 'sent-badge--neg'
                : /pos/.test(raw) ? 'sent-badge--pos'
                : 'sent-badge--neu';
      td.innerHTML = `<span class="sent-badge ${cls}">${escapeHtml(td.textContent.trim())}</span>`;
    });
  });
}

async function setupBriefingTab(ticker) {
  const pane = document.getElementById('tab-briefing');
  pane.innerHTML = `<div class="briefing-loading"><div class="spinner"></div><span>Cargando informe…</span></div>`;

  try {
    const brief = await fetchBriefing(ticker);
    const dateLabel = brief.generated_at.slice(0, 10);
    const windowLabel = brief.days
      ? `Resumen de los últimos ${brief.days} días`
      : 'Informe de noticias';
    pane.innerHTML = `
      <div class="briefing-header">${escapeHtml(windowLabel)}</div>
      <div class="briefing-meta">Generado el ${dateLabel}</div>
      <div class="briefing-body">${marked.parse(brief.markdown)}</div>`;
    _applyBriefingSentimentBadges(pane.querySelector('.briefing-body'));
  } catch (err) {
    const isNotFound = err.message.startsWith('404');
    pane.innerHTML = `<div class="error-state">${
      isNotFound
        ? 'Informe no disponible para este ticker.'
        : escapeHtml(err.message)
    }</div>`;
  }
}

// ── Tabs ──────────────────────────────────────────────────────────────────────
function setupTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.add('hidden'));
      btn.classList.add('active');
      document.getElementById(`tab-${btn.dataset.tab}`).classList.remove('hidden');
    });
  });
}

// ── Data loading ──────────────────────────────────────────────────────────────
async function loadPortfolio() {
  // No tickers selected → empty state; do NOT fall back to full catalog
  if (state.selectedTickers.length === 0) {
    const loading = document.getElementById('grid-loading');
    if (loading) loading.classList.add('hidden');
    document.getElementById('company-grid').innerHTML =
      '<div class="empty-state-selection">Selecciona una o más empresas para monitorizar.</div>';
    renderStats(null);
    state.portfolio = [];
    return;
  }

  const loading = document.getElementById('grid-loading');
  if (loading) loading.classList.remove('hidden');
  try {
    state.portfolio = await fetchPortfolio(state.selectedTickers);
    renderStats(state.portfolio);
    renderPortfolio();
    if (state.detailTicker) {
      updateDetailSentimentFromPortfolio(state.detailTicker);
    }
  } catch (err) {
    document.getElementById('company-grid').innerHTML =
      `<div class="error-state">Error cargando cartera: ${escapeHtml(err.message)}</div>`;
  }
}

// ── Utility ───────────────────────────────────────────────────────────────────
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── Init ──────────────────────────────────────────────────────────────────────
async function init() {
  document.getElementById('btn-back').addEventListener('click', showPortfolioView);
  document.getElementById('btn-all').addEventListener('click', () => {
    state.selectedTickers = state.catalog.map(c => c.ticker);
    renderChips();
    loadPortfolio();
  });
  document.getElementById('btn-none').addEventListener('click', () => {
    state.selectedTickers = [];
    renderChips();
    loadPortfolio();
  });

  // Period selector
  document.querySelectorAll('.period-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const days = +btn.dataset.days;
      if (days === state.activeDays) return;
      state.activeDays = days;
      document.querySelectorAll('.period-btn').forEach(b => b.classList.remove('period-active'));
      btn.classList.add('period-active');
      loadPortfolio();
    });
  });

  setupTabs();

  try {
    state.catalog         = await fetchCatalog();
    state.selectedTickers = state.catalog.map(c => c.ticker);

    // Show most recent data date from catalog
    const maxDate = state.catalog.reduce((m, c) => c.date_to > m ? c.date_to : m, '');
    if (maxDate) {
      const el = document.getElementById('last-updated');
      if (el) el.textContent = `Demo · datos hasta ${maxDate}`;
    }

    renderChips();
    await loadPortfolio();
  } catch (err) {
    document.getElementById('company-grid').innerHTML = `
      <div class="error-state">
        No se puede conectar con la API. ¿Está uvicorn corriendo en el puerto 8000?<br>
        <code>${escapeHtml(err.message)}</code>
      </div>`;
  }
}

document.addEventListener('DOMContentLoaded', init);
