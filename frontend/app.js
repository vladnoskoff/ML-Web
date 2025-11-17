const API_BASE = '';
const form = document.getElementById('analyze-form');
const textarea = document.getElementById('feedback-text');
const resultBlock = document.getElementById('result');
const barsContainer = document.getElementById('probability-bars');
const historyList = document.getElementById('history');
const totalCounter = document.getElementById('total-counter');
const modelName = document.getElementById('model-name');
const modelDetail = document.getElementById('model-detail');
const correctionForm = document.getElementById('correction-form');
const feedbackNotes = document.getElementById('feedback-notes');
const feedbackStatus = document.getElementById('feedback-status');
const bulkForm = document.getElementById('bulk-form');
const bulkFileInput = document.getElementById('bulk-file');
const bulkFileLabel = document.getElementById('bulk-file-label');
const bulkStatus = document.getElementById('bulk-status');
const bulkSummary = document.getElementById('bulk-summary');
const bulkTable = document.getElementById('bulk-table');
const bulkDownload = document.getElementById('bulk-download');
let chart;
let lastPrediction = null;
let lastAnalyzedText = '';
let bulkPredictions = [];

async function fetchJSON(url, options = {}) {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!response.ok) {
    throw new Error(`Ошибка запроса: ${response.status}`);
  }
  return response.json();
}

function renderResult(prediction) {
  const { label, scores } = prediction;
  const mapping = {
    positive: { text: 'Положительная', class: 'result--positive' },
    negative: { text: 'Отрицательная', class: 'result--negative' },
    neutral: { text: 'Нейтральная', class: 'result--neutral' },
  };
  const { text, class: className } = mapping[label] || {
    text: label,
    class: 'result--neutral',
  };
  resultBlock.className = `result ${className}`;
  resultBlock.innerHTML = `<div><p class="muted">Модель определила тональность:</p><h3>${text}</h3></div>`;
  renderBars(scores);
  lastPrediction = prediction;
  setFeedbackEnabled(true);
  selectFeedbackLabel(label);
}

function setFeedbackStatus(message, state = 'info') {
  if (!feedbackStatus) return;
  const classes = ['feedback-form__status'];
  if (state === 'success') classes.push('feedback-form__status--success');
  if (state === 'error') classes.push('feedback-form__status--error');
  feedbackStatus.className = classes.join(' ');
  feedbackStatus.textContent = message;
}

function setFeedbackEnabled(enabled) {
  if (!correctionForm) return;
  correctionForm.classList.toggle('feedback-form--inactive', !enabled);
  const controls = correctionForm.querySelectorAll('input, textarea, button');
  controls.forEach((control) => {
    if (control === feedbackStatus) return;
    control.disabled = !enabled;
  });
  if (!enabled) {
    correctionForm.reset();
    lastPrediction = null;
    lastAnalyzedText = '';
  }
  setFeedbackStatus(
    enabled
      ? 'Если результат неверен, выберите правильную тональность и отправьте исправление.'
      : 'Сначала выполните анализ, чтобы отправить корректировку.'
  );
}

function selectFeedbackLabel(label) {
  if (!correctionForm) return;
  const radio = correctionForm.querySelector(`input[name="user-label"][value="${label}"]`);
  if (radio) {
    radio.checked = true;
  }
}

setFeedbackEnabled(false);

function renderBars(scores) {
  barsContainer.innerHTML = '';
  Object.entries(scores).forEach(([label, value]) => {
    const wrapper = document.createElement('div');
    const labelEl = document.createElement('div');
    labelEl.className = 'bar__label';
    labelEl.innerHTML = `<span>${label}</span><span>${(value * 100).toFixed(1)}%</span>`;
    const bar = document.createElement('div');
    bar.className = 'bar';
    const span = document.createElement('span');
    span.style.width = `${(value * 100).toFixed(1)}%`;
    span.style.background = labelColor(label);
    bar.appendChild(span);
    wrapper.appendChild(labelEl);
    wrapper.appendChild(bar);
    barsContainer.appendChild(wrapper);
  });
}

function toggleHidden(element, visible) {
  if (!element) return;
  if (visible) {
    element.classList.remove('hidden');
  } else {
    element.classList.add('hidden');
  }
}

function labelColor(label) {
  switch (label) {
    case 'positive':
      return 'var(--positive)';
    case 'negative':
      return 'var(--negative)';
    default:
      return 'var(--neutral)';
  }
}

function updateHistory(history) {
  historyList.innerHTML = '';
  history.forEach((item) => {
    const li = document.createElement('li');
    li.innerHTML = `
      <div class="history__label">
        <span>${new Date(item.timestamp).toLocaleTimeString()}</span>
        <span>${item.label}</span>
      </div>
      <p class="history__text">${item.text}</p>
    `;
    historyList.appendChild(li);
  });
}

function updateChart(distribution) {
  const labels = Object.keys(distribution);
  const data = Object.values(distribution).map((v) => Math.round(v * 100));
  const ctx = document.getElementById('stats-chart');
  if (!chart) {
    chart = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels,
        datasets: [
          {
            data,
            backgroundColor: labels.map((label) => labelColor(label)),
          },
        ],
      },
      options: {
        plugins: {
          legend: { position: 'bottom' },
        },
      },
    });
  } else {
    chart.data.labels = labels;
    chart.data.datasets[0].data = data;
    chart.data.datasets[0].backgroundColor = labels.map((label) => labelColor(label));
    chart.update();
  }
}

async function refreshStats() {
  try {
    const stats = await fetchJSON(`${API_BASE}/stats`);
    totalCounter.textContent = stats.total_predictions;
    updateHistory(stats.recent_predictions);
    updateChart(stats.label_distribution);
  } catch (error) {
    console.error(error);
  }
}

async function loadModelInfo() {
  try {
    const info = await fetchJSON(`${API_BASE}/model`);
    modelName.textContent = info.algorithm || 'Модель';
    modelDetail.textContent = info.classes ? `Классы: ${info.classes.join(', ')}` : '';
  } catch (error) {
    modelName.textContent = 'Не удалось получить данные';
    console.error(error);
  }
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const text = textarea.value.trim();
  if (!text) return;
  form.classList.add('loading');
  lastAnalyzedText = text;
  try {
    const prediction = await fetchJSON(`${API_BASE}/predict`, {
      method: 'POST',
      body: JSON.stringify({ text }),
    });
    renderResult(prediction);
    textarea.value = '';
    await refreshStats();
  } catch (error) {
    resultBlock.className = 'result result--negative';
    resultBlock.innerHTML = `<p>Ошибка: ${error.message}</p>`;
    setFeedbackEnabled(false);
  } finally {
    form.classList.remove('loading');
  }
});

if (correctionForm) {
  correctionForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (!lastPrediction || !lastAnalyzedText) {
      setFeedbackStatus('Сначала выполните анализ текста.', 'error');
      return;
    }
    const selected = correctionForm.querySelector('input[name="user-label"]:checked');
    if (!selected) {
      setFeedbackStatus('Выберите корректную тональность.', 'error');
      return;
    }
    const payload = {
      text: lastAnalyzedText,
      predicted_label: lastPrediction.label,
      user_label: selected.value,
      scores: lastPrediction.scores,
      notes: feedbackNotes && feedbackNotes.value.trim() ? feedbackNotes.value.trim() : undefined,
    };
    correctionForm.classList.add('loading');
    try {
      await fetchJSON(`${API_BASE}/feedback`, {
        method: 'POST',
        body: JSON.stringify(payload),
      });
      setFeedbackStatus('Спасибо! Исправление сохранено.', 'success');
      correctionForm.reset();
      selectFeedbackLabel(lastPrediction.label);
    } catch (error) {
      setFeedbackStatus(`Ошибка: ${error.message}`, 'error');
    } finally {
      correctionForm.classList.remove('loading');
    }
  });
}

function setBulkStatus(message, state = 'info') {
  if (!bulkStatus) return;
  bulkStatus.textContent = message;
  const classes = ['bulk-form__status'];
  if (state === 'error') {
    classes.push('feedback-form__status--error');
  }
  bulkStatus.className = classes.join(' ');
}

function renderBulkSummary(summary) {
  if (!bulkSummary) return;
  const inputRows = summary?.input_rows ?? 0;
  const processedRows = summary?.processed_rows ?? 0;
  const skippedRows = summary?.skipped_rows ?? Math.max(0, inputRows - processedRows);
  const countsMarkup = Object.entries(summary?.class_counts || {})
    .map(([label, count]) => `<p><strong>${label}:</strong> ${count}</p>`)
    .join('');
  bulkSummary.innerHTML = `
    <div class="bulk-summary__card">
      <h4>Строк обработано</h4>
      <p><strong>${processedRows}</strong> из ${inputRows}</p>
      <p class="muted">Пропущено: ${skippedRows}</p>
    </div>
    <div class="bulk-summary__card">
      <h4>Распределение классов</h4>
      ${countsMarkup || '<p>Нет данных</p>'}
    </div>
  `;
  toggleHidden(bulkSummary, true);
}

function renderBulkTable(predictions) {
  if (!bulkTable) return;
  if (!predictions.length) {
    bulkTable.innerHTML = '';
    toggleHidden(bulkTable, false);
    return;
  }
  const rows = predictions
    .map(
      (pred) => `
      <tr>
        <td>${pred.row}</td>
        <td>${pred.label}</td>
        <td>${(pred.scores[pred.label] * 100).toFixed(1)}%</td>
        <td>${pred.text}</td>
      </tr>`
    )
    .join('');
  bulkTable.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>Класс</th>
          <th>Уверенность</th>
          <th>Текст</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
  toggleHidden(bulkTable, true);
}

function downloadBulkCSV() {
  if (!bulkPredictions.length) return;
  const header = ['row', 'label', 'confidence', 'text'];
  const lines = [header.join(',')];
  bulkPredictions.forEach((pred) => {
    const confidence = pred.scores[pred.label] ?? 0;
    const values = [pred.row, pred.label, confidence.toFixed(4), pred.text]
      .map((value) => {
        const stringValue = String(value ?? '');
        if (stringValue.includes('"') || stringValue.includes(',') || stringValue.includes('\n')) {
          return '"' + stringValue.replace(/"/g, '""') + '"';
        }
        return stringValue;
      })
      .join(',');
    lines.push(values);
  });
  const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'predictions.csv';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

if (bulkFileInput) {
  bulkFileInput.addEventListener('change', () => {
    if (!bulkFileLabel) return;
    const file = bulkFileInput.files && bulkFileInput.files[0];
    bulkFileLabel.textContent = file ? file.name : 'Выберите CSV-файл';
  });
}

if (bulkDownload) {
  bulkDownload.addEventListener('click', downloadBulkCSV);
}

if (bulkForm) {
  bulkForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    const file = bulkFileInput && bulkFileInput.files ? bulkFileInput.files[0] : null;
    if (!file) {
      setBulkStatus('Выберите CSV-файл.', 'error');
      return;
    }
    const formData = new FormData();
    formData.append('file', file);
    bulkForm.classList.add('loading');
    setBulkStatus('Файл отправлен, ждём результат…');
    try {
      const response = await fetch(`${API_BASE}/predict_file`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || 'Не удалось обработать файл');
      }
      const payload = await response.json();
      bulkPredictions = payload.predictions || [];
      renderBulkSummary(payload.summary || {});
      renderBulkTable(bulkPredictions);
      toggleHidden(bulkDownload, bulkPredictions.length > 0);
      await refreshStats();
      setBulkStatus('Готово! Результаты можно скачать ниже.');
    } catch (error) {
      setBulkStatus(`Ошибка: ${error.message}`, 'error');
      toggleHidden(bulkSummary, false);
      toggleHidden(bulkTable, false);
      toggleHidden(bulkDownload, false);
    } finally {
      bulkForm.classList.remove('loading');
    }
  });
}

loadModelInfo();
refreshStats();
setInterval(refreshStats, 5000);
