const API_BASE = '';
const form = document.getElementById('analyze-form');
const textarea = document.getElementById('feedback-text');
const resultBlock = document.getElementById('result');
const barsContainer = document.getElementById('probability-bars');
const historyList = document.getElementById('history');
const totalCounter = document.getElementById('total-counter');
const modelName = document.getElementById('model-name');
const modelDetail = document.getElementById('model-detail');
let chart;

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
}

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
  } finally {
    form.classList.remove('loading');
  }
});

loadModelInfo();
refreshStats();
setInterval(refreshStats, 5000);
