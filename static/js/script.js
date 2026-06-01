function showLoading(event) {
  const overlay = document.getElementById('loaderOverlay');
  if (overlay) {
    overlay.classList.add('active');
  }

  if (event && event.target) {
    const button = event.target.querySelector("button[type='submit']");
    if (button) {
      button.disabled = true;
      button.innerHTML = 'Analyzing...';
    }
  }
}

function updateRangeValue(labelId, value) {
  const label = document.getElementById(labelId);
  if (label) {
    label.textContent = value;
  }
}

function previewMedia(event, previewId) {
  const container = document.getElementById(previewId);
  if (!container) return;
  const file = event.target.files[0];
  container.innerHTML = '';
  if (!file) {
    container.textContent = 'No media selected.';
    return;
  }

  if (file.type.startsWith('image/')) {
    const img = document.createElement('img');
    img.src = URL.createObjectURL(file);
    img.onload = () => URL.revokeObjectURL(img.src);
    container.appendChild(img);
  } else if (file.type.startsWith('video/')) {
    const video = document.createElement('video');
    video.src = URL.createObjectURL(file);
    video.controls = true;
    video.onload = () => URL.revokeObjectURL(video.src);
    container.appendChild(video);
  } else {
    container.textContent = file.name;
  }
}

function initRangeValues() {
  const rangeFields = [
    {id: 'usageValue', inputName: 'usage_hours'},
    {id: 'sleepValue', inputName: 'sleep_hours'},
    {id: 'mentalValue', inputName: 'mental_score'},
    {id: 'productivityValue', inputName: 'productivity_loss'},
    {id: 'relationshipValue', inputName: 'relationship_conflicts'},
    {id: 'academicValue', inputName: 'academic_performance'},
    {id: 'stressValue', inputName: 'stress_level'},
  ];

  rangeFields.forEach((field) => {
    const input = document.querySelector(`input[name='${field.inputName}']`);
    const label = document.getElementById(field.id);
    if (input && label) {
      label.textContent = input.value;
      input.addEventListener('input', () => {
        label.textContent = input.value;
      });
    }
  });
}

function initCharts() {
  if (typeof chartPayload === 'undefined') {
    return;
  }

  const accuracyCtx = document.getElementById('accuracyChart');
  const pieCtx = document.getElementById('riskPieChart');
  const featureCtx = document.getElementById('featureChart');

  const commonOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: '#cbd5e1' } },
      tooltip: { backgroundColor: 'rgba(15, 23, 42, 0.95)', titleColor: '#fff', bodyColor: '#eef2ff' },
    },
    scales: {
      x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(148, 163, 184, 0.1)' } },
      y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(148, 163, 184, 0.1)' } },
    },
  };

  if (accuracyCtx) {
    new Chart(accuracyCtx, {
      type: 'bar',
      data: {
        labels: chartPayload.accuracyLabels,
        datasets: [
          {
            label: 'Accuracy %',
            data: chartPayload.accuracyValues,
            backgroundColor: ['#38bdf8', '#a78bfa', '#34d399', '#facc15'],
            borderRadius: 16,
            maxBarThickness: 25,
          },
        ],
      },
      options: {
        ...commonOptions,
        scales: commonOptions.scales,
      },
    });
  }

  if (pieCtx) {
    new Chart(pieCtx, {
      type: 'pie',
      data: {
        labels: chartPayload.pieLabels,
        datasets: [
          {
            data: chartPayload.pieValues,
            backgroundColor: ['#22c55e', '#f59e0b', '#ef4444'],
            hoverOffset: 12,
          },
        ],
      },
      options: { ...commonOptions, plugins: { ...commonOptions.plugins, legend: { position: 'bottom', labels: { color: '#cbd5e1' } } } },
    });
  }

  if (featureCtx) {
    new Chart(featureCtx, {
      type: 'line',
      data: {
        labels: chartPayload.featureLabels,
        datasets: [
          {
            label: 'Importance',
            data: chartPayload.featureValues,
            borderColor: '#60a5fa',
            backgroundColor: 'rgba(96, 165, 250, 0.25)',
            fill: true,
            tension: 0.35,
            pointRadius: 4,
            pointBackgroundColor: '#38bdf8',
          },
        ],
      },
      options: {
        ...commonOptions,
        scales: commonOptions.scales,
      },
    });
  }
}

window.addEventListener('load', () => {
  initRangeValues();
  initCharts();

  const forms = document.querySelectorAll('form.needs-validation');
  forms.forEach((form) => {
    form.addEventListener('submit', (event) => {
      if (!form.checkValidity()) {
        event.preventDefault();
        event.stopPropagation();
      }
      form.classList.add('was-validated');
    });
  });
});
