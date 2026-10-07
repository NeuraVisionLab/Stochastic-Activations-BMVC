const YOUTUBE_VIDEO_ID = 'jEi-fqd9yHQ';
const YOUTUBE_WATCH_URL = `https://www.youtube.com/watch?v=${YOUTUBE_VIDEO_ID}`;

const videoShell = document.querySelector('#video-shell');

if (videoShell && YOUTUBE_VIDEO_ID) {
  // YouTube can return player error 153 when an embed is opened directly from
  // file:// because the request has no HTTP referrer. The real embedded player
  // is therefore used on normal http/https hosting, while local file previews
  // get a clean fallback instead of a broken player.
  if (window.location.protocol === 'file:') {
    videoShell.innerHTML = `
      <a class="video-local-fallback" href="${YOUTUBE_WATCH_URL}" target="_blank" rel="noreferrer"
         aria-label="Watch the BMVC 2026 paper presentation on YouTube">
        <img src="https://i.ytimg.com/vi/${encodeURIComponent(YOUTUBE_VIDEO_ID)}/maxresdefault.jpg" alt="Video thumbnail for the BMVC 2026 paper presentation">
        <span class="video-play-button" aria-hidden="true">▶</span>
        <span class="video-local-note">
          <strong>Watch the 8-minute paper presentation</strong>
          <small>Local file preview · the embedded player appears when the site is served over HTTP/HTTPS</small>
        </span>
      </a>`;
  } else {
    videoShell.innerHTML = `
      <iframe
        src="https://www.youtube.com/embed/${encodeURIComponent(YOUTUBE_VIDEO_ID)}?rel=0&playsinline=1"
        title="Stochastic Nonlinearities Improve Uncertainty Estimation — BMVC 2026 paper presentation"
        loading="lazy"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
        allowfullscreen></iframe>`;
  }
}

const resultSlides = [
  {
    label: 'Segmentation performance',
    title: 'Segmentation performance',
    note: 'Table 1 · higher is better',
    columns: ['Dataset', 'Method', 'Correct. ↑', 'Complete. ↑', 'Quality ↑', 'F1 ↑'],
    rows: [
      ['Massachusetts', 'MC Dropout', '91.00', '87.20', '81.90', '78.62'],
      ['Massachusetts', 'DDU', '91.44', '86.10', '80.52', '79.23'],
      ['Massachusetts', 'TTA', '91.46', '86.73', '81.19', '79.43'],
      ['Massachusetts', 'Masksembles', '81.63', '87.67', '81.94', '80.02'],
      ['Massachusetts', 'Deep Ensemble', '92.37', '88.11', '82.67', '80.54'],
      ['Massachusetts', 'Laplace Redux', '91.45', '86.11', '80.52', '79.23'],
      ['Massachusetts', 'Iterative', '90.07', '84.97', '78.39', '78.58'],
      ['Massachusetts', 'Ours', '91.88', '88.24', '82.11', '79.71'],
      ['DRIVE', 'MC Dropout', '96.18', '85.13', '82.53', '79.41'],
      ['DRIVE', 'DDU', '96.13', '85.83', '82.86', '80.81'],
      ['DRIVE', 'TTA', '95.45', '84.41', '82.29', '79.98'],
      ['DRIVE', 'Masksembles', '96.82', '85.11', '82.69', '81.02'],
      ['DRIVE', 'Deep Ensemble', '97.28', '85.84', '83.72', '81.84'],
      ['DRIVE', 'Laplace Redux', '96.21', '84.54', '82.81', '77.56'],
      ['DRIVE', 'Iterative', '96.86', '82.61', '81.87', '80.30'],
      ['DRIVE', 'Ours', '98.38', '85.61', '82.91', '81.00'],
      ['LiTS', 'MC Dropout', '87.09', '71.06', '65.82', '96.41'],
      ['LiTS', 'DDU', '87.45', '78.82', '69.73', '97.02'],
      ['LiTS', 'TTA', '72.04', '64.01', '58.73', '93.87'],
      ['LiTS', 'Masksembles', '88.27', '77.83', '70.75', '98.01'],
      ['LiTS', 'Deep Ensemble', '87.59', '81.38', '72.84', '98.25'],
      ['LiTS', 'Laplace Redux', '87.27', '80.40', '71.43', '97.51'],
      ['LiTS', 'Iterative', '87.98', '74.55', '70.26', '97.39'],
      ['LiTS', 'Ours', '89.39', '79.57', '72.93', '98.11'],
      ['CREMI', 'MC Dropout', '97.40', '94.45', '92.68', '78.58'],
      ['CREMI', 'DDU', '96.41', '94.76', '93.16', '79.48'],
      ['CREMI', 'TTA', '97.04', '94.65', '92.63', '76.23'],
      ['CREMI', 'Masksembles', '96.90', '95.01', '93.56', '81.76'],
      ['CREMI', 'Deep Ensemble', '97.24', '95.83', '93.93', '79.92'],
      ['CREMI', 'Laplace Redux', '97.26', '95.62', '93.76', '79.58'],
      ['CREMI', 'Iterative', '96.02', '87.90', '87.19', '65.73'],
      ['CREMI', 'Ours', '98.47', '96.61', '94.76', '81.94']
    ]
  },
  {
    label: 'Segmentation uncertainty',
    title: 'Segmentation uncertainty & calibration',
    note: 'Table 2 · Corr. ↑ · ECE/Brier ↓',
    columns: ['Dataset', 'Method', 'Corr. ↑', 'ECE ↓', 'Brier ↓', 'Train', 'Infer'],
    rows: [
      ['Massachusetts', 'MC Dropout', '79.73', '0.026', '0.033', '×1', '×5'],
      ['Massachusetts', 'DDU', '75.64', '0.027', '0.031', '×1', '–'],
      ['Massachusetts', 'TTA', '79.90', '0.025', '0.029', '×1', '×5'],
      ['Massachusetts', 'Masksembles', '79.62', '0.024', '0.031', '×1', '×5'],
      ['Massachusetts', 'Deep Ensemble', '80.03', '0.024', '0.027', '×5', '×5'],
      ['Massachusetts', 'Laplace Redux', '77.51', '0.034', '0.034', '×1', '–'],
      ['Massachusetts', 'Iterative', '77.28', '0.028', '0.030', '×3', '×3'],
      ['Massachusetts', 'Ours', '80.38', '0.020', '0.025', '×1', '×5'],
      ['DRIVE', 'MC Dropout', '80.58', '0.032', '0.040', '×1', '×5'],
      ['DRIVE', 'DDU', '80.45', '0.031', '0.034', '×1', '–'],
      ['DRIVE', 'TTA', '83.00', '0.058', '0.031', '×1', '×5'],
      ['DRIVE', 'Masksembles', '81.06', '0.029', '0.035', '×1', '×5'],
      ['DRIVE', 'Deep Ensemble', '81.72', '0.017', '0.031', '×5', '×5'],
      ['DRIVE', 'Laplace Redux', '79.56', '0.086', '0.043', '×1', '–'],
      ['DRIVE', 'Iterative', '79.91', '0.024', '0.035', '×3', '×3'],
      ['DRIVE', 'Ours', '81.80', '0.015', '0.029', '×1', '×5'],
      ['LiTS', 'MC Dropout', '52.28', '0.004', '0.003', '×1', '×5'],
      ['LiTS', 'DDU', '52.55', '0.004', '0.003', '×1', '–'],
      ['LiTS', 'TTA', '52.41', '0.004', '0.003', '×1', '×5'],
      ['LiTS', 'Masksembles', '52.80', '0.001', '0.003', '×1', '×5'],
      ['LiTS', 'Deep Ensemble', '53.30', '0.001', '0.001', '×5', '×5'],
      ['LiTS', 'Laplace Redux', '52.35', '0.003', '0.003', '×1', '–'],
      ['LiTS', 'Iterative', '53.34', '0.002', '0.002', '×3', '×3'],
      ['LiTS', 'Ours', '54.22', '0.001', '0.002', '×1', '×5'],
      ['CREMI', 'MC Dropout', '37.23', '0.035', '0.017', '×1', '×5'],
      ['CREMI', 'DDU', '34.12', '0.016', '0.018', '×1', '–'],
      ['CREMI', 'TTA', '38.70', '0.011', '0.016', '×1', '×5'],
      ['CREMI', 'Masksembles', '36.97', '0.014', '0.017', '×1', '×5'],
      ['CREMI', 'Deep Ensemble', '38.49', '0.009', '0.014', '×5', '×5'],
      ['CREMI', 'Laplace Redux', '35.47', '0.011', '0.015', '×1', '–'],
      ['CREMI', 'Iterative', '39.48', '0.050', '0.023', '×3', '×3'],
      ['CREMI', 'Ours', '42.58', '0.004', '0.009', '×1', '×5']
    ]
  },
  {
    label: 'Segmentation OOD',
    title: 'Cross-dataset segmentation OOD detection',
    note: 'Table 3 · AUROC/AUPR ↑',
    columns: ['Method', 'Mass.→RoadTracer AUROC', 'AUPR', 'DRIVE→CHASE AUROC', 'AUPR', 'CREMI→ISBI AUROC', 'AUPR'],
    rows: [
      ['MC Dropout', '53.07', '48.51', '82.21', '53.07', '87.34', '94.64'],
      ['DDU', '54.63', '52.54', '85.99', '61.77', '88.79', '94.82'],
      ['TTA', '55.05', '56.62', '89.94', '67.31', '93.10', '94.88'],
      ['Masksembles', '54.00', '51.91', '86.35', '64.51', '87.72', '94.14'],
      ['Deep Ensemble', '57.32', '57.85', '89.56', '66.33', '85.73', '94.69'],
      ['Laplace Redux', '52.82', '52.04', '86.46', '57.90', '88.68', '94.77'],
      ['Iterative', '57.04', '49.68', '83.23', '57.32', '88.46', '94.53'],
      ['Ours', '56.74', '56.00', '90.52', '66.31', '95.43', '95.68']
    ]
  },
  {
    label: 'Classification',
    title: 'Classification accuracy & OOD detection',
    note: 'Table 4 · Acc/AUC/AUPR ↑ · ECE/Brier/FPR95 ↓',
    columns: ['Dataset', 'Method', 'Acc ↑', 'ECE ↓', 'Brier ↓', 'AUC ↑', 'AUPR ↑', 'FPR95 ↓'],
    rows: [
      ['CIFAR-10', 'MC Dropout', '93.54', '0.011', '0.094', '84.22', '87.36', '27.10'],
      ['CIFAR-10', 'DDU', '94.12', '0.018', '0.090', '85.20', '89.26', '24.11'],
      ['CIFAR-10', 'TTA', '94.23', '0.011', '0.083', '93.64', '97.29', '17.97'],
      ['CIFAR-10', 'Masksembles', '94.61', '0.011', '0.080', '91.69', '92.88', '18.47'],
      ['CIFAR-10', 'Deep Ensemble', '94.94', '0.009', '0.074', '91.92', '93.15', '14.70'],
      ['CIFAR-10', 'Laplace Redux', '93.83', '0.090', '0.103', '92.60', '94.98', '18.83'],
      ['CIFAR-10', 'Ours', '94.59', '0.009', '0.083', '95.14', '97.54', '16.36'],
      ['MiniImageNet', 'MC Dropout', '64.70', '0.067', '0.212', '81.57', '86.96', '53.94'],
      ['MiniImageNet', 'DDU', '65.83', '0.050', '0.191', '84.21', '88.49', '49.85'],
      ['MiniImageNet', 'TTA', '68.52', '0.044', '0.171', '79.10', '84.41', '54.28'],
      ['MiniImageNet', 'Masksembles', '66.24', '0.048', '0.194', '83.52', '88.06', '48.35'],
      ['MiniImageNet', 'Deep Ensemble', '70.50', '0.039', '0.187', '84.13', '89.20', '47.98'],
      ['MiniImageNet', 'Laplace Redux', '64.48', '0.052', '0.215', '80.39', '86.14', '54.55'],
      ['MiniImageNet', 'Ours', '66.90', '0.031', '0.178', '86.42', '90.66', '45.24'],
      ['BloodMNIST', 'MC Dropout', '98.42', '0.008', '0.027', '93.81', '96.12', '22.36'],
      ['BloodMNIST', 'DDU', '98.52', '0.009', '0.023', '98.42', '98.86', '9.38'],
      ['BloodMNIST', 'TTA', '98.60', '0.010', '0.030', '98.43', '99.09', '6.40'],
      ['BloodMNIST', 'Masksembles', '98.40', '0.010', '0.025', '98.74', '99.11', '7.61'],
      ['BloodMNIST', 'Deep Ensemble', '98.45', '0.010', '0.020', '99.06', '99.50', '5.11'],
      ['BloodMNIST', 'Laplace Redux', '80.35', '0.011', '0.029', '94.23', '97.16', '7.61'],
      ['BloodMNIST', 'Ours', '98.72', '0.008', '0.023', '98.98', '99.53', '4.50'],
      ['PathMNIST', 'MC Dropout', '94.14', '0.034', '0.096', '89.43', '68.70', '25.15'],
      ['PathMNIST', 'DDU', '94.53', '0.041', '0.094', '88.39', '55.97', '24.52'],
      ['PathMNIST', 'TTA', '94.00', '0.041', '0.098', '85.72', '56.60', '20.98'],
      ['PathMNIST', 'Masksembles', '94.74', '0.037', '0.098', '85.36', '57.25', '23.38'],
      ['PathMNIST', 'Deep Ensemble', '95.01', '0.018', '0.076', '86.04', '58.20', '22.09'],
      ['PathMNIST', 'Laplace Redux', '93.77', '0.045', '0.111', '44.68', '27.51', '92.09'],
      ['PathMNIST', 'Ours', '95.40', '0.037', '0.088', '94.87', '80.58', '11.60']
    ]
  }
];

const stage = document.querySelector('#results-stage');
const tabs = document.querySelector('#result-tabs');
const dots = document.querySelector('#result-dots');
const counter = document.querySelector('#results-counter');
const prevButton = document.querySelector('#results-prev');
const nextButton = document.querySelector('#results-next');
const carousel = document.querySelector('#results-carousel');
let activeSlide = 0;
let activeGroup = 0;

function groupedResults(slide) {
  if (slide.columns[0] === 'Dataset') {
    const names = [];
    const groups = new Map();
    slide.rows.forEach(row => {
      const dataset = row[0];
      if (!groups.has(dataset)) {
        groups.set(dataset, []);
        names.push(dataset);
      }
      groups.get(dataset).push(row.slice(1));
    });
    return names.map(name => ({
      name,
      columns: slide.columns.slice(1),
      rows: groups.get(name)
    }));
  }

  if (slide.label === 'Segmentation OOD') {
    const specs = [
      { name: 'Massachusetts → RoadTracer', a: 1, b: 2 },
      { name: 'DRIVE → CHASE-DB1', a: 3, b: 4 },
      { name: 'CREMI → ISBI 2012', a: 5, b: 6 }
    ];
    return specs.map(spec => ({
      name: spec.name,
      columns: ['Method', 'AUROC ↑', 'AUPR ↑'],
      rows: slide.rows.map(row => [row[0], row[spec.a], row[spec.b]])
    }));
  }

  return [{ name: slide.title, columns: slide.columns, rows: slide.rows }];
}

function metricDirection(col) {
  if (col.includes('↓')) return 'lower is better';
  if (col.includes('↑')) return 'higher is better';
  return '';
}

function renderSlide(slideIndex, groupIndex = 0) {
  activeSlide = (slideIndex + resultSlides.length) % resultSlides.length;
  const slide = resultSlides[activeSlide];
  const groups = groupedResults(slide);
  activeGroup = (groupIndex + groups.length) % groups.length;
  const group = groups[activeGroup];

  const header = group.columns.map(col => {
    const helper = metricDirection(col);
    return `<th scope="col"${helper ? ` title="${helper}"` : ''}>${col}</th>`;
  }).join('');

  const body = group.rows.map(row => {
    const isOurs = row[0] === 'Ours';
    const cells = row.map((cell, cellIndex) => {
      const tag = cellIndex === 0 ? 'th' : 'td';
      const scope = tag === 'th' ? ' scope="row"' : '';
      return `<${tag}${scope}>${cell}</${tag}>`;
    }).join('');
    return `<tr class="${isOurs ? 'ours' : ''}">${cells}</tr>`;
  }).join('');

  const groupButtons = groups.map((item, index) => `
    <button type="button" class="dataset-tab${index === activeGroup ? ' active' : ''}" data-group="${index}" aria-pressed="${index === activeGroup}">${item.name}</button>`
  ).join('');

  stage.innerHTML = `
    <article class="result-panel" role="tabpanel" aria-label="${slide.title}: ${group.name}">
      <div class="result-panel-head">
        <div>
          <span class="result-dataset">${group.name}</span>
          <h3>${slide.title}</h3>
        </div>
        <p>${slide.note}</p>
      </div>
      <div class="dataset-tabs" aria-label="Choose dataset">${groupButtons}</div>
      <div class="table-scroll" tabindex="0">
        <table class="results-table">
          <thead><tr>${header}</tr></thead>
          <tbody>${body}</tbody>
        </table>
      </div>
    </article>`;

  stage.querySelectorAll('.dataset-tab').forEach(button => {
    button.addEventListener('click', () => renderSlide(activeSlide, Number(button.dataset.group)));
  });

  [...tabs.children].forEach((tab, i) => {
    tab.setAttribute('aria-selected', i === activeSlide ? 'true' : 'false');
    tab.tabIndex = i === activeSlide ? 0 : -1;
  });

  dots.innerHTML = groups.map((_, index) => `
    <button type="button" class="carousel-dot${index === activeGroup ? ' active' : ''}" data-group="${index}" aria-label="Show result page ${index + 1}"></button>`
  ).join('');
  dots.querySelectorAll('.carousel-dot').forEach(dot => {
    dot.addEventListener('click', () => renderSlide(activeSlide, Number(dot.dataset.group)));
  });

  counter.textContent = `${activeGroup + 1} / ${groups.length}`;
}

resultSlides.forEach((slide, index) => {
  const tab = document.createElement('button');
  tab.type = 'button';
  tab.className = 'carousel-tab';
  tab.role = 'tab';
  tab.textContent = slide.label;
  tab.addEventListener('click', () => renderSlide(index, 0));
  tabs.appendChild(tab);
});

prevButton.addEventListener('click', () => renderSlide(activeSlide, activeGroup - 1));
nextButton.addEventListener('click', () => renderSlide(activeSlide, activeGroup + 1));
carousel.addEventListener('keydown', event => {
  if (event.key === 'ArrowLeft') renderSlide(activeSlide, activeGroup - 1);
  if (event.key === 'ArrowRight') renderSlide(activeSlide, activeGroup + 1);
});
renderSlide(0, 0);

const copyButton = document.querySelector('#copy-bibtex');
const toast = document.querySelector('#toast');
let toastTimer;

copyButton.addEventListener('click', async () => {
  const citation = document.querySelector('#bibtex-code').textContent;
  try {
    await navigator.clipboard.writeText(citation);
    copyButton.textContent = 'Copied';
    toast.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      copyButton.textContent = 'Copy';
      toast.classList.remove('show');
    }, 1800);
  } catch {
    copyButton.textContent = 'Select manually';
  }
});
