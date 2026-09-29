const input = document.querySelector('#media-input');
const zone = document.querySelector('#drop-zone');
const empty = document.querySelector('#drop-empty');
const selected = document.querySelector('#drop-selected');
const frame = document.querySelector('#media-frame');
const error = document.querySelector('#upload-error');
const runButton = document.querySelector('#run-detection');
const status = document.querySelector('#analysis-status');
const verdict = document.querySelector('#result-verdict');
const score = document.querySelector('#result-score');
const note = document.querySelector('#result-note');
const resultPanel = document.querySelector('#result-panel');
const serviceStatus = document.querySelector('#service-status');
const serviceMessage = document.querySelector('#service-message');
const serviceLink = document.querySelector('#service-link');
const localSite = ['127.0.0.1', 'localhost', ''].includes(location.hostname);
const backendHelp = localSite
  ? 'Start the backend with ./start.command, then open http://127.0.0.1:8000.'
  : 'The detector is temporarily unavailable. Please try again later.';
let previewUrl;
let selectedFile;
let request;

function resetResult() {
  if (request) request.abort();
  request = undefined;
  runButton.disabled = false;
  runButton.textContent = 'Run detection';
  status.textContent = 'Ready to analyze';
  status.dataset.state = 'ready';
  document.querySelector('#analysis-title').textContent = 'Awaiting analysis';
  verdict.textContent = 'Waiting for scan';
  resultPanel.dataset.verdict = 'pending';
  score.textContent = '—';
  note.textContent = 'VERUS uses separate validated detectors for AI-generated photos and face-swap videos.';
}

function clearFile() {
  resetResult();
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = undefined;
  selectedFile = undefined;
  input.value = '';
  frame.replaceChildren();
  selected.hidden = true;
  empty.hidden = false;
  error.hidden = true;
}

function showFile(file) {
  error.hidden = true;
  if (!file) return;
  const allowed = ['image/png', 'image/jpeg', 'image/webp', 'video/mp4', 'video/webm', 'video/quicktime'];
  if (!allowed.includes(file.type) || file.size === 0 || file.size > 4 * 1024 * 1024) {
    clearFile();
    error.textContent = 'Choose a non-empty JPG, PNG, WebP, MP4, WebM, or MOV file up to 4 MB.';
    error.hidden = false;
    return;
  }
  resetResult();
  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  const media = document.createElement(file.type.startsWith('video/') ? 'video' : 'img');
  media.src = previewUrl;
  if (media.tagName === 'VIDEO') {
    media.controls = true;
    media.playsInline = true;
  } else {
    media.alt = `Preview of ${file.name}`;
  }
  frame.replaceChildren(media);
  document.querySelector('#file-name').textContent = file.name;
  const size = file.size < 1024 * 1024 ? `${Math.ceil(file.size / 1024)} KB` : `${(file.size / 1024 / 1024).toFixed(1)} MB`;
  document.querySelector('#file-details').textContent = `${file.type} · ${size}`;
  empty.hidden = true;
  selected.hidden = false;
}

runButton.addEventListener('click', async () => {
  if (!selectedFile || request) return;
  const current = new AbortController();
  request = current;
  runButton.disabled = true;
  runButton.textContent = 'Analyzing…';
  status.textContent = 'Analyzing';
  status.dataset.state = 'running';
  document.querySelector('#analysis-title').textContent = 'Looking closer';
  verdict.textContent = 'Analyzing…';
  score.textContent = '—';
  resultPanel.dataset.verdict = 'pending';
  note.textContent = 'The first run may take longer while the model loads.';
  const form = new FormData();
  form.append('file', selectedFile);
  try {
    const response = await fetch('/api/analyze', { method: 'POST', body: form, signal: current.signal });
    const result = response.headers.get('content-type')?.includes('application/json') ? await response.json() : null;
    if (!result) throw new Error(backendHelp);
    if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : 'Analysis failed. Please try again.');
    const names = {
      likely_manipulated: 'LIKELY FAKE',
      no_strong_signal: 'LIKELY REAL',
      inconclusive: 'INCONCLUSIVE'
    };
    document.querySelector('#analysis-title').textContent = 'Analysis complete';
    status.textContent = 'Complete';
    status.dataset.state = 'complete';
    verdict.textContent = names[result.verdict] || 'INCONCLUSIVE';
    resultPanel.dataset.verdict = result.verdict || 'inconclusive';
    score.textContent = result.fake_score == null || !result.calibrated ? '—' : `${(result.fake_score * 100).toFixed(1)}%`;
    if (result.media_type === 'image') {
      const context = result.verdict === 'no_strong_signal'
        ? 'The score is in the likely-real range; this does not prove authenticity.'
        : result.verdict === 'likely_manipulated'
          ? 'The score is in the likely-fake range; this is evidence, not proof.'
          : 'The score falls between the validated decision ranges.';
      note.textContent = `A separate still-image model analyzed five crops of the photo. ${context} Screenshots, edits, low-resolution files, and new generators can still fool it.`;
    } else if (result.frames_with_faces === 0) {
      note.textContent = 'No face was found in the sampled media. This model cannot assess content without a visible face.';
    } else {
      const sample = `${result.frames_with_faces} of ${result.frames_sampled} sampled frame${result.frames_sampled === 1 ? '' : 's'} had a detectable face.`;
      const context = result.verdict === 'no_strong_signal'
        ? " The score falls in the model's likely-real range; this does not prove authenticity."
        : ' This is a model estimate, not proof of manipulation.';
      note.textContent = `${sample}${result.multiple_faces ? ' Only the largest face in each frame was analyzed.' : ''}${context} Verify important claims with the source.`;
    }
  } catch (cause) {
    if (cause.name !== 'AbortError') {
      document.querySelector('#analysis-title').textContent = 'Could not analyze';
      status.textContent = 'Error';
      status.dataset.state = 'error';
      verdict.textContent = 'NO RESULT';
      note.textContent = cause instanceof TypeError ? backendHelp : cause.message || backendHelp;
      if (cause instanceof TypeError || cause.message === backendHelp) showServiceError();
    }
  } finally {
    if (request === current) {
      request = undefined;
      runButton.disabled = false;
      runButton.textContent = 'Run detection';
    }
  }
});

function showServiceError(message = 'Detector disconnected.') {
  serviceStatus.dataset.state = 'error';
  serviceMessage.textContent = `${message} ${backendHelp}`;
  serviceLink.hidden = !localSite;
}

async function checkService(attempt = 0) {
  try {
    const response = await fetch('/api/health', { cache: 'no-store' });
    if (!response.ok || !response.headers.get('content-type')?.includes('application/json')) throw new Error('unavailable');
    const health = await response.json();
    if (!health.model_available) {
      serviceStatus.dataset.state = 'error';
      serviceMessage.textContent = localSite ? 'Model files are missing. Follow the setup steps in README.md.' : 'The detector is temporarily unavailable.';
      serviceLink.hidden = true;
      return;
    }
    if (!health.ready) {
      serviceStatus.dataset.state = 'error';
      serviceMessage.textContent = localSite ? health.problem || 'Detector calibration is incomplete. See README.md.' : 'The detector is temporarily unavailable.';
      serviceLink.hidden = true;
      return;
    }
    serviceStatus.dataset.state = 'ready';
    serviceMessage.textContent = 'Detector connected. Choose a file to see the model estimate.';
    serviceLink.hidden = true;
  } catch {
    if (attempt < 2) {
      serviceStatus.dataset.state = 'loading';
      serviceMessage.textContent = 'Waking the detector…';
      setTimeout(() => checkService(attempt + 1), 2000);
      return;
    }
    showServiceError();
  }
}

checkService();

input.addEventListener('change', () => showFile(input.files[0]));
document.querySelector('#choose-file').addEventListener('click', () => input.click());
document.querySelector('#clear-file').addEventListener('click', clearFile);
zone.addEventListener('dragover', event => {
  event.preventDefault();
  zone.classList.add('drag-over');
});
zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
zone.addEventListener('drop', event => {
  event.preventDefault();
  zone.classList.remove('drag-over');
  showFile(event.dataTransfer.files[0]);
});
window.addEventListener('pagehide', () => {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
});
