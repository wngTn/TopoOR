'use strict';

const copyButton = document.querySelector('#copy-citation');
const citation = document.querySelector('#bibtex');
const copyStatus = document.querySelector('#copy-status');
if (copyButton && citation) {
  copyButton.hidden = false;
  copyButton.addEventListener('click', async () => {
    try {
      if (!navigator.clipboard) throw new Error('Clipboard unavailable');
      await navigator.clipboard.writeText(citation.textContent.trim());
      copyStatus.textContent = 'Citation copied.';
    } catch {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(citation);
      selection.removeAllRanges();
      selection.addRange(range);
      citation.parentElement.focus();
      copyStatus.textContent = 'Citation selected. Press Ctrl+C or ⌘C to copy.';
    }
  });
}

const figureDialog = document.querySelector('#figure-dialog');
if (figureDialog && typeof figureDialog.showModal === 'function') {
  const figureImage = document.querySelector('#enlarged-figure');
  const viewport = document.querySelector('.dialog-image-area');
  const zoomButton = document.querySelector('#figure-zoom');
  const hint = document.querySelector('.figure-hint');
  const compactView = matchMedia('(max-width: 600px), (pointer: coarse) and (max-height: 600px)');
  let trigger;
  const setZoom = (zoomed, point) => {
    viewport.classList.toggle('zoomed', zoomed);
    zoomButton.textContent = zoomed ? 'Fit' : 'Zoom';
    zoomButton.setAttribute('aria-label', zoomed ? 'Fit figure to screen' : 'Zoom into figure');
    zoomButton.setAttribute('aria-pressed', String(zoomed));
    hint.textContent = zoomed ? 'Swipe to explore. Use Fit to see the whole figure.' : 'Use Zoom to explore the details.';
    if (zoomed && point) {
      viewport.scrollTo(point.x * figureImage.clientWidth - viewport.clientWidth / 2,
        point.y * figureImage.clientHeight - viewport.clientHeight / 2);
    } else viewport.scrollTo(0, 0);
  };
  zoomButton.addEventListener('click', () => setZoom(!viewport.classList.contains('zoomed')));
  figureImage.addEventListener('click', event => {
    if (event.pointerType === 'touch' || matchMedia('(pointer: coarse)').matches) return;
    const bounds = figureImage.getBoundingClientRect();
    setZoom(!viewport.classList.contains('zoomed'), {
      x: (event.clientX - bounds.left) / bounds.width,
      y: (event.clientY - bounds.top) / bounds.height
    });
  });
  compactView.addEventListener('change', () => {
    if (figureDialog.open) setZoom(compactView.matches);
  });
  document.querySelectorAll('[data-figure]').forEach(button => {
    button.addEventListener('click', () => {
      trigger = button;
      figureImage.src = button.dataset.source;
      figureImage.alt = button.dataset.alt;
      document.querySelector('#figure-dialog-title').textContent = button.dataset.title;
      figureDialog.showModal();
      document.body.classList.add('figure-open');
      setZoom(compactView.matches);
      document.querySelector('#close-figure').focus({ preventScroll: true });
    });
  });
  document.querySelector('#close-figure').addEventListener('click', () => figureDialog.close());
  figureDialog.addEventListener('click', event => {
    const bounds = figureDialog.getBoundingClientRect();
    if (event.target === figureDialog && (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom)) figureDialog.close();
  });
  figureDialog.addEventListener('close', () => {
    document.body.classList.remove('figure-open');
    setZoom(false);
    trigger?.focus();
  });
}

const themeToggle = document.querySelector('#theme-toggle');
if (themeToggle) {
  const systemTheme = matchMedia('(prefers-color-scheme: dark)');
  const applyTheme = theme => {
    document.documentElement.dataset.theme = theme;
    themeToggle.setAttribute('aria-label', `Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`);
    document.querySelector('meta[name="theme-color"]').content = theme === 'dark' ? '#101725' : '#fafbfd';
  };
  themeToggle.hidden = false;
  applyTheme(document.documentElement.dataset.theme || 'light');
  themeToggle.addEventListener('click', () => {
    const theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    applyTheme(theme);
    try { localStorage.setItem('topoor-theme', theme); } catch { /* Theme still works without storage. */ }
  });
  systemTheme.addEventListener('change', event => {
    let saved;
    try { saved = localStorage.getItem('topoor-theme'); } catch { /* Use system preference. */ }
    if (!saved) applyTheme(event.matches ? 'dark' : 'light');
  });
}

const video = document.querySelector('#teaser-video');
const player = document.querySelector('#teaser-player');
if (video && player) {
  const start = document.querySelector('#video-start');
  const toggle = document.querySelector('#video-toggle');
  const seek = document.querySelector('#video-seek');
  const fullscreen = document.querySelector('#video-fullscreen');
  let pendingSeek = null;
  let inView = false;
  let manuallyPaused = false;
  const clock = value => `${Math.floor(value / 60)}:${String(Math.floor(value % 60)).padStart(2, '0')}`;
  const update = () => {
    const duration = Number.isFinite(video.duration) ? video.duration : 45;
    seek.max = duration;
    seek.value = video.currentTime;
    seek.setAttribute('aria-valuetext', `${clock(video.currentTime)} of ${clock(Math.round(duration))}`);
    document.querySelector('#video-time').textContent = clock(video.currentTime);
    document.querySelector('#video-duration').textContent = clock(Math.round(duration));
    player.classList.toggle('playing', !video.paused);
    start.hidden = video.controls || !video.paused;
    toggle.setAttribute('aria-label', video.paused ? 'Play teaser' : 'Pause teaser');
  };
  const playPause = async () => {
    if (video.paused) {
      manuallyPaused = false;
      try { await video.play(); } catch (error) {
        if (error.name === 'AbortError') return;
        // Preserve native playback as a fallback if the custom play request fails.
        video.controls = true;
        player.classList.remove('enhanced');
        start.hidden = true;
        document.querySelector('#video-controls').hidden = true;
      }
    } else {
      manuallyPaused = true;
      video.pause();
    }
  };
  start.addEventListener('click', () => { toggle.focus(); playPause(); });
  toggle.addEventListener('click', playPause);
  video.addEventListener('click', playPause);
  ['play', 'pause', 'ended', 'timeupdate', 'loadedmetadata'].forEach(event => video.addEventListener(event, update));
  seek.addEventListener('input', () => {
    const position = Number(seek.value);
    if (video.readyState > 0) video.currentTime = position;
    else {
      pendingSeek = position;
      video.preload = 'metadata';
      video.load();
    }
  });
  video.addEventListener('loadedmetadata', () => {
    if (pendingSeek !== null) { video.currentTime = pendingSeek; pendingSeek = null; }
  });
  seek.addEventListener('pointerdown', () => { if (video.readyState === 0) { video.preload = 'metadata'; video.load(); } });
  fullscreen.hidden = !player.requestFullscreen && !video.webkitEnterFullscreen;
  fullscreen.addEventListener('click', async () => {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else if (player.requestFullscreen) await player.requestFullscreen();
      else if (video.webkitEnterFullscreen) video.webkitEnterFullscreen();
    } catch { /* Native fullscreen may be unavailable in embedded browsers. */ }
  });
  document.addEventListener('fullscreenchange', () => fullscreen.setAttribute('aria-label', document.fullscreenElement ? 'Exit video fullscreen' : 'Enter video fullscreen'));
  video.controls = false;
  video.muted = true;
  player.classList.add('enhanced');
  document.querySelector('#video-controls').hidden = false;
  update();

  const syncPlayback = () => {
    if (!inView || document.hidden) {
      video.pause();
      return;
    }
    if (manuallyPaused || !video.paused) return;
    // A browser may reject autoplay; the play button remains available.
    video.play().catch(() => { /* Leave playback to the user's play button. */ });
  };
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(([entry]) => {
      inView = entry.isIntersecting && entry.intersectionRatio >= 0.35;
      syncPlayback();
    }, { threshold: [0, 0.35] });
    observer.observe(video);
    document.addEventListener('visibilitychange', syncPlayback);
  }
}
