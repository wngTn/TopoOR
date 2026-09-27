'use strict';
// Apply the saved preference before paint; otherwise follow the operating system.
(() => {
  let theme;
  try { theme = localStorage.getItem('topoor-theme'); } catch { /* Storage is optional. */ }
  if (theme !== 'light' && theme !== 'dark') {
    theme = matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  document.documentElement.dataset.theme = theme;
})();
