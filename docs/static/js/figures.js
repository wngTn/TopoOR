'use strict';

// SVG exports can reuse IDs for clips, images, and gradients. Scope them per figure.
function namespaceSvg(svg, prefix) {
  const ids = new Map();
  svg.querySelectorAll('[id]').forEach(node => {
    const original = node.id;
    node.id = `${prefix}-${original}`;
    ids.set(original, node.id);
  });
  const rewrite = value => value.replace(/url\(\s*(['"]?)#([^\s)'"]+)\1\s*\)/g, (match, quote, id) => ids.has(id) ? `url(#${ids.get(id)})` : match);
  svg.querySelectorAll('*').forEach(node => {
    [...node.attributes].forEach(attribute => {
      if (attribute.localName === 'href' && attribute.value.startsWith('#')) {
        const target = ids.get(attribute.value.slice(1));
        if (target) attribute.value = `#${target}`;
      } else attribute.value = rewrite(attribute.value);
    });
    if (node.localName === 'style') node.textContent = rewrite(node.textContent);
  });
}

const methodStages = {
    I: ['Multimodal Inputs', 'Calibrated camera views, as well as, audio recordings, robot logs, and monitor images provide observations of the operating room.'],
    II: ['Perception & Initialization', 'Frozen perception modules estimate 3D human poses and object locations. Modality encoders turn multi-modal sensor cues into evidence features.'],
    III: ['Building the Complex', 'Human joints, objects, and evidence form rank-0 cells. Rank-1 edges encode skeletons and relations; rank-2 cells group a person’s skeleton or entities in a surgical interaction. Temporal edges link entities across frames.'],
    IV: ['Higher-Order Attention', 'Cells exchange features with their incident lower- and higher-rank neighbours. Attention weights include a learned bias for the source and target ranks, allowing information to flow between entities and groups.'],
    V: ['(Exemplary) Downstream Tasks', 'Learned heads can use the pooled representation to predict downstream tasks such as the next action and robot phase. Sterility detection applies distance thresholds to sterile and non-sterile entities.']
  };

const resultStages = {
  I: ['Robot-Phase Predictions', 'The timeline shows ground-truth robot phases and predictions from a Vanilla Transformer, SurgLatentGraph, and TopoOR. Colors identify the different phases', 'Robot-Phase-Prediction'],
  II: ['Estimated 3D Entities', 'The operating theater illustrated using estimated human poses and the locations of tools and equipment, with labels identifying roles and objects.', 'Estimated-3D-Entities'],
  III: ['Scene Representations', 'The Hasse-diagrams illustrate unstructured pairwise relations for the Vanilla Transformer, a graph for SurgLatentGraph, and a combinatorial complex for TopoOR. The TopoOR diagram includes rank-0 entities, rank-1 relations, and rank-2 person and functional cells.', 'Hasse-Diagrams']
};

// Enhance each supplied SVG while retaining its image if loading fails.
async function enhanceFigure(selectorId, stages, label) {
  const explorer = document.querySelector(selectorId);
  if (!explorer) return;
  const art = explorer.querySelector('.method-art');
  const fallback = art.querySelector('img');
  const description = explorer.querySelector('figcaption');
  const defaultDescription = description.innerHTML;

  try {
    const response = await fetch(fallback.src);
    if (!response.ok) throw new Error('Figure unavailable');
    const parsed = new DOMParser().parseFromString(await response.text(), 'image/svg+xml');
    if (parsed.querySelector('parsererror')) throw new Error('Invalid figure');
    const svg = document.importNode(parsed.documentElement, true);
    const groups = Object.keys(stages).map(id => svg.querySelector(`#${stages[id][2] || `Part-${id}`}`));
    if (groups.some(group => !group)) throw new Error('Missing figure stages');
    svg.classList.add('method-svg');
    svg.setAttribute('role', 'group');
    svg.setAttribute('aria-label', label);
    namespaceSvg(svg, explorer.id);
    svg.removeAttribute('width');
    svg.removeAttribute('height');
    art.append(svg);

    // Rectangles use each original group's coordinates, so they scale with the SVG.
    await document.fonts.ready;
    const buttons = [...explorer.querySelectorAll('[data-stage]')];
    const selector = explorer.querySelector('.method-stages');
    const aura = document.createElement('div');
    aura.className = 'method-aura';
    aura.setAttribute('aria-hidden', 'true');
    art.append(aura);
    const trail = document.createElement('span');
    trail.className = 'stage-trail';
    trail.setAttribute('aria-hidden', 'true');
    selector.append(trail);
    const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
    const targets = [...buttons, ...groups];
    let hovered = null;
    let focused = null;
    let pinned = null;
    let active;
    let leaveTimer;
    let renderFrame = null;
    let descriptionAnimation;
    const positionHighlight = () => {
      if (!active) return;
      const group = groups.find(item => item.dataset.stage === active);
      const bounds = group.getBoundingClientRect();
      const origin = art.getBoundingClientRect();
      aura.style.width = `${bounds.width + 20}px`;
      aura.style.height = `${bounds.height + 16}px`;
      aura.style.transform = `translate(${bounds.left - origin.left - 10}px, ${bounds.top - origin.top - 8}px)`;
      const button = buttons.find(item => item.dataset.stage === active);
      const buttonBounds = button.getBoundingClientRect();
      const selectorBounds = selector.getBoundingClientRect();
      trail.style.width = `${buttonBounds.width * .6}px`;
      trail.style.transform = `translateX(${buttonBounds.left - selectorBounds.left + buttonBounds.width * .2}px)`;
    };
    const commitSelection = () => {
      const next = hovered || focused || pinned;
      if (active === next) return;
      active = next;
      svg.classList.toggle('has-active-stage', Boolean(active));
      positionHighlight();
      aura.classList.toggle('is-visible', Boolean(active));
      trail.classList.toggle('is-visible', Boolean(active));
      targets.forEach(target => {
        const selected = target.dataset.stage === active;
        target.classList.toggle('is-active', selected);
        target.setAttribute('aria-pressed', String(selected));
      });
      if (!active) {
        description.innerHTML = defaultDescription;
      } else {
        const title = document.createElement('span');
        title.className = 'caption-title';
        title.textContent = `${stages[active][0]}. `;
        description.replaceChildren(title, document.createTextNode(stages[active][1]));
      }
      descriptionAnimation?.cancel();
      if (!reducedMotion.matches && typeof description.animate === 'function') {
        descriptionAnimation = description.animate([
          { opacity: .35, transform: 'translateY(4px)' },
          { opacity: 1, transform: 'translateY(0)' }
        ], { duration: 360, easing: 'cubic-bezier(.22, 1, .36, 1)' });
      }
    };
    const render = () => {
      if (renderFrame !== null) return;
      // Focus, blur, and click can all run during one selection. Paint it once.
      renderFrame = requestAnimationFrame(() => {
        renderFrame = null;
        commitSelection();
      });
    };
    groups.forEach((group, index) => {
      const id = Object.keys(stages)[index];
      group.classList.add('method-stage');
      group.dataset.stage = id;
      // The HTML selector supplies keyboard stops in reading order.
      group.setAttribute('tabindex', '-1');
      group.setAttribute('role', 'button');
      group.setAttribute('aria-label', `Stage ${id}: ${stages[id][0]}`);
      group.setAttribute('aria-controls', description.id);
      const bounds = group.getBBox();
      const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      Object.entries({ x: bounds.x - 18, y: bounds.y - 18, width: bounds.width + 36, height: bounds.height + 36, rx: 38 }).forEach(([name, value]) => rect.setAttribute(name, value));
      rect.classList.add('stage-outline');
      rect.setAttribute('aria-hidden', 'true');
      // A transparent rectangle makes whitespace within each stage interactive.
      group.append(rect);
      group.addEventListener('keydown', event => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          group.dispatchEvent(new MouseEvent('click'));
        }
      });
    });
    targets.forEach(target => {
      target.setAttribute('aria-controls', description.id);
      target.addEventListener('pointerenter', event => {
        if (event.pointerType === 'touch') return;
        clearTimeout(leaveTimer);
        hovered = target.dataset.stage;
        render();
      });
      target.addEventListener('pointerleave', () => {
        // Bridge the gaps between stages without flashing back to the overview.
        leaveTimer = setTimeout(() => { hovered = null; render(); }, 140);
      });
      target.addEventListener('pointerdown', event => {
        clearTimeout(leaveTimer);
        focused = null;
        hovered = event.pointerType === 'touch' ? null : target.dataset.stage;
        render();
      });
      target.addEventListener('focus', () => {
        const keyboardFocus = target.matches(':focus-visible');
        if (keyboardFocus) {
          clearTimeout(leaveTimer);
          hovered = null;
        }
        focused = keyboardFocus ? target.dataset.stage : null;
        render();
      });
      target.addEventListener('blur', () => { focused = null; render(); });
      target.addEventListener('click', () => {
        clearTimeout(leaveTimer);
        pinned = pinned === target.dataset.stage ? null : target.dataset.stage;
        render();
      });
    });
    explorer.addEventListener('keydown', event => {
      if (event.key === 'Escape') {
        clearTimeout(leaveTimer);
        hovered = focused = pinned = null;
        render();
      }
    });
    if ('ResizeObserver' in window) new ResizeObserver(positionHighlight).observe(explorer);
    render();
    fallback.hidden = true;
    selector.hidden = false;
    explorer.classList.add('is-interactive');
  } catch {
    art.querySelector('svg')?.remove();
    art.querySelector('.method-aura')?.remove();
    explorer.querySelector('.stage-trail')?.remove();
    // Keep the original figure available, including when opened from disk.
  }
}

function enhanceWhenVisible(selector, stages, label) {
  const figure = document.querySelector(selector);
  if (!figure) return;
  if (!('IntersectionObserver' in window)) {
    enhanceFigure(selector, stages, label);
    return;
  }
  const observer = new IntersectionObserver(entries => {
    if (!entries.some(entry => entry.isIntersecting)) return;
    observer.disconnect();
    enhanceFigure(selector, stages, label);
  });
  observer.observe(figure);
}

enhanceWhenVisible('#method-explorer', methodStages, 'TopoOR methodology: explore stages I to V');
enhanceWhenVisible('#results-explorer', resultStages, 'Qualitative results: explore panels I to III');
