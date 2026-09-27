(() => {
  const section = document.getElementById('relations');
  if (!section) return;

  const diagram = document.getElementById('relation-diagram');
  const title = document.getElementById('relation-model-title');
  const description = document.getElementById('relation-model-description');
  const selection = document.getElementById('relation-selection');
  const buttons = [...section.querySelectorAll('[data-relation-model]')];
  const points = {
    A: [80, 80], B: [40, 180], C: [145, 210], D: [210, 125],
    E: [280, 60], F: [355, 160], G: [300, 235],
  };
  const outlines = {
    ABCD: 'M64 44 C103 35 165 49 209 90 C245 120 225 175 190 211 C154 247 66 237 22 202 C-2 175 7 123 30 87 C39 70 48 54 64 44 Z',
    DEF: 'M190 106 C202 70 252 31 284 33 C327 35 376 104 386 151 C397 195 348 203 304 189 C266 178 208 186 190 153 C181 138 182 122 190 106 Z',
    CG: 'M124 197 C139 181 169 184 202 188 L302 207 C329 211 335 234 319 250 C304 265 274 262 243 256 L152 239 C124 235 110 216 124 197 Z',
  };
  const graphEdges = ['AB', 'AD', 'BC', 'CD', 'CG', 'DE', 'DF', 'EF'];
  const models = {
    graph: {
      title: 'Graph', description: 'Each edge connects two nodes.',
      example: 'Edges A–B and B–C do not imply an A–C edge.',
      edges: graphEdges, cells: [],
    },
    hypergraph: {
      title: 'Hypergraph',
      description: 'A hyperedge connects any nonempty set of nodes. Its subsets need not be hyperedges.',
      example: 'The hyperedge {A, B, C, D} does not require an A–C edge.',
      edges: [], cells: ['ABCD', 'DEF', 'CG'],
    },
    simplicial: {
      title: 'Simplicial complex',
      description: 'Nodes, edges, and higher-order groups form a hierarchy. Every group requires all its nonempty subsets as relations.',
      example: 'The filled triangle ABC includes edges AB, AC, BC, and all three nodes.',
      edges: ['AB', 'AC', 'BC', 'AD', 'CD', 'CG', 'DE', 'DF', 'EF'],
      cells: ['ABC', 'ACD', 'DEF'],
    },
    combinatorial: {
      title: 'Combinatorial complex',
      description: 'Keeps a hierarchy of relations without requiring every subset to be a relation.',
      example: 'Here, three- and four-node cells both have rank 2; A–C is absent.',
      edges: graphEdges, cells: ['ABCD', 'DEF'],
    },
  };
  let current = 'graph';
  let pinned = null;
  let preview = null;

  function svg(tag, attributes = {}, text = '') {
    const element = document.createElementNS('http://www.w3.org/2000/svg', tag);
    Object.entries(attributes).forEach(([name, value]) => element.setAttribute(name, value));
    if (text) element.textContent = text;
    return element;
  }

  function highlight() {
    const node = preview || pinned;
    const model = models[current];
    const cells = node ? model.cells.filter(cell => cell.includes(node)) : [];
    const edges = node ? model.edges.filter(edge => edge.includes(node)
      || cells.some(cell => [...edge].every(member => cell.includes(member)))) : [];
    const members = new Set([node, ...cells.join(''), ...edges.join('')]);
    diagram.classList.toggle('has-selection', Boolean(node));
    diagram.querySelectorAll('[data-members]').forEach(element => {
      const active = [...cells, ...edges].includes(element.dataset.members);
      element.classList.toggle('is-active', active);
    });
    diagram.querySelectorAll('[data-node]').forEach(element => {
      element.classList.toggle('is-active', Boolean(node) && members.has(element.dataset.node));
      element.classList.toggle('is-selected', element.dataset.node === node);
      element.setAttribute('aria-pressed', String(element.dataset.node === pinned));
    });

    if (!node) {
      selection.textContent = model.example;
    } else if (current === 'graph') {
      selection.textContent = `${node}: ${edges.length} pairwise relations.`;
    } else if (current === 'hypergraph') {
      selection.textContent = `${node}: ${cells.length === 1 ? 'hyperedge' : 'hyperedges'} ${cells.join(', ')}.`;
    } else if (current === 'simplicial') {
      selection.textContent = cells.length
        ? `${node}: ${cells.length === 1 ? 'triangle' : 'triangles'} ${cells.join(', ')} with all ${cells.length === 1 ? 'its' : 'their'} edges and nodes${node === 'C' ? ', plus edge CG' : ''}.`
        : `${node}: a pairwise relation with C.`;
    } else {
      selection.textContent = cells.length
        ? `${node}: rank-2 ${cells.length === 1 ? 'cell' : 'cells'} ${cells.join(', ')} and rank-1 edges.`
        : `${node}: a rank-1 edge connecting it to C.`;
    }
  }

  function render(name) {
    current = name;
    section.dataset.model = name;
    pinned = null;
    preview = null;
    const model = models[name];
    title.textContent = model.title;
    description.textContent = model.description;
    diagram.setAttribute('aria-label', `${model.title}: seven entities and their relations`);
    buttons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.relationModel === name)));
    diagram.replaceChildren(svg('title', {}, `${model.title}: seven entities and their relations`));

    model.cells.forEach((members, index) => {
      const color = ['blue', 'purple', 'green'][index];
      const attributes = {
        class: 'relation-cell', 'data-members': members,
        style: `--relation-color: var(--relation-${color})`,
        fill: `var(--relation-${color})`, stroke: `var(--relation-${color})`,
      };
      const cell = name === 'simplicial'
        ? svg('polygon', { ...attributes, points: [...members].map(member => points[member].join(',')).join(' ') })
        : svg('path', { ...attributes, d: outlines[members] });
      diagram.append(cell);
    });
    model.edges.forEach(members => {
      const [x1, y1] = points[members[0]];
      const [x2, y2] = points[members[1]];
      diagram.append(svg('line', { class: 'relation-edge', 'data-members': members, x1, y1, x2, y2 }));
    });
    if (name === 'combinatorial') {
      diagram.append(svg('text', { class: 'relation-rank', x: 80, y: 32, 'text-anchor': 'middle', 'aria-hidden': 'true' }, 'rank 0'));
      diagram.append(svg('text', { class: 'relation-rank', x: 142, y: 75, 'text-anchor': 'middle', 'aria-hidden': 'true' }, 'rank 2'));
      diagram.append(svg('text', { class: 'relation-rank', x: 296, y: 127, 'text-anchor': 'middle', 'aria-hidden': 'true' }, 'rank 2'));
      diagram.append(svg('text', { class: 'relation-rank', x: 224, y: 263, 'text-anchor': 'middle', 'aria-hidden': 'true' }, 'rank 1'));
    }
    Object.entries(points).forEach(([node, [x, y]]) => {
      const group = svg('g', {
        class: 'relation-node', 'data-node': node, role: 'button', tabindex: 0,
        'aria-label': `Node ${node}. Show its relations.`, 'aria-pressed': 'false',
        transform: `translate(${x} ${y})`,
      });
      group.append(
        svg('circle', { class: 'relation-hit', r: 32, fill: 'transparent', 'pointer-events': 'all' }),
        svg('circle', { class: 'relation-node-mark', r: 17 }),
        svg('text', { 'text-anchor': 'middle', 'dominant-baseline': 'central', 'aria-hidden': 'true' }, node),
      );
      diagram.append(group);
    });
    highlight();
  }

  function nodeAt(target) {
    return target instanceof Element ? target.closest('[data-node]')?.dataset.node : null;
  }

  function toggle(node) {
    pinned = pinned === node ? null : node;
    preview = null;
    highlight();
  }

  buttons.forEach(button => button.addEventListener('click', () => {
    if (models[button.dataset.relationModel]) render(button.dataset.relationModel);
  }));
  diagram.addEventListener('pointerover', event => {
    if (event.pointerType === 'touch') return;
    preview = nodeAt(event.target);
    highlight();
  });
  diagram.addEventListener('pointerout', event => {
    if (event.pointerType === 'touch') return;
    preview = nodeAt(event.relatedTarget);
    highlight();
  });
  diagram.addEventListener('focusin', event => {
    preview = nodeAt(event.target);
    highlight();
  });
  diagram.addEventListener('focusout', event => {
    preview = nodeAt(event.relatedTarget);
    highlight();
  });
  diagram.addEventListener('click', event => {
    const node = nodeAt(event.target);
    if (node) toggle(node);
  });
  section.addEventListener('keydown', event => {
    const node = nodeAt(event.target);
    if (node && (event.key === 'Enter' || event.key === ' ')) {
      event.preventDefault();
      toggle(node);
    } else if (event.key === 'Escape') {
      pinned = null;
      preview = null;
      highlight();
    }
  });

  render(current);
  section.classList.add('relations-ready');
})();
