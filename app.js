import { parseHlo, nodeCategory, reachable, computationLinks, nodeSummary } from './parser.js';
import { instructionGuide, layoutDiagram } from './instruction-guide.js';

const $ = (id) => document.getElementById(id);
const state = { module: null, current: null, selected: null, history: [], zoom: 1, positions: new Map() };
const CARD_W = 218, CARD_H = 91, GAP_X = 94, GAP_Y = 31, PAD_X = 58, PAD_Y = 66;
const escapeHtml = (text) => String(text).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

function loadText(text) {
  const module = parseHlo(text);
  if (!module.computations.length) throw new Error('No computations found. Paste a textual HLO module with %computation { … } blocks.');
  state.module = module;
  state.history = [];
  state.selected = null;
  state.current = null;
  $('module-name').textContent = module.name;
  $('module-stats').textContent = `${module.computations.length} computations · ${module.computations.reduce((n,c) => n+c.nodes.length, 0)} instructions`;
  $('computation-count').textContent = module.computations.length;
  render();
}

function openComputation(name, focusName = null) {
  if (!state.module.byName.has(name)) return;
  if (state.current !== name) state.history.push({ current: state.current, selected: state.selected });
  state.current = name;
  state.selected = focusName;
  state.zoom = 1;
  render();
  $('graph-scroller').scrollTo(0, 0);
}

function openOverview() {
  if (state.current !== null) state.history.push({ current: state.current, selected: state.selected });
  state.current = null;
  state.selected = null;
  render();
  requestAnimationFrame(fitView);
}

function chooseNode(name) {
  state.selected = state.selected === name ? null : name;
  updateFocus();
}

function render() {
  if (state.current === null) {
    $('crumb-current').textContent = 'Overview';
    $('view-kicker').textContent = 'MODULE MAP';
    $('view-title').textContent = 'Computation overview';
    $('view-description').textContent = 'Arrows show which computation an instruction invokes. Click a computation to inspect its instruction dependencies.';
    $('back-button').hidden = !state.history.length;
    $('warning-count').textContent = state.module.warnings.length ? `${state.module.warnings.length} parse notes` : 'Parsed without warnings';
    $('selection-summary').textContent = 'Computation links · instruction data edges are inside each computation';
    document.body.classList.remove('inspect-open');
    renderSidebar();
    renderOverview();
    return;
  }
  const c = state.module.byName.get(state.current);
  $('crumb-current').textContent = c.entry ? 'Entry' : c.name;
  $('view-kicker').textContent = c.entry ? 'ENTRY COMPUTATION' : 'SUBCOMPUTATION';
  $('view-title').textContent = `%${c.name}`;
  $('view-description').textContent = c.entry ? 'Follow the inputs into the loop, then open its body or condition.' : `The ${c.nodes.length} instructions in this computation, shown in data dependency order.`;
  $('back-button').hidden = !state.history.length;
  $('warning-count').textContent = state.module.warnings.length ? `${state.module.warnings.length} parse notes` : 'Parsed without warnings';
  renderSidebar();
  renderGraph(c);
  updateFocus();
}

function renderSidebar() {
  const list = $('computation-list');
  list.replaceChildren();
  const mobile = $('mobile-computations');
  mobile.replaceChildren();
  const module = state.module;
  const overviewOption = document.createElement('option');
  overviewOption.value = '';
  overviewOption.textContent = 'Overview · all computations';
  overviewOption.selected = state.current === null;
  mobile.append(overviewOption);
  const overview = document.createElement('button');
  overview.type = 'button';
  overview.className = `computation-item overview-item${state.current === null ? ' active' : ''}`;
  overview.innerHTML = '<span class="comp-icon">▦</span><span class="comp-copy"><strong>Overview</strong><small>COMPUTATION LINKS</small></span><span class="comp-arrow">›</span>';
  overview.addEventListener('click', openOverview);
  list.append(overview);
  const ordered = [...module.computations].sort((a,b) => Number(b.entry) - Number(a.entry));
  for (const c of ordered) {
    const option = document.createElement('option');
    option.value = c.name;
    option.textContent = `%${c.name}${c.entry ? ' · entry' : ''}`;
    option.selected = c.name === state.current;
    mobile.append(option);
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `computation-item${c.name === state.current ? ' active' : ''}`;
    const role = c.entry ? 'ENTRY' : (c.name.includes('region_0') ? 'WHILE BODY' : c.name.includes('region_1') ? 'WHILE CONDITION' : c.name.includes('fusion') ? 'FUSION' : 'COMPUTATION');
    button.innerHTML = `<span class="comp-icon">${c.entry ? '⌂' : role === 'FUSION' ? '✦' : role.startsWith('WHILE') ? '↻' : '◇'}</span><span class="comp-copy"><strong title="%${escapeHtml(c.name)}">%${escapeHtml(c.name)}</strong><small>${role} · ${c.nodes.length} nodes</small></span><span class="comp-arrow">›</span>`;
    button.addEventListener('click', () => openComputation(c.name));
    list.append(button);
  }
}

function renderOverview() {
  const module = state.module;
  const links = computationLinks(module);
  const incoming = new Map(module.computations.map(c => [c.name, []]));
  const outgoing = new Map(module.computations.map(c => [c.name, []]));
  for (const link of links) { incoming.get(link.to).push(link); outgoing.get(link.from).push(link); }
  const roots = module.computations.filter(c => c.entry || !incoming.get(c.name).length);
  const levels = new Map();
  const queue = [...roots];
  roots.forEach(c => levels.set(c.name, 0));
  while (queue.length) {
    const current = queue.shift();
    const children = [...outgoing.get(current.name)].sort((a,b) =>
      (a.role === 'body' ? -1 : a.role === 'condition' ? 1 : 0) -
      (b.role === 'body' ? -1 : b.role === 'condition' ? 1 : 0));
    for (const link of children) {
      if (levels.has(link.to)) continue;
      levels.set(link.to, levels.get(current.name) + 1);
      queue.push(module.byName.get(link.to));
    }
  }
  // Cyclic or unusual modules may have no root; still show every definition.
  for (const c of module.computations) if (!levels.has(c.name)) levels.set(c.name, 0);
  const columns = new Map();
  for (const c of module.computations) {
    const level = levels.get(c.name);
    if (!columns.has(level)) columns.set(level, []);
    columns.get(level).push(c);
  }
  const roleFor = c => c.entry ? 'ENTRY' : incoming.get(c.name).some(x => x.role === 'body') ? 'WHILE BODY' : incoming.get(c.name).some(x => x.role === 'condition') ? 'WHILE CONDITION' : incoming.get(c.name).some(x => x.op === 'fusion') ? 'FUSION' : 'COMPUTATION';
  for (const group of columns.values()) group.sort((a,b) => {
    const priority = c => roleFor(c) === 'WHILE BODY' ? -1 : roleFor(c) === 'WHILE CONDITION' ? 1 : 0;
    return priority(a) - priority(b) || module.computations.indexOf(a) - module.computations.indexOf(b);
  });
  const W = 212, H = 116, DX = 64, DY = 68, PX = 48, PY = 60;
  const maxRows = Math.max(1, ...[...columns.values()].map(group => group.length));
  const width = PX*2 + (Math.max(...columns.keys())+1)*W + (columns.size-1)*DX;
  const height = PY*2 + maxRows*H + (maxRows-1)*DY;
  const positions = new Map();
  for (const [level, group] of columns) {
    const offset = (maxRows-group.length)*(H+DY)/2;
    group.forEach((c,row) => positions.set(c.name, {x: PX+level*(W+DX), y: PY+offset+row*(H+DY)}));
  }
  const stage = $('graph-stage');
  stage.style.width = `${width}px`; stage.style.height = `${height}px`;
  const svg = $('graph-edges');
  svg.setAttribute('width', width); svg.setAttribute('height', height);
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.replaceChildren();
  const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
  defs.innerHTML = '<marker id="call-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8" /></marker>';
  svg.append(defs);
  for (const link of links) {
    const from = positions.get(link.from), to = positions.get(link.to);
    const x1 = from.x+W, y1 = from.y+H/2, x2 = to.x, y2 = to.y+H/2;
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', `M${x1} ${y1} C${x1+32} ${y1},${x2-32} ${y2},${x2-8} ${y2}`);
    path.setAttribute('class','call-edge'); path.setAttribute('marker-end','url(#call-arrow)');
    const title = document.createElementNS('http://www.w3.org/2000/svg','title');
    title.textContent = `%${link.from} → %${link.to} via %${link.via} (${link.role})`;
    path.append(title); svg.append(path);
    const label = document.createElementNS('http://www.w3.org/2000/svg','text');
    label.setAttribute('x', (x1+x2)/2-4); label.setAttribute('y', (y1+y2)/2-9);
    label.setAttribute('class','call-label'); label.setAttribute('text-anchor','middle');
    label.textContent = link.role === 'condition' ? 'COND' : link.role === 'body' ? 'BODY' : 'CALL';
    svg.append(label);
  }
  const container = $('graph-nodes'); container.replaceChildren();
  for (const c of module.computations) {
    const pos = positions.get(c.name), role = roleFor(c);
    const card = document.createElement('button'); card.type = 'button';
    card.className = `overview-card ${role.toLowerCase().replaceAll(' ','-')}`;
    card.style.left = `${pos.x}px`; card.style.top = `${pos.y}px`;
    card.innerHTML = `<span class="overview-role">${escapeHtml(role)}</span><strong title="%${escapeHtml(c.name)}">%${escapeHtml(c.name)}</strong><span class="overview-meta">${c.nodes.length} instructions <span>Open graph ↗</span></span>`;
    card.addEventListener('click', () => openComputation(c.name));
    container.append(card);
  }
  $('graph-hint-kind').textContent = 'Calls flow left to right';
  $('graph-summary').textContent = `${module.computations.length} computations · ${links.length} call links`;
  setZoom(state.zoom);
}

function layout(c) {
  const levels = new Map();
  const visiting = new Set();
  const depth = (node) => {
    if (levels.has(node.name)) return levels.get(node.name);
    if (visiting.has(node.name)) return 0;
    visiting.add(node.name);
    const result = node.operands.length ? 1 + Math.max(...node.operands.map(name => c.byName.has(name) ? depth(c.byName.get(name)) : -1)) : 0;
    visiting.delete(node.name);
    levels.set(node.name, result);
    return result;
  };
  c.nodes.forEach(depth);
  const columns = new Map();
  for (const node of c.nodes) {
    const level = levels.get(node.name);
    if (!columns.has(level)) columns.set(level, []);
    columns.get(level).push(node);
  }
  // Place nodes close to the vertical center of their inputs. This reduces
  // crossings while preserving the instruction order for independent nodes.
  for (const level of [...columns.keys()].sort((a,b) => a-b)) {
    if (level === 0) continue;
    const previous = columns.get(level-1) || [];
    const rank = new Map(previous.map((n,i) => [n.name, i]));
    columns.get(level).sort((a,b) => {
      const avg = (n) => { const hits = n.operands.map(x => rank.get(x)).filter(x => x !== undefined); return hits.length ? hits.reduce((x,y)=>x+y,0)/hits.length : Infinity; };
      return avg(a) - avg(b) || c.nodes.indexOf(a) - c.nodes.indexOf(b);
    });
  }
  const maxRows = Math.max(...[...columns.values()].map(x => x.length), 1);
  const positions = new Map();
  for (const [level, nodes] of columns) {
    const offset = (maxRows - nodes.length) * (CARD_H + GAP_Y) / 2;
    nodes.forEach((node, row) => positions.set(node.name, { x: PAD_X + level*(CARD_W+GAP_X), y: PAD_Y + offset + row*(CARD_H+GAP_Y) }));
  }
  return { positions, width: PAD_X*2 + (Math.max(...columns.keys())+1)*CARD_W + (columns.size-1)*GAP_X, height: PAD_Y*2 + maxRows*CARD_H + (maxRows-1)*GAP_Y };
}

function renderGraph(c) {
  const { positions, width, height } = layout(c);
  state.positions = positions;
  const stage = $('graph-stage');
  stage.style.width = `${width}px`;
  stage.style.height = `${height}px`;
  const svg = $('graph-edges');
  svg.setAttribute('width', width); svg.setAttribute('height', height);
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.replaceChildren();
  const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
  defs.innerHTML = '<marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7" /></marker>';
  svg.append(defs);
  let edgeCount = 0;
  for (const target of c.nodes) {
    const to = positions.get(target.name);
    target.operands.forEach((sourceName, operandIndex) => {
      const from = positions.get(sourceName);
      if (!from) return;
      const x1 = from.x+CARD_W, y1 = from.y+CARD_H/2;
      const x2 = to.x, y2 = to.y+CARD_H/2 + (operandIndex-(target.operands.length-1)/2)*15;
      const dx = Math.max(48, (x2-x1)*0.48);
      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      path.setAttribute('d', `M${x1} ${y1} C${x1+dx} ${y1},${x2-dx} ${y2},${x2-8} ${y2}`);
      path.setAttribute('class', 'edge');
      path.dataset.from = sourceName; path.dataset.to = target.name;
      path.setAttribute('marker-end','url(#arrow)');
      svg.append(path); edgeCount++;
    });
  }
  const container = $('graph-nodes');
  container.replaceChildren();
  for (const node of c.nodes) {
    const pos = positions.get(node.name);
    const card = document.createElement('button');
    card.type = 'button';
    card.className = `node-card ${nodeCategory(node)}`;
    card.style.left = `${pos.x}px`; card.style.top = `${pos.y}px`;
    card.dataset.node = node.name;
    const detail = nodeSummary(node);
    card.innerHTML = `<span class="node-top"><span class="node-op">${escapeHtml(node.op)}</span>${node.root ? '<span class="root-tag">ROOT</span>' : ''}</span><strong title="%${escapeHtml(node.name)}">%${escapeHtml(node.name)}</strong><span class="node-detail" title="${escapeHtml(detail)}">${escapeHtml(detail)}</span><span class="node-port left"></span><span class="node-port right"></span>`;
    card.addEventListener('click', () => chooseNode(node.name));
    container.append(card);
  }
  $('graph-summary').textContent = `${c.nodes.length} nodes · ${edgeCount} data edges`;
  $('graph-hint-kind').textContent = 'Data flows left to right';
  setZoom(state.zoom);
}

function updateFocus() {
  const c = state.module.byName.get(state.current);
  const selected = c.byName.get(state.selected);
  document.body.classList.toggle('inspect-open', !!selected);
  const up = selected ? reachable(c, selected.name, 'up') : new Set();
  const down = selected ? reachable(c, selected.name, 'down') : new Set();
  for (const card of $('graph-nodes').children) {
    const name = card.dataset.node;
    card.classList.toggle('selected', !!selected && name === selected.name);
    card.classList.toggle('upstream', up.has(name));
    card.classList.toggle('downstream', down.has(name));
    card.classList.toggle('dimmed', !!selected && name !== selected.name && !up.has(name) && !down.has(name));
  }
  for (const edge of $('graph-edges').querySelectorAll('.edge')) {
    const {from,to} = edge.dataset;
    edge.classList.toggle('upstream', !!selected && (to === selected.name || (up.has(from) && up.has(to))));
    edge.classList.toggle('downstream', !!selected && (from === selected.name || (down.has(from) && down.has(to))));
    edge.classList.toggle('dimmed', !!selected && !edge.classList.contains('upstream') && !edge.classList.contains('downstream'));
  }
  $('selection-summary').textContent = selected ? `%${selected.name} · ${up.size} upstream · ${down.size} downstream` : 'Select a node to inspect its dependencies';
  renderInspector(c, selected, up, down);
}

function renderInspector(c, node, up, down) {
  const target = $('inspector-content');
  if (!node) {
    target.innerHTML = '<div class="empty-inspector"><div class="empty-icon">◇</div><strong>Explore the graph</strong><p>Select any node to see its inputs, consumers, raw HLO, and linked computations.</p></div>';
    return;
  }
  const linked = Object.entries(node.calls).map(([role,name]) => `<button class="jump-link" data-comp="${escapeHtml(name)}"><span><small>${escapeHtml(role.toUpperCase())}</small><strong>%${escapeHtml(name)}</strong></span><span>↗</span></button>`).join('');
  const refs = (names) => names.length ? names.map(name => `<button class="reference" data-node="${escapeHtml(name)}"><span>%${escapeHtml(name)}</span><span>↗</span></button>`).join('') : '<p class="none">None in this computation</p>';
  const guide = instructionGuide(node, { computation: c, module: state.module });
  const guideRow = part => `<div class="guide-row part-${part.key}" data-part="${part.key}" tabindex="0"><span class="guide-swatch"></span><div><strong>${escapeHtml(part.label)}</strong><p>${escapeHtml(part.text)}</p></div></div>`;
  const typeKeys = new Set(['tuple','shape','order','tile','space','dest','source','context',...guide.resultGroup.slots?.map(slot => slot.key) || []]);
  const explanations = guide.parts.filter(part => !typeKeys.has(part.key)).map(guideRow).join('');
  const detailRows = details => details.map(detail => `<div class="type-detail"><strong>${escapeHtml(detail.label)}</strong><p>${escapeHtml(detail.text)}</p></div>`).join('');
  const tuple = guide.resultGroup.kind === 'tuple';
  const typeTree = `<details class="type-tree part-${tuple ? 'tuple' : 'shape'}" data-part="${tuple ? 'tuple' : 'shape'}" open>
    <summary><span class="type-tree-title">结果类型 <b>${tuple ? `${guide.resultGroup.slots.length} 项 tuple` : /\[\]/.test(guide.resultGroup.rawType) ? '标量' : '数组'}</b></span><code>${escapeHtml(guide.resultGroup.rawType)}</code></summary>
    <div class="type-tree-children">${tuple ? guide.resultGroup.slots.map(slot => {
      const explanation = guide.parts.find(part => part.key === slot.key);
      return `<details class="type-child part-${slot.key}" data-part="${slot.key}"><summary><span>${escapeHtml(slot.label)}</span><code>${escapeHtml(slot.raw)}</code></summary><div class="type-child-body">${explanation ? `<p class="type-child-intro">${escapeHtml(explanation.text)}</p>` : ''}${detailRows(slot.details)}</div></details>`;
    }).join('') : detailRows(guide.resultGroup.details)}</div></details>`;
  const diagram = layoutDiagram(node) || '';
  const displayType = node.type.startsWith('(') ? nodeSummary(node) : node.type;
  const semanticsLink = node.op === 'copy-start' ? '<a href="https://openxla.org/xla/operation_semantics#copy" target="_blank" rel="noreferrer">Copy</a>' : node.type.startsWith('(') ? '<a href="https://openxla.org/xla/operation_semantics#tuple" target="_blank" rel="noreferrer">Tuple</a>' : '<a href="https://openxla.org/xla/operation_semantics" target="_blank" rel="noreferrer">Operations</a>';
  target.innerHTML = `
    <div class="inspector-title"><span class="type-badge ${nodeCategory(node)}">${escapeHtml(node.op)}</span>${node.root ? '<span class="root-pill">ROOT</span>' : ''}<h3>%${escapeHtml(node.name)}</h3><div class="muted">${escapeHtml(displayType)}</div></div>
    <div class="metric-row"><div><strong>${up.size}</strong><span>Upstream</span></div><div><strong>${down.size}</strong><span>Downstream</span></div><div><strong>${node.line}</strong><span>Source line</span></div></div>
    <div class="inspector-section instruction-section"><h4>HLO instruction</h4><pre class="hlo-code">${guide.html}</pre>${node.op === 'copy-start' ? diagram : ''}<div class="guide-heading">逐段解读 <span>点按彩色片段定位说明</span></div><div class="guide-list">${typeTree}${explanations}</div>${node.op === 'copy-start' ? '' : diagram}<div class="guide-sources">参考：<a href="https://jax-ml.github.io/scaling-book/profiling/#how-to-read-an-xla-op" target="_blank" rel="noreferrer">How to read an XLA op</a> · <a href="https://openxla.org/xla/shapes" target="_blank" rel="noreferrer">Shapes and layout</a> · ${semanticsLink}</div></div>
    <div class="inspector-section"><h4>Direct inputs <span>${node.operands.length}</span></h4>${refs(node.operands)}</div>
    <div class="inspector-section"><h4>Direct consumers <span>${node.users.length}</span></h4>${refs(node.users)}</div>
    ${linked ? `<div class="inspector-section"><h4>Open computation</h4>${linked}</div>` : ''}`;
  const activatePart = key => target.querySelectorAll('[data-part]').forEach(el => el.classList.toggle('active-part', !!key && el.dataset.part === key));
  target.querySelectorAll('.hlo-part, .hlo-slot, .guide-row').forEach(el => {
    el.addEventListener('pointerenter', () => activatePart(el.dataset.part));
    el.addEventListener('pointerleave', () => activatePart(null));
    el.addEventListener('focus', () => activatePart(el.dataset.part));
    el.addEventListener('blur', () => activatePart(null));
    el.addEventListener('click', () => {
      activatePart(el.dataset.part);
      if (!el.classList.contains('guide-row')) {
        const item = target.querySelector(`.type-child[data-part="${el.dataset.part}"]`) ||
          (typeKeys.has(el.dataset.part) ? target.querySelector('.type-tree') : null) ||
          target.querySelector(`.guide-row[data-part="${el.dataset.part}"]`);
        if (item) {
          item.closest('.type-tree')?.setAttribute('open', '');
          if (item.classList.contains('type-child')) item.open = true;
          item.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
        }
      }
    });
  });
  target.querySelectorAll('[data-node]').forEach(button => button.addEventListener('click', () => { state.selected = button.dataset.node; updateFocus(); centerNode(button.dataset.node); }));
  target.querySelectorAll('[data-comp]').forEach(button => button.addEventListener('click', () => openComputation(button.dataset.comp)));
}

function centerNode(name) {
  const pos = state.positions.get(name);
  if (!pos) return;
  const scroller = $('graph-scroller');
  scroller.scrollTo({ left: (pos.x+CARD_W/2)*state.zoom-scroller.clientWidth/2, top: (pos.y+CARD_H/2)*state.zoom-scroller.clientHeight/2, behavior: 'smooth' });
}

function setZoom(zoom) {
  state.zoom = Math.min(1.5, Math.max(0.45, zoom));
  const stage = $('graph-stage');
  stage.style.transform = `scale(${state.zoom})`;
  const width = parseFloat(stage.style.width), height = parseFloat(stage.style.height);
  $('graph-content').style.width = `${width*state.zoom}px`;
  $('graph-content').style.height = `${height*state.zoom}px`;
  $('zoom-value').textContent = `${Math.round(state.zoom*100)}%`;
}

function fitView() {
  const scroller = $('graph-scroller'), stage = $('graph-stage');
  const width = parseFloat(stage.style.width), height = parseFloat(stage.style.height);
  setZoom(Math.min(1, (scroller.clientWidth-32)/width, (scroller.clientHeight-32)/height));
  scroller.scrollTo(0,0);
}

function showSearch() { $('search-overlay').hidden = false; $('search-input').value = ''; renderSearch(); $('search-input').focus(); }
function hideSearch() { $('search-overlay').hidden = true; }
function renderSearch() {
  const query = $('search-input').value.trim().toLowerCase();
  const results = [];
  for (const c of state.module.computations) for (const n of c.nodes) if (!query || `${n.name} ${n.op} ${c.name}`.toLowerCase().includes(query)) results.push({c,n});
  const target = $('search-results'); target.replaceChildren();
  for (const {c,n} of results.slice(0, 30)) {
    const button = document.createElement('button'); button.type = 'button'; button.className = 'search-result';
    button.innerHTML = `<span class="type-badge ${nodeCategory(n)}">${escapeHtml(n.op)}</span><span><strong>%${escapeHtml(n.name)}</strong><small>%${escapeHtml(c.name)}</small></span><span>↗</span>`;
    button.addEventListener('click', () => { hideSearch(); openComputation(c.name, n.name); centerNode(n.name); });
    target.append(button);
  }
  if (!results.length) target.innerHTML = '<div class="search-empty">No matching nodes</div>';
}

$('back-button').addEventListener('click', () => { const prior = state.history.pop(); if (prior) { state.current = prior.current; state.selected = prior.selected; render(); } });
$('mobile-computations').addEventListener('change', e => e.target.value ? openComputation(e.target.value) : openOverview());
$('close-inspector').addEventListener('click', () => { state.selected = null; updateFocus(); });
$('fit-button').addEventListener('click', fitView);
$('zoom-in').addEventListener('click', () => setZoom(state.zoom+0.15));
$('zoom-out').addEventListener('click', () => setZoom(state.zoom-0.15));
$('graph-scroller').addEventListener('dblclick', e => { if (state.current === null || e.target.closest('.node-card, .overview-card')) return; state.selected = null; updateFocus(); });
let pan = null;
$('graph-scroller').addEventListener('pointerdown', e => {
  if (e.button !== 0 || e.target.closest('.node-card, .overview-card')) return;
  const scroller = $('graph-scroller');
  pan = { x: e.clientX, y: e.clientY, left: scroller.scrollLeft, top: scroller.scrollTop };
  scroller.setPointerCapture(e.pointerId);
  scroller.classList.add('panning');
});
$('graph-scroller').addEventListener('pointermove', e => {
  if (!pan) return;
  const scroller = $('graph-scroller');
  scroller.scrollLeft = pan.left - (e.clientX - pan.x);
  scroller.scrollTop = pan.top - (e.clientY - pan.y);
});
for (const event of ['pointerup', 'pointercancel']) $('graph-scroller').addEventListener(event, () => { pan = null; $('graph-scroller').classList.remove('panning'); });
$('open-button').addEventListener('click', () => $('open-dialog').showModal());
$('dialog-close').addEventListener('click', () => $('open-dialog').close());
$('file-input').addEventListener('change', async e => { const file = e.target.files[0]; if (file) $('hlo-input').value = await file.text(); });
$('load-button').addEventListener('click', () => { try { loadText($('hlo-input').value); $('import-error').textContent = ''; $('open-dialog').close(); fitView(); } catch (error) { $('import-error').textContent = error.message; } });
$('search-button').addEventListener('click', showSearch);
$('search-input').addEventListener('input', renderSearch);
$('search-input').addEventListener('keydown', e => { if (e.key === 'Enter') $('search-results').querySelector('button')?.click(); });
$('search-overlay').addEventListener('click', e => { if (e.target === $('search-overlay')) hideSearch(); });
document.addEventListener('keydown', e => { if (e.key === 'Escape') { hideSearch(); return; } if (e.key === '/' && !['INPUT','TEXTAREA'].includes(document.activeElement.tagName)) { e.preventDefault(); showSearch(); } });

fetch('./sample.hlo').then(r => r.text()).then(loadText).then(() => requestAnimationFrame(fitView)).catch(error => { $('module-name').textContent = 'Could not load sample'; $('view-description').textContent = error.message; });
