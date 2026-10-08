// A small, dependency-free parser for the textual HLO shown by XLA.
// It keeps the original instruction line for inspection and only interprets
// the parts needed to draw data dependencies and computation links.
export function parseHlo(source) {
  const lines = source.replace(/\r\n?/g, '\n').split('\n');
  const module = { name: '', computations: [], byName: new Map(), warnings: [] };
  let current = null;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line || line.startsWith('//')) continue;
    if (line.startsWith('HloModule ')) {
      module.name = line.slice(10).split(',')[0].trim();
      continue;
    }
    if (line === '}') { current = null; continue; }
    if (line.endsWith('{') && !line.includes(' = ')) {
      const match = line.match(/^(ENTRY\s+)?%([^\s(]+)\s*\(/);
      if (match) {
        current = { name: match[2], entry: !!match[1], nodes: [], byName: new Map(), header: line };
        module.computations.push(current);
        module.byName.set(current.name, current);
      }
      continue;
    }
    if (!current || !line.includes(' = ')) continue;
    const match = line.match(/^(ROOT\s+)?%([^\s]+)\s*=\s*(.*)$/);
    if (!match) { module.warnings.push(`Line ${i + 1}: unrecognized instruction`); continue; }
    const rhs = match[3];
    const opMatch = /\s([a-z][a-z0-9-]*)\(/g.exec(rhs);
    if (!opMatch) { module.warnings.push(`Line ${i + 1}: operation not found`); continue; }
    const op = opMatch[1];
    const opStart = opMatch.index + opMatch[0].length - 1;
    let depth = 0, end = opStart;
    for (; end < rhs.length; end++) {
      if (rhs[end] === '(') depth++;
      if (rhs[end] === ')' && --depth === 0) break;
    }
    const args = rhs.slice(opStart + 1, end);
    const operands = [...args.matchAll(/%([\w.-]+)/g)].map(m => m[1]);
    const calls = {};
    for (const ref of rhs.slice(end + 1).matchAll(/\b(calls|body|condition)=%([\w.-]+)/g)) calls[ref[1]] = ref[2];
    const index = /\bindex=(\d+)/.exec(rhs.slice(end + 1));
    const kind = /\bkind=([\w]+)/.exec(rhs.slice(end + 1));
    const direction = /\bdirection=([\w]+)/.exec(rhs.slice(end + 1));
    const node = {
      id: `${current.name}/${match[2]}`, name: match[2], op, type: rhs.slice(0, opMatch.index).trim(),
      operands, calls, index: index ? Number(index[1]) : null,
      kind: kind?.[1] || null, direction: direction?.[1] || null,
      root: !!match[1], raw: line, line: i + 1, users: []
    };
    current.nodes.push(node);
    current.byName.set(node.name, node);
  }
  for (const computation of module.computations) {
    for (const node of computation.nodes) {
      for (const operand of node.operands) {
        const sourceNode = computation.byName.get(operand);
        if (sourceNode) sourceNode.users.push(node.name);
        else module.warnings.push(`${computation.name}: %${node.name} references missing %${operand}`);
      }
      for (const target of Object.values(node.calls)) {
        if (!module.byName.has(target)) module.warnings.push(`${computation.name}: missing computation %${target}`);
      }
    }
  }
  if (!module.name) module.name = 'Untitled HLO module';
  return module;
}

export function nodeCategory(node) {
  if (node.op === 'parameter' || node.op === 'constant') return 'input';
  if (node.op === 'while') return 'control';
  if (node.op === 'fusion') return 'fusion';
  if (node.op.startsWith('copy')) return 'transfer';
  if (node.op === 'tuple' || node.op === 'get-tuple-element') return 'tuple';
  return 'compute';
}

export function reachable(computation, startName, direction) {
  const found = new Set();
  const queue = [startName];
  while (queue.length) {
    const name = queue.shift();
    const node = computation.byName.get(name);
    if (!node) continue;
    for (const next of direction === 'up' ? node.operands : node.users) {
      if (!found.has(next) && next !== startName) { found.add(next); queue.push(next); }
    }
  }
  return found;
}

// Edges in the module overview represent a control/call relationship between
// computations. They are intentionally separate from instruction data edges.
export function computationLinks(module) {
  const links = [];
  for (const computation of module.computations) {
    for (const node of computation.nodes) {
      for (const [role, target] of Object.entries(node.calls)) {
        if (module.byName.has(target)) links.push({
          from: computation.name, to: target, role, via: node.name, op: node.op
        });
      }
    }
  }
  return links;
}

export function tupleArity(type) {
  if (!type.startsWith('(')) return null;
  let parens = 0, brackets = 0, braces = 0, count = 1;
  for (const char of type) {
    if (char === '(') parens++;
    else if (char === ')') {
      parens--;
      if (parens === 0) return count;
    } else if (char === '[') brackets++;
    else if (char === ']') brackets--;
    else if (char === '{') braces++;
    else if (char === '}') braces--;
    else if (char === ',' && parens === 1 && brackets === 0 && braces === 0) count++;
  }
  return null;
}

export function copyDirection(node) {
  if (node.op !== 'copy-start' || !node.type.startsWith('(')) return null;
  const slots = [];
  let start = 1, parens = 1, brackets = 0, braces = 0;
  for (let i = 1; i < node.type.length; i++) {
    const char = node.type[i];
    if (char === '(') parens++;
    else if (char === ')') parens--;
    else if (char === '[') brackets++;
    else if (char === ']') brackets--;
    else if (char === '{') braces++;
    else if (char === '}') braces--;
    if ((char === ',' && parens === 1 && !brackets && !braces) || (char === ')' && parens === 0)) {
      slots.push(node.type.slice(start, i).trim());
      start = i + 1;
    }
    if (parens === 0) break;
  }
  if (slots.length < 2 || !slots[0] || !slots[1]) return null;
  const location = shape => {
    const space = /S\((\d+)\)/.exec(shape)?.[1] ?? '0';
    return space === '0' ? 'HBM' : space === '1' ? 'VMEM' : `S(${space})`;
  };
  // copy-start returns (destination, source, context).
  return `${location(slots[1])} → ${location(slots[0])}`;
}

export function nodeSummary(node) {
  if (node.index !== null) return `tuple slot ${node.index}`;
  if (node.kind) return node.kind;
  if (node.direction) return `direction ${node.direction}`;
  if (node.op === 'constant') return `value ${node.raw.match(/constant\(([^)]*)\)/)?.[1] || ''}`;
  const arity = tupleArity(node.type);
  if (arity !== null) {
    if (node.op === 'copy-start') {
      return copyDirection(node) || `${arity}-element tuple`;
    }
    return node.op === 'while' ? `${arity}-value loop state` : `${arity}-element tuple`;
  }
  return node.type.match(/^[a-z][\w]*\[[^\]]*\]/)?.[0] || node.type;
}
