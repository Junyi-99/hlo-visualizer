// A small, dependency-free parser for the textual HLO shown by XLA.
// It keeps the original instruction line for inspection and only interprets
// the parts needed to draw data dependencies and computation links.
import type { Computation, ComputationLink, HloModule, HloNode } from './types';

function openBraceCount(text: string): number {
  let depth = 0, quoted = false, escaped = false;
  for (const char of text) {
    if (quoted) {
      if (escaped) escaped = false;
      else if (char === '\\') escaped = true;
      else if (char === '"') quoted = false;
    } else if (char === '"') quoted = true;
    else if (char === '{') depth++;
    else if (char === '}') depth--;
  }
  return depth;
}

export function parseHlo(source: string): HloModule {
  const lines = source.replace(/\r\n?/g, '\n').split('\n');
  const module: HloModule = { name: '', computations: [], byName: new Map(), warnings: [] };
  let current: Computation | null = null;
  let pending: { text: string; line: number } | null = null;

  const commit = () => {
    if (!pending || !current) return;
    const { text: raw, line: lineNumber } = pending;
    pending = null;
    const match = raw.match(/^(ROOT\s+)?%([^\s]+)\s*=\s*([\s\S]*)$/);
    if (!match) { module.warnings.push(`Line ${lineNumber}: unrecognized instruction`); return; }
    const rhs = match[3];
    const opMatch = /\s([a-z][a-z0-9-]*)\(/g.exec(rhs);
    if (!opMatch) { module.warnings.push(`Line ${lineNumber}: operation not found`); return; }
    const op = opMatch[1];
    const opStart = opMatch.index + opMatch[0].length - 1;
    let depth = 0, end = opStart;
    for (; end < rhs.length; end++) {
      if (rhs[end] === '(') depth++;
      if (rhs[end] === ')' && --depth === 0) break;
    }
    if (end >= rhs.length) { module.warnings.push(`Line ${lineNumber}: unclosed ${op} operands`); return; }
    const args = rhs.slice(opStart + 1, end);
    const operands = [...args.matchAll(/%([\w.-]+)/g)].map(m => m[1]);
    const suffix = rhs.slice(end + 1);
    const calls: Record<string, string> = {};
    for (const ref of suffix.matchAll(/\b(calls|body|condition)=%([\w.-]+)/g)) calls[ref[1]] = ref[2];
    const controlText = /\bcontrol-predecessors=\{([^}]*)\}/.exec(suffix)?.[1] || '';
    const controlPredecessors = [...controlText.matchAll(/%([\w.-]+)/g)].map(m => m[1]);
    const index = /\bindex=(\d+)/.exec(suffix);
    const kind = /\bkind=([\w]+)/.exec(suffix);
    const direction = /\bdirection=([\w]+)/.exec(suffix);
    const node: HloNode = {
      id: `${current.name}/${match[2]}`, name: match[2], op, type: rhs.slice(0, opMatch.index).trim(),
      operands, controlPredecessors, controlSuccessors: [], calls,
      index: index ? Number(index[1]) : null,
      kind: kind?.[1] || null, direction: direction?.[1] || null,
      root: !!match[1], raw, line: lineNumber, users: []
    };
    if (current.byName.has(node.name)) module.warnings.push(`Line ${lineNumber}: duplicate instruction %${node.name}`);
    current.nodes.push(node);
    current.byName.set(node.name, node);
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line || line.startsWith('//')) continue;
    if (line.startsWith('HloModule ')) {
      module.name = line.slice(10).split(',')[0].trim();
      continue;
    }
    if (line === '}' && pending && openBraceCount(pending.text) > 0) {
      pending.text += `\n${line}`;
      continue;
    }
    if (line === '}') { commit(); current = null; continue; }
    if (line.endsWith('{') && /^(ENTRY\s+)?%[^\s(]+\s*\(/.test(line)) {
      commit();
      const match = line.match(/^(ENTRY\s+)?%([^\s(]+)\s*\(/);
      if (match) {
        current = { name: match[2], entry: !!match[1], nodes: [], byName: new Map(), header: line };
        module.computations.push(current);
        module.byName.set(current.name, current);
      }
      continue;
    }
    if (!current) continue;
    if (/^(?:ROOT\s+)?%[^\s]+\s*=/.test(line)) {
      commit();
      pending = { text: line, line: i + 1 };
    } else if (pending) pending.text += `\n${line}`;
    else module.warnings.push(`Line ${i + 1}: unrecognized content in %${current.name}`);
  }
  commit();
  for (const computation of module.computations) {
    for (const node of computation.nodes) {
      for (const operand of node.operands) {
        const sourceNode = computation.byName.get(operand);
        if (sourceNode) sourceNode.users.push(node.name);
        else module.warnings.push(`${computation.name}: %${node.name} references missing %${operand}`);
      }
      for (const predecessor of node.controlPredecessors) {
        const sourceNode = computation.byName.get(predecessor);
        if (sourceNode) sourceNode.controlSuccessors.push(node.name);
        else module.warnings.push(`${computation.name}: %${node.name} references missing control predecessor %${predecessor}`);
      }
      for (const target of Object.values(node.calls)) {
        if (!module.byName.has(target)) module.warnings.push(`${computation.name}: missing computation %${target}`);
      }
    }
  }
  if (!module.name) module.name = 'Untitled HLO module';
  return module;
}

export function nodeCategory(node: HloNode): string {
  if (node.op === 'parameter' || node.op === 'constant') return 'input';
  if (node.op === 'while') return 'control';
  if (node.op === 'fusion') return 'fusion';
  if (node.op.startsWith('copy')) return 'transfer';
  if (node.op === 'tuple' || node.op === 'get-tuple-element') return 'tuple';
  return 'compute';
}

export function reachable(computation: Computation, startName: string, direction: 'up' | 'down'): Set<string> {
  const found = new Set<string>();
  const queue = [startName];
  while (queue.length) {
    const name = queue.shift()!;
    const node = computation.byName.get(name);
    if (!node) continue;
    for (const next of direction === 'up' ? [...node.operands, ...node.controlPredecessors] : [...node.users, ...node.controlSuccessors]) {
      if (!found.has(next) && next !== startName) { found.add(next); queue.push(next); }
    }
  }
  return found;
}

export function dependencyNeighborhood(computation: Computation, startName: string, radius: number): Set<string> {
  const seen = new Set([startName]);
  let frontier = [startName];
  for (let distance = 0; distance < radius; distance++) {
    const next: string[] = [];
    for (const name of frontier) {
      const node = computation.byName.get(name);
      if (!node) continue;
      for (const neighbor of [...node.operands, ...node.controlPredecessors, ...node.users, ...node.controlSuccessors]) {
        if (!seen.has(neighbor)) { seen.add(neighbor); next.push(neighbor); }
      }
    }
    frontier = next;
  }
  return seen;
}

export function shortestDependencyPath(computation: Computation, from: string, to: string): string[] | null {
  if (!computation.byName.has(from) || !computation.byName.has(to)) return null;
  const previous = new Map<string, string | null>([[from, null]]);
  const queue = [from];
  for (let index = 0; index < queue.length; index++) {
    const name = queue[index];
    if (name === to) {
      const path: string[] = [];
      for (let cursor: string | null = to; cursor !== null; cursor = previous.get(cursor) ?? null) path.unshift(cursor);
      return path;
    }
    const node = computation.byName.get(name)!;
    for (const neighbor of [...node.operands, ...node.controlPredecessors, ...node.users, ...node.controlSuccessors]) {
      if (computation.byName.has(neighbor) && !previous.has(neighbor)) { previous.set(neighbor, name); queue.push(neighbor); }
    }
  }
  return null;
}

// Edges in the module overview represent a control/call relationship between
// computations. They are intentionally separate from instruction data edges.
export function computationLinks(module: HloModule): ComputationLink[] {
  const links: ComputationLink[] = [];
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

export function tupleArity(type: string): number | null {
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

export function copyDirection(node: HloNode): string | null {
  if (node.op !== 'copy-start' || !node.type.startsWith('(')) return null;
  const slots: string[] = [];
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
  const location = (shape: string) => {
    const space = /S\((\d+)\)/.exec(shape)?.[1] ?? '0';
    return space === '0' ? 'HBM' : space === '1' ? 'VMEM' : `S(${space})`;
  };
  // copy-start returns (destination, source, context).
  return `${location(slots[1])} → ${location(slots[0])}`;
}

export function nodeSummary(node: HloNode): string {
  if (node.index !== null) return `tuple slot ${node.index}`;
  if (node.kind) return node.kind;
  if (node.direction) return `direction ${node.direction}`;
  if (node.op === 'constant') {
    const value = node.raw.match(/\bconstant\((.*)\)(?:,|$)/)?.[1] || '';
    return `value ${value.length > 36 ? `${value.slice(0, 33)}…` : value}`;
  }
  if (node.op === 'parameter') {
    const label = /\bop_name="([^"/]+)"/.exec(node.raw)?.[1];
    const shape = node.type.match(/^[a-z][\w]*\[[^\]]*\]/)?.[0] || node.type;
    return label ? `${label} · ${shape}` : shape;
  }
  const arity = tupleArity(node.type);
  if (arity !== null) {
    if (node.op === 'copy-start') {
      return copyDirection(node) || `${arity}-element tuple`;
    }
    if (node.op === 'custom-call') {
      const target = /\bcustom_call_target="([^"]+)"/.exec(node.raw)?.[1];
      return `${target || 'custom-call'} · ${arity} outputs`;
    }
    return node.op === 'while' ? `${arity}-value loop state` : `${arity}-element tuple`;
  }
  return node.type.match(/^[a-z][\w]*\[[^\]]*\]/)?.[0] || node.type;
}
