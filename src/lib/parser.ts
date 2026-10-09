// A small, dependency-free parser for the textual HLO shown by XLA.
// It keeps the original instruction line for inspection and only interprets
// the parts needed to draw data dependencies and computation links.
import type { Computation, ComputationLink, HloModule, HloNode, SourceFrame } from './types';
import { memorySpaceLabel } from './memory-space.ts';

function openBraceCount(text: string): number {
  let depth = 0,
    quoted = false,
    escaped = false;
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

// Split on commas that are not nested inside (), [], {} or a quoted string.
export function splitTopLevel(text: string): string[] {
  const pieces: string[] = [];
  let depth = 0,
    quoted = false,
    escaped = false,
    start = 0;
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (quoted) {
      if (escaped) escaped = false;
      else if (char === '\\') escaped = true;
      else if (char === '"') quoted = false;
    } else if (char === '"') quoted = true;
    else if (char === '(' || char === '[' || char === '{') depth++;
    else if (char === ')' || char === ']' || char === '}') depth--;
    else if (char === ',' && depth === 0) {
      pieces.push(text.slice(start, i));
      start = i + 1;
    }
  }
  pieces.push(text.slice(start));
  return pieces;
}

// Attributes whose value names other computations. Newer XLA/JAX printers omit the % sigil.
const CALL_ATTRIBUTES = new Set([
  'calls',
  'to_apply',
  'body',
  'condition',
  'select',
  'scatter',
  'branch_computations',
  'true_computation',
  'false_computation',
  'called_computations',
  'comparator'
]);

export function computationCalls(suffix: string): Record<string, string> {
  const calls: Record<string, string> = {};
  for (const piece of splitTopLevel(suffix)) {
    const match = /^\s*([\w-]+)=(.*)$/s.exec(piece);
    if (!match || !CALL_ATTRIBUTES.has(match[1])) continue;
    const names = [...match[2].matchAll(/%?([\w.-]+)/g)].map(m => m[1]);
    names.forEach((name, index) => {
      calls[names.length > 1 ? `${match[1]}[${index}]` : match[1]] = name;
    });
  }
  return calls;
}

// Operand names in an argument list: "%a, %b", "a, b", "/*index=5*/%f", typed "f32[2]{0} %a",
// or XLA dump projections "%t#0" (element 0 of tuple %t).
export function operandNames(args: string): string[] {
  if (!args.trim()) return [];
  return splitTopLevel(args).flatMap(piece => {
    const name = /%?([\w.-]+)(?:#\d+)?\s*$/.exec(piece.replace(/\/\*.*?\*\//g, ''))?.[1];
    return name ? [name] : [];
  });
}

const field = (text: string, name: string) => Number(new RegExp(`\\b${name}=(\\d+)`).exec(text)?.[1] ?? 0);
const unquote = (text = '') => text.replace(/^"|"$/g, '');

export function parseHlo(source: string): HloModule {
  const lines = source.replace(/\r\n?/g, '\n').split('\n');
  const module: HloModule = { name: '', computations: [], byName: new Map(), warnings: [], scheduled: false, stackFrames: new Map() };
  // FileNames / FunctionNames / FileLocations / StackFrames tables printed before the computations.
  const tables: Record<string, Map<number, string>> = {
    FileNames: new Map(),
    FunctionNames: new Map(),
    FileLocations: new Map(),
    StackFrames: new Map()
  };
  let table: Map<number, string> | null = null;
  let current: Computation | null = null;
  let pending: { text: string; line: number } | null = null;

  const commit = () => {
    if (!pending || !current) return;
    const { text: raw, line: lineNumber } = pending;
    pending = null;
    const match = raw.match(/^(ROOT\s+)?%?([^\s=]+)\s*=\s*([\s\S]*)$/);
    if (!match) {
      module.warnings.push(`Line ${lineNumber}: unrecognized instruction`);
      return;
    }
    const rhs = match[3];
    const opMatch = /\s([a-z][a-z0-9-]*)\(/g.exec(rhs);
    if (!opMatch) {
      module.warnings.push(`Line ${lineNumber}: operation not found`);
      return;
    }
    const op = opMatch[1];
    const opStart = opMatch.index + opMatch[0].length - 1;
    let depth = 0,
      end = opStart;
    for (; end < rhs.length; end++) {
      if (rhs[end] === '(') depth++;
      if (rhs[end] === ')' && --depth === 0) break;
    }
    if (end >= rhs.length) {
      module.warnings.push(`Line ${lineNumber}: unclosed ${op} operands`);
      return;
    }
    const args = rhs.slice(opStart + 1, end);
    const operands = op === 'constant' || op === 'parameter' ? [] : operandNames(args);
    const suffix = rhs.slice(end + 1);
    const calls = computationCalls(suffix);
    const controlText = /\bcontrol-predecessors=\{([^}]*)\}/.exec(suffix)?.[1] || '';
    const controlPredecessors = [...controlText.matchAll(/%?([\w.-]+)/g)].map(m => m[1]);
    const index = /\bindex=(\d+)/.exec(suffix);
    const kind = /\bkind=([\w]+)/.exec(suffix);
    const direction = /\bdirection=([\w]+)/.exec(suffix);
    const node: HloNode = {
      id: `${current.name}/${match[2]}`,
      name: match[2],
      op,
      type: rhs.slice(0, opMatch.index).trim(),
      operands,
      controlPredecessors,
      controlSuccessors: [],
      calls,
      index: index ? Number(index[1]) : null,
      kind: kind?.[1] || null,
      direction: direction?.[1] || null,
      root: !!match[1],
      raw,
      line: lineNumber,
      users: []
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
      module.scheduled = /\bis_scheduled=true\b/.test(line);
      continue;
    }
    if (line === '}' && pending && openBraceCount(pending.text) > 0) {
      pending.text += `\n${line}`;
      continue;
    }
    if (line === '}') {
      commit();
      current = null;
      continue;
    }
    const header =
      line.endsWith('{') && !/\s=\s/.test(line.split('(')[0]) && !(pending && openBraceCount(pending.text) > 0)
        ? /^(ENTRY\s+)?%?([^\s(={]+)\s*[({]/.exec(line)
        : null;
    if (header) {
      commit();
      const match = header;
      if (match) {
        current = { name: match[2], entry: !!match[1], nodes: [], byName: new Map(), header: line };
        module.computations.push(current);
        module.byName.set(current.name, current);
      }
      continue;
    }
    if (!current) {
      if (line in tables) table = tables[line];
      else if (table) {
        const row = /^(\d+)\s+(.*)$/.exec(line);
        if (row) table.set(Number(row[1]), row[2]);
      }
      continue;
    }
    if (/^(?:ROOT\s+)?%?[\w.-]+\s=\s/.test(line) && !(pending && openBraceCount(pending.text) > 0)) {
      commit();
      pending = { text: line, line: i + 1 };
    } else if (pending) pending.text += `\n${line}`;
    else module.warnings.push(`Line ${i + 1}: unrecognized content in %${current.name}`);
  }
  commit();
  for (const [id, frame] of tables.StackFrames) {
    const location = tables.FileLocations.get(field(frame, 'file_location_id')) || '';
    module.stackFrames.set(id, {
      file: unquote(tables.FileNames.get(field(location, 'file_name_id'))),
      func: unquote(tables.FunctionNames.get(field(location, 'function_name_id'))),
      line: field(location, 'line'),
      column: field(location, 'column'),
      // XLA prints parent_frame_id as the parent's id + 1, so 1 means "no parent" (checked against a 3-level call chain).
      parent: Math.max(0, field(frame, 'parent_frame_id') - 1)
    });
  }
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

// Call stack for a metadata stack_frame_id, innermost frame first.
export function sourceStack(module: HloModule, frameId: number): SourceFrame[] {
  const stack: SourceFrame[] = [];
  for (let id = frameId; id && module.stackFrames.has(id) && stack.length < 64; id = module.stackFrames.get(id)!.parent)
    stack.push(module.stackFrames.get(id)!);
  return stack;
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
      if (!found.has(next) && next !== startName) {
        found.add(next);
        queue.push(next);
      }
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
        if (!seen.has(neighbor)) {
          seen.add(neighbor);
          next.push(neighbor);
        }
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
      if (computation.byName.has(neighbor) && !previous.has(neighbor)) {
        previous.set(neighbor, name);
        queue.push(neighbor);
      }
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
        if (module.byName.has(target))
          links.push({
            from: computation.name,
            to: target,
            role,
            via: node.name,
            op: node.op
          });
      }
    }
  }
  return links;
}

export function tupleArity(type: string): number | null {
  if (!type.startsWith('(')) return null;
  let parens = 0,
    brackets = 0,
    braces = 0,
    count = 1;
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

const shapeLocation = (shape: string) => memorySpaceLabel(Number(/S\((\d+)\)/.exec(shape)?.[1] ?? '0'));

export function copyDirection(node: HloNode): string | null {
  if (node.op !== 'copy-start' || !node.type.startsWith('(')) return null;
  const slots: string[] = [];
  let start = 1,
    parens = 1,
    brackets = 0,
    braces = 0;
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
  // copy-start returns (destination, source, context).
  return `${shapeLocation(slots[1])} → ${shapeLocation(slots[0])}`;
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
