// A small, dependency-free parser for the textual HLO shown by XLA.
// It keeps the original instruction line for inspection and only interprets
// the parts needed to draw data dependencies and computation links.
import type { Computation, ComputationLink, HloModule, HloNode, SourceFrame } from './types';
import { memorySpaceLabel, shapeMemorySpace } from './memory-space.ts';

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

// Index of the ")" that closes the "(" at `open`, or -1 when it is never closed.
function closingParen(text: string, open: number): number {
  let depth = 0;
  for (let i = open; i < text.length; i++) {
    if (text[i] === '(') depth++;
    else if (text[i] === ')' && --depth === 0) return i;
  }
  return -1;
}

// The leading array shape of a type, e.g. "f32[8,128]" from "f32[8,128]{1,0:T(8,128)}".
export function leadingShape(type: string): string | null {
  return /^[a-z][\w]*\[[^\]]*\]/.exec(type)?.[0] ?? null;
}

// Top-level element types of a tuple type "(a, b, …)", or null when the type is not a tuple.
export function tupleSlots(type: string): string[] | null {
  if (!type.startsWith('(')) return null;
  const end = closingParen(type, 0);
  return end < 0 ? null : splitTopLevel(type.slice(1, end));
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

function computationCalls(suffix: string): Record<string, string> {
  const calls: Record<string, string> = {};
  for (const piece of splitTopLevel(suffix)) {
    const match = /^\s*([\w-]+)=(.*)$/s.exec(piece);
    if (!match || !CALL_ATTRIBUTES.has(match[1])) continue;

    const attribute = match[1];
    const names = [...match[2].matchAll(/%?([\w.-]+)/g)].map(m => m[1]);
    names.forEach((name, index) => {
      calls[names.length > 1 ? `${attribute}[${index}]` : attribute] = name;
    });
  }
  return calls;
}

// Operand names in an argument list: "%a, %b", "a, b", "/*index=5*/%f", typed "f32[2]{0} %a",
// or XLA dump projections "%t#0" (element 0 of tuple %t).
function operandNames(args: string): string[] {
  if (!args.trim()) return [];
  return splitTopLevel(args).flatMap(piece => {
    const name = /%?([\w.-]+)(?:#\d+)?\s*$/.exec(piece.replace(/\/\*.*?\*\//g, ''))?.[1];
    return name ? [name] : [];
  });
}

// Parses one (possibly multi-line) instruction, or reports why it could not.
function parseInstruction(raw: string, line: number, computation: string, warnings: string[]): HloNode | null {
  const match = raw.match(/^(ROOT\s+)?%?([^\s=]+)\s*=\s*([\s\S]*)$/);
  if (!match) {
    warnings.push(`Line ${line}: unrecognized instruction`);
    return null;
  }
  const [, rootMarker, name, rhs] = match;

  const opMatch = /\s([a-z][a-z0-9-]*)\(/g.exec(rhs);
  if (!opMatch) {
    warnings.push(`Line ${line}: operation not found`);
    return null;
  }
  const op = opMatch[1];
  const argsStart = opMatch.index + opMatch[0].length - 1;
  const argsEnd = closingParen(rhs, argsStart);
  if (argsEnd < 0) {
    warnings.push(`Line ${line}: unclosed ${op} operands`);
    return null;
  }

  const args = rhs.slice(argsStart + 1, argsEnd);
  const suffix = rhs.slice(argsEnd + 1);
  const attribute = (pattern: RegExp) => pattern.exec(suffix)?.[1] || null;
  const controlText = attribute(/\bcontrol-predecessors=\{([^}]*)\}/) ?? '';
  const index = attribute(/\bindex=(\d+)/);

  return {
    id: `${computation}/${name}`,
    name,
    op,
    type: rhs.slice(0, opMatch.index).trim(),
    operands: op === 'constant' || op === 'parameter' ? [] : operandNames(args),
    controlPredecessors: [...controlText.matchAll(/%?([\w.-]+)/g)].map(m => m[1]),
    controlSuccessors: [],
    calls: computationCalls(suffix),
    index: index ? Number(index) : null,
    kind: attribute(/\bkind=([\w]+)/),
    direction: attribute(/\bdirection=([\w]+)/),
    root: !!rootMarker,
    raw,
    line,
    users: []
  };
}

// FileNames / FunctionNames / FileLocations / StackFrames tables printed before the computations.
type SourceTables = Record<'FileNames' | 'FunctionNames' | 'FileLocations' | 'StackFrames', Map<number, string>>;

const field = (text: string, name: string) => Number(new RegExp(`\\b${name}=(\\d+)`).exec(text)?.[1] ?? 0);
const unquote = (text = '') => text.replace(/^"|"$/g, '');

function stackFrames(tables: SourceTables): Map<number, SourceFrame> {
  const frames = new Map<number, SourceFrame>();
  for (const [id, frame] of tables.StackFrames) {
    const location = tables.FileLocations.get(field(frame, 'file_location_id')) || '';
    frames.set(id, {
      file: unquote(tables.FileNames.get(field(location, 'file_name_id'))),
      func: unquote(tables.FunctionNames.get(field(location, 'function_name_id'))),
      line: field(location, 'line'),
      column: field(location, 'column'),
      // XLA prints parent_frame_id as the parent's id + 1, so 1 means "no parent" (checked against a 3-level call chain).
      parent: Math.max(0, field(frame, 'parent_frame_id') - 1)
    });
  }
  return frames;
}

// Fills users / controlSuccessors and warns about names that resolve to nothing.
function linkReferences(module: HloModule) {
  for (const computation of module.computations) {
    for (const node of computation.nodes) {
      for (const operand of node.operands) {
        const source = computation.byName.get(operand);
        if (source) source.users.push(node.name);
        else module.warnings.push(`${computation.name}: %${node.name} references missing %${operand}`);
      }

      for (const predecessor of node.controlPredecessors) {
        const source = computation.byName.get(predecessor);
        if (source) source.controlSuccessors.push(node.name);
        else module.warnings.push(`${computation.name}: %${node.name} references missing control predecessor %${predecessor}`);
      }

      for (const target of Object.values(node.calls)) {
        if (!module.byName.has(target)) module.warnings.push(`${computation.name}: missing computation %${target}`);
      }
    }
  }
}

export function parseHlo(source: string): HloModule {
  const lines = source.replace(/\r\n?/g, '\n').split('\n');
  const module: HloModule = { name: '', computations: [], byName: new Map(), warnings: [], scheduled: false, stackFrames: new Map() };
  const tables: SourceTables = { FileNames: new Map(), FunctionNames: new Map(), FileLocations: new Map(), StackFrames: new Map() };
  let table: Map<number, string> | null = null;
  let current: Computation | null = null;
  // The instruction being read; it may continue over several lines.
  let pending: { text: string; line: number } | null = null;

  // A "{" left open (e.g. in backend_config JSON) means the next lines still belong to the instruction.
  const pendingIsOpen = () => !!pending && openBraceCount(pending.text) > 0;

  const commit = () => {
    if (!pending || !current) return;
    const node = parseInstruction(pending.text, pending.line, current.name, module.warnings);
    pending = null;
    if (!node) return;

    if (current.byName.has(node.name)) module.warnings.push(`Line ${node.line}: duplicate instruction %${node.name}`);
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

    if (line === '}') {
      if (pendingIsOpen()) pending!.text += `\n${line}`;
      else {
        commit();
        current = null;
      }
      continue;
    }

    const isHeader = line.endsWith('{') && !/\s=\s/.test(line.split('(')[0]) && !pendingIsOpen();
    const header = isHeader ? /^(ENTRY\s+)?%?([^\s(={]+)\s*[({]/.exec(line) : null;
    if (header) {
      commit();
      current = { name: header[2], entry: !!header[1], nodes: [], byName: new Map(), header: line };
      module.computations.push(current);
      module.byName.set(current.name, current);
      continue;
    }

    if (!current) {
      if (line in tables) table = tables[line as keyof SourceTables];
      else if (table) {
        const row = /^(\d+)\s+(.*)$/.exec(line);
        if (row) table.set(Number(row[1]), row[2]);
      }
      continue;
    }

    if (/^(?:ROOT\s+)?%?[\w.-]+\s=\s/.test(line) && !pendingIsOpen()) {
      commit();
      pending = { text: line, line: i + 1 };
    } else if (pending) pending.text += `\n${line}`;
    else module.warnings.push(`Line ${i + 1}: unrecognized content in %${current.name}`);
  }
  commit();

  module.stackFrames = stackFrames(tables);
  linkReferences(module);
  if (!module.name) module.name = 'Untitled HLO module';
  return module;
}

// Call stack for a metadata stack_frame_id, innermost frame first.
export function sourceStack(module: HloModule, frameId: number): SourceFrame[] {
  const stack: SourceFrame[] = [];
  let frame = module.stackFrames.get(frameId);
  // The length cap guards against a cyclic parent chain.
  while (frameId && frame && stack.length < 64) {
    stack.push(frame);
    frameId = frame.parent;
    frame = module.stackFrames.get(frameId);
  }
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

// Data and control dependencies of a node, by name.
export const predecessors = (node: HloNode) => [...node.operands, ...node.controlPredecessors];
export const successors = (node: HloNode) => [...node.users, ...node.controlSuccessors];

export function reachable(computation: Computation, startName: string, direction: 'up' | 'down'): Set<string> {
  const found = new Set<string>();
  const queue = [startName];
  while (queue.length) {
    const node = computation.byName.get(queue.shift()!);
    if (!node) continue;

    for (const next of direction === 'up' ? predecessors(node) : successors(node)) {
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

      for (const neighbor of [...predecessors(node), ...successors(node)]) {
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
    for (const neighbor of [...predecessors(node), ...successors(node)]) {
      if (computation.byName.has(neighbor) && !previous.has(neighbor)) {
        previous.set(neighbor, name);
        queue.push(neighbor);
      }
    }
  }
  return null;
}

// Overview edges: one computation calling another. Kept separate from instruction data edges.
export function computationLinks(module: HloModule): ComputationLink[] {
  const links: ComputationLink[] = [];
  for (const computation of module.computations) {
    for (const node of computation.nodes) {
      for (const [role, target] of Object.entries(node.calls)) {
        if (module.byName.has(target)) links.push({ from: computation.name, to: target, role, via: node.name, op: node.op });
      }
    }
  }
  return links;
}

const shapeLocation = (shape: string) => memorySpaceLabel(shapeMemorySpace(shape).space);

export function copyDirection(node: HloNode): string | null {
  if (node.op !== 'copy-start') return null;
  // copy-start returns (destination, source, context).
  const [destination, source] = tupleSlots(node.type)?.map(slot => slot.trim()) ?? [];
  if (!destination || !source) return null;
  return `${shapeLocation(source)} → ${shapeLocation(destination)}`;
}

export function nodeSummary(node: HloNode): string {
  if (node.index !== null) return `tuple slot ${node.index}`;
  if (node.kind) return node.kind;
  if (node.direction) return `direction ${node.direction}`;

  const shape = leadingShape(node.type) || node.type;
  if (node.op === 'constant') {
    const value = node.raw.match(/\bconstant\((.*)\)(?:,|$)/)?.[1] || '';
    return `value ${value.length > 36 ? `${value.slice(0, 33)}…` : value}`;
  }
  if (node.op === 'parameter') {
    const label = /\bop_name="([^"/]+)"/.exec(node.raw)?.[1];
    return label ? `${label} · ${shape}` : shape;
  }

  const arity = tupleSlots(node.type)?.length;
  if (arity === undefined) return shape;
  if (node.op === 'copy-start') return copyDirection(node) || `${arity}-element tuple`;
  if (node.op === 'custom-call') {
    const target = /\bcustom_call_target="([^"]+)"/.exec(node.raw)?.[1];
    return `${target || 'custom-call'} · ${arity} outputs`;
  }
  return node.op === 'while' ? `${arity}-value loop state` : `${arity}-element tuple`;
}

// [role, computation name] for each call target that exists in the module.
export function calledComputations(module: HloModule, node: HloNode) {
  return Object.entries(node.calls).filter(([, name]) => module.byName.has(name));
}
