import { computationLinks } from './parser.ts';
import type { Computation, HloModule, HloNode } from './types';
import { memorySpaceLabel } from './memory-space.ts';

export interface MemoryLocation {
  path: string | null;
  space: number;
  label: string;
  explicit: boolean;
}

function tupleItems(type: string): string[] {
  const items: string[] = [];
  let start = 1, parens = 0, brackets = 0, braces = 0;
  for (let index = 1; index < type.length - 1; index++) {
    const char = type[index];
    if (char === '(') parens++;
    else if (char === ')') parens--;
    else if (char === '[') brackets++;
    else if (char === ']') brackets--;
    else if (char === '{') braces++;
    else if (char === '}') braces--;
    else if (char === ',' && !parens && !brackets && !braces) {
      items.push(type.slice(start, index).trim());
      start = index + 1;
    }
  }
  items.push(type.slice(start, -1).trim());
  return items;
}

export function memoryLocations(type: string): MemoryLocation[] {
  const visit = (shape: string, path: string | null): MemoryLocation[] => {
    const trimmed = shape.replace(/\/\*[^*]*\*\//g, '').trim(); // long tuples carry /*index=N*/ comments
    if (trimmed.startsWith('(') && trimmed.endsWith(')')) {
      return tupleItems(trimmed).flatMap((item, index) => visit(item, path === null ? String(index) : `${path}.${index}`));
    }
    if (!/^[a-z][\w]*\[[^\]]*\]/.test(trimmed) || trimmed.startsWith('token[')) return []; // tokens hold no data
    const match = /\bS\((\d+)\)/.exec(trimmed);
    const space = match ? Number(match[1]) : 0;
    return [{
      path, space, explicit: !!match,
      label: memorySpaceLabel(space)
    }];
  };
  return visit(type, null);
}

// Calls whose callee runs inside the caller's kernel or per element, so its values never get buffers of their own:
// fusion bodies, and reducers / comparators / scatter combiners (XLA assigns those thread-local storage).
const INLINE_CALLERS = new Set(['fusion', 'reduce', 'reduce-window', 'scatter', 'sort', 'select-and-scatter', 'map', 'all-reduce', 'reduce-scatter', 'custom-call']);

// Whether a node's result lives in a buffer of its own:
//   buffer       its layout's S(n) is where XLA placed it
//   fusion       computed inside a fusion kernel; never written to memory as such
//   thread-local a value inside a reducer / comparator, held in thread-local storage
//   unassigned   lowered (unscheduled) HLO; memory spaces are only assigned during optimization
export type BufferStatus = 'buffer' | 'fusion' | 'thread-local' | 'unassigned';

export function bufferStatus(module: HloModule, computation: Computation, node: HloNode): BufferStatus {
  if (!module.scheduled) return 'unassigned';
  const callers = computationLinks(module).filter(link => link.to === computation.name && INLINE_CALLERS.has(link.op));
  if (!callers.length) return 'buffer';
  if (callers.some(link => link.op === 'fusion')) return 'fusion';
  return node.op === 'constant' ? 'buffer' : 'thread-local'; // constants get a global buffer even inside reducers
}

// Memory locations to show for a node, or [] when the HLO text cannot tell us one.
// Checked against XLA's buffer assignment for every compiled example (memory-location.test.js).
export function nodeMemoryLocations(module: HloModule, computation: Computation, node: HloNode): MemoryLocation[] {
  return bufferStatus(module, computation, node) === 'buffer' ? memoryLocations(node.type) : [];
}
