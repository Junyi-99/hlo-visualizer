import { computationLinks, leadingShape, splitTopLevel } from './parser.ts';
import type { Computation, HloModule, HloNode } from './types';
import { memorySpaceLabel, shapeMemorySpace } from './memory-space.ts';

export interface MemoryLocation {
  path: string | null;
  space: number;
  label: string;
  explicit: boolean;
}

// The untrimmed top-level elements of a tuple type "(a, b, …)".
export function tupleElements(type: string): string[] {
  return splitTopLevel(type.slice(1, -1));
}

function shapeLocations(shape: string, path: string | null): MemoryLocation[] {
  // Long tuples carry /*index=N*/ printer comments.
  const trimmed = shape.replace(/\/\*[^*]*\*\//g, '').trim();

  if (trimmed.startsWith('(') && trimmed.endsWith(')')) {
    return tupleElements(trimmed).flatMap((item, index) => {
      const itemPath = path === null ? String(index) : `${path}.${index}`;
      return shapeLocations(item.trim(), itemPath);
    });
  }

  if (!leadingShape(trimmed) || trimmed.startsWith('token[')) return []; // tokens hold no data

  const { space, explicit } = shapeMemorySpace(trimmed);
  return [{ path, space, explicit, label: memorySpaceLabel(space) }];
}

function memoryLocations(type: string): MemoryLocation[] {
  return shapeLocations(type, null);
}

// Calls whose callee runs inside the caller's kernel or per element, so its values never get buffers of their own:
// fusion bodies, and reducers / comparators / scatter combiners (XLA assigns those thread-local storage).
const INLINE_CALLERS = new Set([
  'fusion',
  'reduce',
  'reduce-window',
  'scatter',
  'sort',
  'select-and-scatter',
  'map',
  'all-reduce',
  'reduce-scatter',
  'custom-call'
]);

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
// Checked against XLA's buffer assignment for every compiled example (examples/tpu-v6e/*.after.memory.json).
export function nodeMemoryLocations(module: HloModule, computation: Computation, node: HloNode): MemoryLocation[] {
  return bufferStatus(module, computation, node) === 'buffer' ? memoryLocations(node.type) : [];
}
