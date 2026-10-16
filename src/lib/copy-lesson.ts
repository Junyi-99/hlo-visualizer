import { nodeMemoryLocations, tupleElements } from './memory-location';
import { reachable } from './parser';
import type { Computation, HloModule, HloNode } from './types';

export interface CopyLessonData {
  source: HloNode | null;
  start: HloNode | null;
  dones: HloNode[];
  consumers: HloNode[];
  tuple: string[];
  direction: string | null;
  unrelated: HloNode[];
}

// Resolve against the original computation, even when the graph displays a merged copy.
export function copyLessonData(module: HloModule, computation: Computation, selected: HloNode): CopyLessonData | null {
  const node = computation.byName.get(selected.name) ?? selected;
  if (node.op !== 'copy-start' && node.op !== 'copy-done') return null;
  const candidate = node.op === 'copy-start' ? node : computation.byName.get(node.operands[0]);
  const start = candidate?.op === 'copy-start' ? candidate : null;
  const dones = start ? computation.nodes.filter(item => item.op === 'copy-done' && item.operands[0] === start.name) : [node];
  const doneNames = new Set(dones.map(item => item.name));
  const consumers = computation.nodes.filter(item => item.operands.some(name => doneNames.has(name)));
  const source = start ? (computation.byName.get(start.operands[0]) ?? null) : null;
  const tuple = start?.type.trim().startsWith('(') ? tupleElements(start.type.trim()) : [];
  const locations = start ? nodeMemoryLocations(module, computation, start) : [];
  const destinationSpace = locations.find(location => location.path === '0');
  const sourceSpace = locations.find(location => location.path === '1');
  const direction = destinationSpace && sourceSpace ? `${sourceSpace.label} → ${destinationSpace.label}` : null;
  const related = new Set<string>();
  for (const done of dones) {
    related.add(done.name);
    for (const name of reachable(computation, done.name, 'up')) related.add(name);
    for (const name of reachable(computation, done.name, 'down')) related.add(name);
  }
  const unrelated = dones.length ? computation.nodes.filter(item => !related.has(item.name)) : [];
  return { source, start, dones, consumers, tuple, direction, unrelated };
}

export const copyEdgeKey = (source: string, target: string) => `${source}\u0000${target}`;

export function copyLessonHighlights(data: CopyLessonData | null, aliases: Map<string, string> = new Map()) {
  const nodes = new Set<string>();
  const edges = new Set<string>();
  const alias = (name: string) => aliases.get(name) ?? name;
  const addEdge = (from: string, to: string) => {
    if (alias(from) !== alias(to)) edges.add(copyEdgeKey(alias(from), alias(to)));
  };
  if (data) {
    for (const node of [data.source, data.start, ...data.dones, ...data.consumers]) {
      if (node) nodes.add(alias(node.name));
    }
    if (data.source && data.start) addEdge(data.source.name, data.start.name);
    for (const done of data.dones) {
      if (data.start) addEdge(data.start.name, done.name);
      for (const consumer of data.consumers) {
        if (consumer.operands.includes(done.name)) addEdge(done.name, consumer.name);
      }
    }
  }
  return { nodes, edges };
}
