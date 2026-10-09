import { computationLinks, predecessors } from './parser.ts';
import type { Computation, ComputationLink, HloModule, HloNode } from './types';

export type LayoutDirection = 'horizontal' | 'vertical';
export type LayoutMode = 'auto' | LayoutDirection;

export const MIN_ZOOM = 0.45;
export const MAX_ZOOM = 1.5;
export const ZOOM_STEP = 0.15;
export const clampZoom = (zoom: number) => Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, zoom));

export interface Point {
  x: number;
  y: number;
}

export interface GraphLayout {
  positions: Map<string, Point>;
  width: number;
  height: number;
}

export const NODE_WIDTH = 218;
export const NODE_HEIGHT = 91;
export const OVERVIEW_WIDTH = 212;
export const OVERVIEW_HEIGHT = 116;

const REDUCING_OPS = ['reduce', 'reduce-window', 'all-reduce', 'reduce-scatter', 'scatter'];

const isComparator = (link: ComputationLink) =>
  link.op === 'sort' || link.role === 'comparator' || (link.op === 'custom-call' && link.role.startsWith('called_computations'));

export function computationRole(computation: Computation, links: ComputationLink[]): string {
  if (computation.entry) return 'ENTRY';

  const incoming = links.filter(link => link.to === computation.name);
  const calledAs = (test: (link: ComputationLink) => boolean) => incoming.some(test);

  if (calledAs(link => link.role === 'body')) return 'WHILE BODY';
  if (calledAs(link => link.role === 'condition')) return 'WHILE CONDITION';
  if (calledAs(link => link.op === 'fusion')) return 'FUSION';
  if (calledAs(link => /^(branch_computations|true_computation|false_computation)/.test(link.role))) return 'BRANCH';
  if (calledAs(isComparator)) return 'COMPARATOR';
  if (calledAs(link => link.role === 'to_apply' && REDUCING_OPS.includes(link.op))) return 'REDUCER';
  if (calledAs(link => link.role === 'select')) return 'SELECT';
  if (calledAs(link => link.role === 'scatter')) return 'SCATTER';
  if (calledAs(link => link.op === 'call')) return 'CALLED';
  return 'COMPUTATION';
}

const LINK_LABELS: Record<string, string> = {
  body: 'BODY',
  condition: 'COND',
  to_apply: 'APPLY',
  calls: 'CALL',
  branch_computations: 'BRANCH',
  true_computation: 'TRUE',
  false_computation: 'FALSE',
  select: 'SELECT',
  scatter: 'SCATTER',
  called_computations: 'CALLED'
};

// Short label drawn on an overview edge, e.g. BODY, COND, APPLY, BRANCH 1.
export function linkLabel(role: string): string {
  const branch = /^branch_computations\[(\d+)\]$/.exec(role);
  if (branch) return `BRANCH ${branch[1]}`;
  return LINK_LABELS[role] || role.toUpperCase();
}

// Spacing for placeLayers. "Layer" runs along the layout direction, "slot" across it.
interface LayerSpacing {
  layerPad: number;
  slotPad: number;
  layerGap: number;
  slotGap: number;
}

// Places layers one after another along the direction and centers each layer's
// items across it. Every layer is as deep as its deepest item.
function placeLayers<T extends { name: string }>(
  layers: T[][],
  direction: LayoutDirection,
  spacing: LayerSpacing,
  size: (item: T) => { width: number; height: number }
): GraphLayout {
  const horizontal = direction === 'horizontal';
  const depth = (item: T) => (horizontal ? size(item).width : size(item).height);
  const breadth = (item: T) => (horizontal ? size(item).height : size(item).width);
  const layerBreadth = (layer: T[]) =>
    layer.reduce((total, item) => total + breadth(item), 0) + Math.max(0, layer.length - 1) * spacing.slotGap;
  const maxBreadth = Math.max(0, ...layers.map(layerBreadth));

  const positions = new Map<string, Point>();
  let along = spacing.layerPad;
  for (const layer of layers) {
    let across = spacing.slotPad + (maxBreadth - layerBreadth(layer)) / 2;
    for (const item of layer) {
      positions.set(item.name, horizontal ? { x: along, y: across } : { x: across, y: along });
      across += breadth(item) + spacing.slotGap;
    }
    along += Math.max(...layer.map(depth)) + spacing.layerGap;
  }

  const length = along - (layers.length ? spacing.layerGap : 0) + spacing.layerPad;
  const span = maxBreadth + spacing.slotPad * 2;
  return horizontal ? { positions, width: length, height: span } : { positions, width: span, height: length };
}

// Groups items by level, in ascending level order; items keep their input order within a level.
function groupByLevel<T extends { name: string }>(items: T[], levels: Map<string, number>): T[][] {
  const groups = new Map<number, T[]>();
  for (const item of items) {
    const level = levels.get(item.name)!;
    if (!groups.has(level)) groups.set(level, []);
    groups.get(level)!.push(item);
  }
  return [...groups.keys()].sort((a, b) => a - b).map(level => groups.get(level)!);
}

// Keeps a while loop's body ahead of its condition.
const loopOrder = (part: string | null) => (part === 'body' ? -1 : part === 'condition' ? 1 : 0);

// Breadth-first call depth from the entry and other uncalled computations.
function callLevels(module: HloModule, links: ComputationLink[]): Map<string, number> {
  const incoming = new Map(module.computations.map(c => [c.name, [] as ComputationLink[]]));
  const outgoing = new Map(module.computations.map(c => [c.name, [] as ComputationLink[]]));
  for (const link of links) {
    incoming.get(link.to)?.push(link);
    outgoing.get(link.from)?.push(link);
  }

  const roots = module.computations.filter(c => c.entry || !incoming.get(c.name)?.length);
  const levels = new Map(roots.map(c => [c.name, 0]));
  const queue = roots.map(c => c.name);
  while (queue.length) {
    const name = queue.shift()!;
    const children = [...outgoing.get(name)!].sort((a, b) => loopOrder(a.role) - loopOrder(b.role));
    for (const link of children) {
      if (levels.has(link.to)) continue;
      levels.set(link.to, levels.get(name)! + 1);
      queue.push(link.to);
    }
  }

  // Computations only reachable through a call cycle.
  for (const c of module.computations) if (!levels.has(c.name)) levels.set(c.name, 0);
  return levels;
}

export function layoutOverview(module: HloModule, direction: LayoutDirection = 'horizontal'): GraphLayout {
  const links = computationLinks(module);
  const layers = groupByLevel(module.computations, callLevels(module, links));

  const rolePart = (c: Computation) => {
    const role = computationRole(c, links);
    return role === 'WHILE BODY' ? 'body' : role === 'WHILE CONDITION' ? 'condition' : null;
  };
  for (const layer of layers) layer.sort((a, b) => loopOrder(rolePart(a)) - loopOrder(rolePart(b)));

  const spacing =
    direction === 'horizontal'
      ? { layerPad: 48, slotPad: 60, layerGap: 64, slotGap: 68 }
      : { layerPad: 60, slotPad: 48, layerGap: 68, slotGap: 64 };
  return placeLayers(layers, direction, spacing, () => ({ width: OVERVIEW_WIDTH, height: OVERVIEW_HEIGHT }));
}

// Longest dependency chain to each node; a cycle is cut where it is found.
function dependencyLevels(computation: Computation): Map<string, number> {
  const levels = new Map<string, number>();
  const visiting = new Set<string>();
  const depth = (node: HloNode): number => {
    if (levels.has(node.name)) return levels.get(node.name)!;
    if (visiting.has(node.name)) return 0;

    visiting.add(node.name);
    const inputs = predecessors(node).map(name => computation.byName.get(name));
    const value = inputs.length ? 1 + Math.max(...inputs.map(input => (input ? depth(input) : -1))) : 0;
    visiting.delete(node.name);
    levels.set(node.name, value);
    return value;
  };
  computation.nodes.forEach(depth);
  return levels;
}

export function layoutInstructions(
  computation: Computation,
  nodeHeights: ReadonlyMap<string, number> = new Map(),
  direction: LayoutDirection = 'horizontal'
): GraphLayout {
  const layers = groupByLevel(computation.nodes, dependencyLevels(computation));

  // Order each layer by the average slot of its inputs in the previous layer, to reduce crossings.
  const order = new Map(computation.nodes.map((node, index) => [node.name, index]));
  for (let i = 1; i < layers.length; i++) {
    const slot = new Map(layers[i - 1].map((node, index) => [node.name, index]));
    const averageSlot = (node: HloNode) => {
      const hits = predecessors(node).flatMap(name => slot.get(name) ?? []);
      return hits.length ? hits.reduce((sum, hit) => sum + hit, 0) / hits.length : Infinity;
    };
    layers[i].sort((a, b) => averageSlot(a) - averageSlot(b) || order.get(a.name)! - order.get(b.name)!);
  }

  const spacing = direction === 'horizontal' ? { layerPad: 58, slotPad: 130 } : { layerPad: 130, slotPad: 58 };
  return placeLayers(layers, direction, { ...spacing, layerGap: 94, slotGap: 31 }, node => ({
    width: NODE_WIDTH,
    height: nodeHeights.get(node.name) ?? NODE_HEIGHT
  }));
}
