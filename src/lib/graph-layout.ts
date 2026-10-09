import { computationLinks } from './parser.ts';
import type { Computation, ComputationLink, HloModule, HloNode } from './types';

export type LayoutDirection = 'horizontal' | 'vertical';
export type LayoutMode = 'auto' | LayoutDirection;

export interface Point { x: number; y: number }
export interface GraphLayout { positions: Map<string, Point>; width: number; height: number }

export const NODE_WIDTH = 218;
export const NODE_HEIGHT = 91;
export const OVERVIEW_WIDTH = 212;
export const OVERVIEW_HEIGHT = 116;

export function computationRole(computation: Computation, links: ComputationLink[]): string {
  if (computation.entry) return 'ENTRY';
  const incoming = links.filter(link => link.to === computation.name);
  if (incoming.some(link => link.role === 'body')) return 'WHILE BODY';
  if (incoming.some(link => link.role === 'condition')) return 'WHILE CONDITION';
  if (incoming.some(link => link.op === 'fusion')) return 'FUSION';
  if (incoming.some(link => /^(branch_computations|true_computation|false_computation)/.test(link.role))) return 'BRANCH';
  if (incoming.some(link => link.op === 'sort' || link.role === 'comparator' || (link.op === 'custom-call' && link.role.startsWith('called_computations')))) return 'COMPARATOR';
  if (incoming.some(link => link.role === 'to_apply' && ['reduce', 'reduce-window', 'all-reduce', 'reduce-scatter', 'scatter'].includes(link.op))) return 'REDUCER';
  if (incoming.some(link => link.role === 'select')) return 'SELECT';
  if (incoming.some(link => link.role === 'scatter')) return 'SCATTER';
  if (incoming.some(link => link.op === 'call')) return 'CALLED';
  return 'COMPUTATION';
}

// Short label drawn on an overview edge, e.g. BODY, COND, APPLY, BRANCH 1.
export function linkLabel(role: string): string {
  const branch = /^branch_computations\[(\d+)\]$/.exec(role);
  if (branch) return `BRANCH ${branch[1]}`;
  return ({ body: 'BODY', condition: 'COND', to_apply: 'APPLY', calls: 'CALL', branch_computations: 'BRANCH', true_computation: 'TRUE',
    false_computation: 'FALSE', select: 'SELECT', scatter: 'SCATTER', called_computations: 'CALLED' } as Record<string, string>)[role] || role.toUpperCase();
}

export function layoutOverview(module: HloModule, direction: LayoutDirection = 'horizontal'): GraphLayout {
  const links = computationLinks(module);
  const incoming = new Map(module.computations.map(c => [c.name, [] as ComputationLink[]]));
  const outgoing = new Map(module.computations.map(c => [c.name, [] as ComputationLink[]]));
  for (const link of links) {
    incoming.get(link.to)?.push(link);
    outgoing.get(link.from)?.push(link);
  }
  const roots = module.computations.filter(c => c.entry || !incoming.get(c.name)?.length);
  const levels = new Map<string, number>();
  const queue = [...roots];
  roots.forEach(c => levels.set(c.name, 0));
  while (queue.length) {
    const current = queue.shift()!;
    const children = [...outgoing.get(current.name)!].sort((a, b) =>
      (a.role === 'body' ? -1 : a.role === 'condition' ? 1 : 0) -
      (b.role === 'body' ? -1 : b.role === 'condition' ? 1 : 0));
    for (const link of children) {
      if (levels.has(link.to)) continue;
      levels.set(link.to, levels.get(current.name)! + 1);
      queue.push(module.byName.get(link.to)!);
    }
  }
  for (const c of module.computations) if (!levels.has(c.name)) levels.set(c.name, 0);
  const columns = new Map<number, Computation[]>();
  for (const c of module.computations) {
    const level = levels.get(c.name)!;
    if (!columns.has(level)) columns.set(level, []);
    columns.get(level)!.push(c);
  }
  for (const group of columns.values()) group.sort((a, b) => {
    const priority = (c: Computation) => computationRole(c, links) === 'WHILE BODY' ? -1 : computationRole(c, links) === 'WHILE CONDITION' ? 1 : 0;
    return priority(a) - priority(b) || module.computations.indexOf(a) - module.computations.indexOf(b);
  });
  const gapX = 64, gapY = 68, padX = 48, padY = 60;
  const maxRows = Math.max(1, ...[...columns.values()].map(group => group.length));
  const width = padX * 2 + (Math.max(0, ...columns.keys()) + 1) * OVERVIEW_WIDTH + Math.max(0, columns.size - 1) * gapX;
  const height = padY * 2 + maxRows * OVERVIEW_HEIGHT + (maxRows - 1) * gapY;
  const positions = new Map<string, Point>();
  for (const [level, group] of columns) {
    const offset = (maxRows - group.length) * (OVERVIEW_HEIGHT + gapY) / 2;
    group.forEach((c, row) => positions.set(c.name, {
      x: padX + level * (OVERVIEW_WIDTH + gapX),
      y: padY + offset + row * (OVERVIEW_HEIGHT + gapY)
    }));
  }
  if (direction === 'vertical') {
    const verticalPositions = new Map<string, Point>();
    for (const [level, group] of columns) {
      const offset = (maxRows - group.length) * (OVERVIEW_WIDTH + gapX) / 2;
      group.forEach((c, index) => verticalPositions.set(c.name, {
        x: padX + offset + index * (OVERVIEW_WIDTH + gapX),
        y: padY + level * (OVERVIEW_HEIGHT + gapY)
      }));
    }
    return { positions: verticalPositions,
      width: padX * 2 + maxRows * OVERVIEW_WIDTH + (maxRows - 1) * gapX,
      height: padY * 2 + Math.max(1, columns.size) * OVERVIEW_HEIGHT + Math.max(0, columns.size - 1) * gapY };
  }
  return { positions, width: Number.isFinite(width) ? width : padX * 2, height };
}

export function layoutInstructions(computation: Computation, nodeHeights: ReadonlyMap<string, number> = new Map(), direction: LayoutDirection = 'horizontal'): GraphLayout {
  const levels = new Map<string, number>();
  const visiting = new Set<string>();
  const depth = (node: HloNode): number => {
    if (levels.has(node.name)) return levels.get(node.name)!;
    if (visiting.has(node.name)) return 0;
    visiting.add(node.name);
    const predecessors = [...node.operands, ...node.controlPredecessors];
    const value = predecessors.length ? 1 + Math.max(...predecessors.map(name => {
      const input = computation.byName.get(name);
      return input ? depth(input) : -1;
    })) : 0;
    visiting.delete(node.name);
    levels.set(node.name, value);
    return value;
  };
  computation.nodes.forEach(depth);
  const columns = new Map<number, HloNode[]>();
  for (const node of computation.nodes) {
    const level = levels.get(node.name)!;
    if (!columns.has(level)) columns.set(level, []);
    columns.get(level)!.push(node);
  }
  for (const level of [...columns.keys()].sort((a, b) => a - b)) {
    if (level === 0) continue;
    const previous = columns.get(level - 1) || [];
    const rank = new Map(previous.map((n, i) => [n.name, i]));
    const average = (n: HloNode) => {
      const hits = [...n.operands, ...n.controlPredecessors].map(x => rank.get(x)).filter((x): x is number => x !== undefined);
      return hits.length ? hits.reduce((x, y) => x + y, 0) / hits.length : Infinity;
    };
    columns.get(level)!.sort((a, b) => average(a) - average(b) || computation.nodes.indexOf(a) - computation.nodes.indexOf(b));
  }
  const gapX = 94, gapY = 31, padX = 58, padY = 130;
  if (direction === 'vertical') {
    const positions = new Map<string, Point>();
    const maxRows = Math.max(1, ...[...columns.values()].map(group => group.length));
    let y = padY;
    for (const level of [...columns.keys()].sort((a, b) => a - b)) {
      const group = columns.get(level)!;
      const offset = (maxRows - group.length) * (NODE_WIDTH + gapY) / 2;
      group.forEach((node, index) => positions.set(node.name, {
        x: padX + offset + index * (NODE_WIDTH + gapY), y
      }));
      y += Math.max(...group.map(node => nodeHeights.get(node.name) ?? NODE_HEIGHT)) + gapX;
    }
    return { positions, width: padX * 2 + maxRows * NODE_WIDTH + (maxRows - 1) * gapY,
      height: y - (columns.size ? gapX : 0) + padY };
  }

  const columnHeight = (group: HloNode[]) => group.reduce((height, node) => height + (nodeHeights.get(node.name) ?? NODE_HEIGHT), 0) + Math.max(0, group.length - 1) * gapY;
  const maxColumnHeight = Math.max(0, ...[...columns.values()].map(columnHeight));
  const positions = new Map<string, Point>();
  for (const [level, group] of columns) {
    let y = padY + (maxColumnHeight - columnHeight(group)) / 2;
    for (const node of group) {
      positions.set(node.name, { x: padX + level * (NODE_WIDTH + gapX), y });
      y += (nodeHeights.get(node.name) ?? NODE_HEIGHT) + gapY;
    }
  }
  return {
    positions,
    width: padX * 2 + (Math.max(0, ...columns.keys()) + 1) * NODE_WIDTH + Math.max(0, columns.size - 1) * gapX,
    height: padY * 2 + maxColumnHeight
  };
}
