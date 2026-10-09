import { copyDirection } from './parser.ts';
import type { Computation, HloNode } from './types';

export interface CopyGroup {
  start: HloNode;
  done: HloNode;
  direction: string;
  shape: string;
}
export interface CopyGrouping {
  computation: Computation;
  groups: Map<string, CopyGroup>;
  aliases: Map<string, string>;
}

// This is a display projection. The original instructions remain available for
// search, source inspection, and switching grouping off.
export function groupCopyPairs(computation: Computation, enabled: boolean): CopyGrouping {
  const groups = new Map<string, CopyGroup>();
  const aliases = new Map<string, string>();
  if (!enabled) return { computation, groups, aliases };
  const dones = new Map<string, HloNode[]>();
  for (const node of computation.nodes) {
    if (node.op !== 'copy-done' || node.operands.length !== 1) continue;
    const start = computation.byName.get(node.operands[0]);
    if (start?.op !== 'copy-start') continue;
    const matches = dones.get(start.name) ?? [];
    matches.push(node);
    dones.set(start.name, matches);
  }
  for (const [name, matches] of dones) {
    if (matches.length !== 1) continue;
    const start = computation.byName.get(name)!, done = matches[0];
    const direction = copyDirection(start) ?? 'Unknown transfer';
    groups.set(done.name, { start, done, direction, shape: done.type.match(/^[a-z][\w]*\[[^\]]*\]/)?.[0] ?? done.type });
    aliases.set(start.name, done.name);
    aliases.set(done.name, done.name);
  }
  if (!groups.size) return { computation, groups, aliases };
  const alias = (name: string) => aliases.get(name) ?? name;
  const nodes = computation.nodes.filter(node => !aliases.has(node.name) || groups.has(node.name)).map(node => {
    const group = groups.get(node.name);
    const members = group ? [group.start, group.done] : [node];
    const remap = (names: string[]) => names.map(alias).filter(name => !group || name !== node.name);
    return {
      ...node,
      op: group ? 'copy' : node.op,
      root: members.some(member => member.root),
      operands: remap(members.flatMap(member => member.operands)),
      controlPredecessors: [...new Set(remap(members.flatMap(member => member.controlPredecessors)))],
      users: [] as string[],
      controlSuccessors: [] as string[]
    };
  });
  const byName = new Map(nodes.map(node => [node.name, node]));
  for (const node of nodes) {
    for (const name of node.operands) {
      const source = byName.get(name);
      if (source && !source.users.includes(node.name)) source.users.push(node.name);
    }
    for (const name of node.controlPredecessors) byName.get(name)?.controlSuccessors.push(node.name);
  }
  return { computation: { ...computation, nodes, byName }, groups, aliases };
}
