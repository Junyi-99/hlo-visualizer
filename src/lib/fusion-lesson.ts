import type { Computation, HloModule, HloNode } from './types';

export interface FusionInput {
  parameter: HloNode;
  index: number | null;
  operand: HloNode | null;
  operandName: string | null;
}

export interface FusionOutput {
  path: number[];
  node: HloNode | null;
  name: string;
  unexpanded?: boolean;
}

export const FUSION_OUTPUT_PREVIEW_LIMIT = 64;
const OUTPUT_DEPTH_LIMIT = 32;
const OUTPUT_VISIT_LIMIT = 4096;

// HloNode.index represents tuple_index, not the position of a parameter.
export function fusionParameterIndex(node: HloNode): number | null {
  if (node.op !== 'parameter') return null;
  const match = /\bparameter\s*\(\s*(\d+)\s*\)/.exec(node.raw);
  if (!match) return null;
  const value = Number(match[1]);
  return Number.isSafeInteger(value) ? value : null;
}

export function fusionLessonData(module: HloModule, caller: Computation, node: HloNode) {
  if (node.op !== 'fusion') return null;
  const name = node.calls.calls ?? Object.values(node.calls)[0];
  const body = name ? module.byName.get(name) : undefined;
  if (!body) return { name, body: null, inputs: [], outputs: [], outputsTruncated: false, internal: [], operations: [], root: null };

  const inputs: FusionInput[] = body.nodes
    .filter(item => item.op === 'parameter')
    .map(parameter => {
      const index = fusionParameterIndex(parameter);
      const operandName = index === null ? null : (node.operands[index] ?? null);
      return { parameter, index, operandName, operand: operandName ? (caller.byName.get(operandName) ?? null) : null };
    })
    .sort((a, b) => (a.index ?? Infinity) - (b.index ?? Infinity));

  const root = body.nodes.find(item => item.root) ?? null;
  const returned = new Set<string>();
  // Classify the full tuple DAG once, independently of the bounded path preview.
  // Mark nodes when enqueuing so shared tuple children and cycles stay linear.
  const pending = root ? [root.name] : [];
  if (root) returned.add(root.name);
  while (pending.length) {
    const item = body.byName.get(pending.pop()!);
    if (item?.op !== 'tuple') continue;
    for (const operand of item.operands) {
      if (returned.has(operand)) continue;
      returned.add(operand);
      pending.push(operand);
    }
  }

  const outputs: FusionOutput[] = [];
  let outputsTruncated = false;
  let visits = 0;
  const visitOutput = (outputName: string, path: number[], ancestors: Set<string>) => {
    if (outputs.length >= FUSION_OUTPUT_PREVIEW_LIMIT || visits >= OUTPUT_VISIT_LIMIT) {
      outputsTruncated = true;
      return;
    }
    visits++;
    const item = body.byName.get(outputName) ?? null;
    if (item?.op === 'tuple') {
      if (ancestors.has(outputName) || path.length >= OUTPUT_DEPTH_LIMIT) {
        outputsTruncated = true;
        outputs.push({ path, node: item, name: outputName, unexpanded: true });
        return;
      }
      const next = new Set(ancestors).add(outputName);
      for (let index = 0; index < item.operands.length; index++) {
        if (outputs.length >= FUSION_OUTPUT_PREVIEW_LIMIT || visits >= OUTPUT_VISIT_LIMIT) {
          outputsTruncated = true;
          break;
        }
        visitOutput(item.operands[index], [...path, index], next);
      }
    } else outputs.push({ path, node: item, name: outputName });
  };
  if (root) visitOutput(root.name, [], new Set());
  const internal = body.nodes.filter(item => item.op !== 'parameter' && !returned.has(item.name));
  const counts = new Map<string, number>();
  for (const item of body.nodes) {
    if (item.op !== 'parameter') counts.set(item.op, (counts.get(item.op) ?? 0) + 1);
  }
  const operations = [...counts].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  return { name, body, inputs, outputs, outputsTruncated, internal, operations, root };
}
