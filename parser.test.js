import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parseHlo, reachable, computationLinks, nodeSummary, dependencyNeighborhood, shortestDependencyPath, sourceStack } from './src/lib/parser.ts';

test('sample loop dependencies and computation links', () => {
  const module = parseHlo(readFileSync(new URL('./sample.hlo', import.meta.url), 'utf8'));
  assert.equal(module.computations.length, 6);
  assert.deepEqual(module.warnings, []);
  const entry = module.byName.get('main.3');
  assert.equal(entry.nodes.length, 14);
  assert.deepEqual(entry.byName.get('while').operands, ['tuple.14']);
  assert.deepEqual(entry.byName.get('while').calls, {
    condition: 'wide.region_1.2', body: 'wide.region_0.1.sunk'
  });
  assert.equal(reachable(entry, 'while', 'up').size, 10);
  assert.equal(reachable(entry, 'while', 'down').size, 3);
  assert.deepEqual(module.byName.get('wide.region_0.1.sunk').byName.get('fusion.8').operands,
    ['copy.11', 'get-tuple-element.41']);
  assert.deepEqual(computationLinks(module).map(({from,to,role,via}) => [from,to,role,via]), [
    ['fused_computation.clone.1', 'bitcast_fusion.clone.1', 'calls', 'fusion.6'],
    ['fused_computation.clone.1', 'bitcast_fusion.1.clone.1', 'calls', 'fusion.7'],
    ['wide.region_0.1.sunk', 'fused_computation.clone.1', 'calls', 'fusion.8'],
    ['main.3', 'wide.region_1.2', 'condition', 'while'],
    ['main.3', 'wide.region_0.1.sunk', 'body', 'while']
  ]);
  assert.equal(nodeSummary(entry.byName.get('tuple.14')), '4-element tuple');
  assert.equal(nodeSummary(entry.byName.get('copy-start')), 'VMEM → HBM');
  assert.equal(nodeSummary(entry.byName.get('copy-start.1')), 'HBM → VMEM');
  assert.equal(nodeSummary(entry.byName.get('copy-start.2')), 'HBM → VMEM');
  assert.equal(nodeSummary(entry.byName.get('while')), '4-value loop state');
});

test('backend configuration JSON does not create false operands', () => {
  const source = `HloModule x
ENTRY %main (a: s32[]) -> s32[] {
  %a = s32[] parameter(0)
  ROOT %copy = s32[] copy(%a), backend_config={"aliasing_operands":{"lists":[{"indices":["0","2"]}]}}
}`;
  const module = parseHlo(source);
  assert.deepEqual(module.warnings, []);
  assert.deepEqual(module.computations[0].byName.get('copy').operands, ['a']);
});

test('control dependencies are distinct from data inputs and participate in graph traversal', () => {
  const source = `HloModule controls
ENTRY %main (a: s32[]) -> s32[] {
  %a = s32[] parameter(0)
  %gate = s32[] constant(1)
  %middle = s32[] copy(%a), control-predecessors={%gate}
  ROOT %out = s32[] add(%middle, %gate)
}`;
  const module = parseHlo(source);
  const computation = module.byName.get('main');
  assert.deepEqual(module.warnings, []);
  assert.deepEqual(computation.byName.get('middle').operands, ['a']);
  assert.deepEqual(computation.byName.get('middle').controlPredecessors, ['gate']);
  assert.deepEqual(computation.byName.get('gate').controlSuccessors, ['middle']);
  assert.deepEqual([...dependencyNeighborhood(computation, 'a', 1)], ['a', 'middle']);
  assert.deepEqual(shortestDependencyPath(computation, 'a', 'gate'), ['a', 'middle', 'gate']);
  assert.equal(reachable(computation, 'gate', 'down').has('middle'), true);
});

test('multiline attributes retain source and missing control references produce diagnostics', () => {
  const source = `HloModule multiline
ENTRY %main (a: s32[]) -> s32[] {
  %a = s32[] parameter(0)
  ROOT %out = s32[] copy(%a),
    metadata={op_name="jit(main)/copy"},
    control-predecessors={%missing}
}`;
  const module = parseHlo(source);
  assert.equal(module.byName.get('main').byName.get('out').raw.split('\n').length, 3);
  assert.deepEqual(module.byName.get('main').byName.get('out').controlPredecessors, ['missing']);
  assert.match(module.warnings[0], /missing control predecessor %missing/);
});

test('metadata stack_frame_id resolves through the module source table, innermost first', () => {
  for (const stage of ['before', 'after']) {
    const module = parseHlo(readFileSync(new URL(`./examples/tpu-v6e/nested_calls.${stage}.hlo`, import.meta.url), 'utf8'));
    const sine = module.computations.flatMap(c => c.nodes).find(node => node.op === 'sine');
    const frame = Number(/stack_frame_id=(\d+)/.exec(sine.raw)[1]);
    assert.deepEqual(sourceStack(module, frame).map(f => `${f.func}:${f.line}`), ['_inner:194', '_middle:198', 'nested_calls:203', '<module>:209']);
    assert.equal(sourceStack(module, frame)[0].file, 'scripts/dump_hlo.py');
  }
});
