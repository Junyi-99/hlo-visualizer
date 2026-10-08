import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parseHlo, reachable, computationLinks, nodeSummary } from './parser.js';

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
