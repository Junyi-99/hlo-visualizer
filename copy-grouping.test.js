import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parseHlo, reachable, shortestDependencyPath } from './src/lib/parser.ts';
import { groupCopyPairs } from './src/lib/copy-grouping.ts';

const module = parseHlo(readFileSync(new URL('./sample.hlo', import.meta.url), 'utf8'));
const entry = module.byName.get('main.3');
test('copy pairs are projected as dashed-card data without modifying the parsed computation', () => {
  const original = JSON.stringify(entry.nodes);
  const off = groupCopyPairs(entry, false);
  assert.equal(off.computation, entry);
  const grouped = groupCopyPairs(entry, true);
  assert.equal(grouped.groups.size, 3);
  assert.equal(grouped.computation.nodes.length, entry.nodes.length - 3);
  assert.equal(JSON.stringify(entry.nodes), original);
  assert.equal(grouped.aliases.get('copy-start.1'), 'copy-done.1');
  assert.equal(grouped.groups.get('copy-done.1').direction, 'HBM → VMEM');
  assert.equal(grouped.groups.get('copy-done').direction, 'VMEM → HBM');
  assert.equal(grouped.groups.get('copy-done').shape, 'bf16[128,128]');
  assert.equal(grouped.computation.byName.get('copy-done').root, true);
  assert(!grouped.computation.byName.has('copy-start'));
});
test('grouping preserves external inputs, consumers, paths and control dependencies', () => {
  const source = `HloModule grouping, is_scheduled=true
ENTRY %main {
  %p = f32[8]{0} parameter(0)
  %guard = f32[8]{0} parameter(1)
  %cs = (f32[8]{0:S(1)}, f32[8]{0}, u32[]{:S(2)}) copy-start(%p), control-predecessors={%guard}
  %done = f32[8]{0:S(1)} copy-done(%cs)
  ROOT %out = f32[8]{0:S(1)} add(%done, %done), control-predecessors={%cs}
}`;
  const original = parseHlo(source).computations[0];
  const result = groupCopyPairs(original, true).computation;
  assert.deepEqual(result.byName.get('done').operands, ['p']);
  assert.deepEqual(result.byName.get('done').controlPredecessors, ['guard']);
  assert.deepEqual(result.byName.get('out').operands, ['done', 'done']);
  assert.deepEqual(result.byName.get('out').controlPredecessors, ['done']);
  assert.deepEqual(result.byName.get('done').users, ['out']);
  assert.deepEqual(result.byName.get('done').controlSuccessors, ['out']);
  assert(reachable(result, 'out', 'up').has('guard'));
  assert.deepEqual(shortestDependencyPath(result, 'p', 'out'), ['p', 'done', 'out']);
});
test('pairing follows operands rather than suffixes; unmatched and ambiguous events stay separate', () => {
  const source = `HloModule pairing
ENTRY %main {
  %p = f32[2]{0} parameter(0)
  %start.9 = (f32[2]{0:S(1)}, f32[2]{0}, u32[]{}) copy-start(%p)
  %finish.2 = f32[2]{0:S(1)} copy-done(%start.9)
  %orphan = (f32[2]{0}, f32[2]{0:S(1)}, u32[]{}) copy-start(%p)
  %shared = (f32[2]{0:S(1)}, f32[2]{0}, u32[]{}) copy-start(%p)
  %done.1 = f32[2]{0:S(1)} copy-done(%shared)
  %done.2 = f32[2]{0:S(1)} copy-done(%shared)
  ROOT %result = (f32[2]{0:S(1)}, f32[2]{0:S(1)}) tuple(%finish.2, %done.2)
}`;
  const result = groupCopyPairs(parseHlo(source).computations[0], true);
  assert.equal(result.groups.size, 1);
  assert.equal(result.aliases.get('start.9'), 'finish.2');
  for (const name of ['orphan', 'shared', 'done.1', 'done.2']) assert(result.computation.byName.has(name));
});
