import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { memoryLocations, nodeMemoryLocations } from './src/lib/memory-location.ts';
import { parseHlo } from './src/lib/parser.ts';
import { instructionGuide } from './src/lib/instruction-guide.ts';
import { memorySpaceLabel } from './src/lib/memory-space.ts';

const module = parseHlo(readFileSync(new URL('./sample.hlo', import.meta.url), 'utf8'));

test('array nodes show their result memory space', () => {
  const entry = module.byName.get('main.3');
  assert.deepEqual(
    memoryLocations(entry.byName.get('a.1').type).map(item => item.label),
    ['HBM']
  );
  assert.deepEqual(
    memoryLocations(entry.byName.get('copy-done.1').type).map(item => item.label),
    ['VMEM']
  );
});

test('tuple nodes show memory for each result slot', () => {
  const entry = module.byName.get('main.3');
  const locations = memoryLocations(entry.byName.get('copy-start.1').type);
  assert.deepEqual(
    locations.map(item => [item.path, item.label]),
    [
      ['0', 'VMEM'],
      ['1', 'HBM'],
      ['2', 'SFLAG']
    ]
  );
  assert.deepEqual(
    memoryLocations('(f32[2]{0}, (bf16[2]{0:S(1)}, s32[]{:S(5)}))').map(item => [item.path, item.label]),
    [
      ['0', 'HBM'],
      ['1.0', 'VMEM'],
      ['1.1', 'HOST']
    ]
  );
});

// Ground truth: examples/tpu-v6e/*.memory.json, extracted from XLA's buffer assignment of the same compilation
// (scripts/extract_memory_truth.py). Keys are instruction names and shape indices; values are memory spaces.
const dir = new URL('./examples/tpu-v6e/', import.meta.url);
const spaceLabel = memorySpaceLabel;

// Data leaves of a result type, keyed like XLA shape indices ("" for an array, "1,0" for a nested slot).
function dataLeaves(type, index = []) {
  const shape = type.replace(/\/\*[^*]*\*\//g, '').trim();
  if (!shape.startsWith('(')) return shape.startsWith('token[') ? [] : [index.join(',')];
  let depth = 0,
    start = 1;
  const items = [];
  for (let i = 1; i < shape.length - 1; i++) {
    if ('([{'.includes(shape[i])) depth++;
    else if (')]}'.includes(shape[i])) depth--;
    else if (shape[i] === ',' && !depth) {
      items.push(shape.slice(start, i));
      start = i + 1;
    }
  }
  items.push(shape.slice(start, -1));
  return items.flatMap((item, i) => dataLeaves(item, [...index, i]));
}

for (const file of readdirSync(dir)
  .filter(name => name.endsWith('.after.hlo'))
  .sort()) {
  test(`${file}: memory labels match XLA buffer assignment`, () => {
    const module = parseHlo(readFileSync(new URL(file, dir), 'utf8'));
    const truth = JSON.parse(readFileSync(new URL(file.replace('.hlo', '.memory.json'), dir), 'utf8'));
    assert.equal(module.scheduled, true);
    let checked = 0;
    for (const computation of module.computations)
      for (const node of computation.nodes) {
        const shown = new Map(
          nodeMemoryLocations(module, computation, node).map(location => [
            location.path === null ? '' : location.path.replaceAll('.', ','),
            location.label
          ])
        );
        const actual = truth[node.name] || {};
        for (const leaf of dataLeaves(node.type)) {
          const space = actual[leaf];
          const where = `%${computation.name}/%${node.name} {${leaf}}`;
          if (space === undefined || space === 'thread-local')
            assert.equal(shown.get(leaf), undefined, `${where} has no buffer of its own, so no location is shown`);
          else {
            assert.equal(shown.get(leaf), spaceLabel(space), `${where} is in ${spaceLabel(space)}`);
            checked++;
          }
        }
        for (const leaf of shown.keys())
          assert.ok(dataLeaves(node.type).includes(leaf), `%${node.name} shows a location for a real data leaf`);
      }
    assert.ok(checked > 0);
  });
}

for (const file of readdirSync(dir)
  .filter(name => name.endsWith('.after.hlo'))
  .sort()) {
  test(`${file}: the inspector's memory-space text agrees with XLA buffer assignment`, () => {
    const module = parseHlo(readFileSync(new URL(file, dir), 'utf8'));
    const truth = JSON.parse(readFileSync(new URL(file.replace('.hlo', '.memory.json'), dir), 'utf8'));
    for (const computation of module.computations)
      for (const node of computation.nodes) {
        const group = instructionGuide(node, { module, computation }).resultGroup;
        const rows = group.kind === 'array' ? [['', group.details]] : group.slots.map((slot, index) => [String(index), slot.details]);
        for (const [leaf, details] of rows) {
          const text = details.find(detail => detail.key === 'space')?.text || '';
          const space = truth[node.name]?.[leaf];
          const where = `%${computation.name}/%${node.name} {${leaf}}`;
          if (typeof space === 'number')
            assert.match(text, new RegExp(spaceLabel(space).replace(/[()]/g, '\\$&')), `${where} says ${spaceLabel(space)}`);
          else
            assert.doesNotMatch(text, /^(HBM|VMEM|SFLAG|HOST|SMEM|S\(\d+\)) \(/, `${where} has no buffer and must not claim one: ${text}`);
        }
      }
  });
}

test('lowered (unscheduled) modules show no memory locations: spaces are not assigned yet', () => {
  for (const file of readdirSync(dir).filter(name => name.endsWith('.before.hlo'))) {
    const module = parseHlo(readFileSync(new URL(file, dir), 'utf8'));
    assert.equal(module.scheduled, false);
    for (const computation of module.computations)
      for (const node of computation.nodes) assert.deepEqual(nodeMemoryLocations(module, computation, node), [], `${file} %${node.name}`);
  }
});

test('/*index=N*/ comments inside long tuple types do not hide slots', () => {
  assert.deepEqual(
    memoryLocations('(f32[2]{0}, f32[2]{0}, f32[2]{0}, f32[2]{0}, f32[2]{0}, /*index=5*/f32[2]{0:S(1)})')
      .map(item => [item.path, item.label])
      .at(-1),
    ['5', 'VMEM']
  );
  assert.deepEqual(
    memoryLocations('(f32[]{:T(128)}, token[])').map(item => item.path),
    ['0']
  );
});

test('TPU memory-space numbers have names backed by evidence (see src/lib/memory-space.ts)', () => {
  assert.deepEqual([0, 1, 2, 5, 6, 3].map(memorySpaceLabel), ['HBM', 'VMEM', 'SFLAG', 'HOST', 'SMEM', 'S(3)']);
});
