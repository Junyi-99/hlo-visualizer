import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { memoryLocations } from './src/lib/memory-location.ts';
import { parseHlo } from './src/lib/parser.ts';

const module = parseHlo(readFileSync(new URL('./sample.hlo', import.meta.url), 'utf8'));

test('array nodes show their result memory space', () => {
  const entry = module.byName.get('main.3');
  assert.deepEqual(memoryLocations(entry.byName.get('a.1').type).map(item => item.label), ['HBM']);
  assert.deepEqual(memoryLocations(entry.byName.get('copy-done.1').type).map(item => item.label), ['VMEM']);
});

test('tuple nodes show memory for each result slot', () => {
  const entry = module.byName.get('main.3');
  const locations = memoryLocations(entry.byName.get('copy-start.1').type);
  assert.deepEqual(locations.map(item => [item.path, item.label]), [
    ['0', 'VMEM'], ['1', 'HBM'], ['2', 'S(2)']
  ]);
  assert.deepEqual(memoryLocations('(f32[2]{0}, (bf16[2]{0:S(1)}, s32[]{:S(5)}))')
    .map(item => [item.path, item.label]), [['0', 'HBM'], ['1.0', 'VMEM'], ['1.1', 'HOST']]);
});
