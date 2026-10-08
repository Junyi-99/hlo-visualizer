import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parseHlo } from './parser.js';
import { instructionGuide, layoutDiagram } from './instruction-guide.js';

const module = parseHlo(readFileSync(new URL('./sample.hlo', import.meta.url), 'utf8'));
const entry = module.byName.get('main.3');
const visibleText = html => html.replace(/<[^>]*>/g, '')
  .replace(/&amp;|&lt;|&gt;|&quot;|&#39;/g, entity => ({
    '&amp;':'&', '&lt;':'<', '&gt;':'>', '&quot;':'"', '&#39;':"'"
  })[entity]);

test('copy-start guide preserves the original instruction and identifies parts', () => {
  const node = entry.byName.get('copy-start');
  const guide = instructionGuide(node);
  assert.equal(visibleText(guide.html), node.raw);
  assert.deepEqual(guide.parts.map(part => part.key), [
    'name','tuple','dest','source','context','shape','order','tile','space','op','operand','priority'
  ]);
  assert.match(guide.html, /data-part="priority"[^>]*>dma_priority=1/);
});

test('layout diagram reflects VMEM to HBM and physical offsets', () => {
  const diagram = layoutDiagram(entry.byName.get('copy-start'));
  assert.match(diagram, /VMEM S\(1\) → HBM S\(0\)/);
  assert.match(diagram, />256<\/text>/);
  assert.match(diagram, /16 个外层 tile/);
  assert.match(layoutDiagram(entry.byName.get('copy-start.1')), /HBM S\(0\) → VMEM S\(1\)/);
});

test('while condition tuple parameter is broken into four explained slots', () => {
  const computation = module.byName.get('wide.region_1.2');
  const node = computation.byName.get('wide.arg_tuple.3');
  const guide = instructionGuide(node, { module, computation });
  assert.equal(visibleText(guide.html), node.raw);
  assert.deepEqual(guide.parts.filter(part => part.key.startsWith('slot-')).map(part => part.key),
    ['slot-0','slot-1','slot-2','slot-3']);
  assert.match(guide.parts.find(part => part.key === 'slot-0').text, /%copy\.10.*%add\.2.*%get-tuple-element\.15/);
  assert.match(guide.parts.find(part => part.key === 'slot-1').text, /bf16\[128,128\].*VMEM.*没有直接取出/);
  assert.match(guide.parts.find(part => part.key === 'tile').text, /T\(128\).*不会把标量变成 128 个逻辑元素/);
  assert.match(guide.parts.find(part => part.key === 'parameter-index').text, /整个 tuple 是一个参数/);
  assert.equal(guide.resultGroup.kind, 'tuple');
  assert.equal(guide.resultGroup.slots.length, 4);
  assert.deepEqual(guide.resultGroup.slots[0].details.map(detail => detail.label), ['元素类型与逻辑形状','物理分块','内存空间','循环状态来源','本计算中的使用']);
  assert.match(layoutDiagram(node), /第 1、2 项 · VMEM S\(1\)/);
});

test('every result type exposes a hierarchy, including copy tuples and single arrays', () => {
  const copy = entry.byName.get('copy-start');
  const copyGroup = instructionGuide(copy).resultGroup;
  assert.equal(copyGroup.kind, 'tuple');
  assert.deepEqual(copyGroup.slots.map(slot => slot.key), ['dest','source','context']);
  assert.match(copyGroup.slots[0].details[0].text, /bf16\[128,128\]/);

  const array = entry.byName.get('copy-done');
  const arrayGuide = instructionGuide(array);
  assert.equal(visibleText(arrayGuide.html), array.raw);
  assert.equal(arrayGuide.resultGroup.kind, 'array');
  assert.deepEqual(arrayGuide.resultGroup.details.map(detail => detail.label), ['元素类型与逻辑形状','维度顺序','物理分块','内存空间']);
});

test('priority inside backend_config is highlighted without changing the raw line', () => {
  const node = entry.byName.get('copy-start');
  const full = { ...node, raw: node.raw.replace(', dma_priority=1', ', backend_config={"flag_configs":[],"dma_priority":1}') };
  const guide = instructionGuide(full);
  assert.match(guide.html, /data-part="priority"[^>]*>&quot;dma_priority&quot;:1/);
  assert.match(guide.parts.find(part => part.key === 'priority').text, /dma_priority=1/);
});
