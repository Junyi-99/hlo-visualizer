import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { nodeSummary, parseHlo } from './src/lib/parser.ts';
import { instructionGuide, layoutDiagram } from './src/lib/instruction-guide.ts';
import { extractHloMetadata } from './src/lib/metadata.ts';

const module = parseHlo(readFileSync(new URL('./sample.hlo', import.meta.url), 'utf8'));
const entry = module.byName.get('main.3');
const visibleText = html =>
  html.replace(/<[^>]*>/g, '').replace(
    /&amp;|&lt;|&gt;|&quot;|&#39;/g,
    entity =>
      ({
        '&amp;': '&',
        '&lt;': '<',
        '&gt;': '>',
        '&quot;': '"',
        '&#39;': "'"
      })[entity]
  );

test('copy-start guide preserves the original instruction and identifies parts', () => {
  const node = entry.byName.get('copy-start');
  const guide = instructionGuide(node);
  assert.equal(visibleText(guide.html), node.raw);
  assert.deepEqual(
    guide.parts.map(part => part.key),
    ['name', 'tuple', 'dest', 'source', 'context', 'shape', 'order', 'tile', 'space', 'op', 'operand', 'priority']
  );
  assert.match(guide.html, /data-part="priority"[^>]*>dma_priority=1/);
});

test('layout diagram reflects VMEM to HBM and physical offsets', () => {
  const diagram = layoutDiagram(entry.byName.get('copy-start'));
  assert.match(diagram, /VMEM S\(1\) → HBM S\(0\)/);
  assert.match(diagram, />256<\/text>/);
  assert.match(diagram, /16 outer tiles/);
  assert.match(layoutDiagram(entry.byName.get('copy-start.1')), /HBM S\(0\) → VMEM S\(1\)/);
});

test('while condition tuple parameter is broken into four explained slots', () => {
  const computation = module.byName.get('wide.region_1.2');
  const node = computation.byName.get('wide.arg_tuple.3');
  const guide = instructionGuide(node, { module, computation });
  assert.equal(visibleText(guide.html), node.raw);
  assert.deepEqual(
    guide.parts.filter(part => part.key.startsWith('slot-')).map(part => part.key),
    ['slot-0', 'slot-1', 'slot-2', 'slot-3']
  );
  assert.match(guide.parts.find(part => part.key === 'slot-0').text, /%copy\.10.*%add\.2.*%get-tuple-element\.15/);
  assert.match(guide.parts.find(part => part.key === 'slot-1').text, /bf16\[128,128\].*VMEM.*does not read this element directly/);
  assert.match(guide.parts.find(part => part.key === 'tile').text, /T\(128\).*does not turn a scalar into 128 logical elements/);
  assert.match(guide.parts.find(part => part.key === 'parameter-index').text, /the whole tuple is one parameter/);
  assert.equal(guide.resultGroup.kind, 'tuple');
  assert.equal(guide.resultGroup.slots.length, 4);
  assert.deepEqual(
    guide.resultGroup.slots[0].details.map(detail => detail.label),
    ['Element type and logical shape', 'Physical tiling', 'Memory space', 'Loop state source', 'Use in this computation']
  );
  assert.match(layoutDiagram(node), /Elements 1, 2 · VMEM S\(1\)/);
});

test('every result type exposes a hierarchy, including copy tuples and single arrays', () => {
  const copy = entry.byName.get('copy-start');
  const copyGroup = instructionGuide(copy).resultGroup;
  assert.equal(copyGroup.kind, 'tuple');
  assert.deepEqual(
    copyGroup.slots.map(slot => slot.key),
    ['dest', 'source', 'context']
  );
  assert.match(copyGroup.slots[0].details[0].text, /bf16\[128,128\]/);

  const array = entry.byName.get('copy-done');
  const arrayGuide = instructionGuide(array);
  assert.equal(visibleText(arrayGuide.html), array.raw);
  assert.equal(arrayGuide.resultGroup.kind, 'array');
  assert.deepEqual(
    arrayGuide.resultGroup.details.map(detail => detail.label),
    ['Element type and logical shape', 'Dimension order', 'Physical tiling', 'Memory space']
  );
});

test('highlighted layout fragments map to result type detail rows', () => {
  const guide = instructionGuide(entry.byName.get('constant.5'));
  assert.match(guide.html, /data-part="tile"[^>]*>T\(128\)/);
  assert.deepEqual(
    guide.resultGroup.details.map(detail => detail.key),
    ['shape', 'tile', 'space']
  );
  assert.equal(guide.resultGroup.details.find(detail => detail.key === 'tile').label, 'Physical tiling');
});

test('constant literal has a matching instruction fragment and explanation', () => {
  const node = entry.byName.get('constant.5');
  const guide = instructionGuide(node);
  assert.equal(visibleText(guide.html), node.raw);
  assert.match(guide.html, /data-part="literal"[^>]*>\(0\)<\/span>/);
  assert.match(guide.parts.find(part => part.key === 'literal').text, /0.*the value given directly/);
});

test('fusion kind and called computation are explained and linked to instruction fragments', () => {
  const computation = module.byName.get('wide.region_0.1.sunk');
  const node = computation.byName.get('fusion.8');
  const guide = instructionGuide(node, { module, computation });
  assert.equal(visibleText(guide.html), node.raw);
  assert.match(guide.html, /data-part="fusion-kind"[^>]*>kind=kOutput<\/span>/);
  assert.match(guide.html, /data-part="called-computation"[^>]*>calls=%fused_computation\.clone\.1<\/span>/);
  assert.match(guide.parts.find(part => part.key === 'fusion-kind').text, /Output fusion/);
  assert.match(
    guide.parts.find(part => part.key === 'called-computation').text,
    /parameter 0 ← %copy\.11; parameter 1 ← %get-tuple-element\.41\. Its ROOT %convolution\.3 defines the fusion's result/
  );
});

test('array result type is highlighted through its closing layout brace', () => {
  const node = entry.byName.get('copy-done.2');
  const guide = instructionGuide(node);
  assert.equal(visibleText(guide.html), node.raw);
  assert.match(guide.html, /<span class="hlo-type part-shape" data-part="shape"[^>]*>bf16\[128,128\]\{/);
  assert.match(guide.html, /S\(1\)<\/span>}<\/span> <span class="hlo-part part-op"/);
});

test('priority inside backend_config is highlighted without changing the raw line', () => {
  const node = entry.byName.get('copy-start');
  const full = { ...node, raw: node.raw.replace(', dma_priority=1', ', backend_config={"flag_configs":[],"dma_priority":1}') };
  const guide = instructionGuide(full);
  assert.match(guide.html, /data-part="priority"[^>]*>&quot;dma_priority&quot;:1/);
  assert.match(guide.parts.find(part => part.key === 'priority').text, /dma_priority=1/);
});

test('splash attention custom call keeps six dependencies and hides only the encoded body', () => {
  const body = 'A'.repeat(500);
  const source = `HloModule jit_splash_attention_kernel
ENTRY %main.28 (q: bf16[8,2048,128]) -> bf16[8,2048,128] {
  %constant.4 = s8[1,2,2]{2,1,0} constant({ { { 0, 0 }, { 0, 1 } } })
  %constant.5 = s8[1,2,2]{2,1,0} constant({ { { 1, 0 }, { 2, 1 } } })
  %Arg_0.1 = bf16[8,2048,128]{2,1,0} parameter(0), metadata={op_name="q"}
  %Arg_1.2 = bf16[8,2048,128]{2,1,0} parameter(1), metadata={op_name="k"}
  %Arg_2.3 = bf16[8,2048,128]{2,1,0} parameter(2), metadata={op_name="v"}
  %constant.6 = s32[2048]{0} constant({...})
  %broadcast.0 = s32[2048,128]{1,0} broadcast(%constant.6), dimensions={0}
  %custom-call.0 = (f32[1024,128]{1,0}, f32[1024,128]{1,0}, f32[1024,128]{1,0}, bf16[8,2048,128]{2,1,0}) custom-call(%constant.4, %constant.5, %Arg_0.1, %Arg_1.2, %Arg_2.3, /*index=5*/%broadcast.0), custom_call_target="tpu_custom_call", operand_layout_constraints={s8[1,2,2]{2,1,0}}, metadata={op_name="splash_mha_fwd/pallas_call" source_line=1100}, backend_config={"custom_call_config":{"body":"${body}","serialization_format":1}}
  ROOT %get-tuple-element.0 = bf16[8,2048,128]{2,1,0} get-tuple-element(%custom-call.0), index=3
}`;
  const parsed = parseHlo(source);
  const computation = parsed.byName.get('main.28');
  const node = computation.byName.get('custom-call.0');
  const guide = instructionGuide(node, { module: parsed, computation });
  assert.deepEqual(parsed.warnings, []);
  assert.deepEqual(node.operands, ['constant.4', 'constant.5', 'Arg_0.1', 'Arg_1.2', 'Arg_2.3', 'broadcast.0']);
  assert.match(nodeSummary(computation.byName.get('Arg_0.1')), /^q · bf16\[8,2048,128\]/);
  assert.equal(computation.byName.get('get-tuple-element.0').index, 3);
  const outputGuide = instructionGuide(computation.byName.get('get-tuple-element.0'));
  assert.match(outputGuide.html, /data-part="tuple-index"[^>]*>index=3/);
  assert.match(outputGuide.parts.find(part => part.key === 'tuple-index').text, /element 3 .*%custom-call\.0/);
  const broadcastGuide = instructionGuide(computation.byName.get('broadcast.0'));
  assert.match(broadcastGuide.html, /data-part="dimensions"[^>]*>dimensions=\{0\}/);
  assert.match(broadcastGuide.parts.find(part => part.key === 'dimensions').text, /input dimensions.*output/);
  assert.match(nodeSummary(node), /tpu_custom_call · 4 outputs/);
  assert.equal(visibleText(guide.html), node.raw);
  assert.ok(!guide.compactHtml.includes(body));
  assert.match(guide.compactHtml, /\[500 encoded characters\]/);
  assert.match(guide.compactHtml, /metadata=\{…\}/);
  const metadata = extractHloMetadata(node.raw);
  assert.deepEqual(metadata.fields, [
    { name: 'op_name', value: 'splash_mha_fwd/pallas_call' },
    { name: 'source_line', value: '1100' }
  ]);
  assert.equal(guide.resultGroup.slots.length, 4);
  assert.match(guide.resultGroup.slots[3].details.at(-1).text, /%get-tuple-element\.0/);
  for (const key of ['operand', 'target', 'constraints', 'metadata', 'backend']) {
    assert.ok(
      guide.parts.some(part => part.key === key),
      `missing ${key} explanation`
    );
    assert.match(guide.html, new RegExp(`data-part="${key}"`));
  }
});
