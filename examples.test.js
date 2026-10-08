// Every bundled example must parse cleanly and be explained segment by segment.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { computationLinks, parseHlo } from './src/lib/parser.ts';
import { instructionGuide } from './src/lib/instruction-guide.ts';

const dir = new URL('./examples/tpu-v6e/', import.meta.url);
const files = readdirSync(dir).filter(name => name.endsWith('.hlo')).sort();
const decode = text => text.replace(/&amp;|&lt;|&gt;|&quot;|&#39;/g, entity => ({ '&amp;':'&', '&lt;':'<', '&gt;':'>', '&quot;':'"', '&#39;':"'" })[entity]);

// Visible text of the guide html, with the characters that sit outside every data-part segment blanked out.
function uncoveredText(html) {
  let text = '', uncovered = '', depth = 0;
  const stack = [];
  for (const [, closing, attributes, chunk] of html.matchAll(/<(\/?)span([^>]*)>|([^<]+)/g)) {
    if (chunk !== undefined) {
      const value = decode(chunk);
      text += value;
      uncovered += depth ? ' '.repeat(value.length) : value;
    } else if (closing) { if (stack.pop()) depth--; }
    else { const part = attributes.includes('data-part='); stack.push(part); if (part) depth++; }
  }
  return { text, uncovered };
}

test('examples cover compiled and lowered HLO from many programs', () => {
  assert.ok(files.length >= 40, `expected the TPU examples, found ${files.length}`);
  assert.ok(files.some(name => name.endsWith('.before.hlo')) && files.some(name => name.endsWith('.after.hlo')));
});

for (const file of files) {
  test(`${file}: parses, links computations and explains every instruction segment`, () => {
    const module = parseHlo(readFileSync(new URL(file, dir), 'utf8'));
    assert.deepEqual(module.warnings, []);
    assert.ok(module.computations.some(computation => computation.entry), 'has an ENTRY computation');
    const called = new Set(computationLinks(module).map(link => link.to));
    for (const computation of module.computations) {
      if (!computation.entry) assert.ok(called.has(computation.name), `%${computation.name} is called by some instruction`);
      for (const node of computation.nodes) {
        const where = `%${computation.name}/%${node.name}`;
        const guide = instructionGuide(node, { module, computation });
        const { text, uncovered } = uncoveredText(guide.html);
        assert.equal(text, node.raw, `${where} renders its source exactly`);
        assert.deepEqual(uncovered.match(/[A-Za-z0-9_]+/g)?.filter(word => word !== 'ROOT') ?? [], [], `${where} has no unexplained text`);
        const segments = new Set([...guide.html.matchAll(/data-part="([^"]+)"/g)].map(match => match[1]));
        const explained = new Set(guide.parts.map(part => part.key));
        for (const key of segments) assert.ok(explained.has(key), `${where} explains segment ${key}`);
        for (const key of explained) assert.ok(segments.has(key), `${where} explanation ${key} points at a segment`);
        for (const part of guide.parts) assert.doesNotMatch(part.text, /see the XLA operation semantics doc/, `${where} ${part.key} has a specific explanation`);
      }
    }
  });
}
