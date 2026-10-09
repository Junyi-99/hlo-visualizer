import type { Computation, GuidePart, HloModule, HloNode, InstructionGuide, ResultGroup, TypeDetail, TypeSlot } from './types';
import { extractHloMetadata } from './metadata.ts';
import { splitTopLevel } from './parser.ts';
import { attributeKey, explainAttribute, opDescription } from './hlo-reference.ts';
import { bufferStatus, type BufferStatus } from './memory-location.ts';
import { memorySpaceLabel, shapeMemoryText } from './memory-space.ts';

interface GuideContext {
  computation?: Computation;
  module?: HloModule;
}

const escapeHtml = (value: unknown) =>
  String(value).replace(
    /[&<>"']/g,
    c => (({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }) as Record<string, string>)[c]
  );
const STYLED_KEYS =
  /^(name|tuple|dest|source|context|shape|order|tile|space|op|operand|priority|literal|parameter-index|target|constraints|metadata|backend|dimensions|tuple-index|fusion-kind|called-computation|slot-\d+)$/;
// Keys without a CSS color get a stable hue so neighbouring segments stay distinguishable.
const hue = (key: string) => [...key].reduce((h, c) => (h * 31 + c.charCodeAt(0)) % 360, 7);
export const partColor = (key: string) => (STYLED_KEYS.test(key) ? undefined : `hsl(${hue(key)} 45% 40%)`);
const partStyle = (key: string) => (partColor(key) ? ` style="--part:${partColor(key)}"` : '');
const open = (key: string) => `<span class="hlo-part part-${key}" data-part="${key}" tabindex="0"${partStyle(key)}>`;
const markHtml = (key: string, html: string) => `${open(key)}${html}</span>`;
const mark = (key: string, text: string) => markHtml(key, escapeHtml(text));

function splitTuple(text: string) {
  const parts: { body: string; separator: string }[] = [];
  let parens = 0,
    brackets = 0,
    braces = 0,
    start = 1;
  for (let i = 1; i < text.length - 1; i++) {
    const char = text[i];
    if (char === '(') parens++;
    else if (char === ')') parens--;
    else if (char === '[') brackets++;
    else if (char === ']') brackets--;
    else if (char === '{') braces++;
    else if (char === '}') braces--;
    else if (char === ',' && !parens && !brackets && !braces) {
      parts.push({ body: text.slice(start, i), separator: ',' });
      start = i + 1;
    }
  }
  parts.push({ body: text.slice(start, -1), separator: '' });
  return parts;
}

function layoutHtml(layout: string) {
  let output = '',
    position = 0;
  const regex = /\d+(?:,\d+)*|T(?:\([^)]*\))+|S\(\d+\)/g;
  for (const match of layout.matchAll(regex)) {
    output += escapeHtml(layout.slice(position, match.index));
    const key = match[0].startsWith('T') ? 'tile' : match[0].startsWith('S') ? 'space' : 'order';
    output += mark(key, match[0]);
    position = match.index + match[0].length;
  }
  return output + escapeHtml(layout.slice(position));
}

function shapeHtml(shape: string) {
  const match = /^(\s*)([a-z][\w]*\[[^\]]*\])(\{[^}]*\})?(.*)$/s.exec(shape);
  if (!match) return escapeHtml(shape);
  const type = escapeHtml(match[2]) + (match[3] ? `{${layoutHtml(match[3].slice(1, -1))}}` : '');
  return escapeHtml(match[1]) + `<span class="hlo-type part-shape" data-part="shape" tabindex="0">${type}</span>` + escapeHtml(match[4]);
}

// Attributes after the operand list. Each top-level `name=value` becomes one explained segment.
function suffixHtml(suffix: string) {
  const pieces = splitTopLevel(suffix);
  const attributes: { name: string; value: string }[] = [];
  const html = pieces
    .map((piece, index) => {
      const match = /^(\s*)([\w-]+)=([\s\S]*?)(\s*)$/.exec(piece);
      const separator = index < pieces.length - 1 ? ',' : '';
      if (!match) return escapeHtml(piece) + separator;
      const [, lead, name, value, trail] = match;
      attributes.push({ name, value });
      const text = `${name}=${value}`;
      let inner = escapeHtml(text);
      if (name === 'backend_config') {
        // The DMA priority inside backend_config gets its own nested segment.
        const priority = /"dma_priority"\s*:\s*"?\d+"?/.exec(text);
        if (priority) {
          inner =
            escapeHtml(text.slice(0, priority.index)) +
            mark('priority', priority[0]) +
            escapeHtml(text.slice(priority.index + priority[0].length));
          attributes.push({ name: 'dma_priority', value: /\d+/.exec(priority[0])![0] });
        }
      }
      return escapeHtml(lead) + markHtml(attributeKey(name), inner) + escapeHtml(trail) + separator;
    })
    .join('');
  return { html, attributes };
}

function compactEncodedBody(html: string) {
  return html.replace(
    /(&quot;body&quot;\s*:\s*&quot;)([A-Za-z0-9+/=]{300,})(&quot;)/,
    (_match, before: string, body: string, after: string) => `${before}[${body.length.toLocaleString('en-US')} encoded characters]${after}`
  );
}

// Operand references, with or without the % sigil, plus printer comments such as /*index=5*/.
function operandHtml(args: string) {
  return splitTopLevel(args)
    .map((piece, index, all) => {
      const separator = index < all.length - 1 ? ',' : '';
      const match = /^([\s\S]*?)(%?[\w.-]+(?:#\d+)?)(\s*)$/.exec(piece);
      if (!match) return escapeHtml(piece) + separator;
      const head = match[1]
        .replace(/\/\*[^*]*\*\//g, comment => `\u0000${comment}\u0000`)
        .split('\u0000')
        .map(chunk => (chunk.startsWith('/*') ? mark('operand-index', chunk) : escapeHtml(chunk)))
        .join('');
      return head + mark('operand', match[2]) + escapeHtml(match[3]) + separator;
    })
    .join('');
}

const memorySpace = (shape: string) => Number(/S\((\d+)\)/.exec(shape)?.[1] || 0);

function memoryLabel(shape: string) {
  const value = memorySpace(shape);
  return value === 0 ? 'HBM (S(0) omitted)' : `${memorySpaceLabel(value)} (S(${value}))`;
}

const DTYPES: Record<string, string> = {
  pred: 'boolean',
  s4: '4-bit signed integer',
  s8: '8-bit signed integer',
  s16: '16-bit signed integer',
  s32: '32-bit signed integer',
  s64: '64-bit signed integer',
  u4: '4-bit unsigned integer',
  u8: '8-bit unsigned integer',
  u16: '16-bit unsigned integer',
  u32: '32-bit unsigned integer',
  u64: '64-bit unsigned integer',
  f16: '16-bit half-precision float',
  bf16: '16-bit brain float (bfloat16)',
  f32: '32-bit float',
  f64: '64-bit float',
  f8e4m3fn: '8-bit float (e4m3)',
  f8e5m2: '8-bit float (e5m2)',
  c64: '64-bit complex (two f32)',
  c128: '128-bit complex (two f64)',
  token: 'token (only orders side effects; holds no data)'
};

// What to say about where a value lives. `status` is null when the guide has no module context.
function placement(shape: string, status: BufferStatus | null, node: HloNode) {
  const space = /S\((\d+)\)/.exec(shape)?.[1];
  const marked = space === undefined ? 'the layout has no S(n)' : `the layout says S(${space})`;
  switch (status) {
    case 'fusion':
      if (node.op === 'parameter')
        return `A fusion parameter: it is the operand passed in by the caller and takes no buffer of its own (${marked}).`;
      return `No buffer of its own: this instruction is computed inside a fusion kernel and its result is not written back to HBM or VMEM by itself (${marked}, which is not a storage location).${node.root ? ' Its result is the output of this fusion; see the fusion instruction that calls it for the actual location.' : ''}`;
    case 'thread-local':
      return `A value inside a reducer or comparator; XLA gives it thread-local storage, not HBM or VMEM (${marked}).`;
    case 'unassigned':
      return `Lowered (unscheduled) HLO has not been through memory-space assignment yet (${marked}); see the compiled HLO for the actual location.`;
    default:
      return `${shapeMemoryText(shape)}${status ? '' : ' (assuming this instruction has a buffer of its own)'}`;
  }
}

function shapeDescription(shape: string, includeMemory = true) {
  const match = /([a-z][\w]*)\[([^\]]*)\]/.exec(shape);
  if (!match) return "see the raw HLO for this element's type";
  const dtype = DTYPES[match[1]] || match[1];
  const logical = match[1] === 'token' ? '' : match[2] ? `, a ${match[2].replaceAll(',', '×')} array` : ', a scalar (logical shape [])';
  return `${match[1]}[${match[2]}]: ${dtype}${logical}${includeMemory ? `; ${memoryLabel(shape)}` : ''}`;
}

function typeDetails(shape: string, status: BufferStatus | null = null, node?: HloNode): TypeDetail[] {
  const order = /\{(\d+(?:,\d+)*)(?=[:}])/.exec(shape)?.[1];
  const tile = /T(?:\([^)]*\))+/.exec(shape)?.[0];
  return [
    { key: 'shape', label: 'Element type and logical shape', text: shapeDescription(shape, false) },
    ...(order
      ? [{ key: 'order', label: 'Dimension order', text: `${order}: minor-to-major; the dimensions listed first vary fastest.` }]
      : []),
    ...(tile
      ? [
          {
            key: 'tile',
            label: 'Physical tiling',
            text: shape.includes('[]')
              ? `${tile} is a layout marker on a scalar; it does not mean ${tile.match(/\d+/)?.[0] || ''} logical elements.`
              : `${tile} describes the array's physical tiling.`
          }
        ]
      : []),
    { key: 'space', label: 'Memory space', text: node ? placement(shape, status, node) : shapeMemoryText(shape) }
  ];
}

function whileState(context: GuideContext) {
  const { module, computation } = context || {};
  if (!module || !computation) return null;
  for (const caller of module.computations)
    for (const instruction of caller.nodes) {
      if (instruction.op !== 'while' || !Object.values(instruction.calls).includes(computation.name)) continue;
      const initial = caller.byName.get(instruction.operands[0]);
      const body = module.byName.get(instruction.calls.body);
      const next = body?.nodes.find(node => node.root && node.op === 'tuple');
      return { initial: initial?.op === 'tuple' ? initial.operands : [], next: next?.operands || [] };
    }
  return null;
}

const operandList = (node: HloNode) => node.operands.map(name => `%${name}`).join(', ');

export function instructionGuide(node: HloNode, context: GuideContext = {}): InstructionGuide {
  const raw = node.raw;
  const prefix = /^(ROOT\s+)?(%?[\w.-]+)(\s*=\s*)/.exec(raw);
  const opAt = prefix ? raw.indexOf(` ${node.op}(`, prefix[0].length) : -1;
  if (!prefix || opAt < 0) return { html: escapeHtml(raw), parts: [], resultGroup: null };
  const typeText = raw.slice(prefix[0].length, opAt);
  const argsStart = opAt + node.op.length + 2;
  let depth = 1,
    argsEnd = argsStart;
  for (; argsEnd < raw.length; argsEnd++) {
    if (raw[argsEnd] === '(') depth++;
    else if (raw[argsEnd] === ')' && --depth === 0) break;
  }
  const args = raw.slice(argsStart, argsEnd);
  const suffix = raw.slice(argsEnd + 1);
  const tupleType = typeText.startsWith('(') && typeText.endsWith(')');
  const copyStart = node.op === 'copy-start' && tupleType;
  const opPart: GuidePart = { key: 'op', label: 'Operation', text: opDescription(node) };
  const status = context.module && context.computation ? bufferStatus(context.module, context.computation, node) : null;
  const hasBuffer = status === null || status === 'buffer'; // only then does S(n) say where the value is
  let resultType: string,
    parts: GuidePart[],
    tupleGroup: { rawType: string; slots: (TypeSlot & GuidePart)[] } | null = null;
  if (copyStart) {
    const slots = splitTuple(typeText);
    const outputShape = /([a-z][\w]*)\[([^\]]*)\]/.exec(slots[0]?.body || '');
    const order = /\{(\d+(?:,\d+)*):/.exec(slots[0]?.body || '')?.[1];
    const tile = /T(?:\([^)]*\))+/.exec(slots[0]?.body || '')?.[0];
    const tileLevels = tile && [...tile.matchAll(/\(([^)]*)\)/g)].map(match => match[1].replaceAll(',', '×'));
    resultType =
      `<span class="hlo-type part-tuple" data-part="tuple" tabindex="0">` +
      mark('tuple', '(') +
      slots
        .map(
          ({ body, separator }, index) =>
            `<span class="hlo-slot part-${['dest', 'source', 'context'][index] || 'shape'}" data-part="${['dest', 'source', 'context'][index] || 'shape'}">${shapeHtml(body)}</span>${escapeHtml(separator)}`
        )
        .join('') +
      mark('tuple', ')') +
      '</span>';
    parts = [
      { key: 'name', label: 'Result name', text: `${prefix[2]} names this instruction's result; later instructions refer to it.` },
      {
        key: 'tuple',
        label: 'Three results',
        text: 'The parentheses mean the result is a tuple: the copy destination, the copy source and a context, all used by copy-done.'
      },
      {
        key: 'dest',
        label: 'Element 0 · destination',
        text: `The buffer that holds the data after the copy; here it is in ${memoryLabel(slots[0]?.body || '')}.`
      },
      {
        key: 'source',
        label: 'Element 1 · source',
        text: `The buffer the data is copied from; here it is in ${memoryLabel(slots[1]?.body || '')}.`
      },
      {
        key: 'context',
        label: 'Element 2 · context',
        text: `${(slots[2]?.body || 'u32[]').trim()} is the sync flag of the asynchronous copy; copy-done waits on it. It is in ${memoryLabel(slots[2]?.body || '')}.`
      },
      {
        key: 'shape',
        label: 'Type and shape',
        text: outputShape
          ? `${outputShape[1]}[${outputShape[2]}] is the element type and logical shape of the copied tensor; u32[] is a scalar.`
          : 'The element type and logical shape of the copied tensor; u32[] is a scalar.'
      },
      {
        key: 'order',
        label: 'Dimension order',
        text: order
          ? `${order} is the minor-to-major physical order; the dimension listed first varies fastest.`
          : 'The numbers in braces give the minor-to-major physical order.'
      },
      {
        key: 'tile',
        label: 'Tiling',
        text:
          tileLevels?.length === 2
            ? `${tile} is two-level tiling: outer ${tileLevels[0]}, inner ${tileLevels[1]}; it may add padding.`
            : tile
              ? `${tile} describes the tiled physical layout and any padding.`
              : 'T(...) describes the tiled physical layout and any padding.'
      },
      {
        key: 'space',
        label: 'Memory space',
        text: 'S(n) is the memory space number. On TPU, S(1) is VMEM, S(2) is SFLAG (sync flags) and S(6) is SMEM (scalar memory); no S(n) means the default S(0), which is HBM.'
      },
      opPart,
      {
        key: 'operand',
        label: 'Input',
        text: `${operandList(node) || 'The operand in parentheses'} provides the data to copy; use Direct inputs below to jump to it.`
      }
    ];
    parts = parts.filter(
      part => (part.key !== 'tile' || tile) && (part.key !== 'order' || order) && (part.key !== 'space' || /S\(/.test(typeText))
    );
  } else if (tupleType) {
    const slots = splitTuple(typeText);
    const state = node.op === 'parameter' ? whileState(context) : null;
    const used = new Map<number, string[]>();
    if (context.computation) {
      for (const name of node.users) {
        const consumer = context.computation.byName.get(name);
        if (consumer?.op === 'get-tuple-element' && consumer.index !== null) {
          if (!used.has(consumer.index)) used.set(consumer.index, []);
          used.get(consumer.index)!.push(name);
        }
      }
    }
    resultType =
      `<span class="hlo-type part-tuple" data-part="tuple" tabindex="0">` +
      mark('tuple', '(') +
      slots
        .map(
          ({ body, separator }, index) =>
            `<span class="hlo-slot part-slot-${index % 4}" data-part="slot-${index}">${shapeHtml(body)}</span>${escapeHtml(separator)}`
        )
        .join('') +
      mark('tuple', ')') +
      '</span>';
    const slotParts = slots.map(({ body }, index) => {
      const origin = state
        ? `On the first iteration it comes from ${state.initial[index] ? `%${state.initial[index]}` : 'the initial state'}; later iterations take ${state.next[index] ? `%${state.next[index]}` : 'the matching element'} returned by the body.`
        : node.op === 'tuple' && node.operands[index]
          ? `This element comes from %${node.operands[index]}.`
          : '';
      const readers = used.has(index)
        ? `This computation reads it through ${used
            .get(index)!
            .map(name => `%${name}`)
            .join(', ')}.`
        : node.op === 'parameter' && context.computation
          ? 'This computation does not read this element directly.'
          : '';
      const details = [
        ...typeDetails(body, status, node),
        ...(origin ? [{ label: node.op === 'parameter' ? 'Loop state source' : 'Data source', text: origin }] : []),
        ...(readers ? [{ label: 'Use in this computation', text: readers }] : [])
      ];
      return {
        key: `slot-${index}`,
        label: `Element ${index}`,
        raw: body.trim(),
        text: [`${shapeDescription(body, hasBuffer)}.`, origin, readers].filter(Boolean).join(' '),
        details
      };
    });
    tupleGroup = { rawType: typeText, slots: slotParts };
    parts = [
      { key: 'name', label: 'Result name', text: `${prefix[2]} names this instruction's result.` },
      {
        key: 'tuple',
        label: `${slots.length}-element tuple`,
        text: `The outer parentheses make a tuple with ${slots.length} elements. Each element has its own type, shape and layout; elements are numbered from 0.`
      },
      ...slotParts,
      {
        key: 'shape',
        label: 'Type and logical shape',
        text: 'Each tuple element declares its own element type and logical shape; expand an element to see it.'
      },
      {
        key: 'order',
        label: 'Physical dimension order',
        text: 'The numbers in braces give the minor-to-major order; the dimensions listed first vary fastest.'
      },
      {
        key: 'tile',
        label: 'T(...) tiling',
        text: /\bs32\[\]\{[^}]*T\(128\)/.test(typeText)
          ? 'T(128) is a layout marker on scalars; it does not turn a scalar into 128 logical elements. Other T(...) describe the physical tiling of arrays.'
          : 'T(...) describes physical tiling; it does not change the logical shape.'
      },
      { key: 'space', label: 'S(...) memory space', text: 'S(n) gives the memory space; no S(n) means the default space.' },
      node.op === 'parameter'
        ? { ...opPart, text: 'parameter declares one input of the current computation; it creates no new data here.' }
        : opPart
    ];
    if (node.op === 'parameter')
      parts.push({
        key: 'parameter-index',
        label: 'Parameter number',
        text: `parameter(${args}) is input ${args} of the current computation; the whole tuple is one parameter, not ${slots.length} separate ones.`
      });
    if (node.op !== 'parameter' && node.operands.length)
      parts.push({
        key: 'operand',
        label: 'Inputs',
        text: `This instruction depends directly on ${operandList(node)}; the incoming edges in the graph are these references.`
      });
    if (!/\{\d+(?:,\d+)*[:}]/.test(typeText)) parts = parts.filter(part => part.key !== 'order');
    if (!/T\(/.test(typeText)) parts = parts.filter(part => part.key !== 'tile');
    if (!/S\(/.test(typeText)) parts = parts.filter(part => part.key !== 'space');
  } else {
    resultType = shapeHtml(typeText);
    const details = typeDetails(typeText, status, node);
    const detail = (key: string) => details.find(item => item.key === key);
    parts = [
      { key: 'name', label: 'Result name', text: `${prefix[2]} names this instruction's result.` },
      {
        key: 'shape',
        label: 'Result type',
        text: `${shapeDescription(typeText, hasBuffer)}. After the equals sign come the result's data type, logical shape and optional physical layout.`
      },
      ...(/\{\d+(?:,\d+)*[:}]/.test(typeText) ? [{ key: 'order', label: 'Dimension order', text: detail('order')!.text }] : []),
      ...(/T\(/.test(typeText) ? [{ key: 'tile', label: 'Physical tiling', text: detail('tile')!.text }] : []),
      ...(/S\(/.test(typeText) ? [{ key: 'space', label: 'Memory space', text: detail('space')!.text }] : []),
      opPart,
      {
        key: 'operand',
        label: 'Inputs',
        text: `${operandList(node)} in parentheses ${node.operands.length > 1 ? 'are the instructions this one depends on, in operand order' : 'is the instruction this one depends on'}.${/#\d/.test(args) ? ' %t#N is XLA dump syntax for element N of tuple %t (like get-tuple-element).' : ''}`
      }
    ];
    if (node.op === 'parameter')
      parts.push({
        key: 'parameter-index',
        label: 'Parameter number',
        text: `parameter(${args}) is input ${args} of the current computation.`
      });
    if (node.op === 'constant')
      parts.push({
        key: 'literal',
        label: 'Constant value',
        text: `${args.length > 80 ? `${args.slice(0, 77)}…` : args} in parentheses is the value given directly by this instruction; the result type sets its data type and shape, and it depends on no other instruction.`
      });
    if (!node.operands.length || node.op === 'constant' || node.op === 'parameter') parts = parts.filter(part => part.key !== 'operand');
  }
  if (/\/\*[^*]*\*\//.test(args))
    parts.push({
      key: 'operand-index',
      label: 'Index comment',
      text: '/*index=N*/ is a comment the HLO printer inserts every few operands to number the operand that follows (from 0); it is not an operand.'
    });
  const suffixParts = suffixHtml(suffix);
  for (const { name, value } of suffixParts.attributes) {
    const explanation = explainAttribute(name, value, { node, module: context.module, computation: context.computation });
    if (!parts.some(part => part.key === explanation.key)) parts.push(explanation);
  }
  const resultGroup: ResultGroup = tupleType
    ? {
        kind: 'tuple',
        rawType: typeText,
        slots:
          tupleGroup?.slots ||
          splitTuple(typeText).map(({ body }, index) => ({
            key: ['dest', 'source', 'context'][index] || `slot-${index}`,
            label: `Element ${index}${copyStart ? [' · destination', ' · source', ' · context'][index] || '' : ''}`,
            raw: body.trim(),
            details: typeDetails(body, status, node)
          }))
      }
    : { kind: 'array', rawType: typeText, details: typeDetails(typeText, status, node) };
  const argumentsHtml =
    node.op === 'constant'
      ? mark('literal', `(${args})`)
      : '(' + (node.op === 'parameter' ? mark('parameter-index', args) : operandHtml(args)) + ')';
  const html =
    escapeHtml(prefix[1] || '') +
    mark('name', prefix[2]) +
    escapeHtml(prefix[3]) +
    resultType +
    ' ' +
    mark('op', node.op) +
    argumentsHtml +
    suffixParts.html;
  const metadata = extractHloMetadata(suffix);
  const compactHtml = compactEncodedBody(metadata ? html.replace(mark('metadata', metadata.raw), mark('metadata', 'metadata={…}')) : html);
  return { html, compactHtml: compactHtml === html ? undefined : compactHtml, parts, resultGroup };
}

export function layoutDiagram(node: HloNode): string | null {
  if (!['copy-start', 'parameter'].includes(node.op) || !/\{1,0:T\(8,128\)\(2,1\)/.test(node.type)) return null;
  const slots = splitTuple(node.type);
  if (!slots.length) return null;
  const location = memorySpaceLabel;
  const copyStart = node.op === 'copy-start';
  const matrixSlots = slots.map((slot, index) => ({ slot, index })).filter(({ slot }) => /\{1,0:T\(8,128\)\(2,1\)/.test(slot.body));
  const reference = matrixSlots[0]?.slot.body;
  if (!reference) return null;
  const locationText = copyStart
    ? `${location(memorySpace(slots[1].body))} S(${memorySpace(slots[1].body)}) → ${location(memorySpace(slots[0].body))} S(${memorySpace(slots[0].body)})`
    : `Element${matrixSlots.length > 1 ? 's' : ''} ${matrixSlots.map(({ index }) => index).join(', ')} · ${location(memorySpace(reference))} S(${memorySpace(reference)})`;
  const dimensions = /\[(\d+),(\d+)\]/.exec(reference);
  const tileCount = dimensions ? Math.ceil(Number(dimensions[1]) / 8) * Math.ceil(Number(dimensions[2]) / 128) : null;
  const cells = [];
  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 8; col++) {
      const offset = Math.floor(row / 2) * 256 + col * 2 + (row % 2);
      const fill = row < 2 ? (col < 4 ? '#e5f0fa' : '#fff0df') : col < 4 ? '#e9f4e8' : '#f7e9f1';
      cells.push(
        `<rect x="${34 + col * 32}" y="${34 + row * 32}" width="32" height="32" fill="${fill}" stroke="#d6dfea"/><text x="${50 + col * 32}" y="${55 + row * 32}" text-anchor="middle" class="layout-number">${offset}</text>`
      );
    }
  }
  for (let row = 0; row < 4; row += 2)
    for (let col = 0; col < 8; col++)
      cells.push(
        `<rect x="${35 + col * 32}" y="${35 + row * 32}" width="30" height="62" rx="4" fill="none" stroke="#df806f" stroke-width="1.5"/>`
      );
  for (let row = 0; row < 4; row++) cells.push(`<text x="26" y="${55 + row * 32}" text-anchor="end" class="layout-axis">r${row}</text>`);
  for (let col = 0; col < 8; col++) cells.push(`<text x="${50 + col * 32}" y="27" text-anchor="middle" class="layout-axis">c${col}</text>`);
  return `<div class="layout-diagram"><div class="layout-diagram-head"><strong>Array layout · T(8,128)(2,1)</strong><span>${locationText}</span></div><svg viewBox="0 0 310 174" role="img" aria-label="First four rows and eight columns of an eight by 128 tile; numbers show physical order and red outlines show two by one inner tiles"><title>8×128 tile, with 2×1 inner tiles</title>${cells.join('')}</svg><p>The top-left 4 rows × 8 columns of the matrix; a full outer tile is 8×128. Numbers are physical offsets within the tile; red outlines are the 2×1 inner tiles. ${tileCount === null ? 'The full array is made of many such tiles.' : `Each such matrix has ${tileCount} outer tiles.`}</p></div>`;
}
