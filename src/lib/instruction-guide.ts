import type { Computation, GuidePart, HloModule, HloNode, InstructionGuide, ResultGroup, TypeDetail, TypeSlot } from './types';
import { extractHloMetadata } from './metadata.ts';
import { closingParen, splitTopLevel } from './parser.ts';
import { attributeKey, explainAttribute, opDescription } from './hlo-reference.ts';
import { bufferStatus, tupleElements, type BufferStatus } from './memory-location.ts';
import { memorySpaceLabel, shapeMemorySpace, shapeMemoryText } from './memory-space.ts';

interface GuideContext {
  computation?: Computation;
  module?: HloModule;
}

// ───── HTML segments ─────

const HTML_ESCAPES: Record<string, string> = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
const escapeHtml = (value: unknown) => String(value).replace(/[&<>"']/g, c => HTML_ESCAPES[c]);

const STYLED_KEYS =
  /^(name|tuple|dest|source|context|shape|order|tile|space|op|operand|priority|literal|parameter-index|target|constraints|metadata|backend|dimensions|tuple-index|fusion-kind|called-computation|slot-\d+)$/;

// Keys without a CSS color get a stable hue so neighbouring segments stay distinguishable.
const hue = (key: string) => [...key].reduce((h, c) => (h * 31 + c.charCodeAt(0)) % 360, 7);

export const partColor = (key: string) => (STYLED_KEYS.test(key) ? undefined : `hsl(${hue(key)} 45% 40%)`);

function openPart(key: string) {
  const color = partColor(key);
  const style = color ? ` style="--part:${color}"` : '';
  return `<span class="hlo-part part-${key}" data-part="${key}" tabindex="0"${style}>`;
}

const markHtml = (key: string, html: string) => `${openPart(key)}${html}</span>`;
const mark = (key: string, text: string) => markHtml(key, escapeHtml(text));

// ───── Type html ─────

const ARRAY_SHAPE = /([a-z][\w]*)\[([^\]]*)\]/;
const TILE = /T(?:\([^)]*\))+/;
const COPY_SLOT_KEYS = ['dest', 'source', 'context'];

function layoutKey(token: string) {
  if (token.startsWith('T')) return 'tile';
  if (token.startsWith('S')) return 'space';
  return 'order';
}

function layoutHtml(layout: string) {
  let output = '',
    position = 0;

  for (const match of layout.matchAll(/\d+(?:,\d+)*|T(?:\([^)]*\))+|S\(\d+\)/g)) {
    output += escapeHtml(layout.slice(position, match.index));
    output += mark(layoutKey(match[0]), match[0]);
    position = match.index + match[0].length;
  }

  return output + escapeHtml(layout.slice(position));
}

function shapeHtml(shape: string) {
  const match = /^(\s*)([a-z][\w]*\[[^\]]*\])(\{[^}]*\})?(.*)$/s.exec(shape);
  if (!match) return escapeHtml(shape);

  const [, lead, array, layout, rest] = match;
  const type = escapeHtml(array) + (layout ? `{${layoutHtml(layout.slice(1, -1))}}` : '');
  return escapeHtml(lead) + `<span class="hlo-type part-shape" data-part="shape" tabindex="0">${type}</span>` + escapeHtml(rest);
}

// `slotAttributes` gives the attributes of the span around each element.
function tupleTypeHtml(slots: string[], slotAttributes: (index: number) => string) {
  const elements = slots
    .map((slot, index) => `<span ${slotAttributes(index)}>${shapeHtml(slot)}</span>${index < slots.length - 1 ? ',' : ''}`)
    .join('');
  return `<span class="hlo-type part-tuple" data-part="tuple" tabindex="0">${mark('tuple', '(')}${elements}${mark('tuple', ')')}</span>`;
}

// ───── Operands and attributes html ─────

// Wraps the "dma_priority": N inside backend_config in its own nested segment.
function backendConfigHtml(text: string, attributes: { name: string; value: string }[]) {
  const priority = /"dma_priority"\s*:\s*"?\d+"?/.exec(text);
  if (!priority) return escapeHtml(text);

  attributes.push({ name: 'dma_priority', value: /\d+/.exec(priority[0])![0] });
  const before = text.slice(0, priority.index);
  const after = text.slice(priority.index + priority[0].length);
  return escapeHtml(before) + mark('priority', priority[0]) + escapeHtml(after);
}

// Attributes after the operand list. Each top-level `name=value` becomes one explained segment.
function suffixHtml(suffix: string) {
  const pieces = splitTopLevel(suffix);
  const attributes: { name: string; value: string }[] = [];

  const html = pieces
    .map((piece, index) => {
      const separator = index < pieces.length - 1 ? ',' : '';
      const match = /^(\s*)([\w-]+)=([\s\S]*?)(\s*)$/.exec(piece);
      if (!match) return escapeHtml(piece) + separator;

      const [, lead, name, value, trail] = match;
      attributes.push({ name, value });
      const text = `${name}=${value}`;
      const inner = name === 'backend_config' ? backendConfigHtml(text, attributes) : escapeHtml(text);
      return escapeHtml(lead) + markHtml(attributeKey(name), inner) + escapeHtml(trail) + separator;
    })
    .join('');

  return { html, attributes };
}

// Collapses a long base64 kernel "body" in backend_config to its length.
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

      const [, head, operand, trail] = match;
      const headHtml = head
        .replace(/\/\*[^*]*\*\//g, comment => `\u0000${comment}\u0000`)
        .split('\u0000')
        .map(chunk => (chunk.startsWith('/*') ? mark('operand-index', chunk) : escapeHtml(chunk)))
        .join('');
      return headHtml + mark('operand', operand) + escapeHtml(trail) + separator;
    })
    .join('');
}

// ───── Type explanations ─────

function memoryLabel(shape: string) {
  const { space } = shapeMemorySpace(shape);
  return space === 0 ? 'HBM (S(0) omitted)' : `${memorySpaceLabel(space)} (S(${space}))`;
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
  const { space, explicit } = shapeMemorySpace(shape);
  const marked = explicit ? `the layout says S(${space})` : 'the layout has no S(n)';

  switch (status) {
    case 'fusion': {
      if (node.op === 'parameter')
        return `A fusion parameter: it is the operand passed in by the caller and takes no buffer of its own (${marked}).`;
      const rootNote = node.root
        ? ' Its result is the output of this fusion; see the fusion instruction that calls it for the actual location.'
        : '';
      return `No buffer of its own: this instruction is computed inside a fusion kernel and its result is not written back to HBM or VMEM by itself (${marked}, which is not a storage location).${rootNote}`;
    }
    case 'thread-local':
      return `A value inside a reducer or comparator; XLA gives it thread-local storage, not HBM or VMEM (${marked}).`;
    case 'unassigned':
      return `Lowered (unscheduled) HLO has not been through memory-space assignment yet (${marked}); see the compiled HLO for the actual location.`;
    default:
      return `${shapeMemoryText(shape)}${status ? '' : ' (assuming this instruction has a buffer of its own)'}`;
  }
}

function shapeDescription(shape: string, includeMemory = true) {
  const match = ARRAY_SHAPE.exec(shape);
  if (!match) return "see the raw HLO for this element's type";

  const [, elementType, dims] = match;
  const dtype = DTYPES[elementType] || elementType;
  let logical = '';
  if (elementType !== 'token') logical = dims ? `, a ${dims.replaceAll(',', '×')} array` : ', a scalar (logical shape [])';
  const memory = includeMemory ? `; ${memoryLabel(shape)}` : '';

  return `${elementType}[${dims}]: ${dtype}${logical}${memory}`;
}

function typeDetails(shape: string, status: BufferStatus | null = null, node?: HloNode): TypeDetail[] {
  const order = /\{(\d+(?:,\d+)*)(?=[:}])/.exec(shape)?.[1];
  const tile = TILE.exec(shape)?.[0];
  const details: TypeDetail[] = [{ key: 'shape', label: 'Element type and logical shape', text: shapeDescription(shape, false) }];

  if (order)
    details.push({ key: 'order', label: 'Dimension order', text: `${order}: minor-to-major; the dimensions listed first vary fastest.` });

  if (tile) {
    const text = shape.includes('[]')
      ? `${tile} is a layout marker on a scalar; it does not mean ${tile.match(/\d+/)?.[0] || ''} logical elements.`
      : `${tile} describes the array's physical tiling.`;
    details.push({ key: 'tile', label: 'Physical tiling', text });
  }

  details.push({ key: 'space', label: 'Memory space', text: node ? placement(shape, status, node) : shapeMemoryText(shape) });
  return details;
}

// ───── Instruction guide ─────

// The pieces of one instruction line that every kind of result type needs.
interface GuideInput {
  node: HloNode;
  context: GuideContext;
  name: string;
  typeText: string;
  args: string;
  status: BufferStatus | null;
  hasBuffer: boolean;
  opPart: GuidePart;
}

interface TypeGuide {
  resultType: string;
  parts: GuidePart[];
  slots?: TypeSlot[];
}

const operandList = (node: HloNode) => node.operands.map(name => `%${name}`).join(', ');

// Initial and next-iteration operands of the while loop whose body or condition is this computation.
function whileState(context: GuideContext) {
  const { module, computation } = context;
  if (!module || !computation) return null;

  for (const caller of module.computations) {
    for (const instruction of caller.nodes) {
      if (instruction.op !== 'while' || !Object.values(instruction.calls).includes(computation.name)) continue;
      const initial = caller.byName.get(instruction.operands[0]);
      const body = module.byName.get(instruction.calls.body);
      const next = body?.nodes.find(node => node.root && node.op === 'tuple');
      return { initial: initial?.op === 'tuple' ? initial.operands : [], next: next?.operands || [] };
    }
  }

  return null;
}

// Tuple index -> names of the get-tuple-element users that read it.
function tupleReaders(node: HloNode, computation?: Computation) {
  const readers = new Map<number, string[]>();
  if (!computation) return readers;

  for (const name of node.users) {
    const consumer = computation.byName.get(name);
    if (consumer?.op !== 'get-tuple-element' || consumer.index === null) continue;
    if (!readers.has(consumer.index)) readers.set(consumer.index, []);
    readers.get(consumer.index)!.push(name);
  }

  return readers;
}

function copyStartGuide({ node, name, typeText, status, opPart }: GuideInput): TypeGuide {
  const slots = tupleElements(typeText);
  const [dest = '', source = '', context = ''] = slots;
  const outputShape = ARRAY_SHAPE.exec(dest);
  const order = /\{(\d+(?:,\d+)*):/.exec(dest)?.[1];
  const tile = TILE.exec(dest)?.[0];

  const resultType = tupleTypeHtml(slots, index => {
    const key = COPY_SLOT_KEYS[index] || 'shape';
    return `class="hlo-slot part-${key}" data-part="${key}"`;
  });

  const shapeText = outputShape
    ? `${outputShape[1]}[${outputShape[2]}] is the element type and logical shape of the copied tensor; u32[] is a scalar.`
    : 'The element type and logical shape of the copied tensor; u32[] is a scalar.';

  const parts: GuidePart[] = [
    { key: 'name', label: 'Result name', text: `${name} names this instruction's result; later instructions refer to it.` },
    {
      key: 'tuple',
      label: 'Three results',
      text: 'The parentheses mean the result is a tuple: the copy destination, the copy source and a context, all used by copy-done.'
    },
    {
      key: 'dest',
      label: 'Element 0 · destination',
      text: `The buffer that holds the data after the copy; here it is in ${memoryLabel(dest)}.`
    },
    {
      key: 'source',
      label: 'Element 1 · source',
      text: `The buffer the data is copied from; here it is in ${memoryLabel(source)}.`
    },
    {
      key: 'context',
      label: 'Element 2 · context',
      text: `${(context || 'u32[]').trim()} is the sync flag of the asynchronous copy; copy-done waits on it. It is in ${memoryLabel(context)}.`
    },
    { key: 'shape', label: 'Type and shape', text: shapeText }
  ];

  if (order)
    parts.push({
      key: 'order',
      label: 'Dimension order',
      text: `${order} is the minor-to-major physical order; the dimension listed first varies fastest.`
    });

  if (tile) {
    const levels = [...tile.matchAll(/\(([^)]*)\)/g)].map(match => match[1].replaceAll(',', '×'));
    const text =
      levels.length === 2
        ? `${tile} is two-level tiling: outer ${levels[0]}, inner ${levels[1]}; it may add padding.`
        : `${tile} describes the tiled physical layout and any padding.`;
    parts.push({ key: 'tile', label: 'Tiling', text });
  }

  if (/S\(/.test(typeText))
    parts.push({
      key: 'space',
      label: 'Memory space',
      text: 'S(n) is the memory space number. On TPU, S(1) is VMEM, S(2) is SFLAG (sync flags) and S(6) is SMEM (scalar memory); no S(n) means the default S(0), which is HBM.'
    });

  parts.push(opPart, {
    key: 'operand',
    label: 'Input',
    text: `${operandList(node) || 'The operand in parentheses'} provides the data to copy; use Direct inputs below to jump to it.`
  });

  const slotSuffixes = [' · destination', ' · source', ' · context'];
  const typeSlots = slots.map((slot, index) => ({
    key: COPY_SLOT_KEYS[index] || `slot-${index}`,
    label: `Element ${index}${slotSuffixes[index] || ''}`,
    raw: slot.trim(),
    details: typeDetails(slot, status, node)
  }));

  return { resultType, parts, slots: typeSlots };
}

// Where a tuple element's value comes from: the loop state for a while parameter, or the tuple operand.
function slotOrigin(node: HloNode, state: ReturnType<typeof whileState>, index: number) {
  if (state) {
    const initial = state.initial[index] ? `%${state.initial[index]}` : 'the initial state';
    const next = state.next[index] ? `%${state.next[index]}` : 'the matching element';
    return `On the first iteration it comes from ${initial}; later iterations take ${next} returned by the body.`;
  }
  if (node.op === 'tuple' && node.operands[index]) return `This element comes from %${node.operands[index]}.`;
  return '';
}

function tupleGuide({ node, context, name, typeText, args, status, hasBuffer, opPart }: GuideInput): TypeGuide {
  const slots = tupleElements(typeText);
  const state = node.op === 'parameter' ? whileState(context) : null;
  const readers = tupleReaders(node, context.computation);

  const resultType = tupleTypeHtml(slots, index => `class="hlo-slot part-slot-${index % 4}" data-part="slot-${index}"`);

  const slotParts = slots.map((slot, index) => {
    const origin = slotOrigin(node, state, index);

    let reads = '';
    if (readers.has(index)) {
      const names = readers
        .get(index)!
        .map(reader => `%${reader}`)
        .join(', ');
      reads = `This computation reads it through ${names}.`;
    } else if (node.op === 'parameter' && context.computation) {
      reads = 'This computation does not read this element directly.';
    }

    const details: TypeDetail[] = typeDetails(slot, status, node);
    if (origin) details.push({ label: node.op === 'parameter' ? 'Loop state source' : 'Data source', text: origin });
    if (reads) details.push({ label: 'Use in this computation', text: reads });

    return {
      key: `slot-${index}`,
      label: `Element ${index}`,
      raw: slot.trim(),
      text: [`${shapeDescription(slot, hasBuffer)}.`, origin, reads].filter(Boolean).join(' '),
      details
    };
  });

  const tileText = /\bs32\[\]\{[^}]*T\(128\)/.test(typeText)
    ? 'T(128) is a layout marker on scalars; it does not turn a scalar into 128 logical elements. Other T(...) describe the physical tiling of arrays.'
    : 'T(...) describes physical tiling; it does not change the logical shape.';

  const parts: GuidePart[] = [
    { key: 'name', label: 'Result name', text: `${name} names this instruction's result.` },
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
    }
  ];

  if (/\{\d+(?:,\d+)*[:}]/.test(typeText))
    parts.push({
      key: 'order',
      label: 'Physical dimension order',
      text: 'The numbers in braces give the minor-to-major order; the dimensions listed first vary fastest.'
    });
  if (/T\(/.test(typeText)) parts.push({ key: 'tile', label: 'T(...) tiling', text: tileText });
  if (/S\(/.test(typeText))
    parts.push({ key: 'space', label: 'S(...) memory space', text: 'S(n) gives the memory space; no S(n) means the default space.' });

  if (node.op === 'parameter') {
    parts.push(
      { ...opPart, text: 'parameter declares one input of the current computation; it creates no new data here.' },
      {
        key: 'parameter-index',
        label: 'Parameter number',
        text: `parameter(${args}) is input ${args} of the current computation; the whole tuple is one parameter, not ${slots.length} separate ones.`
      }
    );
  } else {
    parts.push(opPart);
    if (node.operands.length)
      parts.push({
        key: 'operand',
        label: 'Inputs',
        text: `This instruction depends directly on ${operandList(node)}; the incoming edges in the graph are these references.`
      });
  }

  return { resultType, parts, slots: slotParts };
}

function arrayGuide({ node, name, typeText, args, status, hasBuffer, opPart }: GuideInput): TypeGuide {
  const details = typeDetails(typeText, status, node);
  const detail = (key: string) => details.find(item => item.key === key)!.text;

  const parts: GuidePart[] = [
    { key: 'name', label: 'Result name', text: `${name} names this instruction's result.` },
    {
      key: 'shape',
      label: 'Result type',
      text: `${shapeDescription(typeText, hasBuffer)}. After the equals sign come the result's data type, logical shape and optional physical layout.`
    }
  ];
  if (/\{\d+(?:,\d+)*[:}]/.test(typeText)) parts.push({ key: 'order', label: 'Dimension order', text: detail('order') });
  if (/T\(/.test(typeText)) parts.push({ key: 'tile', label: 'Physical tiling', text: detail('tile') });
  if (/S\(/.test(typeText)) parts.push({ key: 'space', label: 'Memory space', text: detail('space') });
  parts.push(opPart);

  const takesOperands = node.op !== 'constant' && node.op !== 'parameter';
  if (takesOperands && node.operands.length) {
    const dependsOn =
      node.operands.length > 1 ? 'are the instructions this one depends on, in operand order' : 'is the instruction this one depends on';
    const projection = /#\d/.test(args) ? ' %t#N is XLA dump syntax for element N of tuple %t (like get-tuple-element).' : '';
    parts.push({ key: 'operand', label: 'Inputs', text: `${operandList(node)} in parentheses ${dependsOn}.${projection}` });
  }

  if (node.op === 'parameter')
    parts.push({
      key: 'parameter-index',
      label: 'Parameter number',
      text: `parameter(${args}) is input ${args} of the current computation.`
    });

  if (node.op === 'constant') {
    const literal = args.length > 80 ? `${args.slice(0, 77)}…` : args;
    parts.push({
      key: 'literal',
      label: 'Constant value',
      text: `${literal} in parentheses is the value given directly by this instruction; the result type sets its data type and shape, and it depends on no other instruction.`
    });
  }

  return { resultType: shapeHtml(typeText), parts };
}

// Splits "ROOT %name = <type> op(<args>)<suffix>" at the op's matching closing parenthesis.
function splitInstruction(node: HloNode) {
  const raw = node.raw;
  const prefix = /^(ROOT\s+)?(%?[\w.-]+)(\s*=\s*)/.exec(raw);
  const opAt = prefix ? raw.indexOf(` ${node.op}(`, prefix[0].length) : -1;
  if (!prefix || opAt < 0) return null;

  const argsStart = opAt + node.op.length + 2;
  const argsEnd = closingParen(raw, argsStart - 1);
  if (argsEnd < 0) return null;

  const [, root = '', name, equals] = prefix;
  return {
    root,
    name,
    equals,
    typeText: raw.slice(prefix[0].length, opAt),
    args: raw.slice(argsStart, argsEnd),
    suffix: raw.slice(argsEnd + 1)
  };
}

export function instructionGuide(node: HloNode, context: GuideContext = {}): InstructionGuide {
  const line = splitInstruction(node);
  if (!line) return { html: escapeHtml(node.raw), parts: [], resultGroup: null };

  const { root, name, equals, typeText, args, suffix } = line;
  const status = context.module && context.computation ? bufferStatus(context.module, context.computation, node) : null;
  const input: GuideInput = {
    node,
    context,
    name,
    typeText,
    args,
    status,
    hasBuffer: status === null || status === 'buffer', // only then does S(n) say where the value is
    opPart: { key: 'op', label: 'Operation', text: opDescription(node) }
  };

  const isTuple = typeText.startsWith('(') && typeText.endsWith(')');
  let guide: TypeGuide;
  if (isTuple && node.op === 'copy-start') guide = copyStartGuide(input);
  else if (isTuple) guide = tupleGuide(input);
  else guide = arrayGuide(input);
  const { resultType, parts } = guide;

  if (/\/\*[^*]*\*\//.test(args))
    parts.push({
      key: 'operand-index',
      label: 'Index comment',
      text: '/*index=N*/ is a comment the HLO printer inserts every few operands to number the operand that follows (from 0); it is not an operand.'
    });

  const suffixParts = suffixHtml(suffix);
  for (const { name: attribute, value } of suffixParts.attributes) {
    const explanation = explainAttribute(attribute, value, { node, module: context.module, computation: context.computation });
    if (!parts.some(part => part.key === explanation.key)) parts.push(explanation);
  }

  const resultGroup: ResultGroup = guide.slots
    ? { kind: 'tuple', rawType: typeText, slots: guide.slots }
    : { kind: 'array', rawType: typeText, details: typeDetails(typeText, status, node) };

  let argumentsHtml: string;
  if (node.op === 'constant') argumentsHtml = mark('literal', `(${args})`);
  else if (node.op === 'parameter') argumentsHtml = `(${mark('parameter-index', args)})`;
  else argumentsHtml = `(${operandHtml(args)})`;

  const html =
    escapeHtml(root) + mark('name', name) + escapeHtml(equals) + resultType + ' ' + mark('op', node.op) + argumentsHtml + suffixParts.html;

  const metadata = extractHloMetadata(suffix);
  const withoutMetadata = metadata ? html.replace(mark('metadata', metadata.raw), mark('metadata', 'metadata={…}')) : html;
  const compactHtml = compactEncodedBody(withoutMetadata);

  return { html, compactHtml: compactHtml === html ? undefined : compactHtml, parts, resultGroup };
}

// ───── Layout diagram ─────

const TILED_MATRIX = /\{1,0:T\(8,128\)\(2,1\)/;
const CELL = 32;

function cellFill(row: number, col: number) {
  if (row < 2) return col < 4 ? '#e5f0fa' : '#fff0df';
  return col < 4 ? '#e9f4e8' : '#f7e9f1';
}

// Cells, inner-tile outlines and axis labels for the top-left 4×8 corner of an 8×128 tile.
function tileCornerSvg() {
  const cells = [];

  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 8; col++) {
      const offset = Math.floor(row / 2) * 256 + col * 2 + (row % 2);
      cells.push(
        `<rect x="${34 + col * CELL}" y="${34 + row * CELL}" width="32" height="32" fill="${cellFill(row, col)}" stroke="#d6dfea"/><text x="${50 + col * CELL}" y="${55 + row * CELL}" text-anchor="middle" class="layout-number">${offset}</text>`
      );
    }
  }

  for (let row = 0; row < 4; row += 2) {
    for (let col = 0; col < 8; col++)
      cells.push(
        `<rect x="${35 + col * CELL}" y="${35 + row * CELL}" width="30" height="62" rx="4" fill="none" stroke="#df806f" stroke-width="1.5"/>`
      );
  }

  for (let row = 0; row < 4; row++) cells.push(`<text x="26" y="${55 + row * CELL}" text-anchor="end" class="layout-axis">r${row}</text>`);
  for (let col = 0; col < 8; col++)
    cells.push(`<text x="${50 + col * CELL}" y="27" text-anchor="middle" class="layout-axis">c${col}</text>`);

  return cells.join('');
}

// "VMEM S(1)" for a shape's memory space.
function spaceTag(shape: string) {
  const { space } = shapeMemorySpace(shape);
  return `${memorySpaceLabel(space)} S(${space})`;
}

export function layoutDiagram(node: HloNode): string | null {
  if (!['copy-start', 'parameter'].includes(node.op) || !TILED_MATRIX.test(node.type)) return null;

  const slots = tupleElements(node.type);
  const matrixSlots = slots.map((slot, index) => ({ slot, index })).filter(({ slot }) => TILED_MATRIX.test(slot));
  const reference = matrixSlots[0]?.slot;
  if (!reference) return null;

  let locationText: string;
  if (node.op === 'copy-start') {
    locationText = `${spaceTag(slots[1])} → ${spaceTag(slots[0])}`;
  } else {
    const indices = matrixSlots.map(({ index }) => index).join(', ');
    locationText = `Element${matrixSlots.length > 1 ? 's' : ''} ${indices} · ${spaceTag(reference)}`;
  }

  const dimensions = /\[(\d+),(\d+)\]/.exec(reference);
  const tileCount = dimensions ? Math.ceil(Number(dimensions[1]) / 8) * Math.ceil(Number(dimensions[2]) / 128) : null;
  const tileCountText =
    tileCount === null ? 'The full array is made of many such tiles.' : `Each such matrix has ${tileCount} outer tiles.`;

  const ariaLabel =
    'First four rows and eight columns of an eight by 128 tile; numbers show physical order and red outlines show two by one inner tiles';
  return `<div class="layout-diagram"><div class="layout-diagram-head"><strong>Array layout · T(8,128)(2,1)</strong><span>${locationText}</span></div><svg viewBox="0 0 310 174" role="img" aria-label="${ariaLabel}"><title>8×128 tile, with 2×1 inner tiles</title>${tileCornerSvg()}</svg><p>The top-left 4 rows × 8 columns of the matrix; a full outer tile is 8×128. Numbers are physical offsets within the tile; red outlines are the 2×1 inner tiles. ${tileCountText}</p></div>`;
}
