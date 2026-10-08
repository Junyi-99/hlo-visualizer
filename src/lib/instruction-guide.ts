import type { Computation, GuidePart, HloModule, HloNode, InstructionGuide, ResultGroup, TypeDetail, TypeSlot } from './types';
import { extractHloMetadata } from './metadata.ts';
import { splitTopLevel } from './parser.ts';
import { attributeKey, explainAttribute, opDescription } from './hlo-reference.ts';
import { bufferStatus, type BufferStatus } from './memory-location.ts';
import { memorySpaceLabel, shapeMemoryText } from './memory-space.ts';

interface GuideContext { computation?: Computation; module?: HloModule }

const escapeHtml = (value: unknown) => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'} as Record<string, string>)[c]);
const STYLED_KEYS = /^(name|tuple|dest|source|context|shape|order|tile|space|op|operand|priority|literal|parameter-index|target|constraints|metadata|backend|dimensions|tuple-index|fusion-kind|called-computation|slot-\d+)$/;
// Keys without a CSS color get a stable hue so neighbouring segments stay distinguishable.
const hue = (key: string) => [...key].reduce((h, c) => (h * 31 + c.charCodeAt(0)) % 360, 7);
export const partColor = (key: string) => STYLED_KEYS.test(key) ? undefined : `hsl(${hue(key)} 45% 40%)`;
const partStyle = (key: string) => partColor(key) ? ` style="--part:${partColor(key)}"` : '';
const open = (key: string) => `<span class="hlo-part part-${key}" data-part="${key}" tabindex="0"${partStyle(key)}>`;
const markHtml = (key: string, html: string) => `${open(key)}${html}</span>`;
const mark = (key: string, text: string) => markHtml(key, escapeHtml(text));

function splitTuple(text: string) {
  const parts: { body: string; separator: string }[] = [];
  let parens = 0, brackets = 0, braces = 0, start = 1;
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
  let output = '', position = 0;
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
  const html = pieces.map((piece, index) => {
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
        inner = escapeHtml(text.slice(0, priority.index)) + mark('priority', priority[0]) + escapeHtml(text.slice(priority.index + priority[0].length));
        attributes.push({ name: 'dma_priority', value: /\d+/.exec(priority[0])![0] });
      }
    }
    return escapeHtml(lead) + markHtml(attributeKey(name), inner) + escapeHtml(trail) + separator;
  }).join('');
  return { html, attributes };
}

function compactEncodedBody(html: string) {
  return html.replace(/(&quot;body&quot;\s*:\s*&quot;)([A-Za-z0-9+/=]{300,})(&quot;)/,
    (_match, before: string, body: string, after: string) => `${before}[${body.length.toLocaleString('en-US')} encoded characters]${after}`);
}

// Operand references, with or without the % sigil, plus printer comments such as /*index=5*/.
function operandHtml(args: string) {
  return splitTopLevel(args).map((piece, index, all) => {
    const separator = index < all.length - 1 ? ',' : '';
    const match = /^([\s\S]*?)(%?[\w.-]+(?:#\d+)?)(\s*)$/.exec(piece);
    if (!match) return escapeHtml(piece) + separator;
    const head = match[1].replace(/\/\*[^*]*\*\//g, comment => `\u0000${comment}\u0000`).split('\u0000')
      .map(chunk => chunk.startsWith('/*') ? mark('operand-index', chunk) : escapeHtml(chunk)).join('');
    return head + mark('operand', match[2]) + escapeHtml(match[3]) + separator;
  }).join('');
}

function memoryLabel(shape: string) {
  const value = Number(/S\((\d+)\)/.exec(shape)?.[1] || 0);
  return value === 0 ? 'HBM（省略 S(0)）' : `${memorySpaceLabel(value)}（S(${value})）`;
}

const DTYPES: Record<string, string> = {
  pred: '布尔值', s4: '4 位有符号整数', s8: '8 位有符号整数', s16: '16 位有符号整数', s32: '32 位有符号整数', s64: '64 位有符号整数',
  u4: '4 位无符号整数', u8: '8 位无符号整数', u16: '16 位无符号整数', u32: '32 位无符号整数', u64: '64 位无符号整数',
  f16: '16 位半精度浮点数', bf16: '16 位脑浮点数（bfloat16）', f32: '32 位浮点数', f64: '64 位浮点数',
  f8e4m3fn: '8 位浮点数（e4m3）', f8e5m2: '8 位浮点数（e5m2）', c64: '64 位复数（两个 f32）', c128: '128 位复数（两个 f64）', token: '令牌（只用于排序副作用，不含数据）',
};

// What to say about where a value lives. `status` is null when the guide has no module context.
function placement(shape: string, status: BufferStatus | null, node: HloNode) {
  const space = /S\((\d+)\)/.exec(shape)?.[1];
  const marked = space === undefined ? '布局里没有写 S(n)' : `布局里写的是 S(${space})`;
  switch (status) {
    case 'fusion':
      if (node.op === 'parameter') return `fusion 的输入参数，就是调用方传入的操作数，本身不另占缓冲区；${marked}。`;
      return `没有独立缓冲区：这条指令在 fusion 内核内部计算，结果不会单独写回 HBM 或 VMEM。${marked}，这不代表实际存放位置。${node.root ? '它的结果就是这个 fusion 的输出，实际位置看调用它的 fusion 指令。' : ''}`;
    case 'thread-local': return `规约、比较等函数里的值，XLA 为它分配 thread-local 存储，不在 HBM 或 VMEM 中；${marked}。`;
    case 'unassigned': return `编译前（未调度）的 HLO 还没有经过内存空间分配；${marked}，实际位置要看编译后的 HLO。`;
    default: return `${shapeMemoryText(shape)}${status ? '' : '（前提是这条指令有自己的缓冲区）'}`;
  }
}

function shapeDescription(shape: string, includeMemory = true) {
  const match = /([a-z][\w]*)\[([^\]]*)\]/.exec(shape);
  if (!match) return '该项的具体类型以原始 HLO 为准';
  const dtype = DTYPES[match[1]] || match[1];
  const logical = match[1] === 'token' ? '' : match[2] ? `，${match[2].replaceAll(',', '×')} 的数组` : '，标量（逻辑形状 []）';
  return `${match[1]}[${match[2]}]：${dtype}${logical}${includeMemory ? `；${memoryLabel(shape)}` : ''}`;
}

function typeDetails(shape: string, status: BufferStatus | null = null, node?: HloNode): TypeDetail[] {
  const order = /\{(\d+(?:,\d+)*)(?=[:}])/.exec(shape)?.[1];
  const tile = /T(?:\([^)]*\))+/.exec(shape)?.[0];
  return [
    { key:'shape', label:'元素类型与逻辑形状', text:shapeDescription(shape, false) },
    ...(order ? [{ key:'order', label:'维度顺序', text:`${order}：minor-to-major；排在前面的维度变化最快。` }] : []),
    ...(tile ? [{ key:'tile', label:'物理分块', text:shape.includes('[]') ? `${tile} 是标量的布局标记，不表示 ${tile.match(/\d+/)?.[0] || ''} 个逻辑元素。` : `${tile} 描述数组的物理 tiling。` }] : []),
    { key:'space', label:'内存空间', text:node ? placement(shape, status, node) : shapeMemoryText(shape) }
  ];
}

function whileState(context: GuideContext) {
  const { module, computation } = context || {};
  if (!module || !computation) return null;
  for (const caller of module.computations) for (const instruction of caller.nodes) {
    if (instruction.op !== 'while' || !Object.values(instruction.calls).includes(computation.name)) continue;
    const initial = caller.byName.get(instruction.operands[0]);
    const body = module.byName.get(instruction.calls.body);
    const next = body?.nodes.find(node => node.root && node.op === 'tuple');
    return { initial: initial?.op === 'tuple' ? initial.operands : [], next: next?.operands || [] };
  }
  return null;
}

const operandList = (node: HloNode) => node.operands.map(name => `%${name}`).join('、');

export function instructionGuide(node: HloNode, context: GuideContext = {}): InstructionGuide {
  const raw = node.raw;
  const prefix = /^(ROOT\s+)?(%?[\w.-]+)(\s*=\s*)/.exec(raw);
  const opAt = prefix ? raw.indexOf(` ${node.op}(`, prefix[0].length) : -1;
  if (!prefix || opAt < 0) return { html: escapeHtml(raw), parts: [], resultGroup: null };
  const typeText = raw.slice(prefix[0].length, opAt);
  const argsStart = opAt + node.op.length + 2;
  let depth = 1, argsEnd = argsStart;
  for (; argsEnd < raw.length; argsEnd++) {
    if (raw[argsEnd] === '(') depth++;
    else if (raw[argsEnd] === ')' && --depth === 0) break;
  }
  const args = raw.slice(argsStart, argsEnd);
  const suffix = raw.slice(argsEnd + 1);
  const tupleType = typeText.startsWith('(') && typeText.endsWith(')');
  const copyStart = node.op === 'copy-start' && tupleType;
  const opPart: GuidePart = { key:'op', label:'操作', text:opDescription(node) };
  const status = context.module && context.computation ? bufferStatus(context.module, context.computation, node) : null;
  const hasBuffer = status === null || status === 'buffer'; // only then does S(n) say where the value is
  let resultType: string, parts: GuidePart[], tupleGroup: { rawType: string; slots: (TypeSlot & GuidePart)[] } | null = null;
  if (copyStart) {
    const slots = splitTuple(typeText);
    const memory = (shape: string) => Number(/S\((\d+)\)/.exec(shape)?.[1] || 0);
    const memoryLabel = (number: number) => number === 0 ? 'HBM（S(0) 省略）' : `${memorySpaceLabel(number)}（S(${number})）`;
    const outputShape = /([a-z][\w]*)\[([^\]]*)\]/.exec(slots[0]?.body || '');
    const order = /\{(\d+(?:,\d+)*):/.exec(slots[0]?.body || '')?.[1];
    const tile = /T(?:\([^)]*\))+/.exec(slots[0]?.body || '')?.[0];
    const tileLevels = tile && [...tile.matchAll(/\(([^)]*)\)/g)].map(match => match[1].replaceAll(',', '×'));
    resultType = `<span class="hlo-type part-tuple" data-part="tuple" tabindex="0">` + mark('tuple', '(') + slots.map(({body,separator}, index) =>
      `<span class="hlo-slot part-${['dest','source','context'][index] || 'shape'}" data-part="${['dest','source','context'][index] || 'shape'}">${shapeHtml(body)}</span>${escapeHtml(separator)}`).join('') + mark('tuple', ')') + '</span>';
    parts = [
      { key:'name', label:'结果名', text:`${prefix[2]} 是这条指令的结果标识，后续节点可引用它。` },
      { key:'tuple', label:'三项结果', text:'括号表示返回一个 tuple；依次包含复制目标、复制源和上下文，供 copy-done 使用。' },
      { key:'dest', label:'第 0 项 · 目标', text:`复制完成后数据所在的目标缓冲区；这里位于 ${memoryLabel(memory(slots[0]?.body || ''))}。` },
      { key:'source', label:'第 1 项 · 源', text:`复制前数据所在的源缓冲区；这里位于 ${memoryLabel(memory(slots[1]?.body || ''))}。` },
      { key:'context', label:'第 2 项 · 上下文', text:`${(slots[2]?.body || 'u32[]').trim()} 是异步复制的同步标志，copy-done 据此等待复制完成；它位于 ${memoryLabel(memory(slots[2]?.body || ''))}。` },
      { key:'shape', label:'类型与形状', text:outputShape ? `${outputShape[1]}[${outputShape[2]}] 是结果张量的元素类型和逻辑形状；u32[] 表示标量。` : '这里写的是结果张量的元素类型和逻辑形状；u32[] 表示标量。' },
      { key:'order', label:'维度顺序', text:order ? `${order} 是 minor-to-major 物理顺序；排在最前面的维度变化最快。` : '大括号内的数字表示 minor-to-major 物理顺序。' },
      { key:'tile', label:'分块布局', text:tileLevels?.length === 2 ? `${tile} 是两层 tiling：外层 ${tileLevels[0]}，内层 ${tileLevels[1]}；可能引入填充。` : tile ? `${tile} 描述分块的物理布局与可能的填充。` : 'T(...) 描述分块的物理布局与可能的填充。' },
      { key:'space', label:'内存空间', text:'S(n) 是内存空间编号。在 TPU 上，S(1) 是 VMEM，S(2) 是 SFLAG（同步标志），S(6) 是 SMEM（标量内存）；未写 S(n) 即默认的 S(0)，也就是 HBM。' },
      opPart,
      { key:'operand', label:'输入', text:`${operandList(node) || '括号内的节点'} 提供要复制的数据；点击上方 Direct inputs 可跳到该节点。` },
    ];
    parts = parts.filter(part => (part.key !== 'tile' || tile) && (part.key !== 'order' || order) && (part.key !== 'space' || /S\(/.test(typeText)));
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
    resultType = `<span class="hlo-type part-tuple" data-part="tuple" tabindex="0">` + mark('tuple', '(') + slots.map(({body,separator}, index) =>
      `<span class="hlo-slot part-slot-${index % 4}" data-part="slot-${index}">${shapeHtml(body)}</span>${escapeHtml(separator)}`).join('') + mark('tuple', ')') + '</span>';
    const slotParts = slots.map(({body}, index) => {
      const origin = state ? `首次进入 while 时来自 ${state.initial[index] ? `%${state.initial[index]}` : '初始状态'}；后续迭代由 body 返回 ${state.next[index] ? `%${state.next[index]}` : '对应项'}。` :
        node.op === 'tuple' && node.operands[index] ? `这一项来自 %${node.operands[index]}。` : '';
      const readers = used.has(index) ? `此 computation 通过 ${used.get(index)!.map(name=>`%${name}`).join('、')} 读取它。` : node.op === 'parameter' && context.computation ? '此 computation 没有直接取出这一项。' : '';
      const details = [
        ...typeDetails(body, status, node),
        ...(origin ? [{ label:node.op === 'parameter' ? '循环状态来源' : '数据来源', text:origin }] : []),
        ...(readers ? [{ label:'本计算中的使用', text:readers }] : [])
      ];
      return { key:`slot-${index}`, label:`第 ${index} 项`, raw:body.trim(), text:`${shapeDescription(body, hasBuffer)}。${origin}${readers}`, details };
    });
    tupleGroup = { rawType: typeText, slots: slotParts };
    parts = [
      { key:'name', label:'结果名', text:`${prefix[2]} 是这条指令的结果标识。` },
      { key:'tuple', label:`${slots.length} 项 tuple`, text:`外层括号表示一个有 ${slots.length} 个位置的 tuple。每一项都有自己的类型、形状和布局；序号从 0 开始。` },
      ...slotParts,
      { key:'shape', label:'类型与逻辑形状', text:'tuple 的每一项分别声明元素类型和逻辑形状；展开子项可逐项查看。' },
      { key:'order', label:'物理维度顺序', text:'大括号内的数字是 minor-to-major 顺序；排在前面的维度变化最快。' },
      { key:'tile', label:'T(...) 分块', text:/\bs32\[\]\{[^}]*T\(128\)/.test(typeText) ? 'T(128) 是标量的布局标记，不会把标量变成 128 个逻辑元素；其他 T(...) 描述数组的物理分块。' : 'T(...) 描述物理分块；它不会改变数组的逻辑形状。' },
      { key:'space', label:'S(...) 内存空间', text:'S(n) 指定内存空间；未写 S(n) 通常是默认空间。' },
      node.op === 'parameter' ? { ...opPart, text:'parameter 声明当前 computation 的一个输入，不在这里创建新的数据。' } : opPart
    ];
    if (node.op === 'parameter') parts.push({ key:'parameter-index', label:'参数序号', text:`parameter(${args}) 表示当前 computation 的第 ${args} 个输入；这里整个 tuple 是一个参数，而不是 ${slots.length} 个独立参数。` });
    if (node.op !== 'parameter' && node.operands.length) parts.push({ key:'operand', label:'输入依赖', text:`这条指令直接依赖 ${operandList(node)}；图中的输入边对应这些引用。` });
    if (!/\{\d+(?:,\d+)*[:}]/.test(typeText)) parts = parts.filter(part => part.key !== 'order');
    if (!/T\(/.test(typeText)) parts = parts.filter(part => part.key !== 'tile');
    if (!/S\(/.test(typeText)) parts = parts.filter(part => part.key !== 'space');
  } else {
    resultType = shapeHtml(typeText);
    const details = typeDetails(typeText, status, node);
    const detail = (key: string) => details.find(item => item.key === key);
    parts = [
      { key:'name', label:'结果名', text:`${prefix[2]} 是这条指令的结果标识。` },
      { key:'shape', label:'结果类型', text:`${shapeDescription(typeText, hasBuffer)}。等号后面是结果的数据类型、逻辑形状与可选的物理布局。` },
      ...(/\{\d+(?:,\d+)*[:}]/.test(typeText) ? [{ key:'order', label:'维度顺序', text:detail('order')!.text }] : []),
      ...(/T\(/.test(typeText) ? [{ key:'tile', label:'物理分块', text:detail('tile')!.text }] : []),
      ...(/S\(/.test(typeText) ? [{ key:'space', label:'内存空间', text:detail('space')!.text }] : []),
      opPart,
      { key:'operand', label:'输入', text:`括号内的 ${operandList(node)} 是这条指令依赖的上游节点${node.operands.length > 1 ? '，按操作数顺序排列' : ''}。${/#\d/.test(args) ? '%t#N 是 XLA dump 的写法，表示取 tuple %t 的第 N 项（相当于 get-tuple-element）。' : ''}` }
    ];
    if (node.op === 'parameter') parts.push({ key:'parameter-index', label:'参数序号', text:`parameter(${args}) 表示当前 computation 的第 ${args} 个输入。` });
    if (node.op === 'constant') parts.push({ key:'literal', label:'常量值', text:`括号中的 ${args.length > 80 ? `${args.slice(0, 77)}…` : args} 是这条指令直接给出的值；结果类型决定它的数据类型与形状，不依赖上游节点。` });
    if (!node.operands.length || node.op === 'constant' || node.op === 'parameter') parts = parts.filter(part => part.key !== 'operand');
  }
  if (/\/\*[^*]*\*\//.test(args)) parts.push({ key:'operand-index', label:'序号注释', text:'/*index=N*/ 是 HLO 打印器每隔几个操作数插入的注释，标出紧随其后的操作数序号（从 0 开始），不是操作数本身。' });
  const suffixParts = suffixHtml(suffix);
  for (const { name, value } of suffixParts.attributes) {
    const explanation = explainAttribute(name, value, { node, module: context.module, computation: context.computation });
    if (!parts.some(part => part.key === explanation.key)) parts.push(explanation);
  }
  const resultGroup: ResultGroup = tupleType ? {
    kind:'tuple', rawType:typeText,
    slots:tupleGroup?.slots || splitTuple(typeText).map(({body}, index) => ({
      key:['dest','source','context'][index] || `slot-${index}`,
      label:`第 ${index} 项${copyStart ? [' · 目标',' · 源',' · 上下文'][index] || '' : ''}`,
      raw:body.trim(),
      details:typeDetails(body, status, node)
    }))
  } : { kind:'array', rawType:typeText, details:typeDetails(typeText, status, node) };
  const argumentsHtml = node.op === 'constant' ? mark('literal', `(${args})`) :
    '(' + (node.op === 'parameter' ? mark('parameter-index', args) : operandHtml(args)) + ')';
  const html = escapeHtml(prefix[1] || '') + mark('name', prefix[2]) + escapeHtml(prefix[3]) + resultType +
    ' ' + mark('op', node.op) + argumentsHtml + suffixParts.html;
  const metadata = extractHloMetadata(suffix);
  const compactHtml = compactEncodedBody(metadata ? html.replace(mark('metadata', metadata.raw), mark('metadata', 'metadata={…}')) : html);
  return { html, compactHtml: compactHtml === html ? undefined : compactHtml, parts, resultGroup };
}

export function layoutDiagram(node: HloNode): string | null {
  if (!['copy-start','parameter'].includes(node.op) || !/\{1,0:T\(8,128\)\(2,1\)/.test(node.type)) return null;
  const slots = splitTuple(node.type);
  if (!slots.length) return null;
  const space = (shape: string) => Number(/S\((\d+)\)/.exec(shape)?.[1] || 0);
  const location = memorySpaceLabel;
  const copyStart = node.op === 'copy-start';
  const matrixSlots = slots.map((slot,index) => ({ slot,index })).filter(({slot}) => /\{1,0:T\(8,128\)\(2,1\)/.test(slot.body));
  const reference = matrixSlots[0]?.slot.body;
  if (!reference) return null;
  const locationText = copyStart ? `${location(space(slots[1].body))} S(${space(slots[1].body)}) → ${location(space(slots[0].body))} S(${space(slots[0].body)})` :
    `第 ${matrixSlots.map(({index})=>index).join('、')} 项 · ${location(space(reference))} S(${space(reference)})`;
  const dimensions = /\[(\d+),(\d+)\]/.exec(reference);
  const tileCount = dimensions ? Math.ceil(Number(dimensions[1])/8) * Math.ceil(Number(dimensions[2])/128) : null;
  const cells = [];
  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 8; col++) {
      const offset = Math.floor(row / 2) * 256 + col * 2 + row % 2;
      const fill = row < 2 ? col < 4 ? '#e5f0fa' : '#fff0df' : col < 4 ? '#e9f4e8' : '#f7e9f1';
      cells.push(`<rect x="${34 + col*32}" y="${34 + row*32}" width="32" height="32" fill="${fill}" stroke="#d6dfea"/><text x="${50 + col*32}" y="${55 + row*32}" text-anchor="middle" class="layout-number">${offset}</text>`);
    }
  }
  for (let row = 0; row < 4; row += 2) for (let col = 0; col < 8; col++)
    cells.push(`<rect x="${35 + col*32}" y="${35 + row*32}" width="30" height="62" rx="4" fill="none" stroke="#df806f" stroke-width="1.5"/>`);
  for (let row = 0; row < 4; row++) cells.push(`<text x="26" y="${55+row*32}" text-anchor="end" class="layout-axis">r${row}</text>`);
  for (let col = 0; col < 8; col++) cells.push(`<text x="${50+col*32}" y="27" text-anchor="middle" class="layout-axis">c${col}</text>`);
  return `<div class="layout-diagram"><div class="layout-diagram-head"><strong>Array layout · T(8,128)(2,1)</strong><span>${locationText}</span></div><svg viewBox="0 0 310 174" role="img" aria-label="First four rows and eight columns of an eight by 128 tile; numbers show physical order and red outlines show two by one inner tiles"><title>8×128 tile, with 2×1 inner tiles</title>${cells.join('')}</svg><p>矩阵项的左上角 4 行 × 8 列示意；完整外层 tile 是 8×128。数字是 tile 内的物理偏移，红框是 2×1 内层 tile。${tileCount === null ? '完整数组由多个这样的 tile 拼成。' : `每个这样的矩阵共含 ${tileCount} 个外层 tile。`}</p></div>`;
}
