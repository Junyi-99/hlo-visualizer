import type { Computation, GuidePart, HloModule, HloNode, InstructionGuide, ResultGroup, TypeDetail, TypeSlot } from './types';
import { extractHloMetadata } from './metadata.ts';

interface GuideContext { computation?: Computation; module?: HloModule }

const escapeHtml = (value: unknown) => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'} as Record<string, string>)[c]);
const mark = (key: string, text: string) => `<span class="hlo-part part-${key}" data-part="${key}" tabindex="0">${escapeHtml(text)}</span>`;

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

function suffixHtml(text: string) {
  const metadata = extractHloMetadata(text);
  if (metadata) return suffixFieldsHtml(text.slice(0, metadata.start)) + mark('metadata', metadata.raw) + suffixFieldsHtml(text.slice(metadata.end));
  return suffixFieldsHtml(text);
}

function suffixFieldsHtml(text: string) {
  let result = '', position = 0;
  const fields = /custom_call_target="[^"]*"|operand_layout_constraints=|backend_config=|\bdimensions=\{[^}]*\}|\bindex=\d+|\bkind=k[\w]+|\bcalls=%[\w.-]+|"?dma_priority"?\s*[:=]\s*"?\d+"?/g;
  for (const match of text.matchAll(fields)) {
    const key = match[0].startsWith('custom_call_target') ? 'target' :
      match[0].startsWith('operand_layout_constraints') ? 'constraints' :
      match[0].startsWith('backend_config') ? 'backend' :
      match[0].startsWith('dimensions=') ? 'dimensions' :
      match[0].startsWith('index=') ? 'tuple-index' :
      match[0].startsWith('kind=') ? 'fusion-kind' :
      match[0].startsWith('calls=') ? 'called-computation' : 'priority';
    result += escapeHtml(text.slice(position, match.index)) + mark(key, match[0]);
    position = match.index + match[0].length;
  }
  return result + escapeHtml(text.slice(position));
}

function compactEncodedBody(html: string) {
  return html.replace(/(&quot;body&quot;\s*:\s*&quot;)([A-Za-z0-9+/=]{300,})(&quot;)/,
    (_match, before: string, body: string, after: string) => `${before}[${body.length.toLocaleString('en-US')} encoded characters]${after}`);
}

function operandHtml(args: string) {
  let result = '', position = 0;
  for (const match of args.matchAll(/%[\w.-]+/g)) {
    result += escapeHtml(args.slice(position, match.index)) + mark('operand', match[0]);
    position = match.index + match[0].length;
  }
  return result + escapeHtml(args.slice(position));
}

function memoryLabel(shape: string) {
  const value = Number(/S\((\d+)\)/.exec(shape)?.[1] || 0);
  return value === 0 ? 'HBM（省略 S(0)）' : value === 1 ? 'VMEM（S(1)）' : `S(${value})（后端专用空间）`;
}

function shapeDescription(shape: string, includeMemory = true) {
  const match = /([a-z][\w]*)\[([^\]]*)\]/.exec(shape);
  if (!match) return '该项的具体类型以原始 HLO 为准';
  const dtype = ({ s32:'32 位有符号整数', u32:'32 位无符号整数', bf16:'16 位脑浮点数', f32:'32 位浮点数', pred:'布尔值' } as Record<string, string>)[match[1]] || match[1];
  const logical = match[2] ? `${match[2].replaceAll(',', '×')} 的数组` : '标量（逻辑形状 []）';
  return `${match[1]}[${match[2]}]：${dtype}，${logical}${includeMemory ? `；${memoryLabel(shape)}` : ''}`;
}

function typeDetails(shape: string): TypeDetail[] {
  const order = /\{(\d+(?:,\d+)*)(?=[:}])/.exec(shape)?.[1];
  const tile = /T(?:\([^)]*\))+/.exec(shape)?.[0];
  const space = /S\((\d+)\)/.exec(shape)?.[1];
  return [
    { key:'shape', label:'元素类型与逻辑形状', text:shapeDescription(shape, false) },
    ...(order ? [{ key:'order', label:'维度顺序', text:`${order}：minor-to-major；排在前面的维度变化最快。` }] : []),
    ...(tile ? [{ key:'tile', label:'物理分块', text:shape.includes('[]') ? `${tile} 是标量的布局标记，不表示 ${tile.match(/\d+/)?.[0] || ''} 个逻辑元素。` : `${tile} 描述数组的物理 tiling。` }] : []),
    { key:'space', label:'内存空间', text:space === undefined ? '未写 S(n)：通常是默认内存空间 S(0)，在 TPU 上为 HBM。' : memoryLabel(shape) }
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

export function instructionGuide(node: HloNode, context: GuideContext = {}): InstructionGuide {
  const raw = node.raw;
  const prefix = /^(ROOT\s+)?(%[\w.-]+)(\s*=\s*)/.exec(raw);
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
  let resultType: string, parts: GuidePart[], tupleGroup: { rawType: string; slots: (TypeSlot & GuidePart)[] } | null = null;
  if (copyStart) {
    const slots = splitTuple(typeText);
    const memory = (shape: string) => Number(/S\((\d+)\)/.exec(shape)?.[1] || 0);
    const memoryLabel = (number: number) => number === 0 ? 'HBM（S(0) 省略）' : number === 1 ? 'VMEM（S(1)）' : `S(${number})（后端专用空间）`;
    const outputShape = /([a-z][\w]*)\[([^\]]*)\]/.exec(slots[0]?.body || '');
    const order = /\{(\d+(?:,\d+)*):/.exec(slots[0]?.body || '')?.[1];
    const tile = /T(?:\([^)]*\))+/.exec(slots[0]?.body || '')?.[0];
    const tileLevels = tile && [...tile.matchAll(/\(([^)]*)\)/g)].map(match => match[1].replaceAll(',', '×'));
    const priority = /"?dma_priority"?\s*[:=]\s*"?(\d+)/.exec(raw)?.[1];
    resultType = `<span class="hlo-type part-tuple" data-part="tuple" tabindex="0">` + mark('tuple', '(') + slots.map(({body,separator}, index) =>
      `<span class="hlo-slot part-${['dest','source','context'][index] || 'shape'}" data-part="${['dest','source','context'][index] || 'shape'}">${shapeHtml(body)}</span>${escapeHtml(separator)}`).join('') + mark('tuple', ')') + '</span>';
    parts = [
      { key:'name', label:'结果名', text:`%${node.name} 是这条指令的结果标识，后续节点可引用它。` },
      { key:'tuple', label:'三项结果', text:'括号表示返回一个 tuple；依次包含复制目标、复制源和上下文，供 copy-done 使用。' },
      { key:'dest', label:'第 0 项 · 目标', text:`复制完成后数据所在的目标缓冲区；这里位于 ${memoryLabel(memory(slots[0]?.body || ''))}。` },
      { key:'source', label:'第 1 项 · 源', text:`复制前数据所在的源缓冲区；这里位于 ${memoryLabel(memory(slots[1]?.body || ''))}。` },
      { key:'context', label:'第 2 项 · 上下文', text:'u32[] 是 32 位无符号标量，作为异步复制的上下文标记；这里的 S(2) 是后端专用内存空间编号。' },
      { key:'shape', label:'类型与形状', text:outputShape ? `${outputShape[1]}[${outputShape[2]}] 是结果张量的元素类型和逻辑形状；u32[] 表示标量。` : '这里写的是结果张量的元素类型和逻辑形状；u32[] 表示标量。' },
      { key:'order', label:'维度顺序', text:order ? `${order} 是 minor-to-major 物理顺序；排在最前面的维度变化最快。` : '大括号内的数字表示 minor-to-major 物理顺序。' },
      { key:'tile', label:'分块布局', text:tileLevels?.length === 2 ? `${tile} 是两层 tiling：外层 ${tileLevels[0]}，内层 ${tileLevels[1]}；可能引入填充。` : tile ? `${tile} 描述分块的物理布局与可能的填充。` : 'T(...) 描述分块的物理布局与可能的填充。' },
      { key:'space', label:'内存空间', text:'S(1) 在 TPU 上表示 VMEM；未写 S(n) 通常等同 S(0)，即 HBM。S(2) 的具体用途由后端定义。' },
      { key:'op', label:'操作', text:'copy-start 启动异步复制；对应的 copy-done 等待/完成复制并产生目标张量。' },
      { key:'operand', label:'输入', text:`${node.operands.map(x=>`%${x}`).join('、') || '括号内的节点'} 提供要复制的数据；点击上方 Direct inputs 可跳到该节点。` },
      { key:'priority', label:'DMA 优先级', text:`dma_priority=${priority ?? '?'} 是后端配置中的 DMA 优先级值；数值如何影响调度由具体后端决定。` }
    ];
    parts = parts.filter(part => (part.key !== 'priority' || priority !== undefined) && (part.key !== 'tile' || tile) && (part.key !== 'order' || order));
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
        ...typeDetails(body),
        ...(origin ? [{ label:node.op === 'parameter' ? '循环状态来源' : '数据来源', text:origin }] : []),
        ...(readers ? [{ label:'本计算中的使用', text:readers }] : [])
      ];
      return { key:`slot-${index}`, label:`第 ${index} 项`, raw:body.trim(), text:`${shapeDescription(body)}。${origin}${readers}`, details };
    });
    tupleGroup = { rawType: typeText, slots: slotParts };
    parts = [
      { key:'name', label:'结果名', text:`%${node.name} 是这条指令的结果标识。` },
      { key:'tuple', label:`${slots.length} 项 tuple`, text:`外层括号表示一个有 ${slots.length} 个位置的 tuple。每一项都有自己的类型、形状和布局；序号从 0 开始。` },
      ...slotParts,
      { key:'shape', label:'类型与逻辑形状', text:'tuple 的每一项分别声明元素类型和逻辑形状；展开子项可逐项查看。' },
      { key:'order', label:'物理维度顺序', text:'大括号内的数字是 minor-to-major 顺序；排在前面的维度变化最快。' },
      { key:'tile', label:'T(...) 分块', text:/\bs32\[\]\{[^}]*T\(128\)/.test(typeText) ? 'T(128) 是标量的布局标记，不会把标量变成 128 个逻辑元素；其他 T(...) 描述数组的物理分块。' : 'T(...) 描述物理分块；它不会改变数组的逻辑形状。' },
      { key:'space', label:'S(...) 内存空间', text:'S(n) 指定内存空间；未写 S(n) 通常是默认空间。' },
      { key:'op', label:'操作', text:node.op === 'parameter' ? 'parameter 声明当前 computation 的一个输入，不在这里创建新的数据。' : node.op === 'custom-call' ? 'custom-call 把这一步交给后端实现；文本 HLO 只显示调用边界和声明的结果。' : `${node.op} 是这条 HLO 指令的操作。` }
    ];
    if (node.op === 'parameter') parts.push({ key:'parameter-index', label:'参数序号', text:`parameter(${args}) 表示当前 computation 的第 ${args} 个输入；这里整个 tuple 是一个参数，而不是 ${slots.length} 个独立参数。` });
    if (node.op !== 'parameter' && node.operands.length) parts.push({ key:'operand', label:'输入依赖', text:`这条指令直接依赖 ${node.operands.map(name => `%${name}`).join('、')}；图中的输入边对应这些引用。` });
    if (node.op === 'custom-call') {
      const target = /\bcustom_call_target="([^"]+)"/.exec(suffix)?.[1];
      const opName = /\bop_name="([^"]+)"/.exec(suffix)?.[1];
      const sourceLine = /\bsource_line=(\d+)/.exec(suffix)?.[1];
      parts.push(
        { key:'target', label:'后端目标', text:`${target || 'custom_call_target'} 是注册的后端调用目标；它本身不展开内部计算图。` },
        { key:'constraints', label:'输入布局约束', text:'operand_layout_constraints 依输入顺序列出后端期望的每个操作数布局。' },
        { key:'metadata', label:'来源信息', text:`${opName ? `JAX 操作路径：${opName}。` : 'metadata 可记录源操作。'}${sourceLine ? ` 源码行号：${sourceLine}。` : ''}` },
        { key:'backend', label:'后端配置', text:'backend_config 包含后端专用的序列化配置；其中编码的 body 不能仅凭外层 HLO 推导出内部指令依赖。' }
      );
      parts = parts.filter(part => part.key !== 'target' || !!target).filter(part => part.key !== 'constraints' || suffix.includes('operand_layout_constraints=')).filter(part => part.key !== 'metadata' || suffix.includes('metadata=')).filter(part => part.key !== 'backend' || suffix.includes('backend_config='));
    }
    if (!/\{\d+(?:,\d+)*:/.test(typeText)) parts = parts.filter(part => part.key !== 'order');
    if (!/T\(/.test(typeText)) parts = parts.filter(part => part.key !== 'tile');
    if (!/S\(/.test(typeText)) parts = parts.filter(part => part.key !== 'space');
  } else {
    resultType = shapeHtml(typeText);
    parts = [
      { key:'name', label:'结果名', text:`%${node.name} 是这条指令的结果标识。` },
      { key:'shape', label:'结果类型', text:'等号后面是结果的数据类型、逻辑形状与可选的物理布局。' },
      { key:'op', label:'操作', text:node.op === 'broadcast' ? 'broadcast 把输入值扩展到结果形状；dimensions 指定输入轴在输出中的对应位置。' : node.op === 'get-tuple-element' ? 'get-tuple-element 从输入 tuple 中选出一个位置作为本节点的结果。' : `${node.op} 是执行的 HLO 操作。` },
      { key:'operand', label:'输入', text:'括号内的 %名称表示这条指令依赖的上游节点。' }
    ];
    if (node.op === 'parameter') parts.push({ key:'parameter-index', label:'参数序号', text:`parameter(${args}) 表示当前 computation 的第 ${args} 个输入。` });
    if (node.op === 'constant') parts.push({ key:'literal', label:'常量值', text:`括号中的 ${args} 是这条指令直接给出的值；结果类型决定它的数据类型与形状，不依赖上游节点。` });
    if (node.op === 'get-tuple-element' && node.index !== null) parts.push({ key:'tuple-index', label:'读取 tuple 的位置', text:`index=${node.index} 从 ${node.operands[0] ? `%${node.operands[0]}` : '输入 tuple'} 取出第 ${node.index} 项（从 0 开始）；此节点的结果类型对应那一项。` });
    if (node.op === 'broadcast') {
      const dimensions = /\bdimensions=\{([^}]*)\}/.exec(suffix)?.[1];
      if (dimensions !== undefined) parts.push({ key:'dimensions', label:'广播维度', text:`dimensions={${dimensions}} 表示输入维度依次映射到输出的这些维度；其余输出维度由广播扩展。` });
    }
    if (!node.operands.length) parts = parts.filter(part => part.key !== 'operand');
  }
  if (node.op === 'fusion' && node.kind && suffix.includes(`kind=${node.kind}`)) {
    const meaning = node.kind === 'kOutput'
      ? '输出融合：融合计算的主操作后还可以有处理结果的指令，最终由内部 ROOT 给出输出。'
      : node.kind === 'kLoop'
        ? '循环融合：后端可把融合计算实现为一个循环。'
        : '这是该 fusion 的类型，用于指导后端如何实现融合计算。';
    parts.push({ key:'fusion-kind', label:'Fusion 类型', text:`kind=${node.kind}。${meaning}` });
  }
  const calledName = node.calls.calls;
  if (calledName && suffix.includes(`calls=%${calledName}`)) {
    const callee = context.module?.byName.get(calledName);
    const root = callee?.nodes.find(instruction => instruction.root);
    const parameterMapping = node.operands.map((operand, index) => `参数 ${index} ← %${operand}`).join('；');
    parts.push({
      key:'called-computation', label:'被调用的 computation',
      text:`calls=%${calledName} 指向此 fusion 内部执行的 computation。${parameterMapping ? `${parameterMapping}。` : ''}${root ? `它的 ROOT %${root.name} 定义 fusion 的结果。` : '它的 ROOT 定义 fusion 的结果。'}`
    });
  }
  if (node.op !== 'custom-call' && suffix.includes('metadata=')) {
    const opName = /\bop_name="([^"]+)"/.exec(suffix)?.[1];
    const sourceLine = /\bsource_line=(\d+)/.exec(suffix)?.[1];
    parts.push({ key:'metadata', label:'来源信息', text:`metadata 记录源操作信息。${opName ? `op_name=${opName}。` : ''}${sourceLine ? `源码行号 ${sourceLine}。` : ''}` });
  }
  if (node.op !== 'custom-call' && suffix.includes('backend_config=')) {
    parts.push({ key:'backend', label:'后端配置', text:'backend_config 是后端专用配置；展开完整指令可查看其字段。' });
  }
  const resultGroup: ResultGroup = tupleType ? {
    kind:'tuple', rawType:typeText,
    slots:tupleGroup?.slots || splitTuple(typeText).map(({body}, index) => ({
      key:['dest','source','context'][index] || `slot-${index}`,
      label:`第 ${index} 项${copyStart ? [' · 目标',' · 源',' · 上下文'][index] || '' : ''}`,
      raw:body.trim(),
      details:typeDetails(body)
    }))
  } : { kind:'array', rawType:typeText, details:typeDetails(typeText) };
  const argumentsHtml = node.op === 'constant' ? mark('literal', `(${args})`) :
    '(' + (node.op === 'parameter' ? mark('parameter-index', args) : operandHtml(args)) + ')';
  const html = escapeHtml(prefix[1] || '') + mark('name', prefix[2]) + escapeHtml(prefix[3]) + resultType +
    ' ' + mark('op', node.op) + argumentsHtml + suffixHtml(suffix);
  const metadata = extractHloMetadata(suffix);
  const compactHtml = compactEncodedBody(metadata ? html.replace(mark('metadata', metadata.raw), mark('metadata', 'metadata={…}')) : html);
  return { html, compactHtml: compactHtml === html ? undefined : compactHtml, parts, resultGroup };
}

export function layoutDiagram(node: HloNode): string | null {
  if (!['copy-start','parameter'].includes(node.op) || !/\{1,0:T\(8,128\)\(2,1\)/.test(node.type)) return null;
  const slots = splitTuple(node.type);
  if (!slots.length) return null;
  const space = (shape: string) => Number(/S\((\d+)\)/.exec(shape)?.[1] || 0);
  const location = (number: number) => number === 0 ? 'HBM' : number === 1 ? 'VMEM' : `S(${number})`;
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
