// Plain-language guide for one LLO instruction: the opcode and its modifiers, every operand form,
// and the compiler's trailing comments. The TPU ISA is not public; opcode and modifier meanings
// come from how they are used in final_bundles dumps and from Google's published TPU material,
// and are worded accordingly.
import type { LloInstruction, Unit } from './parser';

export interface LloGuidePart {
  key: string;
  label: string;
  text: string;
}

export interface LloGuide {
  summary: string;
  html: string;
  parts: LloGuidePart[];
}

const HTML_ESCAPES: Record<string, string> = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
const escape = (value: string) => value.replace(/[&<>"']/g, c => HTML_ESCAPES[c]);
const mark = (key: string, text: string) => `<span class="hlo-part part-${key}" data-part="${key}" tabindex="0">${escape(text)}</span>`;

// ───── Opcodes ─────

const EUP = ' Runs on the EUP (transcendental unit); the result is read back with vpop.eup.';

// Keyed by the longest dotted prefix of the opcode that names the operation; the remaining tokens are modifiers.
const OPS: Record<string, string> = {
  vld: 'loads one vector register (8 sublanes × 128 lanes × 32 bit) from VMEM at the given address.',
  vst: 'stores a vector register to VMEM at the given address.',
  'vst.msk': 'stores a vector register to VMEM, writing only the elements selected by the mask operand.',
  vstv: 'broadcasts a scalar register value into a vector register.',
  vsel: 'elementwise select: takes on_true where the mask is set and on_false elsewhere.',
  vmov: 'moves an immediate or register into a vector register.',
  vmmov: 'moves an immediate into a vector mask register.',
  vphi: 'SSA phi: the value depends on which predecessor block ran (loop entry or back edge); realised as register copies.',
  sphi: 'SSA phi for a scalar register: the value depends on which predecessor block ran.',
  vpush: 'pushes a vector value toward the scalar unit; a later spop reads it as a scalar.',
  spop: 'pops the value sent by vpush into a scalar register.',
  vlaneseq: 'writes the lane index 0…127 into every sublane (a vector iota).',
  vcmask: 'builds a constant vector mask from an immediate; the comment shows the selected [sublanes, lanes].',
  'vc.u32': 'carry-out mask of an unsigned 32-bit add, used to emulate 64-bit arithmetic.',
  vadd: 'elementwise add.',
  vsub: 'elementwise subtract (first operand minus second).',
  vmul: 'elementwise multiply.',
  'vmul.u32.u64': 'unsigned 32 × 32 → 64-bit multiply; .low and .high select the half of the product.',
  vmax: 'elementwise maximum.',
  vmin: 'elementwise minimum.',
  vand: 'elementwise bitwise AND.',
  vor: 'elementwise bitwise OR.',
  vxor: 'elementwise bitwise XOR.',
  vshll: 'elementwise logical shift left.',
  vshrl: 'elementwise logical shift right (zero fill).',
  vshra: 'elementwise arithmetic shift right (sign fill).',
  vclz: 'counts leading zero bits of each element.',
  vpcnt: 'counts set bits of each element.',
  vnez: 'mask of the elements that are not zero.',
  vcmp: 'elementwise compare producing a vector mask.',
  vcvt: 'converts the element type by value, rounding as needed.',
  vtrunc: 'rounds each element toward zero.',
  vfloor: 'rounds each element down.',
  vceil: 'rounds each element up.',
  vclampa: 'clamps each element into [low, high].',
  vclamps: 'clamps each element into [−bound, +bound].',
  vweird: 'mask of the elements that are NaN or ±Inf.',
  vrcp: `reciprocal 1/x.${EUP}`,
  vrsqrt: `reciprocal square root 1/√x.${EUP}`,
  vpow2: `2 raised to x.${EUP}`,
  vlog2: `base-2 logarithm.${EUP}`,
  vtanh: `hyperbolic tangent.${EUP}`,
  verf: `error function erf(x).${EUP}`,
  vsinq: `sine of a range-reduced argument.${EUP}`,
  vcosq: `cosine of a range-reduced argument.${EUP}`,
  vpack: 'packs two 32-bit vectors into one vector of narrower elements.',
  vunpack: 'unpacks narrower elements of a packed vector into 32-bit elements.',
  vcombine: 'merges two vector registers into one.',
  vmpackc: 'packs two vector masks into one.',
  vmand: 'bitwise AND of two vector masks.',
  vmor: 'bitwise OR of two vector masks.',
  vmxor: 'bitwise XOR of two vector masks.',
  vmneg: 'inverts a vector mask.',
  'vrot.lane': 'rotates the 128 lanes by a dynamic amount on the XLU; the result is read with vpop.permute.',
  'vrot.slane': 'rotates the 8 sublanes by an immediate amount.',
  'vbcast.lane': 'broadcasts one lane across all lanes on the XLU; the result is read with vpop.permute.',
  'vset.pattern.permute': 'loads a lane-permutation pattern into the XLU permute control register.',
  vperm: 'permutes lanes on the XLU with the loaded pattern; the result is read with vpop.permute.',
  vxpose: 'transposes a block of vector registers on the XLU; .start/.cont/.end chain the steps and vpop.trf reads the results.',
  'vadd.xlane': 'cross-lane reduction: sums the 128 lanes on the XLU; the result is read with vpop.xlane.',
  vpop: 'reads a finished result from a unit FIFO into a vector register.',
  vmatprep: 'prepares a vector register for the MXU input path: .subr before a vmatpush of weights, .mubr before a vmatmul of activations.',
  vmatpush: 'pushes one vector of the stationary (weight) matrix into the MXU staging register.',
  vmatmul: 'streams a vector through the MXU against the held matrix; products accumulate in the MXU and are read with vpop.mrf.',
  smov: 'moves an immediate, address or register into a scalar register.',
  sadd: 'scalar add.',
  ssub: 'scalar subtract (first operand minus second).',
  smul: 'scalar multiply.',
  smin: 'scalar minimum.',
  sand: 'scalar bitwise AND.',
  sor: 'scalar bitwise OR.',
  sshll: 'scalar logical shift left.',
  sshrl: 'scalar logical shift right.',
  sshra: 'scalar arithmetic shift right.',
  scmp: 'scalar compare producing a predicate.',
  sld: 'loads a scalar from SMEM.',
  sst: 'stores a scalar to SMEM.',
  sfence: 'memory fence: earlier memory operations complete before later ones start.',
  'shalt.err': 'halts the core with an error when the predicate holds (a runtime check, for example DMA bounds).',
  'sbr.rel': 'branches to the target region when the predicate holds.',
  scalar_lea: 'computes an address (base plus offset) in the named memory space.',
  scalar_select: 'picks on_true or on_false by the predicate.',
  int_to_ptr: 'reinterprets an integer as an address in the named memory space.',
  pmov: 'moves an immediate into a predicate register.',
  por: 'OR of two predicates.',
  pnand: 'NAND of two predicates.',
  vsyncadd: 'adds an immediate to a sync flag (the counter a DMA increments as data lands).',
  vsyncpa: 'sync-flag operation on a DMA completion flag; issued before the DMA (exact semantics undocumented).',
  dma: 'starts an asynchronous DMA; completion is tracked on the sync flag given as dst_syncflagno.',
  'dma.general': 'starts a descriptor-driven (strided) DMA; completion is tracked on the sync flags given.',
  'dma.done.wait': 'blocks until the DMA tracked by the sync flag has delivered the given number of granules.',
  inlined_call_operand: 'binds one operand of the HLO instruction: its shape, operand index and the memory space it lives in.'
};

const MODIFIERS: Record<string, string> = {
  f32: '32-bit float elements',
  bf16: 'bfloat16 elements',
  hf16: 'IEEE half-precision elements',
  s32: 'signed 32-bit integer elements',
  u32: 'unsigned 32-bit integer elements',
  u64: '64-bit unsigned result',
  b32: '32-bit elements (bit pattern, type-agnostic)',
  b16: '16-bit elements',
  b8: '8-bit elements',
  s8: 'signed 8-bit elements',
  s4: 'signed 4-bit elements',
  msk: 'masked: a vector mask operand selects the active elements',
  xpose: 'pushes the matrix transposed',
  msra: 'matrix staging register A, the destination of vmatpush',
  gmra: 'the matrix register the multiply reads (the latched weights)',
  vlgmr: 'first latches the staged matrix (msra) into the multiply register (gmr)',
  mubr: 'MXU input path for streamed operands (vmatmul)',
  subr: 'MXU input path for stationary operands (vmatpush)',
  mrf: 'matmul result FIFO of the MXU',
  trf: 'transpose result FIFO of the XLU',
  eup: 'result FIFO of the EUP transcendental unit',
  permute: 'permute result of the XLU',
  xlane: 'cross-lane reduction result of the XLU',
  lane: 'operates along the 128-lane axis',
  slane: 'operates along the 8-sublane axis',
  c: 'compact packing: the halves are placed side by side',
  i: 'interleaved packing: elements of the two inputs alternate',
  l: 'low half of each packed element',
  h: 'high half of each packed element',
  low: 'low 32 bits of the result',
  high: 'high 32 bits of the result',
  start: 'first step of a multi-register XLU operation',
  cont: 'continuation step of a multi-register XLU operation',
  end: 'last step of a multi-register XLU operation',
  totalorder: 'integer compare (total order)',
  partialorder: 'IEEE float compare: anything compared with NaN is false',
  eq: 'equal',
  ne: 'not equal',
  lt: 'less than',
  le: 'less than or equal',
  gt: 'greater than',
  ge: 'greater than or equal',
  vmem: 'VMEM, the on-core vector memory',
  hbm: 'HBM, the off-core high-bandwidth memory',
  smem: 'SMEM, the scalar memory',
  sflag: 'sync-flag memory',
  hbm_to_vmem: 'copies from HBM into VMEM',
  vmem_to_hbm: 'copies from VMEM into HBM',
  vmem_to_smem: 'copies from VMEM into SMEM'
};

const UNIT_SUMMARY: Record<Unit, string> = {
  MXU: 'matrix unit operation.',
  XLU: 'cross-lane unit operation.',
  DMA: 'DMA operation.',
  VLOAD: 'vector load.',
  VSTORE: 'vector store.',
  VPU: 'vector operation.',
  SCALAR: 'scalar operation.',
  OTHER: 'operation without a recognised unit.'
};

function modifierText(token: string): string {
  if (MODIFIERS[token]) return MODIFIERS[token];
  const unit = /^(mxu|xlu)(\d)$/.exec(token);
  if (unit) return `executes on ${unit[1].toUpperCase()} ${unit[2]}`;
  if (/^\d$/.test(token)) return `byte ${token} of each 32-bit element`;
  if (token === 'start.end') return 'single-step XLU operation (start and end in one)';
  return 'modifier without a known meaning';
}

// Splits "vmatmul.mubr.f32.gmra.mxu0" into its base operation and modifier tokens.
function splitOpcode(opcode: string): { base: string; description: string; modifiers: string[] } {
  const tokens = opcode.split('.').filter(Boolean);
  for (let count = tokens.length; count > 0; count--) {
    const base = tokens.slice(0, count).join('.');
    const lookup = base.replace(/^(vmatpush)\d$/, '$1').replace(/^(vclamp[as])-f32$/, '$1');
    if (OPS[lookup]) {
      const modifiers = tokens.slice(count);
      if (/^vclamp[as]-f32$/.test(base)) modifiers.unshift('f32');
      return { base, description: OPS[lookup], modifiers };
    }
  }
  return { base: tokens[0], description: '', modifiers: tokens.slice(1) };
}

// ───── Operands ─────

const REGISTER_KIND: [RegExp, string][] = [
  [/^%vm/, 'vector mask register'],
  [/^%v/, 'vector register (8 sublanes × 128 lanes)'],
  [/^%s/, 'scalar register'],
  [/^%p/, 'predicate register'],
  [/^%\d+$/, 'handle of another instruction (no register value; links FIFO producers to their vpop)']
];
const registerKind = (name: string) => REGISTER_KIND.find(([pattern]) => pattern.test(name))?.[1] ?? 'register';

const ARGS: Record<string, string> = {
  vm: 'vector mask selecting the active elements',
  on_true_vy: 'value taken where the mask is set',
  on_false_vx: 'value taken where the mask is clear',
  vst_source: 'vector register whose contents are stored',
  vx: 'input vector register',
  width: 'transpose width in lanes',
  pattern: 'permute pattern loaded by vset.pattern.permute',
  source: 'vector being permuted',
  lhs_vy: 'left multiplicand',
  rhs_vx: 'right multiplicand',
  low: 'low 32 bits of the product, computed by the matching .low instruction',
  hbm: 'address in HBM',
  vmem: 'address in VMEM',
  smem: 'address in SMEM',
  src: 'source address',
  dst: 'destination address',
  size_in_granules: 'transfer size in 32-byte granules',
  dst_syncflagno: 'sync flag incremented as data lands at the destination',
  src_syncflagno: 'sync flag incremented as data leaves the source',
  dst_syncflagno_1: 'second destination sync flag',
  src_stride: 'source stride of the strided transfer',
  dst_stride: 'destination stride of the strided transfer',
  steps_per_stride: 'elements moved per stride step',
  stridedescaddr: 'address of the stride descriptor',
  overrides: 'descriptor field overrides',
  predicate: 'predicate deciding the selection',
  on_true: 'value when the predicate holds',
  on_false: 'value when the predicate does not hold'
};

const NOTES: [RegExp, string][] = [
  [/^Coalesced (load|store)\.$/, 'several adjacent accesses were merged into this one.'],
  [/^materialized constant$/, 'a constant placed in a register for later use.'],
  [/phi copy|copy for cssa/, 'register copy that realises an SSA phi, carrying a value around a loop or join.'],
  [/^Set PCR instruction$/, 'writes the XLU permute control register.'],
  [/^operand (\d+)/, 'operand $1 of the HLO instruction this program implements.'],
  [/dma-wait/, 'wait inserted by the emitter so the data has arrived before use.'],
  [/^iteration index, stage = (\d+)/, 'loop counter of a software-pipelined loop, pipeline stage $1.'],
  [/^smod/, 'strength-reduced modulo.'],
  [/^BoundsCheck/, 'runtime bounds check guarding the DMA named in the note.'],
  [/sublane mask fusion/, 'two partial-sublane loads merged into one with a combined mask.'],
  [/combine.*load/, 'several loads combined into one.'],
  [/^core id$/, 'reads the id of this core.'],
  [/loop exit test/, 'decides whether the loop continues.'],
  [/(Start|End).*region/, 'region boundary (regions are the compiler’s basic blocks).'],
  [/^\[\d+:\d+,\d+:\d+\]$/, 'the [sublanes, lanes] selected by the mask.'],
  [/^\(strided\) load/, 'strided load: offset, stride and sublane count.'],
  [/^(entry|exit) bundle:/, '$1 bundle of the HLO instruction’s program.'],
  [/^aligned_sublanes/, 'sublane alignment note.'],
  [/^base_bounds:/, 'DMA descriptor: bounds, strides and padding of the transfer.']
];

function noteText(note: string): string {
  for (const [pattern, text] of NOTES) {
    const match = pattern.exec(note);
    if (match) return text.replace(/\$(\d)/g, (_, index) => match[Number(index)] ?? '');
  }
  return 'compiler note.';
}

const sublanes = (mask: number) => {
  const bits = [...Array(8).keys()].filter(bit => mask & (1 << bit));
  if (bits.length === 8) return 'all 8 sublanes';
  if (!bits.length) return 'no sublane';
  const contiguous = bits.every((bit, index) => bit === bits[0] + index);
  return contiguous && bits.length > 1 ? `sublanes ${bits[0]}–${bits.at(-1)}` : `sublane${bits.length > 1 ? 's' : ''} ${bits.join(', ')}`;
};

function literalText(text: string, opcode: string): string {
  if (text.startsWith('$0x')) return `immediate ${text} = ${parseInt(text.slice(1), 16)}.`;
  const value = Number(text);
  if (/^\d+$/.test(text) && value >= 2 ** 31 && value < 2 ** 32 && /\.s32\b/.test(opcode))
    return `immediate ${text}, which is ${value - 2 ** 32} as a signed 32-bit value.`;
  return `immediate ${text}.`;
}

function addressText(space: string, inner: string): string {
  const unit = space === 'vmem' ? ' VMEM offsets count 512-byte sublane rows; a full vector register covers 8.' : '';
  const match = /^(.*?)\s*\+\s*\$0x([\da-f]+)$/i.exec(inner);
  if (match) return `${space.toUpperCase()} address: ${match[1]} plus 0x${match[2]} (${parseInt(match[2], 16)}).${unit}`;
  return `${space.toUpperCase()} address held by ${inner}.${unit}`;
}

// Returns the index just past the bracket that closes the one at `open`.
function closing(text: string, open: number): number {
  let depth = 0;
  for (let i = open; i < text.length; i++) {
    if (text[i] === '[') depth++;
    else if (text[i] === ']' && --depth === 0) return i + 1;
  }
  return text.length;
}

interface Scanner {
  html: string;
  parts: Map<string, LloGuidePart>;
}

function add(scanner: Scanner, key: string, label: string, text: string) {
  const existing = scanner.parts.get(key);
  if (!existing) scanner.parts.set(key, { key, label, text });
  else if (!existing.text.includes(text)) existing.text += ` ${text}`;
}

const VALUE = /\[#allocation[\w.-]*\]|%[\w.$-]+|\$0x[\da-fA-F]+|-?\d+(?:\.\d+)?(?:e[+-]?\d+)?/y;
const TOKENS: [RegExp, (scanner: Scanner, match: RegExpExecArray, opcode: string) => void][] = [
  [
    /\/\*([a-z_0-9]+)=\*\//iy,
    (scanner, match) => {
      const label = match[1];
      VALUE.lastIndex = match.index + match[0].length;
      const value = VALUE.exec(match.input);
      const text = match[0] + (value?.[0] ?? '');
      const key = `arg-${label.replace(/_/g, '-')}`;
      scanner.html += mark(key, text);
      add(scanner, key, label, `${ARGS[label] ?? 'named operand'}${value ? `: ${value[0]}` : ''}.`);
      match[0] = text;
    }
  ],
  [
    /\/\*\s*([^*]|\*(?!\/))*?\s*\*\//y,
    (scanner, match) => {
      const note = match[0].slice(2, -2).trim();
      scanner.html += mark('metadata', match[0]);
      add(scanner, 'metadata', 'Compiler note', `“${note.length > 100 ? `${note.slice(0, 100)}…` : note}”: ${noteText(note)}`);
    }
  ],
  [
    /\[(vmem|smem|hbm):\[([^\]]*)\]((?:\s+\w+:\$\w+)*)\s*\]/y,
    (scanner, match) => {
      const [, space, inner, modifiers] = match;
      scanner.html += escape('[') + mark('address', `${space}:[${inner}]`);
      add(scanner, 'address', 'Address', addressText(space, inner));
      for (const modifier of modifiers.matchAll(/(\w+):\$(\w+)/g)) {
        const [whole, name, value] = modifier;
        scanner.html += ' ';
        if (name === 'sm') {
          scanner.html += mark('mask', whole);
          add(scanner, 'mask', 'Sublane mask', `sm:$${value} enables ${sublanes(parseInt(value, 16))}; bit i enables sublane i.`);
        } else if (name === 'ss') {
          scanner.html += mark('stride', whole);
          add(scanner, 'stride', 'Sublane stride', `ss:$${value}: consecutive sublanes come from rows ${value} apart.`);
        } else {
          scanner.html += mark('sps', whole);
          add(scanner, 'sps', name, `${whole}: sublane access parameter (undocumented; appears with strided loads).`);
        }
      }
      scanner.html += escape(']');
    }
  ],
  [
    /\[shape:/y,
    (scanner, match) => {
      const end = closing(match.input, match.index);
      const text = match.input.slice(match.index, end);
      scanner.html += mark('shape', text);
      add(
        scanner,
        'shape',
        'Operand binding',
        'HLO shape of the operand, its position among the instruction’s operands, and whether it is an input or output.'
      );
      match[0] = text;
    }
  ],
  [
    /\[thread:\$(\d+)\]/y,
    (scanner, match) => {
      scanner.html += mark('thread', match[0]);
      add(scanner, 'thread', 'DMA thread', `issued on DMA queue ${match[1]}.`);
    }
  ],
  [
    /\[resolvable:\$\w+\]/y,
    (scanner, match) => {
      scanner.html += mark('flag', match[0]);
      add(scanner, 'flag', 'Flags', `${match[0]}: the address can be resolved statically.`);
    }
  ],
  [
    /\[(\d+)\/(\d+)\]/y,
    (scanner, match) => {
      scanner.html += mark('step', match[0]);
      add(scanner, 'step', 'Step', `step ${match[1]} of ${match[2]} in a multi-register XLU operation.`);
    }
  ],
  [
    /\[?#allocation[\w.-]*\]?/y,
    (scanner, match) => {
      scanner.html += mark('allocation', match[0]);
      add(
        scanner,
        'allocation',
        'Allocation',
        'buffer the compiler reserved inside this program; its space, shape and size are listed under Allocations.'
      );
    }
  ],
  [
    /\((!?)(%p[\w.$-]+)\)/y,
    (scanner, match) => {
      scanner.html += mark('predicate', match[0]);
      add(
        scanner,
        'predicate',
        'Predicate',
        `executes only when ${match[2]} is ${match[1] ? 'false' : 'true'}; otherwise the slot does nothing.`
      );
    }
  ],
  [
    /\((strided|narrow|short|[a-z_]+)\)/y,
    (scanner, match) => {
      const meaning: Record<string, string> = {
        strided: 'the access walks sublanes with the given stride',
        narrow: 'transpose of a block narrower than 128 lanes',
        short: 'transpose of a block shorter than a full register set'
      };
      scanner.html += mark('flag', match[0]);
      add(scanner, 'flag', 'Flags', `${match[0]}: ${meaning[match[1]] ?? 'compiler flag'}.`);
    }
  ],
  [
    /target\s*=\s*\$[\w.-]+/y,
    (scanner, match) => {
      scanner.html += mark('target', match[0]);
      add(scanner, 'target', 'Branch target', 'the region executed next when the branch is taken.');
    }
  ],
  [
    /%[\w.$-]+/y,
    (scanner, match) => {
      scanner.html += mark('operand', match[0]);
      add(scanner, 'operand', 'Register inputs', `${match[0]}: ${registerKind(match[0])}.`);
    }
  ],
  [
    /\$0x[\da-fA-F]+|-?\d+(?:\.\d+)?(?:e[+-]?\d+)?|-?inf|nan/y,
    (scanner, match, opcode) => {
      scanner.html += mark('literal', match[0]);
      add(scanner, 'literal', 'Immediates', literalText(match[0], opcode));
    }
  ]
];

export function lloGuide(instruction: LloInstruction): LloGuide {
  const { raw, opcode, unit } = instruction;
  const scanner: Scanner = { html: '', parts: new Map() };
  const head = /^(?:(%[\w.$-]+)\s*=\s*)?([a-zA-Z][\w.-]*)/.exec(raw);
  if (!head) return { summary: UNIT_SUMMARY[unit], html: escape(raw), parts: [] };

  if (head[1]) {
    scanner.html += mark('name', head[1]);
    const kind = /^%\d+$/.test(head[1])
      ? 'instruction id: this instruction produces no register value (a store, push, DMA or branch); later FIFO reads refer to it.'
      : `result ${registerKind(head[1])}.`;
    add(scanner, 'name', 'Result', kind);
    scanner.html += escape(raw.slice(head[1].length, head.index + head[0].length - head[2].length));
  }

  const { base, description, modifiers } = splitOpcode(head[2]);
  scanner.html += mark('op', base);
  add(scanner, 'op', base, description || UNIT_SUMMARY[unit]);
  modifiers.forEach((token, index) => {
    const key = `mod-${index}`;
    scanner.html += escape('.') + mark(key, token);
    add(scanner, key, `.${token}`, `${modifierText(token)}.`);
  });

  let position = head.index + head[0].length;
  while (position < raw.length) {
    let matched = false;
    for (const [pattern, handle] of TOKENS) {
      pattern.lastIndex = position;
      const match = pattern.exec(raw);
      if (!match) continue;
      handle(scanner, match, opcode);
      position += match[0].length;
      matched = true;
      break;
    }
    if (!matched) scanner.html += escape(raw[position++]);
  }

  const summary = description || UNIT_SUMMARY[unit];
  return { summary: summary[0].toUpperCase() + summary.slice(1), html: scanner.html, parts: [...scanner.parts.values()] };
}
