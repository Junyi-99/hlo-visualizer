// Plain-language reference for HLO opcodes and instruction attributes.
// Semantics follow the XLA operation semantics doc: https://openxla.org/xla/operation_semantics

import type { Computation, HloModule, HloNode } from './types';
import { sourceStack } from './parser.ts';

const OPS: Record<string, string> = {
  // Inputs, structure
  parameter: 'declares one input of the current computation; it computes nothing.',
  constant: 'a literal value written in the instruction; it depends on no input.',
  iota: 'fills an integer sequence 0, 1, 2, … along one dimension (repeated along the others); often used for indices and masks.',
  tuple: 'groups several values into one tuple; no data is copied.',
  'get-tuple-element': 'takes one element out of an input tuple.',
  bitcast: 'reinterprets the same memory with a new shape or layout; no data moves.',
  reshape: 'changes the logical shape while keeping the element count and row-major order.',
  copy: 'copies the data into a new buffer, usually to change its physical layout or memory space.',
  'copy-start': 'starts an asynchronous copy (for example between HBM and VMEM); its tuple result goes to the matching copy-done.',
  'copy-done': 'waits for the asynchronous copy started by copy-start and yields the copied array.',
  'opt-barrier': 'optimization barrier: returns its input unchanged but stops the compiler from moving or merging work across it.',
  'after-all': 'joins tokens into a new token, used to order side-effecting operations such as send/recv.',
  'add-dependency': 'returns the first operand unchanged and adds a dependency on the second (usually a token) for ordering.',
  domain: 'marks a boundary for properties such as sharding; the data is unchanged.',

  // Elementwise arithmetic
  add: 'elementwise addition.',
  subtract: 'elementwise subtraction (first minus second).',
  multiply: 'elementwise multiplication.',
  divide: 'elementwise division (first divided by second); integers round toward zero.',
  remainder: 'elementwise remainder with the sign of the dividend (like C fmod).',
  power: 'elementwise power: first operand raised to the second.',
  maximum: 'elementwise maximum.',
  minimum: 'elementwise minimum.',
  negate: 'elementwise negation.',
  abs: 'elementwise absolute value (modulus for complex numbers).',
  sign: 'elementwise sign: -1 for negative, 0 for zero, 1 for positive.',
  clamp: 'clamps the second operand elementwise to [first operand, third operand].',
  atan2: 'elementwise atan2(first, second): the angle of the point (x = second, y = first).',
  exponential: 'elementwise e^x.',
  'exponential-minus-one': 'elementwise e^x − 1 (more accurate than subtracting when x is small).',
  log: 'elementwise natural logarithm.',
  'log-plus-one': 'elementwise ln(1 + x) (more accurate when x is small).',
  logistic: 'elementwise sigmoid: 1 / (1 + e^−x).',
  sqrt: 'elementwise square root.',
  rsqrt: 'elementwise reciprocal square root 1/√x.',
  cbrt: 'elementwise cube root.',
  tanh: 'elementwise hyperbolic tangent.',
  tan: 'elementwise tangent.',
  sine: 'elementwise sine.',
  cosine: 'elementwise cosine.',
  erf: 'elementwise error function erf(x).',
  floor: 'elementwise round down.',
  ceil: 'elementwise round up.',
  'round-nearest-even': 'elementwise round to the nearest integer, ties to even.',
  'round-nearest-afz': 'elementwise round to the nearest integer, ties away from zero.',
  'is-finite': 'elementwise test for a finite value (not inf or NaN); the result is pred.',
  real: 'real part of a complex value.',
  imag: 'imaginary part of a complex value.',
  complex: 'builds complex values from two real arrays (first is the real part, second the imaginary part).',
  convert: 'converts the element type (for example f32 → bf16) by value, rounding as needed.',
  'bitcast-convert': 'reinterprets the bits as another element type (for example f32 bits as s32); no numeric conversion.',
  'reduce-precision':
    'rounds floating-point values to fewer exponent and mantissa bits to emulate lower precision; the result type is unchanged.',
  'stochastic-convert': 'converts floating point to a lower-precision type with stochastic rounding.',

  // Bitwise and logical
  and: 'elementwise bitwise AND (logical AND for pred).',
  or: 'elementwise bitwise OR (logical OR for pred).',
  xor: 'elementwise bitwise XOR.',
  not: 'elementwise bitwise NOT (logical NOT for pred).',
  'shift-left': 'elementwise left shift by the second operand.',
  'shift-right-logical': 'elementwise logical right shift, filling with zeros.',
  'shift-right-arithmetic': 'elementwise arithmetic right shift, filling with the sign bit.',
  popcnt: 'elementwise count of 1 bits.',
  'count-leading-zeros': 'elementwise count of leading 0 bits.',
  compare: 'compares two operands elementwise; the result is a pred (boolean) array and direction says how to compare.',
  select: 'elementwise choice: where the first operand (pred) is true take the second operand, otherwise the third.',

  // Shape manipulation
  broadcast: 'expands the input to the result shape; dimensions says where the input axes go in the output.',
  transpose: 'reorders dimensions in the order given by dimensions.',
  reverse: 'reverses element order along the given dimensions.',
  slice: 'takes a sub-array with static start, limit and stride.',
  'dynamic-slice': 'takes a fixed-size block starting at indices known at run time; out-of-range starts are clamped into range.',
  'dynamic-update-slice': 'writes the second operand into the first at run-time indices and returns the whole updated array.',
  pad: 'pads each dimension at both ends and between elements with the second operand; negative padding removes elements.',
  concatenate: 'joins the inputs end to end along the dimension in dimensions.',
  gather:
    'collects slices from the input by index (like x[idx]); the *_dims attributes map indices and slices to input and output dimensions.',
  scatter: 'writes updates into the input at indices, or combines them with to_apply, for example x.at[idx].add(u).',

  // Reductions, windows, sorting
  reduce:
    'reduces along the dimensions in dimensions with the binary function to_apply (sum, max, …); those dimensions are removed from the result.',
  'reduce-window': 'reduces each sliding window with to_apply (pooling, prefix sums, …); window gives the window size and stride.',
  'select-and-scatter':
    'the backward pass of window pooling: select picks one position per window, then the source value is combined into it with scatter (for example the gradient of max pooling).',
  sort: 'sorts along the dimension in dimensions; with several operands they are permuted together, ordered by the to_apply comparator.',
  topk: 'takes the k largest (or smallest) values along the last dimension, with their indices.',
  map: 'applies the scalar function to_apply at every element position.',

  // Linear algebra
  dot: 'matrix multiplication / tensor contraction: multiplies and sums over the contracting dimensions, pairing batch dimensions.',
  convolution: 'convolution; on TPU, matrix multiplications are often written as convolutions too. See window and dim_labels.',
  fft: 'fast Fourier transform; see fft_type and fft_length.',
  cholesky: 'Cholesky decomposition: factors a symmetric positive-definite matrix into a triangular matrix times its transpose.',
  'triangular-solve': 'solves a triangular linear system (a is triangular); the options give the side and transposition.',

  // Control flow and calls
  while: "loop: runs body repeatedly until condition returns false; the loop state is this instruction's operand and result.",
  conditional: 'branch: the first operand (pred or branch index) selects one branch computation, which receives the remaining operands.',
  call: "calls the computation in to_apply with this instruction's operands as its parameters.",
  fusion: 'runs several instructions as one kernel; the computation in calls holds them.',
  'custom-call':
    'hands this step to an implementation registered with the backend (a Pallas kernel, a library call, …); the HLO text only shows the call boundary.',

  // Random numbers
  'rng-bit-generator': 'generates random bits from a state with the given algorithm; the result is (new state, random bits).',
  rng: 'generates random numbers from the given distribution.',
  'rng-get-and-update-state': 'reads and advances the global random number generator state.',

  // Collectives and communication
  'all-reduce': 'reduces values at the same position across a device group (sum, …); every device gets the result.',
  'all-reduce-start': 'start of an asynchronous all-reduce.',
  'all-reduce-done': 'waits for an asynchronous all-reduce.',
  'all-gather': "concatenates every device's data along one dimension; every device gets the full result.",
  'all-gather-start': 'start of an asynchronous all-gather.',
  'all-gather-done': 'waits for an asynchronous all-gather.',
  'reduce-scatter': 'reduces across the device group, then splits the result along one dimension across the devices.',
  'all-to-all': 'each device splits its data into blocks and sends one block to every device in the group.',
  'collective-permute': 'sends data point to point between devices following source_target_pairs.',
  'collective-permute-start': 'start of an asynchronous collective-permute.',
  'collective-permute-done': 'waits for an asynchronous collective-permute.',
  'collective-broadcast': "broadcasts one device's data to every device in the group.",
  'partition-id': "returns this device's SPMD partition id.",
  'replica-id': "returns this device's replica id.",
  send: 'sends data (to another device or the host); the result goes to send-done.',
  'send-done': 'waits for the matching send.',
  recv: 'receives data (from another device or the host); the result goes to recv-done.',
  'recv-done': 'waits for the matching recv and yields the received data.',
  infeed: 'reads data from the host input queue.',
  outfeed: 'writes data to the host output queue.',

  // Async wrappers, dynamic shapes
  'async-start': 'starts a computation (in calls) asynchronously.',
  'async-update': 'advances an asynchronous computation.',
  'async-done': 'waits for an asynchronous computation and takes its result.',
  'get-dimension-size': 'returns the run-time size of a dimension (dynamic shapes).',
  'set-dimension-size': 'sets the run-time size of a dimension (dynamic shapes).'
};

const ASYNC = /^(.*)-(start|done|update)$/;
const ASYNC_PHASE: Record<string, string> = { start: 'start', done: 'completion', update: 'update' };

export function opDescription(node: HloNode): string {
  const known = OPS[node.op];
  if (known) return `${node.op}: ${known}`;

  const [, base, phase] = ASYNC.exec(node.op) ?? [];
  if (base && OPS[base]) return `${node.op}: the ${ASYNC_PHASE[phase]} of an asynchronous ${base}. ${base} itself: ${OPS[base]}`;

  return `${node.op} is this instruction's HLO operation; see the XLA operation semantics doc.`;
}

// ───── Attribute helpers ─────

const list = (value: string) =>
  value
    .replace(/^\{|\}$/g, '')
    .split(',')
    .map(x => x.trim())
    .filter(Boolean);

const dimsText = (value: string) => {
  const dims = list(value);
  return dims.length ? `dimension${dims.length > 1 ? 's' : ''} ${dims.join(', ')}` : '(none)';
};

const names = (value: string) => [...value.matchAll(/%?([\w.-]+)/g)].map(m => `%${m[1]}`);

const ROOT_MEANING: Record<string, string> = {
  add: 'sum',
  maximum: 'max',
  minimum: 'min',
  multiply: 'product',
  and: 'logical AND',
  or: 'logical OR',
  compare: 'comparison',
  select: 'selection'
};

function rootSummary(name: string, module?: HloModule): string {
  const callee = module?.byName.get(name.replace(/^%/, ''));
  const root = callee?.nodes.find(node => node.root);
  if (!root) return '';

  const meaning = ROOT_MEANING[root.op];
  return ` Its ROOT is ${root.op}${meaning ? ` (${meaning})` : ''}.`;
}

// Letters in a convolution's dim_labels; any other character is a spatial dimension.
const ACTIVATION_ROLES: Record<string, string> = { b: 'batch', f: 'feature' };
const KERNEL_ROLES: Record<string, string> = { o: 'output feature', i: 'input feature' };

const dimRoles = (labels: string, kind: 'lhs' | 'rhs') => {
  const roles = kind === 'rhs' ? KERNEL_ROLES : ACTIVATION_ROLES;
  return [...labels].map((ch, index) => `dim ${index} = ${roles[ch] ?? `spatial ${ch}`}`).join(', ');
};

function dimLabels(value: string): string {
  const [inputs, output] = value.split('->');
  const [lhs, rhs] = (inputs || '').split('_');
  const legend = 'b = batch, f = feature, o/i = kernel output/input feature, digits = spatial';
  return (
    `dim_labels=${value} gives the role of each dimension by position (${legend}). ` +
    `Input ${lhs}: ${dimRoles(lhs || '', 'lhs')}; kernel ${rhs}: ${dimRoles(rhs || '', 'rhs')}; output ${output}: ${dimRoles(output || '', 'lhs')}.`
  );
}

const WINDOW_FIELDS: Record<string, string> = {
  size: 'window size',
  stride: 'stride',
  pad: 'edge padding (low_high)',
  lhs_dilate: 'input dilation (gaps between elements, used for transposed convolution)',
  rhs_dilate: 'window dilation (atrous convolution)',
  rhs_reversal: 'window reversal'
};

function windowText(value: string): string {
  const parts = [...value.replace(/^\{|\}$/g, '').matchAll(/(\w+)=(\S+)/g)].map(
    ([, key, v]) => `${WINDOW_FIELDS[key] || key} ${v.split('x').join(' × ')}`
  );
  return `window describes the sliding window, one value per spatial dimension separated by x: ${parts.join('; ')}.`;
}

function sliceText(value: string): string {
  const ranges = [...value.matchAll(/\[(-?\d+):(-?\d+)(?::(-?\d+))?\]/g)].map(([, start, limit, stride], index) => {
    const step = stride && stride !== '1' ? ` with stride ${stride}` : '';
    return `dim ${index} takes [${start}, ${limit})${step}`;
  });
  return `slice gives [start:limit:stride] per dimension, limit exclusive: ${ranges.join('; ')}.`;
}

function paddingText(value: string): string {
  const dims = value.split('x').map((dim, index) => {
    const [low, high, interior] = dim.split('_');
    const between = interior && interior !== '0' ? `, ${interior} between elements` : '';
    return `dim ${index}: ${low} before, ${high} after${between}`;
  });
  return `padding is low_high[_interior] per dimension, dimensions separated by x; negative values remove elements: ${dims.join('; ')}.`;
}

function metadataText(value: string, module?: HloModule): string {
  const opName = /\bop_name="([^"]+)"/.exec(value)?.[1];
  const opType = /\bop_type="([^"]+)"/.exec(value)?.[1];
  const file = /\bsource_file="([^"]+)"/.exec(value)?.[1];
  const line = /\bsource_line=(\d+)/.exec(value)?.[1];
  const frame = /\bstack_frame_id=(\d+)/.exec(value)?.[1];

  let source = '';
  if (file) source = ` Source file ${file}${line ? `, line ${line}` : ''}.`;
  else if (line) source = ` Source line ${line}.`;

  let frameText = '';
  if (frame) {
    const stack = module ? sourceStack(module, Number(frame)) : [];
    const chain = stack.map(f => `${f.func}:${f.line}`).join(' ← ');
    const where = stack.length ? ` Source: ${stack[0].file}:${stack[0].line} (${stack[0].func}); call chain ${chain}.` : '';
    frameText = ` stack_frame_id=${frame} points into the StackFrames table at the top of the module.${where}`;
  }

  return (
    'metadata records which front-end operation produced this instruction; it does not affect the computation.' +
    (opName ? ` op_name=${opName} (the operation path in JAX).` : '') +
    (opType ? ` op_type=${opType}.` : '') +
    source +
    frameText
  );
}

const BACKEND_FIELD_LIMIT = 8;

function backendConfigText(value: string, node: HloNode): string {
  const fields = [...new Set([...value.matchAll(/"(\w+)":/g)].map(m => m[1]))].slice(0, BACKEND_FIELD_LIMIT);
  const more = fields.length === BACKEND_FIELD_LIMIT ? ', …' : '';
  const fieldList = fields.length ? ` Fields: ${fields.join(', ')}${more}.` : '';
  const encodedBody =
    node.op === 'custom-call' && /"body"\s*:/.test(value)
      ? ' The encoded body is the serialized backend kernel; its internal dependencies cannot be read from the outer HLO.'
      : '';
  return `backend_config is backend-specific configuration (JSON) where the compiler records kernel, memory and scheduling decisions; it does not change what the data means.${fieldList}${encodedBody}`;
}

const COMPARE: Record<string, string> = {
  EQ: 'equal to',
  NE: 'not equal to',
  LT: 'less than',
  LE: 'less than or equal to',
  GT: 'greater than',
  GE: 'greater than or equal to'
};

const FFT: Record<string, string> = {
  FFT: 'forward complex-to-complex FFT',
  IFFT: 'inverse complex-to-complex FFT',
  RFFT: 'forward real-to-complex FFT (keeps the non-negative frequencies; the last dimension becomes n/2+1)',
  IRFFT: 'inverse complex-to-real FFT'
};

const FUSION_KIND: Record<string, string> = {
  kLoop: 'Loop fusion: elementwise work fused into one loop; each output element is computed independently.',
  kInput: 'Input fusion: built around a reduction, with the elementwise work feeding it fused in.',
  kOutput:
    'Output fusion: the main operation (convolution, matmul, …) can be followed by instructions that process its result; the inner ROOT gives the output.',
  kCustom: 'Custom fusion: the backend generates the kernel from a specific pattern (convolution templates, Pallas, TPU-specific code).'
};

interface AttributeContext {
  node: HloNode;
  module?: HloModule;
  computation?: Computation;
}

interface AttributeExplanation {
  key: string;
  label: string;
  text: string;
}

// Part keys kept stable for existing styling.
const KEY: Record<string, string> = {
  metadata: 'metadata',
  backend_config: 'backend',
  custom_call_target: 'target',
  operand_layout_constraints: 'constraints',
  dimensions: 'dimensions',
  index: 'tuple-index',
  kind: 'fusion-kind',
  calls: 'called-computation',
  dma_priority: 'priority'
};
export const attributeKey = (name: string) => KEY[name] || `attr-${name.replace(/[^\w-]/g, '').replace(/_/g, '-')}`;

const CUSTOM_CALL_TARGETS: Record<string, string> = {
  tpu_custom_call: 'a Pallas / Mosaic kernel on TPU',
  AssumeGatherIndicesInBound:
    'tells the compiler the gather indices are in range so bounds handling can be skipped; the data passes through unchanged',
  Sharding: 'a sharding annotation; the data passes through unchanged',
  SPMDFullToShardShape: 'converts a full shape to a per-shard shape for SPMD',
  SPMDShardToFullShape: 'converts a per-shard shape back to the full shape for SPMD',
  TopK: 'the k largest values and their indices',
  xla_python_cpu_callback: 'a callback into a Python function on the host',
  xla_ffi_python_cpu_callback: 'a callback into a Python function on the host',
  MoveToHost: 'moves data to host memory',
  MoveToDevice: 'moves data back to device memory'
};

// What `to_apply` means for each op that takes one.
const TO_APPLY_ROLES: Record<string, string> = {
  reduce: 'the reducer: combines two scalars into one',
  'reduce-window': 'the reducer applied within each window',
  'all-reduce': 'the reducer applied across devices',
  'reduce-scatter': 'the reducer applied across devices',
  scatter: 'the combiner: decides how the old value and the update combine (for example add; returning the update overwrites)',
  sort: 'the comparator: returns pred, true when the first element goes first',
  call: "the called computation; its parameters are this instruction's operands",
  map: 'the scalar function applied to every element'
};

const TRANSPOSE_A: Record<string, string> = {
  NO_TRANSPOSE: 'a',
  TRANSPOSE: 'the transpose of a'
};

const yes = (plain: string) => plain === 'true';

function dimensionsText(value: string, node: HloNode): string {
  const dims = dimsText(value);
  const byOp: Record<string, string> = {
    broadcast: `dimensions=${value} maps the input dimensions, in order, to output ${dims}; the other output dimensions are broadcast.`,
    reduce: `Reduces over input ${dims}; they are removed from the result.`,
    transpose: `Result dimension i is input dimension dimensions[i], i.e. input ${dims} in that order.`,
    reverse: `Reverses element order along ${dims}.`,
    concatenate: `Joins the inputs end to end along ${dims}.`,
    sort: `Sorts along ${dims}.`,
    'all-gather': `Concatenates the devices' data along ${dims}.`,
    'reduce-scatter': `After reducing, splits the result along ${dims} across devices.`,
    map: `Applies the function elementwise over ${dims}.`
  };
  return byOp[node.op] || `dimensions=${value} gives the dimensions this instruction works on: ${dims}.`;
}

function fusionCallsText(callee: string, node: HloNode, module?: HloModule): string {
  const mapping = node.operands.map((operand, index) => `parameter ${index} ← %${operand}`).join('; ');
  const root = module?.byName.get(callee.slice(1))?.nodes.find(instruction => instruction.root);
  const rootText = root ? ` Its ROOT %${root.name} defines the fusion's result.` : " Its ROOT defines the fusion's result.";
  return `calls=${callee} is the computation inside this fusion.${mapping ? ` ${mapping}.` : ''}${rootText}`;
}

export function explainAttribute(name: string, value: string, { node, module }: AttributeContext): AttributeExplanation {
  const key = attributeKey(name);
  const at = (label: string, text: string) => ({ key, label, text });
  const plain = value.replace(/^"|"$/g, '');
  const callee = names(value)[0] ?? plain;

  switch (name) {
    case 'metadata':
      return at('Source', metadataText(value, module));
    case 'backend_config':
      return at('Backend config', backendConfigText(value, node));
    case 'dma_priority':
      return at(
        'DMA priority',
        `dma_priority=${plain} is the priority of this DMA (asynchronous copy); how it affects scheduling is up to the backend.`
      );
    case 'custom_call_target': {
      const target = CUSTOM_CALL_TARGETS[plain];
      return at(
        'Backend target',
        `custom_call_target=${value} is the registered backend target${target ? `: ${target}` : ''}; its internals are not shown as a graph.`
      );
    }
    case 'operand_layout_constraints':
      return at(
        'Operand layouts',
        'operand_layout_constraints lists, in operand order, the layout the backend requires for each operand; the compiler converts the data before the call.'
      );
    case 'dimensions':
      return at(node.op === 'broadcast' ? 'Broadcast dimensions' : 'Dimensions', dimensionsText(value, node));
    case 'index':
      return at(
        'Tuple index',
        `index=${plain} takes element ${plain} (counting from 0) of ${node.operands[0] ? `%${node.operands[0]}` : 'the input tuple'}; this node's result type is that element's type.`
      );
    case 'kind':
      return at(
        'Fusion kind',
        `kind=${plain}. ${FUSION_KIND[plain] || 'The kind of this fusion, which guides how the backend implements it.'}`
      );
    case 'calls':
      if (node.op === 'fusion') return at('Called computation', fusionCallsText(callee, node, module));
      return at('Called computation', `calls=${callee} is the computation this instruction runs.${rootSummary(callee, module)}`);
    case 'to_apply': {
      const role = TO_APPLY_ROLES[node.op] || 'the sub-computation this instruction uses';
      return at('Applied function', `to_apply=${callee} is ${role}.${rootSummary(callee, module)}`);
    }
    case 'condition':
      return at('Loop condition', `condition=${callee} takes the current loop state and returns pred; the loop stops when it is false.`);
    case 'body':
      return at('Loop body', `body=${callee} takes the current loop state and returns the next one (same type as the state).`);
    case 'branch_computations': {
      const branches = names(value);
      return at(
        'Branches',
        `branch_computations lists ${branches.length} branches in order: ${branches.map((b, i) => `${i} → ${b}`).join(', ')}. The first operand picks the branch; an out-of-range index runs the last branch.`
      );
    }
    case 'true_computation':
      return at('True branch', `Runs ${callee} when the condition is true.`);
    case 'false_computation':
      return at('False branch', `Runs ${callee} when the condition is false.`);
    case 'select':
      return at('Select function', `select=${callee} compares elements within each window to pick one position (returns pred).`);
    case 'scatter':
      return at(
        'Scatter function',
        `scatter=${callee} combines the source value into the selected position.${rootSummary(callee, module)}`
      );
    case 'called_computations':
      return at(
        'Called computations',
        `called_computations=${value} lists the sub-computations this custom-call uses (for example the TopK comparator).`
      );
    case 'iota_dimension':
      return at(
        'Iota dimension',
        `iota_dimension=${plain}: counts 0, 1, 2, … along dimension ${plain}; values repeat along the other dimensions.`
      );
    case 'direction':
      return at('Comparison', `direction=${plain}: tests elementwise whether the first operand is ${COMPARE[plain] || plain} the second.`);
    case 'order': {
      const meaning =
        plain === 'TOTAL'
          ? 'compares floats with a total order, so NaN and -0/+0 also have a fixed place (-NaN < -inf < … < -0 < +0 < … < +inf < +NaN) and sorting is reproducible'
          : 'compares floats with the usual IEEE partial order; any comparison with NaN is false';
      return at('Comparison order', `order=${plain}: ${meaning}.`);
    }
    case 'type':
      return at(
        'Comparison type',
        `type=${plain} sets the numeric meaning of the comparison (FLOAT; TOTALORDER, which also orders NaN and ±0; SIGNED; UNSIGNED).`
      );
    case 'slice':
      return at('Slice range', sliceText(value));
    case 'padding':
      return at('Padding', paddingText(value));
    case 'window':
      return at('Window', windowText(value));
    case 'dim_labels':
      return at('Dimension roles', dimLabels(plain));
    case 'dynamic_slice_sizes':
      return at('Slice sizes', `dynamic_slice_sizes=${value}: the size taken in each dimension from the run-time start.`);
    case 'lhs_contracting_dims':
      return at(
        'LHS contracting dims',
        `Left operand ${dimsText(value)} are multiplied with the right contracting dimensions and summed; they disappear from the result.`
      );
    case 'rhs_contracting_dims':
      return at(
        'RHS contracting dims',
        `Right operand ${dimsText(value)} are multiplied with the left contracting dimensions and summed; they disappear from the result.`
      );
    case 'lhs_batch_dims':
      return at(
        'LHS batch dims',
        `Left operand ${dimsText(value)} are batch dimensions, paired with the right batch dimensions and computed independently; they come first in the result.`
      );
    case 'rhs_batch_dims':
      return at('RHS batch dims', `Right operand ${dimsText(value)} are batch dimensions, paired with the left batch dimensions.`);
    case 'operand_precision':
      return at(
        'Precision',
        `operand_precision=${value} gives the precision for each operand: default uses the backend default (on TPU, f32 is usually one bf16 pass); high/highest use several bf16 passes, and highest comes closest to full f32.`
      );
    case 'feature_group_count':
      return at(
        'Feature groups',
        `feature_group_count=${plain}: splits the input features into ${plain} groups convolved separately (grouped / depthwise convolution).`
      );
    case 'batch_group_count':
      return at('Batch groups', `batch_group_count=${plain}: groups the convolution by batch, common when computing kernel gradients.`);
    case 'offset_dims':
      return at('Offset dims', `offset_dims=${value}: result ${dimsText(value)} index into each gathered slice.`);
    case 'collapsed_slice_dims':
      return at(
        'Collapsed slice dims',
        `collapsed_slice_dims=${value}: input ${dimsText(value)} have size 1 in each slice and are dropped from the result.`
      );
    case 'start_index_map':
      return at(
        'Start index map',
        `start_index_map=${value}: component k of an index vector is the start index for input dimension start_index_map[k]; here the components map to input ${dimsText(value)}.`
      );
    case 'index_vector_dim':
      return at(
        'Index vector dim',
        `index_vector_dim=${plain}: dimension ${plain} of the indices array holds the components of each index vector${node.op === 'gather' || node.op === 'scatter' ? ' (equal to the rank of the indices means each index is a scalar)' : ''}.`
      );
    case 'slice_sizes':
      return at('Slice sizes', `slice_sizes=${value}: the size of the slice taken from the input for each index (per input dimension).`);
    case 'operand_batching_dims':
      return at('Operand batch dims', `operand_batching_dims=${value}: input dimensions paired batch by batch with the indices.`);
    case 'start_indices_batching_dims':
      return at('Index batch dims', `start_indices_batching_dims=${value}: indices dimensions paired with the input batch dimensions.`);
    case 'update_window_dims':
      return at('Update window dims', `update_window_dims=${value}: updates ${dimsText(value)} index into the window being written.`);
    case 'inserted_window_dims':
      return at(
        'Inserted window dims',
        `inserted_window_dims=${value}: input ${dimsText(value)} have size 1 in the window and are omitted from updates.`
      );
    case 'scatter_dims_to_operand_dims':
      return at(
        'Scatter dims to operand dims',
        `scatter_dims_to_operand_dims=${value}: component k of an index vector is the write start for input dimension scatter_dims_to_operand_dims[k]; here the components map to input ${dimsText(value)}.`
      );
    case 'input_batching_dims':
      return at('Input batch dims', `input_batching_dims=${value}: input dimensions paired batch by batch with the indices.`);
    case 'scatter_indices_batching_dims':
      return at('Index batch dims', `scatter_indices_batching_dims=${value}: indices dimensions paired with the input batch dimensions.`);
    case 'indices_are_sorted':
      return at(
        'Sorted indices',
        `indices_are_sorted=${plain}: declares the indices ${yes(plain) ? 'are sorted' : 'may be unsorted'}, which the compiler can use to optimize.`
      );
    case 'unique_indices':
      return at(
        'Unique indices',
        `unique_indices=${plain}: declares the indices ${yes(plain) ? 'never repeat (scatter needs no conflict handling)' : 'may repeat'}.`
      );
    case 'is_stable':
      return at(
        'Stable sort',
        `is_stable=${plain}: ${yes(plain) ? 'equal elements keep their original order' : 'the order of equal elements is not guaranteed'}.`
      );
    case 'k':
      return at('k', `k=${plain}: takes ${plain} elements.`);
    case 'largest':
      return at('Largest', `largest=${plain}: ${yes(plain) ? 'takes the k largest' : 'takes the k smallest'}.`);
    case 'fft_type':
      return at('FFT type', `fft_type=${plain}: ${FFT[plain] || plain}.`);
    case 'fft_length':
      return at('FFT length', `fft_length=${value}: the lengths of the innermost dimensions being transformed.`);
    case 'exponent_bits':
      return at('Exponent bits', `exponent_bits=${plain}: keeps ${plain} exponent bits after rounding.`);
    case 'mantissa_bits':
      return at('Mantissa bits', `mantissa_bits=${plain}: keeps ${plain} mantissa bits after rounding (bf16 has 7, f16 has 10).`);
    case 'lower':
      return at('Triangle', `lower=${plain}: uses the ${yes(plain) ? 'lower' : 'upper'} triangle.`);
    case 'left_side':
      return at('Side', `left_side=${plain}: ${yes(plain) ? 'solves op(a)·x = b' : 'solves x·op(a) = b'}.`);
    case 'unit_diagonal':
      return at(
        'Unit diagonal',
        `unit_diagonal=${plain}: ${yes(plain) ? 'assumes the diagonal of a is 1 and does not read it' : 'uses the diagonal of a'}.`
      );
    case 'transpose_a':
      return at('Transpose a', `transpose_a=${plain}: op(a) is ${TRANSPOSE_A[plain] ?? 'the conjugate transpose of a'}.`);
    case 'algorithm':
      return at(
        'RNG algorithm',
        `algorithm=${plain}: the algorithm used to generate random bits (rng_three_fry, rng_philox, or rng_default, chosen by the backend).`
      );
    case 'distribution':
      return at('Distribution', `distribution=${plain}: the distribution of the random numbers.`);
    case 'channel_id':
      return at(
        'Channel id',
        `channel_id=${plain}: the communication channel; a matching send/recv pair or one collective uses the same id.`
      );
    case 'is_host_transfer':
      return at(
        'Host transfer',
        `is_host_transfer=${plain}: ${yes(plain) ? 'the other end of this transfer is the host (CPU), for example the data behind jax.debug.print' : 'transfers between devices'}.`
      );
    case 'replica_groups':
      return at(
        'Replica groups',
        `replica_groups=${value}: the device groups taking part; devices in the same group communicate with each other.`
      );
    case 'use_global_device_ids':
      return at(
        'Global device ids',
        `use_global_device_ids=${plain}: the ids in replica_groups are ${yes(plain) ? 'global device ids' : 'replica ids'}.`
      );
    case 'source_target_pairs':
      return at(
        'Source-target pairs',
        `source_target_pairs=${value}: each {source, target} pair sends from the source device to the target device.`
      );
    case 'constrain_layout':
      return at('Constrain layout', `constrain_layout=${plain}: whether every device must use the same layout.`);
    case 'split_dimension':
      return at('Split dimension', `split_dimension=${plain}: all-to-all splits blocks along dimension ${plain}.`);
    case 'concat_dimension':
      return at('Concat dimension', `concat_dimension=${plain}: received blocks are joined along dimension ${plain}.`);
    case 'sharding':
      return at(
        'Sharding',
        `sharding=${value} describes how the value is laid out across devices: {replicated} puts a full copy on every device, maximal puts it on one device, devices=[…] gives the split.`
      );
    case 'frontend_attributes':
      return at(
        'Front-end attributes',
        'frontend_attributes are key-value pairs added by the front end (JAX) and carried through by XLA, for host transfers, scheduling hints and similar; they do not change the computation.'
      );
    case 'cross_program_prefetch_index':
      return at(
        'Cross-program prefetch',
        `cross_program_prefetch_index=${plain}: cross-program prefetch copies this parameter (usually weights) to faster memory right at program start, overlapping with compute; ${plain} numbers this prefetch.`
      );
    case 'control-predecessors':
      return at(
        'Control dependencies',
        `control-predecessors=${value}: these instructions must run before this one; only the order is constrained, no data flows.`
      );
    case 'custom_call_has_side_effect':
      return at(
        'Side effects',
        `custom_call_has_side_effect=${plain}: ${yes(plain) ? 'the call has side effects, so the compiler will not remove or duplicate it' : 'no side effects'}.`
      );
    case 'api_version':
      return at('API version', `api_version=${plain}: the calling convention of this custom-call.`);
    case 'output_to_operand_aliasing':
      return at(
        'Output aliasing',
        `output_to_operand_aliasing=${value}: an output shares its buffer with an operand (in-place update), written {output index}: (operand number, {index within operand}).`
      );
    case 'schedule':
    case 'custom_call_schedule':
      return at('Schedule hint', `${name}=${plain}: asks the scheduler to run this instruction early or late.`);
    case 'async_execution_thread':
      return at('Execution thread', `async_execution_thread=${value}: the execution thread the asynchronous computation runs on.`);
    case 'statistics':
      return at('Statistics', 'statistics records compiler statistics; it does not affect the computation.');
    case 'origin':
      return at('Origin', 'origin records which compiler step produced this instruction.');
    default: {
      const shown = value.length > 60 ? `${value.slice(0, 57)}…` : value;
      return at(name, `${name}=${shown} is an attribute of this ${node.op} instruction; see the XLA operation semantics doc.`);
    }
  }
}
