// Plain-language reference for HLO opcodes and instruction attributes.
// Semantics follow the XLA operation semantics doc: https://openxla.org/xla/operation_semantics
import type { Computation, HloModule, HloNode } from './types';
import { sourceStack } from './parser.ts';

const OPS: Record<string, string> = {
  // Inputs, structure
  parameter: '声明当前 computation 的一个输入；不计算新数据。',
  constant: '直接给出一个常量值，不依赖任何输入。',
  iota: '生成沿某一维递增的整数序列 0, 1, 2, …（其余维度上重复），常用来做下标、掩码。',
  tuple: '把多个值打包成一个 tuple；不复制数据，只是把它们组合成一个结果。',
  'get-tuple-element': '从输入 tuple 中取出指定位置的元素。',
  bitcast: '用新的形状或布局重新解释同一块内存，不移动、不复制数据。',
  reshape: '改变逻辑形状但保持元素总数和行优先顺序不变。',
  copy: '把数据复制到一个新缓冲区，通常是为了改变物理布局或内存位置。',
  'copy-start': '启动异步复制（例如 HBM 与 VMEM 之间）；结果 tuple 交给对应的 copy-done。',
  'copy-done': '等待对应 copy-start 发起的异步复制完成，产出复制后的数组。',
  'opt-barrier': '优化屏障：原样返回输入，但阻止编译器把计算跨过这里重排或合并。',
  'after-all': '合并多个 token，产生一个新 token，用来给有副作用的操作（send/recv 等）排顺序。',
  'add-dependency': '原样返回第一个操作数，并额外依赖第二个操作数（通常是 token），用于排顺序。',
  domain: '标记分片等属性的边界，不改变数据。',
  // Elementwise arithmetic
  add: '逐元素相加。', subtract: '逐元素相减（第一个减第二个）。', multiply: '逐元素相乘。', divide: '逐元素相除（第一个除以第二个）；整数时向零取整。',
  remainder: '逐元素取余，结果符号与被除数相同（类似 C 的 fmod）。', power: '逐元素求幂：第一个操作数的第二个操作数次方。',
  maximum: '逐元素取较大值。', minimum: '逐元素取较小值。', negate: '逐元素取相反数。', abs: '逐元素取绝对值（复数时取模）。',
  sign: '逐元素取符号：负数为 -1，零为 0，正数为 1。', clamp: '把第二个操作数逐元素截断到 [第一个操作数, 第三个操作数] 区间。',
  atan2: '逐元素计算 atan2(第一个, 第二个)，即点 (x=第二个, y=第一个) 的辐角。',
  exponential: '逐元素计算 e^x。', 'exponential-minus-one': '逐元素计算 e^x − 1（x 很小时比直接相减更精确）。',
  log: '逐元素自然对数。', 'log-plus-one': '逐元素计算 ln(1 + x)（x 很小时更精确）。', logistic: '逐元素 sigmoid：1 / (1 + e^−x)。',
  sqrt: '逐元素平方根。', rsqrt: '逐元素平方根的倒数 1/√x。', cbrt: '逐元素立方根。', tanh: '逐元素双曲正切。', tan: '逐元素正切。',
  sine: '逐元素正弦。', cosine: '逐元素余弦。', erf: '逐元素误差函数 erf(x)。',
  floor: '逐元素向下取整。', ceil: '逐元素向上取整。', 'round-nearest-even': '逐元素四舍五入到最近整数，恰好一半时取偶数。',
  'round-nearest-afz': '逐元素四舍五入，恰好一半时远离零。', 'is-finite': '逐元素判断是否为有限数（不是 inf 或 NaN），结果为 pred。',
  real: '取复数的实部。', imag: '取复数的虚部。', complex: '用两个实数数组组成复数（第一个为实部，第二个为虚部）。',
  convert: '逐元素转换元素类型（例如 f32 → bf16），按数值转换并按需舍入。',
  'bitcast-convert': '按位重新解释元素类型（例如 f32 的比特当作 s32），不做数值转换。',
  'reduce-precision': '把浮点值舍入到更少的指数位和尾数位来模拟低精度，结果类型不变。',
  'stochastic-convert': '用随机舍入把浮点转换成较低精度类型。',
  // Bitwise and logical
  and: '逐元素按位与（pred 时为逻辑与）。', or: '逐元素按位或（pred 时为逻辑或）。', xor: '逐元素按位异或。', not: '逐元素按位取反（pred 时为逻辑非）。',
  'shift-left': '逐元素左移，移位数来自第二个操作数。', 'shift-right-logical': '逐元素逻辑右移，高位补 0。',
  'shift-right-arithmetic': '逐元素算术右移，高位补符号位。', popcnt: '逐元素统计二进制中 1 的个数。', 'count-leading-zeros': '逐元素统计二进制最高位起连续 0 的个数。',
  compare: '逐元素比较两个操作数，结果是 pred（布尔）数组；比较方式由 direction 指定。',
  select: '逐元素三选一：第一个操作数（pred）为真时取第二个操作数，否则取第三个。',
  // Shape manipulation
  broadcast: '把输入值扩展到结果形状；dimensions 指定输入轴在输出中的对应位置。',
  transpose: '按 dimensions 给出的顺序重排维度。', reverse: '沿指定维度翻转元素顺序。',
  slice: '按静态的起点、终点和步长截取一块子数组。',
  'dynamic-slice': '从运行时给出的起始下标截取固定大小的一块；起点越界时会被截回到合法范围。',
  'dynamic-update-slice': '把第二个操作数写入第一个操作数中从运行时下标开始的位置，返回更新后的整个数组。',
  pad: '在各维的两端以及元素之间填充值（第二个操作数）；负的填充量表示裁掉。',
  concatenate: '沿 dimensions 指定的维度把输入依次拼接。',
  gather: '按索引从输入中收集切片（如 x[idx]）；各 *_dims 属性描述索引和切片怎样对应到输入、输出维度。',
  scatter: '按索引把 updates 写回（或用 to_apply 合并进）输入的对应位置，例如 x.at[idx].add(u)。',
  // Reductions, windows, sorting
  reduce: '沿 dimensions 指定的维度，用 to_apply 给出的二元函数规约（求和、求最大值等）；这些维度从结果中去掉。',
  'reduce-window': '在滑动窗口内用 to_apply 规约（池化、前缀和等）；窗口大小和步长见 window。',
  'select-and-scatter': '窗口池化的反向：在每个窗口内用 select 选出一个位置，再把 source 的值用 scatter 函数累加到那里（如最大池化的梯度）。',
  sort: '沿 dimensions 指定的维度排序；多个操作数时按同一顺序一起重排，先后由 to_apply 比较函数决定。',
  topk: '沿最后一维取最大（或最小）的 k 个值及其下标。',
  map: '对每个元素位置应用 to_apply 给出的标量函数。',
  // Linear algebra
  dot: '矩阵乘法 / 张量缩并：沿 contracting 维相乘求和，batch 维逐批对应。',
  convolution: '卷积；TPU 上矩阵乘法也常写成卷积形式。窗口与维度含义见 window 和 dim_labels。',
  fft: '快速傅里叶变换；变换类型和长度见 fft_type、fft_length。',
  cholesky: 'Cholesky 分解：把对称正定矩阵分解为下三角（或上三角）矩阵与其转置之积。',
  'triangular-solve': '解三角线性方程组（a 为三角矩阵）；方向与是否转置见选项。',
  // Control flow and calls
  while: '循环：反复执行 body，直到 condition 返回 false；循环状态就是这条指令的操作数和结果。',
  conditional: '条件分支：根据第一个操作数（pred 或分支下标）选择执行一个分支 computation，其余操作数作为该分支的参数。',
  call: '调用 to_apply 指定的 computation，参数依次为本指令的操作数。',
  fusion: '把多条指令合并成一个内核执行；内部计算见 calls 指向的 computation。',
  'custom-call': '把这一步交给后端注册的自定义实现（如 Pallas 内核、库函数），文本 HLO 只显示调用边界。',
  // Random numbers
  'rng-bit-generator': '用指定算法从状态生成随机比特，结果为（新状态, 随机比特）。',
  rng: '按指定分布生成随机数。',
  'rng-get-and-update-state': '读取并推进随机数生成器的全局状态。',
  // Collectives and communication
  'all-reduce': '在设备组内对同一位置的值做规约（如求和），每个设备都得到结果。',
  'all-reduce-start': '异步 all-reduce 的开始。', 'all-reduce-done': '等待异步 all-reduce 完成。',
  'all-gather': '在设备组内沿某一维拼接各设备的数据，每个设备都得到完整结果。',
  'all-gather-start': '异步 all-gather 的开始。', 'all-gather-done': '等待异步 all-gather 完成。',
  'reduce-scatter': '先在设备组内规约，再把结果沿某一维切开分给各设备。',
  'all-to-all': '每个设备把数据切成若干块分别发给组内其他设备。',
  'collective-permute': '按 source_target_pairs 在设备之间点对点发送数据。',
  'collective-permute-start': '异步 collective-permute 的开始。', 'collective-permute-done': '等待异步 collective-permute 完成。',
  'collective-broadcast': '把一个设备的数据广播给组内所有设备。',
  'partition-id': '返回当前设备在 SPMD 分区中的编号。', 'replica-id': '返回当前设备的副本编号。',
  send: '发送数据（给另一设备或主机）；结果交给 send-done。', 'send-done': '等待对应的 send 完成。',
  recv: '接收数据（来自另一设备或主机）；结果交给 recv-done。', 'recv-done': '等待对应的 recv 完成，并产出收到的数据。',
  infeed: '从主机输入队列读取数据。', outfeed: '把数据写入主机输出队列。',
  'async-start': '启动一个异步执行的 computation（calls 指定）。', 'async-update': '推进异步计算。', 'async-done': '等待异步计算完成并取得结果。',
  'get-dimension-size': '返回某一维的运行时大小（动态形状）。', 'set-dimension-size': '设置某一维的运行时大小（动态形状）。',
  'optimization-barrier': '优化屏障：原样返回输入，阻止编译器跨过这里重排。',
};

const ASYNC = /^(.*)-(start|done|update)$/;

export function opDescription(node: HloNode): string {
  const known = OPS[node.op];
  if (known) return `${node.op}：${known}`;
  const async = ASYNC.exec(node.op);
  if (async && OPS[async[1]]) return `${node.op}：异步版本的 ${async[1]} 的${async[2] === 'start' ? '开始' : async[2] === 'done' ? '完成' : '推进'}部分。${async[1]} 本身：${OPS[async[1]]}`;
  return `${node.op} 是这条指令的 HLO 操作；说明见 XLA operation semantics 文档。`;
}

// ───── Attribute helpers ─────

const list = (value: string) => value.replace(/^\{|\}$/g, '').split(',').map(x => x.trim()).filter(Boolean);
const dimsText = (value: string) => { const dims = list(value); return dims.length ? `第 ${dims.join('、')} 维` : '（无）'; };
const names = (value: string) => [...value.matchAll(/%?([\w.-]+)/g)].map(m => `%${m[1]}`);

function rootSummary(name: string, module?: HloModule): string {
  const callee = module?.byName.get(name.replace(/^%/, ''));
  const root = callee?.nodes.find(node => node.root);
  if (!root) return '';
  const meaning: Record<string, string> = { add: '求和', maximum: '取最大值', minimum: '取最小值', multiply: '求积', and: '逻辑与', or: '逻辑或', compare: '比较', select: '选择' };
  return `它的 ROOT 是 ${root.op}${meaning[root.op] ? `（${meaning[root.op]}）` : ''}。`;
}

function dimLabels(value: string): string {
  const [inputs, output] = value.split('->');
  const [lhs, rhs] = (inputs || '').split('_');
  const describe = (labels: string, kind: 'lhs' | 'rhs') => [...labels].map((ch, index) => {
    const role = kind === 'rhs' ? ch === 'o' ? '输出特征' : ch === 'i' ? '输入特征' : `空间维 ${ch}` : ch === 'b' ? 'batch' : ch === 'f' ? '特征' : `空间维 ${ch}`;
    return `第 ${index} 维=${role}`;
  }).join('，');
  return `dim_labels=${value} 按位置标出每个张量各维的角色（b=batch，f=特征，o/i=卷积核的输出/输入特征，数字=空间维）。输入 ${lhs}：${describe(lhs || '', 'lhs')}；卷积核 ${rhs}：${describe(rhs || '', 'rhs')}；输出 ${output}：${describe(output || '', 'lhs')}。`;
}

function windowText(value: string): string {
  const fields: Record<string, string> = { size: '窗口大小', stride: '步长', pad: '边缘填充（前_后）', lhs_dilate: '输入膨胀（元素间插空，用于转置卷积）', rhs_dilate: '窗口膨胀（空洞卷积）', rhs_reversal: '窗口翻转' };
  const parts = [...value.replace(/^\{|\}$/g, '').matchAll(/(\w+)=(\S+)/g)].map(([, key, v]) => `${fields[key] || key} ${v.split('x').join(' × ')}`);
  return `window 描述滑动窗口，每个字段按空间维用 x 分隔：${parts.join('；')}。`;
}

function sliceText(value: string): string {
  const ranges = [...value.matchAll(/\[(-?\d+):(-?\d+)(?::(-?\d+))?\]/g)].map(([, start, limit, stride], index) =>
    `第 ${index} 维取 [${start}, ${limit})${stride && stride !== '1' ? `，步长 ${stride}` : ''}`);
  return `slice 按维给出 [起点:终点:步长]，终点不含：${ranges.join('；')}。`;
}

function paddingText(value: string): string {
  const dims = value.split('x').map((dim, index) => {
    const [low, high, interior] = dim.split('_');
    return `第 ${index} 维前补 ${low}、后补 ${high}${interior && interior !== '0' ? `、元素间插 ${interior}` : ''}`;
  });
  return `padding 每维写成 前_后[_中间]，维之间用 x 分隔；负数表示裁掉：${dims.join('；')}。`;
}

const COMPARE: Record<string, string> = { EQ: '等于', NE: '不等于', LT: '小于', LE: '小于等于', GT: '大于', GE: '大于等于' };
const FFT: Record<string, string> = { FFT: '复数到复数的正向 FFT', IFFT: '复数到复数的逆 FFT', RFFT: '实数到复数的正向 FFT（只保留非负频率，最后一维长度变为 n/2+1）', IRFFT: '复数到实数的逆 FFT' };
const FUSION_KIND: Record<string, string> = {
  kLoop: '循环融合：逐元素等操作融合成一个循环，每个输出元素独立计算。',
  kInput: '输入融合：以规约等操作为核心，把它前面的逐元素运算一起融合进来。',
  kOutput: '输出融合：融合计算的主操作（如卷积、矩阵乘）之后还可以有处理结果的指令，最终由内部 ROOT 给出输出。',
  kCustom: '自定义融合：由后端按特定模式生成内核（如卷积模板、Pallas、TPU 专用实现）。',
};

export interface AttributeContext { node: HloNode; module?: HloModule; computation?: Computation }
export interface AttributeExplanation { key: string; label: string; text: string }

// Part keys kept stable for existing styling and tests.
const KEY: Record<string, string> = {
  metadata: 'metadata', backend_config: 'backend', custom_call_target: 'target', operand_layout_constraints: 'constraints',
  dimensions: 'dimensions', index: 'tuple-index', kind: 'fusion-kind', calls: 'called-computation', dma_priority: 'priority',
};
export const attributeKey = (name: string) => KEY[name] || `attr-${name.replace(/[^\w-]/g, '').replace(/_/g, '-')}`;

export function explainAttribute(name: string, value: string, { node, module }: AttributeContext): AttributeExplanation {
  const key = attributeKey(name);
  const at = (label: string, text: string) => ({ key, label, text });
  const plain = value.replace(/^"|"$/g, '');
  switch (name) {
    case 'metadata': {
      const opName = /\bop_name="([^"]+)"/.exec(value)?.[1];
      const opType = /\bop_type="([^"]+)"/.exec(value)?.[1];
      const file = /\bsource_file="([^"]+)"/.exec(value)?.[1];
      const line = /\bsource_line=(\d+)/.exec(value)?.[1];
      const frame = /\bstack_frame_id=(\d+)/.exec(value)?.[1];
      const stack = frame && module ? sourceStack(module, Number(frame)) : [];
      const where = stack.length ? `源码位置 ${stack[0].file}:${stack[0].line}（${stack[0].func}）；调用链 ${stack.map(f => `${f.func}:${f.line}`).join(' ← ')}。` : '';
      return at('来源信息', `metadata 记录这条指令来自哪个前端操作，不影响计算。${opName ? `op_name=${opName}（JAX 中的操作路径）。` : ''}${opType ? `op_type=${opType}。` : ''}${file ? `源文件 ${file}${line ? ` 第 ${line} 行` : ''}。` : line ? `源码行号 ${line}。` : ''}${frame ? `stack_frame_id=${frame} 指向模块开头 StackFrames 表里的调用栈。${where}` : ''}`);
    }
    case 'backend_config': {
      const fields = [...value.matchAll(/"(\w+)":/g)].map(m => m[1]).filter((v, i, a) => a.indexOf(v) === i).slice(0, 8);
      const custom = node.op === 'custom-call' && /"body"\s*:/.test(value) ? '其中编码的 body 是后端内核的序列化内容，不能仅凭外层 HLO 推导出内部指令依赖。' : '';
      return at('后端配置', `backend_config 是后端专用的配置（JSON），编译器用它记录内核、内存和调度相关的决定，不改变数据语义。${fields.length ? `包含字段：${fields.join('、')}${fields.length === 8 ? ' 等' : ''}。` : ''}${custom}`);
    }
    case 'dma_priority':
      return at('DMA 优先级', `dma_priority=${plain} 是这次 DMA（异步复制）的优先级值；数值如何影响调度由具体后端决定。`);
    case 'custom_call_target': {
      const targets: Record<string, string> = {
        tpu_custom_call: 'TPU 上的 Pallas / Mosaic 内核',
        AssumeGatherIndicesInBound: '告诉编译器 gather 的下标都在范围内，可省去越界处理；数据原样返回',
        Sharding: '分片标注，数据原样返回', SPMDFullToShardShape: 'SPMD 分片时把完整形状转成分片形状',
        SPMDShardToFullShape: 'SPMD 分片时把分片形状转回完整形状', TopK: '取前 k 大的值与下标',
        xla_python_cpu_callback: '回调到主机上的 Python 函数', xla_ffi_python_cpu_callback: '回调到主机上的 Python 函数',
        MoveToHost: '把数据移到主机内存', MoveToDevice: '把数据移回设备内存',
      };
      return at('后端目标', `custom_call_target=${value} 是注册的后端调用目标${targets[plain] ? `：${targets[plain]}` : ''}；它本身不展开内部计算图。`);
    }
    case 'operand_layout_constraints':
      return at('输入布局约束', 'operand_layout_constraints 依输入顺序列出后端要求的每个操作数的布局；编译器会在调用前把数据转换成这些布局。');
    case 'dimensions': {
      const dims = dimsText(value);
      const text: Record<string, string> = {
        broadcast: `dimensions=${value} 表示输入维度依次映射到输出的${dims}；其余输出维度由广播扩展。`,
        reduce: `沿输入的${dims}规约，这些维度从结果中去掉。`,
        transpose: `结果的第 i 维取输入的第 dimensions[i] 维，即依次取输入的${dims}。`,
        reverse: `沿${dims}翻转元素顺序。`, concatenate: `沿${dims}把各输入首尾相接。`, sort: `沿${dims}排序。`,
        'all-gather': `沿${dims}拼接各设备的数据。`, 'reduce-scatter': `规约后沿${dims}切开分给各设备。`, map: `在${dims}上逐元素应用函数。`,
      };
      return at(node.op === 'broadcast' ? '广播维度' : '作用维度', text[node.op] || `dimensions=${value} 指定这条指令作用的维度：${dims}。`);
    }
    case 'index':
      return at('读取 tuple 的位置', `index=${plain} 从 ${node.operands[0] ? `%${node.operands[0]}` : '输入 tuple'} 取出第 ${plain} 项（从 0 开始）；此节点的结果类型对应那一项。`);
    case 'kind':
      return at('Fusion 类型', `kind=${plain}。${FUSION_KIND[plain] || '这是该 fusion 的类型，用于指导后端如何实现融合计算。'}`);
    case 'calls': {
      const callee = names(value)[0];
      if (node.op !== 'fusion') return at('被调用的 computation', `calls=${callee} 是这条指令执行的 computation。${rootSummary(callee, module)}`);
      const mapping = node.operands.map((operand, index) => `参数 ${index} ← %${operand}`).join('；');
      const root = module?.byName.get(callee.slice(1))?.nodes.find(instruction => instruction.root);
      return at('被调用的 computation', `calls=${callee} 指向此 fusion 内部执行的 computation。${mapping ? `${mapping}。` : ''}${root ? `它的 ROOT %${root.name} 定义 fusion 的结果。` : '它的 ROOT 定义 fusion 的结果。'}`);
    }
    case 'to_apply': {
      const callee = names(value)[0];
      const role: Record<string, string> = {
        reduce: '规约函数：把两个标量合并成一个', 'reduce-window': '窗口内的规约函数', 'all-reduce': '设备间的规约函数', 'reduce-scatter': '设备间的规约函数',
        scatter: '合并函数：决定原值与更新值如何合并（如相加；直接覆盖时返回更新值）', sort: '比较函数：返回 pred，为真表示第一个元素应排在前面',
        call: '被调用的 computation，参数依次为本指令的操作数', map: '对每个元素应用的标量函数',
      };
      return at('调用的函数', `to_apply=${callee} 是${role[node.op] || '这条指令使用的子计算'}。${rootSummary(callee, module)}`);
    }
    case 'condition': return at('循环条件', `condition=${names(value)[0]} 接收当前循环状态，返回 pred；为 false 时循环结束。`);
    case 'body': return at('循环体', `body=${names(value)[0]} 接收当前循环状态，返回下一轮的状态（类型与状态相同）。`);
    case 'branch_computations': {
      const branches = names(value);
      return at('分支', `branch_computations 按顺序列出 ${branches.length} 个分支：${branches.map((b, i) => `${i} → ${b}`).join('，')}。第一个操作数给出分支下标，越界时执行最后一个分支。`);
    }
    case 'true_computation': return at('真分支', `条件为 true 时执行 ${names(value)[0]}。`);
    case 'false_computation': return at('假分支', `条件为 false 时执行 ${names(value)[0]}。`);
    case 'select': return at('选择函数', `select=${names(value)[0]} 在每个窗口内比较元素，选出一个位置（返回 pred）。`);
    case 'scatter': return at('累加函数', `scatter=${names(value)[0]} 把 source 的值合并到被选中的位置。${rootSummary(names(value)[0], module)}`);
    case 'called_computations': return at('关联的 computation', `called_computations=${value} 列出这个 custom-call 会用到的子计算（如 TopK 的比较函数）。`);
    case 'iota_dimension': return at('递增的维度', `iota_dimension=${plain}：沿第 ${plain} 维填入 0, 1, 2, …；其余维度上值相同。`);
    case 'direction': return at('比较方式', `direction=${plain}：逐元素判断第一个操作数是否${COMPARE[plain] || plain}第二个操作数。`);
    case 'order': return at('比较顺序', `order=${plain}：${plain === 'TOTAL' ? '按全序比较浮点数，NaN 和 -0/+0 也有确定的先后（-NaN < -inf < … < -0 < +0 < … < +inf < +NaN），排序时结果稳定可复现' : '按普通 IEEE 部分序比较浮点数，NaN 与任何值比较都为 false'}。`);
    case 'type': return at('比较类型', `type=${plain} 指定比较的数值语义（FLOAT、TOTALORDER 让 NaN 和 ±0 也有确定顺序、SIGNED、UNSIGNED）。`);
    case 'slice': return at('切片范围', sliceText(value));
    case 'padding': return at('填充量', paddingText(value));
    case 'window': return at('窗口', windowText(value));
    case 'dim_labels': return at('维度角色', dimLabels(plain));
    case 'dynamic_slice_sizes': return at('切片大小', `dynamic_slice_sizes=${value}：从运行时起点开始，各维截取的大小。`);
    case 'lhs_contracting_dims': return at('左侧缩并维', `左操作数的${dimsText(value)}与右侧缩并维逐一相乘求和，从结果中消失。`);
    case 'rhs_contracting_dims': return at('右侧缩并维', `右操作数的${dimsText(value)}与左侧缩并维逐一相乘求和，从结果中消失。`);
    case 'lhs_batch_dims': return at('左侧 batch 维', `左操作数的${dimsText(value)}是 batch 维，与右侧 batch 维逐一对应、各批独立计算，排在结果最前面。`);
    case 'rhs_batch_dims': return at('右侧 batch 维', `右操作数的${dimsText(value)}是 batch 维，与左侧 batch 维逐一对应。`);
    case 'operand_precision': return at('计算精度', `operand_precision=${value} 依次给出每个操作数的计算精度：default 用后端默认精度（TPU 上 f32 通常按一遍 bf16 计算），high/highest 用多遍 bf16 提高精度，highest 最接近完整 f32。`);
    case 'feature_group_count': return at('特征分组', `feature_group_count=${plain}：把输入特征分成 ${plain} 组分别卷积（分组卷积 / depthwise）。`);
    case 'batch_group_count': return at('batch 分组', `batch_group_count=${plain}：按 batch 分组卷积，常见于卷积核梯度的计算。`);
    case 'offset_dims': return at('切片在结果中的维度', `offset_dims=${value}：结果中的${dimsText(value)}对应每次取出的切片内部的维度。`);
    case 'collapsed_slice_dims': return at('折叠的切片维', `collapsed_slice_dims=${value}：输入的${dimsText(value)}在切片中大小为 1，结果里去掉这些维。`);
    case 'start_index_map': return at('索引对应的输入维', `start_index_map=${value}：索引向量的第 k 个分量是输入第 start_index_map[k] 维的起始下标；这里各分量依次对应输入的${dimsText(value)}。`);
    case 'index_vector_dim': return at('索引向量所在维', `index_vector_dim=${plain}：索引数组的第 ${plain} 维存放每个索引向量的分量${node.op === 'gather' || node.op === 'scatter' ? '（等于索引数组维数时表示每个索引是标量）' : ''}。`);
    case 'slice_sizes': return at('切片大小', `slice_sizes=${value}：每个索引从输入取出的切片大小（按输入维度）。`);
    case 'operand_batching_dims': return at('输入 batch 维', `operand_batching_dims=${value}：输入中与索引逐批对应的维度。`);
    case 'start_indices_batching_dims': return at('索引 batch 维', `start_indices_batching_dims=${value}：索引中与输入 batch 维对应的维度。`);
    case 'update_window_dims': return at('更新窗口维', `update_window_dims=${value}：updates 的${dimsText(value)}对应写入窗口内部的维度。`);
    case 'inserted_window_dims': return at('插入的窗口维', `inserted_window_dims=${value}：输入的${dimsText(value)}在窗口中大小为 1，updates 里省略了这些维。`);
    case 'scatter_dims_to_operand_dims': return at('索引对应的输入维', `scatter_dims_to_operand_dims=${value}：索引向量的第 k 个分量是输入第 scatter_dims_to_operand_dims[k] 维的写入起点；这里各分量依次对应输入的${dimsText(value)}。`);
    case 'input_batching_dims': return at('输入 batch 维', `input_batching_dims=${value}：输入中与索引逐批对应的维度。`);
    case 'scatter_indices_batching_dims': return at('索引 batch 维', `scatter_indices_batching_dims=${value}：索引中与输入 batch 维对应的维度。`);
    case 'indices_are_sorted': return at('下标已排序', `indices_are_sorted=${plain}：声明下标${plain === 'true' ? '已排好序' : '不保证有序'}，编译器可据此优化。`);
    case 'unique_indices': return at('下标唯一', `unique_indices=${plain}：声明下标${plain === 'true' ? '互不重复（scatter 无需处理冲突）' : '可能重复'}。`);
    case 'is_stable': return at('稳定排序', `is_stable=${plain}：${plain === 'true' ? '相等元素保持原有相对顺序' : '相等元素的顺序不保证'}。`);
    case 'k': return at('取前 k 个', `k=${plain}：取 ${plain} 个元素。`);
    case 'largest': return at('取最大', `largest=${plain}：${plain === 'true' ? '取最大的 k 个' : '取最小的 k 个'}。`);
    case 'fft_type': return at('FFT 类型', `fft_type=${plain}：${FFT[plain] || plain}。`);
    case 'fft_length': return at('FFT 长度', `fft_length=${value}：参与变换的最内层各维的长度。`);
    case 'exponent_bits': return at('指数位数', `exponent_bits=${plain}：舍入后保留 ${plain} 位指数。`);
    case 'mantissa_bits': return at('尾数位数', `mantissa_bits=${plain}：舍入后保留 ${plain} 位尾数（例如 bf16 是 7，f16 是 10）。`);
    case 'lower': return at('三角方向', `lower=${plain}：使用${plain === 'true' ? '下' : '上'}三角部分。`);
    case 'left_side': return at('求解方向', `left_side=${plain}：${plain === 'true' ? '解 op(a)·x = b' : '解 x·op(a) = b'}。`);
    case 'unit_diagonal': return at('单位对角', `unit_diagonal=${plain}：${plain === 'true' ? '假定 a 的对角元为 1，不读取对角线' : '使用 a 的对角元'}。`);
    case 'transpose_a': return at('是否转置 a', `transpose_a=${plain}：op(a) ${plain === 'NO_TRANSPOSE' ? '就是 a' : plain === 'TRANSPOSE' ? '是 a 的转置' : '是 a 的共轭转置'}。`);
    case 'algorithm': return at('随机算法', `algorithm=${plain}：生成随机比特使用的算法（如 rng_three_fry、rng_philox、rng_default 由后端决定）。`);
    case 'distribution': return at('分布', `distribution=${plain}：随机数服从的分布。`);
    case 'channel_id': return at('通道编号', `channel_id=${plain}：通信通道编号；配对的 send/recv 或同一集体操作用同一编号匹配。`);
    case 'is_host_transfer': return at('主机传输', `is_host_transfer=${plain}：${plain === 'true' ? '这次传输的另一端是主机（CPU），例如 jax.debug.print 的数据回传' : '在设备之间传输'}。`);
    case 'replica_groups': return at('设备分组', `replica_groups=${value}：参与集体通信的设备分组；同一组内的设备之间互相通信。`);
    case 'use_global_device_ids': return at('全局设备编号', `use_global_device_ids=${plain}：replica_groups 中的编号${plain === 'true' ? '是全局设备 ID' : '是副本号'}。`);
    case 'source_target_pairs': return at('收发对', `source_target_pairs=${value}：每对 {源, 目标} 表示从源设备发送到目标设备。`);
    case 'constrain_layout': return at('约束布局', `constrain_layout=${plain}：是否要求各设备使用相同的布局。`);
    case 'split_dimension': return at('切分维', `split_dimension=${plain}：all-to-all 沿第 ${plain} 维切块发送。`);
    case 'concat_dimension': return at('拼接维', `concat_dimension=${plain}：收到的块沿第 ${plain} 维拼接。`);
    case 'sharding': return at('分片', `sharding=${value} 描述该值在设备间如何分布：{replicated} 每个设备一份完整副本，maximal 只放在指定设备，devices=[…] 给出切分方式。`);
    case 'frontend_attributes': return at('前端属性', `frontend_attributes 是前端（JAX）附加的键值信息，XLA 原样携带，用于主机传输、调度提示等；不改变计算。`);
    case 'cross_program_prefetch_index': return at('跨程序预取', `cross_program_prefetch_index=${plain}：跨程序预取，把这个参数（通常是权重）在程序一开始就复制到更快的内存，复制与计算重叠；${plain} 是这个预取的序号。`);
    case 'control-predecessors': return at('控制依赖', `control-predecessors=${value}：这些指令必须在本指令之前执行；只约束顺序，不传递数据。`);
    case 'custom_call_has_side_effect': return at('有副作用', `custom_call_has_side_effect=${plain}：${plain === 'true' ? '该调用有副作用，编译器不会删除或重复执行它' : '没有副作用'}。`);
    case 'api_version': return at('调用约定', `api_version=${plain}：custom-call 使用的调用约定版本。`);
    case 'output_to_operand_aliasing': return at('输出别名', `output_to_operand_aliasing=${value}：输出与某个操作数共享同一缓冲区（原地更新），格式为 {输出索引}: (操作数序号, {操作数内索引})。`);
    case 'schedule': case 'custom_call_schedule': return at('调度提示', `${name}=${plain}：建议调度器尽早或尽晚执行这条指令。`);
    case 'async_execution_thread': return at('异步执行线程', `async_execution_thread=${value}：异步计算在哪个执行线程上运行。`);
    case 'statistics': return at('统计信息', `statistics 记录编译过程中的统计数据，不影响计算。`);
    case 'origin': return at('来源', `origin 记录这条指令由哪个编译步骤产生。`);
    default: return at(name, `${name}=${value.length > 60 ? `${value.slice(0, 57)}…` : value} 是这条 ${node.op} 指令的属性；具体含义见 XLA operation semantics 文档。`);
  }
}

export const KNOWN_ATTRIBUTES = new Set(['metadata', 'backend_config', 'dma_priority', 'custom_call_target', 'operand_layout_constraints', 'dimensions', 'index', 'kind', 'calls',
  'to_apply', 'condition', 'body', 'branch_computations', 'true_computation', 'false_computation', 'select', 'scatter', 'called_computations', 'iota_dimension', 'direction',
  'type', 'order', 'slice', 'padding', 'window', 'dim_labels', 'dynamic_slice_sizes', 'lhs_contracting_dims', 'rhs_contracting_dims', 'lhs_batch_dims', 'rhs_batch_dims',
  'operand_precision', 'feature_group_count', 'batch_group_count', 'offset_dims', 'collapsed_slice_dims', 'start_index_map', 'index_vector_dim', 'slice_sizes',
  'operand_batching_dims', 'start_indices_batching_dims', 'update_window_dims', 'inserted_window_dims', 'scatter_dims_to_operand_dims', 'input_batching_dims',
  'scatter_indices_batching_dims', 'indices_are_sorted', 'unique_indices', 'is_stable', 'k', 'largest', 'fft_type', 'fft_length', 'exponent_bits', 'mantissa_bits',
  'lower', 'left_side', 'unit_diagonal', 'transpose_a', 'algorithm', 'distribution', 'channel_id', 'is_host_transfer', 'replica_groups', 'use_global_device_ids',
  'source_target_pairs', 'constrain_layout', 'split_dimension', 'concat_dimension', 'sharding', 'frontend_attributes', 'cross_program_prefetch_index',
  'control-predecessors', 'custom_call_has_side_effect', 'api_version', 'output_to_operand_aliasing', 'schedule', 'custom_call_schedule', 'async_execution_thread',
  'statistics', 'origin']);
