// TPU names for the memory-space number S(n) in an HLO layout. A layout without S(n) is space 0.
// 0 and 5 are XLA-wide constants (default, host). 1, 2 and 6 are TPU backend numbers, checked on v6e:
//   S(1) VMEM  - where memory-space assignment puts prefetched buffers (copy-start destinations)
//   S(2) SFLAG - libtpu asserts FindMemorySpace(copy_start, {2}) == kSflag, and every copy-start/slice-start/
//                send/recv context element in the examples is S(2)
//   S(6) SMEM  - a Pallas operand with memory_space=SMEM is copied to S(6) before the kernel
//                (scripts/probe_memory_spaces.py); scalar loop counters and predicates also land there
const SPACES: Record<number, { label: string; text: string }> = {
  0: { label: 'HBM', text: '高带宽内存，TPU 的主存，容量最大。' },
  1: { label: 'VMEM', text: 'TensorCore 旁的片上向量内存，计算直接读写，容量小、速度快。' },
  2: { label: 'SFLAG', text: '同步标志存储，记录异步 DMA（copy-start、send 等）是否完成，对应的 *-done 据此等待。' },
  5: { label: 'HOST', text: '主机（CPU）内存。' },
  6: { label: 'SMEM', text: '标量内存，存放循环计数、下标、比较结果等标量，供标量单元使用。' },
};

export const memorySpaceLabel = (space: number) => SPACES[space]?.label ?? `S(${space})`;

// e.g. "VMEM（S(1)）：TensorCore 旁的…" or "HBM（未写 S(n)，即默认空间 S(0)）：…"
export function memorySpaceText(space: number, explicit = space !== 0) {
  const marker = explicit ? `S(${space})` : '未写 S(n)，即默认空间 S(0)';
  const known = SPACES[space];
  return known ? `${known.label}（${marker}）：${known.text}` : `S(${space})：TPU 后端私有的内存空间编号，含义未公开。`;
}

// Memory space of an array shape string such as "f32[8]{0:T(128)S(1)}".
export function shapeMemoryText(shape: string) {
  const match = /S\((\d+)\)/.exec(shape);
  return memorySpaceText(Number(match?.[1] ?? 0), !!match);
}
