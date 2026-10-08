// TPU names for the memory-space number S(n) in an HLO layout. A layout without S(n) is space 0.
// 0 and 5 are XLA-wide constants (default, host). 1, 2 and 6 are TPU backend numbers, checked on v6e:
//   S(1) VMEM  - where memory-space assignment puts prefetched buffers (copy-start destinations)
//   S(2) SFLAG - libtpu asserts FindMemorySpace(copy_start, {2}) == kSflag, and every copy-start/slice-start/
//                send/recv context element in the examples is S(2)
//   S(6) SMEM  - a Pallas operand with memory_space=SMEM is copied to S(6) before the kernel
//                (scripts/probe_memory_spaces.py); scalar loop counters and predicates also land there
const SPACES: Record<number, { label: string; text: string }> = {
  0: { label: 'HBM', text: 'high-bandwidth memory, the TPU main memory and the largest.' },
  1: { label: 'VMEM', text: 'on-chip vector memory next to the TensorCore; compute reads and writes it directly. Small and fast.' },
  2: { label: 'SFLAG', text: 'sync flag memory; records whether an asynchronous DMA (copy-start, send, …) has finished, and the matching *-done waits on it.' },
  5: { label: 'HOST', text: 'host (CPU) memory.' },
  6: { label: 'SMEM', text: 'scalar memory for loop counters, indices, comparison results and other scalars used by the scalar unit.' },
};

export const memorySpaceLabel = (space: number) => SPACES[space]?.label ?? `S(${space})`;

// e.g. "VMEM (S(1)): on-chip vector memory…" or "HBM (no S(n), so the default space S(0)): …"
export function memorySpaceText(space: number, explicit = space !== 0) {
  const marker = explicit ? `S(${space})` : 'no S(n), so the default space S(0)';
  const known = SPACES[space];
  return known ? `${known.label} (${marker}): ${known.text}` : `S(${space}): a memory space number private to the TPU backend; its meaning is not public.`;
}

// Memory space of an array shape string such as "f32[8]{0:T(128)S(1)}".
export function shapeMemoryText(shape: string) {
  const match = /S\((\d+)\)/.exec(shape);
  return memorySpaceText(Number(match?.[1] ?? 0), !!match);
}
