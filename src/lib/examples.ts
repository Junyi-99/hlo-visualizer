// HLO examples bundled from examples/<topology>/<program>.<stage>.hlo; each file is fetched only when chosen.
// Regenerate with scripts/dump_hlo.py on a TPU VM. Ids are <topology>/<program>.<stage>.
import dumpSource from '../../scripts/dump_hlo.py?raw';

const files = import.meta.glob('../../examples/*/*.hlo', { query: '?raw', import: 'default' }) as Record<string, () => Promise<string>>;

interface HloExample {
  id: string;
  topology: string;
  program: string;
  stage: 'after' | 'before';
  load: () => Promise<string>;
}

export const examples: HloExample[] = Object.entries(files)
  .map(([path, load]) => {
    const [topology, file] = path.split('/').slice(-2);
    const [program, stage] = file.replace(/\.hlo$/, '').split('.');
    const example: HloExample = {
      id: `${topology}/${program}.${stage}`,
      topology,
      program,
      stage: stage === 'before' ? 'before' : 'after',
      load
    };
    return example;
  })
  .sort((a, b) => a.topology.localeCompare(b.topology) || a.program.localeCompare(b.program));

export const hloTopologies = [...new Set(examples.map(item => item.topology))];

// The JAX program behind an example: its @program block in scripts/dump_hlo.py, with the 1-based line it starts on.
export function exampleSource(program: string): { code: string; line: number } | null {
  const lines = dumpSource.split('\n');
  const def = lines.findIndex(line => line.startsWith(`def ${program}(`));
  if (def < 0) return null;
  let start = def;
  while (start > 0 && lines[start - 1].trim()) start--;
  let end = def + 1;
  while (end < lines.length && (!lines[end] || /^\s/.test(lines[end]))) end++;
  return { code: lines.slice(start, end).join('\n').trimEnd(), line: start + 1 };
}

// Links from before the topology prefix (?example=attention.after) still resolve to the v6e-1 module.
export const findExample = (id: string) => examples.find(item => item.id === id) ?? examples.find(item => item.id === `v6e-1/${id}`);
