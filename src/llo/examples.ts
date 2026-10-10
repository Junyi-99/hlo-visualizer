// Final-bundle examples are loaded only when selected.
const files = import.meta.glob('../../examples/llo-v6e-*/*.llo', { query: '?raw', import: 'default' }) as Record<
  string,
  () => Promise<string>
>;

export interface LloExample {
  id: string;
  topology: string;
  program: string;
  load: () => Promise<string>;
}

export const lloExamples: LloExample[] = Object.entries(files)
  .map(([path, load]) => {
    const parts = path.split('/');
    const topology = parts.at(-2)!.replace(/^llo-/, '');
    const program = parts.at(-1)!.replace(/\.llo$/, '');
    return { id: `${topology}/${program}`, topology, program, load };
  })
  .sort((a, b) => a.topology.localeCompare(b.topology) || a.program.localeCompare(b.program));
