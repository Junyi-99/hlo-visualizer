// HLO examples bundled from examples/; each file is fetched only when chosen.
// Regenerate with scripts/dump_hlo.py on a TPU VM.
const files = import.meta.glob('../../examples/*/*.hlo', { query: '?raw', import: 'default' }) as Record<string, () => Promise<string>>;

interface HloExample {
  id: string;
  program: string;
  stage: 'after' | 'before';
  load: () => Promise<string>;
}

export const examples: HloExample[] = Object.entries(files)
  .map(([path, load]) => {
    const id = path
      .split('/')
      .pop()!
      .replace(/\.hlo$/, '');
    const [program, stage] = id.split('.');
    const example: HloExample = { id, program, stage: stage === 'before' ? 'before' : 'after', load };
    return example;
  })
  .sort((a, b) => a.program.localeCompare(b.program));
