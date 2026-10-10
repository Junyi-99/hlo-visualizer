export type Unit = 'MXU' | 'XLU' | 'DMA' | 'VLOAD' | 'VSTORE' | 'VPU' | 'SCALAR' | 'OTHER';

export interface LloInstruction {
  id: string;
  line: number;
  raw: string;
  opcode: string;
  output: string | null;
  references: string[];
  allocations: string[];
  unit: Unit;
  bundle: string | null;
  // Loop nesting of the bundle, from the `>` markers after its address.
  depth: number;
  region: string;
}

interface LloBundle {
  address: string;
  depth: number;
  instructions: LloInstruction[];
  region: string;
}

interface LloAllocation {
  name: string;
  space: string;
  shape: string;
  size: string;
  region: string;
}

interface LloRegion {
  name: string;
  label: string;
  instructions: LloInstruction[];
  bundles: LloBundle[];
  allocations: LloAllocation[];
}

export interface LloProgram {
  regions: LloRegion[];
  instructions: LloInstruction[];
  warnings: string[];
  // The HLO instruction this program implements, from the entry-bundle comment of final-bundle dumps.
  hlo: { name: string; opcode: string; operands: string[] } | null;
}

export const units: Unit[] = ['MXU', 'XLU', 'DMA', 'VLOAD', 'VSTORE', 'VPU', 'SCALAR', 'OTHER'];

function instructionUnit(opcode: string): Unit {
  const op = opcode.toLowerCase();
  if (/^(vmat|mxu)/.test(op) || /\.mxu\d*/.test(op)) return 'MXU';
  if (/^(vxpose|vpop\.trf|xlu)/.test(op) || /\.xlu\d*/.test(op)) return 'XLU';
  if (/^(dma\.|dma$)/.test(op)) return 'DMA';
  if (/^(vld|vload)/.test(op)) return 'VLOAD';
  if (/^(vst|vstore)/.test(op)) return 'VSTORE';
  if (/^(v[a-z]|vpop)/.test(op)) return 'VPU';
  if (/^(s[a-z]|int_to_ptr)/.test(op)) return 'SCALAR';
  return 'OTHER';
}

function unique(matches: IterableIterator<RegExpExecArray>): string[] {
  return [...new Set([...matches].map(match => match[0]))];
}

const withoutComments = (text: string) => text.replace(/\/\*[\s\S]*?\*\//g, ' ');

// A final bundle can contain many instructions separated by `;;`. Older dumps
// also put one instruction per line. Ignore separators inside multiline comments.
function bundleStatements(body: string): { raw: string; lineOffset: number }[] {
  const statements: { raw: string; lineOffset: number }[] = [];
  let start = 0;
  let comment = false;
  const add = (end: number) => {
    const raw = body.slice(start, end).trim();
    if (raw) statements.push({ raw, lineOffset: body.slice(0, start).split('\n').length - 1 });
  };
  for (let i = 0; i < body.length; i++) {
    if (!comment && body.slice(i, i + 2) === '/*') {
      comment = true;
      i++;
      continue;
    }
    if (comment && body.slice(i, i + 2) === '*/') {
      comment = false;
      i++;
      continue;
    }
    if (comment) continue;
    if (body.slice(i, i + 2) === ';;') {
      add(i);
      start = i + 2;
      i++;
      continue;
    }
    if (body[i] === '\n' && /^\s*%[\w.$-]+\s*=/.test(body.slice(i + 1))) {
      add(i);
      start = i + 1;
    }
  }
  add(body.length);
  return statements;
}

export function parseLlo(source: string): LloProgram {
  const entry = /entry bundle: %([\w.-]+) = ([\w-]+)\(([^)]*)\)/.exec(source);
  const program: LloProgram = {
    regions: [],
    instructions: [],
    warnings: [],
    hlo: entry
      ? {
          name: entry[1],
          opcode: entry[2],
          operands: entry[3]
            .split(',')
            .map(item => item.trim())
            .filter(Boolean)
        }
      : null
  };
  const lines = source.replace(/\r\n?/g, '\n').split('\n');
  let region: LloRegion | null = null;
  let bundle: LloBundle | null = null;
  let pendingBundle: { body: string; depth: number; comment: boolean; line: number } | null = null;

  const ensureRegion = () => {
    if (!region) {
      region = { name: 'program', label: 'Program', instructions: [], bundles: [], allocations: [] };
      program.regions.push(region);
    }
    return region;
  };

  const addInstruction = (raw: string, line: number) => {
    const trimmed = raw.trim();
    const code = withoutComments(trimmed).trim();
    const match = /^(?:(%[\w.$-]+)\s*=\s*)?([a-zA-Z][\w.-]*)(?=\s|\[|$)/.exec(code);
    if (!match) return false;
    const current = ensureRegion();
    const references = unique(code.slice(match[0].length).matchAll(/%[\w.$-]+/g)).filter(ref => ref !== match[1]);
    const instruction: LloInstruction = {
      id: `i${program.instructions.length}`,
      line,
      raw: trimmed,
      output: match[1] ?? null,
      opcode: match[2],
      references,
      allocations: unique(code.matchAll(/#allocation[\w.-]*/g)),
      unit: instructionUnit(match[2]),
      bundle: bundle?.address ?? null,
      depth: bundle?.depth ?? 0,
      region: current.name
    };
    current.instructions.push(instruction);
    program.instructions.push(instruction);
    bundle?.instructions.push(instruction);
    return true;
  };

  const consumeBundle = (text: string) => {
    if (!pendingBundle) return;
    let end = text.length;
    for (let i = 0; i < text.length; i++) {
      if (!pendingBundle.comment && text.slice(i, i + 2) === '/*') {
        pendingBundle.comment = true;
        i++;
        continue;
      }
      if (pendingBundle.comment && text.slice(i, i + 2) === '*/') {
        pendingBundle.comment = false;
        i++;
        continue;
      }
      if (pendingBundle.comment) continue;
      if (text[i] === '{') pendingBundle.depth++;
      if (text[i] === '}' && --pendingBundle.depth === 0) {
        end = i;
        break;
      }
    }
    pendingBundle.body += text.slice(0, end);
    if (end < text.length) {
      for (const statement of bundleStatements(pendingBundle.body))
        addInstruction(statement.raw, pendingBundle.line + statement.lineOffset);
      pendingBundle = null;
      bundle = null;
    }
  };

  for (let index = 0; index < lines.length; index++) {
    const line = lines[index].trim();
    if (pendingBundle) {
      consumeBundle(`\n${line}`);
      continue;
    }
    if (!line || line.startsWith('//')) continue;
    const regionMatch = /^\$([\w.-]+)\s*:\s*(?:#\{([^}]*)\})?/.exec(line);
    if (regionMatch) {
      region = {
        name: regionMatch[1],
        label: regionMatch[2] || regionMatch[1],
        instructions: [],
        bundles: [],
        allocations: []
      };
      program.regions.push(region);
      bundle = null;
      continue;
    }
    const allocationMatch = /^(#allocation[\w.-]*)\s+\[(.*)\]$/.exec(line);
    if (allocationMatch) {
      ensureRegion().allocations.push({
        name: allocationMatch[1],
        region: region!.name,
        shape: /\bshape\s*=\s*'([^']*)'/.exec(allocationMatch[2])?.[1] ?? '',
        space: /\bspace\s*=\s*([\w.-]+)/.exec(allocationMatch[2])?.[1] ?? '',
        size: /\bsize\s*=\s*([\w.$-]+)/.exec(allocationMatch[2])?.[1] ?? ''
      });
      continue;
    }
    const bundleMatch = /^(0x[\da-fA-F]+|\d+)\s*:\s*(>*)\s*\{(.*)$/.exec(line);
    if (bundleMatch) {
      const current = ensureRegion();
      bundle = { address: bundleMatch[1], depth: bundleMatch[2].length, region: current.name, instructions: [] };
      current.bundles.push(bundle);
      pendingBundle = { body: '', depth: 1, comment: false, line: index + 1 };
      consumeBundle(bundleMatch[3]);
      continue;
    }
    if (line === '}' || line === '{' || line.startsWith('#') || line.startsWith('/*')) continue;
    if (!region && !/^%[\w.$-]+\s*=/.test(line)) continue;
    if (!addInstruction(line, index + 1)) {
      // Wrapped operands continue the preceding instruction. Keep their exact text in the inspector.
      const previous = program.instructions.at(-1);
      if (previous && previous.region === region?.name) {
        previous.raw += `\n${line}`;
        const code = withoutComments(previous.raw);
        previous.references = unique(code.matchAll(/%[\w.$-]+/g)).filter(ref => ref !== previous.output);
        previous.allocations = unique(code.matchAll(/#allocation[\w.-]*/g));
      }
    }
  }

  if (pendingBundle) program.warnings.push(`Line ${pendingBundle.line}: unclosed bundle ${bundle?.address}`);

  if (!program.instructions.length)
    program.warnings.push('No LLO instructions found. Open a textual LLO pass dump or final_bundles.txt file.');
  return program;
}

export function definitionOf(program: LloProgram, instruction: LloInstruction, reference: string): LloInstruction | null {
  // Registers can be reused. The closest earlier definition in the same region is the relevant one.
  for (let i = program.instructions.indexOf(instruction) - 1; i >= 0; i--) {
    const candidate = program.instructions[i];
    if (candidate.region === instruction.region && candidate.output === reference) return candidate;
  }
  return null;
}
