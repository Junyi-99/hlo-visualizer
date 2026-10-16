export interface BackendField {
  path: string;
  label: string;
  value: string;
  explanation: string;
  range?: { start: number; end: number };
}

export interface BackendConfig {
  raw: string;
  decoded: boolean;
  fields: BackendField[];
  error?: string;
}

function quotedEnd(text: string, start: number): number {
  for (let i = start + 1; i < text.length; i++) {
    if (text[i] === '\\') i++;
    else if (text[i] === '"') return i + 1;
  }
  return text.length;
}

// Skip strings, including metadata strings that merely mention backend_config.
function attributeStart(text: string): number {
  for (let i = 0; i < text.length; i++) {
    if (text[i] === '"') i = quotedEnd(text, i) - 1;
    else if ((i === 0 || /[\s,]/.test(text[i - 1])) && text.startsWith('backend_config', i)) {
      const match = /^backend_config\s*=\s*/.exec(text.slice(i));
      if (match) return i + match[0].length;
    }
  }
  return -1;
}

function valueEnd(text: string, start: number): number {
  if (text[start] === '"') return quotedEnd(text, start);
  if (text[start] !== '{' && text[start] !== '[') {
    const tail = /[,\s\]}]/.exec(text.slice(start));
    return tail ? start + tail.index : text.length;
  }
  let depth = 0;
  for (let i = start; i < text.length; i++) {
    if (text[i] === '"') i = quotedEnd(text, i) - 1;
    else if (text[i] === '{' || text[i] === '[') depth++;
    else if ((text[i] === '}' || text[i] === ']') && --depth === 0) return i + 1;
  }
  return text.length;
}

type JsonObject = Record<string, unknown>;
const object = (value: unknown): value is JsonObject => value !== null && typeof value === 'object' && !Array.isArray(value);
const display = (value: unknown): string => {
  if (typeof value === 'string') return value;
  try {
    return JSON.stringify(value);
  } catch {
    return 'Nested value — see raw JSON';
  }
};

// Index exact field slices in the supplied JSON rather than searching for repeated key names.
function fieldRanges(raw: string): Map<string, { start: number; end: number }> {
  const ranges = new Map<string, { start: number; end: number }>();
  const whitespace = (at: number) => {
    while (/\s/.test(raw[at] || '') && at < raw.length) at++;
    return at;
  };
  const walk = (at: number, path: string, depth: number): number => {
    at = whitespace(at);
    if (depth > 64) return valueEnd(raw, at);
    if (raw[at] !== '{' && raw[at] !== '[') return valueEnd(raw, at);
    const isArray = raw[at] === '[';
    const close = isArray ? ']' : '}';
    let index = 0;
    at = whitespace(at + 1);
    while (at < raw.length && raw[at] !== close) {
      const start = at;
      let key = String(index++);
      if (!isArray) {
        const end = quotedEnd(raw, at);
        key = JSON.parse(raw.slice(at, end));
        at = whitespace(whitespace(end) + 1);
      }
      const safeKey = key.replaceAll('\\', '\\\\').replaceAll('.', '\\.');
      const childPath = path ? `${path}.${safeKey}` : safeKey;
      at = walk(at, childPath, depth + 1);
      ranges.set(childPath, { start, end: at });
      at = whitespace(at);
      if (raw[at] === ',') at = whitespace(at + 1);
      else break;
    }
    return at + 1;
  };
  walk(0, '', 0);
  return ranges;
}

export function parseBackendConfig(text: string): BackendConfig | null {
  const start = attributeStart(text);
  if (start < 0) return null;
  let raw = text.slice(start, valueEnd(text, start));
  let decoded = false;
  let config: unknown;
  try {
    config = JSON.parse(raw);
    // HLO can store the JSON as an escaped string instead of an inline object.
    if (typeof config === 'string') {
      raw = config;
      decoded = true;
      config = JSON.parse(raw);
    }
  } catch {
    return { raw, decoded, fields: [], error: 'This backend configuration is not readable JSON. Its content is available below.' };
  }
  if (!object(config))
    return { raw, decoded, fields: [], error: 'This configuration is not a JSON object. Its content is available below.' };

  const ranges = fieldRanges(raw);
  const fields: BackendField[] = [];
  const add = (container: JsonObject, key: string, path: string, label: string, explanation: string) => {
    if (!Object.hasOwn(container, key)) return;
    fields.push({ path, label, value: display(container[key]), explanation, range: ranges.get(path) });
  };
  if (object(config.window_config)) {
    const window = config.window_config;
    const estimate = window.estimated_cycles;
    const positiveEstimate =
      (typeof estimate === 'number' || typeof estimate === 'string') && Number.isFinite(Number(estimate)) && Number(estimate) > 0;
    const validModel = typeof window.cost_model_type !== 'string' || !/INVALID|UNKNOWN|UNSPECIFIED/i.test(window.cost_model_type);
    add(
      window,
      'estimated_cycles',
      'window_config.estimated_cycles',
      'Estimated cycles',
      positiveEstimate && validModel
        ? 'Compiler estimate for this node’s recorded window configuration, in cycles. Not measured runtime; do not sum these as program time.'
        : 'The file records this value, but it does not establish a usable runtime estimate. Zero or an invalid model does not mean the operation is free.'
    );
    const descriptions: [string, string, string][] = [
      [
        'cost_model_type',
        'Cost model',
        'Model identifier recorded by the compiler. Meaning and availability can vary by compiler version.'
      ],
      [
        'buffering_level',
        'Buffering level',
        'Buffering setting chosen for this window configuration. The value alone does not prove how much transfer overlaps computation.'
      ],
      [
        'input_window_bounds',
        'Input window bounds',
        'Recorded input window bounds. These are compiler window coordinates; do not equate them with the logical tensor shape.'
      ],
      ['output_window_bounds', 'Output window bounds', 'Recorded output window bounds for this configuration.'],
      ['kernel_window_bounds', 'Kernel window bounds', 'Recorded kernel window bounds for this configuration.'],
      ['iteration_bounds', 'Iteration bounds', 'Recorded iteration bounds for the window configuration; not a measured execution count.']
    ];
    for (const [key, label, explanation] of descriptions) add(window, key, `window_config.${key}`, label, explanation);
  }
  if (object(config.convolution_algorithm_config)) {
    add(
      config.convolution_algorithm_config,
      'emitter',
      'convolution_algorithm_config.emitter',
      'Kernel emitter',
      'Backend emitter identifier selected by the compiler. This name is a compiler implementation detail, not a hardware instruction.'
    );
  }
  add(
    config,
    'dma_priority',
    'dma_priority',
    'DMA priority',
    'Recorded DMA priority value. The numeric ordering and scheduling policy are backend specific; this is not a transfer duration.'
  );
  if (Array.isArray(config.used_scoped_memory_configs)) {
    config.used_scoped_memory_configs.forEach((region, index) => {
      if (!object(region)) return;
      for (const [key, label, explanation] of [
        [
          'memory_space',
          'Space ID',
          'Raw scoped-memory space identifier. It is not decoded using HLO S(n) or the internal libtpu MemorySpace enum.'
        ],
        ['offset', 'Offset', 'Offset recorded for this scoped memory region.'],
        ['size', 'Size', 'Size recorded for this scoped memory region. This is not the node’s total memory usage.']
      ])
        add(region, key, `used_scoped_memory_configs.${index}.${key}`, `Region ${index + 1} · ${label}`, explanation);
    });
  }
  return { raw, decoded, fields };
}
