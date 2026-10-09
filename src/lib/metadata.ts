export interface HloMetadata {
  raw: string;
  start: number;
  end: number;
  fields: { name: string; value: string }[];
}

export function extractHloMetadata(text: string): HloMetadata | null {
  const match = /\bmetadata=\{/.exec(text);
  if (!match) return null;
  const opening = match.index + match[0].length - 1;
  let depth = 0,
    quoted = false,
    escaped = false,
    closing = -1;
  for (let index = opening; index < text.length; index++) {
    const char = text[index];
    if (quoted) {
      if (escaped) escaped = false;
      else if (char === '\\') escaped = true;
      else if (char === '"') quoted = false;
    } else if (char === '"') quoted = true;
    else if (char === '{') depth++;
    else if (char === '}' && --depth === 0) {
      closing = index;
      break;
    }
  }
  if (closing < 0) return null;
  const contents = text.slice(opening + 1, closing);
  const fields = [...contents.matchAll(/\b([A-Za-z_]\w*)=("(?:\\.|[^"\\])*"|[^\s,}]+)/g)].map(([, name, rawValue]) => ({
    name,
    value: rawValue.startsWith('"') ? rawValue.slice(1, -1) : rawValue
  }));
  return { raw: text.slice(match.index, closing + 1), start: match.index, end: closing + 1, fields };
}

export function hloOpName(text: string): string | null {
  return extractHloMetadata(text)?.fields.find(field => field.name === 'op_name')?.value || null;
}

export function lastOpNameSegment(opName: string): string {
  return opName.replace(/\/+$/, '').split('/').at(-1) || opName;
}
