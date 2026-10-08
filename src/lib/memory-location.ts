export interface MemoryLocation {
  path: string | null;
  space: number;
  label: string;
  explicit: boolean;
}

function tupleItems(type: string): string[] {
  const items: string[] = [];
  let start = 1, parens = 0, brackets = 0, braces = 0;
  for (let index = 1; index < type.length - 1; index++) {
    const char = type[index];
    if (char === '(') parens++;
    else if (char === ')') parens--;
    else if (char === '[') brackets++;
    else if (char === ']') brackets--;
    else if (char === '{') braces++;
    else if (char === '}') braces--;
    else if (char === ',' && !parens && !brackets && !braces) {
      items.push(type.slice(start, index).trim());
      start = index + 1;
    }
  }
  items.push(type.slice(start, -1).trim());
  return items;
}

export function memoryLocations(type: string): MemoryLocation[] {
  const visit = (shape: string, path: string | null): MemoryLocation[] => {
    const trimmed = shape.trim();
    if (trimmed.startsWith('(') && trimmed.endsWith(')')) {
      return tupleItems(trimmed).flatMap((item, index) => visit(item, path === null ? String(index) : `${path}.${index}`));
    }
    if (!/^[a-z][\w]*\[[^\]]*\]/.test(trimmed)) return [];
    const match = /\bS\((\d+)\)/.exec(trimmed);
    const space = match ? Number(match[1]) : 0;
    return [{
      path, space, explicit: !!match,
      label: space === 0 ? 'HBM' : space === 1 ? 'VMEM' : space === 5 ? 'HOST' : `S(${space})`
    }];
  };
  return visit(type, null);
}
