import { tupleElements } from './memory-location';

export interface LayoutSpec {
  elementType: string;
  dimensions: number[];
  minorToMajor: number[];
  tiles: number[][];
}

export interface LayoutSlot {
  path: string;
  raw: string;
}

export function layoutSlots(type: string, path = ''): LayoutSlot[] {
  const raw = type.replace(/\/\*[^*]*\*\//g, '').trim();
  if (raw.startsWith('(') && raw.endsWith(')')) {
    return tupleElements(raw).flatMap((part, index) => layoutSlots(part, path ? `${path}.${index}` : String(index)));
  }
  return /^[a-z]\w*\[/.test(raw) && !raw.startsWith('token[') ? [{ path, raw }] : [];
}

export type LayoutParse = { spec: LayoutSpec; reason?: never } | { spec?: never; reason: string };

export function parseLayout(raw: string): LayoutParse {
  const match = /^([a-z]\w*)\[([^\]]*)\](?:\{([^}]*)\})?$/.exec(raw.trim());
  if (!match) return { reason: 'This result shape uses a format this explorer cannot map yet.' };
  const [, elementType, shape, layout] = match;
  if (!shape.trim())
    return { reason: 'A scalar has one logical value. Tile markers on scalars do not describe a grid of logical elements.' };
  if (!/^(?:bf16|f16|f32|f64|[su](?:8|16|32|64)|pred)$/.test(elementType)) {
    return { reason: 'This element type is not supported by the layout explorer yet.' };
  }
  if (!/^\d+(?:\s*,\s*\d+)*$/.test(shape))
    return { reason: 'Dynamic or symbolic dimensions need runtime bounds; no physical mapping is inferred here.' };
  const dimensions = shape.split(',').map(Number);
  if (dimensions.length > 8 || dimensions.some(d => !Number.isSafeInteger(d) || d < 1)) {
    return { reason: 'The explorer supports positive, static array dimensions up to rank 8.' };
  }
  if (layout === undefined)
    return { reason: 'No physical layout is recorded on this result. A logical shape alone does not establish memory order.' };
  const [order, ...suffixParts] = layout.split(':');
  if (!/^\d+(?:\s*,\s*\d+)*$/.test(order)) return { reason: 'This physical dimension ordering is not supported.' };
  const minorToMajor = order.split(',').map(Number);
  if (
    minorToMajor.length !== dimensions.length ||
    new Set(minorToMajor).size !== dimensions.length ||
    minorToMajor.some(d => d >= dimensions.length)
  ) {
    return { reason: 'The recorded dimension order is not a permutation of this array’s dimensions.' };
  }
  const suffix = suffixParts.join(':');
  const tileText = /T(?:\([^)]*\))+/.exec(suffix)?.[0] ?? '';
  const other = suffix
    .replace(tileText, '')
    .replace(/S\(\d+\)/g, '')
    .trim();
  if (other)
    return { reason: 'This layout has additional physical-format annotations. The explorer does not assume their storage effects.' };
  const tileStrings = [...tileText.matchAll(/\(([^)]*)\)/g)].map(m => m[1]);
  if (tileStrings.some(t => !/^\d+(?:\s*,\s*\d+)*$/.test(t))) {
    return { reason: 'Dimension-combining tiles (* or -1) are not supported by this explorer yet.' };
  }
  const tiles = tileStrings.map(t => t.split(',').map(Number));
  let rank = dimensions.length;
  for (const tile of tiles) {
    if (tile.length > rank || tile.some(t => !Number.isSafeInteger(t) || t < 1) || tiles.length > 8) {
      return { reason: 'The tiling dimensions cannot be mapped safely by this explorer.' };
    }
    rank += tile.length;
  }
  const spec = { elementType, dimensions, minorToMajor, tiles };
  if (!layoutGeometry(spec)) return { reason: 'This layout is too large for exact integer offsets in the explorer.' };
  return { spec };
}

const majorToMinor = (spec: LayoutSpec) => spec.minorToMajor.map((_, i) => spec.minorToMajor[spec.minorToMajor.length - 1 - i]);

const product = (values: number[]) => values.reduce((result, value) => result * value, 1);

export interface LayoutGeometry {
  stages: number[][];
  logicalElements: number;
  storageElements: number;
}

// Every tile applies to a suffix of the current physical dimensions. Repeating
// this transform also handles later tiles that reach across previous tile axes.
export function layoutGeometry(spec: LayoutSpec): LayoutGeometry | null {
  const stages = [majorToMinor(spec).map(d => spec.dimensions[d])];
  for (const tile of spec.tiles) {
    const previous = stages[stages.length - 1];
    const start = previous.length - tile.length;
    stages.push([...previous.slice(0, start), ...tile.map((t, i) => Math.ceil(previous[start + i] / t)), ...tile]);
  }
  const logicalElements = product(spec.dimensions);
  const storageElements = product(stages[stages.length - 1]);
  if (!Number.isSafeInteger(logicalElements) || !Number.isSafeInteger(storageElements)) return null;
  return { stages, logicalElements, storageElements };
}

export function layoutCoordinates(spec: LayoutSpec, logical: number[]): number[][] {
  const stages = [majorToMinor(spec).map(d => logical[d])];
  for (const tile of spec.tiles) {
    const previous = stages[stages.length - 1];
    const start = previous.length - tile.length;
    stages.push([
      ...previous.slice(0, start),
      ...tile.map((t, i) => Math.floor(previous[start + i] / t)),
      ...tile.map((t, i) => previous[start + i] % t)
    ]);
  }
  return stages;
}

export function physicalOffset(spec: LayoutSpec, logical: number[], geometry = layoutGeometry(spec)!): number {
  const coordinates = layoutCoordinates(spec, logical).at(-1)!;
  return coordinates.reduce((offset, coordinate, index) => offset * geometry.stages.at(-1)![index] + coordinate, 0);
}

// null denotes padding, including padding introduced by a later tiling stage.
export function logicalAtOffset(spec: LayoutSpec, offset: number, geometry = layoutGeometry(spec)!): number[] | null {
  if (!Number.isSafeInteger(offset) || offset < 0 || offset >= geometry.storageElements) return null;
  let remainder = offset;
  let coordinates = geometry.stages.at(-1)!.map(() => 0);
  for (let i = coordinates.length - 1; i >= 0; i--) {
    coordinates[i] = remainder % geometry.stages.at(-1)![i];
    remainder = Math.floor(remainder / geometry.stages.at(-1)![i]);
  }
  for (let level = spec.tiles.length - 1; level >= 0; level--) {
    const tile = spec.tiles[level];
    const previous = geometry.stages[level];
    const start = previous.length - tile.length;
    coordinates = [...coordinates.slice(0, start), ...tile.map((t, i) => coordinates[start + i] * t + coordinates[previous.length + i])];
    if (coordinates.some((c, i) => c >= previous[i])) return null;
  }
  const logical = spec.dimensions.map(() => 0);
  majorToMinor(spec).forEach((dimension, index) => {
    logical[dimension] = coordinates[index];
  });
  return logical;
}
