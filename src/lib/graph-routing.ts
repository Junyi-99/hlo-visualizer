import type { ELK, ElkNode } from 'elkjs/lib/elk-api';
import { computationLinks } from './parser.ts';
import { NODE_HEIGHT, NODE_WIDTH, OVERVIEW_HEIGHT, OVERVIEW_WIDTH, linkLabel } from './graph-layout.ts';
import type { GraphLayout, LayoutDirection, Point } from './graph-layout.ts';
import type { Computation, HloModule } from './types';

export interface RoutedEdge {
  id: string;
  /** Position of this edge in the input edge list. */
  index: number;
  source: string;
  target: string;
  control: boolean;
  points: Point[];
  label?: Point;
}

export interface RoutedLayout extends GraphLayout {
  edges: RoutedEdge[];
}

interface LayoutNode {
  name: string;
  width: number;
  height: number;
}

interface LayoutEdge {
  source: string;
  target: string;
  control?: boolean;
  label?: string;
}

// Canvas size when ELK has nothing to lay out.
const EMPTY_WIDTH = 116;
const EMPTY_HEIGHT = 120;

const ELK_OPTIONS = {
  'elk.algorithm': 'layered',
  'elk.edgeRouting': 'SPLINES',
  'elk.layered.edgeRouting.splines.mode': 'SLOPPY',
  'elk.padding': '[top=72,left=48,bottom=48,right=48]',
  'elk.spacing.nodeNode': '36',
  'elk.spacing.edgeNode': '18',
  'elk.spacing.edgeEdge': '12',
  'elk.layered.spacing.nodeNodeBetweenLayers': '72',
  'elk.layered.spacing.edgeNodeBetweenLayers': '18',
  'elk.layered.spacing.edgeEdgeBetweenLayers': '12',
  'elk.layered.crossingMinimization.strategy': 'LAYER_SWEEP',
  'elk.layered.nodePlacement.strategy': 'BRANDES_KOEPF',
  'elk.layered.nodePlacement.bk.fixedAlignment': 'BALANCED',
  'elk.layered.mergeEdges': 'false',
  'elk.randomSeed': '1'
};

const elkPort = (id: string, side: string) => ({ id, width: 0, height: 0, layoutOptions: { 'elk.port.side': side } });

// Use internal numeric IDs: HLO names can contain punctuation used by port IDs.
export async function routeGraph(engine: ELK, nodes: LayoutNode[], edges: LayoutEdge[], direction: LayoutDirection): Promise<RoutedLayout> {
  if (!nodes.length) return { positions: new Map(), edges: [], width: EMPTY_WIDTH, height: EMPTY_HEIGHT };

  const ids = new Map(nodes.map((node, index) => [node.name, `n${index}`]));
  const validEdges = edges.map((edge, index) => ({ ...edge, index })).filter(edge => ids.has(edge.source) && ids.has(edge.target));
  const vertical = direction === 'vertical';
  // One port per edge end, fixed to the card side facing the layout direction.
  const portsOf = (name: string) =>
    validEdges.flatMap((edge, index) => [
      ...(edge.target === name ? [elkPort(`e${index}:in`, vertical ? 'NORTH' : 'WEST')] : []),
      ...(edge.source === name ? [elkPort(`e${index}:out`, vertical ? 'SOUTH' : 'EAST')] : [])
    ]);

  const graph: ElkNode = {
    id: 'graph',
    layoutOptions: { ...ELK_OPTIONS, 'elk.direction': vertical ? 'DOWN' : 'RIGHT' },
    children: nodes.map(node => ({
      id: ids.get(node.name)!,
      width: node.width,
      height: node.height,
      layoutOptions: { 'elk.portConstraints': 'FIXED_SIDE' },
      ports: portsOf(node.name)
    })),
    edges: validEdges.map((edge, index) => ({
      id: `e${index}`,
      sources: [`e${index}:out`],
      targets: [`e${index}:in`],
      ...(edge.label ? { labels: [{ text: edge.label, width: edge.label.length * 7 + 8, height: 16 }] } : {})
    }))
  };

  const result = await engine.layout(graph);

  const positions = new Map<string, Point>();
  result.children?.forEach(node => positions.set(nodes[Number(node.id.slice(1))].name, { x: node.x ?? 0, y: node.y ?? 0 }));
  return {
    positions,
    width: result.width ?? EMPTY_WIDTH,
    height: result.height ?? EMPTY_HEIGHT,
    edges: (result.edges ?? []).map(edge => {
      const definition = validEdges[Number(edge.id.slice(1))];
      const section = edge.sections?.[0];
      if (!section) throw new Error(`Missing route for ${definition.source} → ${definition.target}`);

      const label = edge.labels?.[0];
      return {
        id: edge.id,
        index: definition.index,
        source: definition.source,
        target: definition.target,
        control: !!definition.control,
        points: [section.startPoint, ...(section.bendPoints ?? []), section.endPoint],
        label: label ? { x: (label.x ?? 0) + (label.width ?? 0) / 2, y: (label.y ?? 0) + 12 } : undefined
      };
    })
  };
}

export function instructionGraph(computation: Computation, heights: ReadonlyMap<string, number>) {
  return {
    nodes: computation.nodes.map(node => ({ name: node.name, width: NODE_WIDTH, height: heights.get(node.name) ?? NODE_HEIGHT })),
    edges: computation.nodes.flatMap(node => [
      ...node.operands.map(source => ({ source, target: node.name, control: false })),
      ...node.controlPredecessors.map(source => ({ source, target: node.name, control: true }))
    ])
  };
}

export function overviewGraph(module: HloModule) {
  return {
    nodes: module.computations.map(node => ({ name: node.name, width: OVERVIEW_WIDTH, height: OVERVIEW_HEIGHT })),
    edges: computationLinks(module).map(link => ({ source: link.from, target: link.to, label: linkLabel(link.role) }))
  };
}

// ELK spline routes are cubic Bézier controls in triples after the start point. Keeping
// them makes cross-layer edges follow their planned lanes instead of one curve across cards.
function curvePoints(points: Point[], direction: LayoutDirection, arrowInset = 5): Point[] {
  // A straight route has no controls; its endpoint handles are set below.
  if (points.length === 2) points = [points[0], points[0], points[1], points[1]];
  if (points.length < 4 || (points.length - 1) % 3 !== 0) return [];

  const route = points.map(point => ({ ...point }));
  const first = route[0],
    last = route.at(-1)!;
  const axis = direction === 'horizontal' ? 'x' : 'y';
  const cross = direction === 'horizontal' ? 'y' : 'x';
  const handle = (from: Point, to: Point) => Math.min(64, Math.max(16, Math.abs(to[axis] - from[axis]) * 0.45));

  // Point the endpoint handles straight out of the card sides, so arrows enter squarely.
  const outgoing = handle(first, route[3]);
  const incoming = handle(route[route.length - 4], last);
  route[1][axis] = first[axis] + outgoing;
  route[1][cross] = first[cross];
  last[axis] -= arrowInset;
  route[route.length - 2][axis] = last[axis] - incoming;
  route[route.length - 2][cross] = last[cross];
  return route;
}

export function edgePath(points: Point[], direction: LayoutDirection = 'horizontal', arrowInset = 5): string {
  const route = curvePoints(points, direction, arrowInset);
  // Not a cubic spline: draw the polyline rather than dropping the dependency.
  if (!route.length) return points.map((point, index) => `${index ? 'L' : 'M'}${point.x} ${point.y}`).join(' ');

  let path = `M${route[0].x} ${route[0].y}`;
  for (let i = 1; i < route.length; i += 3) {
    const a = route[i],
      b = route[i + 1],
      c = route[i + 2];
    path += ` C${a.x} ${a.y},${b.x} ${b.y},${c.x} ${c.y}`;
  }
  return path;
}

// Shifting endpoints and their adjacent controls keeps dragged curves attached without elbows.
// Arrange restores the automatic layout.
export function movedEdgePoints(edge: RoutedEdge, offsets: ReadonlyMap<string, Point>): Point[] {
  const source = offsets.get(edge.source),
    target = offsets.get(edge.target);
  if (!source && !target) return edge.points;

  const points = edge.points.map(point => ({ ...point }));
  for (const [offset, indices] of [
    [source, [0, 1]],
    [target, [points.length - 2, points.length - 1]]
  ] as const) {
    if (!offset) continue;
    for (const index of indices) {
      points[index].x += offset.x;
      points[index].y += offset.y;
    }
  }
  return points;
}

// A worker failure should still leave dependencies visible and the viewer usable.
export function fallbackRoutes(layout: GraphLayout, nodes: LayoutNode[], edges: LayoutEdge[], direction: LayoutDirection): RoutedLayout {
  const horizontal = direction === 'horizontal';
  const byName = new Map(nodes.map(node => [node.name, node]));
  const valid = edges
    .map((edge, index) => ({ ...edge, index }))
    .filter(edge => layout.positions.has(edge.source) && layout.positions.has(edge.target));

  // Spread the edges at each card side evenly along it, in edge order.
  const counts = new Map<string, number>();
  for (const edge of valid) {
    for (const key of [edge.source + ':out', edge.target + ':in']) counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  const used = new Map<string, number>();
  const port = (name: string, side: 'in' | 'out') => {
    const node = byName.get(name)!,
      point = layout.positions.get(name)!;
    const key = name + ':' + side;
    const index = used.get(key) ?? 0;
    used.set(key, index + 1);

    const fraction = (index + 1) / ((counts.get(key) ?? 1) + 1);
    const outSide = side === 'out';
    return horizontal
      ? { x: point.x + (outSide ? node.width : 0), y: point.y + node.height * fraction }
      : { x: point.x + node.width * fraction, y: point.y + (outSide ? node.height : 0) };
  };

  return {
    ...layout,
    edges: valid.map((edge, index) => {
      const start = port(edge.source, 'out'),
        end = port(edge.target, 'in');
      // An S-shaped cubic whose controls meet halfway between the cards.
      const mid = horizontal ? (start.x + end.x) / 2 : (start.y + end.y) / 2;
      const controls = horizontal
        ? [
            { x: mid, y: start.y },
            { x: mid, y: end.y }
          ]
        : [
            { x: start.x, y: mid },
            { x: end.x, y: mid }
          ];
      return {
        id: `e${index}`,
        index: edge.index,
        source: edge.source,
        target: edge.target,
        control: !!edge.control,
        points: [start, ...controls, end],
        label: edge.label ? { x: (start.x + end.x) / 2, y: (start.y + end.y) / 2 - 8 } : undefined
      };
    })
  };
}
