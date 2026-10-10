import { Fragment, forwardRef, useEffect, useImperativeHandle, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import {
  calledComputations,
  computationLinks,
  dependencyNeighborhood,
  nodeCategory,
  reachable,
  shortestDependencyPath
} from '../lib/parser';
import {
  clampZoom,
  ZOOM_STEP,
  computationRole,
  linkLabel,
  MIN_ZOOM,
  NODE_HEIGHT,
  NODE_WIDTH,
  type LayoutDirection,
  type LayoutMode,
  type Point
} from '../lib/graph-layout';
import { hloOpName, lastOpNameSegment } from '../lib/metadata';
import { nodeMemoryLocations, type MemoryLocation } from '../lib/memory-location';
import { memorySpaceText } from '../lib/memory-space';
import type { Computation, ComputationLink, HloModule, HloNode } from '../lib/types';
import { copyGroupCaption, type CopyGroup, type CopyGrouping } from '../lib/copy-grouping';
import { useGraphLayouts } from '../hooks/useGraphLayouts';
import { edgePath, movedEdgePoints, type RoutedEdge, type RoutedLayout } from '../lib/graph-routing';
import { ComputationExplorer } from './ComputationExplorer';

const CANVAS_PADDING = 32;
const VISIBLE_MARGIN = 18;

const cx = (...classes: (string | false | null | undefined)[]) => classes.filter(Boolean).join(' ');

// Direction-independent key, so a path edge matches either orientation.
const edgeKey = (a: string, b: string) => [a, b].sort().join('\u0000');

function memoryTooltip(location: MemoryLocation) {
  const target = location.path === null ? 'Result' : `Result element ${location.path}`;
  return `${target}: ${memorySpaceText(location.space, location.explicit)}`;
}

// Connection points of each node, relative to the node's layout position.
function nodePorts(layout: RoutedLayout) {
  const ports = new Map<string, Point[]>();
  for (const edge of layout.edges) {
    const ends = [
      [edge.source, edge.points[0]],
      [edge.target, edge.points.at(-1)!]
    ] as const;
    for (const [name, point] of ends) {
      const base = layout.positions.get(name);
      if (!base) continue;
      const list = ports.get(name) ?? [];
      list.push({ x: point.x - base.x, y: point.y - base.y });
      ports.set(name, list);
    }
  }
  return ports;
}

export interface GraphHandle {
  fit: () => void;
  arrange: () => void;
  centerNode: (name: string) => void;
}

interface GraphCanvasProps {
  copyGrouping: CopyGrouping | null;
  autoGroup: boolean;
  layoutMode: LayoutMode;
  module: HloModule;
  current: string | null;
  selected: string | null;
  zoom: number;
  useOpName: boolean;
  showLastNameOnly: boolean;
  showMemoryLocation: boolean;
  onZoom: (value: number) => void;
  onSelect: (name: string | null) => void;
  onComputation: (name: string) => void;
}

export const GraphCanvas = forwardRef<GraphHandle, GraphCanvasProps>(function GraphCanvas(
  {
    copyGrouping,
    autoGroup,
    layoutMode,
    module,
    current,
    selected: originalSelected,
    zoom,
    useOpName,
    showLastNameOnly,
    showMemoryLocation,
    onZoom,
    onSelect,
    onComputation
  },
  ref
) {
  const computation = copyGrouping?.computation ?? null;
  const selected = originalSelected ? (copyGrouping?.aliases.get(originalSelected) ?? originalSelected) : null;
  const nodeKey = (name: string) => `${computation?.name}/${name}`;

  const [focusRadius, setFocusRadius] = useState<number | null>(null);
  const [hops, setHops] = useState(1);
  const [pathPicking, setPathPicking] = useState(false);
  const [pathTarget, setPathTarget] = useState<string | null>(null);
  const [expandedNodeName, setExpandedNodeName] = useState<string | null>(null);
  const [nodeHeights, setNodeHeights] = useState<Record<string, number>>({});
  const [viewport, setViewport] = useState<{ width: number; height: number } | null>(null);
  const [arrangeRevision, setArrangeRevision] = useState(0);
  const [nodeOffsets, setNodeOffsets] = useState<Record<string, Point>>({});
  const [draggingNode, setDraggingNode] = useState<string | null>(null);

  const scrollerRef = useRef<HTMLDivElement>(null);
  const zoomRef = useRef(zoom);
  const renderedZoomRef = useRef(zoom);
  const gestureStartZoomRef = useRef(0);
  const pendingZoomAnchorRef = useRef<{ x: number; y: number; canvasX: number; canvasY: number } | null>(null);
  const panRef = useRef<{ x: number; y: number; left: number; top: number } | null>(null);
  const dragRef = useRef<{ name: string; pointerId: number; x: number; y: number; position: Point; moved: boolean } | null>(null);
  const suppressClickRef = useRef<string | null>(null);
  zoomRef.current = zoom;
  renderedZoomRef.current = zoom;

  const path = useMemo(
    () => (computation && selected && pathTarget ? shortestDependencyPath(computation, selected, pathTarget) : null),
    [computation, selected, pathTarget]
  );
  const pathEdges = useMemo(() => (path ? new Set(path.slice(1).map((name, index) => edgeKey(path[index], name))) : null), [path]);

  const visibleNames = useMemo(() => {
    if (!computation || !selected) return null;
    if (pathTarget) return new Set(path || [selected, pathTarget]);
    if (focusRadius !== null && !pathPicking) return dependencyNeighborhood(computation, selected, focusRadius);
    return null;
  }, [computation, selected, pathTarget, path, focusRadius, pathPicking]);

  const visibleNodes = useMemo(
    () => (computation ? computation.nodes.filter(node => !visibleNames || visibleNames.has(node.name)) : []),
    [computation, visibleNames]
  );
  const viewComputation = useMemo(
    () => (computation ? { ...computation, nodes: visibleNodes, byName: new Map(visibleNodes.map(node => [node.name, node])) } : null),
    [computation, visibleNodes]
  );

  const { layouts, pending, failed, retry } = useGraphLayouts(module, viewComputation, nodeHeights);

  // Auto picks the direction that fits the canvas better when the graph or its layout changes,
  // not on every resize: opening the inspector must not flip the graph under the click.
  const [autoPick, setAutoPick] = useState<{
    layouts: typeof layouts;
    arrangeRevision: number;
    measured: boolean;
    direction: LayoutDirection;
  } | null>(null);
  let autoDirection = autoPick?.direction ?? 'horizontal';
  if (!autoPick || autoPick.layouts !== layouts || autoPick.arrangeRevision !== arrangeRevision || autoPick.measured !== !!viewport) {
    const { width, height } = viewport ?? { width: 1, height: 1 };
    const score = (candidate: RoutedLayout) => Math.min(width / candidate.width, height / candidate.height);
    autoDirection = score(layouts.vertical) > score(layouts.horizontal) * 1.08 ? 'vertical' : 'horizontal';
    setAutoPick({ layouts, arrangeRevision, measured: !!viewport, direction: autoDirection });
  }

  const direction = layoutMode === 'auto' ? autoDirection : layoutMode;
  const layout = layouts[direction];
  const vertical = direction === 'vertical';
  const links = useMemo(() => computationLinks(module), [module]);
  const nodeHeight = (name: string) => (computation ? (nodeHeights[nodeKey(name)] ?? NODE_HEIGHT) : NODE_HEIGHT);

  const positions = useMemo(() => {
    if (!computation) return layout.positions;
    const adjusted = new Map(layout.positions);
    for (const node of visibleNodes) {
      const base = layout.positions.get(node.name);
      const offset = nodeOffsets[node.id];
      if (base && offset) adjusted.set(node.name, { x: base.x + offset.x, y: base.y + offset.y });
    }
    return adjusted;
  }, [layout, computation, visibleNodes, nodeOffsets]);

  const viewRef = useRef({ positions, zoom, nodeHeights });
  viewRef.current = { positions, zoom, nodeHeights };

  const stageWidth = computation
    ? Math.max(layout.width, ...[...positions.values()].map(point => point.x + NODE_WIDTH + 58))
    : layout.width;
  const stageHeight = computation
    ? Math.max(layout.height, ...[...positions].map(([name, point]) => point.y + nodeHeight(name) + 66))
    : layout.height;

  const selectedNode = selected ? computation?.byName.get(selected) : undefined;
  const expandedNode = expandedNodeName ? computation?.byName.get(expandedNodeName) : undefined;
  const upstream = selectedNode && computation ? reachable(computation, selectedNode.name, 'up') : new Set<string>();
  const downstream = selectedNode && computation ? reachable(computation, selectedNode.name, 'down') : new Set<string>();

  useEffect(() => {
    setPathPicking(false);
    setPathTarget(null);
    setExpandedNodeName(null);
  }, [selected, current, module, autoGroup]);

  useEffect(() => {
    setNodeOffsets({});
  }, [module, current, direction, autoGroup]);

  useEffect(() => {
    const scroller = scrollerRef.current;
    if (!scroller) return;
    const observer = new ResizeObserver(() => {
      const width = Math.max(1, scroller.clientWidth - CANVAS_PADDING);
      const height = Math.max(1, scroller.clientHeight - CANVAS_PADDING);
      setViewport(prior => (prior?.width === width && prior.height === height ? prior : { width, height }));
    });
    observer.observe(scroller);
    return () => observer.disconnect();
  }, []);

  // Measured card heights feed back into the layout.
  useEffect(() => {
    if (!computation) return;
    const cards = scrollerRef.current?.querySelectorAll<HTMLButtonElement>('.node-card');
    if (!cards) return;
    const observer = new ResizeObserver(entries => {
      setNodeHeights(prior => {
        let changed = false;
        const next = { ...prior };
        for (const entry of entries) {
          const card = entry.target as HTMLElement;
          const id = card.dataset.nodeId;
          if (!id || next[id] === card.offsetHeight) continue;
          next[id] = card.offsetHeight;
          changed = true;
        }
        return changed ? next : prior;
      });
    });
    cards.forEach(card => observer.observe(card));
    return () => observer.disconnect();
  }, [computation, visibleNodes, useOpName, showLastNameOnly, showMemoryLocation]);

  // Keep the canvas point under the cursor fixed while zooming.
  useLayoutEffect(() => {
    const anchor = pendingZoomAnchorRef.current;
    const scroller = scrollerRef.current;
    if (!anchor || !scroller) return;
    scroller.scrollLeft = anchor.canvasX * zoom - anchor.x;
    scroller.scrollTop = anchor.canvasY * zoom - anchor.y;
    pendingZoomAnchorRef.current = null;
  }, [zoom]);

  // Ctrl+wheel (trackpad pinch) and Safari gesture events. Listeners are native so they can preventDefault.
  useEffect(() => {
    const scroller = scrollerRef.current;
    if (!scroller) return;

    const zoomAt = (clientX: number, clientY: number, next: number) => {
      if (Math.abs(next - zoomRef.current) < 0.001) return;
      const rect = scroller.getBoundingClientRect();
      const x = clientX - rect.left;
      const y = clientY - rect.top;
      pendingZoomAnchorRef.current = {
        x,
        y,
        canvasX: (scroller.scrollLeft + x) / renderedZoomRef.current,
        canvasY: (scroller.scrollTop + y) / renderedZoomRef.current
      };
      zoomRef.current = next;
      onZoom(next);
    };

    const wheel = (event: WheelEvent) => {
      if (!event.ctrlKey || gestureStartZoomRef.current) return;
      event.preventDefault();
      zoomAt(event.clientX, event.clientY, clampZoom(zoomRef.current * Math.exp(-event.deltaY / 100)));
    };
    const gestureStart = (event: Event) => {
      event.preventDefault();
      gestureStartZoomRef.current = zoomRef.current;
    };
    const gestureChange = (event: Event) => {
      event.preventDefault();
      const gesture = event as Event & { scale: number; clientX: number; clientY: number };
      const startZoom = gestureStartZoomRef.current || zoomRef.current;
      zoomAt(gesture.clientX, gesture.clientY, clampZoom(startZoom * gesture.scale));
    };
    const gestureEnd = () => {
      gestureStartZoomRef.current = 0;
    };

    scroller.addEventListener('wheel', wheel, { passive: false });
    scroller.addEventListener('gesturestart', gestureStart, { passive: false });
    scroller.addEventListener('gesturechange', gestureChange, { passive: false });
    scroller.addEventListener('gestureend', gestureEnd);
    return () => {
      scroller.removeEventListener('wheel', wheel);
      scroller.removeEventListener('gesturestart', gestureStart);
      scroller.removeEventListener('gesturechange', gestureChange);
      scroller.removeEventListener('gestureend', gestureEnd);
    };
  }, [onZoom]);

  const fit = () => {
    const scroller = scrollerRef.current;
    if (!scroller) return;
    const fitWidth = (scroller.clientWidth - CANVAS_PADDING) / stageWidth;
    const fitHeight = (scroller.clientHeight - CANVAS_PADDING) / stageHeight;
    onZoom(Math.max(MIN_ZOOM, Math.min(1, fitWidth, fitHeight)));
    scroller.scrollTo(0, 0);
  };

  useImperativeHandle(ref, () => ({
    fit,
    arrange() {
      setNodeOffsets({});
      setArrangeRevision(value => value + 1);
      if (failed) retry();
    },
    centerNode(name) {
      const displayName = copyGrouping?.aliases.get(name) ?? name;
      const position = positions.get(displayName);
      const scroller = scrollerRef.current;
      if (!position || !scroller) return;
      scroller.scrollTo({
        left: (position.x + NODE_WIDTH / 2) * zoom - scroller.clientWidth / 2,
        top: (position.y + nodeHeight(displayName) / 2) * zoom - scroller.clientHeight / 2,
        behavior: 'smooth'
      });
    }
  }));

  useEffect(() => {
    const frame = requestAnimationFrame(fit);
    return () => cancelAnimationFrame(frame);
    // Refit only when the layout changes; a zoom button must not refit.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layout, arrangeRevision]);

  // Scroll the selection into view, also when the inspector docks or the viewport resizes.
  // Reads viewRef so dragging a node does not recenter.
  useEffect(() => {
    if (!selected || !computation) return;
    const scroller = scrollerRef.current;
    if (!scroller) return;

    const ensureVisible = () => {
      const view = viewRef.current;
      const position = view.positions.get(selected);
      if (!position) return;
      const height = view.nodeHeights[nodeKey(selected)] ?? NODE_HEIGHT;
      const scale = view.zoom;
      const left = position.x * scale;
      const right = (position.x + NODE_WIDTH) * scale;
      const top = position.y * scale;
      const bottom = (position.y + height) * scale;
      const visible =
        left >= scroller.scrollLeft + VISIBLE_MARGIN &&
        right <= scroller.scrollLeft + scroller.clientWidth - VISIBLE_MARGIN &&
        top >= scroller.scrollTop + VISIBLE_MARGIN &&
        bottom <= scroller.scrollTop + scroller.clientHeight - VISIBLE_MARGIN;
      if (visible) return;
      scroller.scrollTo({
        left: Math.max(0, (position.x + NODE_WIDTH / 2) * scale - scroller.clientWidth / 2),
        top: Math.max(0, (position.y + height / 2) * scale - scroller.clientHeight / 2),
        behavior: 'smooth'
      });
    };

    const frame = requestAnimationFrame(ensureVisible);
    const observer = new ResizeObserver(ensureVisible);
    observer.observe(scroller);
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected, current, nodeHeights, layout]);

  const ports = nodePorts(layout);
  const offsets = new Map(visibleNodes.flatMap(node => (nodeOffsets[node.id] ? [[node.name, nodeOffsets[node.id]] as const] : [])));
  const shownEdges = pathEdges ? layout.edges.filter(edge => pathEdges.has(edgeKey(edge.source, edge.target))) : layout.edges;

  const startPan = (event: React.PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0 || (event.target instanceof Element && event.target.closest('.node-card, .overview-card'))) return;
    const scroller = scrollerRef.current!;
    scroller.focus({ preventScroll: true });
    panRef.current = { x: event.clientX, y: event.clientY, left: scroller.scrollLeft, top: scroller.scrollTop };
    scroller.setPointerCapture(event.pointerId);
    scroller.classList.add('panning');
  };

  const movePan = (event: React.PointerEvent<HTMLDivElement>) => {
    const pan = panRef.current;
    const scroller = scrollerRef.current;
    if (!pan || !scroller) return;
    scroller.scrollLeft = pan.left - (event.clientX - pan.x);
    scroller.scrollTop = pan.top - (event.clientY - pan.y);
  };

  const endPan = () => {
    panRef.current = null;
    scrollerRef.current?.classList.remove('panning');
  };

  const deselectOnBackground = (event: React.MouseEvent<HTMLDivElement>) => {
    const onCard = event.target instanceof Element && event.target.closest('.node-card');
    if (computation && !onCard) onSelect(null);
  };

  const startNodeDrag = (event: React.PointerEvent<HTMLButtonElement>, name: string) => {
    if (event.button !== 0) return;
    event.stopPropagation();
    suppressClickRef.current = null;
    const position = positions.get(name);
    if (!position) return;
    dragRef.current = { name, pointerId: event.pointerId, x: event.clientX, y: event.clientY, position, moved: false };
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const moveNode = (event: React.PointerEvent<HTMLButtonElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId || !computation) return;
    const dx = event.clientX - drag.x;
    const dy = event.clientY - drag.y;
    if (!drag.moved && Math.hypot(dx, dy) < 4) return;

    drag.moved = true;
    setDraggingNode(drag.name);
    const base = layout.positions.get(drag.name)!;
    const x = Math.max(12, drag.position.x + dx / zoom);
    const y = Math.max(12, drag.position.y + dy / zoom);
    setNodeOffsets(prior => ({ ...prior, [nodeKey(drag.name)]: { x: x - base.x, y: y - base.y } }));
  };

  const endNodeDrag = (event: React.PointerEvent<HTMLButtonElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    if (drag.moved) suppressClickRef.current = drag.name;
    dragRef.current = null;
    setDraggingNode(null);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  };

  const clickNode = (name: string) => {
    // A drag ends with a click on the same card; swallow it.
    if (suppressClickRef.current === name) {
      suppressClickRef.current = null;
      return;
    }
    if (pathPicking && selected && name !== selected) {
      setPathTarget(name);
      setPathPicking(false);
      return;
    }
    onSelect(selected === name ? null : name);
  };

  const handleZoomKey = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.altKey || event.ctrlKey || event.metaKey) return;
    const actions: Record<string, () => void> = {
      '+': () => onZoom(clampZoom(zoom + ZOOM_STEP)),
      '=': () => onZoom(clampZoom(zoom + ZOOM_STEP)),
      '-': () => onZoom(clampZoom(zoom - ZOOM_STEP)),
      _: () => onZoom(clampZoom(zoom - ZOOM_STEP)),
      '0': fit
    };
    const action = actions[event.key];
    if (!action) return;
    event.preventDefault();
    action();
  };

  const changeHops = (value: number) => {
    setHops(value);
    if (focusRadius !== null) setFocusRadius(value);
  };

  const toggleFocus = () => {
    setFocusRadius(focusRadius === null ? hops : null);
    setPathTarget(null);
    setPathPicking(false);
  };

  const togglePathPicking = () => {
    setPathTarget(null);
    setPathPicking(value => !value);
  };

  const hint = computation
    ? { action: 'Data flow · Drag nodes · Background pans', counts: `${visibleNodes.length} nodes · ${shownEdges.length} edges` }
    : { action: 'Calls · Drag to pan', counts: `${module.computations.length} computations · ${links.length} links` };

  return (
    <section
      className={cx('graph-shell', vertical ? 'flow-vertical' : 'flow-horizontal', expandedNode && 'explorer-open')}
      aria-label="HLO graph"
      aria-busy={pending}
    >
      {pending && (
        <div className="layout-status" role="status">
          Arranging graph…
        </div>
      )}
      {failed && (
        <div className="layout-status" role="status">
          Could not arrange graph.{' '}
          <button type="button" className="subtle-button" onClick={retry}>
            Retry
          </button>
        </div>
      )}

      <div
        ref={scrollerRef}
        className="graph-scroller overscroll-contain"
        role="region"
        tabIndex={0}
        aria-label="Graph canvas: + and - to zoom, 0 to fit"
        aria-keyshortcuts="+ - 0"
        onKeyDown={handleZoomKey}
        onPointerDown={startPan}
        onPointerMove={movePan}
        onPointerUp={endPan}
        onPointerCancel={endPan}
        onDoubleClick={deselectOnBackground}
      >
        <div className="graph-content" style={{ width: stageWidth * zoom, height: stageHeight * zoom }}>
          <div className="graph-stage" style={{ width: stageWidth, height: stageHeight, transform: `scale(${zoom})` }}>
            <EdgeLayer width={stageWidth} height={stageHeight}>
              {computation
                ? shownEdges.map(edge => {
                    const up = !!selectedNode && (edge.target === selected || (upstream.has(edge.source) && upstream.has(edge.target)));
                    const down =
                      !!selectedNode && (edge.source === selected || (downstream.has(edge.source) && downstream.has(edge.target)));
                    const className = cx(
                      'edge',
                      edge.control && 'control-edge',
                      pathEdges && 'path-edge',
                      up && 'upstream',
                      down && 'downstream',
                      selectedNode && !pathEdges && !up && !down && 'dimmed'
                    );
                    return (
                      <path
                        key={edge.id}
                        className={className}
                        d={edgePath(movedEdgePoints(edge, offsets), direction)}
                        markerEnd="url(#arrow)"
                      />
                    );
                  })
                : layout.edges.map(edge => <OverviewEdge key={edge.id} edge={edge} link={links[edge.index]} direction={direction} />)}
            </EdgeLayer>

            <div className="graph-nodes">
              {computation
                ? visibleNodes.map(node => {
                    const onPath = !!path?.includes(node.name);
                    const isSelected = selected === node.name;
                    const related = isSelected || onPath || upstream.has(node.name) || downstream.has(node.name);
                    return (
                      <NodeCard
                        key={node.name}
                        module={module}
                        computation={computation}
                        node={node}
                        position={positions.get(node.name)!}
                        ports={ports.get(node.name)}
                        copyGroup={copyGrouping?.groups.get(node.name)}
                        useOpName={useOpName}
                        showLastNameOnly={showLastNameOnly}
                        showMemoryLocation={showMemoryLocation}
                        stateClass={cx(
                          draggingNode === node.name && 'dragging',
                          isSelected && 'selected',
                          onPath && 'path-node',
                          upstream.has(node.name) && 'upstream',
                          downstream.has(node.name) && 'downstream',
                          selectedNode && !related && 'dimmed'
                        )}
                        expanded={expandedNodeName === node.name}
                        onPointerDown={event => startNodeDrag(event, node.name)}
                        onPointerMove={moveNode}
                        onPointerEnd={endNodeDrag}
                        onClick={() => clickNode(node.name)}
                        onToggleExpand={() => setExpandedNodeName(expandedNodeName === node.name ? null : node.name)}
                      />
                    );
                  })
                : module.computations.map(entry => (
                    <OverviewCard
                      key={entry.name}
                      computation={entry}
                      position={layout.positions.get(entry.name)!}
                      links={links}
                      onOpen={() => onComputation(entry.name)}
                    />
                  ))}
            </div>
          </div>
        </div>
      </div>

      {computation && (
        <GraphControls
          hops={hops}
          focused={focusRadius !== null}
          selected={!!selected}
          pathPicking={pathPicking}
          hasPathTarget={!!pathTarget}
          status={controlsStatus({
            pathPicking,
            pathTarget,
            path,
            selected,
            focused: focusRadius !== null,
            visibleCount: visibleNodes.length,
            totalCount: computation.nodes.length
          })}
          onHopsChange={changeHops}
          onToggleFocus={toggleFocus}
          onTogglePathPicking={togglePathPicking}
          onClearPath={() => setPathTarget(null)}
        />
      )}

      {expandedNode && (
        <ComputationExplorer
          autoGroup={autoGroup}
          key={expandedNode.id}
          module={module}
          rootNode={expandedNode}
          onClose={() => setExpandedNodeName(null)}
          onOpenFull={onComputation}
        />
      )}

      <div className="graph-hint">
        <span className="flow-icon">{vertical ? '↓' : '→'}</span> {hint.action} <span className="hint-divider">·</span> Pinch / + / − zoom ·
        0 fit <span className="hint-divider">·</span> {hint.counts}
      </div>
    </section>
  );
});

function ArrowMarker({ id }: { id: string }) {
  return (
    <marker id={id} markerWidth="7" markerHeight="7" viewBox="0 0 7 7" markerUnits="userSpaceOnUse" refX="6" refY="3.5" orient="auto">
      <path d="M1 1 L6 3.5 L1 6 Z" />
    </marker>
  );
}

function EdgeLayer({ width, height, children }: { width: number; height: number; children: ReactNode }) {
  return (
    <svg className="graph-edges" width={width} height={height} viewBox={`0 0 ${width} ${height}`} aria-hidden="true">
      <defs>
        <ArrowMarker id="arrow" />
        <ArrowMarker id="call-arrow" />
      </defs>
      {children}
    </svg>
  );
}

function OverviewEdge({ edge, link, direction }: { edge: RoutedEdge; link: ComputationLink; direction: LayoutDirection }) {
  return (
    <g>
      <path className="call-edge" d={edgePath(edge.points, direction)} markerEnd="url(#call-arrow)">
        <title>
          %{edge.source} → %{edge.target} via %{link.via} ({link.role})
        </title>
      </path>
      {edge.label && (
        <text className="call-label" x={edge.label.x} y={edge.label.y} textAnchor="middle">
          {linkLabel(link.role)}
        </text>
      )}
    </g>
  );
}

interface NodeCardProps {
  module: HloModule;
  computation: Computation;
  node: HloNode;
  position: Point;
  ports: Point[] | undefined;
  copyGroup: CopyGroup | undefined;
  useOpName: boolean;
  showLastNameOnly: boolean;
  showMemoryLocation: boolean;
  stateClass: string;
  expanded: boolean;
  onPointerDown: (event: React.PointerEvent<HTMLButtonElement>) => void;
  onPointerMove: (event: React.PointerEvent<HTMLButtonElement>) => void;
  onPointerEnd: (event: React.PointerEvent<HTMLButtonElement>) => void;
  onClick: () => void;
  onToggleExpand: () => void;
}

function NodeCard({
  module,
  computation,
  node,
  position,
  ports,
  copyGroup,
  useOpName,
  showLastNameOnly,
  showMemoryLocation,
  stateClass,
  expanded,
  onPointerDown,
  onPointerMove,
  onPointerEnd,
  onClick,
  onToggleExpand
}: NodeCardProps) {
  const caption = copyGroupCaption(node, copyGroup);
  const opName = useOpName && !copyGroup ? hloOpName(node.raw) : null;
  const locations = showMemoryLocation ? nodeMemoryLocations(module, computation, node) : [];
  const canExpand = calledComputations(module, node).length > 0;

  let label = caption.label;
  let title = `%${node.name}`;
  if (copyGroup) title = `%${copyGroup.start.name} + %${copyGroup.done.name}`;
  if (opName) {
    label = showLastNameOnly ? lastOpNameSegment(opName) : opName.replaceAll('/', '/\n');
    title = `${opName}\nHLO: %${node.name}`;
  }

  return (
    <Fragment>
      <button
        type="button"
        data-node-id={node.id}
        className={cx('node-card', nodeCategory(node), copyGroup && 'copy-group', canExpand && 'has-expand', stateClass)}
        style={{ left: position.x, top: position.y }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerEnd}
        onPointerCancel={onPointerEnd}
        onClick={onClick}
      >
        <span className="node-top">
          <span className="node-op">{node.op}</span>
          {node.root && <span className="root-tag">ROOT</span>}
        </span>
        <strong className={opName ? 'op-name' : undefined} title={title}>
          {label}
        </strong>
        <span className="node-detail" title={caption.detail}>
          {caption.detail}
        </span>
        {locations.length > 0 && (
          <span className="node-memory-list">
            {locations.map(location => (
              <span
                key={location.path ?? 'result'}
                className={`node-memory memory-space-${location.space}`}
                title={memoryTooltip(location)}
              >
                {location.path !== null && <span className="node-memory-path">{location.path}</span>}
                {location.label}
              </span>
            ))}
          </span>
        )}
        {copyGroup && <span className="copy-group-members">copy-start + copy-done</span>}
        {ports?.map((port, index) => (
          <span key={index} className="node-port" aria-hidden="true" style={{ left: port.x - 4, top: port.y - 4 }} />
        ))}
      </button>

      {canExpand && (
        <button
          type="button"
          className={cx('node-expand', expanded && 'active')}
          style={{ left: position.x + NODE_WIDTH - 78, top: position.y + 9 }}
          aria-label={`${expanded ? 'Close expansion of' : 'Expand calls of'} %${node.name}`}
          aria-expanded={expanded}
          aria-controls="computation-explorer"
          onPointerDown={event => event.stopPropagation()}
          onClick={event => {
            event.stopPropagation();
            onToggleExpand();
          }}
        >
          {expanded ? 'Close' : 'Expand'}
        </button>
      )}
    </Fragment>
  );
}

interface OverviewCardProps {
  computation: Computation;
  position: Point;
  links: ComputationLink[];
  onOpen: () => void;
}

function OverviewCard({ computation, position, links, onOpen }: OverviewCardProps) {
  const role = computationRole(computation, links);
  return (
    <button
      type="button"
      className={`overview-card ${role.toLowerCase().replaceAll(' ', '-')}`}
      style={{ left: position.x, top: position.y }}
      onClick={onOpen}
    >
      <span className="overview-role">{role}</span>
      <strong title={`%${computation.name}`}>%{computation.name}</strong>
      <span className="overview-meta">
        {computation.nodes.length} instructions <span>Open graph ↗</span>
      </span>
    </button>
  );
}

function controlsStatus(state: {
  pathPicking: boolean;
  pathTarget: string | null;
  path: string[] | null;
  selected: string | null;
  focused: boolean;
  visibleCount: number;
  totalCount: number;
}) {
  if (state.pathPicking) return 'Click the destination node';
  if (state.pathTarget) return state.path ? `${state.path.length - 1} hops` : 'No dependency path';
  if (!state.focused) return '';
  return state.selected ? `${state.visibleCount} of ${state.totalCount} nodes` : 'Select a node';
}

interface GraphControlsProps {
  hops: number;
  focused: boolean;
  selected: boolean;
  pathPicking: boolean;
  hasPathTarget: boolean;
  status: string;
  onHopsChange: (value: number) => void;
  onToggleFocus: () => void;
  onTogglePathPicking: () => void;
  onClearPath: () => void;
}

function GraphControls({
  hops,
  focused,
  selected,
  pathPicking,
  hasPathTarget,
  status,
  onHopsChange,
  onToggleFocus,
  onTogglePathPicking,
  onClearPath
}: GraphControlsProps) {
  return (
    <div className="graph-controls">
      <select aria-label="Neighborhood size" value={hops} onChange={event => onHopsChange(Number(event.target.value))}>
        <option value="1">1 hop</option>
        <option value="2">2 hops</option>
        <option value="3">3 hops</option>
      </select>
      <button
        type="button"
        disabled={!selected && !focused}
        className={focused ? 'active' : ''}
        aria-pressed={focused}
        onClick={onToggleFocus}
      >
        {focused ? 'Show all' : 'View'}
      </button>
      <button
        type="button"
        disabled={!selected}
        aria-pressed={pathPicking}
        className={pathPicking ? 'active' : ''}
        onClick={onTogglePathPicking}
      >
        Find path
      </button>
      {hasPathTarget && (
        <button type="button" onClick={onClearPath}>
          Clear path
        </button>
      )}
      <span>{status}</span>
    </div>
  );
}
