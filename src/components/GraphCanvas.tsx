import { Fragment, forwardRef, useEffect, useImperativeHandle, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { computationLinks, dependencyNeighborhood, nodeCategory, nodeSummary, reachable, shortestDependencyPath } from '../lib/parser';
import {
  clampZoom,
  computationRole,
  linkLabel,
  MIN_ZOOM,
  NODE_HEIGHT,
  NODE_WIDTH,
  type LayoutDirection,
  type LayoutMode
} from '../lib/graph-layout';
import { hloOpName, lastOpNameSegment } from '../lib/metadata';
import { nodeMemoryLocations, type MemoryLocation } from '../lib/memory-location';
import { memorySpaceText } from '../lib/memory-space';
import type { HloModule } from '../lib/types';
import type { CopyGrouping } from '../lib/copy-grouping';
import { useGraphLayouts } from '../hooks/useGraphLayouts';
import { edgePath, movedEdgePoints } from '../lib/graph-routing';
import { ComputationExplorer } from './ComputationExplorer';

function memoryTooltip(location: MemoryLocation) {
  return `${location.path === null ? 'Result' : `Result element ${location.path}`}: ${memorySpaceText(location.space, location.explicit)}`;
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
  const [focusRadius, setFocusRadius] = useState<number | null>(null);
  const [hops, setHops] = useState(1);
  const [pathPicking, setPathPicking] = useState(false);
  const [pathTarget, setPathTarget] = useState<string | null>(null);
  const [expandedNodeName, setExpandedNodeName] = useState<string | null>(null);
  useEffect(() => {
    setPathPicking(false);
    setPathTarget(null);
  }, [selected, current, module, autoGroup]);
  useEffect(() => {
    setExpandedNodeName(null);
  }, [selected, current, module, autoGroup]);
  const path = useMemo(
    () => (computation && selected && pathTarget ? shortestDependencyPath(computation, selected, pathTarget) : null),
    [computation, selected, pathTarget]
  );
  const pathEdges = useMemo(
    () => (path ? new Set(path.slice(1).map((name, index) => [path[index], name].sort().join('\u0000'))) : null),
    [path]
  );
  const visibleNames = useMemo(
    () =>
      computation && selected
        ? pathTarget
          ? new Set(path || [selected, pathTarget])
          : focusRadius !== null && !pathPicking
            ? dependencyNeighborhood(computation, selected, focusRadius)
            : null
        : null,
    [computation, selected, pathTarget, path, focusRadius, pathPicking]
  );
  const visibleNodes = useMemo(
    () => (computation ? computation.nodes.filter(node => !visibleNames || visibleNames.has(node.name)) : []),
    [computation, visibleNames]
  );
  const viewComputation = useMemo(
    () => (computation ? { ...computation, nodes: visibleNodes, byName: new Map(visibleNodes.map(node => [node.name, node])) } : null),
    [computation, visibleNodes]
  );
  const [nodeHeights, setNodeHeights] = useState<Record<string, number>>({});
  const [viewport, setViewport] = useState<{ width: number; height: number } | null>(null);
  const [arrangeRevision, setArrangeRevision] = useState(0);
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
    const score = (candidate: typeof layouts.horizontal) => Math.min(width / candidate.width, height / candidate.height);
    autoDirection = score(layouts.vertical) > score(layouts.horizontal) * 1.08 ? 'vertical' : 'horizontal';
    setAutoPick({ layouts, arrangeRevision, measured: !!viewport, direction: autoDirection });
  }
  const direction = layoutMode === 'auto' ? autoDirection : layoutMode;
  const layout = layouts[direction];
  const vertical = direction === 'vertical';
  const links = useMemo(() => computationLinks(module), [module]);
  const scrollerRef = useRef<HTMLDivElement>(null);
  const zoomRef = useRef(zoom);
  const renderedZoomRef = useRef(zoom);
  const gestureStartZoomRef = useRef(0);
  const pendingZoomAnchorRef = useRef<{ x: number; y: number; canvasX: number; canvasY: number } | null>(null);
  zoomRef.current = zoom;
  renderedZoomRef.current = zoom;
  const panRef = useRef<{ x: number; y: number; left: number; top: number } | null>(null);
  const [nodeOffsets, setNodeOffsets] = useState<Record<string, { x: number; y: number }>>({});
  const [draggingNode, setDraggingNode] = useState<string | null>(null);
  const dragRef = useRef<{
    name: string;
    pointerId: number;
    x: number;
    y: number;
    position: { x: number; y: number };
    moved: boolean;
  } | null>(null);
  const suppressClickRef = useRef<string | null>(null);
  useEffect(() => {
    setNodeOffsets({});
  }, [module, current, direction, autoGroup]);
  useEffect(() => {
    const scroller = scrollerRef.current;
    if (!scroller) return;
    const observer = new ResizeObserver(() => {
      const width = Math.max(1, scroller.clientWidth - 32),
        height = Math.max(1, scroller.clientHeight - 32);
      setViewport(prior => (prior?.width === width && prior.height === height ? prior : { width, height }));
    });
    observer.observe(scroller);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    if (!computation) return;
    const cards = scrollerRef.current?.querySelectorAll<HTMLButtonElement>('.node-card');
    if (!cards) return;
    const observer = new ResizeObserver(entries => {
      setNodeHeights(prior => {
        let changed = false;
        const next = { ...prior };
        for (const entry of entries) {
          const id = (entry.target as HTMLElement).dataset.nodeId;
          if (!id) continue;
          const height = (entry.target as HTMLElement).offsetHeight;
          if (next[id] !== height) {
            next[id] = height;
            changed = true;
          }
        }
        return changed ? next : prior;
      });
    });
    cards.forEach(card => observer.observe(card));
    return () => observer.disconnect();
  }, [computation, visibleNodes, useOpName, showLastNameOnly, showMemoryLocation]);
  const nodeHeight = (name: string) => (computation ? (nodeHeights[`${computation.name}/${name}`] ?? NODE_HEIGHT) : NODE_HEIGHT);
  const positions = useMemo(() => {
    if (!computation) return layout.positions;
    const adjusted = new Map(layout.positions);
    for (const node of visibleNodes) {
      const base = layout.positions.get(node.name);
      const offset = nodeOffsets[`${computation.name}/${node.name}`];
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
  const selectedNode = selected && computation?.byName.get(selected);
  const expandedNode = expandedNodeName && computation?.byName.get(expandedNodeName);
  const upstream = selectedNode && computation ? reachable(computation, selectedNode.name, 'up') : new Set<string>();
  const downstream = selectedNode && computation ? reachable(computation, selectedNode.name, 'down') : new Set<string>();

  useLayoutEffect(() => {
    const anchor = pendingZoomAnchorRef.current;
    const scroller = scrollerRef.current;
    if (!anchor || !scroller) return;
    scroller.scrollLeft = anchor.canvasX * zoom - anchor.x;
    scroller.scrollTop = anchor.canvasY * zoom - anchor.y;
    pendingZoomAnchorRef.current = null;
  }, [zoom]);
  useEffect(() => {
    const scroller = scrollerRef.current;
    if (!scroller) return;
    const zoomAt = (clientX: number, clientY: number, next: number) => {
      const currentZoom = zoomRef.current;
      if (Math.abs(next - currentZoom) < 0.001) return;
      const rect = scroller.getBoundingClientRect();
      const x = clientX - rect.left,
        y = clientY - rect.top;
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
      zoomAt(gesture.clientX, gesture.clientY, clampZoom((gestureStartZoomRef.current || zoomRef.current) * gesture.scale));
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
    onZoom(Math.max(MIN_ZOOM, Math.min(1, (scroller.clientWidth - 32) / stageWidth, (scroller.clientHeight - 32) / stageHeight)));
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
    // The view changes its layout. A zoom button only changes zoom, so it must not refit.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layout, arrangeRevision]);
  useEffect(() => {
    if (!selected || !computation) return;
    const scroller = scrollerRef.current;
    if (!scroller) return;
    const ensureVisible = () => {
      const view = viewRef.current;
      const position = view.positions.get(selected);
      if (!scroller || !position) return;
      const height = view.nodeHeights[`${computation.name}/${selected}`] ?? NODE_HEIGHT;
      const scale = view.zoom;
      const left = position.x * scale,
        right = (position.x + NODE_WIDTH) * scale;
      const top = position.y * scale,
        bottom = (position.y + height) * scale;
      if (
        left < scroller.scrollLeft + 18 ||
        right > scroller.scrollLeft + scroller.clientWidth - 18 ||
        top < scroller.scrollTop + 18 ||
        bottom > scroller.scrollTop + scroller.clientHeight - 18
      ) {
        scroller.scrollTo({
          left: Math.max(0, (position.x + NODE_WIDTH / 2) * scale - scroller.clientWidth / 2),
          top: Math.max(0, (position.y + height / 2) * scale - scroller.clientHeight / 2),
          behavior: 'smooth'
        });
      }
    };
    const frame = requestAnimationFrame(ensureVisible);
    const observer = new ResizeObserver(ensureVisible);
    observer.observe(scroller);
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
    };
    // Resize tracks the canvas when the inspector docks or the viewport changes.
    // Node dragging updates viewRef without triggering a recenter.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected, current, nodeHeights, layout]);

  const offsets = new Map(
    visibleNodes.flatMap(node => {
      const offset = nodeOffsets[node.id];
      return offset ? [[node.name, offset] as const] : [];
    })
  );
  const ports = new Map<string, { x: number; y: number }[]>();
  for (const edge of layout.edges) {
    for (const [name, point] of [
      [edge.source, edge.points[0]],
      [edge.target, edge.points.at(-1)!]
    ] as const) {
      const base = layout.positions.get(name);
      if (!base) continue;
      const list = ports.get(name) ?? [];
      list.push({ x: point.x - base.x, y: point.y - base.y });
      ports.set(name, list);
    }
  }
  const instructionEdges = layout.edges
    .filter(edge => !pathEdges || pathEdges.has([edge.source, edge.target].sort().join('\u0000')))
    .map(edge => {
      const highlightedUp = !!selectedNode && (edge.target === selected || (upstream.has(edge.source) && upstream.has(edge.target)));
      const highlightedDown = !!selectedNode && (edge.source === selected || (downstream.has(edge.source) && downstream.has(edge.target)));
      return (
        <path
          key={edge.id}
          className={`edge${edge.control ? ' control-edge' : ''}${pathEdges ? ' path-edge' : ''}${highlightedUp ? ' upstream' : ''}${highlightedDown ? ' downstream' : ''}${selectedNode && !pathEdges && !highlightedUp && !highlightedDown ? ' dimmed' : ''}`}
          d={edgePath(movedEdgePoints(edge, offsets), direction)}
          markerEnd="url(#arrow)"
        />
      );
    });
  const overviewEdges = layout.edges.map(edge => {
    const link = links[edge.index];
    return (
      <g key={edge.id}>
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
  });

  const startPan = (event: React.PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0 || (event.target instanceof Element && event.target.closest('.node-card, .overview-card'))) return;
    const scroller = scrollerRef.current!;
    scroller.focus({ preventScroll: true });
    panRef.current = { x: event.clientX, y: event.clientY, left: scroller.scrollLeft, top: scroller.scrollTop };
    scroller.setPointerCapture(event.pointerId);
    scroller.classList.add('panning');
  };
  const movePan = (event: React.PointerEvent<HTMLDivElement>) => {
    const pan = panRef.current,
      scroller = scrollerRef.current;
    if (!pan || !scroller) return;
    scroller.scrollLeft = pan.left - (event.clientX - pan.x);
    scroller.scrollTop = pan.top - (event.clientY - pan.y);
  };
  const endPan = () => {
    panRef.current = null;
    scrollerRef.current?.classList.remove('panning');
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
    const dx = event.clientX - drag.x,
      dy = event.clientY - drag.y;
    if (!drag.moved && Math.hypot(dx, dy) < 4) return;
    drag.moved = true;
    setDraggingNode(drag.name);
    const base = layout.positions.get(drag.name)!;
    const x = Math.max(12, drag.position.x + dx / zoom);
    const y = Math.max(12, drag.position.y + dy / zoom);
    setNodeOffsets(prior => ({ ...prior, [`${computation.name}/${drag.name}`]: { x: x - base.x, y: y - base.y } }));
  };
  const endNodeDrag = (event: React.PointerEvent<HTMLButtonElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    if (drag.moved) suppressClickRef.current = drag.name;
    dragRef.current = null;
    setDraggingNode(null);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  };
  const handleZoomKey = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.altKey || event.ctrlKey || event.metaKey) return;
    if (event.key === '+' || event.key === '=') {
      event.preventDefault();
      onZoom(clampZoom(zoom + 0.15));
    } else if (event.key === '-' || event.key === '_') {
      event.preventDefault();
      onZoom(clampZoom(zoom - 0.15));
    } else if (event.key === '0') {
      event.preventDefault();
      fit();
    }
  };

  return (
    <section
      className={`graph-shell ${vertical ? 'flow-vertical' : 'flow-horizontal'}${expandedNode ? ' explorer-open' : ''}`}
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
          <button type="button" onClick={retry}>
            Retry
          </button>
        </div>
      )}
      <div
        ref={scrollerRef}
        className="graph-scroller overscroll-contain"
        tabIndex={0}
        aria-label="Graph canvas: + and - to zoom, 0 to fit"
        onKeyDown={handleZoomKey}
        onPointerDown={startPan}
        onPointerMove={movePan}
        onPointerUp={endPan}
        onPointerCancel={endPan}
        onDoubleClick={event => {
          if (computation && !(event.target instanceof Element && event.target.closest('.node-card'))) onSelect(null);
        }}
      >
        <div className="graph-content" style={{ width: stageWidth * zoom, height: stageHeight * zoom }}>
          <div className="graph-stage" style={{ width: stageWidth, height: stageHeight, transform: `scale(${zoom})` }}>
            <svg
              className="graph-edges"
              width={stageWidth}
              height={stageHeight}
              viewBox={`0 0 ${stageWidth} ${stageHeight}`}
              aria-hidden="true"
            >
              <defs>
                <marker
                  id="arrow"
                  markerWidth="7"
                  markerHeight="7"
                  viewBox="0 0 7 7"
                  markerUnits="userSpaceOnUse"
                  refX="6"
                  refY="3.5"
                  orient="auto"
                >
                  <path d="M1 1 L6 3.5 L1 6 Z" />
                </marker>
                <marker
                  id="call-arrow"
                  markerWidth="7"
                  markerHeight="7"
                  viewBox="0 0 7 7"
                  markerUnits="userSpaceOnUse"
                  refX="6"
                  refY="3.5"
                  orient="auto"
                >
                  <path d="M1 1 L6 3.5 L1 6 Z" />
                </marker>
              </defs>
              {computation ? instructionEdges : overviewEdges}
            </svg>
            <div className="graph-nodes">
              {computation
                ? visibleNodes.map(node => {
                    const position = positions.get(node.name)!;
                    const copyGroup = copyGrouping?.groups.get(node.name);
                    const detail = copyGroup ? copyGroup.shape : nodeSummary(node);
                    const locations = showMemoryLocation ? nodeMemoryLocations(module, computation, node) : [];
                    const opName = useOpName && !copyGroup ? hloOpName(node.raw) : null;
                    const label = copyGroup
                      ? copyGroup.direction
                      : opName
                        ? showLastNameOnly
                          ? lastOpNameSegment(opName)
                          : opName.replaceAll('/', '/\n')
                        : `%${node.name}`;
                    const canExpand = Object.values(node.calls).some(name => module.byName.has(name));
                    const isExpanded = expandedNodeName === node.name;
                    return (
                      <Fragment key={node.name}>
                        <button
                          type="button"
                          data-node-id={node.id}
                          className={`node-card ${nodeCategory(node)}${copyGroup ? ' copy-group' : ''}${canExpand ? ' has-expand' : ''}${draggingNode === node.name ? ' dragging' : ''}${selected === node.name ? ' selected' : ''}${path?.includes(node.name) ? ' path-node' : ''}${upstream.has(node.name) ? ' upstream' : ''}${downstream.has(node.name) ? ' downstream' : ''}${selectedNode && selected !== node.name && !path?.includes(node.name) && !upstream.has(node.name) && !downstream.has(node.name) ? ' dimmed' : ''}`}
                          style={{ left: position.x, top: position.y }}
                          onPointerDown={event => startNodeDrag(event, node.name)}
                          onPointerMove={moveNode}
                          onPointerUp={endNodeDrag}
                          onPointerCancel={endNodeDrag}
                          onClick={() => {
                            if (suppressClickRef.current === node.name) {
                              suppressClickRef.current = null;
                              return;
                            }
                            if (pathPicking && selected && node.name !== selected) {
                              setPathTarget(node.name);
                              setPathPicking(false);
                              return;
                            }
                            onSelect(selected === node.name ? null : node.name);
                          }}
                        >
                          <span className="node-top">
                            <span className="node-op">{node.op}</span>
                            {node.root && <span className="root-tag">ROOT</span>}
                          </span>
                          <strong
                            className={opName ? 'op-name' : undefined}
                            title={
                              copyGroup
                                ? `%${copyGroup.start.name} + %${copyGroup.done.name}`
                                : opName
                                  ? `${opName}\nHLO: %${node.name}`
                                  : `%${node.name}`
                            }
                          >
                            {label}
                          </strong>
                          <span className="node-detail" title={detail}>
                            {detail}
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
                          {ports.get(node.name)?.map((port, index) => (
                            <span key={index} className="node-port" style={{ left: port.x - 4, top: port.y - 4 }} />
                          ))}
                        </button>
                        {canExpand && (
                          <button
                            type="button"
                            className={`node-expand${isExpanded ? ' active' : ''}`}
                            style={{ left: position.x + NODE_WIDTH - 78, top: position.y + 9 }}
                            aria-label={`${isExpanded ? 'Close expansion of' : 'Expand calls of'} %${node.name}`}
                            aria-expanded={isExpanded}
                            aria-controls="computation-explorer"
                            onPointerDown={event => event.stopPropagation()}
                            onClick={event => {
                              event.stopPropagation();
                              setExpandedNodeName(isExpanded ? null : node.name);
                            }}
                          >
                            {isExpanded ? 'Close' : 'Expand'}
                          </button>
                        )}
                      </Fragment>
                    );
                  })
                : module.computations.map(c => {
                    const position = layout.positions.get(c.name)!;
                    const role = computationRole(c, links);
                    return (
                      <button
                        type="button"
                        key={c.name}
                        className={`overview-card ${role.toLowerCase().replaceAll(' ', '-')}`}
                        style={{ left: position.x, top: position.y }}
                        onClick={() => onComputation(c.name)}
                      >
                        <span className="overview-role">{role}</span>
                        <strong title={`%${c.name}`}>%{c.name}</strong>
                        <span className="overview-meta">
                          {c.nodes.length} instructions <span>Open graph ↗</span>
                        </span>
                      </button>
                    );
                  })}
            </div>
          </div>
        </div>
      </div>
      {computation && (
        <div className="graph-controls">
          <select
            aria-label="Neighborhood size"
            value={hops}
            onChange={event => {
              const value = Number(event.target.value);
              setHops(value);
              if (focusRadius !== null) setFocusRadius(value);
            }}
          >
            <option value="1">1 hop</option>
            <option value="2">2 hops</option>
            <option value="3">3 hops</option>
          </select>
          <button
            type="button"
            disabled={!selected && focusRadius === null}
            className={focusRadius !== null ? 'active' : ''}
            aria-pressed={focusRadius !== null}
            onClick={() => {
              setFocusRadius(focusRadius === null ? hops : null);
              setPathTarget(null);
              setPathPicking(false);
            }}
          >
            {focusRadius === null ? 'View' : 'Show all'}
          </button>
          <button
            type="button"
            disabled={!selected}
            className={pathPicking ? 'active' : ''}
            onClick={() => {
              setPathTarget(null);
              setPathPicking(value => !value);
            }}
          >
            Find path
          </button>
          {pathTarget && (
            <button type="button" onClick={() => setPathTarget(null)}>
              Clear path
            </button>
          )}
          <span>
            {pathPicking
              ? 'Click the destination node'
              : pathTarget
                ? path
                  ? `${path.length - 1} hops`
                  : 'No dependency path'
                : selected && focusRadius !== null
                  ? `${visibleNodes.length} of ${computation.nodes.length} nodes`
                  : !selected && focusRadius !== null
                    ? 'Select a node'
                    : ''}
          </span>
        </div>
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
        <span className="flow-icon">{vertical ? '↓' : '→'}</span>{' '}
        {computation ? 'Data flow · Drag nodes · Background pans' : 'Calls · Drag to pan'} <span className="hint-divider">·</span> Pinch / +
        / − zoom · 0 fit <span className="hint-divider">·</span>{' '}
        {computation
          ? `${visibleNodes.length} nodes · ${instructionEdges.length} edges`
          : `${module.computations.length} computations · ${links.length} links`}
      </div>
    </section>
  );
});
