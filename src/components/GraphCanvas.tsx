import { Fragment, forwardRef, useEffect, useImperativeHandle, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { computationLinks, dependencyNeighborhood, nodeCategory, nodeSummary, reachable, shortestDependencyPath } from '../lib/parser';
import { computationRole, linkLabel, layoutInstructions, layoutOverview, NODE_HEIGHT, NODE_WIDTH, OVERVIEW_HEIGHT, OVERVIEW_WIDTH } from '../lib/graph-layout';
import { hloOpName, lastOpNameSegment } from '../lib/metadata';
import { memoryLocations, type MemoryLocation } from '../lib/memory-location';
import type { HloModule } from '../lib/types';
import { ComputationExplorer } from './ComputationExplorer';

function memoryTooltip(location: MemoryLocation) {
  const result = location.path === null ? '结果' : `结果第 ${location.path} 项`;
  const detail = !location.explicit ? '省略 S(0)' :
    location.label.startsWith('S(') ? '后端专用内存空间' : `S(${location.space})`;
  return `${result}：${location.label}（${detail}）`;
}

export interface GraphHandle {
  fit: () => void;
  centerNode: (name: string) => void;
}

interface GraphCanvasProps {
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
  { module, current, selected, zoom, useOpName, showLastNameOnly, showMemoryLocation, onZoom, onSelect, onComputation }, ref
) {
  const computation = current ? module.byName.get(current) : null;
  const [focusRadius, setFocusRadius] = useState<number | null>(null);
  const [hops, setHops] = useState(1);
  const [pathPicking, setPathPicking] = useState(false);
  const [pathTarget, setPathTarget] = useState<string | null>(null);
  const [expandedNodeName, setExpandedNodeName] = useState<string | null>(null);
  useEffect(() => { setPathPicking(false); setPathTarget(null); }, [selected, current, module]);
  useEffect(() => { setExpandedNodeName(null); }, [selected, current, module]);
  const path = useMemo(() => computation && selected && pathTarget ? shortestDependencyPath(computation, selected, pathTarget) : null, [computation, selected, pathTarget]);
  const pathEdges = useMemo(() => path ? new Set(path.slice(1).map((name, index) => [path[index], name].sort().join('\u0000'))) : null, [path]);
  const visibleNames = useMemo(() => computation && selected
    ? pathTarget ? new Set(path || [selected, pathTarget]) : focusRadius !== null && !pathPicking ? dependencyNeighborhood(computation, selected, focusRadius) : null
    : null, [computation, selected, pathTarget, path, focusRadius, pathPicking]);
  const visibleNodes = useMemo(() => computation ? computation.nodes.filter(node => !visibleNames || visibleNames.has(node.name)) : [], [computation, visibleNames]);
  const viewComputation = useMemo(() => computation ? { ...computation, nodes: visibleNodes, byName: new Map(visibleNodes.map(node => [node.name, node])) } : null, [computation, visibleNodes]);
  const [nodeHeights, setNodeHeights] = useState<Record<string, number>>({});
  const layout = useMemo(() => viewComputation ? layoutInstructions(viewComputation, new Map(visibleNodes.map(node => [node.name, nodeHeights[node.id] ?? NODE_HEIGHT]))) : layoutOverview(module), [viewComputation, visibleNodes, module, nodeHeights]);
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
  const dragRef = useRef<{ name: string; pointerId: number; x: number; y: number; position: { x: number; y: number }; moved: boolean } | null>(null);
  const suppressClickRef = useRef<string | null>(null);
  useEffect(() => { setNodeOffsets({}); }, [module]);
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
          if (next[id] !== height) { next[id] = height; changed = true; }
        }
        return changed ? next : prior;
      });
    });
    cards.forEach(card => observer.observe(card));
    return () => observer.disconnect();
  }, [visibleNodes, useOpName, showLastNameOnly, showMemoryLocation]);
  const nodeHeight = (name: string) => computation ? nodeHeights[`${computation.name}/${name}`] ?? NODE_HEIGHT : NODE_HEIGHT;
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
  const stageWidth = computation ? Math.max(layout.width, ...[...positions.values()].map(point => point.x + NODE_WIDTH + 58)) : layout.width;
  const stageHeight = computation ? Math.max(layout.height, ...[...positions].map(([name, point]) => point.y + nodeHeight(name) + 66)) : layout.height;
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
      const current = zoomRef.current;
      if (Math.abs(next - current) < 0.001) return;
      const rect = scroller.getBoundingClientRect();
      const x = clientX - rect.left, y = clientY - rect.top;
      pendingZoomAnchorRef.current = {
        x, y,
        canvasX: (scroller.scrollLeft + x) / renderedZoomRef.current,
        canvasY: (scroller.scrollTop + y) / renderedZoomRef.current
      };
      zoomRef.current = next;
      onZoom(next);
    };
    const wheel = (event: WheelEvent) => {
      if (!event.ctrlKey || gestureStartZoomRef.current) return;
      event.preventDefault();
      zoomAt(event.clientX, event.clientY, Math.max(0.45, Math.min(1.5, zoomRef.current * Math.exp(-event.deltaY / 100))));
    };
    const gestureStart = (event: Event) => {
      event.preventDefault();
      gestureStartZoomRef.current = zoomRef.current;
    };
    const gestureChange = (event: Event) => {
      event.preventDefault();
      const gesture = event as Event & { scale: number; clientX: number; clientY: number };
      zoomAt(gesture.clientX, gesture.clientY, Math.max(0.45, Math.min(1.5, (gestureStartZoomRef.current || zoomRef.current) * gesture.scale)));
    };
    const gestureEnd = () => { gestureStartZoomRef.current = 0; };
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
    onZoom(Math.max(0.45, Math.min(1, (scroller.clientWidth - 32) / stageWidth, (scroller.clientHeight - 32) / stageHeight)));
    scroller.scrollTo(0, 0);
  };
  useImperativeHandle(ref, () => ({
    fit,
    centerNode(name) {
      const position = positions.get(name);
      const scroller = scrollerRef.current;
      if (!position || !scroller) return;
      scroller.scrollTo({
        left: (position.x + NODE_WIDTH / 2) * zoom - scroller.clientWidth / 2,
        top: (position.y + nodeHeight(name) / 2) * zoom - scroller.clientHeight / 2,
        behavior: 'smooth'
      });
    }
  }));
  useEffect(() => {
    const frame = requestAnimationFrame(fit);
    return () => cancelAnimationFrame(frame);
    // The view changes its layout. A zoom button only changes zoom, so it must not refit.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layout]);
  useEffect(() => {
    if (!selected || !computation) return;
    const scroller = scrollerRef.current;
    if (!scroller) return;
    const ensureVisible = () => {
      const { positions, zoom, nodeHeights } = viewRef.current;
      const position = positions.get(selected);
      if (!scroller || !position) return;
      const height = nodeHeights[`${computation.name}/${selected}`] ?? NODE_HEIGHT;
      const left = position.x * zoom, right = (position.x + NODE_WIDTH) * zoom;
      const top = position.y * zoom, bottom = (position.y + height) * zoom;
      if (left < scroller.scrollLeft + 18 || right > scroller.scrollLeft + scroller.clientWidth - 18 ||
          top < scroller.scrollTop + 18 || bottom > scroller.scrollTop + scroller.clientHeight - 18) {
        scroller.scrollTo({
          left: Math.max(0, (position.x + NODE_WIDTH / 2) * zoom - scroller.clientWidth / 2),
          top: Math.max(0, (position.y + height / 2) * zoom - scroller.clientHeight / 2),
          behavior: 'smooth'
        });
      }
    };
    const frame = requestAnimationFrame(ensureVisible);
    const observer = new ResizeObserver(ensureVisible);
    observer.observe(scroller);
    return () => { cancelAnimationFrame(frame); observer.disconnect(); };
    // Resize tracks the canvas when the inspector docks or the viewport changes.
    // Node dragging updates viewRef without triggering a recenter.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected, current, nodeHeights]);

  const instructionEdges = visibleNodes.flatMap(target => [
    ...target.operands.map(sourceName => ({ sourceName, control: false })),
    ...target.controlPredecessors.map(sourceName => ({ sourceName, control: true }))
  ].map(({ sourceName, control }, index) => {
    const from = positions.get(sourceName), to = positions.get(target.name);
    if (!from || !to) return null;
    if (pathEdges && !pathEdges.has([sourceName, target.name].sort().join('\u0000'))) return null;
    const x1 = from.x + NODE_WIDTH, y1 = from.y + nodeHeight(sourceName) / 2;
    const edgeCount = target.operands.length + target.controlPredecessors.length;
    const x2 = to.x, y2 = to.y + nodeHeight(target.name) / 2 + (index - (edgeCount - 1) / 2) * 15;
    const dx = Math.max(48, (x2 - x1) * 0.48);
    const highlightedUp = !!selectedNode && (target.name === selected || (upstream.has(sourceName) && upstream.has(target.name)));
    const highlightedDown = !!selectedNode && (sourceName === selected || (downstream.has(sourceName) && downstream.has(target.name)));
    return <path key={`${sourceName}-${target.name}-${index}`} className={`edge${control ? ' control-edge' : ''}${pathEdges ? ' path-edge' : ''}${highlightedUp ? ' upstream' : ''}${highlightedDown ? ' downstream' : ''}${selectedNode && !pathEdges && !highlightedUp && !highlightedDown ? ' dimmed' : ''}`}
      d={`M${x1} ${y1} C${x1 + dx} ${y1},${x2 - dx} ${y2},${x2 - 8} ${y2}`} markerEnd="url(#arrow)" />;
  }));

  const overviewEdges = links.map((link, index) => {
    const from = layout.positions.get(link.from), to = layout.positions.get(link.to);
    if (!from || !to) return null;
    const x1 = from.x + OVERVIEW_WIDTH, y1 = from.y + OVERVIEW_HEIGHT / 2;
    const x2 = to.x, y2 = to.y + OVERVIEW_HEIGHT / 2;
    return <g key={`${link.from}-${link.to}-${index}`}>
      <path className="call-edge" d={`M${x1} ${y1} C${x1 + 32} ${y1},${x2 - 32} ${y2},${x2 - 8} ${y2}`} markerEnd="url(#call-arrow)">
        <title>%{link.from} → %{link.to} via %{link.via} ({link.role})</title>
      </path>
      <text className="call-label" x={(x1 + x2) / 2 - 4} y={(y1 + y2) / 2 - 9} textAnchor="middle">{linkLabel(link.role)}</text>
    </g>;
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
    const pan = panRef.current, scroller = scrollerRef.current;
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
    const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
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
      onZoom(Math.min(1.5, zoom + 0.15));
    } else if (event.key === '-' || event.key === '_') {
      event.preventDefault();
      onZoom(Math.max(0.45, zoom - 0.15));
    } else if (event.key === '0') {
      event.preventDefault();
      fit();
    }
  };

  return <section className={`graph-shell${expandedNode ? ' explorer-open' : ''}`} aria-label="HLO graph">
    <div ref={scrollerRef} className="graph-scroller overscroll-contain" tabIndex={0} aria-label="Graph canvas: + and - to zoom, 0 to fit" onKeyDown={handleZoomKey} onPointerDown={startPan} onPointerMove={movePan} onPointerUp={endPan} onPointerCancel={endPan}
      onDoubleClick={event => { if (computation && !(event.target instanceof Element && event.target.closest('.node-card'))) onSelect(null); }}>
      <div className="graph-content" style={{ width: stageWidth * zoom, height: stageHeight * zoom }}>
        <div className="graph-stage" style={{ width: stageWidth, height: stageHeight, transform: `scale(${zoom})` }}>
          <svg className="graph-edges" width={stageWidth} height={stageHeight} viewBox={`0 0 ${stageWidth} ${stageHeight}`} aria-hidden="true">
            <defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7" /></marker>
              <marker id="call-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8" /></marker></defs>
            {computation ? instructionEdges : overviewEdges}
          </svg>
          <div className="graph-nodes">
            {computation ? visibleNodes.map(node => {
              const position = positions.get(node.name)!;
              const detail = nodeSummary(node);
              const locations = showMemoryLocation ? memoryLocations(node.type) : [];
              const opName = useOpName ? hloOpName(node.raw) : null;
              const label = opName ? (showLastNameOnly ? lastOpNameSegment(opName) : opName.replaceAll('/', '/\n')) : `%${node.name}`;
              const canExpand = Object.values(node.calls).some(name => module.byName.has(name));
              const isExpanded = expandedNodeName === node.name;
              return <Fragment key={node.name}><button type="button" data-node-id={node.id} className={`node-card ${nodeCategory(node)}${canExpand ? ' has-expand' : ''}${draggingNode === node.name ? ' dragging' : ''}${selected === node.name ? ' selected' : ''}${path?.includes(node.name) ? ' path-node' : ''}${upstream.has(node.name) ? ' upstream' : ''}${downstream.has(node.name) ? ' downstream' : ''}${selectedNode && selected !== node.name && !path?.includes(node.name) && !upstream.has(node.name) && !downstream.has(node.name) ? ' dimmed' : ''}`}
                style={{ left: position.x, top: position.y }} onPointerDown={event => startNodeDrag(event, node.name)} onPointerMove={moveNode} onPointerUp={endNodeDrag} onPointerCancel={endNodeDrag}
                onClick={() => { if (suppressClickRef.current === node.name) { suppressClickRef.current = null; return; } if (pathPicking && selected && node.name !== selected) { setPathTarget(node.name); setPathPicking(false); return; } onSelect(selected === node.name ? null : node.name); }}>
                <span className="node-top"><span className="node-op">{node.op}</span>{node.root && <span className="root-tag">ROOT</span>}</span>
                <strong className={opName ? 'op-name' : undefined} title={opName ? `${opName}\nHLO: %${node.name}` : `%${node.name}`}>{label}</strong><span className="node-detail" title={detail}>{detail}</span>
                {locations.length > 0 && <span className="node-memory-list">{locations.map(location => <span key={location.path ?? 'result'} className={`node-memory memory-space-${location.space}`} title={memoryTooltip(location)}>
                  {location.path !== null && <span className="node-memory-path">{location.path}</span>}{location.label}
                </span>)}</span>}
                <span className="node-port left" /><span className="node-port right" />
              </button>{canExpand && <button type="button" className={`node-expand${isExpanded ? ' active' : ''}`} style={{ left: position.x + NODE_WIDTH - 78, top: position.y + 9 }}
                aria-label={`${isExpanded ? 'Close expansion of' : 'Expand calls of'} %${node.name}`} aria-expanded={isExpanded} aria-controls="computation-explorer"
                onPointerDown={event => event.stopPropagation()} onClick={event => { event.stopPropagation(); setExpandedNodeName(isExpanded ? null : node.name); }}>
                {isExpanded ? 'Close' : 'Expand'}
              </button>}</Fragment>;
            }) : module.computations.map(c => {
              const position = layout.positions.get(c.name)!;
              const role = computationRole(c, links);
              return <button type="button" key={c.name} className={`overview-card ${role.toLowerCase().replaceAll(' ', '-')}`} style={{ left: position.x, top: position.y }} onClick={() => onComputation(c.name)}>
                <span className="overview-role">{role}</span><strong title={`%${c.name}`}>%{c.name}</strong>
                <span className="overview-meta">{c.nodes.length} instructions <span>Open graph ↗</span></span>
              </button>;
            })}
          </div>
        </div>
      </div>
    </div>
    {computation && <div className="graph-controls">
      <select aria-label="Neighborhood size" value={hops} onChange={event => { const value = Number(event.target.value); setHops(value); if (focusRadius !== null) setFocusRadius(value); }}>
        <option value="1">1 hop</option><option value="2">2 hops</option><option value="3">3 hops</option>
      </select>
      <button type="button" disabled={!selected && focusRadius === null} className={focusRadius !== null ? 'active' : ''} aria-pressed={focusRadius !== null}
        onClick={() => { setFocusRadius(focusRadius === null ? hops : null); setPathTarget(null); setPathPicking(false); }}>{focusRadius === null ? 'View' : 'Show all'}</button>
      <button type="button" disabled={!selected} className={pathPicking ? 'active' : ''} onClick={() => { setPathTarget(null); setPathPicking(value => !value); }}>Find path</button>
      {pathTarget && <button type="button" onClick={() => setPathTarget(null)}>Clear path</button>}
      <span>{pathPicking ? 'Click the destination node' : pathTarget ? path ? `${path.length - 1} hops` : 'No dependency path' : selected && focusRadius !== null ? `${visibleNodes.length} of ${computation.nodes.length} nodes` : !selected && focusRadius !== null ? 'Select a node' : ''}</span>
    </div>}
    {expandedNode && <ComputationExplorer key={expandedNode.id} module={module} rootNode={expandedNode} onClose={() => setExpandedNodeName(null)} onOpenFull={onComputation} />}
    <div className="graph-hint"><span className="flow-icon">→</span> {computation ? 'Data flow · Drag nodes · Background pans' : 'Calls · Drag to pan'} <span className="hint-divider">·</span> Pinch / + / − zoom · 0 fit <span className="hint-divider">·</span> {computation ? `${visibleNodes.length} nodes · ${instructionEdges.filter(Boolean).length} edges` : `${module.computations.length} computations · ${links.length} links`}</div>
  </section>;
});
