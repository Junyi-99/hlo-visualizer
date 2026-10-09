import { useEffect, useMemo, useRef, useState } from 'react';
import { instructionGuide, layoutDiagram, partColor } from '../lib/instruction-guide';
import { extractHloMetadata } from '../lib/metadata.ts';
import { nodeCategory, nodeSummary, sourceStack } from '../lib/parser';
import type { Computation, HloModule, HloNode } from '../lib/types';
import { Drawer } from 'vaul';
import { TypeTree } from './TypeTree';
import type { CopyGroup } from '../lib/copy-grouping';
import { Icon } from './Icon';

interface InspectorProps {
  copyGroup?: CopyGroup | null;
  module: HloModule;
  computation: Computation | null;
  node: HloNode | null;
  upstreamCount: number;
  downstreamCount: number;
  onClose: () => void;
  onNode: (name: string) => void;
  onComputation: (name: string) => void;
}

function ReferenceList({ names, onNode }: { names: string[]; onNode: (name: string) => void }) {
  return names.length ? (
    <>
      {names.map((name, index) => (
        <button type="button" className="reference" key={`${index}-${name}`} onClick={() => onNode(name)}>
          <span>%{name}</span>
          <span>↗</span>
        </button>
      ))}
    </>
  ) : (
    <p className="none">None in this computation</p>
  );
}

const phoneQuery = '(max-width: 700px)';

function useIsPhone() {
  const [isPhone, setIsPhone] = useState(() => window.matchMedia(phoneQuery).matches);
  useEffect(() => {
    const media = window.matchMedia(phoneQuery);
    const onChange = () => setIsPhone(media.matches);
    media.addEventListener('change', onChange);
    return () => media.removeEventListener('change', onChange);
  }, []);
  return isPhone;
}

const MIN_WIDTH = 300;
const MIN_GRAPH_WIDTH = 360;

function maxInspectorWidth() {
  const sidebar = window.matchMedia('(min-width: 1101px)').matches
    ? parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--sidebar-w')) || 0
    : 0;
  return Math.max(MIN_WIDTH, window.innerWidth - sidebar - MIN_GRAPH_WIDTH);
}
const clampWidth = (width: number) => Math.round(Math.min(maxInspectorWidth(), Math.max(MIN_WIDTH, width)));

// The docked inspector's width is the --inspector-w custom property; unset means the responsive default.
// Dragging writes the property directly so the graph does not re-render on every pointer move.
function useInspectorWidth() {
  const [width, setWidth] = useState<number | null>(() => {
    try {
      const saved = Number(localStorage.getItem('inspector-width'));
      return saved > 0 ? saved : null;
    } catch {
      return null;
    }
  });
  useEffect(() => {
    const root = document.documentElement.style;
    if (width === null) root.removeProperty('--inspector-w');
    else root.setProperty('--inspector-w', `${width}px`);
    try {
      if (width === null) localStorage.removeItem('inspector-width');
      else localStorage.setItem('inspector-width', String(width));
    } catch {
      /* storage blocked */
    }
  }, [width]);
  return [width, setWidth] as const;
}

function ResizeHandle({ panelRef }: { panelRef: React.RefObject<HTMLElement | null> }) {
  const [width, setWidth] = useInspectorWidth();
  const dragRef = useRef<{ pointerId: number; x: number; width: number; next: number } | null>(null);
  const [measured, setMeasured] = useState(MIN_WIDTH);
  useEffect(() => {
    const panel = panelRef.current;
    if (!panel) return;
    const observer = new ResizeObserver(() => setMeasured(panel.getBoundingClientRect().width));
    observer.observe(panel);
    return () => observer.disconnect();
  }, [panelRef]);
  const current = () => panelRef.current?.getBoundingClientRect().width ?? MIN_WIDTH;
  return (
    <div
      className="inspector-resize"
      role="separator"
      aria-orientation="vertical"
      aria-label="Resize node details"
      tabIndex={0}
      aria-valuemin={MIN_WIDTH}
      aria-valuenow={Math.round(width ?? measured)}
      title="Drag to resize · double-click to reset"
      onPointerDown={event => {
        if (event.button !== 0) return;
        event.preventDefault();
        const start = current();
        dragRef.current = { pointerId: event.pointerId, x: event.clientX, width: start, next: start };
        event.currentTarget.setPointerCapture(event.pointerId);
        document.documentElement.classList.add('resizing-inspector');
      }}
      onPointerMove={event => {
        const drag = dragRef.current;
        if (!drag || drag.pointerId !== event.pointerId) return;
        drag.next = clampWidth(drag.width + drag.x - event.clientX);
        document.documentElement.style.setProperty('--inspector-w', `${drag.next}px`);
      }}
      onPointerUp={event => {
        const drag = dragRef.current;
        if (!drag || drag.pointerId !== event.pointerId) return;
        dragRef.current = null;
        document.documentElement.classList.remove('resizing-inspector');
        if (drag.next !== drag.width) setWidth(drag.next);
      }}
      onPointerCancel={() => {
        dragRef.current = null;
        document.documentElement.classList.remove('resizing-inspector');
        if (width === null) document.documentElement.style.removeProperty('--inspector-w');
        else document.documentElement.style.setProperty('--inspector-w', `${width}px`);
      }}
      onDoubleClick={() => setWidth(null)}
      onKeyDown={event => {
        const step = event.shiftKey ? 80 : 20;
        if (event.key === 'ArrowLeft') setWidth(clampWidth(current() + step));
        else if (event.key === 'ArrowRight') setWidth(clampWidth(current() - step));
        else if (event.key === 'Home' || event.key === 'Enter') setWidth(null);
        else return;
        event.preventDefault();
      }}
    />
  );
}

export function Inspector({
  copyGroup,
  module,
  computation,
  node,
  upstreamCount,
  downstreamCount,
  onClose,
  onNode,
  onComputation
}: InspectorProps) {
  const isPhone = useIsPhone();
  const panelRef = useRef<HTMLElement>(null);
  const shown = node && computation ? { copyGroup, module, computation, node, upstreamCount, downstreamCount } : null;
  // On phones the inspector is a draggable bottom drawer; the last node stays rendered while it slides away.
  const [lastShown, setLastShown] = useState(shown);
  if (shown && !sameDetails(shown, lastShown)) setLastShown(shown);
  const renderDetails = (props: DetailsProps | null) =>
    props && <InstructionDetails key={props.node.id} {...props} onNode={onNode} onComputation={onComputation} />;
  const details = renderDetails(shown);
  if (isPhone)
    return (
      <Drawer.Root
        open={!!details}
        onOpenChange={open => {
          if (!open) onClose();
        }}
      >
        <Drawer.Portal>
          <Drawer.Overlay className="drawer-overlay" />
          <Drawer.Content className="inspector-drawer" aria-describedby={undefined}>
            <div className="drawer-handle" aria-hidden="true" />
            <div className="inspector-header">
              <Drawer.Title>Node details</Drawer.Title>
              <button className="icon-button" type="button" aria-label="Close node details" onClick={onClose}>
                <Icon name="close" />
              </button>
            </div>
            <div className="inspector-content overscroll-contain">{details || renderDetails(lastShown)}</div>
          </Drawer.Content>
        </Drawer.Portal>
      </Drawer.Root>
    );
  return (
    <aside ref={panelRef} className={`inspector${node ? ' inspector-visible' : ''}`}>
      {node && <ResizeHandle panelRef={panelRef} />}
      <div className="inspector-header">
        <div>
          <div className="eyebrow">INSPECTOR</div>
          <h2>Node details</h2>
        </div>
        <button className="icon-button" type="button" aria-label="Clear selection" onClick={onClose}>
          <Icon name="close" />
        </button>
      </div>
      <div className="inspector-content overscroll-contain">
        {details || (
          <div className="empty-inspector">
            <div className="empty-icon">
              <Icon name="graph" size={28} />
            </div>
            <strong>Explore the graph</strong>
            <p>Select any node to see its inputs, consumers, raw HLO, and linked computations.</p>
          </div>
        )}
      </div>
    </aside>
  );
}

type DetailsProps = Omit<InspectorProps, 'onClose' | 'onNode' | 'onComputation'> & { node: HloNode; computation: Computation };

const sameDetails = (a: DetailsProps, b: DetailsProps | null) =>
  !!b && (Object.keys(a) as (keyof DetailsProps)[]).every(key => a[key] === b[key]);

function codeTarget(target: EventTarget) {
  const part = target instanceof Element ? target.closest<HTMLElement>('[data-part]') : null;
  return { key: part?.dataset.part || null, slot: part?.closest<HTMLElement>('.hlo-slot')?.dataset.part || null };
}

function InstructionDetails({
  copyGroup,
  module,
  computation,
  node,
  upstreamCount,
  downstreamCount,
  onNode,
  onComputation
}: DetailsProps & Pick<InspectorProps, 'onNode' | 'onComputation'>) {
  const guide = useMemo(() => instructionGuide(node, { computation, module }), [node, computation, module]);
  const [activePart, setActivePart] = useState<string | null>(null);
  const [activeSlot, setActiveSlot] = useState<string | null>(null);
  const [showFullInstruction, setShowFullInstruction] = useState(false);
  const detailsRef = useRef<HTMLDivElement>(null);
  const codeRef = useRef<HTMLPreElement>(null);
  const group = guide.resultGroup;
  const metadata = extractHloMetadata(node.raw);
  const frameId = Number(metadata?.fields.find(field => field.name === 'stack_frame_id')?.value || 0);
  const stack = frameId ? sourceStack(module, frameId) : [];
  const typeKeys = new Set([
    'tuple',
    'shape',
    'order',
    'tile',
    'space',
    'dest',
    'source',
    'context',
    ...(group?.kind === 'tuple' ? group.slots.map(slot => slot.key) : [])
  ]);
  const explanations = guide.parts.filter(part => !typeKeys.has(part.key) && !(metadata && part.key === 'metadata'));
  const diagram = layoutDiagram(node);
  const semanticsUrl =
    node.op === 'copy-start'
      ? 'https://openxla.org/xla/operation_semantics#copy'
      : node.type.startsWith('(')
        ? 'https://openxla.org/xla/operation_semantics#tuple'
        : 'https://openxla.org/xla/operation_semantics';
  const semanticsLabel = node.op === 'copy-start' ? 'Copy' : node.type.startsWith('(') ? 'Tuple' : 'Operations';
  const displayType = node.type.startsWith('(') ? nodeSummary(node) : node.type;
  const displayedHtml = showFullInstruction ? guide.html : guide.compactHtml || guide.html;

  useEffect(() => {
    const elements = [...(codeRef.current?.querySelectorAll<HTMLElement>('[data-part]') || [])];
    const hasMatchingPart = elements.some(
      element =>
        element.dataset.part === activePart && (!activeSlot || element.closest<HTMLElement>('.hlo-slot')?.dataset.part === activeSlot)
    );
    elements.forEach(element => {
      const key = element.dataset.part;
      const slot = element.closest<HTMLElement>('.hlo-slot')?.dataset.part;
      const matchesPart = key === activePart && (!activeSlot || slot === activeSlot || key === activeSlot);
      const matchesSlot = element.classList.contains('hlo-slot') && key === activeSlot;
      const fallbackType = !hasMatchingPart && !activeSlot && activePart === 'space' && element.classList.contains('hlo-type');
      element.classList.toggle('active-part', !!(matchesPart || matchesSlot || fallbackType));
    });
    // oxlint-disable-next-line react/exhaustive-effect-dependencies -- dangerouslySetInnerHTML replaces the spans, so re-apply highlights when the HTML changes
  }, [activePart, activeSlot, displayedHtml]);

  const activate = (part: string | null, slot: string | null = null) => {
    setActivePart(part);
    setActiveSlot(slot);
  };

  const activateFromCode = (event: React.MouseEvent<HTMLPreElement>) => {
    const { key, slot } = codeTarget(event.target);
    if (!key) return;
    activate(key, slot);
    const container = detailsRef.current;
    const slotItem = slot ? container?.querySelector<HTMLDetailsElement>(`.type-child[data-part="${slot}"]`) : null;
    const item =
      (slotItem || container?.querySelector('.type-tree'))?.querySelector<HTMLElement>(`.type-detail[data-part="${key}"]`) ||
      slotItem ||
      container?.querySelector<HTMLElement>(`.type-child[data-part="${key}"]`) ||
      (typeKeys.has(key) ? container?.querySelector<HTMLElement>('.type-tree') : null) ||
      container?.querySelector<HTMLElement>(`.guide-row[data-part="${key}"]`) ||
      container?.querySelector<HTMLElement>(`.metadata-section[data-part="${key}"]`);
    if (!item) return;
    const tree = item.closest<HTMLDetailsElement>('.type-tree');
    if (tree) tree.open = true;
    if (slotItem) slotItem.open = true;
    if (item instanceof HTMLDetailsElement) item.open = true;
    item.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  };

  return (
    <div ref={detailsRef}>
      {copyGroup && (
        <section className="inspector-section copy-group-summary">
          <h4>Grouped copy</h4>
          <strong>{copyGroup.direction}</strong>
          <code>{copyGroup.shape}</code>
          <p>Two instructions displayed as one copy.</p>
          <ReferenceList names={[copyGroup.start.name, copyGroup.done.name]} onNode={onNode} />
        </section>
      )}
      <div className="inspector-title">
        <span className={`type-badge ${nodeCategory(node)}`}>{node.op}</span>
        {node.root && <span className="root-pill">ROOT</span>}
        <h3>%{node.name}</h3>
        <div className="muted">{displayType}</div>
      </div>
      <div className="metric-row">
        <div>
          <strong>{upstreamCount}</strong>
          <span>Upstream</span>
        </div>
        <div>
          <strong>{downstreamCount}</strong>
          <span>Downstream</span>
        </div>
        <div>
          <strong>{node.line}</strong>
          <span>Source line</span>
        </div>
      </div>
      <div className="inspector-section instruction-section">
        <h4>HLO instruction</h4>
        <pre
          ref={codeRef}
          className="hlo-code"
          onClick={activateFromCode}
          onMouseOver={event => {
            const { key, slot } = codeTarget(event.target);
            activate(key, slot);
          }}
          onMouseLeave={() => activate(null)}
          dangerouslySetInnerHTML={{ __html: displayedHtml }}
        />
        {guide.compactHtml && (
          <button className="instruction-toggle" type="button" onClick={() => setShowFullInstruction(value => !value)}>
            {showFullInstruction ? 'Hide encoded config' : 'Show full HLO instruction'}
          </button>
        )}
        {node.op === 'copy-start' && diagram && <div dangerouslySetInnerHTML={{ __html: diagram }} />}
        <div className="guide-heading">Explanation</div>
        <div className="guide-list">
          {group && <TypeTree group={group} parts={guide.parts} activePart={activePart} activeSlot={activeSlot} onActivate={activate} />}
          {explanations.map(part => (
            <div
              key={part.key}
              className={`guide-row part-${part.key}${activePart === part.key ? ' active-part' : ''}`}
              data-part={part.key}
              tabIndex={0}
              style={partColor(part.key) ? ({ '--part': partColor(part.key) } as React.CSSProperties) : undefined}
              onMouseEnter={() => activate(part.key)}
              onMouseLeave={() => activate(null)}
              onFocus={() => activate(part.key)}
              onBlur={() => activate(null)}
            >
              <span className="guide-swatch" />
              <div>
                <strong>{part.label}</strong>
                <p>{part.text}</p>
              </div>
            </div>
          ))}
        </div>
        {node.op !== 'copy-start' && diagram && <div dangerouslySetInnerHTML={{ __html: diagram }} />}
        <div className="guide-sources">
          References:{' '}
          <a href="https://jax-ml.github.io/scaling-book/profiling/#how-to-read-an-xla-op" target="_blank" rel="noreferrer">
            How to read an XLA op
          </a>{' '}
          ·{' '}
          <a href="https://openxla.org/xla/shapes" target="_blank" rel="noreferrer">
            Shapes and layout
          </a>{' '}
          ·{' '}
          <a href={semanticsUrl} target="_blank" rel="noreferrer">
            {semanticsLabel}
          </a>
        </div>
      </div>
      {metadata && (
        <div
          className={`inspector-section metadata-section part-metadata${activePart === 'metadata' ? ' active-part' : ''}`}
          data-part="metadata"
          tabIndex={0}
          onMouseEnter={() => activate('metadata')}
          onMouseLeave={() => activate(null)}
          onFocus={() => activate('metadata')}
          onBlur={() => activate(null)}
        >
          <h4>HLO metadata</h4>
          <p className="metadata-intro">Where this instruction came from in the source program.</p>
          <div className="metadata-fields">
            {metadata.fields.map(field => (
              <div className="metadata-field" key={field.name}>
                <span>{field.name}</span>
                <code>{field.value}</code>
              </div>
            ))}
            {stack.length > 0 && (
              <div className="metadata-field">
                <span>source (stack_frame_id={frameId}, innermost first)</span>
                <ol className="source-stack">
                  {stack.map((frame, index) => (
                    <li key={index}>
                      <code>{frame.func}</code>{' '}
                      <small>
                        {frame.file}:{frame.line}:{frame.column}
                      </small>
                    </li>
                  ))}
                </ol>
              </div>
            )}
          </div>
          <details className="metadata-original">
            <summary>Show raw metadata</summary>
            <pre>{metadata.raw}</pre>
          </details>
        </div>
      )}
      <div className="inspector-section">
        <h4>
          Direct inputs <span>{node.operands.length}</span>
        </h4>
        <ReferenceList names={node.operands} onNode={onNode} />
      </div>
      <div className="inspector-section">
        <h4>
          Direct consumers <span>{node.users.length}</span>
        </h4>
        <ReferenceList names={node.users} onNode={onNode} />
      </div>
      {!!node.controlPredecessors.length && (
        <div className="inspector-section">
          <h4>
            Control predecessors <span>{node.controlPredecessors.length}</span>
          </h4>
          <ReferenceList names={node.controlPredecessors} onNode={onNode} />
        </div>
      )}
      {!!node.controlSuccessors.length && (
        <div className="inspector-section">
          <h4>
            Control successors <span>{node.controlSuccessors.length}</span>
          </h4>
          <ReferenceList names={node.controlSuccessors} onNode={onNode} />
        </div>
      )}
      {!!Object.keys(node.calls).length && (
        <div className="inspector-section">
          <h4>Open computation</h4>
          {Object.entries(node.calls).map(([role, name]) => (
            <button type="button" className="jump-link" key={role} onClick={() => onComputation(name)}>
              <span>
                <small>{role.toUpperCase()}</small>
                <strong>%{name}</strong>
              </span>
              <span>↗</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
