import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { Drawer } from 'vaul';
import { instructionGuide, layoutDiagram, partColor } from '../lib/instruction-guide';
import { extractHloMetadata, type HloMetadata } from '../lib/metadata';
import { nodeCategory, nodeSummary, sourceStack } from '../lib/parser';
import type { CopyGroup } from '../lib/copy-grouping';
import type { Computation, GuidePart, HloModule, HloNode, InstructionGuide } from '../lib/types';
import { useMediaQuery } from '../hooks/useMediaQuery';
import { useStoredState } from '../hooks/useStoredState';
import { Icon } from './Icon';
import { partClass, TypeTree } from './TypeTree';

interface InspectorProps {
  copyGroup?: CopyGroup | null;
  module: HloModule;
  computation: Computation | null;
  node: HloNode | null;
  upstreamCount: number;
  downstreamCount: number;
  lloHref: string | null;
  onClose: () => void;
  onNode: (name: string) => void;
  onComputation: (name: string) => void;
}

type DetailsProps = Omit<InspectorProps, 'onClose' | 'onNode' | 'onComputation'> & { node: HloNode; computation: Computation };

type Activate = (part: string | null, slot?: string | null) => void;

// Resizable width

const MIN_WIDTH = 300;
const MIN_GRAPH_WIDTH = 360;

function maxInspectorWidth() {
  const sidebarVisible = window.matchMedia('(min-width: 1101px)').matches;
  const sidebar = sidebarVisible ? parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--sidebar-w')) || 0 : 0;
  return Math.max(MIN_WIDTH, window.innerWidth - sidebar - MIN_GRAPH_WIDTH);
}

const clampWidth = (width: number) => Math.round(Math.min(maxInspectorWidth(), Math.max(MIN_WIDTH, width)));

// The docked width lives in --inspector-w (unset = responsive default). Dragging writes it directly
// so the graph does not re-render on every pointer move.
function applyInspectorWidth(width: number | null) {
  const style = document.documentElement.style;
  if (width === null) style.removeProperty('--inspector-w');
  else style.setProperty('--inspector-w', `${width}px`);
}

const setResizing = (resizing: boolean) => document.documentElement.classList.toggle('resizing-inspector', resizing);

const readWidth = (saved: string | null) => (Number(saved) > 0 ? Number(saved) : null);
const writeWidth = (width: number | null) => (width === null ? null : String(width));

function useInspectorWidth() {
  const [width, setWidth] = useStoredState('inspector-width', readWidth, writeWidth);

  useEffect(() => applyInspectorWidth(width), [width]);

  return [width, setWidth] as const;
}

function ResizeHandle({ panelRef }: { panelRef: React.RefObject<HTMLElement | null> }) {
  const [width, setWidth] = useInspectorWidth();
  const [measured, setMeasured] = useState(MIN_WIDTH);
  const dragRef = useRef<{ pointerId: number; x: number; width: number; next: number } | null>(null);

  useEffect(() => {
    const panel = panelRef.current;
    if (!panel) return;
    const observer = new ResizeObserver(() => setMeasured(panel.getBoundingClientRect().width));
    observer.observe(panel);
    return () => observer.disconnect();
  }, [panelRef]);

  const panelWidth = () => panelRef.current?.getBoundingClientRect().width ?? MIN_WIDTH;

  const onPointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;
    event.preventDefault();
    const start = panelWidth();
    dragRef.current = { pointerId: event.pointerId, x: event.clientX, width: start, next: start };
    event.currentTarget.setPointerCapture(event.pointerId);
    setResizing(true);
  };

  const onPointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    drag.next = clampWidth(drag.width + drag.x - event.clientX);
    applyInspectorWidth(drag.next);
  };

  const onPointerUp = (event: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    dragRef.current = null;
    setResizing(false);
    if (drag.next !== drag.width) setWidth(drag.next);
  };

  const onPointerCancel = () => {
    dragRef.current = null;
    setResizing(false);
    applyInspectorWidth(width);
  };

  const onKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    const step = event.shiftKey ? 80 : 20;
    if (event.key === 'ArrowLeft') setWidth(clampWidth(panelWidth() + step));
    else if (event.key === 'ArrowRight') setWidth(clampWidth(panelWidth() - step));
    else if (event.key === 'Home' || event.key === 'Enter') setWidth(null);
    else return;
    event.preventDefault();
  };

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
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerCancel}
      onDoubleClick={() => setWidth(null)}
      onKeyDown={onKeyDown}
    />
  );
}

// Inspector shell: docked panel, or a bottom drawer on phones

const sameDetails = (a: DetailsProps, b: DetailsProps | null) =>
  !!b && (Object.keys(a) as (keyof DetailsProps)[]).every(key => a[key] === b[key]);

export function Inspector({
  copyGroup,
  module,
  computation,
  node,
  upstreamCount,
  downstreamCount,
  lloHref,
  onClose,
  onNode,
  onComputation
}: InspectorProps) {
  const isPhone = useMediaQuery('(max-width: 700px)');
  const panelRef = useRef<HTMLElement>(null);

  const shown = node && computation ? { copyGroup, module, computation, node, upstreamCount, downstreamCount, lloHref } : null;

  // The drawer keeps the last node rendered while it slides away.
  const [lastShown, setLastShown] = useState(shown);
  if (shown && !sameDetails(shown, lastShown)) setLastShown(shown);

  const renderDetails = (props: DetailsProps | null) =>
    props && <InstructionDetails key={props.node.id} {...props} onNode={onNode} onComputation={onComputation} />;
  const details = renderDetails(shown);

  if (isPhone) {
    return (
      <InspectorDrawer open={!!details} onClose={onClose}>
        {details || renderDetails(lastShown)}
      </InspectorDrawer>
    );
  }

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
      <div className="inspector-content overscroll-contain">{details || <EmptyInspector />}</div>
    </aside>
  );
}

function InspectorDrawer({ open, onClose, children }: { open: boolean; onClose: () => void; children: ReactNode }) {
  return (
    <Drawer.Root
      open={open}
      onOpenChange={isOpen => {
        if (!isOpen) onClose();
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
          <div className="inspector-content overscroll-contain">{children}</div>
        </Drawer.Content>
      </Drawer.Portal>
    </Drawer.Root>
  );
}

function EmptyInspector() {
  return (
    <div className="empty-inspector">
      <div className="empty-icon">
        <Icon name="graph" size={28} />
      </div>
      <strong>Explore the graph</strong>
      <p>Select any node to see its inputs, consumers, raw HLO, and linked computations.</p>
    </div>
  );
}

// Instruction details

// Parts explained by the result-type tree rather than as standalone guide rows.
const TYPE_PART_KEYS = ['tuple', 'shape', 'order', 'tile', 'space', 'dest', 'source', 'context'];

const SEMANTICS_URL = 'https://openxla.org/xla/operation_semantics';

function codeTarget(target: EventTarget) {
  const part = target instanceof Element ? target.closest<HTMLElement>('[data-part]') : null;
  return { key: part?.dataset.part || null, slot: part?.closest<HTMLElement>('.hlo-slot')?.dataset.part || null };
}

// Hover or focus highlights the matching span in the HLO code.
const highlightProps = (key: string, activate: Activate) => ({
  tabIndex: 0,
  onMouseEnter: () => activate(key),
  onMouseLeave: () => activate(null),
  onFocus: () => activate(key),
  onBlur: () => activate(null)
});

// Finds the explanation for a clicked code part, most specific first.
function findExplanation(container: HTMLElement, key: string, slot: string | null, typeKeys: Set<string>) {
  const slotItem = slot ? container.querySelector<HTMLDetailsElement>(`.type-child[data-part="${slot}"]`) : null;
  const typeTree = container.querySelector<HTMLDetailsElement>('.type-tree');
  const item =
    (slotItem || typeTree)?.querySelector<HTMLElement>(`.type-detail[data-part="${key}"]`) ||
    slotItem ||
    container.querySelector<HTMLElement>(`.type-child[data-part="${key}"]`) ||
    (typeKeys.has(key) ? typeTree : null) ||
    container.querySelector<HTMLElement>(`.guide-row[data-part="${key}"]`) ||
    container.querySelector<HTMLElement>(`.metadata-section[data-part="${key}"]`);
  return { item, slotItem };
}

function InstructionDetails({
  copyGroup,
  module,
  computation,
  node,
  upstreamCount,
  downstreamCount,
  lloHref,
  onNode,
  onComputation
}: DetailsProps & Pick<InspectorProps, 'onNode' | 'onComputation'>) {
  const [activePart, setActivePart] = useState<string | null>(null);
  const [activeSlot, setActiveSlot] = useState<string | null>(null);
  const detailsRef = useRef<HTMLDivElement>(null);

  const guide = useMemo(() => instructionGuide(node, { computation, module }), [node, computation, module]);
  const metadata = extractHloMetadata(node.raw);
  const slotKeys = guide.resultGroup?.kind === 'tuple' ? guide.resultGroup.slots.map(slot => slot.key) : [];
  const typeKeys = new Set([...TYPE_PART_KEYS, ...slotKeys]);
  const explanations = guide.parts.filter(part => !typeKeys.has(part.key) && !(metadata && part.key === 'metadata'));

  const activate: Activate = (part, slot = null) => {
    setActivePart(part);
    setActiveSlot(slot);
  };

  const revealExplanation = (key: string, slot: string | null) => {
    const container = detailsRef.current;
    if (!container) return;
    const { item, slotItem } = findExplanation(container, key, slot, typeKeys);
    if (!item) return;

    const tree = item.closest<HTMLDetailsElement>('.type-tree');
    if (tree) tree.open = true;
    if (slotItem) slotItem.open = true;
    if (item instanceof HTMLDetailsElement) item.open = true;
    item.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  };

  const onCodeClick = (event: React.MouseEvent<HTMLPreElement>) => {
    const { key, slot } = codeTarget(event.target);
    if (!key) return;
    activate(key, slot);
    revealExplanation(key, slot);
  };

  return (
    <div ref={detailsRef}>
      {copyGroup && <CopyGroupSummary copyGroup={copyGroup} onNode={onNode} />}
      <InstructionHeader node={node} upstreamCount={upstreamCount} downstreamCount={downstreamCount} />
      <InstructionSection
        node={node}
        guide={guide}
        explanations={explanations}
        activePart={activePart}
        activeSlot={activeSlot}
        activate={activate}
        onCodeClick={onCodeClick}
      />
      {metadata && <MetadataSection metadata={metadata} module={module} active={activePart === 'metadata'} activate={activate} />}
      <ReferenceSection title="Direct inputs" names={node.operands} onNode={onNode} />
      <ReferenceSection title="Direct consumers" names={node.users} onNode={onNode} />
      {!!node.controlPredecessors.length && (
        <ReferenceSection title="Control predecessors" names={node.controlPredecessors} onNode={onNode} />
      )}
      {!!node.controlSuccessors.length && <ReferenceSection title="Control successors" names={node.controlSuccessors} onNode={onNode} />}
      {!!Object.keys(node.calls).length && <CalledComputations calls={node.calls} onComputation={onComputation} />}
      {lloHref && (
        <div className="inspector-section">
          <h4>TPU backend program</h4>
          <a className="jump-link" href={lloHref}>
            <span>
              <small>LLO FINAL BUNDLES</small>
              <strong>%{node.name}</strong>
            </span>
            <span>→</span>
          </a>
        </div>
      )}
    </div>
  );
}

function CopyGroupSummary({ copyGroup, onNode }: { copyGroup: CopyGroup; onNode: (name: string) => void }) {
  return (
    <section className="inspector-section copy-group-summary">
      <h4>Grouped copy</h4>
      <strong>{copyGroup.direction}</strong>
      <code>{copyGroup.shape}</code>
      <p>Two instructions displayed as one copy.</p>
      <ReferenceList names={[copyGroup.start.name, copyGroup.done.name]} onNode={onNode} />
    </section>
  );
}

function InstructionHeader({ node, upstreamCount, downstreamCount }: { node: HloNode; upstreamCount: number; downstreamCount: number }) {
  const displayType = node.type.startsWith('(') ? nodeSummary(node) : node.type;
  const metrics = [
    { label: 'Upstream', value: upstreamCount },
    { label: 'Downstream', value: downstreamCount },
    { label: 'Source line', value: node.line }
  ];

  return (
    <>
      <div className="inspector-title">
        <span className={`type-badge ${nodeCategory(node)}`}>{node.op}</span>
        {node.root && <span className="root-pill">ROOT</span>}
        <h3>%{node.name}</h3>
        <div className="muted">{displayType}</div>
      </div>
      <div className="metric-row">
        {metrics.map(metric => (
          <div key={metric.label}>
            <strong>{metric.value}</strong>
            <span>{metric.label}</span>
          </div>
        ))}
      </div>
    </>
  );
}

function InstructionSection({
  node,
  guide,
  explanations,
  activePart,
  activeSlot,
  activate,
  onCodeClick
}: {
  node: HloNode;
  guide: InstructionGuide;
  explanations: GuidePart[];
  activePart: string | null;
  activeSlot: string | null;
  activate: Activate;
  onCodeClick: (event: React.MouseEvent<HTMLPreElement>) => void;
}) {
  const [showFullInstruction, setShowFullInstruction] = useState(false);
  const html = showFullInstruction ? guide.html : guide.compactHtml || guide.html;

  // copy-start's diagram explains the instruction itself, so it comes before the explanation.
  const diagramHtml = layoutDiagram(node);
  const diagram = diagramHtml && <div dangerouslySetInnerHTML={{ __html: diagramHtml }} />;
  const diagramFirst = node.op === 'copy-start';

  return (
    <div className="inspector-section instruction-section">
      <h4>HLO instruction</h4>
      <HloCode html={html} activePart={activePart} activeSlot={activeSlot} activate={activate} onClick={onCodeClick} />
      {guide.compactHtml && (
        <button className="instruction-toggle" type="button" onClick={() => setShowFullInstruction(value => !value)}>
          {showFullInstruction ? 'Hide encoded config' : 'Show full HLO instruction'}
        </button>
      )}
      {diagramFirst && diagram}

      <div className="guide-heading">Explanation</div>
      <div className="guide-list">
        {guide.resultGroup && (
          <TypeTree group={guide.resultGroup} parts={guide.parts} activePart={activePart} activeSlot={activeSlot} onActivate={activate} />
        )}
        {explanations.map(part => (
          <GuideRow key={part.key} part={part} active={activePart === part.key} activate={activate} />
        ))}
      </div>
      {!diagramFirst && diagram}

      <GuideReferences node={node} />
    </div>
  );
}

function HloCode({
  html,
  activePart,
  activeSlot,
  activate,
  onClick
}: {
  html: string;
  activePart: string | null;
  activeSlot: string | null;
  activate: Activate;
  onClick: (event: React.MouseEvent<HTMLPreElement>) => void;
}) {
  const codeRef = useRef<HTMLPreElement>(null);

  useEffect(() => {
    const elements = [...(codeRef.current?.querySelectorAll<HTMLElement>('[data-part]') || [])];
    const slotOf = (element: HTMLElement) => element.closest<HTMLElement>('.hlo-slot')?.dataset.part;
    const hasMatchingPart = elements.some(
      element => element.dataset.part === activePart && (!activeSlot || slotOf(element) === activeSlot)
    );

    elements.forEach(element => {
      const key = element.dataset.part;
      const matchesPart = key === activePart && (!activeSlot || slotOf(element) === activeSlot || key === activeSlot);
      const matchesSlot = element.classList.contains('hlo-slot') && key === activeSlot;
      // A memory space with no span of its own highlights the whole type.
      const fallbackType = !hasMatchingPart && !activeSlot && activePart === 'space' && element.classList.contains('hlo-type');
      element.classList.toggle('active-part', matchesPart || matchesSlot || fallbackType);
    });
    // oxlint-disable-next-line react/exhaustive-effect-dependencies -- new html replaces the spans, so re-apply highlights
  }, [activePart, activeSlot, html]);

  return (
    <pre
      ref={codeRef}
      className="hlo-code"
      onClick={onClick}
      onMouseOver={event => {
        const { key, slot } = codeTarget(event.target);
        activate(key, slot);
      }}
      onMouseLeave={() => activate(null)}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

export function GuideRow({
  part,
  active,
  activate
}: {
  part: Pick<GuidePart, 'key' | 'label' | 'text'>;
  active: boolean;
  activate: Activate;
}) {
  const color = partColor(part.key);
  const style = color ? ({ '--part': color } as React.CSSProperties) : undefined;

  return (
    <div className={partClass('guide-row', part.key, active)} data-part={part.key} style={style} {...highlightProps(part.key, activate)}>
      <span className="guide-swatch" />
      <div>
        <strong>{part.label}</strong>
        <p>{part.text}</p>
      </div>
    </div>
  );
}

function GuideReferences({ node }: { node: HloNode }) {
  const semantics =
    node.op === 'copy-start'
      ? { href: `${SEMANTICS_URL}#copy`, label: 'Copy' }
      : node.type.startsWith('(')
        ? { href: `${SEMANTICS_URL}#tuple`, label: 'Tuple' }
        : { href: SEMANTICS_URL, label: 'Operations' };

  return (
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
      <a href={semantics.href} target="_blank" rel="noreferrer">
        {semantics.label}
      </a>
    </div>
  );
}

function MetadataSection({
  metadata,
  module,
  active,
  activate
}: {
  metadata: HloMetadata;
  module: HloModule;
  active: boolean;
  activate: Activate;
}) {
  const frameId = Number(metadata.fields.find(field => field.name === 'stack_frame_id')?.value || 0);
  const stack = frameId ? sourceStack(module, frameId) : [];

  return (
    <div
      className={partClass('inspector-section metadata-section', 'metadata', active)}
      data-part="metadata"
      {...highlightProps('metadata', activate)}
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
  );
}

function ReferenceList({ names, onNode }: { names: string[]; onNode: (name: string) => void }) {
  if (!names.length) return <p className="none">None in this computation</p>;

  return names.map((name, index) => (
    <button type="button" className="reference" key={`${index}-${name}`} onClick={() => onNode(name)}>
      <span>%{name}</span>
      <span>↗</span>
    </button>
  ));
}

function ReferenceSection({ title, names, onNode }: { title: string; names: string[]; onNode: (name: string) => void }) {
  return (
    <div className="inspector-section">
      <h4>
        {title} <span>{names.length}</span>
      </h4>
      <ReferenceList names={names} onNode={onNode} />
    </div>
  );
}

function CalledComputations({ calls, onComputation }: { calls: HloNode['calls']; onComputation: (name: string) => void }) {
  return (
    <div className="inspector-section">
      <h4>Open computation</h4>
      {Object.entries(calls).map(([role, name]) => (
        <button type="button" className="jump-link" key={role} onClick={() => onComputation(name)}>
          <span>
            <small>{role.toUpperCase()}</small>
            <strong>%{name}</strong>
          </span>
          <span>↗</span>
        </button>
      ))}
    </div>
  );
}
