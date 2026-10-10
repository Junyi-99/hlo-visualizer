import { useCallback, useEffect, useRef, useState } from 'react';
import { ZOOM_STEP, type LayoutMode } from '../lib/graph-layout';
import { examples, hloTopologies } from '../lib/examples';
import type { Computation, HloModule } from '../lib/types';
import { Icon, type IconName } from './Icon';
import { useComputationRoles } from '../hooks/useComputationRoles';
import { useDismiss } from '../hooks/useDismiss';
import { useMediaQuery } from '../hooks/useMediaQuery';
import { useStoredState } from '../hooks/useStoredState';
import { SidebarToggle } from './AppSidebar';

interface ToolbarProps {
  autoGroup: boolean;
  onAutoGroup: (enabled: boolean) => void;
  layoutMode: LayoutMode;
  onLayoutMode: (mode: LayoutMode) => void;
  module: HloModule;
  current: string | null;
  canGoBack: boolean;
  zoom: number;
  useOpName: boolean;
  showLastNameOnly: boolean;
  showMemoryLocation: boolean;
  onUseOpName: (value: boolean) => void;
  onShowLastNameOnly: (value: boolean) => void;
  onShowMemoryLocation: (value: boolean) => void;
  onBack: () => void;
  onComputation: (name: string | null) => void;
  onFit: () => void;
  onArrange: () => void;
  onZoom: (delta: number) => void;
  onSearch: () => void;
  onImport: () => void;
  exampleId: string | null;
  onExample: (id: string) => void;
  sidebarCollapsed: boolean;
  onToggleSidebar: () => void;
  lloHref: string;
  onShowSource: (() => void) | null;
}

const MEMORY_LOCATION_UNAVAILABLE =
  'XLA decides whether each value lives in HBM or VMEM while compiling. This is lowered HLO (no is_scheduled=true in the module header), so that information does not exist yet; load compiled HLO instead.';

type Theme = 'system' | 'light' | 'dark';
const themeOrder: Theme[] = ['system', 'light', 'dark'];
const themeLabel: Record<Theme, string> = { system: 'Theme: follow system', light: 'Theme: light', dark: 'Theme: dark' };
const themeIcon: Record<Theme, IconName> = { system: 'system', light: 'sun', dark: 'moon' };

const readTheme = (saved: string | null): Theme => (saved === 'light' || saved === 'dark' ? saved : 'system');
const writeTheme = (theme: Theme) => (theme === 'system' ? null : theme);

const exampleGroups = hloTopologies.flatMap(topology => [
  { topology, stage: 'after', label: `TPU ${topology} · compiled (after optimizations)` },
  { topology, stage: 'before', label: `TPU ${topology} · before optimizations` }
]);

function ThemeButton() {
  const [theme, setTheme] = useStoredState('theme', readTheme, writeTheme);
  const prefersDark = useMediaQuery('(prefers-color-scheme: dark)');

  useEffect(() => {
    const systemTheme = prefersDark ? 'dark' : 'light';
    document.documentElement.dataset.theme = theme === 'system' ? systemTheme : theme;
  }, [theme, prefersDark]);

  const next = themeOrder[(themeOrder.indexOf(theme) + 1) % themeOrder.length];

  return (
    <button
      className="icon-button theme-button"
      type="button"
      title={`${themeLabel[theme]} (click for ${next})`}
      aria-label={`${themeLabel[theme]}. Switch to ${next}`}
      onClick={() => setTheme(next)}
    >
      <Icon name={themeIcon[theme]} />
    </button>
  );
}

function ExampleSelect({ exampleId, onExample }: Pick<ToolbarProps, 'exampleId' | 'onExample'>) {
  return (
    <select
      className="example-select"
      aria-label="Load an HLO example"
      title="Load an HLO example"
      value={exampleId ?? ''}
      onChange={event => event.target.value && onExample(event.target.value)}
    >
      <option value="" disabled>
        Examples
      </option>
      {exampleGroups.map(({ topology, stage, label }) => (
        <optgroup key={label} label={label}>
          {examples
            .filter(example => example.topology === topology && example.stage === stage)
            .map(example => (
              <option key={example.id} value={example.id}>
                {example.program}
              </option>
            ))}
        </optgroup>
      ))}
    </select>
  );
}

type DisplayMenuProps = Pick<
  ToolbarProps,
  'module' | 'useOpName' | 'showLastNameOnly' | 'showMemoryLocation' | 'onUseOpName' | 'onShowLastNameOnly' | 'onShowMemoryLocation'
>;

function DisplayMenu({
  module,
  useOpName,
  showLastNameOnly,
  showMemoryLocation,
  onUseOpName,
  onShowLastNameOnly,
  onShowMemoryLocation
}: DisplayMenuProps) {
  const [open, setOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const close = useCallback(() => setOpen(false), []);

  useDismiss(menuRef, open, close);

  const triggerClass = `subtle-button display-trigger${open ? ' active' : ''}`;

  return (
    <div className="display-menu" ref={menuRef}>
      <button
        className={triggerClass}
        type="button"
        aria-expanded={open}
        aria-controls="display-options"
        onClick={() => setOpen(value => !value)}
      >
        Display <Icon name="down" size={14} />
      </button>

      {open && (
        <div className="display-popover" id="display-options" role="group" aria-label="Node display options">
          <strong>Node labels</strong>
          <label className="display-option">
            <input type="checkbox" checked={useOpName} onChange={event => onUseOpName(event.target.checked)} />
            Use op_name in metadata
          </label>
          <label className="display-option">
            <input
              type="checkbox"
              checked={showLastNameOnly}
              disabled={!useOpName}
              onChange={event => onShowLastNameOnly(event.target.checked)}
            />
            Show last name only
          </label>
          <label className="display-option" title={module.scheduled ? undefined : MEMORY_LOCATION_UNAVAILABLE}>
            <input
              type="checkbox"
              checked={showMemoryLocation && module.scheduled}
              disabled={!module.scheduled}
              onChange={event => onShowMemoryLocation(event.target.checked)}
            />
            Show memory location
            {!module.scheduled && <small className="display-note">Lowered HLO: XLA picks HBM or VMEM only when compiling</small>}
          </label>
        </div>
      )}
    </div>
  );
}

type ViewActionsProps = DisplayMenuProps &
  Pick<
    ToolbarProps,
    'canGoBack' | 'onBack' | 'autoGroup' | 'onAutoGroup' | 'layoutMode' | 'onLayoutMode' | 'onArrange' | 'onFit' | 'zoom' | 'onZoom'
  >;

function ViewActions({
  canGoBack,
  onBack,
  autoGroup,
  onAutoGroup,
  layoutMode,
  onLayoutMode,
  onArrange,
  onFit,
  zoom,
  onZoom,
  ...displayProps
}: ViewActionsProps) {
  return (
    <div className="view-actions">
      {canGoBack && (
        <button className="subtle-button" type="button" onClick={onBack}>
          <Icon name="back" />
          Back
        </button>
      )}

      <DisplayMenu {...displayProps} />

      <label className="auto-group-toggle" title="Combine matching copy-start and copy-done instructions">
        <input type="checkbox" checked={autoGroup} onChange={event => onAutoGroup(event.target.checked)} />
        Auto Group
      </label>

      <select
        className="layout-select"
        aria-label="Graph layout"
        title="Graph direction: automatic, left to right, or top to bottom"
        value={layoutMode}
        onChange={event => onLayoutMode(event.target.value as LayoutMode)}
      >
        <option value="auto">Auto layout</option>
        <option value="horizontal">Horizontal →</option>
        <option value="vertical">Vertical ↓</option>
      </select>

      <button className="subtle-button" type="button" title="Restore automatic node positions and tidy routes" onClick={onArrange}>
        Arrange
      </button>
      <button className="subtle-button" type="button" onClick={onFit}>
        Fit view
      </button>

      <div className="zoom-controls">
        <button className="icon-button" type="button" aria-label="Zoom out" onClick={() => onZoom(-ZOOM_STEP)}>
          <Icon name="minus" />
        </button>
        <span className="zoom-value">{Math.round(zoom * 100)}%</span>
        <button className="icon-button" type="button" aria-label="Zoom in" onClick={() => onZoom(ZOOM_STEP)}>
          <Icon name="plus" />
        </button>
      </div>
    </div>
  );
}

function crumbLabel(computation: Computation | null | undefined) {
  if (!computation) return 'Overview';
  return computation.entry ? 'Entry' : computation.name;
}

function eyebrowLabel(computation: Computation | null | undefined, role: string | undefined) {
  if (!computation) return 'MODULE MAP';
  return computation.entry ? 'ENTRY COMPUTATION' : role;
}

export function Toolbar({
  module,
  current,
  onComputation,
  onSearch,
  onImport,
  exampleId,
  onExample,
  sidebarCollapsed,
  onToggleSidebar,
  lloHref,
  onShowSource,
  ...viewActionProps
}: ToolbarProps) {
  const roles = useComputationRoles(module);

  const computation = current ? module.byName.get(current) : null;
  const role = computation ? roles.get(computation.name) : undefined;

  return (
    <>
      <header className="topbar">
        <div className="topbar-leading">
          <SidebarToggle controls="hlo-sidebar" collapsed={sidebarCollapsed} onToggle={onToggleSidebar} />
          <div className="crumb">
            <span>Workspace</span>
            <span className="slash">/</span>
            <strong>{crumbLabel(computation)}</strong>
          </div>
        </div>

        <div className="top-actions">
          <a
            className="icon-button repo-link"
            href="https://github.com/Junyi-99/hlo-visualizer"
            target="_blank"
            rel="noreferrer"
            aria-label="Source code on GitHub"
            title="Source code on GitHub"
          >
            <Icon name="github" />
          </a>
          <ThemeButton />
          <ExampleSelect exampleId={exampleId} onExample={onExample} />
          <button className="search-button" type="button" title="Search nodes (/)" aria-label="Search nodes" onClick={onSearch}>
            <Icon name="search" />
            <span className="search-label">Search nodes</span>
            <kbd>/</kbd>
          </button>
          <button className="primary-button" type="button" aria-label="Open HLO" onClick={onImport}>
            <Icon name="plus" />
            <span>Open HLO</span>
          </button>
        </div>
      </header>

      <section className="heading">
        <div>
          <div className="eyebrow">{eyebrowLabel(computation, role)}</div>
          <h2>{computation ? `%${computation.name}` : 'Computation overview'}</h2>
          <div className="mobile-navigation">
            <select
              id="mobile-computations"
              aria-label="Choose computation"
              value={current || ''}
              onChange={event => onComputation(event.target.value || null)}
            >
              <option value="">Overview · all computations</option>
              {module.computations.map(c => (
                <option key={c.name} value={c.name}>
                  %{c.name}
                  {c.entry ? ' · entry' : ''}
                </option>
              ))}
            </select>
            {onShowSource && (
              <button type="button" className="link-button" onClick={onShowSource}>
                Source
              </button>
            )}
            <a href={lloHref}>LLO Visualizer →</a>
          </div>
        </div>

        <div className="heading-controls">
          <ViewActions module={module} {...viewActionProps} />
        </div>
      </section>
    </>
  );
}
