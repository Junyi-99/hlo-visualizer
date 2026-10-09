import { useEffect, useRef, useState } from 'react';
import { computationLinks } from '../lib/parser';
import { computationRole, type LayoutMode } from '../lib/graph-layout';
import { examples } from '../lib/examples';
import type { HloModule } from '../lib/types';
import { Icon } from './Icon';
import { useStoredState } from '../hooks/useStoredState';

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
}

type Theme = 'system' | 'light' | 'dark';
const themeOrder: Theme[] = ['system', 'light', 'dark'];
const themeLabel = { system: 'Theme: follow system', light: 'Theme: light', dark: 'Theme: dark' };

const readTheme = (saved: string | null): Theme => (saved === 'light' || saved === 'dark' ? saved : 'system');
const writeTheme = (theme: Theme) => (theme === 'system' ? null : theme);

function ThemeButton() {
  const [theme, setTheme] = useStoredState('theme', readTheme, writeTheme);
  useEffect(() => {
    const media = matchMedia('(prefers-color-scheme: dark)');
    const apply = () => {
      document.documentElement.dataset.theme = theme === 'system' ? (media.matches ? 'dark' : 'light') : theme;
    };
    apply();
    if (theme !== 'system') return;
    media.addEventListener('change', apply);
    return () => media.removeEventListener('change', apply);
  }, [theme]);
  const next = themeOrder[(themeOrder.indexOf(theme) + 1) % themeOrder.length];
  return (
    <button
      className="icon-button theme-button"
      type="button"
      title={`${themeLabel[theme]} (click for ${next})`}
      aria-label={`${themeLabel[theme]}. Switch to ${next}`}
      onClick={() => setTheme(next)}
    >
      <Icon name={theme === 'system' ? 'system' : theme === 'light' ? 'sun' : 'moon'} />
    </button>
  );
}

export function Toolbar({
  autoGroup,
  onAutoGroup,
  layoutMode,
  onLayoutMode,
  module,
  current,
  canGoBack,
  zoom,
  useOpName,
  showLastNameOnly,
  showMemoryLocation,
  onUseOpName,
  onShowLastNameOnly,
  onShowMemoryLocation,
  onBack,
  onComputation,
  onFit,
  onArrange,
  onZoom,
  onSearch,
  onImport,
  exampleId,
  onExample
}: ToolbarProps) {
  const [displayOpen, setDisplayOpen] = useState(false);
  const displayRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!displayOpen) return;
    const dismiss = (event: PointerEvent) => {
      if (!displayRef.current?.contains(event.target as Node)) setDisplayOpen(false);
    };
    const onEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setDisplayOpen(false);
    };
    document.addEventListener('pointerdown', dismiss);
    document.addEventListener('keydown', onEscape);
    return () => {
      document.removeEventListener('pointerdown', dismiss);
      document.removeEventListener('keydown', onEscape);
    };
  }, [displayOpen]);
  const computation = current ? module.byName.get(current) : null;
  const links = computationLinks(module);
  const role = computation && computationRole(computation, links);
  return (
    <>
      <header className="topbar">
        <div className="crumb">
          <span>Workspace</span>
          <span className="slash">/</span>
          <strong>{!computation ? 'Overview' : computation.entry ? 'Entry' : computation.name}</strong>
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
            {(['after', 'before'] as const).map(stage => (
              <optgroup
                key={stage}
                label={stage === 'after' ? 'TPU v6e · compiled (after optimizations)' : 'TPU v6e · before optimizations'}
              >
                {examples
                  .filter(example => example.stage === stage)
                  .map(example => (
                    <option key={example.id} value={example.id}>
                      {example.program}
                    </option>
                  ))}
              </optgroup>
            ))}
          </select>
          <button className="search-button" type="button" title="Search nodes (/)" onClick={onSearch}>
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
          <div className="eyebrow">{!computation ? 'MODULE MAP' : computation.entry ? 'ENTRY COMPUTATION' : role}</div>
          <h2>{computation ? `%${computation.name}` : 'Computation overview'}</h2>
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
        </div>
        <div className="heading-controls">
          <div className="view-actions">
            {canGoBack && (
              <button className="subtle-button" type="button" onClick={onBack}>
                <Icon name="back" />
                Back
              </button>
            )}
            <div className="display-menu" ref={displayRef}>
              <button
                className={`subtle-button display-trigger${displayOpen ? ' active' : ''}`}
                type="button"
                aria-expanded={displayOpen}
                aria-controls="display-options"
                onClick={() => setDisplayOpen(value => !value)}
              >
                Display <Icon name="down" size={14} />
              </button>
              {displayOpen && (
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
                  <label
                    className="display-option"
                    title={
                      module.scheduled
                        ? undefined
                        : 'XLA decides whether each value lives in HBM or VMEM while compiling. This is lowered HLO (no is_scheduled=true in the module header), so that information does not exist yet; load compiled HLO instead.'
                    }
                  >
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
              <button className="icon-button" type="button" aria-label="Zoom out" onClick={() => onZoom(-0.15)}>
                <Icon name="minus" />
              </button>
              <span className="zoom-value">{Math.round(zoom * 100)}%</span>
              <button className="icon-button" type="button" aria-label="Zoom in" onClick={() => onZoom(0.15)}>
                <Icon name="plus" />
              </button>
            </div>
          </div>
        </div>
      </section>
    </>
  );
}
