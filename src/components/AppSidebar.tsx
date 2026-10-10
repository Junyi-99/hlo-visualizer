import type { ReactNode } from 'react';
import { Icon, type IconName } from './Icon';

// Shared sidebar shell for the HLO and LLO apps: brand, slot for content, cross-link to the other view.
// Width, slide-out animation and collapsed state live in styles.css under .sidebar / .sidebar-collapsed.
export function AppSidebar({
  id,
  collapsed,
  icon,
  title,
  subtitle,
  switchTo,
  children
}: {
  id: string;
  collapsed: boolean;
  icon: IconName;
  title: string;
  subtitle: string;
  switchTo: { href: string; label: string };
  children: ReactNode;
}) {
  return (
    <aside className="sidebar" id={id} inert={collapsed}>
      <div className="brand">
        <div className="brand-mark">
          <Icon name={icon} size={18} />
        </div>
        <div>
          <strong>{title}</strong>
          <span>{subtitle}</span>
        </div>
      </div>
      {children}
      <a className="sidebar-switch" href={switchTo.href}>
        {switchTo.label}
      </a>
    </aside>
  );
}

export function SidebarToggle({ controls, collapsed, onToggle }: { controls: string; collapsed: boolean; onToggle: () => void }) {
  const label = collapsed ? 'Show sidebar' : 'Hide sidebar';
  return (
    <button
      className="icon-button sidebar-toggle"
      type="button"
      title={label}
      aria-label={label}
      aria-controls={controls}
      aria-expanded={!collapsed}
      onClick={onToggle}
    >
      <Icon name="sidebar" />
    </button>
  );
}
