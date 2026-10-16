import { useMemo, useRef, useState } from 'react';
import { parseBackendConfig, type BackendConfig, type BackendField } from '../lib/backend-config';
import type { HloNode } from '../lib/types';
import './BackendConfigDetails.css';

export function BackendConfigDetails({ node }: { node: HloNode }) {
  const config = useMemo(() => parseBackendConfig(node.raw), [node.raw]);
  return config ? <CompilerRecords key={node.id + node.raw} config={config} /> : null;
}

function CompilerRecords({ config }: { config: BackendConfig }) {
  const [selected, setSelected] = useState<BackendField | null>(null);
  const rawRef = useRef<HTMLDetailsElement>(null);
  const markRef = useRef<HTMLElement>(null);
  const reveal = (field: BackendField) => {
    setSelected(field);
    if (rawRef.current) rawRef.current.open = true;
    requestAnimationFrame(() => markRef.current?.scrollIntoView({ block: 'nearest' }));
  };
  const range = selected?.range;

  return (
    <details className="backend-records">
      <summary>
        Compiler records <span>backend_config</span>
      </summary>
      <div className="backend-records-body">
        {config.error ? (
          <p role="status">{config.error}</p>
        ) : config.fields.length === 0 ? (
          <p>No supported teaching fields are present. Other backend fields remain available in the JSON.</p>
        ) : (
          <dl className="backend-fields">
            {config.fields.map(field => (
              <div key={field.path}>
                <dt>
                  <button type="button" onClick={() => reveal(field)} title={field.path}>
                    {field.label}
                  </button>
                </dt>
                <dd>
                  <code>{field.value}</code>
                  <p>{field.explanation}</p>
                </dd>
              </div>
            ))}
          </dl>
        )}
        <details ref={rawRef} className="backend-raw">
          <summary>{config.decoded ? 'Decoded backend JSON' : 'Raw backend JSON'}</summary>
          {selected && <p className="backend-selected">{selected.path}</p>}
          <pre tabIndex={0} aria-label="Backend configuration JSON">
            <code>
              {range ? (
                <>
                  {config.raw.slice(0, range.start)}
                  <mark ref={markRef}>{config.raw.slice(range.start, range.end)}</mark>
                  {config.raw.slice(range.end)}
                </>
              ) : (
                config.raw
              )}
            </code>
          </pre>
        </details>
        <p className="backend-source">
          Background:{' '}
          <a href="https://gh.evko.io/crucible-notes/libtpu/cost/tpu-hlo-cost-analysis.html" target="_blank" rel="noreferrer">
            libtpu cost analysis notes
          </a>
        </p>
      </div>
    </details>
  );
}
