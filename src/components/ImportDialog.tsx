import { useEffect, useRef, useState, type ChangeEvent, type MouseEvent, type SyntheticEvent } from 'react';
import { Icon } from './Icon';

interface ImportDialogProps {
  onClose: () => void;
  onLoad: (source: string) => void;
}

export function ImportDialog({ onClose, onLoad }: ImportDialogProps) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const [source, setSource] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    const dialog = dialogRef.current;
    dialog?.showModal();
    return () => dialog?.close();
  }, []);

  const load = () => {
    try {
      onLoad(source);
      setError('');
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  };

  const readFile = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) setSource(await file.text());
  };

  // Escape fires `cancel`; let the parent unmount the dialog instead of the browser closing it.
  const cancel = (event: SyntheticEvent<HTMLDialogElement>) => {
    event.preventDefault();
    onClose();
  };

  const closeOnBackdrop = (event: MouseEvent<HTMLDialogElement>) => {
    if (event.target === event.currentTarget) onClose();
  };

  return (
    <dialog ref={dialogRef} onCancel={cancel} onClick={closeOnBackdrop}>
      <div className="dialog-head">
        <div>
          <div className="eyebrow">IMPORT MODULE</div>
          <h2>Open HLO text</h2>
        </div>
        <button className="icon-button" type="button" aria-label="Close" onClick={onClose}>
          <Icon name="close" />
        </button>
      </div>

      <p>
        Paste XLA <code>HloModule</code> text or choose a local <code>.hlo</code> / <code>.txt</code> file. Processing stays in your
        browser.
      </p>
      <textarea
        spellCheck={false}
        autoCapitalize="none"
        autoCorrect="off"
        placeholder={'HloModule …\n\nENTRY %main (...) -> ... {'}
        value={source}
        onChange={event => setSource(event.target.value)}
      />

      <div className="dialog-actions">
        <label className="subtle-button file-label">
          Choose file
          <input type="file" accept=".hlo,.txt,.log,text/plain" onChange={readFile} />
        </label>
        <span id="import-error" role="alert">
          {error}
        </span>
        <button className="primary-button" type="button" onClick={load}>
          Visualize module →
        </button>
      </div>
    </dialog>
  );
}
