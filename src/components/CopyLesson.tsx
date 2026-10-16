import { useState } from 'react';
import { copyLessonData } from '../lib/copy-lesson';
import type { Computation, HloModule, HloNode } from '../lib/types';
import './CopyLesson.css';

interface CopyLessonProps {
  module: HloModule;
  computation: Computation;
  node: HloNode;
  onNode: (name: string) => void;
}

export function CopyLesson({ module, computation, node, onNode }: CopyLessonProps) {
  const [step, setStep] = useState(1);
  const lesson = copyLessonData(module, computation, node);
  if (!lesson) return null;
  const stages = [
    {
      label: 'Source',
      text: 'The source value supplies the data. The copy changes where the data is stored, not its mathematical value.',
      nodes: lesson.source ? [lesson.source] : [],
      missing: 'The source instruction is not present in this computation.'
    },
    {
      label: 'Start',
      text: 'copy-start initiates an asynchronous transfer and returns a tuple describing the destination, source and completion context. The destination is not ready to use yet.',
      nodes: lesson.start ? [lesson.start] : [],
      missing: 'No matching copy-start was found in this computation.'
    },
    {
      label: 'Wait',
      text: 'copy-done waits for the matching transfer to finish and returns the destination value. A consumer of this result depends on completion.',
      nodes: lesson.dones,
      missing: 'No matching copy-done was found in this computation.'
    },
    {
      label: 'Use',
      text: 'These instructions directly consume the completed copy. Further uses may appear downstream or in a called computation.',
      nodes: lesson.consumers,
      missing: lesson.dones.some(done => done.root)
        ? 'The completed copy is returned as this computation’s ROOT.'
        : 'No direct consumer of a completed copy is present here.'
    }
  ];
  const stage = stages[step];
  const nodeButtons = (nodes: HloNode[]) => (
    <div className="copy-lesson-nodes">
      {nodes.map(item => (
        <button key={item.name} type="button" onClick={() => onNode(item.name)} title={`Select %${item.name}`}>
          %{item.name}
        </button>
      ))}
    </div>
  );

  return (
    <section className="inspector-section copy-lesson" aria-label="How this asynchronous copy works">
      <h4>Follow the copy</h4>
      {lesson.direction && <strong className="copy-lesson-direction">{lesson.direction}</strong>}
      <p>
        {lesson.direction === 'HBM → VMEM'
          ? 'This transfer brings data into on-chip vector memory. Starting it early can prepare data for later computation.'
          : 'Separating the start from the wait gives the compiler an opportunity to overlap data movement with other work.'}
      </p>
      <div className="copy-lesson-steps" aria-label="Copy stages">
        {stages.map((item, index) => (
          <button key={item.label} type="button" aria-pressed={step === index} onClick={() => setStep(index)}>
            <span>{index + 1}</span>
            {item.label}
          </button>
        ))}
      </div>
      <div className="copy-lesson-stage" aria-live="polite">
        <p>{stage.text}</p>
        {stage.nodes.length ? nodeButtons(stage.nodes) : <p className="copy-lesson-note">{stage.missing}</p>}
      </div>
      {!!lesson.tuple.length && (
        <details className="copy-lesson-tuple">
          <summary>Inside the start tuple</summary>
          <dl>
            {['Destination', 'Source', 'Context'].map(
              (label, index) =>
                lesson.tuple[index] && (
                  <div key={label}>
                    <dt>
                      {index} · {label}
                    </dt>
                    <dd>
                      <code>{lesson.tuple[index].trim()}</code>
                      <span>
                        {
                          [
                            'The buffer being filled; copy-done makes the completed value available.',
                            'The original buffer, kept available while the transfer is in progress.',
                            'Backend state used to track completion; on TPU this is commonly a sync flag.'
                          ][index]
                        }
                      </span>
                    </dd>
                  </div>
                )
            )}
          </dl>
        </details>
      )}
      {!!lesson.unrelated.length && (
        <details className="copy-lesson-independent">
          <summary>{lesson.unrelated.length} instructions have no dependency path to or from the wait</summary>
          <p>
            Considering data and control edges within this computation. They may offer overlap opportunities, but resource limits and
            scheduling still apply.
          </p>
          {nodeButtons(lesson.unrelated.slice(0, 8))}
          {lesson.unrelated.length > 8 && <p className="copy-lesson-note">Showing the first 8 instructions.</p>}
        </details>
      )}
      <p className="copy-lesson-note">
        The highlighted chain shows logical dependencies. Graph positions and instruction order do not measure transfer time or prove
        execution overlap.
      </p>
      <div className="copy-lesson-links">
        <a href="https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html" target="_blank" rel="noreferrer">
          TPU pipelining ↗
        </a>
        <a href="https://gh.evko.io/crucible-notes/libtpu/compiler/msa-overview.html" target="_blank" rel="noreferrer">
          MSA notes ↗
        </a>
      </div>
    </section>
  );
}
