"""Extract where XLA actually placed each value, from an --xla_dump_to directory.

Writes examples/v6e-1/<program>.after.memory.json:
  {"<instruction>": {"<shape index>": <memory space> | "thread-local"}}
Only positions that own or alias a buffer appear; instructions absent from the file have no buffer
(e.g. values inside fusion bodies). This is the ground truth for the app's memory labels.

Usage: python scripts/dump_hlo.py OUT                      (on the TPU VM; also writes OUT/xla_dump)
       python scripts/extract_memory_truth.py OUT/xla_dump examples/v6e-1
"""

import glob
import json
import os
import re
import sys
from pathlib import Path

dump, examples = sys.argv[1], sys.argv[2]
for example in sorted(glob.glob(os.path.join(examples, "*.after.hlo"))):
    program = os.path.basename(example).split(".")[0]
    values_files = glob.glob(
        os.path.join(
            dump, f"*.jit_{program}.*after_optimizations-buffer-assignment-values.txt"
        )
    )
    if len(values_files) != 1:
        sys.exit(
            f"{program}: expected one buffer assignment dump, found {len(values_files)}"
        )
    values_file = values_files[0]
    hlo_file = values_file.replace("-buffer-assignment-values.txt", ".txt")
    # XLA's dump writes tuple projections inline (%t#0) and spells out async wrapper computations;
    # compile().as_text() materializes the projections as get-tuple-element and hides the wrappers
    # behind sugar like slice-start. Every other instruction must appear in both.
    names = lambda text: set(
        re.findall(r"^\s*(?:ROOT )?%([\w.-]+) = ", text, re.MULTILINE)
    )
    example_text = Path(example).read_text()
    projections = {
        m.group(1): (m.group(2), m.group(3))
        for m in re.finditer(
            r"^\s*(?:ROOT )?%([\w.-]+) = \S+ get-tuple-element\(%([\w.-]+)\), index=(\d+)",
            example_text,
            re.MULTILINE,
        )
    }
    dumped, shown = names(Path(hlo_file).read_text()), names(example_text)
    if not (shown - dumped) <= set(projections):
        sys.exit(f"{program}: dump and example are different compilations")

    # Values that live in thread-local allocations (scalars inside reducer/comparator computations).
    thread_local = set()
    allocation_local = False
    for line in (
        Path(values_file.replace("-values.txt", ".txt")).read_text().splitlines()
    ):
        if line.startswith("allocation "):
            allocation_local = "thread-local" in line
        elif allocation_local and (match := re.match(r"\s*value: <(\d+) ", line)):
            thread_local.add(match.group(1))

    truth = {}
    value = None
    section = None
    for line in Path(values_file).read_text().splitlines():
        if match := re.match(
            r"<(\d+) .*? @(\d+)>", line
        ):  # e.g. <41 cond.8.clone{0} (phi) @1>
            value = (match.group(1), int(match.group(2)))
        elif line.strip() in ("positions:", "uses:"):
            section = line.strip()
        elif line.startswith(" from instruction"):
            section = None
        elif (
            section == "positions:"
            and value
            and (match := re.match(r"\s+([\w.-]+)(?: \{([\d,]*)\})?\s*$", line))
        ):
            space = "thread-local" if value[0] in thread_local else value[1]
            truth.setdefault(match.group(1), {})[match.group(2) or ""] = space
    # A projection shares the buffer of the tuple element it reads (resolve chains of projections).
    pending = {
        name: source for name, source in projections.items() if name not in truth
    }
    while pending:
        resolved = {
            name
            for name, (tuple_name, index) in pending.items()
            if index in truth.get(tuple_name, {})
        }
        if not resolved:
            break
        for name in resolved:
            tuple_name, index = pending.pop(name)
            truth[name] = {"": truth[tuple_name][index]}
    with open(example.replace(".hlo", ".memory.json"), "w") as out:
        json.dump(truth, out, indent=0, sort_keys=True)
        out.write("\n")
    print(program, len(truth), "instructions with buffers")
