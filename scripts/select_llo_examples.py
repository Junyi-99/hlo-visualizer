"""Publish verified final-bundle LLO programs for the LLO view.

Input is the OUT directory produced by dump_hlo.py with LLO_DUMP_DIR set
(OUT/manifest.json, OUT/<program>.after.hlo, OUT/llo/<program>/*-final_bundles.txt).

Each final-bundle file names its HLO instruction in the entry-bundle comment,
e.g. "entry bundle: %fusion.5 = fusion(%copy-done.2, %fusion.10, %fusion.4)".
A program is kept only when that name, opcode and operand list occur in the
run's own compiled HLO; the dump directory also collects programs of other
modules compiled by the same example (warm-ups, helpers), and TLP or
<late-*> programs that have no HLO instruction.

Programs are written to public/llo/<topology>/<program>/<instruction>.llo and
listed in src/llo/llo-index.json, which also records whether the instruction
exists in the HLO example shipped under examples/<topology> (for cross links).

Usage: python scripts/select_llo_examples.py OUT v6e-1 [--hlo examples/v6e-1]
"""

import argparse
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENTRY = re.compile(r"entry bundle: %([\w.-]+) = ([\w-]+)\(([^)]*)\)")
BUNDLE = re.compile(r"^\s*(?:0x[0-9a-f]+|\d+)\s*:\s*\{", re.IGNORECASE | re.MULTILINE)


def operands(text):
    return [item.strip() for item in text.split(",") if item.strip()]


def hlo_has(hlo, name, opcode, args):
    pattern = re.compile(rf"%{re.escape(name)} = .*?\b{re.escape(opcode)}\(([^)]*)\)")
    match = pattern.search(hlo)
    return bool(match) and operands(match.group(1)) == args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("topology")
    parser.add_argument(
        "--hlo", type=Path, help="HLO examples dir, default examples/<topology>"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "public/llo")
    parser.add_argument("--index", type=Path, default=ROOT / "src/llo/llo-index.json")
    args = parser.parse_args()
    hlo_dir = args.hlo or ROOT / "examples" / args.topology

    manifest = json.loads((args.source / "manifest.json").read_text())
    target = args.out / args.topology
    if target.exists():
        shutil.rmtree(target)
    entries = []
    skipped = {"no entry comment": 0, "other module": 0, "duplicate": 0}

    for program in manifest["passed"]:
        own_hlo = (args.source / f"{program}.after.hlo").read_text()
        shipped = hlo_dir / f"{program}.after.hlo"
        shipped_hlo = shipped.read_text() if shipped.exists() else ""
        kept = {}
        for path in sorted((args.source / "llo" / program).glob("*-final_bundles.txt")):
            content = path.read_text()
            match = ENTRY.search(content)
            if not match:
                skipped["no entry comment"] += 1
                continue
            name, opcode, arg_text = match.groups()
            if not hlo_has(own_hlo, name, opcode, operands(arg_text)):
                skipped["other module"] += 1
                continue
            if name in kept:
                skipped["duplicate"] += 1
                if kept[name][1] != content:
                    print(
                        f"warning: {program}/{name}: differing duplicates, keeping {kept[name][0].name}"
                    )
                continue
            kept[name] = (path, content)
            entries.append(
                {
                    "id": f"{args.topology}/{program}/{name}",
                    "topology": args.topology,
                    "example": program,
                    "instruction": name,
                    "opcode": opcode,
                    "bundles": len(BUNDLE.findall(content)),
                    "bytes": len(content.encode()),
                    "default": False,
                    "hloExample": f"{args.topology}/{program}.after"
                    if hlo_has(shipped_hlo, name, opcode, operands(arg_text))
                    else None,
                }
            )
            (target / program).mkdir(parents=True, exist_ok=True)
            (target / program / f"{name}.llo").write_text(
                f"// {program} · {manifest['topology']} · {path.name}\n{content}"
            )
        if not kept:
            raise RuntimeError(f"{program}: no verified final-bundle programs")
        largest = max(
            (entry for entry in entries if entry["example"] == program),
            key=lambda entry: entry["bytes"],
        )
        largest["default"] = True

    index = json.loads(args.index.read_text()) if args.index.exists() else []
    index = [entry for entry in index if entry["topology"] != args.topology] + entries
    index.sort(
        key=lambda entry: (entry["topology"], entry["example"], entry["instruction"])
    )
    args.index.write_text(json.dumps(index, indent=2) + "\n")
    total = sum(entry["bytes"] for entry in entries)
    print(
        f"{args.topology}: kept {len(entries)} programs ({total / 1e6:.1f} MB), skipped {skipped}"
    )


if __name__ == "__main__":
    main()
