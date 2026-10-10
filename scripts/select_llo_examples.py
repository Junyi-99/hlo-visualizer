"""Pick one representative final-bundle program per JAX example.

Input is the OUT directory produced by dump_hlo.py with LLO_DUMP_DIR set.
The full set of LLO programs remains in OUT/llo; this script makes small,
browser-friendly examples from the largest non-copy backend program.

Usage: python scripts/select_llo_examples.py OUT examples/llo-v6e-1
"""

import json
import re
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
target.mkdir(parents=True, exist_ok=True)
manifest = json.loads((source / "manifest.json").read_text())
selected = {}

for program in manifest["passed"]:
    files = list((source / "llo" / program).glob("*-final_bundles.txt"))
    if not files:
        raise RuntimeError(f"{program}: no final-bundle files")
    candidates = [path for path in files if not re.search(r"-(?:TLP|copy[^-]*|<late-[^>]*>)-", path.name, re.I)]
    chosen = max(candidates or files, key=lambda path: path.stat().st_size)
    content = chosen.read_text()
    header = f"// {program} · {manifest['topology']} · {chosen.name}\n"
    (target / f"{program}.llo").write_text(header + content)
    selected[program] = {
        "source": chosen.name,
        "bytes": len(content.encode()),
        "bundles": len(re.findall(r"^\s*(?:0x[0-9a-f]+|\d+)\s*:\s*\{", content, re.I | re.M)),
    }

(target / "manifest.json").write_text(json.dumps({"jax": manifest["jax"], "topology": manifest["topology"],
                                                 "selection": "largest non-copy final-bundle program per example",
                                                 "examples": selected}, indent=2) + "\n")
print(f"selected {len(selected)} programs for {manifest['topology']} in {target}")
