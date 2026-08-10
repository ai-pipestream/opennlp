#!/usr/bin/env bash
# Regenerates opennlp.tools.pii.IanaTlds from the live IANA root-zone TLD list.
# Fails closed if the fetch does not succeed; never invents a stale hand-typed list.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/opennlp-core/opennlp-runtime/src/main/java/opennlp/tools/pii/IanaTlds.java"
URL="https://data.iana.org/TLD/tlds-alpha-by-domain.txt"
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT
curl -fsSL "$URL" -o "$TMP"
python3 - "$TMP" "$OUT" <<'PY'
import re
import sys
from pathlib import Path

src_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
lines = src_path.read_text().splitlines()
if not lines or not lines[0].startswith("# Version"):
    raise SystemExit("IANA TLD list missing Version header")
version_line = lines[0][2:]
tlds = [line.strip() for line in lines[1:] if line.strip() and not line.startswith("#")]
if not tlds:
    raise SystemExit("IANA TLD list contained no entries")
if tlds != sorted(tlds):
    raise SystemExit("IANA TLD list is not sorted; refusing to bake")
packed = ",".join(tlds)
chunks = []
i = 0
limit = 90
while i < len(packed):
    end = min(i + limit, len(packed))
    if end < len(packed):
        cut = packed.rfind(",", i, end + 1)
        if cut <= i:
            cut = packed.find(",", i + 1)
        end = cut + 1
        chunks.append(packed[i:end])
        i = end
    else:
        chunks.append(packed[i:end])
        i = end
parts = []
for n, c in enumerate(chunks):
    parts.append(('      "' if n == 0 else '          + "') + c + '"')
tlds_const = "\n".join(parts) + ";"
text = out_path.read_text()
text = re.sub(
    r"Snapshot .*?; regenerated",
    f"Snapshot {version_line}; regenerated",
    text,
    count=1,
)
text = re.sub(
    r"private static final String TLDS =\n(?:.*\n)*?.*;",
    "private static final String TLDS =\n" + tlds_const,
    text,
    count=1,
)
out_path.write_text(text)
print(f"Wrote {len(tlds)} TLDs in {len(chunks)} chunks to {out_path}")
PY
echo "Regenerated $OUT"
