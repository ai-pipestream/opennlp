#!/usr/bin/env bash
# Regenerates opennlp.tools.pii.IanaTlds from the live IANA root-zone TLD list.
# Fails closed if the fetch does not succeed; never invents a stale hand-typed list.
set -euo pipefail
if [[ $# -gt 1 || ($# -eq 1 && $1 != "--check") ]]; then
  echo "Usage: $0 [--check]" >&2
  exit 2
fi
MODE="${1:-write}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/opennlp-core/opennlp-runtime/src/main/java/opennlp/tools/pii/IanaTlds.java"
URL="https://data.iana.org/TLD/tlds-alpha-by-domain.txt"
TMP_SOURCE="$(mktemp)"
TMP_OUT="$(mktemp)"
trap 'rm -f "$TMP_SOURCE" "$TMP_OUT"' EXIT
curl -fsSL "$URL" -o "$TMP_SOURCE"
cp "$OUT" "$TMP_OUT"
python3 - "$TMP_SOURCE" "$TMP_OUT" <<'PY'
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
for tld in tlds:
    if not (1 <= len(tld) <= 63):
        raise SystemExit(f"IANA TLD has invalid length: {tld!r}")
    if tld[0] == "-" or tld[-1] == "-":
        raise SystemExit(f"IANA TLD has an edge hyphen: {tld!r}")
    if any(not ("A" <= c <= "Z" or "0" <= c <= "9" or c == "-") for c in tld):
        raise SystemExit(f"IANA TLD is not uppercase ASCII: {tld!r}")
if tlds != sorted(tlds):
    raise SystemExit("IANA TLD list is not sorted; refusing to bake")
if len(tlds) != len(set(tlds)):
    raise SystemExit("IANA TLD list contains duplicates; refusing to bake")
packed = ",".join(tlds)
if packed.split(",") != tlds:
    raise SystemExit("packed IANA TLD list does not round-trip")
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
text, version_replacements = re.subn(
    r"Snapshot .*?; regenerated",
    f"Snapshot {version_line}; regenerated",
    text,
    count=1,
)
text, table_replacements = re.subn(
    r"private static final String TLDS =\n(?:.*\n)*?.*;",
    "private static final String TLDS =\n" + tlds_const,
    text,
    count=1,
)
if version_replacements != 1 or table_replacements != 1:
    raise SystemExit(
        "expected exactly one snapshot and one TLD table replacement, got "
        f"{version_replacements} and {table_replacements}"
    )
table = re.search(r"private static final String TLDS =\n((?:.*\n)*?.*;)", text)
if table is None:
    raise SystemExit("generated Java source has no TLD table")
reparsed = "".join(re.findall(r'\"([^\"]*)\"', table.group(1))).split(",")
if reparsed != tlds:
    raise SystemExit("generated Java TLD table does not round-trip")
out_path.write_text(text)
print(f"Wrote {len(tlds)} TLDs in {len(chunks)} chunks to {out_path}")
PY
if [[ "$MODE" == "--check" ]]; then
  if ! diff -u "$OUT" "$TMP_OUT"; then
    echo "IANA TLD snapshot is stale; run dev/fetch-iana-tlds.sh" >&2
    exit 1
  fi
  echo "IANA TLD snapshot is current"
else
  chmod --reference="$OUT" "$TMP_OUT"
  mv "$TMP_OUT" "$OUT"
  echo "Regenerated $OUT"
fi
