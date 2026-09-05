#!/usr/bin/env bash
# Regenerates PhoneNumberLengths from libphonenumber's live metadata.
# Set PHONE_METADATA_SOURCE, PHONE_METADATA_REVISION, and PHONE_METADATA_DATE
# together to reproduce a downloaded snapshot without network access.
set -euo pipefail

if [[ $# -gt 1 || ($# -eq 1 && $1 != "--check") ]]; then
  echo "Usage: $0 [--check]" >&2
  exit 2
fi

MODE="${1:-write}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/opennlp-core/opennlp-runtime/src/main/java/opennlp/tools/pii/PhoneNumberLengths.java"
TMP_SOURCE="$(mktemp)"
TMP_OUT="$(mktemp)"
trap 'rm -f "$TMP_SOURCE" "$TMP_OUT"' EXIT

if [[ -n "${PHONE_METADATA_SOURCE:-}" ]]; then
  if [[ -z "${PHONE_METADATA_REVISION:-}" || -z "${PHONE_METADATA_DATE:-}" ]]; then
    echo "Offline generation requires PHONE_METADATA_REVISION and PHONE_METADATA_DATE" >&2
    exit 2
  fi
  cp "$PHONE_METADATA_SOURCE" "$TMP_SOURCE"
  REVISION="$PHONE_METADATA_REVISION"
  SNAPSHOT_DATE="$PHONE_METADATA_DATE"
else
  REPOSITORY="https://github.com/google/libphonenumber.git"
  REVISION="$(git ls-remote "$REPOSITORY" refs/heads/master | awk '{print $1}')"
  if [[ ! "$REVISION" =~ ^[0-9a-f]{40}$ ]]; then
    echo "Could not resolve the libphonenumber master revision" >&2
    exit 1
  fi
  URL="https://raw.githubusercontent.com/google/libphonenumber/$REVISION/resources/PhoneNumberMetadata.xml"
  curl -fsSL "$URL" -o "$TMP_SOURCE"
  SNAPSHOT_DATE="$(curl -fsSL "https://api.github.com/repos/google/libphonenumber/commits/$REVISION" \
      | python3 -c 'import json, sys; print(json.load(sys.stdin)["commit"]["committer"]["date"][:10])')"
fi

cp "$OUT" "$TMP_OUT"
python3 - "$TMP_SOURCE" "$TMP_OUT" "$REVISION" "$SNAPSHOT_DATE" <<'PY'
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

source_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
revision, snapshot_date = sys.argv[3], sys.argv[4]
if not re.fullmatch(r"[0-9a-f]{7,40}", revision):
    raise SystemExit("invalid libphonenumber revision")
if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", snapshot_date):
    raise SystemExit("invalid libphonenumber snapshot date")

root = ET.parse(source_path).getroot()
if root.tag != "phoneNumberMetadata":
    raise SystemExit(f"unexpected metadata root: {root.tag!r}")

def expand_lengths(value):
    lengths = set()
    for match in re.finditer(r"(\d+)(?:-(\d+))?", value):
        low = int(match.group(1))
        high = int(match.group(2) or match.group(1))
        if low > high or low < 1 or high > 31:
            raise SystemExit(f"invalid national number length range: {match.group(0)!r}")
        lengths.update(range(low, high + 1))
    return lengths

by_code = {}
territories = root.findall("./territories/territory")
if len(territories) < 200:
    raise SystemExit(f"metadata contained only {len(territories)} territories")
for territory in territories:
    code_text = territory.get("countryCode", "")
    if not code_text.isascii() or not code_text.isdigit():
        raise SystemExit(f"invalid calling code: {code_text!r}")
    code = int(code_text)
    if not 1 <= code <= 999:
        raise SystemExit(f"calling code out of range: {code}")
    lengths = set()
    for possible in territory.findall(".//possibleLengths"):
        lengths.update(expand_lengths(possible.get("national", "")))
    if not lengths:
        raise SystemExit(f"calling code {code} has no national lengths")
    by_code.setdefault(code, set()).update(lengths)
if len(by_code) < 200:
    raise SystemExit(f"metadata contained only {len(by_code)} calling codes")

pairs = [(code, sum(1 << length for length in lengths))
         for code, lengths in sorted(by_code.items())]
tokens = []
for code, mask in pairs:
    tokens.extend((str(code), f"0x{mask:X}"))
lines = []
line = "      "
for token in tokens:
    addition = token + ","
    if len(line) + len(addition) + 1 > 98:
        lines.append(line.rstrip())
        line = "      " + addition + " "
    else:
        line += addition + " "
lines.append(line.rstrip())
table = "\n".join(lines)

text = out_path.read_text()
text, comment_count = re.subn(
    r"revision \{@code [0-9a-f]+\} of \d{4}-\d{2}-\d{2}",
    f"revision {{@code {revision[:12]}}} of {snapshot_date}",
    text,
    count=1,
)
text, table_count = re.subn(
    r"(?<=private static final int\[\] CODE_AND_MASK = \{\n)(?:.*\n)*?(?=  \};)",
    table + "\n",
    text,
    count=1,
)
if comment_count != 1 or table_count != 1:
    raise SystemExit(
        f"expected one comment and table replacement, got {comment_count} and {table_count}"
    )
generated = re.search(r"CODE_AND_MASK = \{(.*?)\};", text, re.DOTALL)
if generated is None:
    raise SystemExit("generated Java source has no table")
round_trip = [int(value, 0) for value in re.findall(r"0x[0-9A-F]+|\d+", generated.group(1))]
expected = [value for pair in pairs for value in pair]
if round_trip != expected:
    raise SystemExit("generated phone table does not round-trip")
out_path.write_text(text)
print(f"Wrote {len(pairs)} calling codes from {revision[:12]} to {out_path}")
PY

if [[ "$MODE" == "--check" ]]; then
  if ! diff -u "$OUT" "$TMP_OUT"; then
    echo "Phone metadata snapshot is stale; run dev/fetch-phone-number-lengths.sh" >&2
    exit 1
  fi
  echo "Phone metadata snapshot is current"
else
  chmod --reference="$OUT" "$TMP_OUT"
  mv "$TMP_OUT" "$OUT"
  echo "Regenerated $OUT"
fi
