#!/usr/bin/env bash
# Regenerates IbanLengths from the live SWIFT ISO 13616 registry TXT file.
# Set IBAN_REGISTRY_SOURCE to reproduce an already downloaded snapshot.
set -euo pipefail

if [[ $# -gt 1 || ($# -eq 1 && $1 != "--check") ]]; then
  echo "Usage: $0 [--check]" >&2
  exit 2
fi

MODE="${1:-write}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/opennlp-core/opennlp-runtime/src/main/java/opennlp/tools/pii/IbanLengths.java"
URL="https://www.swift.com/swift-resource/11971/download?language=en"
TMP_SOURCE="$(mktemp)"
TMP_OUT="$(mktemp)"
trap 'rm -f "$TMP_SOURCE" "$TMP_OUT"' EXIT

if [[ -n "${IBAN_REGISTRY_SOURCE:-}" ]]; then
  cp "$IBAN_REGISTRY_SOURCE" "$TMP_SOURCE"
else
  curl --connect-timeout 20 --max-time 120 --retry 3 --retry-all-errors -fsSL \
      "$URL" -o "$TMP_SOURCE"
fi

cp "$OUT" "$TMP_OUT"
python3 - "$TMP_SOURCE" "$TMP_OUT" <<'PY'
import csv
import hashlib
import io
import re
import sys
from pathlib import Path

source_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
data = source_path.read_bytes()
if len(data) < 10_000:
    raise SystemExit(f"SWIFT registry is unexpectedly small: {len(data)} bytes")
for encoding in ("utf-8-sig", "utf-16", "cp1252"):
    try:
        decoded = data.decode(encoding)
        break
    except UnicodeDecodeError:
        continue
else:
    raise SystemExit("SWIFT registry has an unsupported text encoding")

rows = {}
for row in csv.reader(io.StringIO(decoded), delimiter="\t"):
    if row:
        rows[row[0].replace("\N{NO-BREAK SPACE}", " ").strip()] = [value.strip() for value in row[1:]]
countries = rows.get("IBAN prefix country code (ISO 3166)")
lengths = rows.get("IBAN length")
if countries is None or lengths is None:
    raise SystemExit("SWIFT registry lacks the country or IBAN length row")
if len(countries) != len(lengths) or len(countries) < 80:
    raise SystemExit(
        f"SWIFT registry has mismatched or too few entries: {len(countries)} and {len(lengths)}"
    )

entries = []
for country, length_text in zip(countries, lengths):
    if not re.fullmatch(r"[A-Z]{2}", country):
        raise SystemExit(f"invalid IBAN country code: {country!r}")
    if not re.fullmatch(r"\d{2}", length_text):
        raise SystemExit(f"invalid IBAN length for {country}: {length_text!r}")
    length = int(length_text)
    if not 15 <= length <= 34:
        raise SystemExit(f"IBAN length out of range for {country}: {length}")
    entries.append((country, length))
if entries != sorted(entries):
    raise SystemExit("SWIFT registry entries are not sorted by country code")
if len(entries) != len({country for country, _ in entries}):
    raise SystemExit("SWIFT registry contains duplicate country codes")

registry = "".join(f"{country}{length:02d}" for country, length in entries)
chunks = [registry[index:index + 76] for index in range(0, len(registry), 76)]
parts = []
for index, chunk in enumerate(chunks):
    prefix = "      \"" if index == 0 else "          + \""
    parts.append(prefix + chunk + "\"")
table = "\n".join(parts) + ";"
digest = hashlib.sha256(registry.encode("ascii")).hexdigest()

text = out_path.read_text()
text, comment_count = re.subn(
    r"Country-length projection SHA-256:\n \* \{@code [0-9a-f]+\}\.",
    f"Country-length projection SHA-256:\n * {{@code {digest}}}.",
    text,
    count=1,
    flags=re.DOTALL,
)
text, table_count = re.subn(
    r"private static final String REGISTRY =\n(?:.*\n)*?.*;",
    "private static final String REGISTRY =\n" + table,
    text,
    count=1,
)
if comment_count != 1 or table_count != 1:
    raise SystemExit(
        f"expected one comment and table replacement, got {comment_count} and {table_count}"
    )
generated = re.search(r"private static final String REGISTRY =\n((?:.*\n)*?.*;)", text)
if generated is None:
    raise SystemExit("generated Java source has no registry table")
round_trip = "".join(re.findall(r'\"([^\"]*)\"', generated.group(1)))
if round_trip != registry:
    raise SystemExit("generated IBAN table does not round-trip")
out_path.write_text(text)
print(f"Wrote {len(entries)} IBAN countries with SHA-256 {digest[:12]} to {out_path}")
PY

if [[ "$MODE" == "--check" ]]; then
  if ! diff -u "$OUT" "$TMP_OUT"; then
    echo "SWIFT IBAN registry snapshot is stale; run dev/fetch-iban-lengths.sh" >&2
    exit 1
  fi
  echo "SWIFT IBAN registry snapshot is current"
else
  chmod --reference="$OUT" "$TMP_OUT"
  mv "$TMP_OUT" "$OUT"
  echo "Regenerated $OUT"
fi
