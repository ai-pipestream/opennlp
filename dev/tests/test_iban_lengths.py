# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Synthetic SWIFT registry inputs for encoding and field validation."""

import codecs
import csv
import hashlib
import io
import re
import unittest

from generator_test_case import GeneratorTestCase


COUNTRY_ROW = "IBAN prefix country code (ISO 3166)"
LENGTH_ROW = "IBAN length"


class IbanLengthsTest(GeneratorTestCase):
    """Exercise the registry command with no external registry file."""

    script_name = "fetch-iban-lengths.sh"
    java_name = "IbanLengths.java"

    def setUp(self):
        """Create 80 synthetic sorted country entries spanning supported lengths."""
        super().setUp()
        self.countries = [chr(65 + index // 26) + chr(65 + index % 26) for index in range(80)]
        self.lengths = [str(15 + index % 20) for index in range(80)]
        self.rows = [[COUNTRY_ROW, *self.countries], [LENGTH_ROW, *self.lengths],
                     ["Description", "Café " + "x" * 11000]]
        self.encoding = "utf-8"
        self.bom = b""
        self.parity = None
        self.suffix = ""
        self.environment["IBAN_REGISTRY_SOURCE"] = str(self.source)

    def write_source(self):
        """Encode a tab-separated registry, optionally with a byte-order mark."""
        text = io.StringIO()
        csv.writer(text, delimiter="\t").writerows(self.rows)
        data = self.bom + (text.getvalue() + self.suffix).encode(self.encoding)
        if self.parity is not None and len(data) % 2 != self.parity:
            data += b" "
        self.source.write_bytes(data)

    def assert_table(self):
        """Check every generated entry and the projection digest."""
        source = self.output.read_text(encoding="utf-8")
        body = source.partition("private static final String REGISTRY =")[2].partition(";")[0]
        generated = "".join(re.findall(r'"([^"]*)"', body))
        expected = "".join(country + length for country, length in zip(self.countries, self.lengths))
        self.assertEqual(expected, generated)
        self.assertIn(hashlib.sha256(expected.encode("ascii")).hexdigest(), source)

    def test_write_then_check(self):
        """A current registry check is read-only and generation is repeatable."""
        self.assert_repeatable()
        self.assert_table()
        self.assertFalse(self.curl_log.exists())

    def test_stale_check(self):
        """A stale check returns failure without replacing the table."""
        result = self.run_generator("--check")
        self.assertEqual(1, result.returncode)
        self.assertIn("snapshot is stale", result.stderr)
        self.assertEqual(self.original, self.output.read_bytes())

    def test_normalized_row_labels(self):
        """Nonbreaking spaces and surrounding whitespace are accepted in row labels."""
        self.rows[0][0] = " " + COUNTRY_ROW.replace(" ", "\N{NO-BREAK SPACE}") + " "
        self.rows[1][0] = " " + LENGTH_ROW + " "
        self.assert_repeatable()
        self.assert_table()

    def test_duplicate_country_row(self):
        """A repeated country row cannot replace an earlier row."""
        self.rows.append([COUNTRY_ROW, *self.countries])
        self.assert_rejected("duplicate")

    def test_duplicate_length_row(self):
        """Even identical duplicate length rows are rejected."""
        self.rows.append([LENGTH_ROW.replace(" ", "\N{NO-BREAK SPACE}"), *self.lengths])
        self.assert_rejected("duplicate")

    def test_malformed_quoted_row(self):
        """An unclosed quoted field after the required rows is rejected."""
        self.suffix = 'Extra\t"unterminated'
        self.assert_rejected("registry")

    def test_too_few_countries(self):
        """Incomplete country and length rows cannot replace the table."""
        self.rows[0].pop()
        self.rows[1].pop()
        self.assert_rejected("too few entries")

    def test_mismatched_rows(self):
        """Country and length cells must have a one-to-one correspondence."""
        self.rows[1].pop()
        self.assert_rejected("mismatched")

    def test_duplicate_country(self):
        """Repeated countries are rejected even when their entries are sorted."""
        self.rows[0][2] = self.rows[0][1]
        self.assert_rejected("duplicate country codes")

    def test_unsorted_country(self):
        """Source order must be alphabetical to keep columns checkable."""
        self.rows[0][1], self.rows[0][2] = self.rows[0][2], self.rows[0][1]
        self.assert_rejected("not sorted")

    def test_missing_row(self):
        """Both required rows must be present."""
        del self.rows[1]
        self.assert_rejected("lacks")

    def test_small_source(self):
        """An unexpectedly short registry is rejected before parsing."""
        self.rows = [["Description", "short file"]]
        self.assert_rejected("unexpectedly small")


def encoded_case(encoding, bom, parity):
    """Build a separately reported supported-encoding test."""
    def check(self):
        """Require the same country projection regardless of text encoding."""
        self.encoding, self.bom, self.parity = encoding, bom, parity
        self.assert_repeatable()
        self.assert_table()
    return check


def invalid_field_case(row, value):
    """Build a field validation test with a malformed first entry."""
    def check(self):
        """Reject invalid country or length text without rewriting the table."""
        self.rows[row][1] = value
        self.assert_rejected("IBAN")
    return check


for index, arguments in enumerate((
        ("utf-8", b"", None), ("utf-8-sig", b"", None),
        ("utf-16-le", codecs.BOM_UTF16_LE, None), ("utf-16-be", codecs.BOM_UTF16_BE, None),
        ("cp1252", b"", 0), ("cp1252", b"", 1))):
    setattr(IbanLengthsTest, f"test_encoding_{index}_{arguments[0]}", encoded_case(*arguments))
for row, invalid in ((0, ("A", "ABC", "aA", "12", "\u00c9A")),
                     (1, ("", "1", "014", "14", "35", "2x", "-2", "\u0662\u0662"))):
    for index, value in enumerate(invalid):
        setattr(IbanLengthsTest, f"test_invalid_field_{row}_{index}_{value}",
                invalid_field_case(row, value))


if __name__ == "__main__":
    unittest.main()
