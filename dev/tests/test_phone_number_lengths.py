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

"""Offline checks for the phone metadata generator, using synthetic XML."""

import re
import unittest
import xml.etree.ElementTree as ET

from generator_test_case import GeneratorTestCase

REVISION = "0123456789abcdef0123456789abcdef01234567"
SNAPSHOT_DATE = "2024-02-29"


class PhoneNumberLengthsTest(GeneratorTestCase):
    """Run the complete shell command against disposable source trees."""

    script_name = "fetch-phone-number-lengths.sh"
    java_name = "PhoneNumberLengths.java"

    def setUp(self):
        """Create an isolated checkout and 200 synthetic calling-code entries."""
        super().setUp()
        self.xml = ET.Element("phoneNumberMetadata")
        self.territories = ET.SubElement(self.xml, "territories")
        for code in range(100, 300):
            territory = ET.SubElement(self.territories, "territory",
                                      countryCode=str(code))
            description = ET.SubElement(territory, "fixedLine")
            ET.SubElement(description, "possibleLengths", national="10")
        self.first = self.territories[0]
        self.lengths = self.first.find("fixedLine/possibleLengths")
        self.environment.update(PHONE_METADATA_SOURCE=str(self.source),
                                PHONE_METADATA_REVISION=REVISION,
                                PHONE_METADATA_DATE=SNAPSHOT_DATE)

    def write_source(self):
        """Write the synthetic XML before each generator run."""
        ET.ElementTree(self.xml).write(self.source, encoding="utf-8")

    def read_table(self):
        """Read code/mask pairs from the generated Java initializer."""
        source = self.output.read_text(encoding="utf-8")
        body = source.partition("CODE_AND_MASK = {")[2].partition("};")[0]
        self.assertTrue(body)
        values = [int(token, 0) for token in re.findall(r"0x[0-9A-F]+|[0-9]+", body)]
        self.assertEqual(0, len(values) % 2)
        codes = values[::2]
        self.assertEqual(sorted(set(codes)), codes)
        return dict(zip(codes, values[1::2]))

    def test_write_then_check(self):
        """Generation is repeatable, and a current check does not rewrite the file."""
        self.assert_repeatable()
        generated = self.output.read_bytes()
        self.assertNotEqual(self.original, generated)
        self.assertIn(f"revision {{@code {REVISION[:12]}}} of {SNAPSHOT_DATE}",
                      generated.decode("utf-8"))
        self.assertEqual({code: 1 << 10 for code in range(100, 300)}, self.read_table())

    def test_stale_check_preserves_source(self):
        """A differing snapshot returns a failure without changing the table."""
        result = self.run_generator("--check")
        self.assertEqual(1, result.returncode)
        self.assertIn("snapshot is stale", result.stderr)
        self.assertEqual(self.original, self.output.read_bytes())

    def test_shared_codes_and_multiple_types(self):
        """Shared-code territories and phone types contribute their national lengths."""
        self.lengths.set("national", "[8-10]")
        self.lengths.set("localOnly", "5,6")
        mobile = ET.SubElement(self.first, "mobile")
        ET.SubElement(mobile, "possibleLengths", national="10,12", localOnly="7")
        shared = ET.SubElement(self.territories, "territory", countryCode="100")
        ET.SubElement(ET.SubElement(shared, "fixedLine"), "possibleLengths", national="13")
        result = self.run_generator()
        self.assertEqual(0, result.returncode, result.stderr)
        expected = {code: 1 << 10 for code in range(101, 300)}
        expected[100] = sum(1 << length for length in (8, 9, 10, 12, 13))
        self.assertEqual(expected, self.read_table())

    def test_missing_national_attribute(self):
        """A valid sibling type does not hide a missing national-length attribute."""
        del self.lengths.attrib["national"]
        ET.SubElement(ET.SubElement(self.first, "mobile"), "possibleLengths", national="10")
        self.assert_rejected("national")

    def test_missing_lengths(self):
        """A territory without any national lengths is rejected."""
        self.first.remove(self.first[0])
        self.assert_rejected("has no national lengths")

    def test_wrong_root(self):
        """The input must have the expected metadata root."""
        self.xml.tag = "html"
        self.assert_rejected("unexpected metadata root")

    def test_truncated_territories(self):
        """An incomplete source cannot replace the table."""
        self.territories.remove(self.territories[-1])
        self.assert_rejected("only 199 territories")

    def test_too_few_distinct_codes(self):
        """Territory count does not substitute for distinct calling-code count."""
        self.first.set("countryCode", "101")
        self.assert_rejected("only 199 calling codes")

    def test_missing_revision(self):
        """Offline input requires its revision without starting a download."""
        del self.environment["PHONE_METADATA_REVISION"]
        self.assert_rejected("requires PHONE_METADATA_REVISION")

    def test_missing_date(self):
        """Offline input requires its snapshot date without starting a download."""
        del self.environment["PHONE_METADATA_DATE"]
        self.assert_rejected("PHONE_METADATA_DATE")

    def test_missing_output_marker(self):
        """A source layout mismatch leaves the destination file intact."""
        self.original = self.original.replace(b"CODE_AND_MASK = {", b"RENAMED = {")
        self.output.write_bytes(self.original)
        self.assert_rejected("expected one comment and table replacement")


def length_case(value, expected):
    """Build an independently named syntax acceptance or rejection test."""
    def check(self):
        """Check the exact mask, or reject the input without writing source."""
        self.lengths.set("national", value)
        if expected is None:
            self.assert_rejected("national")
        else:
            result = self.run_generator()
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual(sum(1 << length for length in expected), self.read_table()[100])
    return check


def attribute_case(name, value, message):
    """Build a rejection test for calling-code or snapshot attributes."""
    def check(self):
        """Reject a malformed attribute without changing the table."""
        if name == "countryCode":
            self.first.set(name, value)
        else:
            self.environment[name] = value
        self.assert_rejected(message)
    return check


ACCEPTED_LENGTHS = (
    ("1", {1}), ("31", {31}), ("8,10", {8, 10}),
    ("[8-10]", {8, 9, 10}), ("[3-5],7,9,[11-14]", {3, 4, 5, 7, 9, 11, 12, 13, 14}),
    ("[1-31]", set(range(1, 32))),
)
INVALID_LENGTHS = (
    "", "words", "10garbage12", "x10", "10x", "10;12", "10 12", "10.12", "-1", "+10",
    "1e1", "0xA", "10,", ",10", "10,,12", "10-12", "[10-12", "10-12]", "[10]",
    "[[10-12]]", "[10-12-14]", "[10-10]", "[10-11]", "[12-10]", "10,10", "[8-10],10",
    "[8-10],[10-12]", "[0-2]", "0", "32", "[30-32]", "\u0661\u0660", "10,\u0661\u0662",
)
INVALID_CODES = ("0", "1000", "0100", "001", "-1", "+1", "1x", "\u0661")
INVALID_DATES = ("2024-02-30", "2023-02-29", "2024-13-01", "2024-00-01", "0000-01-01",
                 "2024-2-01", "20240229", "\u0662\u0660\u0662\u0664-02-29")
INVALID_REVISIONS = ("abc123", "a" * 41, "A" * 40, "not-a-revision")

for index, (value, expected) in enumerate(ACCEPTED_LENGTHS):
    setattr(PhoneNumberLengthsTest, f"test_valid_lengths_{index}_{value}",
            length_case(value, expected))
for index, value in enumerate(INVALID_LENGTHS):
    setattr(PhoneNumberLengthsTest, f"test_invalid_lengths_{index}_{value}",
            length_case(value, None))
for attribute, values, message in (
        ("countryCode", INVALID_CODES, "calling code"),
        ("PHONE_METADATA_DATE", INVALID_DATES, "snapshot date"),
        ("PHONE_METADATA_REVISION", INVALID_REVISIONS, "revision")):
    for index, value in enumerate(values):
        setattr(PhoneNumberLengthsTest, f"test_invalid_{attribute}_{index}_{value}",
                attribute_case(attribute, value, message))


if __name__ == "__main__":
    unittest.main()
