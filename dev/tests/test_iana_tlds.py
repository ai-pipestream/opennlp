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

"""Synthetic IANA lists for header, label and source-preservation tests."""

import re
import unittest

from generator_test_case import GeneratorTestCase


HEADER = "# Version 2026090601, Last Updated Mon Sep  7 07:07:01 2026 UTC"


class IanaTldsTest(GeneratorTestCase):
    """Exercise the TLD generator with a local substitute for downloads."""

    script_name = "fetch-iana-tlds.sh"
    java_name = "IanaTlds.java"

    def setUp(self):
        """Create a complete-size synthetic list without copying the root zone."""
        super().setUp()
        self.header = HEADER
        self.labels = [f"TEST{index:04d}" for index in range(1000)]
        self.environment["IANA_TLDS_SOURCE"] = str(self.source)

    def write_source(self):
        """Write a list in the IANA download's line-based format."""
        self.source.write_text("\n".join([self.header, *self.labels]) + "\n", encoding="utf-8")

    def assert_table(self):
        """Check the exact generated label sequence and safe snapshot header."""
        source = self.output.read_text(encoding="utf-8")
        body = source.partition("private static final String TLDS =")[2].partition(";")[0]
        packed = "".join(re.findall(r'"([^"]*)"', body))
        self.assertEqual(self.labels, packed.split(","))
        self.assertIn("Snapshot " + self.header[2:] + "; regenerated", source)

    def test_write_then_check(self):
        """Generation and read-only checks work without downloading metadata."""
        self.assert_repeatable()
        self.assert_table()
        self.assertFalse(self.curl_log.exists())

    def test_downloaded_source(self):
        """The download path processes the same synthetic bytes."""
        del self.environment["IANA_TLDS_SOURCE"]
        self.assert_repeatable()
        self.assert_table()
        self.assertTrue(self.curl_log.exists())

    def test_download_failure(self):
        """A failed download does not replace the source table."""
        del self.environment["IANA_TLDS_SOURCE"]
        self.environment["GENERATOR_CURL_EXIT"] = "22"
        result = self.run_generator()
        self.assertEqual(22, result.returncode)
        self.assertEqual(self.original, self.output.read_bytes())

    def test_stale_check(self):
        """A stale check leaves the checked-in table intact."""
        result = self.run_generator("--check")
        self.assertEqual(1, result.returncode)
        self.assertIn("snapshot is stale", result.stderr)
        self.assertEqual(self.original, self.output.read_bytes())

    def test_duplicate_label(self):
        """Duplicate labels are rejected even when sorted."""
        self.labels[1] = self.labels[0]
        self.assert_rejected("duplicates")

    def test_unsorted_labels(self):
        """The source must retain alphabetical order."""
        self.labels[0], self.labels[1] = self.labels[1], self.labels[0]
        self.assert_rejected("not sorted")

    def test_truncated_list(self):
        """A truncated list does not replace the table."""
        self.labels.pop()
        self.assert_rejected("entries")

    def test_empty_list(self):
        """An empty list is rejected before source generation."""
        self.labels.clear()
        self.assert_rejected("entries")

    def test_long_labels_and_punycode(self):
        """Maximum-length labels and punycode retain their exact characters."""
        self.labels += ["W" * 63, "XN--P1AI"]
        self.assert_repeatable()
        self.assert_table()

    def test_missing_source_marker(self):
        """An unexpected Java template cannot be rewritten partially."""
        self.original = self.original.replace(b"String TLDS =", b"String RENAMED =")
        self.output.write_bytes(self.original)
        self.assert_rejected("expected exactly one snapshot")


def invalid_header_case(header):
    """Build a named rejection case for malformed snapshot metadata."""
    def check(self):
        """Reject the header before it can become part of a Java comment."""
        self.header = header
        self.assert_rejected("header")
    return check


def invalid_label_case(label):
    """Build a named rejection case for an invalid ASCII label."""
    def check(self):
        """Reject malformed labels before checking source order."""
        self.labels[0] = label
        self.assert_rejected("IANA TLD")
    return check


INVALID_HEADERS = (
    "", "Version 2026090601", "# Version", "# Version invalid", "# Version 1 */ invalid /*",
    HEADER + " */", HEADER + "\\u002a\\u002f", HEADER.replace("UTC", "PST"),
    HEADER.replace("07:07:01", "25:07:01"), HEADER.replace("Sep  7", "Feb 30"),
    HEADER.replace("Mon", "Tue"), HEADER.replace("2026090601", "\u0662" * 10),
)
INVALID_LABELS = ("A" * 64, "-COM", "COM-", "coM", "A_B", "A.B", "A B", "\u00c9XAMPLE", "COM*/")
for index, value in enumerate(INVALID_HEADERS):
    setattr(IanaTldsTest, f"test_invalid_header_{index}", invalid_header_case(value))
for index, value in enumerate(INVALID_LABELS):
    setattr(IanaTldsTest, f"test_invalid_label_{index}", invalid_label_case(value))


if __name__ == "__main__":
    unittest.main()
