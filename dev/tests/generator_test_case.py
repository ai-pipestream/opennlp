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

"""Temporary checkouts and a local download substitute for generator tests."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
JAVA_DIRECTORY = Path("opennlp-core/opennlp-runtime/src/main/java/opennlp/tools/pii")
DOWNLOAD_COMMAND = """#!/usr/bin/env python3
import os
from pathlib import Path
import shutil
import sys

Path(os.environ['GENERATOR_CURL_LOG']).write_text(repr(sys.argv[1:]), encoding='utf-8')
status = int(os.environ['GENERATOR_CURL_EXIT'])
if status:
    raise SystemExit(status)
output = sys.argv[sys.argv.index('-o') + 1]
shutil.copyfile(os.environ['GENERATOR_TEST_SOURCE'], output)
"""


class GeneratorTestCase(unittest.TestCase):
    """Run a generator without touching repository output or contacting a server."""

    def setUp(self):
        """Copy the selected generator and its Java template into a temporary tree."""
        temporary = tempfile.TemporaryDirectory(prefix="reference generator ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.script = self.root / "dev" / self.script_name
        self.script.parent.mkdir()
        shutil.copyfile(ROOT / "dev" / self.script_name, self.script)
        self.output = self.root / JAVA_DIRECTORY / self.java_name
        self.output.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / JAVA_DIRECTORY / self.java_name, self.output)
        self.original = self.output.read_bytes()
        self.source = self.root / "input source"
        self.input_bytes = b""
        self.curl_log = self.root / "download.log"
        binary_directory = self.root / "bin"
        binary_directory.mkdir()
        curl = binary_directory / "curl"
        curl.write_text(DOWNLOAD_COMMAND, encoding="utf-8")
        curl.chmod(0o755)
        self.environment = os.environ.copy()
        for name in ("PHONE_METADATA_SOURCE", "PHONE_METADATA_REVISION", "PHONE_METADATA_DATE",
                     "IBAN_REGISTRY_SOURCE", "IANA_TLDS_SOURCE"):
            self.environment.pop(name, None)
        self.environment.update(PATH=str(binary_directory) + os.pathsep + os.environ["PATH"],
                                GENERATOR_CURL_LOG=str(self.curl_log), GENERATOR_CURL_EXIT="0",
                                GENERATOR_TEST_SOURCE=str(self.source))

    def write_source(self):
        """Write the case's input bytes before starting the generator."""
        self.source.write_bytes(self.input_bytes)

    def run_generator(self, *arguments):
        """Write the fixture and run with a bounded process timeout."""
        self.write_source()
        return subprocess.run(["bash", str(self.script), *arguments], cwd=self.root,
                              env=self.environment, capture_output=True, text=True, timeout=15)

    def assert_rejected(self, message):
        """Require an actionable error and an unchanged destination file."""
        result = self.run_generator()
        self.assertNotEqual(0, result.returncode, result.stdout)
        self.assertIn(message, result.stderr)
        self.assertEqual(self.original, self.output.read_bytes())

    def assert_repeatable(self):
        """Require stable generation and a read-only successful snapshot check."""
        result = self.run_generator()
        self.assertEqual(0, result.returncode, result.stderr)
        generated = self.output.read_bytes()
        modified = self.output.stat().st_mtime_ns
        checked = self.run_generator("--check")
        self.assertEqual(0, checked.returncode, checked.stderr)
        self.assertEqual(generated, self.output.read_bytes())
        self.assertEqual(modified, self.output.stat().st_mtime_ns)
        self.assertEqual(0, self.run_generator().returncode)
        self.assertEqual(generated, self.output.read_bytes())
