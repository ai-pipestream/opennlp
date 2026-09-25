/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Records Hunspell's results for the OpenNLP Hunspell stemmer tests.
//
// Usage: hunspell-record [--analyze] <affix file> <dictionary file> < inputs
//
// Reads one input per line from standard input, in the encoding the affix file
// declares with SET, and prints one line per input:
//
//   <input> TAB accepted|rejected [TAB <stem>]...
//
// accepted or rejected is the result of Hunspell::spell(). The stems are the
// results of Hunspell::stem(). With --analyze, the results of
// Hunspell::analyze() are printed in place of the stems.
//
// See dev/README-hunspell-dictionaries.md for how to build and run it.

#include <iostream>
#include <string>
#include <vector>

#include <hunspell.hxx>

namespace {

const char* const ANALYZE_OPTION = "--analyze";
const char* const ACCEPTED = "accepted";
const char* const REJECTED = "rejected";
const char FIELD_SEPARATOR = '\t';

// Prints the usage line and returns the exit status for a usage error.
int usage(const char* program) {
  std::cerr << "usage: " << program << " [" << ANALYZE_OPTION
            << "] <affix file> <dictionary file> < inputs" << std::endl;
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  int next = 1;
  bool analyze = false;
  if (argc > next && std::string(argv[next]) == ANALYZE_OPTION) {
    analyze = true;
    next++;
  }
  if (argc - next != 2) {
    return usage(argv[0]);
  }
  Hunspell hunspell(argv[next], argv[next + 1]);
  std::string input;
  while (std::getline(std::cin, input)) {
    if (!input.empty() && input.back() == '\r') {
      input.pop_back();
    }
    std::cout << input << FIELD_SEPARATOR
              << (hunspell.spell(input) ? ACCEPTED : REJECTED);
    const std::vector<std::string> results =
        analyze ? hunspell.analyze(input) : hunspell.stem(input);
    for (const std::string& result : results) {
      std::cout << FIELD_SEPARATOR << result;
    }
    std::cout << '\n';
  }
  return 0;
}
