/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
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

package opennlp.tools.formats.glossary;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

import opennlp.tools.glossary.GlossaryEntry;
import opennlp.tools.util.InvalidFormatException;
import opennlp.tools.util.StringUtil;

/**
 * A {@link GlossaryReader} for CSV and TSV term lists, the de facto interchange shape
 * for glossaries: one row per entry, the first column the identifier, the second the
 * term. Columns beyond the second are ignored, leaving room for metadata columns such
 * as language or status.
 *
 * <p>Parsing follows RFC&#160;4180: fields may be quoted, a quoted field may contain
 * the delimiter, doubled quotes, and line breaks, and rows end with LF or CRLF. Input
 * is decoded as UTF-8 and a leading byte order mark (the Excel export signature) is
 * stripped. Blank lines are skipped. Rows with fewer than two columns, blank
 * identifiers or terms, and quotes left open at end of input fail loud as
 * {@link InvalidFormatException} naming the offending line.</p>
 *
 * <p>The reader holds no per-call state and is safe to share across threads.</p>
 *
 * @see <a href="https://www.rfc-editor.org/rfc/rfc4180">RFC 4180</a>
 * @since 3.0.0
 */
public final class CsvGlossaryReader implements GlossaryReader {

  private static final char QUOTE = '"';
  private static final char BYTE_ORDER_MARK = '\uFEFF';

  /** The character separating fields; comma for CSV, tab for TSV. */
  private final char delimiter;

  /** Whether the first row is a header to drop. */
  private final boolean skipHeader;

  /**
   * Builds a comma-separated reader that treats the first row as data.
   */
  public CsvGlossaryReader() {
    this(',', false);
  }

  /**
   * Builds a reader for a delimiter and header convention.
   *
   * @param delimiter The field separator, for example {@code ','} or {@code '\t'}.
   *                  Must not be the quote character or a line break.
   * @param skipHeader Whether to drop the first row as a header.
   * @throws IllegalArgumentException Thrown if {@code delimiter} is the quote
   *         character or a line break.
   */
  public CsvGlossaryReader(char delimiter, boolean skipHeader) {
    if (delimiter == QUOTE || delimiter == '\n' || delimiter == '\r') {
      throw new IllegalArgumentException(
          "delimiter must not be the quote character or a line break");
    }
    this.delimiter = delimiter;
    this.skipHeader = skipHeader;
  }

  @Override
  public List<GlossaryEntry> read(InputStream in) throws IOException {
    if (in == null) {
      throw new IllegalArgumentException("in must not be null");
    }
    final List<GlossaryEntry> entries = new ArrayList<>();
    final BufferedReader reader =
        new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8));

    final List<String> fields = new ArrayList<>();
    final StringBuilder field = new StringBuilder();
    boolean inQuotes = false;
    boolean recordHasContent = false;
    boolean headerPending = skipHeader;
    int line = 1;
    int recordLine = 1;

    int read = reader.read();
    if (read == BYTE_ORDER_MARK) {
      read = reader.read();
    }
    while (read >= 0) {
      final char c = (char) read;
      if (inQuotes) {
        if (c == QUOTE) {
          final int next = reader.read();
          if (next == QUOTE) {
            field.append(QUOTE);
          } else {
            inQuotes = false;
            read = next;
            continue;
          }
        } else {
          if (c == '\n') {
            line++;
          }
          field.append(c);
        }
      } else if (c == QUOTE && field.isEmpty()) {
        inQuotes = true;
        recordHasContent = true;
      } else if (c == delimiter) {
        fields.add(field.toString());
        field.setLength(0);
        recordHasContent = true;
      } else if (c == '\n' || c == '\r') {
        if (c == '\r') {
          final int next = reader.read();
          if (next != '\n' && next >= 0) {
            line++;
            if (recordHasContent || !field.isEmpty()) {
              headerPending = endRecord(fields, field, entries, recordLine, headerPending);
            }
            recordHasContent = false;
            recordLine = line;
            read = next;
            continue;
          }
        }
        line++;
        if (recordHasContent || !field.isEmpty()) {
          headerPending = endRecord(fields, field, entries, recordLine, headerPending);
        }
        recordHasContent = false;
        recordLine = line;
      } else {
        field.append(c);
      }
      read = reader.read();
    }
    if (inQuotes) {
      throw new InvalidFormatException(
          "quote opened on line " + recordLine + " is never closed");
    }
    if (recordHasContent || !field.isEmpty()) {
      endRecord(fields, field, entries, recordLine, headerPending);
    }
    return entries;
  }

  /**
   * Finishes one record: validates the column shape, converts it to an entry unless
   * it is the pending header, and resets the field buffers.
   *
   * @param fields The completed fields of the record; emptied by this call.
   * @param field The trailing field in progress; emptied by this call.
   * @param entries The sink for accepted entries.
   * @param recordLine The line the record started on, for error messages.
   * @param headerPending Whether this record is the header to drop.
   * @return The header flag for the next record: always {@code false}.
   * @throws InvalidFormatException Thrown if the record has fewer than two columns or
   *         a blank id or term.
   */
  private boolean endRecord(List<String> fields, StringBuilder field,
      List<GlossaryEntry> entries, int recordLine, boolean headerPending)
      throws InvalidFormatException {
    fields.add(field.toString());
    field.setLength(0);
    try {
      if (headerPending) {
        return false;
      }
      if (fields.size() < 2) {
        throw new InvalidFormatException(
            "expected at least an id and a term column on line " + recordLine
                + ", found " + fields.size() + " column(s)");
      }
      final String id = fields.get(0);
      final String term = fields.get(1);
      if (StringUtil.isBlank(id)) {
        throw new InvalidFormatException("blank id on line " + recordLine);
      }
      if (StringUtil.isBlank(term)) {
        throw new InvalidFormatException("blank term on line " + recordLine);
      }
      entries.add(new GlossaryEntry(id, term));
      return false;
    } finally {
      fields.clear();
    }
  }
}
