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

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import javax.xml.XMLConstants;
import javax.xml.stream.XMLInputFactory;
import javax.xml.stream.XMLStreamConstants;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.XMLStreamReader;

import opennlp.tools.glossary.GlossaryEntry;
import opennlp.tools.util.InvalidFormatException;
import opennlp.tools.util.StringUtil;

/**
 * A {@link GlossaryReader} for TBX termbases (ISO 30042, TermBase eXchange), the
 * interchange format CAT tools and public termbases such as IATE export. Both element
 * vocabularies are understood by local name: the TBX&#160;2 martif shape
 * ({@code termEntry}, {@code langSet}, {@code tig} or nested {@code ntig}) and the
 * TBX&#160;3 shape ({@code conceptEntry}, {@code langSec}, {@code termSec}), with or
 * without the ISO 30042 namespace.
 *
 * <p>A termbase is multilingual while a matcher wants one language's surface forms, so
 * the reader selects one language at construction. Selection follows BCP&#160;47
 * lookup: the configured tag matches a {@code xml:lang} that is equal
 * (case-insensitively) or that extends it with a subtag, so {@code en} selects
 * {@code en-US} but {@code en-GB} does not. Every {@code term} of a selected language
 * section becomes one {@link GlossaryEntry} carrying the enclosing entry's {@code id}
 * attribute, so synonyms arrive as aliases sharing one id, which the matchers already
 * understand. A language absent from the file reads as an empty list; the matcher
 * constructors reject an empty glossary, so the miss cannot pass silently.</p>
 *
 * <p>Real exports carry a {@code DOCTYPE} referencing the TBX DTD. The declaration is
 * tolerated and internal entities resolve, but nothing external is ever fetched:
 * external entity references fail closed as {@link InvalidFormatException}, and the
 * external DTD subset resolves to nothing. Entries without an {@code id} attribute,
 * blank terms, and malformed XML also fail loud as {@link InvalidFormatException}.</p>
 *
 * <p>The reader holds no per-call state and is safe to share across threads.</p>
 *
 * @see <a href="https://www.iso.org/standard/62510.html">ISO 30042 (TBX)</a>
 * @since 3.0.0
 */
public final class TbxGlossaryReader implements GlossaryReader {

  /** The lowercased BCP 47 tag selecting which language sections to read. */
  private final String language;

  /**
   * Builds a reader for one language of a termbase.
   *
   * @param languageTag The BCP 47 tag of the language to read, for example {@code en}
   *                    or {@code en-US}. Must not be {@code null} or blank. Matching
   *                    is case-insensitive and by subtag prefix.
   * @throws IllegalArgumentException Thrown if {@code languageTag} is {@code null}
   *         or blank.
   */
  public TbxGlossaryReader(String languageTag) {
    if (languageTag == null || StringUtil.isBlank(languageTag)) {
      throw new IllegalArgumentException("languageTag must not be null or blank");
    }
    this.language = languageTag.toLowerCase(Locale.ROOT);
  }

  @Override
  public List<GlossaryEntry> read(InputStream in) throws IOException {
    if (in == null) {
      throw new IllegalArgumentException("in must not be null");
    }
    final List<GlossaryEntry> entries = new ArrayList<>();
    final XMLStreamReader xml;
    try {
      xml = createSecureFactory().createXMLStreamReader(in);
    } catch (XMLStreamException e) {
      throw new InvalidFormatException("not parseable as XML: " + e.getMessage(), e);
    }
    try {
      readEntries(xml, entries);
    } catch (XMLStreamException e) {
      throw new InvalidFormatException("malformed TBX content: " + e.getMessage(), e);
    } finally {
      try {
        xml.close();
      } catch (XMLStreamException e) {
        // The stream itself stays the caller's to close; a close failure of the
        // parser wrapper after a successful read carries no information.
      }
    }
    return entries;
  }

  /**
   * Walks the document and collects the selected language's terms.
   *
   * @param xml The open stream reader positioned before the root.
   * @param entries The sink for entries in file order.
   * @throws XMLStreamException Thrown if the XML is malformed.
   * @throws InvalidFormatException Thrown if the content violates the termbase shape.
   */
  private void readEntries(XMLStreamReader xml, List<GlossaryEntry> entries)
      throws XMLStreamException, InvalidFormatException {
    boolean sawRoot = false;
    String entryId = null;
    boolean inSelectedLanguage = false;
    while (xml.hasNext()) {
      final int event = xml.next();
      if (event == XMLStreamConstants.START_ELEMENT) {
        final String name = xml.getLocalName();
        if (!sawRoot) {
          if (!"martif".equals(name) && !"tbx".equals(name)) {
            throw new InvalidFormatException(
                "expected a martif or tbx root element, found: " + name);
          }
          sawRoot = true;
        } else if ("termEntry".equals(name) || "conceptEntry".equals(name)) {
          entryId = xml.getAttributeValue(null, "id");
          if (entryId == null || StringUtil.isBlank(entryId)) {
            throw new InvalidFormatException(
                "term entry without an id attribute; ids become GlossaryEntry"
                    + " identifiers and cannot be invented");
          }
        } else if ("langSet".equals(name) || "langSec".equals(name)) {
          inSelectedLanguage = matchesLanguage(
              xml.getAttributeValue(XMLConstants.XML_NS_URI, "lang"));
        } else if ("term".equals(name) && inSelectedLanguage && entryId != null) {
          final String term = collectText(xml).trim();
          if (StringUtil.isBlank(term)) {
            throw new InvalidFormatException(
                "blank term in entry \"" + entryId + "\"");
          }
          entries.add(new GlossaryEntry(entryId, term));
        }
      } else if (event == XMLStreamConstants.END_ELEMENT) {
        final String name = xml.getLocalName();
        if ("termEntry".equals(name) || "conceptEntry".equals(name)) {
          entryId = null;
        } else if ("langSet".equals(name) || "langSec".equals(name)) {
          inSelectedLanguage = false;
        }
      }
    }
  }

  /**
   * Collects the text of the current element including nested inline markup, leaving
   * the reader on the element's end tag.
   *
   * @param xml The reader positioned on a start element.
   * @return The concatenated character data of the element and its descendants.
   * @throws XMLStreamException Thrown if the XML is malformed.
   */
  private static String collectText(XMLStreamReader xml) throws XMLStreamException {
    final StringBuilder text = new StringBuilder();
    int depth = 1;
    while (depth > 0) {
      final int event = xml.next();
      if (event == XMLStreamConstants.START_ELEMENT) {
        depth++;
      } else if (event == XMLStreamConstants.END_ELEMENT) {
        depth--;
      } else if (event == XMLStreamConstants.CHARACTERS
          || event == XMLStreamConstants.CDATA
          || event == XMLStreamConstants.ENTITY_REFERENCE) {
        text.append(xml.getText());
      }
    }
    return text.toString();
  }

  /**
   * Applies BCP 47 lookup between the configured tag and a section's language.
   *
   * @param xmlLang The {@code xml:lang} value of the section, possibly {@code null}.
   * @return {@code true} if the section belongs to the configured language.
   */
  private boolean matchesLanguage(String xmlLang) {
    if (xmlLang == null) {
      return false;
    }
    final String folded = xmlLang.toLowerCase(Locale.ROOT);
    return folded.equals(language) || folded.startsWith(language + "-");
  }

  /**
   * Configures a factory that tolerates the DOCTYPE real exports carry while never
   * fetching anything external: external entity support is off and the external DTD
   * subset resolves to an empty stream.
   *
   * @return The hardened factory.
   */
  private static XMLInputFactory createSecureFactory() {
    final XMLInputFactory factory = XMLInputFactory.newFactory();
    factory.setProperty(XMLInputFactory.IS_SUPPORTING_EXTERNAL_ENTITIES, Boolean.FALSE);
    factory.setProperty(XMLInputFactory.SUPPORT_DTD, Boolean.TRUE);
    factory.setProperty(XMLInputFactory.IS_COALESCING, Boolean.TRUE);
    factory.setXMLResolver((publicId, systemId, baseUri, namespace) ->
        new ByteArrayInputStream(new byte[0]));
    return factory;
  }
}
