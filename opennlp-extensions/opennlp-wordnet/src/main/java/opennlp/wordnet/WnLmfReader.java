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
package opennlp.wordnet;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.xml.XMLConstants;
import javax.xml.stream.Location;
import javax.xml.stream.XMLInputFactory;
import javax.xml.stream.XMLStreamConstants;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.XMLStreamReader;

import opennlp.tools.util.InvalidFormatException;
import opennlp.tools.wordnet.LexicalKnowledgeBase;
import opennlp.tools.wordnet.Synset;
import opennlp.tools.wordnet.WordNetPOS;
import opennlp.tools.wordnet.WordNetRelation;

/**
 * Reads a WN-LMF XML document (the Global WordNet Association interchange format, used by Open
 * English WordNet and many other language wordnets) into a {@link LexicalKnowledgeBase}.
 *
 * <p>The reader is clean-room, built from the published format documentation, and uses only the
 * JDK's StAX parser. It reads the subset of the format the {@link LexicalKnowledgeBase} contract
 * serves: lexical entries (lemma, part of speech, senses), synsets (definition and typed
 * relations), and sense relations, which are lifted to the synset level as documented on
 * {@link WordNetRelation}. Elements outside that subset ({@code Pronunciation}, {@code Form},
 * {@code Example}, {@code SyntacticBehaviour}, {@code ILIDefinition}, and similar) are skipped.
 * Relations of type {@code other}, the format's escape hatch for relations typed only by
 * metadata, are also skipped: the contract has no untyped relation slot. Every other unknown
 * relation type fails loud.</p>
 *
 * <p><b>Security.</b> The parser is hardened against XXE per the OWASP-documented posture for
 * formats that carry a DOCTYPE: a DOCTYPE declaration is tokenized and skipped, but nothing it
 * names is ever resolved. External entities and the external DTD subset are both disabled
 * ({@code IS_SUPPORTING_EXTERNAL_ENTITIES} and {@code ACCESS_EXTERNAL_DTD} are off), and the
 * {@link javax.xml.stream.XMLResolver} throws on any resolution attempt regardless, so no
 * network or filesystem access can be triggered by a DOCTYPE, internal or external. Open English
 * WordNet releases ship with a DOCTYPE line referencing the schema DTD; this reader parses such
 * a file unmodified.</p>
 *
 * <p><b>Errors.</b> Malformed structure fails loud with an {@link InvalidFormatException}
 * naming the resource and, where the parser provides one, the line: missing required
 * attributes, a duplicate lexical entry, sense, or synset id, a sense pointing to an undeclared
 * synset, a relation to an undeclared target, an unknown part-of-speech code, an unknown
 * relation type, a synset with no member entries. I/O failures propagate as
 * {@link IOException}.</p>
 *
 * <p>Part-of-speech code {@code s} (adjective satellite) normalizes to
 * {@link WordNetPOS#ADJECTIVE}, and a {@code similar} relation whose source is a verb synset
 * maps to {@link WordNetRelation#VERB_GROUP} (that is how WN-LMF documents derived from
 * Princeton data express verb groups); on any other part of speech it maps to
 * {@link WordNetRelation#SIMILAR_TO}.</p>
 *
 * <p>The returned lexicon is immutable and safe for concurrent lookups.</p>
 */
public final class WnLmfReader {

  private static final Map<String, WordNetRelation> RELATION_NAMES = relationNames();

  /** The format's escape-hatch relation type; carries no type the contract can express. */
  private static final String OTHER_RELATION = "other";

  private WnLmfReader() {
  }

  /**
   * Reads a WN-LMF XML file.
   *
   * @param file The XML file. Must not be {@code null} and must exist.
   * @return The loaded lexicon.
   * @throws IllegalArgumentException Thrown if {@code file} is {@code null} or missing.
   * @throws InvalidFormatException Thrown if the document is malformed; the message names the
   *     file and, where available, the line.
   * @throws IOException Thrown if reading the file fails.
   */
  public static LexicalKnowledgeBase read(Path file) throws IOException {
    if (file == null) {
      throw new IllegalArgumentException("File must not be null");
    }
    if (!Files.isRegularFile(file)) {
      throw new IllegalArgumentException("File does not exist or is not a regular file: " + file);
    }
    try (InputStream in = new BufferedInputStream(Files.newInputStream(file))) {
      return read(in, file.toString());
    }
  }

  /**
   * Reads a WN-LMF XML document from a stream. The stream is not closed.
   *
   * @param in           The document stream. Must not be {@code null}.
   * @param resourceName The name used in error messages. Must not be {@code null}.
   * @return The loaded lexicon.
   * @throws IllegalArgumentException Thrown if an argument is {@code null}.
   * @throws InvalidFormatException Thrown if the document is malformed; the message names the
   *     resource and, where available, the line.
   * @throws IOException Thrown if reading the stream fails.
   */
  public static LexicalKnowledgeBase read(InputStream in, String resourceName) throws IOException {
    if (in == null) {
      throw new IllegalArgumentException("In must not be null");
    }
    if (resourceName == null) {
      throw new IllegalArgumentException("ResourceName must not be null");
    }
    final Parser parser = new Parser(resourceName);
    try {
      final XMLStreamReader reader = hardenedFactory().createXMLStreamReader(in);
      try {
        parser.parse(reader);
      } finally {
        reader.close();
      }
    } catch (XMLStreamException e) {
      // StAX wraps a failing stream read in an XMLStreamException; surface it as the I/O
      // failure it is instead of misreporting it as a malformed document.
      final Throwable nested = e.getNestedException() == null ? e.getCause()
          : e.getNestedException();
      if (nested instanceof IOException io) {
        throw io;
      }
      throw parser.malformed(e.getLocation(), "XML error: " + e.getMessage(), e);
    }
    return parser.build();
  }

  private static XMLInputFactory hardenedFactory() {
    final XMLInputFactory factory = XMLInputFactory.newFactory();
    // XXE hardening, the OWASP-documented posture for DOCTYPE-bearing formats: the internal
    // subset is not processed (so no custom entity gets declared at all), and both the
    // external DTD subset and external entities are denied, so a DOCTYPE is tokenized and
    // skipped but nothing it names is ever fetched.
    factory.setProperty(XMLInputFactory.SUPPORT_DTD, Boolean.FALSE);
    factory.setProperty(XMLInputFactory.IS_SUPPORTING_EXTERNAL_ENTITIES, Boolean.FALSE);
    factory.setProperty(XMLConstants.ACCESS_EXTERNAL_DTD, "");
    factory.setProperty(XMLInputFactory.IS_COALESCING, Boolean.TRUE);
    factory.setXMLResolver((publicId, systemId, baseUri, namespace) -> {
      throw new XMLStreamException("External entity resolution is disabled, refusing " + systemId);
    });
    return factory;
  }

  // The streaming parse state and post-parse resolution.
  private static final class Parser {

    private final String resourceName;

    // Entry state.
    private final Set<String> entryIds = new HashSet<>();
    private final Map<String, String> lemmaByEntryId = new HashMap<>();
    private final Map<String, WordNetPOS> posByEntryId = new HashMap<>();
    private final Map<String, String> synsetBySenseId = new HashMap<>();
    private final Map<InMemoryWordNetLexicon.LemmaKey, List<String>> senseOrder =
        new LinkedHashMap<>();
    private final List<RawSenseRelation> senseRelations = new ArrayList<>();
    private final Map<String, RawSynset> rawSynsets = new LinkedHashMap<>();
    // Fallback membership (entry ids per synset in document order) when members is absent.
    private final Map<String, List<String>> entryIdsBySynset = new HashMap<>();

    // Cursor state.
    private String currentEntryId;
    private String currentEntryLemma;
    private WordNetPOS currentEntryPos;
    private String currentSenseId;
    private RawSynset currentSynset;

    Parser(String resourceName) {
      this.resourceName = resourceName;
    }

    void parse(XMLStreamReader reader) throws XMLStreamException, InvalidFormatException {
      while (reader.hasNext()) {
        final int event = reader.next();
        // A DTD event (the DOCTYPE declaration) is intentionally not handled here: with
        // external entities and the external subset denied in the factory, it carries nothing
        // that can affect parsing, so it is skipped exactly like a comment or an ignored
        // element.
        if (event == XMLStreamConstants.START_ELEMENT) {
          startElement(reader);
        } else if (event == XMLStreamConstants.END_ELEMENT) {
          endElement(reader.getLocalName());
        }
      }
    }

    private void startElement(XMLStreamReader reader)
        throws XMLStreamException, InvalidFormatException {
      final String name = reader.getLocalName();
      switch (name) {
        case "LexicalEntry" -> {
          currentEntryId = requireAttribute(reader, "id");
          if (!entryIds.add(currentEntryId)) {
            throw malformed(reader.getLocation(),
                "Duplicate lexical entry id " + currentEntryId, null);
          }
          currentEntryLemma = null;
          currentEntryPos = null;
        }
        case "Lemma" -> {
          if (currentEntryId == null) {
            throw malformed(reader.getLocation(), "Lemma outside a LexicalEntry", null);
          }
          currentEntryLemma = requireAttribute(reader, "writtenForm");
          currentEntryPos = parsePos(requireAttribute(reader, "partOfSpeech"),
              reader.getLocation());
          lemmaByEntryId.put(currentEntryId, currentEntryLemma);
          posByEntryId.put(currentEntryId, currentEntryPos);
        }
        case "Sense" -> {
          if (currentEntryLemma == null) {
            throw malformed(reader.getLocation(),
                "Sense before its entry's Lemma in LexicalEntry " + currentEntryId, null);
          }
          currentSenseId = requireAttribute(reader, "id");
          final String synsetId = requireAttribute(reader, "synset");
          if (synsetBySenseId.putIfAbsent(currentSenseId, synsetId) != null) {
            throw malformed(reader.getLocation(), "Duplicate sense id " + currentSenseId, null);
          }
          entryIdsBySynset.computeIfAbsent(synsetId, unused -> new ArrayList<>(2))
              .add(currentEntryId);
          final List<String> order = senseOrder.computeIfAbsent(
              InMemoryWordNetLexicon.LemmaKey.of(currentEntryLemma, currentEntryPos),
              unused -> new ArrayList<>(2));
          if (!order.contains(synsetId)) {
            order.add(synsetId);
          }
        }
        case "SenseRelation" -> {
          if (currentSenseId == null) {
            throw malformed(reader.getLocation(), "SenseRelation outside a Sense", null);
          }
          senseRelations.add(new RawSenseRelation(currentSenseId,
              requireAttribute(reader, "relType"), requireAttribute(reader, "target"),
              line(reader.getLocation())));
        }
        case "Synset" -> {
          final String id = requireAttribute(reader, "id");
          final WordNetPOS pos = parsePos(requireAttribute(reader, "partOfSpeech"),
              reader.getLocation());
          currentSynset = new RawSynset(id, pos, reader.getAttributeValue(null, "members"),
              line(reader.getLocation()));
          if (rawSynsets.putIfAbsent(id, currentSynset) != null) {
            throw malformed(reader.getLocation(), "Duplicate synset id " + id, null);
          }
        }
        case "Definition" -> {
          if (currentSynset != null && currentSynset.gloss == null) {
            currentSynset.gloss = reader.getElementText();
          }
        }
        case "SynsetRelation" -> {
          if (currentSynset == null) {
            throw malformed(reader.getLocation(), "SynsetRelation outside a Synset", null);
          }
          final String relType = requireAttribute(reader, "relType");
          final String target = requireAttribute(reader, "target");
          // The escape-hatch type is skipped here exactly as it is for sense relations in
          // build(): documented skip, not rejection.
          if (!OTHER_RELATION.equals(relType)) {
            currentSynset.relations.add(
                new RawRelation(relType, target, line(reader.getLocation())));
          }
        }
        default -> {
          // Pronunciation, Form, Example, SyntacticBehaviour, ILIDefinition, and other
          // elements outside the contract subset are skipped.
        }
      }
    }

    private void endElement(String name) {
      switch (name) {
        case "LexicalEntry" -> {
          currentEntryId = null;
          currentEntryLemma = null;
          currentEntryPos = null;
        }
        case "Sense" -> currentSenseId = null;
        case "Synset" -> currentSynset = null;
        default -> {
          // Nothing to close for skipped elements.
        }
      }
    }

    LexicalKnowledgeBase build() throws InvalidFormatException {
      // Every sense must point to a declared synset, with a consistent part of speech.
      for (final Map.Entry<String, String> sense : synsetBySenseId.entrySet()) {
        final RawSynset target = rawSynsets.get(sense.getValue());
        if (target == null) {
          throw malformed(null,
              "Sense " + sense.getKey() + " references undeclared synset " + sense.getValue(),
              null);
        }
      }
      // Lift sense relations to the synset level.
      for (final RawSenseRelation relation : senseRelations) {
        if (OTHER_RELATION.equals(relation.relType)) {
          continue;
        }
        final String sourceSynsetId = synsetBySenseId.get(relation.sourceSenseId);
        final String targetSynsetId = synsetBySenseId.get(relation.targetSenseId);
        if (targetSynsetId == null) {
          throw malformed(null, "SenseRelation at line " + relation.line + " from sense "
              + relation.sourceSenseId + " references undeclared sense " + relation.targetSenseId,
              null);
        }
        final RawSynset source = rawSynsets.get(sourceSynsetId);
        source.relations.add(new RawRelation(relation.relType, targetSynsetId, relation.line));
      }
      // Resolve raw synsets into contract synsets.
      final Map<String, Synset> synsetsById = new LinkedHashMap<>(rawSynsets.size() * 2);
      for (final RawSynset raw : rawSynsets.values()) {
        final Map<WordNetRelation, List<String>> relations = resolveRelations(raw);
        synsetsById.put(raw.id,
            new Synset(raw.id, raw.pos, memberLemmas(raw), raw.gloss == null ? "" : raw.gloss,
                relations));
      }
      return new InMemoryWordNetLexicon(synsetsById, senseOrder);
    }

    private Map<WordNetRelation, List<String>> resolveRelations(RawSynset raw)
        throws InvalidFormatException {
      final Map<WordNetRelation, LinkedHashSet<String>> typed = new LinkedHashMap<>();
      for (final RawRelation relation : raw.relations) {
        final WordNetRelation type = parseRelation(relation.relType, raw.pos, relation.line);
        final RawSynset target = rawSynsets.get(relation.target);
        if (target == null) {
          throw malformed(null, "Relation " + relation.relType + " at line " + relation.line
              + " on synset " + raw.id + " references undeclared synset " + relation.target, null);
        }
        // The target's canonical id instance, not the pointer's own string: a full lexicon
        // carries hundreds of thousands of pointers, and sharing the synset table's instances
        // keeps only one copy of each id in memory.
        typed.computeIfAbsent(type, unused -> new LinkedHashSet<>()).add(target.id);
      }
      final Map<WordNetRelation, List<String>> relations = new LinkedHashMap<>(typed.size() * 2);
      for (final Map.Entry<WordNetRelation, LinkedHashSet<String>> entry : typed.entrySet()) {
        relations.put(entry.getKey(), List.copyOf(entry.getValue()));
      }
      return relations;
    }

    private List<String> memberLemmas(RawSynset raw) throws InvalidFormatException {
      final List<String> entryIds;
      if (raw.members != null && !raw.members.isEmpty()) {
        entryIds = LemmaFolding.splitOnSpaces(raw.members);
      } else {
        final List<String> fromSenses = entryIdsBySynset.get(raw.id);
        entryIds = fromSenses == null ? List.of() : fromSenses;
      }
      if (entryIds.isEmpty()) {
        throw malformed(null, "Synset " + raw.id + " at line " + raw.line
            + " has no member entries", null);
      }
      final List<String> lemmas = new ArrayList<>(entryIds.size());
      for (final String entryId : entryIds) {
        final String lemma = lemmaByEntryId.get(entryId);
        if (lemma == null) {
          throw malformed(null, "Synset " + raw.id + " at line " + raw.line
              + " lists undeclared member entry " + entryId, null);
        }
        if (raw.pos != posByEntryId.get(entryId)) {
          throw malformed(null, "Synset " + raw.id + " at line " + raw.line
              + " has part of speech " + raw.pos + " but member entry " + entryId
              + " has " + posByEntryId.get(entryId), null);
        }
        if (!lemmas.contains(lemma)) {
          lemmas.add(lemma);
        }
      }
      return lemmas;
    }

    private WordNetPOS parsePos(String code, Location location) throws InvalidFormatException {
      // The adjective satellite code normalizes to ADJECTIVE; see WordNetPOS.
      return switch (code) {
        case "n" -> WordNetPOS.NOUN;
        case "v" -> WordNetPOS.VERB;
        case "a", "s" -> WordNetPOS.ADJECTIVE;
        case "r" -> WordNetPOS.ADVERB;
        default -> throw malformed(location, "Unknown part-of-speech code: " + code, null);
      };
    }

    private WordNetRelation parseRelation(String relType, WordNetPOS sourcePos, int line)
        throws InvalidFormatException {
      // Documents derived from Princeton data express verb groups as similar on verb synsets.
      if ("similar".equals(relType)) {
        return sourcePos == WordNetPOS.VERB ? WordNetRelation.VERB_GROUP
            : WordNetRelation.SIMILAR_TO;
      }
      final WordNetRelation relation = RELATION_NAMES.get(relType);
      if (relation == null) {
        throw malformed(null, "Unknown relation type " + relType + " at line " + line, null);
      }
      return relation;
    }

    private String requireAttribute(XMLStreamReader reader, String attribute)
        throws InvalidFormatException {
      final String value = reader.getAttributeValue(null, attribute);
      if (value == null || value.isEmpty()) {
        throw malformed(reader.getLocation(), "Element " + reader.getLocalName()
            + " is missing required attribute " + attribute, null);
      }
      return value;
    }

    InvalidFormatException malformed(Location location, String message, Throwable cause) {
      final int line = line(location);
      final String prefix = line < 0 ? "Malformed WN-LMF document " + resourceName + ": "
          : "Malformed WN-LMF document " + resourceName + " at line " + line + ": ";
      return cause == null ? new InvalidFormatException(prefix + message)
          : new InvalidFormatException(prefix + message, cause);
    }

    private static int line(Location location) {
      return location == null ? -1 : location.getLineNumber();
    }
  }

  private static final class RawSynset {
    private final String id;
    private final WordNetPOS pos;
    private final String members;
    private final int line;
    private final List<RawRelation> relations = new ArrayList<>(4);
    private String gloss;

    RawSynset(String id, WordNetPOS pos, String members, int line) {
      this.id = id;
      this.pos = pos;
      this.members = members;
      this.line = line;
    }
  }

  private record RawRelation(String relType, String target, int line) {
  }

  private record RawSenseRelation(String sourceSenseId, String relType, String targetSenseId,
                                  int line) {
  }

  private static Map<String, WordNetRelation> relationNames() {
    final Map<String, WordNetRelation> names = new HashMap<>();
    names.put("antonym", WordNetRelation.ANTONYM);
    names.put("hypernym", WordNetRelation.HYPERNYM);
    names.put("instance_hypernym", WordNetRelation.INSTANCE_HYPERNYM);
    names.put("hyponym", WordNetRelation.HYPONYM);
    names.put("instance_hyponym", WordNetRelation.INSTANCE_HYPONYM);
    names.put("holo_member", WordNetRelation.MEMBER_HOLONYM);
    names.put("holo_substance", WordNetRelation.SUBSTANCE_HOLONYM);
    names.put("holo_part", WordNetRelation.PART_HOLONYM);
    names.put("mero_member", WordNetRelation.MEMBER_MERONYM);
    names.put("mero_substance", WordNetRelation.SUBSTANCE_MERONYM);
    names.put("mero_part", WordNetRelation.PART_MERONYM);
    names.put("attribute", WordNetRelation.ATTRIBUTE);
    names.put("derivation", WordNetRelation.DERIVATIONALLY_RELATED);
    names.put("entails", WordNetRelation.ENTAILMENT);
    names.put("is_entailed_by", WordNetRelation.ENTAILED_BY);
    names.put("causes", WordNetRelation.CAUSE);
    names.put("is_caused_by", WordNetRelation.CAUSED_BY);
    names.put("also", WordNetRelation.ALSO_SEE);
    names.put("participle", WordNetRelation.PARTICIPLE);
    names.put("pertainym", WordNetRelation.PERTAINYM);
    names.put("domain_topic", WordNetRelation.DOMAIN_TOPIC);
    names.put("has_domain_topic", WordNetRelation.MEMBER_OF_DOMAIN_TOPIC);
    names.put("domain_region", WordNetRelation.DOMAIN_REGION);
    names.put("has_domain_region", WordNetRelation.MEMBER_OF_DOMAIN_REGION);
    // The usage domain carries both its current WN-LMF name and the legacy alias.
    names.put("exemplifies", WordNetRelation.DOMAIN_USAGE);
    names.put("domain_usage", WordNetRelation.DOMAIN_USAGE);
    names.put("is_exemplified_by", WordNetRelation.MEMBER_OF_DOMAIN_USAGE);
    names.put("has_domain_usage", WordNetRelation.MEMBER_OF_DOMAIN_USAGE);
    return Map.copyOf(names);
  }
}
