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
import java.util.Optional;
import java.util.Set;
import javax.xml.XMLConstants;
import javax.xml.namespace.QName;
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
import opennlp.wordnet.WnLmfRawModel.Kind;
import opennlp.wordnet.WnLmfRawModel.RawEntry;
import opennlp.wordnet.WnLmfRawModel.RawExternalEntry;
import opennlp.wordnet.WnLmfRawModel.RawExternalSense;
import opennlp.wordnet.WnLmfRawModel.RawExternalSynset;
import opennlp.wordnet.WnLmfRawModel.RawLexicon;
import opennlp.wordnet.WnLmfRawModel.RawRelation;
import opennlp.wordnet.WnLmfRawModel.RawResource;
import opennlp.wordnet.WnLmfRawModel.RawSense;
import opennlp.wordnet.WnLmfRawModel.RawSenseRelation;
import opennlp.wordnet.WnLmfRawModel.RawSynset;

/**
 * Reads a WN-LMF XML document (the Global WordNet Association
 * <a href="https://globalwordnet.github.io/schemas/">interchange format</a>, used by
 * <a href="https://github.com/globalwordnet/english-wordnet">Open English WordNet</a> and many
 * other language wordnets) into a {@link LexicalKnowledgeBase} using the JDK StAX parser.
 *
 * <p>It reads lexical entries, synsets with their definitions and every typed relation in WN-LMF
 * 1.4, and sense relations, which are lifted to the synset level as documented on
 * {@link WordNetRelation}. Elements outside that subset are skipped, as are relations of type
 * {@code other}, the format's untyped escape hatch. Any other unknown relation type fails loud.</p>
 *
 * <p>The parser is hardened against XXE: DTD processing and external entities are disabled, so a
 * DOCTYPE is skipped but nothing it names is fetched or resolved.</p>
 *
 * <p>Malformed structure fails loud with an {@link InvalidFormatException} naming the resource
 * and, where the parser provides one, the line; I/O failures propagate as {@link IOException}.
 * Part-of-speech code {@code s} normalizes to {@link WordNetPOS#ADJECTIVE}, and a {@code similar}
 * relation on a verb synset maps to {@link WordNetRelation#VERB_GROUP} rather than
 * {@link WordNetRelation#SIMILAR_TO}. Use {@link #readResource(Path)} when a document contains
 * several lexicons; the single-lexicon {@code read} methods reject that shape instead of merging
 * language-specific indexes. WN-LMF {@code Requires} declarations are preserved as dependency
 * metadata but never resolved or loaded. Returned resources and lexicons are immutable and safe
 * for concurrent lookups.</p>
 *
 * <p>A WN-LMF {@code LexiconExtension} is an additive overlay on a base lexicon supplied in a
 * separate document. The overloads that take a {@link WnLmfResolver} compose extensions against
 * the bases the resolver supplies: new entries, senses, and synsets are added, declarations such
 * as {@code ExternalLexicalEntry}, {@code ExternalSense}, and {@code ExternalSynset} attach
 * additive content to base entities, and the composed result is an ordinary {@link WnLmfLexicon}
 * whose {@link WnLmfLexicon#extensionOf() extensionOf} carries the {@code Extends} reference.
 * Extension chains compose up to 16 levels, with cycles detected by exact id and version. The
 * overloads without a resolver perform no I/O of their own and reject
 * a {@code LexiconExtension} clearly. {@code ExternalLemma}, {@code Form}, and
 * {@code ExternalForm} stay outside the knowledge-base projection, exactly like the forms,
 * examples, counts, and secondary definitions the ordinary reader already skips; they never
 * affect lookup.</p>
 */
public final class WnLmfReader {

  /**
   * The maximum number of {@code LexiconExtension} levels one composition may stack: a document
   * whose base is itself an extension counts one level per extension. The bound exists so a
   * malicious or broken extension chain cannot recurse without limit; deeper chains fail loud
   * with an {@link InvalidFormatException} naming this bound.
   */
  private static final int MAX_EXTENSION_DEPTH = 16;

  /** The WN-LMF relation names this reader accepts, mapped to the contract relations. */
  private static final Map<String, WordNetRelation> RELATION_NAMES = relationNames();

  /** Relations declared only for SynsetRelation in WN-LMF 1.4. */
  private static final Set<String> SYNSET_ONLY_RELATIONS = Set.of(
      "attribute", "be_in_state", "causes", "classified_by", "classifies",
      "co_agent_instrument", "co_agent_patient", "co_agent_result", "co_instrument_agent",
      "co_instrument_patient", "co_instrument_result", "co_patient_agent",
      "co_patient_instrument", "co_result_agent", "co_result_instrument", "co_role",
      "direction", "entails", "eq_synonym", "holo_location", "holo_member", "holo_part",
      "holo_portion", "holo_substance", "holonym", "hypernym", "hyponym", "in_manner",
      "instance_hypernym", "instance_hyponym", "involved", "involved_agent",
      "involved_direction", "involved_instrument", "involved_location", "involved_patient",
      "involved_result", "involved_source_direction", "involved_target_direction",
      "ir_synonym", "is_caused_by", "is_entailed_by", "is_subevent_of", "manner_of",
      "mero_location", "mero_member", "mero_part", "mero_portion", "mero_substance",
      "meronym", "patient", "restricted_by", "restricts", "role", "source_direction",
      "state_of", "subevent", "target_direction");

  /** Relations declared only for SenseRelation in WN-LMF 1.4. */
  private static final Set<String> SENSE_ONLY_RELATIONS = Set.of(
      "body_part", "by_means_of", "derivation", "destination", "event", "has_metaphor",
      "has_metonym", "material", "metaphor", "metonym", "participle", "pertainym",
      "property", "secondary_aspect_ip", "secondary_aspect_pi", "simple_aspect_ip",
      "simple_aspect_pi", "state", "undergoer", "uses", "vehicle");

  /** The format's escape-hatch relation type; carries no type the contract can express. */
  private static final String OTHER_RELATION = "other";

  /** The element declaring a lexical entry; opened and closed by the same handlers. */
  private static final String LEXICAL_ENTRY_ELEMENT = "LexicalEntry";

  /** The element declaring a sense; opened and closed by the same handlers. */
  private static final String SENSE_ELEMENT = "Sense";

  /** The element declaring a synset; opened and closed by the same handlers. */
  private static final String SYNSET_ELEMENT = "Synset";

  /** The element declaring one independently queryable lexicon. */
  private static final String LEXICON_ELEMENT = "Lexicon";

  /** The element declaring an extension composed against a separately supplied base. */
  private static final String LEXICON_EXTENSION_ELEMENT = "LexiconExtension";

  /** The element declaring a lexicon dependency. */
  private static final String REQUIRES_ELEMENT = "Requires";

  /** The element declaring an extension's base lexicon. */
  private static final String EXTENDS_ELEMENT = "Extends";

  /** The element referencing a base lexical entry to attach additive content to. */
  private static final String EXTERNAL_LEXICAL_ENTRY_ELEMENT = "ExternalLexicalEntry";

  /** The element referencing a base sense to attach added relations to. */
  private static final String EXTERNAL_SENSE_ELEMENT = "ExternalSense";

  /** The element referencing a base synset to attach added content to. */
  private static final String EXTERNAL_SYNSET_ELEMENT = "ExternalSynset";

  /** The identifier attribute shared by entries, senses, and synsets. */
  private static final String ID_ATTRIBUTE = "id";

  /** The human-readable Lexicon label attribute. */
  private static final String LABEL_ATTRIBUTE = "label";

  /** The BCP 47 Lexicon language attribute. */
  private static final String LANGUAGE_ATTRIBUTE = "language";

  /** The Lexicon version attribute. */
  private static final String VERSION_ATTRIBUTE = "version";

  /** The required lexicon reference attribute. */
  private static final String REF_ATTRIBUTE = "ref";

  /** The optional retrieval-hint attribute on Requires and Extends. */
  private static final String URL_ATTRIBUTE = "url";

  /** The part-of-speech attribute shared by lemmas and synsets. */
  private static final String PART_OF_SPEECH_ATTRIBUTE = "partOfSpeech";

  /** The relation-type attribute shared by sense and synset relations. */
  private static final String REL_TYPE_ATTRIBUTE = "relType";

  /** The relation-target attribute shared by sense and synset relations. */
  private static final String TARGET_ATTRIBUTE = "target";

  /** The opening of every malformed-document message, before the resource name. */
  private static final String MALFORMED_PREFIX = "Malformed WN-LMF document ";

  /** Not instantiable. */
  private WnLmfReader() {
  }

  /**
   * Reads a WN-LMF XML file.
   *
   * @param file The XML file. Must not be {@code null} and must exist.
   * @return The loaded lexicon.
   * @throws IllegalArgumentException Thrown if {@code file} is {@code null} or missing.
   * @throws InvalidFormatException Thrown if the document is malformed, contains an unsupported
   *     {@code LexiconExtension}, or contains more than one lexicon. The message names the file
   *     and, where available, the line.
   * @throws IOException Thrown if reading the file fails.
   */
  public static LexicalKnowledgeBase read(Path file) throws IOException {
    final WnLmfResource resource = readResource(file);
    return onlyLexicon(resource, file.toString());
  }

  /**
   * Reads a WN-LMF XML file, composing a {@code LexiconExtension} against the base lexicons the
   * resolver supplies.
   *
   * @param file     The XML file. Must not be {@code null} and must exist.
   * @param resolver The resolver for {@code Extends} references. Must not be {@code null}.
   * @return The loaded lexicon; for an extension, the composed base-plus-extension graph.
   * @throws IllegalArgumentException Thrown if {@code file} is {@code null} or missing or
   *     {@code resolver} is {@code null}.
   * @throws InvalidFormatException Thrown if a document is malformed, composition is invalid,
   *     or the file contains more than one lexicon.
   * @throws IOException Thrown if reading the file fails or the resolver cannot supply a base.
   */
  public static LexicalKnowledgeBase read(Path file, WnLmfResolver resolver) throws IOException {
    final WnLmfResource resource = readResource(file, resolver);
    return onlyLexicon(resource, file.toString());
  }

  /**
   * Reads every lexicon in a WN-LMF XML file without merging their lookup indexes.
   *
   * @param file The XML file. Must not be {@code null} and must exist.
   * @return The lexical resource in document order.
   * @throws IllegalArgumentException Thrown if {@code file} is {@code null} or missing.
   * @throws InvalidFormatException Thrown if the document is malformed or contains an unsupported
   *     {@code LexiconExtension}.
   * @throws IOException Thrown if reading the file fails.
   */
  public static WnLmfResource readResource(Path file) throws IOException {
    return readResourceFromFile(file, null);
  }

  /**
   * Reads every lexicon in a WN-LMF XML file, composing each {@code LexiconExtension} against
   * the base lexicons the resolver supplies. Sibling extensions compose independently and are
   * never merged with each other.
   *
   * @param file     The XML file. Must not be {@code null} and must exist.
   * @param resolver The resolver for {@code Extends} references. Must not be {@code null}.
   * @return The lexical resource in document order, extensions composed.
   * @throws IllegalArgumentException Thrown if {@code file} is {@code null} or missing or
   *     {@code resolver} is {@code null}.
   * @throws InvalidFormatException Thrown if a document is malformed or composition is invalid.
   * @throws IOException Thrown if reading the file fails or the resolver cannot supply a base.
   */
  public static WnLmfResource readResource(Path file, WnLmfResolver resolver) throws IOException {
    if (resolver == null) {
      throw new IllegalArgumentException("Resolver must not be null");
    }
    return readResourceFromFile(file, resolver);
  }

  /**
   * Reads a WN-LMF XML document from a stream. The stream is not closed.
   *
   * @param in           The document stream. Must not be {@code null}.
   * @param resourceName The name used in error messages. Must not be {@code null}.
   * @return The loaded lexicon.
   * @throws IllegalArgumentException Thrown if an argument is {@code null}.
   * @throws InvalidFormatException Thrown if the document is malformed, contains an unsupported
   *     {@code LexiconExtension}, or contains more than one lexicon. The message names the
   *     resource and, where available, the line.
   * @throws IOException Thrown if reading the stream fails.
   */
  public static LexicalKnowledgeBase read(InputStream in, String resourceName) throws IOException {
    final WnLmfResource resource = readResource(in, resourceName);
    return onlyLexicon(resource, resourceName);
  }

  /**
   * Reads a WN-LMF XML document from a stream, composing a {@code LexiconExtension} against the
   * base lexicons the resolver supplies. The stream is not closed; resolver-supplied sources
   * are closed by this reader.
   *
   * @param in           The document stream. Must not be {@code null}.
   * @param resourceName The name used in error messages. Must not be {@code null}.
   * @param resolver     The resolver for {@code Extends} references. Must not be {@code null}.
   * @return The loaded lexicon; for an extension, the composed base-plus-extension graph.
   * @throws IllegalArgumentException Thrown if an argument is {@code null}.
   * @throws InvalidFormatException Thrown if a document is malformed, composition is invalid,
   *     or the document contains more than one lexicon.
   * @throws IOException Thrown if reading a stream fails or the resolver cannot supply a base.
   */
  public static LexicalKnowledgeBase read(InputStream in, String resourceName,
      WnLmfResolver resolver) throws IOException {
    final WnLmfResource resource = readResource(in, resourceName, resolver);
    return onlyLexicon(resource, resourceName);
  }

  /**
   * Reads every lexicon in a WN-LMF document without merging their lookup indexes. The stream is
   * not closed.
   *
   * @param in           The document stream. Must not be {@code null}.
   * @param resourceName The name used in error messages. Must not be {@code null}.
   * @return The lexical resource in document order.
   * @throws IllegalArgumentException Thrown if an argument is {@code null}.
   * @throws InvalidFormatException Thrown if the document is malformed or contains an unsupported
   *     {@code LexiconExtension}.
   * @throws IOException Thrown if reading the stream fails.
   */
  public static WnLmfResource readResource(InputStream in, String resourceName)
      throws IOException {
    return readResourceInternal(in, resourceName, null);
  }

  /**
   * Reads every lexicon in a WN-LMF document, composing each {@code LexiconExtension} against
   * the base lexicons the resolver supplies. The stream is not closed; resolver-supplied
   * sources are closed by this reader. Sibling extensions compose independently and are never
   * merged with each other; resolver calls are cached per exact id and version for the duration
   * of this read.
   *
   * @param in           The document stream. Must not be {@code null}.
   * @param resourceName The name used in error messages. Must not be {@code null}.
   * @param resolver     The resolver for {@code Extends} references. Must not be {@code null}.
   * @return The lexical resource in document order, extensions composed.
   * @throws IllegalArgumentException Thrown if an argument is {@code null}.
   * @throws InvalidFormatException Thrown if a document is malformed or composition is invalid.
   * @throws IOException Thrown if reading a stream fails or the resolver cannot supply a base.
   */
  public static WnLmfResource readResource(InputStream in, String resourceName,
      WnLmfResolver resolver) throws IOException {
    if (resolver == null) {
      throw new IllegalArgumentException("Resolver must not be null");
    }
    return readResourceInternal(in, resourceName, resolver);
  }

  /**
   * Opens a file and reads its resource with an optional resolver.
   *
   * @param file     The XML file.
   * @param resolver The resolver, or {@code null} for the no-resolver contract; the public
   *                 resolver-aware overloads validated their argument already.
   * @return The lexical resource.
   * @throws IOException Thrown if reading fails.
   */
  private static WnLmfResource readResourceFromFile(Path file, WnLmfResolver resolver)
      throws IOException {
    if (file == null) {
      throw new IllegalArgumentException("File must not be null");
    }
    if (!Files.isRegularFile(file)) {
      throw new IllegalArgumentException("File does not exist or is not a regular file: " + file);
    }
    try (InputStream in = new BufferedInputStream(Files.newInputStream(file))) {
      return readResourceInternal(in, file.toString(), resolver);
    }
  }

  /**
   * Parses one document into the raw model, then materializes every lexicon, composing
   * extensions when a resolver is present.
   *
   * @param in           The document stream; not closed.
   * @param resourceName The name used in error messages.
   * @param resolver     The resolver, or {@code null} to reject extensions.
   * @return The lexical resource in document order.
   * @throws InvalidFormatException Thrown if a document is malformed or composition is invalid.
   * @throws IOException Thrown if reading a stream fails or the resolver cannot supply a base.
   */
  private static WnLmfResource readResourceInternal(InputStream in, String resourceName,
      WnLmfResolver resolver) throws IOException {
    if (in == null) {
      throw new IllegalArgumentException("In must not be null");
    }
    if (resourceName == null) {
      throw new IllegalArgumentException("ResourceName must not be null");
    }
    final RawResource raw = parseRaw(in, resourceName, resolver != null);
    rejectSameDocumentBase(raw, resourceName);
    final Composition composition =
        resolver == null ? null : new Composition(resolver, resourceName);
    final List<WnLmfLexicon> lexicons = new ArrayList<>(raw.lexicons().size());
    for (final RawLexicon lexicon : raw.lexicons()) {
      if (lexicon.kind() == Kind.LEXICON) {
        lexicons.add(descriptor(lexicon, resourceName));
      } else {
        // A null composition cannot reach here: without a resolver the parser rejects a
        // LexiconExtension, so no raw lexicon of that kind exists.
        lexicons.add(descriptor(composition.compose(lexicon), resourceName));
      }
    }
    return new WnLmfResource(lexicons);
  }

  /**
   * Materializes a raw lexicon into its public descriptor.
   *
   * @param lexicon      The raw lexicon; for an extension, already composed.
   * @param resourceName The name used in error messages.
   * @return The public lexicon descriptor.
   * @throws InvalidFormatException Thrown if the lexicon content is invalid.
   */
  private static WnLmfLexicon descriptor(RawLexicon lexicon, String resourceName)
      throws InvalidFormatException {
    return new WnLmfLexicon(lexicon.id(), lexicon.label(), lexicon.language(),
        lexicon.version(), lexicon.metadata(), lexicon.requires(),
        Optional.ofNullable(lexicon.extendsRef()),
        new Materializer(lexicon, resourceName).materialize());
  }

  /**
   * Rejects a document that contains both an extension and the base it extends: the format
   * forbids that shape, because external ids would be ambiguous.
   *
   * @param resource     The parsed document.
   * @param resourceName The name used in error messages.
   * @throws InvalidFormatException Thrown if an extension's base shares the document.
   */
  private static void rejectSameDocumentBase(RawResource resource, String resourceName)
      throws InvalidFormatException {
    for (final RawLexicon extension : resource.lexicons()) {
      if (extension.kind() != Kind.EXTENSION) {
        continue;
      }
      for (final RawLexicon other : resource.lexicons()) {
        if (other != extension
            && other.id().equals(extension.extendsRef().ref())
            && other.version().equals(extension.extendsRef().version())) {
          throw malformed(resourceName, -1, "LexiconExtension " + extension.id() + " extends "
              + other.id() + " version " + other.version() + ", which occurs in the same"
              + " document; a base lexicon may not share a file with its extension", null);
        }
      }
    }
  }

  /**
   * Streams one document into the raw model.
   *
   * @param in              The document stream; not closed.
   * @param resourceName    The name used in error messages.
   * @param allowExtensions Whether a {@code LexiconExtension} may be parsed; without a resolver
   *                        it is rejected instead.
   * @return The raw resource in document order.
   * @throws InvalidFormatException Thrown if the document is malformed.
   * @throws IOException Thrown if reading the stream fails.
   */
  private static RawResource parseRaw(InputStream in, String resourceName,
      boolean allowExtensions) throws IOException {
    final Parser parser = new Parser(resourceName, allowExtensions);
    try {
      final XMLStreamReader reader = hardenedFactory().createXMLStreamReader(in);
      try {
        parser.parse(reader);
      } finally {
        reader.close();
      }
    } catch (XMLStreamException e) {
      // StAX wraps a failing stream read in an XMLStreamException; surface it as the I/O failure.
      final Throwable nested = e.getNestedException() == null ? e.getCause()
          : e.getNestedException();
      if (nested instanceof IOException io) {
        throw io;
      }
      throw parser.malformedAt(e.getLocation(), "XML error: " + e.getMessage(), e);
    }
    return parser.resource();
  }

  /**
   * Returns the only knowledge base in a resource.
   *
   * @param resource     The parsed resource.
   * @param resourceName The resource name used in the rejection message.
   * @return The resource's only knowledge base.
   * @throws InvalidFormatException Thrown if the resource contains more than one lexicon.
   */
  private static LexicalKnowledgeBase onlyLexicon(WnLmfResource resource, String resourceName)
      throws InvalidFormatException {
    if (resource.lexicons().size() != 1) {
      throw new InvalidFormatException("WN-LMF resource " + resourceName + " contains "
          + resource.lexicons().size() + " lexicons; use WnLmfReader.readResource to preserve "
          + "their boundaries");
    }
    return resource.lexicons().get(0).knowledgeBase();
  }

  /**
   * Builds an XXE-hardened StAX factory: the DTD internal subset is not processed and external
   * entities and the external DTD subset are denied, so a DOCTYPE is skipped but never resolved.
   *
   * @return The hardened factory.
   */
  private static XMLInputFactory hardenedFactory() {
    final XMLInputFactory factory = XMLInputFactory.newFactory();
    factory.setProperty(XMLInputFactory.SUPPORT_DTD, Boolean.FALSE);
    factory.setProperty(XMLInputFactory.IS_SUPPORTING_EXTERNAL_ENTITIES, Boolean.FALSE);
    factory.setProperty(XMLConstants.ACCESS_EXTERNAL_DTD, "");
    factory.setProperty(XMLInputFactory.IS_COALESCING, Boolean.TRUE);
    factory.setXMLResolver((publicId, systemId, baseUri, namespace) -> {
      throw new XMLStreamException("External entity resolution is disabled, refusing " + systemId);
    });
    return factory;
  }

  /**
   * Builds a malformed-document exception naming the resource and, when known, the line.
   *
   * @param resourceName The resource name.
   * @param line         The line number, or {@code -1} when unknown.
   * @param message      The failure detail.
   * @param cause        The underlying cause, or {@code null}.
   * @return The exception to throw.
   */
  private static InvalidFormatException malformed(String resourceName, int line, String message,
      Throwable cause) {
    final String prefix = line < 0 ? MALFORMED_PREFIX + resourceName + ": "
        : MALFORMED_PREFIX + resourceName + " at line " + line + ": ";
    return cause == null ? new InvalidFormatException(prefix + message)
        : new InvalidFormatException(prefix + message, cause);
  }

  /**
   * Extracts a line number from a parser location.
   *
   * @param location The location, or {@code null}.
   * @return The line number, or {@code -1} when unknown.
   */
  private static int line(Location location) {
    return location == null ? -1 : location.getLineNumber();
  }

  /** Streams one document into raw lexicons; performs no cross-lexicon resolution. */
  private static final class Parser {

    private final String resourceName;
    private final boolean allowExtensions;
    private final List<RawLexicon> lexicons = new ArrayList<>();
    private final Set<String> lexiconIds = new HashSet<>();
    private final Set<String> documentIds = new HashSet<>();

    // Current Lexicon or LexiconExtension identity.
    private Kind currentKind;
    private String currentLexiconId;
    private String currentLexiconLabel;
    private String currentLexiconLanguage;
    private String currentLexiconVersion;
    private Map<QName, String> currentLexiconMetadata;
    private int currentLexiconLine;
    private final List<WnLmfDependency> currentRequires = new ArrayList<>();
    private WnLmfDependency currentExtends;

    // Current lexicon content, in document order. Duplicate ids of every kind are caught by
    // the document-wide id claim, whose messages name the element kind.
    private final Map<String, RawEntry> entries = new LinkedHashMap<>();
    private final Map<String, RawSense> senses = new LinkedHashMap<>();
    private final Map<String, RawSynset> synsets = new LinkedHashMap<>();
    private final List<RawExternalEntry> externalEntries = new ArrayList<>();
    private final List<RawExternalSynset> externalSynsets = new ArrayList<>();

    // Cursor state.
    private String currentEntryId;
    private String currentEntryLemma;
    private WordNetPOS currentEntryPos;
    private int currentEntryLine;
    private String currentSenseId;
    private String currentSenseEntryId;
    private String currentSenseSynset;
    private int currentSenseLine;
    private final List<RawSenseRelation> currentSenseRelations = new ArrayList<>();
    private SynsetBuilder currentSynset;
    private ExternalEntryBuilder currentExternalEntry;
    private ExternalSenseBuilder currentExternalSense;
    private ExternalSynsetBuilder currentExternalSynset;

    /**
     * Creates a parser.
     *
     * @param resourceName    The name used in error messages.
     * @param allowExtensions Whether a {@code LexiconExtension} may be parsed.
     */
    Parser(String resourceName, boolean allowExtensions) {
      this.resourceName = resourceName;
      this.allowExtensions = allowExtensions;
    }

    /**
     * Streams the document, dispatching start and end elements.
     *
     * @param reader The StAX reader.
     * @throws XMLStreamException Thrown if the stream read fails.
     * @throws InvalidFormatException Thrown if the document is malformed.
     */
    void parse(XMLStreamReader reader) throws XMLStreamException, InvalidFormatException {
      while (reader.hasNext()) {
        final int event = reader.next();
        // A DTD event carries nothing that can affect parsing once the factory is hardened.
        if (event == XMLStreamConstants.START_ELEMENT) {
          startElement(reader);
        } else if (event == XMLStreamConstants.END_ELEMENT) {
          endElement(reader.getLocalName());
        }
      }
      if (currentLexiconId != null) {
        throw malformedAt(reader.getLocation(), "Unclosed Lexicon " + currentLexiconId, null);
      }
      if (lexicons.isEmpty()) {
        throw malformedAt(reader.getLocation(), "Document contains no Lexicon", null);
      }
    }

    /**
     * Handles one start element, updating cursor state and collecting raw content.
     *
     * @param reader The StAX reader positioned on the start element.
     * @throws XMLStreamException Thrown if reading element text fails.
     * @throws InvalidFormatException Thrown if the element violates the format.
     */
    private void startElement(XMLStreamReader reader)
        throws XMLStreamException, InvalidFormatException {
      final String name = reader.getLocalName();
      switch (name) {
        case LEXICON_ELEMENT -> openLexicon(reader, Kind.LEXICON);
        case LEXICON_EXTENSION_ELEMENT -> {
          if (!allowExtensions) {
            throw malformedAt(reader.getLocation(), "LexiconExtension is not supported without"
                + " a resolver; use a read or readResource overload that takes a WnLmfResolver",
                null);
          }
          openLexicon(reader, Kind.EXTENSION);
        }
        case EXTENDS_ELEMENT -> {
          if (currentKind != Kind.EXTENSION) {
            throw malformedAt(reader.getLocation(), "Extends outside a LexiconExtension", null);
          }
          if (currentExtends != null) {
            throw malformedAt(reader.getLocation(),
                "Repeated Extends in LexiconExtension " + currentLexiconId, null);
          }
          currentExtends = dependency(reader);
        }
        case REQUIRES_ELEMENT -> {
          requireLexicon(reader, REQUIRES_ELEMENT);
          currentRequires.add(dependency(reader));
        }
        case LEXICAL_ENTRY_ELEMENT -> {
          requireLexicon(reader, LEXICAL_ENTRY_ELEMENT);
          currentEntryId = requireAttribute(reader, ID_ATTRIBUTE);
          claimDocumentId(currentEntryId, "lexical entry", reader.getLocation());
          currentEntryLemma = null;
          currentEntryPos = null;
          currentEntryLine = line(reader.getLocation());
        }
        case EXTERNAL_LEXICAL_ENTRY_ELEMENT -> {
          requireExtension(reader, EXTERNAL_LEXICAL_ENTRY_ELEMENT);
          final String id = requireAttribute(reader, ID_ATTRIBUTE);
          claimDocumentId(id, "external lexical entry", reader.getLocation());
          currentExternalEntry = new ExternalEntryBuilder(id, line(reader.getLocation()));
        }
        case "Lemma" -> {
          if (currentEntryId == null) {
            throw malformedAt(reader.getLocation(), "Lemma outside a LexicalEntry", null);
          }
          currentEntryLemma = requireAttribute(reader, "writtenForm");
          currentEntryPos = parsePos(requireAttribute(reader, PART_OF_SPEECH_ATTRIBUTE),
              reader.getLocation());
        }
        case SENSE_ELEMENT -> {
          final String owner;
          if (currentExternalEntry != null) {
            owner = currentExternalEntry.id;
          } else {
            if (currentEntryLemma == null) {
              throw malformedAt(reader.getLocation(),
                  "Sense before its entry's Lemma in LexicalEntry " + currentEntryId, null);
            }
            owner = currentEntryId;
          }
          final String id = requireAttribute(reader, ID_ATTRIBUTE);
          final String synsetId = requireAttribute(reader, "synset");
          claimDocumentId(id, "sense", reader.getLocation());
          currentSenseId = id;
          currentSenseEntryId = owner;
          currentSenseSynset = synsetId;
          currentSenseLine = line(reader.getLocation());
          currentSenseRelations.clear();
        }
        case EXTERNAL_SENSE_ELEMENT -> {
          if (currentExternalEntry == null) {
            throw malformedAt(reader.getLocation(),
                "ExternalSense outside an ExternalLexicalEntry", null);
          }
          final String id = requireAttribute(reader, ID_ATTRIBUTE);
          claimDocumentId(id, "external sense", reader.getLocation());
          currentExternalSense = new ExternalSenseBuilder(id, line(reader.getLocation()));
        }
        case "SenseRelation" -> {
          if (currentExternalSense == null && currentSenseId == null) {
            throw malformedAt(reader.getLocation(), "SenseRelation outside a Sense", null);
          }
          final RawSenseRelation relation = new RawSenseRelation(
              requireAttribute(reader, REL_TYPE_ATTRIBUTE),
              requireAttribute(reader, TARGET_ATTRIBUTE), line(reader.getLocation()));
          if (currentExternalSense != null) {
            currentExternalSense.relations.add(relation);
          } else {
            currentSenseRelations.add(relation);
          }
        }
        case SYNSET_ELEMENT -> {
          requireLexicon(reader, SYNSET_ELEMENT);
          final String id = requireAttribute(reader, ID_ATTRIBUTE);
          final WordNetPOS pos = parsePos(requireAttribute(reader, PART_OF_SPEECH_ATTRIBUTE),
              reader.getLocation());
          claimDocumentId(id, "synset", reader.getLocation());
          currentSynset = new SynsetBuilder(id, pos,
              reader.getAttributeValue(null, "members"), line(reader.getLocation()));
        }
        case EXTERNAL_SYNSET_ELEMENT -> {
          requireExtension(reader, EXTERNAL_SYNSET_ELEMENT);
          final String id = requireAttribute(reader, ID_ATTRIBUTE);
          claimDocumentId(id, "external synset", reader.getLocation());
          currentExternalSynset = new ExternalSynsetBuilder(id, line(reader.getLocation()));
        }
        case "Definition" -> {
          if (currentSynset != null && currentSynset.gloss == null) {
            currentSynset.gloss = reader.getElementText();
          } else if (currentExternalSynset != null && currentExternalSynset.definition == null) {
            currentExternalSynset.definition = reader.getElementText();
          }
        }
        case "SynsetRelation" -> {
          if (currentSynset == null && currentExternalSynset == null) {
            throw malformedAt(reader.getLocation(), "SynsetRelation outside a Synset", null);
          }
          final String relType = requireAttribute(reader, REL_TYPE_ATTRIBUTE);
          final String target = requireAttribute(reader, TARGET_ATTRIBUTE);
          // The escape-hatch type is a documented skip, not a rejection.
          if (!OTHER_RELATION.equals(relType)) {
            final RawRelation relation =
                new RawRelation(relType, target, line(reader.getLocation()));
            if (currentSynset != null) {
              currentSynset.relations.add(relation);
            } else {
              currentExternalSynset.relations.add(relation);
            }
          }
        }
        default -> {
          // Pronunciation, Form, ExternalForm, ExternalLemma, Example, SyntacticBehaviour,
          // ILIDefinition, and other elements outside the contract subset are skipped.
        }
      }
    }

    /**
     * Builds raw records and clears cursor state when a tracked element closes.
     *
     * @param name The local name of the closing element.
     * @throws InvalidFormatException Thrown if closing a lexicon exposes invalid content.
     */
    private void endElement(String name) throws InvalidFormatException {
      switch (name) {
        case LEXICON_ELEMENT, LEXICON_EXTENSION_ELEMENT -> closeLexicon();
        case LEXICAL_ENTRY_ELEMENT -> {
          if (currentEntryLemma != null) {
            entries.put(currentEntryId, new RawEntry(currentEntryId, currentEntryLemma,
                currentEntryPos, currentEntryLine));
          }
          currentEntryId = null;
          currentEntryLemma = null;
          currentEntryPos = null;
        }
        case SENSE_ELEMENT -> {
          if (currentSenseId != null) {
            senses.put(currentSenseId, new RawSense(currentSenseId, currentSenseEntryId,
                currentSenseSynset, currentSenseRelations, currentSenseLine));
          }
          currentSenseId = null;
          currentSenseEntryId = null;
          currentSenseSynset = null;
          currentSenseRelations.clear();
        }
        case EXTERNAL_SENSE_ELEMENT -> {
          if (currentExternalSense != null) {
            currentExternalEntry.externalSenses.add(new RawExternalSense(
                currentExternalSense.id, currentExternalSense.relations,
                currentExternalSense.line));
          }
          currentExternalSense = null;
        }
        case EXTERNAL_LEXICAL_ENTRY_ELEMENT -> {
          if (currentExternalEntry != null) {
            externalEntries.add(new RawExternalEntry(currentExternalEntry.id,
                currentExternalEntry.externalSenses, currentExternalEntry.line));
          }
          currentExternalEntry = null;
        }
        case SYNSET_ELEMENT -> {
          if (currentSynset != null) {
            synsets.put(currentSynset.id, new RawSynset(currentSynset.id, currentSynset.pos,
                currentSynset.members, List.of(), currentSynset.gloss,
                currentSynset.relations, currentSynset.line));
          }
          currentSynset = null;
        }
        case EXTERNAL_SYNSET_ELEMENT -> {
          if (currentExternalSynset != null) {
            externalSynsets.add(new RawExternalSynset(currentExternalSynset.id,
                currentExternalSynset.definition, currentExternalSynset.relations,
                currentExternalSynset.line));
          }
          currentExternalSynset = null;
        }
        default -> {
          // Nothing to close for skipped elements.
        }
      }
    }

    /**
     * Opens one Lexicon or LexiconExtension and captures its identity and metadata.
     *
     * @param reader The reader positioned on the start element.
     * @param kind   The lexicon form being opened.
     * @throws InvalidFormatException Thrown if the element is nested, repeats an id, or omits a
     *     required attribute.
     */
    private void openLexicon(XMLStreamReader reader, Kind kind) throws InvalidFormatException {
      if (currentLexiconId != null) {
        throw malformedAt(reader.getLocation(),
            "Nested " + reader.getLocalName() + " inside " + currentLexiconId, null);
      }
      final String id = requireAttribute(reader, ID_ATTRIBUTE);
      if (!lexiconIds.add(id)) {
        throw malformedAt(reader.getLocation(), "Duplicate lexicon id " + id, null);
      }
      claimDocumentId(id, "lexicon", reader.getLocation());
      currentKind = kind;
      currentLexiconId = id;
      currentLexiconLabel = requireAttribute(reader, LABEL_ATTRIBUTE);
      currentLexiconLanguage = requireAttribute(reader, LANGUAGE_ATTRIBUTE);
      currentLexiconVersion = requireAttribute(reader, VERSION_ATTRIBUTE);
      currentLexiconLine = line(reader.getLocation());
      final Map<QName, String> metadata = new LinkedHashMap<>();
      for (int i = 0; i < reader.getAttributeCount(); i++) {
        final QName attribute = reader.getAttributeName(i);
        if (!isIdentityAttribute(attribute)) {
          metadata.put(attribute, reader.getAttributeValue(i));
        }
      }
      currentLexiconMetadata = Map.copyOf(metadata);
    }

    /**
     * Builds the current lexicon's raw record and clears its parse state.
     *
     * @throws InvalidFormatException Thrown if the lexicon content is invalid.
     */
    private void closeLexicon() throws InvalidFormatException {
      if (currentLexiconId == null) {
        return;
      }
      if (currentKind == Kind.EXTENSION && currentExtends == null) {
        throw malformedAt(null,
            "LexiconExtension " + currentLexiconId + " is missing its Extends element", null);
      }
      try {
        lexicons.add(new RawLexicon(currentKind, currentLexiconId, currentLexiconLabel,
            currentLexiconLanguage, currentLexiconVersion, currentLexiconMetadata,
            currentRequires, currentExtends, entries, senses, synsets, externalEntries,
            externalSynsets, currentLexiconLine));
      } finally {
        clearLexiconState();
      }
    }

    /**
     * Requires lexical content to be enclosed by a Lexicon or LexiconExtension element.
     *
     * @param reader  The reader positioned on the content element.
     * @param element The element name used in the rejection message.
     * @throws InvalidFormatException Thrown if no lexicon is open.
     */
    private void requireLexicon(XMLStreamReader reader, String element)
        throws InvalidFormatException {
      if (currentLexiconId == null) {
        throw malformedAt(reader.getLocation(), element + " outside a Lexicon", null);
      }
    }

    /**
     * Requires an external declaration to be enclosed by a LexiconExtension.
     *
     * @param reader  The reader positioned on the declaration element.
     * @param element The element name used in the rejection message.
     * @throws InvalidFormatException Thrown if no extension is open.
     */
    private void requireExtension(XMLStreamReader reader, String element)
        throws InvalidFormatException {
      if (currentKind != Kind.EXTENSION) {
        throw malformedAt(reader.getLocation(),
            element + " outside a LexiconExtension", null);
      }
    }

    /**
     * Reads a Requires or Extends element into a dependency descriptor.
     *
     * @param reader The reader positioned on the element.
     * @return The dependency, with its url when the source declared a non-empty one.
     * @throws InvalidFormatException Thrown if ref or version is absent or empty.
     */
    private WnLmfDependency dependency(XMLStreamReader reader) throws InvalidFormatException {
      final String url = reader.getAttributeValue(null, URL_ATTRIBUTE);
      return new WnLmfDependency(
          requireAttribute(reader, REF_ATTRIBUTE),
          requireAttribute(reader, VERSION_ATTRIBUTE),
          Optional.ofNullable(url).filter(value -> !value.isEmpty()));
    }

    /**
     * Claims an XML ID across the complete LexicalResource.
     *
     * @param id       The identifier to claim.
     * @param kind     The element kind used in a duplicate error.
     * @param location The source location.
     * @throws InvalidFormatException Thrown if another parsed element already carries the id.
     */
    private void claimDocumentId(String id, String kind, Location location)
        throws InvalidFormatException {
      if (!documentIds.add(id)) {
        throw malformedAt(location, "Duplicate " + kind + " id " + id, null);
      }
    }

    /**
     * Tests whether an attribute is exposed directly on {@link WnLmfLexicon}.
     *
     * @param attribute The attribute name.
     * @return {@code true} for an unqualified id, label, language, or version attribute.
     */
    private boolean isIdentityAttribute(QName attribute) {
      if (!attribute.getNamespaceURI().isEmpty()) {
        return false;
      }
      return switch (attribute.getLocalPart()) {
        case ID_ATTRIBUTE, LABEL_ATTRIBUTE, LANGUAGE_ATTRIBUTE, VERSION_ATTRIBUTE -> true;
        default -> false;
      };
    }

    /** Clears every field whose scope is one Lexicon or LexiconExtension element. */
    private void clearLexiconState() {
      currentKind = null;
      currentLexiconId = null;
      currentLexiconLabel = null;
      currentLexiconLanguage = null;
      currentLexiconVersion = null;
      currentLexiconMetadata = null;
      currentRequires.clear();
      currentExtends = null;
      entries.clear();
      senses.clear();
      synsets.clear();
      externalEntries.clear();
      externalSynsets.clear();
      currentEntryId = null;
      currentEntryLemma = null;
      currentEntryPos = null;
      currentSenseId = null;
      currentSenseEntryId = null;
      currentSenseSynset = null;
      currentSenseRelations.clear();
      currentSynset = null;
      currentExternalEntry = null;
      currentExternalSense = null;
      currentExternalSynset = null;
    }

    /**
     * Returns the parsed raw resource.
     *
     * @return The immutable raw resource in document order.
     */
    RawResource resource() {
      return new RawResource(lexicons);
    }

    /**
     * Maps a WN-LMF part-of-speech code to a {@link WordNetPOS}; code {@code s} normalizes to
     * {@link WordNetPOS#ADJECTIVE}.
     *
     * @param code     The part-of-speech code.
     * @param location The parser location, for error reporting.
     * @return The part of speech.
     * @throws InvalidFormatException Thrown if the code is unknown.
     */
    private WordNetPOS parsePos(String code, Location location) throws InvalidFormatException {
      return switch (code) {
        case "n" -> WordNetPOS.NOUN;
        case "v" -> WordNetPOS.VERB;
        case "a", "s" -> WordNetPOS.ADJECTIVE;
        case "r" -> WordNetPOS.ADVERB;
        default -> throw malformedAt(location, "Unknown part-of-speech code: " + code, null);
      };
    }

    /**
     * Reads a required attribute from the current element.
     *
     * @param reader    The StAX reader.
     * @param attribute The attribute name.
     * @return The non-empty attribute value.
     * @throws InvalidFormatException Thrown if the attribute is absent or empty.
     */
    private String requireAttribute(XMLStreamReader reader, String attribute)
        throws InvalidFormatException {
      final String value = reader.getAttributeValue(null, attribute);
      if (value == null || value.isEmpty()) {
        throw malformedAt(reader.getLocation(), "Element " + reader.getLocalName()
            + " is missing required attribute " + attribute, null);
      }
      return value;
    }

    /**
     * Builds a malformed-document exception naming this parser's resource.
     *
     * @param location The parser location, or {@code null} when unavailable.
     * @param message  The failure detail.
     * @param cause    The underlying cause, or {@code null}.
     * @return The exception to throw.
     */
    InvalidFormatException malformedAt(Location location, String message, Throwable cause) {
      return malformed(resourceName, line(location), message, cause);
    }

    /** Mutable Synset parse state, converted to a {@link RawSynset} when the element closes. */
    private static final class SynsetBuilder {
      private final String id;
      private final WordNetPOS pos;
      private final String members;
      private final int line;
      private final List<RawRelation> relations = new ArrayList<>(4);
      private String gloss;

      SynsetBuilder(String id, WordNetPOS pos, String members, int line) {
        this.id = id;
        this.pos = pos;
        this.members = members;
        this.line = line;
      }
    }

    /** Mutable ExternalLexicalEntry parse state. */
    private static final class ExternalEntryBuilder {
      private final String id;
      private final int line;
      private final List<RawExternalSense> externalSenses = new ArrayList<>(2);

      ExternalEntryBuilder(String id, int line) {
        this.id = id;
        this.line = line;
      }
    }

    /** Mutable ExternalSense parse state. */
    private static final class ExternalSenseBuilder {
      private final String id;
      private final int line;
      private final List<RawSenseRelation> relations = new ArrayList<>(2);

      ExternalSenseBuilder(String id, int line) {
        this.id = id;
        this.line = line;
      }
    }

    /** Mutable ExternalSynset parse state. */
    private static final class ExternalSynsetBuilder {
      private final String id;
      private final int line;
      private final List<RawRelation> relations = new ArrayList<>(2);
      private String definition;

      ExternalSynsetBuilder(String id, int line) {
        this.id = id;
        this.line = line;
      }
    }
  }

  /**
   * Materializes one raw lexicon (parsed or composed) into the immutable knowledge base:
   * validates sense targets, lifts sense relations to the synset level, and builds the contract
   * synsets with their member lemmas and typed relations.
   */
  private static final class Materializer {

    private final RawLexicon lexicon;
    private final String resourceName;

    /**
     * Creates a materializer.
     *
     * @param lexicon      The raw lexicon; for an extension, already composed.
     * @param resourceName The name used in error messages.
     */
    Materializer(RawLexicon lexicon, String resourceName) {
      this.lexicon = lexicon;
      this.resourceName = resourceName;
    }

    /**
     * Resolves the raw lexicon into an immutable knowledge base.
     *
     * @return The loaded lexicon.
     * @throws InvalidFormatException Thrown if a sense or relation references an undeclared
     *     target or a declared member is invalid.
     */
    LexicalKnowledgeBase materialize() throws InvalidFormatException {
      // Every sense must point to a declared synset; part-of-speech consistency between a
      // synset and its member entries is checked in memberLemmas.
      for (final RawSense sense : lexicon.senses().values()) {
        if (!lexicon.synsets().containsKey(sense.synsetId())) {
          throw fail("Sense " + sense.id() + " references undeclared synset "
              + sense.synsetId());
        }
      }
      // Lift sense relations to the synset level and index membership and sense order, all in
      // one pass over the senses in their materialization order.
      final Map<String, List<LiftableRelation>> liftedBySynset = new HashMap<>();
      final Map<InMemoryWordNetLexicon.LemmaKey, List<String>> senseOrder =
          new LinkedHashMap<>();
      final Map<String, List<String>> entryIdsBySynset = new HashMap<>();
      for (final RawSense sense : lexicon.senses().values()) {
        for (final RawSenseRelation relation : sense.relations()) {
          if (OTHER_RELATION.equals(relation.relType())) {
            continue;
          }
          final RawSense target = lexicon.senses().get(relation.target());
          if (target == null) {
            throw fail("SenseRelation at line " + relation.line() + " from sense "
                + sense.id() + " references undeclared sense " + relation.target());
          }
          liftedBySynset.computeIfAbsent(sense.synsetId(), unused -> new ArrayList<>(2))
              .add(new LiftableRelation(relation.relType(), target.synsetId(),
                  relation.line(), true));
        }
        final RawEntry entry = entry(sense);
        entryIdsBySynset.computeIfAbsent(sense.synsetId(), unused -> new ArrayList<>(2))
            .add(entry.id());
        final List<String> order = senseOrder.computeIfAbsent(
            InMemoryWordNetLexicon.LemmaKey.of(entry.lemma(), entry.pos()),
            unused -> new ArrayList<>(2));
        if (!order.contains(sense.synsetId())) {
          order.add(sense.synsetId());
        }
      }
      // Resolve raw synsets into contract synsets.
      final Map<String, Synset> synsetsById = new LinkedHashMap<>(lexicon.synsets().size() * 2);
      for (final RawSynset raw : lexicon.synsets().values()) {
        final Map<WordNetRelation, List<String>> relations = resolveRelations(raw,
            liftedBySynset.getOrDefault(raw.id(), List.of()));
        synsetsById.put(raw.id(),
            new Synset(raw.id(), raw.pos(), memberLemmas(raw, entryIdsBySynset),
                raw.gloss() == null ? "" : raw.gloss(), relations));
      }
      return new InMemoryWordNetLexicon(synsetsById, senseOrder);
    }

    /**
     * Resolves a sense's owning entry.
     *
     * @param sense The sense.
     * @return The owning entry.
     */
    private RawEntry entry(RawSense sense) {
      final RawEntry entry = lexicon.entries().get(sense.entryId());
      if (entry == null) {
        // Reachable only through a reader bug: composition validates external entries and the
        // parser binds every ordinary sense to its enclosing entry.
        throw new IllegalStateException("Sense " + sense.id()
            + " belongs to unresolved lexical entry " + sense.entryId());
      }
      return entry;
    }

    /**
     * Resolves a raw synset's own and lifted relations into typed target-id lists, deduplicated
     * in source order.
     *
     * @param raw    The raw synset.
     * @param lifted The sense relations lifted onto this synset, in sense order.
     * @return The typed relations for the contract synset.
     * @throws InvalidFormatException Thrown if a relation type is unknown or its target is
     *     undeclared.
     */
    private Map<WordNetRelation, List<String>> resolveRelations(RawSynset raw,
        List<LiftableRelation> lifted) throws InvalidFormatException {
      final List<LiftableRelation> combined =
          new ArrayList<>(raw.relations().size() + lifted.size());
      for (final RawRelation relation : raw.relations()) {
        combined.add(new LiftableRelation(relation.relType(), relation.target(),
            relation.line(), false));
      }
      combined.addAll(lifted);
      final Map<WordNetRelation, LinkedHashSet<String>> typed = new LinkedHashMap<>();
      for (final LiftableRelation relation : combined) {
        final WordNetRelation type = parseRelation(
            relation.relType(), raw.pos(), relation.line(), relation.senseRelation());
        final RawSynset target = lexicon.synsets().get(relation.target());
        if (target == null) {
          throw fail("Relation " + relation.relType() + " at line " + relation.line()
              + " on synset " + raw.id() + " references undeclared synset "
              + relation.target());
        }
        // Share the synset table's id instance so only one copy of each id is retained.
        typed.computeIfAbsent(type, unused -> new LinkedHashSet<>()).add(target.id());
      }
      final Map<WordNetRelation, List<String>> relations = new LinkedHashMap<>(typed.size() * 2);
      for (final Map.Entry<WordNetRelation, LinkedHashSet<String>> entry : typed.entrySet()) {
        relations.put(entry.getKey(), List.copyOf(entry.getValue()));
      }
      return relations;
    }

    /**
     * Resolves a synset's member sense ids to their entry lemmas, from the {@code members}
     * attribute when present and otherwise from the senses that pointed at the synset, then
     * appends the entry ids composition added. Legacy documents that put lexical-entry ids in
     * {@code members} remain accepted.
     *
     * @param raw               The raw synset.
     * @param entryIdsBySynset  The fallback membership index built from the senses.
     * @return The member lemmas in source order, deduplicated.
     * @throws InvalidFormatException Thrown if the synset names an undeclared member or a
     *     member's part of speech disagrees with the synset's.
     */
    private List<String> memberLemmas(RawSynset raw, Map<String, List<String>> entryIdsBySynset)
        throws InvalidFormatException {
      final List<String> memberIds = new ArrayList<>();
      if (raw.members() != null && !raw.members().isEmpty()) {
        memberIds.addAll(LemmaFolding.splitOnSpaces(raw.members()));
      } else {
        memberIds.addAll(entryIdsBySynset.getOrDefault(raw.id(), List.of()));
      }
      memberIds.addAll(raw.extraMembers());
      final List<String> lemmas = new ArrayList<>(memberIds.size());
      for (final String memberId : memberIds) {
        final RawSense sense = lexicon.senses().get(memberId);
        final String entryId = sense == null ? memberId : sense.entryId();
        final RawEntry entry = lexicon.entries().get(entryId);
        if (entry == null) {
          throw fail("Synset " + raw.id() + " at line " + raw.line()
              + " lists undeclared member sense or entry " + memberId);
        }
        if (raw.pos() != entry.pos()) {
          throw fail("Synset " + raw.id() + " at line " + raw.line()
              + " has part of speech " + raw.pos() + " but member entry " + entry.id()
              + " has " + entry.pos());
        }
        if (!lemmas.contains(entry.lemma())) {
          lemmas.add(entry.lemma());
        }
      }
      return lemmas;
    }

    /**
     * Maps a WN-LMF relation name to a {@link WordNetRelation}, enforcing whether the DTD permits
     * it on a SenseRelation or SynsetRelation. A {@code similar} relation on a verb synset maps to
     * {@link WordNetRelation#VERB_GROUP}, otherwise to {@link WordNetRelation#SIMILAR_TO}.
     *
     * @param relType   The relation name.
     * @param sourcePos The part of speech of the source synset.
     * @param line      The document line, for error reporting.
     * @param senseRelation Whether the relation originated on a Sense.
     * @return The mapped relation.
     * @throws InvalidFormatException Thrown if the relation name is unknown.
     */
    private WordNetRelation parseRelation(String relType, WordNetPOS sourcePos, int line,
                                          boolean senseRelation)
        throws InvalidFormatException {
      if (senseRelation && SYNSET_ONLY_RELATIONS.contains(relType)) {
        throw fail("Relation type " + relType
            + " is not legal on SenseRelation at line " + line);
      }
      if (!senseRelation && SENSE_ONLY_RELATIONS.contains(relType)) {
        throw fail("Relation type " + relType
            + " is not legal on SynsetRelation at line " + line);
      }
      if ("similar".equals(relType)) {
        return !senseRelation && sourcePos == WordNetPOS.VERB ? WordNetRelation.VERB_GROUP
            : WordNetRelation.SIMILAR_TO;
      }
      final WordNetRelation relation = RELATION_NAMES.get(relType);
      if (relation == null) {
        throw fail("Unknown relation type " + relType + " at line " + line);
      }
      return relation;
    }

    /**
     * Builds a materialization failure naming the resource.
     *
     * @param message The failure detail.
     * @return The exception to throw.
     */
    private InvalidFormatException fail(String message) {
      return malformed(resourceName, -1, message, null);
    }

    /** One relation ready for typing: a synset relation or a lifted sense relation. */
    private record LiftableRelation(String relType, String target, int line,
                                    boolean senseRelation) {
    }
  }

  /**
   * One top-level read's composition state: the resolver, its per-(id, version) cache, and the
   * resolution stack that bounds chains and detects cycles.
   */
  private static final class Composition {

    private final WnLmfResolver resolver;
    private final String resourceName;
    private final Map<ResolutionKey, RawLexicon> cache = new HashMap<>();
    private final LinkedHashSet<ResolutionKey> stack = new LinkedHashSet<>();

    /**
     * Creates the composition state for one top-level read.
     *
     * @param resolver     The caller-supplied resolver.
     * @param resourceName The top-level resource name used in error messages.
     */
    Composition(WnLmfResolver resolver, String resourceName) {
      this.resolver = resolver;
      this.resourceName = resourceName;
    }

    /**
     * Composes one extension against its resolved base, recursively composing base chains.
     *
     * @param extension The parsed extension.
     * @return The composed raw lexicon, carrying no external declarations.
     * @throws InvalidFormatException Thrown if composition is invalid, cyclic, or too deep.
     * @throws IOException Thrown if the resolver cannot supply a base.
     */
    RawLexicon compose(RawLexicon extension) throws IOException {
      final ResolutionKey key = new ResolutionKey(extension.id(), extension.version());
      if (stack.contains(key)) {
        throw malformed(resourceName, -1, "Extension dependency cycle: " + path(key), null);
      }
      if (stack.size() >= MAX_EXTENSION_DEPTH) {
        throw malformed(resourceName, -1, "Extension chain at " + key
            + " exceeds the maximum composition depth of " + MAX_EXTENSION_DEPTH, null);
      }
      stack.add(key);
      try {
        return merge(base(extension.extendsRef()), extension);
      } finally {
        stack.remove(key);
      }
    }

    /**
     * Resolves one base reference to a composed raw lexicon, caching per exact id and version.
     *
     * @param reference The {@code Extends} reference.
     * @return The base lexicon; if the resolved lexicon is itself an extension, its composed
     *         form.
     * @throws InvalidFormatException Thrown if the resolved document is malformed or lacks an
     *     exact id and version match.
     * @throws IOException Thrown if the resolver cannot supply the document.
     * @throws IllegalStateException Thrown if the resolver violates its contract by returning
     *     {@code null} or a consumed source.
     */
    private RawLexicon base(WnLmfDependency reference) throws IOException {
      final ResolutionKey key = new ResolutionKey(reference.ref(), reference.version());
      final RawLexicon cached = cache.get(key);
      if (cached != null) {
        return cached;
      }
      final WnLmfSource source = resolver.resolve(reference);
      if (source == null) {
        throw new IllegalStateException("Resolver returned null for lexicon " + reference.ref()
            + " version " + reference.version() + " instead of a source or an IOException");
      }
      final String sourceName = source.name();
      final RawResource resolved;
      try (source) {
        resolved = parseRaw(source.consume(), sourceName, true);
      }
      rejectSameDocumentBase(resolved, sourceName);
      RawLexicon match = null;
      for (final RawLexicon lexicon : resolved.lexicons()) {
        if (lexicon.id().equals(reference.ref())
            && lexicon.version().equals(reference.version())) {
          match = lexicon;
          break;
        }
      }
      if (match == null) {
        throw malformed(resourceName, -1, "Resolved resource " + sourceName
            + " does not contain lexicon " + reference.ref() + " version "
            + reference.version(), null);
      }
      final RawLexicon result = match.kind() == Kind.EXTENSION ? compose(match) : match;
      cache.put(key, result);
      return result;
    }

    /**
     * Additively merges one extension into its composed base. The base inputs stay untouched;
     * every changed entity is a fresh record.
     *
     * @param base      The composed base lexicon.
     * @param extension The parsed extension.
     * @return The composed raw lexicon carrying the extension's identity.
     * @throws InvalidFormatException Thrown if a validation rule fails.
     */
    private RawLexicon merge(RawLexicon base, RawLexicon extension)
        throws InvalidFormatException {
      for (final String id : extension.entries().keySet()) {
        requireNewId(base, extension, id, "lexical entry");
      }
      for (final String id : extension.senses().keySet()) {
        requireNewId(base, extension, id, "sense");
      }
      for (final String id : extension.synsets().keySet()) {
        requireNewId(base, extension, id, "synset");
      }

      // Resolve the declared external synsets against the base.
      final Map<String, RawExternalSynset> externalSynsets = new LinkedHashMap<>();
      for (final RawExternalSynset external : extension.externalSynsets()) {
        if (!base.synsets().containsKey(external.id())) {
          throw fail("ExternalSynset " + external.id() + " at line " + external.line()
              + " does not resolve to a synset in base " + base.id() + " version "
              + base.version() + kindHint(base, external.id()));
        }
        externalSynsets.put(external.id(), external);
      }

      // Resolve the declared external entries and senses against the base.
      final Set<String> declaredExternalSenses = new HashSet<>();
      final Map<String, List<RawSenseRelation>> addedSenseRelations = new LinkedHashMap<>();
      for (final RawExternalEntry external : extension.externalEntries()) {
        if (!base.entries().containsKey(external.id())) {
          throw fail("ExternalLexicalEntry " + external.id() + " at line " + external.line()
              + " does not resolve to a lexical entry in base " + base.id() + " version "
              + base.version() + kindHint(base, external.id()));
        }
        for (final RawExternalSense externalSense : external.externalSenses()) {
          final RawSense baseSense = base.senses().get(externalSense.id());
          if (baseSense == null) {
            throw fail("ExternalSense " + externalSense.id() + " at line "
                + externalSense.line() + " does not resolve to a sense in base " + base.id()
                + " version " + base.version() + kindHint(base, externalSense.id()));
          }
          if (!baseSense.entryId().equals(external.id())) {
            throw fail("ExternalSense " + externalSense.id() + " at line "
                + externalSense.line() + " is not a sense of lexical entry " + external.id()
                + "; it belongs to " + baseSense.entryId());
          }
          declaredExternalSenses.add(externalSense.id());
          addedSenseRelations
              .computeIfAbsent(externalSense.id(), unused -> new ArrayList<>(2))
              .addAll(externalSense.relations());
        }
      }

      // Every target of new content must be new content or explicitly declared external
      // content; silently reaching into the base would hide missing declarations.
      for (final RawSense sense : extension.senses().values()) {
        requireSynsetTarget(extension, externalSynsets, sense.synsetId(),
            "Sense " + sense.id(), sense.line());
        requireSenseTargets(extension, declaredExternalSenses, sense.id(), sense.relations());
      }
      for (final RawSynset synset : extension.synsets().values()) {
        for (final RawRelation relation : synset.relations()) {
          requireSynsetTarget(extension, externalSynsets, relation.target(),
              "Relation " + relation.relType() + " on synset " + synset.id(),
              relation.line());
        }
      }
      for (final RawExternalSynset external : extension.externalSynsets()) {
        for (final RawRelation relation : external.relations()) {
          requireSynsetTarget(extension, externalSynsets, relation.target(),
              "Relation " + relation.relType() + " on ExternalSynset " + external.id(),
              relation.line());
        }
      }
      for (final Map.Entry<String, List<RawSenseRelation>> added
          : addedSenseRelations.entrySet()) {
        requireSenseTargets(extension, declaredExternalSenses, added.getKey(),
            added.getValue());
      }

      // Merge entries: base first, new entries after, in extension source order.
      final Map<String, RawEntry> entries = new LinkedHashMap<>(base.entries());
      entries.putAll(extension.entries());

      // Merge senses: augmented base senses keep their position, new senses follow all base
      // senses so lookup returns base senses first.
      final Map<String, RawSense> senses = new LinkedHashMap<>(base.senses());
      for (final Map.Entry<String, List<RawSenseRelation>> added
          : addedSenseRelations.entrySet()) {
        final RawSense baseSense = base.senses().get(added.getKey());
        final List<RawSenseRelation> relations =
            new ArrayList<>(baseSense.relations().size() + added.getValue().size());
        relations.addAll(baseSense.relations());
        relations.addAll(added.getValue());
        senses.put(baseSense.id(), new RawSense(baseSense.id(), baseSense.entryId(),
            baseSense.synsetId(), relations, baseSense.line()));
      }
      senses.putAll(extension.senses());

      // New senses on declared external base synsets contribute their entry's lemma after the
      // base members.
      final Map<String, List<String>> extraMembers = new LinkedHashMap<>();
      for (final RawSense sense : extension.senses().values()) {
        if (externalSynsets.containsKey(sense.synsetId())) {
          extraMembers.computeIfAbsent(sense.synsetId(), unused -> new ArrayList<>(2))
              .add(sense.entryId());
        }
      }

      // Merge synsets: augmented base synsets keep their position with base-first relation
      // order; a base gloss wins and only an empty one is filled.
      final Map<String, RawSynset> synsets = new LinkedHashMap<>(base.synsets());
      for (final RawExternalSynset external : externalSynsets.values()) {
        final RawSynset baseSynset = base.synsets().get(external.id());
        final List<RawRelation> relations =
            new ArrayList<>(baseSynset.relations().size() + external.relations().size());
        relations.addAll(baseSynset.relations());
        relations.addAll(external.relations());
        final String gloss = baseSynset.gloss() == null || baseSynset.gloss().isEmpty()
            ? external.definition() == null ? baseSynset.gloss() : external.definition()
            : baseSynset.gloss();
        final List<String> extras = new ArrayList<>(baseSynset.extraMembers());
        extras.addAll(extraMembers.getOrDefault(external.id(), List.of()));
        synsets.put(baseSynset.id(), new RawSynset(baseSynset.id(), baseSynset.pos(),
            baseSynset.members(), extras, gloss, relations, baseSynset.line()));
      }
      synsets.putAll(extension.synsets());

      return new RawLexicon(Kind.EXTENSION, extension.id(), extension.label(),
          extension.language(), extension.version(), extension.metadata(),
          extension.requires(), extension.extendsRef(), entries, senses, synsets,
          List.of(), List.of(), extension.line());
    }

    /**
     * Rejects a new id that collides with any base entry, sense, or synset id.
     *
     * @param base      The composed base.
     * @param extension The extension introducing the id.
     * @param id        The new id.
     * @param kind      The element kind used in the rejection message.
     * @throws InvalidFormatException Thrown on a collision.
     */
    private void requireNewId(RawLexicon base, RawLexicon extension, String id, String kind)
        throws InvalidFormatException {
      if (base.entries().containsKey(id) || base.senses().containsKey(id)
          || base.synsets().containsKey(id)) {
        throw fail("New " + kind + " id " + id + " in extension " + extension.id()
            + " collides with content of base " + base.id() + " version " + base.version());
      }
    }

    /**
     * Requires a synset target to be new extension content or a declared external synset.
     *
     * @param extension       The extension under composition.
     * @param externalSynsets The declared external synsets by id.
     * @param target          The referenced synset id.
     * @param what            The referencing construct, for the rejection message.
     * @param line            The referencing line.
     * @throws InvalidFormatException Thrown if the target is neither.
     */
    private void requireSynsetTarget(RawLexicon extension,
        Map<String, RawExternalSynset> externalSynsets, String target, String what, int line)
        throws InvalidFormatException {
      if (!extension.synsets().containsKey(target) && !externalSynsets.containsKey(target)) {
        throw fail(what + " at line " + line + " references synset " + target
            + ", which is neither declared in the extension nor as an ExternalSynset");
      }
    }

    /**
     * Requires every sense-relation target to be a new extension sense or a declared external
     * sense. The format's {@code other} escape hatch is exempt, exactly as it is for ordinary
     * lexicons.
     *
     * @param extension              The extension under composition.
     * @param declaredExternalSenses The declared external sense ids.
     * @param sourceId               The relation source, for the rejection message.
     * @param relations              The relations to check.
     * @throws InvalidFormatException Thrown if a target is neither.
     */
    private void requireSenseTargets(RawLexicon extension, Set<String> declaredExternalSenses,
        String sourceId, List<RawSenseRelation> relations) throws InvalidFormatException {
      for (final RawSenseRelation relation : relations) {
        if (OTHER_RELATION.equals(relation.relType())) {
          continue;
        }
        if (!extension.senses().containsKey(relation.target())
            && !declaredExternalSenses.contains(relation.target())) {
          throw fail("SenseRelation " + relation.relType() + " at line " + relation.line()
              + " from sense " + sourceId + " references sense " + relation.target()
              + ", which is neither declared in the extension nor as an ExternalSense");
        }
      }
    }

    /**
     * Names the entity kind a misdeclared external id actually resolves to, when it resolves
     * at all.
     *
     * @param base The composed base.
     * @param id   The external id.
     * @return A clause naming the actual kind, or an empty string.
     */
    private String kindHint(RawLexicon base, String id) {
      if (base.entries().containsKey(id)) {
        return "; it names a lexical entry";
      }
      if (base.senses().containsKey(id)) {
        return "; it names a sense";
      }
      if (base.synsets().containsKey(id)) {
        return "; it names a synset";
      }
      return "";
    }

    /**
     * Renders the resolution path for a cycle message.
     *
     * @param repeated The key that closed the cycle.
     * @return The path from the outermost extension to the repeated key.
     */
    private String path(ResolutionKey repeated) {
      final StringBuilder path = new StringBuilder();
      for (final ResolutionKey key : stack) {
        path.append(key).append(" -> ");
      }
      return path.append(repeated).toString();
    }

    /**
     * Builds a composition failure naming the top-level resource.
     *
     * @param message The failure detail.
     * @return The exception to throw.
     */
    private InvalidFormatException fail(String message) {
      return malformed(resourceName, -1, message, null);
    }

    /** One exact base identity: the caching and cycle-detection unit. */
    private record ResolutionKey(String ref, String version) {

      @Override
      public String toString() {
        return ref + " " + version;
      }
    }
  }

  /**
   * Builds the WN-LMF relation-name to {@link WordNetRelation} table.
   *
   * @return The immutable name table.
   */
  private static Map<String, WordNetRelation> relationNames() {
    final Map<String, WordNetRelation> names = new HashMap<>();
    names.put("agent", WordNetRelation.AGENT);
    names.put("also", WordNetRelation.ALSO_SEE);
    names.put("anto_converse", WordNetRelation.ANTO_CONVERSE);
    names.put("anto_gradable", WordNetRelation.ANTO_GRADABLE);
    names.put("anto_simple", WordNetRelation.ANTO_SIMPLE);
    names.put("antonym", WordNetRelation.ANTONYM);
    names.put("attribute", WordNetRelation.ATTRIBUTE);
    names.put("augmentative", WordNetRelation.AUGMENTATIVE);
    names.put("be_in_state", WordNetRelation.BE_IN_STATE);
    names.put("body_part", WordNetRelation.BODY_PART);
    names.put("by_means_of", WordNetRelation.BY_MEANS_OF);
    names.put("causes", WordNetRelation.CAUSE);
    names.put("classified_by", WordNetRelation.CLASSIFIED_BY);
    names.put("classifies", WordNetRelation.CLASSIFIES);
    names.put("co_agent_instrument", WordNetRelation.CO_AGENT_INSTRUMENT);
    names.put("co_agent_patient", WordNetRelation.CO_AGENT_PATIENT);
    names.put("co_agent_result", WordNetRelation.CO_AGENT_RESULT);
    names.put("co_instrument_agent", WordNetRelation.CO_INSTRUMENT_AGENT);
    names.put("co_instrument_patient", WordNetRelation.CO_INSTRUMENT_PATIENT);
    names.put("co_instrument_result", WordNetRelation.CO_INSTRUMENT_RESULT);
    names.put("co_patient_agent", WordNetRelation.CO_PATIENT_AGENT);
    names.put("co_patient_instrument", WordNetRelation.CO_PATIENT_INSTRUMENT);
    names.put("co_result_agent", WordNetRelation.CO_RESULT_AGENT);
    names.put("co_result_instrument", WordNetRelation.CO_RESULT_INSTRUMENT);
    names.put("co_role", WordNetRelation.CO_ROLE);
    names.put("derivation", WordNetRelation.DERIVATIONALLY_RELATED);
    names.put("destination", WordNetRelation.DESTINATION);
    names.put("diminutive", WordNetRelation.DIMINUTIVE);
    names.put("direction", WordNetRelation.DIRECTION);
    names.put("domain_region", WordNetRelation.DOMAIN_REGION);
    names.put("domain_topic", WordNetRelation.DOMAIN_TOPIC);
    names.put("entails", WordNetRelation.ENTAILMENT);
    names.put("eq_synonym", WordNetRelation.EQ_SYNONYM);
    names.put("event", WordNetRelation.EVENT);
    names.put("exemplifies", WordNetRelation.DOMAIN_USAGE);
    names.put("feminine", WordNetRelation.FEMININE);
    names.put("has_augmentative", WordNetRelation.HAS_AUGMENTATIVE);
    names.put("has_diminutive", WordNetRelation.HAS_DIMINUTIVE);
    names.put("has_domain_region", WordNetRelation.MEMBER_OF_DOMAIN_REGION);
    names.put("has_domain_topic", WordNetRelation.MEMBER_OF_DOMAIN_TOPIC);
    names.put("has_feminine", WordNetRelation.HAS_FEMININE);
    names.put("has_masculine", WordNetRelation.HAS_MASCULINE);
    names.put("has_metaphor", WordNetRelation.HAS_METAPHOR);
    names.put("has_metonym", WordNetRelation.HAS_METONYM);
    names.put("has_young", WordNetRelation.HAS_YOUNG);
    names.put("holo_location", WordNetRelation.LOCATION_HOLONYM);
    names.put("holo_member", WordNetRelation.MEMBER_HOLONYM);
    names.put("holo_part", WordNetRelation.PART_HOLONYM);
    names.put("holo_portion", WordNetRelation.PORTION_HOLONYM);
    names.put("holo_substance", WordNetRelation.SUBSTANCE_HOLONYM);
    names.put("holonym", WordNetRelation.HOLONYM);
    names.put("hypernym", WordNetRelation.HYPERNYM);
    names.put("instance_hypernym", WordNetRelation.INSTANCE_HYPERNYM);
    names.put("hyponym", WordNetRelation.HYPONYM);
    names.put("instance_hyponym", WordNetRelation.INSTANCE_HYPONYM);
    names.put("in_manner", WordNetRelation.IN_MANNER);
    names.put("instrument", WordNetRelation.INSTRUMENT);
    names.put("involved", WordNetRelation.INVOLVED);
    names.put("involved_agent", WordNetRelation.INVOLVED_AGENT);
    names.put("involved_direction", WordNetRelation.INVOLVED_DIRECTION);
    names.put("involved_instrument", WordNetRelation.INVOLVED_INSTRUMENT);
    names.put("involved_location", WordNetRelation.INVOLVED_LOCATION);
    names.put("involved_patient", WordNetRelation.INVOLVED_PATIENT);
    names.put("involved_result", WordNetRelation.INVOLVED_RESULT);
    names.put("involved_source_direction", WordNetRelation.INVOLVED_SOURCE_DIRECTION);
    names.put("involved_target_direction", WordNetRelation.INVOLVED_TARGET_DIRECTION);
    names.put("ir_synonym", WordNetRelation.IR_SYNONYM);
    names.put("is_caused_by", WordNetRelation.CAUSED_BY);
    names.put("is_entailed_by", WordNetRelation.ENTAILED_BY);
    names.put("is_exemplified_by", WordNetRelation.MEMBER_OF_DOMAIN_USAGE);
    names.put("is_subevent_of", WordNetRelation.IS_SUBEVENT_OF);
    names.put("location", WordNetRelation.LOCATION);
    names.put("manner_of", WordNetRelation.MANNER_OF);
    names.put("masculine", WordNetRelation.MASCULINE);
    names.put("material", WordNetRelation.MATERIAL);
    names.put("mero_location", WordNetRelation.LOCATION_MERONYM);
    names.put("mero_member", WordNetRelation.MEMBER_MERONYM);
    names.put("mero_part", WordNetRelation.PART_MERONYM);
    names.put("mero_portion", WordNetRelation.PORTION_MERONYM);
    names.put("mero_substance", WordNetRelation.SUBSTANCE_MERONYM);
    names.put("meronym", WordNetRelation.MERONYM);
    names.put("metaphor", WordNetRelation.METAPHOR);
    names.put("metonym", WordNetRelation.METONYM);
    names.put("participle", WordNetRelation.PARTICIPLE);
    names.put("patient", WordNetRelation.PATIENT);
    names.put("pertainym", WordNetRelation.PERTAINYM);
    names.put("property", WordNetRelation.PROPERTY);
    names.put("restricted_by", WordNetRelation.RESTRICTED_BY);
    names.put("restricts", WordNetRelation.RESTRICTS);
    names.put("result", WordNetRelation.RESULT);
    names.put("role", WordNetRelation.ROLE);
    names.put("secondary_aspect_ip", WordNetRelation.SECONDARY_ASPECT_IP);
    names.put("secondary_aspect_pi", WordNetRelation.SECONDARY_ASPECT_PI);
    names.put("simple_aspect_ip", WordNetRelation.SIMPLE_ASPECT_IP);
    names.put("simple_aspect_pi", WordNetRelation.SIMPLE_ASPECT_PI);
    names.put("source_direction", WordNetRelation.SOURCE_DIRECTION);
    names.put("state", WordNetRelation.STATE);
    names.put("state_of", WordNetRelation.STATE_OF);
    names.put("subevent", WordNetRelation.SUBEVENT);
    names.put("target_direction", WordNetRelation.TARGET_DIRECTION);
    names.put("undergoer", WordNetRelation.UNDERGOER);
    names.put("uses", WordNetRelation.USES);
    names.put("vehicle", WordNetRelation.VEHICLE);
    names.put("young", WordNetRelation.YOUNG);
    // Legacy aliases accepted by older WN-LMF producers.
    names.put("domain_usage", WordNetRelation.DOMAIN_USAGE);
    names.put("has_domain_usage", WordNetRelation.MEMBER_OF_DOMAIN_USAGE);
    return Map.copyOf(names);
  }
}
