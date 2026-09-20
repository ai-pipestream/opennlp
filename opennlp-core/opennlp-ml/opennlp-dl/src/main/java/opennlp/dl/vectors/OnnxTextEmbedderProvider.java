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
package opennlp.dl.vectors;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import opennlp.dl.CudaExecutionProviderConfigurer;
import opennlp.dl.ExecutionProviderConfigurer;
import opennlp.dl.ExecutionProviderRequest;
import opennlp.dl.ExecutionProviders;
import opennlp.dl.InferenceOptions;
import opennlp.tools.embeddings.TextEmbedder;
import opennlp.tools.embeddings.TextEmbedderProvider;
import opennlp.tools.util.ext.ProviderSpec;

/**
 * Provides a {@link SentenceVectorsDL} under the name {@value #NAME}. It supports a spec whose
 * location is a local file named {@code *.onnx}, in any letter case, and whose only options are
 * {@value #VOCABULARY_OPTION}, {@value #LOWER_CASE_OPTION}, {@value #POOLING_OPTION},
 * {@value #NORMALIZE_OPTION}, {@value #MAX_LENGTH_OPTION}, {@value #PADDING_OPTION},
 * {@value #EXECUTION_PROVIDERS_OPTION}, {@value #INTRA_OP_NUM_THREADS_OPTION},
 * {@value #INTER_OP_NUM_THREADS_OPTION}, {@value #OPTIMIZATION_LEVEL_OPTION}, {@value #GPU_OPTION}
 * and {@value #GPU_DEVICE_ID_OPTION}. The option values are checked by
 * {@link #create(ProviderSpec)}, which also initializes the ONNX Runtime.
 *
 * <h2>The syntax of {@value #EXECUTION_PROVIDERS_OPTION}</h2>
 *
 * <p>One spec option value carries the whole ordered list that
 * {@link InferenceOptions#setExecutionProviders(List)} takes, because the order is the fallback
 * order and a spec option is a single string. The syntax is:</p>
 *
 * <pre>
 *   list    = request ( ',' request )*
 *   request = id [ '(' option ( ';' option )* ')' ]
 *   option  = name '=' value
 * </pre>
 *
 * <p>So {@code cpu} asks for one execution provider,
 * {@code cuda(device_id=1),cpu} asks for CUDA on the second card with the CPU behind it, and
 * {@code openvino(device_type=GPU.0;cache_dir=/var/cache/ov),cpu} asks an addon for two provider
 * options. An {@code id} is the id of an {@link ExecutionProviderConfigurer}, so an addon execution
 * provider is requested from a spec by naming the id its configurer answers to; the built-in ids are
 * {@value ExecutionProviders#CPU} and {@value ExecutionProviders#CUDA} and everything else comes from
 * a jar on the class path.</p>
 *
 * <p>Two separators rather than one because a provider option value may itself hold a comma:
 * OpenVINO's {@code device_type} takes {@code MULTI:GPU,CPU}. Inside the parentheses of a request
 * {@code ;} separates the options, so {@code ,} there is an ordinary character of a value and
 * {@code openvino(device_type=MULTI:GPU,CPU),cpu} means what it looks like. The first {@code =} of an
 * option separates its name from its value, so a value may hold further ones.</p>
 *
 * <p>Nothing is trimmed and nothing is guessed, the way no other option of this class is trimmed.
 * Malformed input is reported rather than repaired: an empty value, an empty list element, a
 * {@code (} that is not closed, a {@code )} without a {@code (}, a nested {@code (}, characters
 * between a {@code )} and the next {@code ,}, an empty pair of parentheses, an option without an
 * {@code =}, a blank option name, a repeated option name within one request, and an id that could
 * name no configurer each fail {@link #create(ProviderSpec)} with an
 * {@link IllegalArgumentException} naming what was wrong. A space is therefore an ordinary
 * character, and {@code cuda, cpu} is rejected because {@code " cpu"} is not a usable id.</p>
 *
 * <p>An option value may be empty, as {@code cuda(device_id=)} is: provider options are the
 * configurer's business, and {@link ProviderSpec} carries an empty value as readily as any other, so
 * such a request reaches its configurer and is rejected there if the configurer rejects it. An option
 * name may not be blank, which is the one rule {@link ProviderSpec} itself imposes.</p>
 *
 * @since 3.0.0
 */
public final class OnnxTextEmbedderProvider implements TextEmbedderProvider {

  /** The name of this provider. */
  public static final String NAME = "onnx";

  /**
   * The required option that names the vocabulary file, resolved against the directory of the
   * model if it is relative.
   */
  public static final String VOCABULARY_OPTION = "vocabulary";

  /** The option that lower-cases the text, {@code true} by default or {@code false}. */
  public static final String LOWER_CASE_OPTION = "lowerCase";

  /** The option that selects the {@link Pooling}, {@code mean} by default or {@code cls}. */
  public static final String POOLING_OPTION = "pooling";

  /** The option that scales vectors to unit length, {@code true} by default or {@code false}. */
  public static final String NORMALIZE_OPTION = "normalize";

  /**
   * The option that sets the maximum number of tokens per input, at least {@code 2},
   * {@value SentenceVectorsDL#DEFAULT_MAX_LENGTH} by default.
   */
  public static final String MAX_LENGTH_OPTION = "maxLength";

  /**
   * The option that selects the {@link PaddingStrategy}: {@code exact_length}, {@code longest} or
   * {@code max_length}. Left out, the strategy follows the execution provider through
   * {@link PaddingStrategy#defaultFor(java.util.List)}, which gives {@code longest} where the
   * resolved execution provider is an accelerator and {@code exact_length} where it is not.
   */
  public static final String PADDING_OPTION = "padding";

  /**
   * The option that names the ONNX Runtime execution providers to run on, in priority order, with
   * their provider options. Left out, nothing is requested and ONNX Runtime places the session on
   * the CPU, as {@link InferenceOptions#setExecutionProviders(List)} describes. The syntax and what
   * is rejected are documented on this class.
   *
   * <p>This is the option that reaches an addon: an id no jar on the class path answers to fails at
   * construction, and so does an execution provider ONNX Runtime cannot register, because neither
   * this class nor {@link ExecutionProviders} falls back to the CPU.</p>
   *
   * <p>It wins over the deprecated {@value #GPU_OPTION}, which is the precedence
   * {@link ExecutionProviders#resolve(InferenceOptions)} documents for the two settings it maps
   * onto.</p>
   */
  public static final String EXECUTION_PROVIDERS_OPTION = "executionProviders";

  /**
   * The option that sets the number of threads one operator of the session may run on, a
   * non-negative integer of at most {@value InferenceOptions#MAX_NUM_THREADS}. Left out, ONNX
   * Runtime's own default stands. See {@link InferenceOptions#setIntraOpNumThreads(int)} for what
   * the value does and why it is the setting that moves CPU throughput.
   */
  public static final String INTRA_OP_NUM_THREADS_OPTION = "intraOpNumThreads";

  /**
   * The option that sets the number of threads the session may run independent operators on, a
   * non-negative integer of at most {@value InferenceOptions#MAX_NUM_THREADS}. Left out, ONNX
   * Runtime's own default stands. See {@link InferenceOptions#setInterOpNumThreads(int)}.
   */
  public static final String INTER_OP_NUM_THREADS_OPTION = "interOpNumThreads";

  /**
   * The option that sets the graph optimizations ONNX Runtime applies when it loads the model, named
   * by the lower case form of an {@link OrtSession.SessionOptions.OptLevel} constant, for example
   * {@code no_opt} or {@code all_opt}. Left out, ONNX Runtime's own default stands, which already
   * applies every optimization. See {@link InferenceOptions#setOptimizationLevel}.
   */
  public static final String OPTIMIZATION_LEVEL_OPTION = "optimizationLevel";

  /**
   * The option that runs inference on the CUDA execution provider, {@code false} by default or
   * {@code true}. It requires the {@code onnxruntime_gpu} runtime on the classpath, which the
   * {@code opennlp-dl-gpu} module brings in. With the CPU-only runtime, or with a CUDA installation
   * the runtime cannot load, {@link #create(ProviderSpec)} reports the failure rather than falling
   * back to the CPU.
   *
   * @deprecated A flag can name one of the fourteen execution providers ONNX Runtime's Java API
   *     exposes and cannot order a fallback behind it. Use {@value #EXECUTION_PROVIDERS_OPTION},
   *     where {@code cuda} is the same request and {@code cuda(device_id=1),cpu} is one this flag
   *     cannot express. This option keeps working and keeps selecting CUDA, and it is read only
   *     while {@value #EXECUTION_PROVIDERS_OPTION} is absent.
   */
  @Deprecated(since = "3.0.0", forRemoval = true)
  public static final String GPU_OPTION = "gpu";

  /**
   * The option that names the CUDA device to run on, {@code 0} by default. It is only read when
   * {@value #GPU_OPTION} is {@code true}. An id no card answers to makes
   * {@link #create(ProviderSpec)} fail rather than fall back to another card or to the CPU.
   *
   * @deprecated Part of the flag replaced by {@value #EXECUTION_PROVIDERS_OPTION}. The device is the
   *     {@value CudaExecutionProviderConfigurer#DEVICE_ID_OPTION} provider option of a
   *     {@value ExecutionProviders#CUDA} request, which is also where every other CUDA provider
   *     option goes.
   */
  @Deprecated(since = "3.0.0", forRemoval = true)
  public static final String GPU_DEVICE_ID_OPTION = "gpuDeviceId";

  private static final String MODEL_SUFFIX = ".onnx";
  private static final String TRUE = "true";
  private static final String FALSE = "false";
  private static final String MEAN = Pooling.MEAN.name().toLowerCase(Locale.ROOT);
  private static final String CLS = Pooling.CLS.name().toLowerCase(Locale.ROOT);
  private static final String EXACT_LENGTH =
      PaddingStrategy.EXACT_LENGTH.name().toLowerCase(Locale.ROOT);
  private static final String LONGEST = PaddingStrategy.LONGEST.name().toLowerCase(Locale.ROOT);
  private static final String MAX_LENGTH =
      PaddingStrategy.MAX_LENGTH.name().toLowerCase(Locale.ROOT);
  private static final int MIN_MAX_LENGTH = 2;

  /** Separates the requests of {@value #EXECUTION_PROVIDERS_OPTION}, outside a parenthesis. */
  private static final char REQUEST_SEPARATOR = ',';

  /** Opens the provider options of one request. */
  private static final char OPTIONS_OPEN = '(';

  /** Closes the provider options of one request. */
  private static final char OPTIONS_CLOSE = ')';

  /** Separates the provider options of one request, inside the parentheses. */
  private static final char OPTION_SEPARATOR = ';';

  /** Separates the name of a provider option from its value. */
  private static final char OPTION_ASSIGNMENT = '=';

  @Override
  public String name() {
    return NAME;
  }

  /**
   * {@inheritDoc}
   * Checks that the ONNX Runtime classes are present without initializing the runtime.
   */
  @Override
  public boolean isAvailable() {
    try {
      return OrtEnvironment.class.getName() != null;
    } catch (final LinkageError e) {
      return false;
    }
  }

  @SuppressWarnings({"deprecation", "removal"})
  @Override
  public boolean supports(final ProviderSpec spec) {
    if (spec == null) {
      throw new IllegalArgumentException("spec must not be null");
    }
    return spec.path().isPresent() && spec.locationEndsWith(MODEL_SUFFIX)
        && spec.hasOnlyOptions(VOCABULARY_OPTION, LOWER_CASE_OPTION, POOLING_OPTION,
            NORMALIZE_OPTION, MAX_LENGTH_OPTION, PADDING_OPTION, EXECUTION_PROVIDERS_OPTION,
            INTRA_OP_NUM_THREADS_OPTION, INTER_OP_NUM_THREADS_OPTION, OPTIMIZATION_LEVEL_OPTION,
            GPU_OPTION, GPU_DEVICE_ID_OPTION);
  }

  @Override
  public TextEmbedder create(final ProviderSpec spec) throws IOException {
    if (spec == null || !supports(spec)) {
      throw new IllegalArgumentException("spec is not supported: " + spec);
    }
    final Path model = spec.path().orElseThrow();
    if (!Files.isRegularFile(model)) {
      throw new IllegalArgumentException("model must be a regular file: " + model);
    }
    final String vocabulary = spec.option(VOCABULARY_OPTION, null);
    if (vocabulary == null || vocabulary.isBlank()) {
      throw new IllegalArgumentException(VOCABULARY_OPTION + " must not be null or blank");
    }
    final boolean lowerCase = booleanOption(spec, LOWER_CASE_OPTION);
    final boolean normalize = booleanOption(spec, NORMALIZE_OPTION);
    final Pooling pooling = poolingOption(spec);
    final int maxLength = maxLengthOption(spec);
    final InferenceOptions inferenceOptions = inferenceOptions(spec);
    final PaddingStrategy padding = paddingOption(spec);
    final Path vocabularyPath = model.toAbsolutePath().getParent().resolve(vocabulary);
    try {
      // A spec without a padding option leaves the strategy to the execution provider, the way a
      // caller of withDerivedPadding does.
      if (padding == null) {
        return SentenceVectorsDL.withDerivedPadding(model.toFile(), vocabularyPath.toFile(),
            lowerCase, pooling, normalize, maxLength, inferenceOptions);
      }
      return new SentenceVectorsDL(model.toFile(), vocabularyPath.toFile(), lowerCase, pooling,
          normalize, maxLength, padding, inferenceOptions);
    } catch (final OrtException e) {
      throw new IOException("Cannot load the ONNX model " + model, e);
    } catch (final LinkageError e) {
      throw new IOException("Cannot initialize the ONNX Runtime", e);
    }
  }

  /**
   * Reads a boolean option that is {@code true} unless set to {@code false}.
   *
   * @param spec The spec to read.
   * @param name The name of the option.
   * @return The option value.
   * @throws IllegalArgumentException Thrown if the value is neither {@code true} nor
   *     {@code false}.
   */
  private boolean booleanOption(final ProviderSpec spec, final String name) {
    return booleanOption(spec, name, TRUE);
  }

  /**
   * Reads a boolean option with the given default.
   *
   * @param spec The spec to read.
   * @param name The name of the option.
   * @param fallback The value to use if the option is not set, {@code "true"} or {@code "false"}.
   * @return The option value.
   * @throws IllegalArgumentException Thrown if the value is neither {@code true} nor
   *     {@code false}.
   */
  private boolean booleanOption(final ProviderSpec spec, final String name,
      final String fallback) {
    final String value = spec.option(name, fallback);
    if (!TRUE.equals(value) && !FALSE.equals(value)) {
      throw new IllegalArgumentException(name + " must be true or false");
    }
    return TRUE.equals(value);
  }

  /**
   * Builds the {@link InferenceOptions} a spec asks for: which execution providers to run on and
   * how the session may use threads and optimize its graph.
   *
   * <p>{@value #EXECUTION_PROVIDERS_OPTION} wins over {@value #GPU_OPTION} whenever it is present,
   * which is the precedence {@link ExecutionProviders#resolve(InferenceOptions)} documents for the
   * two settings these options map onto: a list the spec author ordered is not quietly prepended
   * with a request from a leftover flag. {@value #GPU_OPTION} and {@value #GPU_DEVICE_ID_OPTION} are
   * read either way, so a malformed value of either is reported rather than ignored, and they only
   * reach ONNX Runtime while the list is absent.</p>
   *
   * <p>Each session setting is left alone unless its option is present, since
   * {@link InferenceOptions} has no value that means "unset again" and an unset setting is how ONNX
   * Runtime's own default is kept.</p>
   *
   * <p>Package-private rather than private because it is the seam a test reaches through: what a
   * spec asked for is otherwise observable only in native session state that ONNX Runtime does not
   * read back.</p>
   *
   * @param spec The spec to read.
   * @return The inference options. Never {@code null}.
   * @throws IllegalArgumentException Thrown if any of these option values is malformed.
   */
  @SuppressWarnings({"deprecation", "removal"})
  InferenceOptions inferenceOptions(final ProviderSpec spec) {
    final InferenceOptions inferenceOptions = new InferenceOptions();
    final List<ExecutionProviderRequest> requested = executionProvidersOption(spec);
    final boolean gpu = booleanOption(spec, GPU_OPTION, FALSE);
    final int deviceId = gpuDeviceIdOption(spec);
    if (requested != null) {
      inferenceOptions.setExecutionProviders(requested);
    } else if (gpu) {
      inferenceOptions.setExecutionProviders(List.of(ExecutionProviderRequest.of(
          ExecutionProviders.CUDA, Map.of(CudaExecutionProviderConfigurer.DEVICE_ID_OPTION,
              Integer.toString(deviceId)))));
    }
    final Integer intraOpNumThreads = numThreadsOption(spec, INTRA_OP_NUM_THREADS_OPTION);
    if (intraOpNumThreads != null) {
      inferenceOptions.setIntraOpNumThreads(intraOpNumThreads);
    }
    final Integer interOpNumThreads = numThreadsOption(spec, INTER_OP_NUM_THREADS_OPTION);
    if (interOpNumThreads != null) {
      inferenceOptions.setInterOpNumThreads(interOpNumThreads);
    }
    final OrtSession.SessionOptions.OptLevel optimizationLevel = optimizationLevelOption(spec);
    if (optimizationLevel != null) {
      inferenceOptions.setOptimizationLevel(optimizationLevel);
    }
    return inferenceOptions;
  }

  /**
   * Reads the {@value #EXECUTION_PROVIDERS_OPTION} option, whose syntax this class documents.
   *
   * @param spec The spec to read.
   * @return The requests in the order the option lists them, or {@code null} if the option is not
   *     set, which is not the same as an empty list and is why this is nullable rather than empty.
   * @throws IllegalArgumentException Thrown if the value is malformed or names an unusable id.
   */
  private List<ExecutionProviderRequest> executionProvidersOption(final ProviderSpec spec) {
    final String value = spec.option(EXECUTION_PROVIDERS_OPTION, null);
    if (value == null) {
      return null;
    }
    if (value.isEmpty()) {
      throw malformedList("must name at least one execution provider", value);
    }
    final List<ExecutionProviderRequest> requests = new ArrayList<>();
    int start = 0;
    while (true) {
      final int end = endOfRequest(value, start);
      if (end == start) {
        throw malformedList("has an empty element", value);
      }
      requests.add(request(value, start, end));
      if (end == value.length()) {
        return List.copyOf(requests);
      }
      start = end + 1;
      if (start == value.length()) {
        throw malformedList("ends with a '" + REQUEST_SEPARATOR + "'", value);
      }
    }
  }

  /**
   * Finds where the request that begins at {@code start} ends: at the next {@value
   * #REQUEST_SEPARATOR} outside a parenthesis, or at the end of the value.
   *
   * <p>A character scan rather than a split, both because a provider option value may hold a
   * {@value #REQUEST_SEPARATOR} and because the unbalanced cases are reported here rather than
   * turning into a silently different list.</p>
   *
   * @param value The whole option value.
   * @param start The first index of the request.
   * @return The index one past the request, which is the index of its separator or the length of
   *     {@code value}.
   * @throws IllegalArgumentException Thrown if the parentheses of the request are unbalanced or
   *     nested.
   */
  private int endOfRequest(final String value, final int start) {
    boolean inOptions = false;
    for (int at = start; at < value.length(); at++) {
      final char current = value.charAt(at);
      if (current == OPTIONS_OPEN) {
        if (inOptions) {
          throw malformedList("has a nested '" + OPTIONS_OPEN + "'", value);
        }
        inOptions = true;
      } else if (current == OPTIONS_CLOSE) {
        if (!inOptions) {
          throw malformedList("has a '" + OPTIONS_CLOSE + "' without a '" + OPTIONS_OPEN + "'",
              value);
        }
        inOptions = false;
      } else if (current == REQUEST_SEPARATOR && !inOptions) {
        return at;
      }
    }
    if (inOptions) {
      throw malformedList("has a '" + OPTIONS_OPEN + "' that is never closed", value);
    }
    return value.length();
  }

  /**
   * Builds one request out of {@code value} between {@code start} and {@code end}, which
   * {@link #endOfRequest(String, int)} has already established holds balanced parentheses.
   *
   * @param value The whole option value.
   * @param start The first index of the request.
   * @param end The index one past the request.
   * @return The request. Never {@code null}.
   * @throws IllegalArgumentException Thrown if the request is malformed or names an unusable id.
   */
  private ExecutionProviderRequest request(final String value, final int start, final int end) {
    final int open = indexOf(value, OPTIONS_OPEN, start, end);
    if (open < 0) {
      return requestOf(value, value.substring(start, end), Map.of());
    }
    if (value.charAt(end - 1) != OPTIONS_CLOSE) {
      throw malformedList("has characters between a '" + OPTIONS_CLOSE + "' and the next '"
          + REQUEST_SEPARATOR + "'", value);
    }
    if (open == start) {
      throw malformedList("has provider options without an execution provider id", value);
    }
    if (open + 1 == end - 1) {
      throw malformedList("has an empty '" + OPTIONS_OPEN + OPTIONS_CLOSE + "'; leave it out to "
          + "request an execution provider without provider options", value);
    }
    return requestOf(value, value.substring(start, open), options(value, open + 1, end - 1));
  }

  /**
   * Reads the provider options of one request out of {@code value} between {@code from} and
   * {@code to}, which are the indexes just inside its parentheses.
   *
   * @param value The whole option value.
   * @param from The first index of the provider options.
   * @param to The index one past the provider options.
   * @return The provider options, in the order the request lists them. Never {@code null}.
   * @throws IllegalArgumentException Thrown if an option has no {@value #OPTION_ASSIGNMENT}, a
   *     blank name or a name that is repeated within this request.
   */
  private Map<String, String> options(final String value, final int from, final int to) {
    final Map<String, String> options = new LinkedHashMap<>();
    int start = from;
    while (true) {
      int end = indexOf(value, OPTION_SEPARATOR, start, to);
      if (end < 0) {
        end = to;
      }
      addOption(options, value, start, end);
      if (end == to) {
        return options;
      }
      start = end + 1;
      if (start == to) {
        throw malformedList("ends its provider options with a '" + OPTION_SEPARATOR + "'", value);
      }
    }
  }

  /**
   * Adds one provider option, {@code name=value}, to {@code options}.
   *
   * <p>The first {@value #OPTION_ASSIGNMENT} separates the name from the value, so a value may hold
   * further ones. The value may be empty, since a configurer is the authority on its own provider
   * options; the name may not be blank, which is the rule {@link ProviderSpec} imposes on every
   * option map in OpenNLP.</p>
   *
   * @param options The options read so far, which this adds to.
   * @param value The whole option value.
   * @param start The first index of this option.
   * @param end The index one past this option.
   * @throws IllegalArgumentException Thrown if the option has no {@value #OPTION_ASSIGNMENT}, a
   *     blank name or a name already present in {@code options}.
   */
  private void addOption(final Map<String, String> options, final String value, final int start,
      final int end) {
    final int assignment = indexOf(value, OPTION_ASSIGNMENT, start, end);
    if (assignment < 0) {
      throw malformedList("has the provider option '" + value.substring(start, end)
          + "' without a '" + OPTION_ASSIGNMENT + "'", value);
    }
    final String name = value.substring(start, assignment);
    if (name.isBlank()) {
      throw malformedList("has a provider option with a blank name", value);
    }
    if (options.put(name, value.substring(assignment + 1, end)) != null) {
      throw malformedList("names the provider option '" + name + "' twice in one request", value);
    }
  }

  /**
   * Builds a request, reporting an unusable id against the option it came from.
   *
   * <p>The id rule is {@link ExecutionProviderRequest}'s, so it is checked there rather than
   * restated here, and only its message is placed in the context of this spec option.</p>
   *
   * @param value The whole option value, for the message.
   * @param id The id read out of the value.
   * @param options The provider options of the request.
   * @return The request. Never {@code null}.
   * @throws IllegalArgumentException Thrown if the id or the provider options are unusable.
   */
  private ExecutionProviderRequest requestOf(final String value, final String id,
      final Map<String, String> options) {
    try {
      return ExecutionProviderRequest.of(id, options);
    } catch (final IllegalArgumentException e) {
      throw new IllegalArgumentException(EXECUTION_PROVIDERS_OPTION + " is malformed: "
          + e.getMessage() + ", in '" + value + "'", e);
    }
  }

  /**
   * {@return the first index of {@code character} in {@code value} between {@code from} and
   * {@code to}, or {@code -1}}
   *
   * <p>A scan over a range rather than {@link String#indexOf(int, int)}, which has no end bound and
   * would find a character belonging to the next request.</p>
   *
   * @param value The whole option value.
   * @param character The character to find.
   * @param from The first index to read.
   * @param to The index one past the last to read.
   */
  private int indexOf(final String value, final char character, final int from, final int to) {
    for (int at = from; at < to; at++) {
      if (value.charAt(at) == character) {
        return at;
      }
    }
    return -1;
  }

  /**
   * Builds the failure for a malformed {@value #EXECUTION_PROVIDERS_OPTION} value.
   *
   * @param fault What is wrong with it, as a phrase completing "the option ...".
   * @param value The whole option value, which is quoted so a spec author can see what was read.
   * @return The exception to throw. Never {@code null}.
   */
  private IllegalArgumentException malformedList(final String fault, final String value) {
    return new IllegalArgumentException(EXECUTION_PROVIDERS_OPTION + " " + fault + ": '" + value
        + "'. The syntax is id[(name=value;name=value)][,id...], for example "
        + "'cuda(device_id=1),cpu'.");
  }

  /**
   * Reads one of the thread count options.
   *
   * @param spec The spec to read.
   * @param name The name of the option.
   * @return The thread count, or {@code null} if the option is not set, which leaves ONNX Runtime's
   *     own default in place.
   * @throws IllegalArgumentException Thrown if the value is not an integer between {@code 0} and
   *     {@value InferenceOptions#MAX_NUM_THREADS}.
   */
  private Integer numThreadsOption(final ProviderSpec spec, final String name) {
    final String value = spec.option(name, null);
    if (value == null) {
      return null;
    }
    try {
      final int numThreads = Integer.parseInt(value);
      if (numThreads >= 0 && numThreads <= InferenceOptions.MAX_NUM_THREADS) {
        return numThreads;
      }
    } catch (final NumberFormatException e) {
      // reported below
    }
    throw new IllegalArgumentException(name + " must be an integer between 0 and "
        + InferenceOptions.MAX_NUM_THREADS);
  }

  /**
   * Reads the {@value #OPTIMIZATION_LEVEL_OPTION} option.
   *
   * <p>The accepted values are derived from {@link OrtSession.SessionOptions.OptLevel} rather than
   * listed here, so a level a later ONNX Runtime adds is accepted without this class being
   * edited.</p>
   *
   * @param spec The spec to read.
   * @return The level, or {@code null} if the option is not set, which leaves ONNX Runtime's own
   *     default in place.
   * @throws IllegalArgumentException Thrown if the value names no level.
   */
  private OrtSession.SessionOptions.OptLevel optimizationLevelOption(final ProviderSpec spec) {
    final String value = spec.option(OPTIMIZATION_LEVEL_OPTION, null);
    if (value == null) {
      return null;
    }
    final StringBuilder accepted = new StringBuilder();
    for (final OrtSession.SessionOptions.OptLevel level
        : OrtSession.SessionOptions.OptLevel.values()) {
      final String name = level.name().toLowerCase(Locale.ROOT);
      if (name.equals(value)) {
        return level;
      }
      if (accepted.length() > 0) {
        accepted.append(", ");
      }
      accepted.append(name);
    }
    throw new IllegalArgumentException(OPTIMIZATION_LEVEL_OPTION + " must be one of " + accepted);
  }

  /**
   * Reads the {@value #GPU_DEVICE_ID_OPTION} option.
   *
   * @param spec The spec to read.
   * @return The device id, {@code 0} if the option is not set.
   * @throws IllegalArgumentException Thrown if the value is not a non-negative integer.
   */
  @SuppressWarnings({"deprecation", "removal"})
  private int gpuDeviceIdOption(final ProviderSpec spec) {
    final String value = spec.option(GPU_DEVICE_ID_OPTION, null);
    if (value == null) {
      return 0;
    }
    try {
      final int deviceId = Integer.parseInt(value);
      if (deviceId >= 0) {
        return deviceId;
      }
    } catch (final NumberFormatException e) {
      // reported below
    }
    throw new IllegalArgumentException(
        GPU_DEVICE_ID_OPTION + " must be a non-negative integer");
  }

  /**
   * Reads the {@value #POOLING_OPTION} option.
   *
   * @param spec The spec to read.
   * @return The pooling, {@link Pooling#MEAN} if the option is not set.
   * @throws IllegalArgumentException Thrown if the value is neither {@code mean} nor
   *     {@code cls}.
   */
  private Pooling poolingOption(final ProviderSpec spec) {
    final String value = spec.option(POOLING_OPTION, MEAN);
    if (MEAN.equals(value)) {
      return Pooling.MEAN;
    }
    if (CLS.equals(value)) {
      return Pooling.CLS;
    }
    throw new IllegalArgumentException(POOLING_OPTION + " must be " + MEAN + " or " + CLS);
  }

  /**
   * Reads the {@value #PADDING_OPTION} option.
   *
   * <p>{@code null} for an option that is not set, which leaves the choice to the execution
   * provider: {@link #create(ProviderSpec)} then goes through
   * {@link SentenceVectorsDL#withDerivedPadding(java.io.File, java.io.File, boolean, Pooling,
   * boolean, int, InferenceOptions)}, so a spec that requests an accelerator without a padding
   * option gets the strategy that gives six to eight times the throughput there. A spec that names a
   * strategy gets that strategy on any execution provider.</p>
   *
   * @param spec The spec to read.
   * @return The padding strategy, or {@code null} if the option is not set.
   * @throws IllegalArgumentException Thrown if the value names no {@link PaddingStrategy}.
   */
  private PaddingStrategy paddingOption(final ProviderSpec spec) {
    final String value = spec.option(PADDING_OPTION, null);
    if (value == null) {
      return null;
    }
    if (EXACT_LENGTH.equals(value)) {
      return PaddingStrategy.EXACT_LENGTH;
    }
    if (LONGEST.equals(value)) {
      return PaddingStrategy.LONGEST;
    }
    if (MAX_LENGTH.equals(value)) {
      return PaddingStrategy.MAX_LENGTH;
    }
    throw new IllegalArgumentException(PADDING_OPTION + " must be " + EXACT_LENGTH + ", "
        + LONGEST + " or " + MAX_LENGTH);
  }

  /**
   * Reads the {@value #MAX_LENGTH_OPTION} option.
   *
   * @param spec The spec to read.
   * @return The maximum length, {@value SentenceVectorsDL#DEFAULT_MAX_LENGTH} if the option is
   *     not set.
   * @throws IllegalArgumentException Thrown if the value is not an integer of at least
   *     {@code 2}.
   */
  private int maxLengthOption(final ProviderSpec spec) {
    final String value = spec.option(MAX_LENGTH_OPTION, null);
    if (value == null) {
      return SentenceVectorsDL.DEFAULT_MAX_LENGTH;
    }
    try {
      final int maxLength = Integer.parseInt(value);
      if (maxLength >= MIN_MAX_LENGTH) {
        return maxLength;
      }
    } catch (final NumberFormatException e) {
      // reported below
    }
    throw new IllegalArgumentException(
        MAX_LENGTH_OPTION + " must be an integer of at least " + MIN_MAX_LENGTH);
  }
}
