/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package opennlp.dl.vectors;

import java.io.File;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import opennlp.dl.AbstractDL;
import opennlp.dl.ExecutionProviderRequest;
import opennlp.dl.ExecutionProviders;
import opennlp.dl.InferenceOptions;
import opennlp.dl.Tokens;
import opennlp.tools.commons.ThreadSafe;
import opennlp.tools.embeddings.EmbeddingException;
import opennlp.tools.embeddings.TextEmbedder;

/**
 * Facilitates the generation of sentence vectors using
 * a sentence-transformers model converted to ONNX.
 *
 * <p>The model inputs follow the standard single-segment BERT
 * encoding: {@code attention_mask} is {@code 1} for every real
 * token and {@code token_type_ids} is {@code 0} throughout.</p>
 *
 * <p>The sentence vector is read from a {@code sentence_embedding} output of shape
 * {@code [batch, hidden]} if the model has one. Otherwise the token vectors of the first
 * {@code [batch, tokens, hidden]} output are pooled with the configured {@link Pooling}. By
 * default the token vectors are averaged and the result is scaled to unit length, as
 * sentence-transformers does for the MiniLM family. Text longer than the maximum length is
 * truncated, keeping the final {@code [SEP]} token.</p>
 *
 * <p><b>Release note (OpenNLP 3.0.0):</b> prior releases sent an
 * all-zero {@code attention_mask} and all-one {@code token_type_ids},
 * so the encoder attended to nothing and the output vectors were
 * incorrect, and they returned the raw {@code [CLS]} vector instead of the pooled
 * sentence vector. Additionally, tokenization now performs BERT basic
 * tokenization (lower casing and accent stripping by default, see
 * {@link opennlp.tools.tokenize.WordpieceEncoder}) before wordpiece.
 * Output vectors change with the corrected encoding, tokenization and pooling;
 * any embeddings persisted from the previous behavior are not
 * comparable with the corrected output and must be re-embedded.</p>
 *
 * <p>This class is thread-safe and may be shared across threads: the inference methods hold no
 * per-call instance state and the underlying {@link OrtSession} supports concurrent execution. The
 * reusable direct buffers each inference stages its inputs into and writes its output back through are
 * confined to the thread that runs the inference, so no two threads share one, and the total such
 * buffers hold is bounded; see {@code OnnxInference}. This thread-safety guarantee applies until
 * {@link #close()} is called; callers must not race {@code close()} with inference methods.</p>
 *
 * <p>{@link #embedAll(List)} turns a call into as few inferences as the configured
 * {@link PaddingStrategy} allows. Under {@link PaddingStrategy#EXACT_LENGTH} it adds no padding and
 * runs one inference per distinct tokenized length; under {@link PaddingStrategy#LONGEST} it runs an
 * entire call of mixed-length inputs as one inference. The strategy leaves the vectors as they are
 * and changes only the tensor shapes.</p>
 *
 * <p>{@link #withDerivedPadding(File, File, boolean, Pooling, boolean, int, InferenceOptions)}
 * takes no {@link PaddingStrategy} and derives one from the execution providers the session was
 * configured with, through {@link PaddingStrategy#defaultFor(List)}: padding to the longest row of a
 * batch on an accelerator, exact length grouping on the CPU, since the two want different shapes by
 * a factor of several. A strategy given by name is applied as given; the execution provider does not
 * override it. Either way the strategy is written to the log at {@code info} next to the execution
 * providers, so what a session runs at can be read rather than inferred.</p>
 *
 * <p>Inference runs where ONNX Runtime puts it, which is the CPU, unless
 * {@link #SentenceVectorsDL(File, File, boolean, Pooling, boolean, int, PaddingStrategy,
 * InferenceOptions)} is given an {@link InferenceOptions} that requests execution providers through
 * {@link InferenceOptions#setExecutionProviders(List)}. They are appended in the order requested,
 * which is the order ONNX Runtime falls back along. The execution provider does not change the
 * vectors beyond the reordering a different kernel implies.</p>
 *
 * <p><b>Vectors from a GPU execution provider depend on the shape of the batch they were produced
 * in.</b> cuBLAS picks its blocking from the tensor width and floating point addition does not
 * associate, so the same text embedded in a batch padded to a longer member and embedded on its own
 * differ by around {@code 1.2e-4} absolute per component; the worst deviation measured was
 * {@code 5.15e-7} in one minus cosine similarity, which is well inside any retrieval or
 * clustering tolerance but is not zero. On the CPU provider the two agree exactly. Anyone caching or
 * persisting vectors has to key the cache by the execution provider and, on a GPU, accept that a
 * vector is reproducible only to that tolerance unless the batch shape is reproduced too. The
 * {@link PaddingStrategy} therefore does change GPU vectors slightly, even though it changes nothing
 * on a CPU.</p>
 */
@ThreadSafe
public class SentenceVectorsDL extends AbstractDL implements TextEmbedder {

  /** The maximum number of tokens per input if none is given, the BERT position limit. */
  public static final int DEFAULT_MAX_LENGTH = 512;

  /**
   * The {@link PaddingStrategy} applied where no caller and no execution provider of the session
   * calls for another one: the constructors that take no {@link InferenceOptions}, which run on the
   * CPU, and any provider list {@link PaddingStrategy#defaultFor(List)} does not classify as an
   * accelerator. It equals {@code PaddingStrategy.defaultFor(List.of())} and is unchanged from
   * earlier releases, so a caller of an older constructor gets the tensor shapes, the inference
   * count and the vectors it has always had.
   *
   * <p>It is not the strategy each session applies. A session on an accelerator derives
   * {@link PaddingStrategy#LONGEST} instead, since padding is six to eight times faster there and
   * slower on the CPU; {@link PaddingStrategy#defaultFor(List)} has the numbers and the rule.</p>
   */
  public static final PaddingStrategy DEFAULT_PADDING = PaddingStrategy.EXACT_LENGTH;

  /**
   * The upper bound on the token positions of one inference, the product of its row count and
   * its padded row length. {@link #embedAll(List)} splits a call that would exceed it into
   * consecutive sub-batches, so the tensors of one inference stay bounded however large the call
   * is. At this bound the three {@code int64} input tensors hold
   * {@value #MAX_BATCH_TOKEN_POSITIONS} elements each, and a {@code [batch, tokens, hidden]}
   * output with a hidden size of 768 holds around 48 MiB of {@code float}. A single input longer
   * than the bound still runs on its own rather than being dropped or truncated further.
   *
   * <p>This is also what decides how large one reusable direct buffer can grow, on the input side and
   * on the output side alike, so it is paired with
   * {@code OnnxInference.DEFAULT_MAX_PINNED_OUTPUT_BYTES} and
   * {@code OnnxInference.DEFAULT_MAX_INPUT_STAGING_BYTES}, which bound how many such buffers one
   * component may hold across its threads. Changing any of the three numbers means revisiting the
   * others.</p>
   */
  public static final int MAX_BATCH_TOKEN_POSITIONS = 16384;

  private static final Logger logger = LoggerFactory.getLogger(SentenceVectorsDL.class);

  private static final String SENTENCE_EMBEDDING = "sentence_embedding";
  private static final int POOLED_RANK = 2;
  private static final int TOKEN_RANK = 3;
  private static final int MIN_LENGTH = 2;

  private final Pooling pooling;
  private final boolean normalize;
  private final int maxLength;
  private final String outputName;
  private final boolean pooledOutput;

  /**
   * The output shape the model declares, as it declares it, with a non-positive entry wherever a
   * dimension is dynamic. {@link #pinnableOutputShape(int, int)} checks a candidate output shape
   * against it, so that a model declaring a fixed dimension where this class would pin a different
   * one falls back to an unpinned read rather than failing the run.
   */
  private final long[] declaredOutputShape;

  private final int dimension;
  private final PaddingStrategy padding;
  private final long padTokenId;

  /**
   * Instantiates a {@link SentenceVectorsDL sentence vector generator} for an
   * uncased model. Input text is lower cased and accent stripped during
   * tokenization, as required by uncased models such as the
   * sentence-transformers MiniLM family.
   *
   * @param model The file name of a sentence vectors ONNX model.
   * @param vocabulary The file name of the vocabulary file for the model.
   *
   * @throws OrtException Thrown if the {@code model} cannot be loaded.
   * @throws IOException Thrown if errors occurred loading the {@code model} or {@code vocabulary}.
   */
  public SentenceVectorsDL(final File model, final File vocabulary)
      throws OrtException, IOException {

    this(model, vocabulary, true);

  }

  /**
   * Instantiates a {@link SentenceVectorsDL sentence vector generator} that averages the token
   * vectors, scales the result to unit length and truncates input to
   * {@value #DEFAULT_MAX_LENGTH} tokens.
   *
   * @param model The file name of a sentence vectors ONNX model.
   * @param vocabulary The file name of the vocabulary file for the model.
   * @param lowerCase {@code true} for uncased models (lower casing and accent
   *     stripping during tokenization), {@code false} for cased models.
   *
   * @throws OrtException Thrown if the {@code model} cannot be loaded.
   * @throws IOException Thrown if errors occurred loading the {@code model} or {@code vocabulary}.
   */
  public SentenceVectorsDL(final File model, final File vocabulary, final boolean lowerCase)
      throws OrtException, IOException {

    this(model, vocabulary, lowerCase, Pooling.MEAN, true, DEFAULT_MAX_LENGTH);

  }

  /**
   * Instantiates a {@link SentenceVectorsDL sentence vector generator} using ONNX models.
   *
   * @param model The file name of a sentence vectors ONNX model.
   * @param vocabulary The file name of the vocabulary file for the model.
   * @param lowerCase {@code true} for uncased models (lower casing and accent
   *     stripping during tokenization), {@code false} for cased models.
   * @param pooling How token vectors are pooled. Not used for a model with a
   *     {@code sentence_embedding} output. Must not be {@code null}.
   * @param normalize {@code true} to scale every vector to unit length.
   * @param maxLength The maximum number of tokens per input, {@code [CLS]} and {@code [SEP]}
   *     included; longer input is truncated. Must be at least {@code 2}.
   *
   * @throws IllegalArgumentException Thrown if {@code pooling} is {@code null}, if
   *     {@code maxLength} is less than {@code 2}, or if the model has no output of shape
   *     {@code [batch, hidden]} or {@code [batch, tokens, hidden]}.
   * @throws OrtException Thrown if the {@code model} cannot be loaded.
   * @throws IOException Thrown if errors occurred loading the {@code model} or {@code vocabulary}.
   */
  public SentenceVectorsDL(final File model, final File vocabulary, final boolean lowerCase,
      final Pooling pooling, final boolean normalize, final int maxLength)
      throws OrtException, IOException {

    this(model, vocabulary, lowerCase, pooling, normalize, maxLength, DEFAULT_PADDING);

  }

  /**
   * Instantiates a {@link SentenceVectorsDL sentence vector generator} using ONNX models,
   * choosing how {@link #embedAll(List)} shapes the tensors of one inference.
   *
   * @param model The file name of a sentence vectors ONNX model.
   * @param vocabulary The file name of the vocabulary file for the model.
   * @param lowerCase {@code true} for uncased models (lower casing and accent
   *     stripping during tokenization), {@code false} for cased models.
   * @param pooling How token vectors are pooled. Not used for a model with a
   *     {@code sentence_embedding} output. Must not be {@code null}.
   * @param normalize {@code true} to scale every vector to unit length.
   * @param maxLength The maximum number of tokens per input, {@code [CLS]} and {@code [SEP]}
   *     included; longer input is truncated. Must be at least {@code 2}.
   * @param padding How the tensors of one inference are shaped when the inputs of a call differ
   *     in length. Must not be {@code null}. Every strategy but
   *     {@link PaddingStrategy#EXACT_LENGTH} requires a padding token in the vocabulary.
   *
   * @throws IllegalArgumentException Thrown if {@code pooling} or {@code padding} is
   *     {@code null}, if {@code maxLength} is less than {@code 2}, if the model has no output of
   *     shape {@code [batch, hidden]} or {@code [batch, tokens, hidden]}, or if {@code padding}
   *     pads and the vocabulary has no padding token.
   * @throws OrtException Thrown if the {@code model} cannot be loaded.
   * @throws IOException Thrown if errors occurred loading the {@code model} or {@code vocabulary}.
   */
  public SentenceVectorsDL(final File model, final File vocabulary, final boolean lowerCase,
      final Pooling pooling, final boolean normalize, final int maxLength,
      final PaddingStrategy padding)
      throws OrtException, IOException {

    this(model, vocabulary, lowerCase, pooling, normalize, maxLength, padding,
        new InferenceOptions());

  }

  /**
   * {@return a {@link SentenceVectorsDL sentence vector generator} on the execution providers the
   * given {@link InferenceOptions} selects, with the {@link PaddingStrategy} derived from those
   * execution providers}
   *
   * <p>This is the entry point for a caller that has an opinion about where inference runs and none
   * about tensor shapes. {@link PaddingStrategy#defaultFor(List)} makes the choice, reading the
   * execution providers {@link ExecutionProviders#resolve(InferenceOptions)} resolves: a session on
   * an accelerator pads each batch to its longest row, six to eight times the throughput on a GPU,
   * and a session on the CPU groups by exact tokenized length, where padding runs at 0.66 to 0.81 of
   * that speed. That method has the figures, and it states what an execution provider id from an
   * addon derives. The strategy is written to the log at {@code info} next to the line naming the
   * execution providers, so the derived choice can be read off a running system.</p>
   *
   * <p>Name the strategy through
   * {@link #SentenceVectorsDL(File, File, boolean, Pooling, boolean, int, PaddingStrategy,
   * InferenceOptions)} to decide it yourself. The padding strategy is the one setting derived here,
   * and a named one always wins: no execution provider setting overrides it.</p>
   *
   * <p>This is a factory rather than another constructor because a constructor of the same arity
   * taking {@link InferenceOptions} in place of {@link PaddingStrategy} would make
   * {@code new SentenceVectorsDL(model, vocabulary, true, pooling, false, 512, null)} ambiguous, and
   * that call compiles against the constructor that takes a {@link PaddingStrategy} today.</p>
   *
   * @param model The file name of a sentence vectors ONNX model.
   * @param vocabulary The file name of the vocabulary file for the model.
   * @param lowerCase {@code true} for uncased models (lower casing and accent
   *     stripping during tokenization), {@code false} for cased models. Overridden by
   *     {@link InferenceOptions#getLowerCase()} when that is set.
   * @param pooling How token vectors are pooled. Not used for a model with a
   *     {@code sentence_embedding} output. Must not be {@code null}.
   * @param normalize {@code true} to scale every vector to unit length.
   * @param maxLength The maximum number of tokens per input, {@code [CLS]} and {@code [SEP]}
   *     included; longer input is truncated. Must be at least {@code 2}.
   * @param inferenceOptions The execution providers to run on and the session settings to create
   *     the session with, which also decide the {@link PaddingStrategy}. Must not be {@code null}.
   *
   * @throws IllegalArgumentException Thrown if {@code pooling} or {@code inferenceOptions} is
   *     {@code null}, if {@code maxLength} is less than {@code 2}, if the model has no output of
   *     shape {@code [batch, hidden]} or {@code [batch, tokens, hidden]}, or if the derived
   *     strategy pads and the vocabulary has no padding token.
   * @throws OrtException Thrown if the {@code model} cannot be loaded, or if a requested execution
   *     provider is not available in the ONNX Runtime on the classpath or cannot be initialized on
   *     the requested device.
   * @throws IOException Thrown if errors occurred loading the {@code model} or {@code vocabulary}.
   *
   * @since 3.0.0
   */
  public static SentenceVectorsDL withDerivedPadding(final File model, final File vocabulary,
      final boolean lowerCase, final Pooling pooling, final boolean normalize, final int maxLength,
      final InferenceOptions inferenceOptions)
      throws OrtException, IOException {

    // resolve() rejects a null inferenceOptions, so the derivation needs no check of its own.
    return new SentenceVectorsDL(model, vocabulary, lowerCase, pooling, normalize, maxLength,
        PaddingStrategy.defaultFor(ExecutionProviders.resolve(inferenceOptions)), inferenceOptions,
        true);

  }

  /**
   * Instantiates a {@link SentenceVectorsDL sentence vector generator} using ONNX models, on the
   * execution provider the given {@link InferenceOptions} selects.
   *
   * <p>This is the only constructor that can move inference off the CPU or change how the session
   * uses threads. Requesting {@link opennlp.dl.ExecutionProviders#CUDA} through
   * {@link InferenceOptions#setExecutionProviders(List)} creates the session with the CUDA
   * execution provider added, and so does the deprecated {@link InferenceOptions#setGpu(boolean)}.
   * Running on the GPU requires the {@code onnxruntime_gpu} runtime on the classpath, which the
   * {@code opennlp-dl-gpu} module brings in; with the CPU-only {@code onnxruntime} runtime, or with
   * a CUDA installation the runtime cannot load, this constructor throws an {@link OrtException}
   * instead of running on the CPU. An unusable device id likewise fails here rather than later.</p>
   *
   * <p>{@code inferenceOptions} is the last parameter so that it extends the constructor chain the
   * same way the earlier parameters did, and so that the other five settings keep the positions
   * they have had since they were introduced. Three groups of its values are read: the execution
   * providers, the session settings ({@link InferenceOptions#setIntraOpNumThreads(int)},
   * {@link InferenceOptions#setInterOpNumThreads(int)} and
   * {@link InferenceOptions#setOptimizationLevel(OrtSession.SessionOptions.OptLevel)}), and, if
   * {@link InferenceOptions#setLowerCase(boolean)} was called, the lower casing behavior, which
   * then wins over the {@code lowerCase} parameter. The rest of {@link InferenceOptions} describes
   * inputs this component does not have: it has no document splitting, and it always sends both an
   * attention mask and token type ids, since the encoding of a sentence-transformers model is
   * fixed. Nothing is read from the object after construction, so a caller may reuse or mutate it
   * afterwards.</p>
   *
   * @param model The file name of a sentence vectors ONNX model.
   * @param vocabulary The file name of the vocabulary file for the model.
   * @param lowerCase {@code true} for uncased models (lower casing and accent
   *     stripping during tokenization), {@code false} for cased models. Overridden by
   *     {@link InferenceOptions#getLowerCase()} when that is set.
   * @param pooling How token vectors are pooled. Not used for a model with a
   *     {@code sentence_embedding} output. Must not be {@code null}.
   * @param normalize {@code true} to scale every vector to unit length.
   * @param maxLength The maximum number of tokens per input, {@code [CLS]} and {@code [SEP]}
   *     included; longer input is truncated. Must be at least {@code 2}.
   * @param padding How the tensors of one inference are shaped when the inputs of a call differ
   *     in length. Must not be {@code null}. Every strategy but
   *     {@link PaddingStrategy#EXACT_LENGTH} requires a padding token in the vocabulary.
   * @param inferenceOptions The execution providers to run on and the session settings to create
   *     the session with. Must not be {@code null}. A default {@link InferenceOptions} requests
   *     nothing and leaves every session setting at ONNX Runtime's own, which is what every other
   *     constructor passes.
   *
   * @throws IllegalArgumentException Thrown if {@code pooling}, {@code padding} or
   *     {@code inferenceOptions} is {@code null}, if {@code maxLength} is less than {@code 2}, if
   *     the model has no output of shape {@code [batch, hidden]} or
   *     {@code [batch, tokens, hidden]}, or if {@code padding} pads and the vocabulary has no
   *     padding token.
   * @throws OrtException Thrown if the {@code model} cannot be loaded, or if a requested execution
   *     provider is not available in the ONNX Runtime on the classpath or cannot be initialized on
   *     the requested device.
   * @throws IOException Thrown if errors occurred loading the {@code model} or {@code vocabulary}.
   *
   * @since 3.0.0
   */
  public SentenceVectorsDL(final File model, final File vocabulary, final boolean lowerCase,
      final Pooling pooling, final boolean normalize, final int maxLength,
      final PaddingStrategy padding, final InferenceOptions inferenceOptions)
      throws OrtException, IOException {

    this(model, vocabulary, lowerCase, pooling, normalize, maxLength, padding, inferenceOptions,
        false);

  }

  /**
   * The constructor every other one ends at, with the origin of the {@link PaddingStrategy} added
   * so that one log entry can name it.
   *
   * @param model The file name of a sentence vectors ONNX model.
   * @param vocabulary The file name of the vocabulary file for the model.
   * @param lowerCase Whether tokenization lower cases and strips accents.
   * @param pooling How token vectors are pooled. Must not be {@code null}.
   * @param normalize {@code true} to scale every vector to unit length.
   * @param maxLength The maximum number of tokens per input. Must be at least {@code 2}.
   * @param padding How the tensors of one inference are shaped. Must not be {@code null}.
   * @param inferenceOptions The execution providers and session settings. Must not be {@code null}.
   * @param paddingDerived {@code true} if {@code padding} came from
   *     {@link PaddingStrategy#defaultFor(List)} rather than from the caller, which is what the log
   *     entry reports. It changes no other behavior: a derived strategy and the same strategy
   *     requested by name act the same.
   *
   * @throws IllegalArgumentException Thrown as the public constructors document.
   * @throws OrtException Thrown as the public constructors document.
   * @throws IOException Thrown as the public constructors document.
   */
  private SentenceVectorsDL(final File model, final File vocabulary, final boolean lowerCase,
      final Pooling pooling, final boolean normalize, final int maxLength,
      final PaddingStrategy padding, final InferenceOptions inferenceOptions,
      final boolean paddingDerived)
      throws OrtException, IOException {

    // sessionOptions() rejects a null inferenceOptions before the session is created, and it is
    // evaluated before resolveLowerCase(), so neither call sees null.
    super(model, vocabulary, sessionOptions(inferenceOptions),
        resolveLowerCase(inferenceOptions, lowerCase));
    try {
      if (pooling == null) {
        throw new IllegalArgumentException("pooling must not be null");
      }
      if (padding == null) {
        throw new IllegalArgumentException("padding must not be null");
      }
      if (maxLength < MIN_LENGTH) {
        throw new IllegalArgumentException("maxLength must be at least " + MIN_LENGTH);
      }
      this.pooling = pooling;
      this.normalize = normalize;
      this.maxLength = maxLength;
      this.padding = padding;
      // Resolved up front so a vocabulary without a padding token fails at construction rather
      // than on the first call that happens to mix lengths.
      this.padTokenId = padding == PaddingStrategy.EXACT_LENGTH ? 0 : resolvePadTokenId(vocab);
      final Map<String, NodeInfo> outputs = session.getOutputInfo();
      this.outputName = selectOutput(outputs);
      final long[] shape = ((TensorInfo) outputs.get(outputName).getInfo()).getShape();
      this.pooledOutput = shape.length == POOLED_RANK;
      this.declaredOutputShape = shape.clone();
      final long declared = shape[shape.length - 1];
      // The bootstrap run below happens while dimension is still its blank final default of 0, so
      // pinnableOutputShape returns null for it and it reads the output unpinned. That is the only
      // way round: the hidden size is what the run is being made to discover.
      this.dimension = declared > 0 && declared <= Integer.MAX_VALUE
          ? (int) declared : run(new Tokens[] {encodeTokens("")})[0].length;
      logPadding(padding, paddingDerived, inferenceOptions);
    } catch (final OrtException | RuntimeException e) {
      try {
        super.close();
      } catch (final OrtException closeFailure) {
        e.addSuppressed(closeFailure);
      }
      throw e;
    }

  }

  /**
   * Writes the {@link PaddingStrategy} of a new component to the log at {@code info}, next to the
   * execution providers it goes with.
   *
   * <p>{@code AbstractDL.configureSession} reports where the session runs; this reports the tensor
   * shapes it runs at, which is the setting that follows from it. Both are written on the ordinary
   * path, not only where something failed, because a default that cannot be observed is how the
   * wrong one stays in place for years. This package described GPU support for two years while each
   * of its sessions ran on the CPU, with no log line to contradict it.</p>
   *
   * @param padding The strategy in force.
   * @param derived {@code true} if it came from {@link PaddingStrategy#defaultFor(List)} rather than
   *     from a constructor argument.
   * @param inferenceOptions The options to read the execution providers back from. Must not be
   *     {@code null}.
   */
  private static void logPadding(final PaddingStrategy padding, final boolean derived,
      final InferenceOptions inferenceOptions) {
    final List<ExecutionProviderRequest> providers = ExecutionProviders.resolve(inferenceOptions);
    logger.info("ONNX sentence vector padding strategy: {}, {}; execution providers: {}",
        padding, derived ? "derived from the execution providers" : "chosen by the caller",
        ExecutionProviders.describe(providers));
  }

  /**
   * Generates vectors given a sentence.
   * 
   * @param sentence The input sentence.
   * @return The sentence vector.
   *
   * @throws IllegalArgumentException Thrown if {@code sentence} is {@code null}.
   * @throws OrtException Thrown if an error occurs during inference.
   */
  public float[] getVectors(final String sentence) throws OrtException {

    if (sentence == null) {
      throw new IllegalArgumentException("sentence must not be null");
    }
    return run(new Tokens[] {encode(sentence)})[0];

  }

  /**
   * {@inheritDoc}
   *
   * <p>Empty or unrecognized input is still run through the model as the wrapped
   * {@code [CLS] [SEP]} sequence rather than returning a zero vector.</p>
   */
  @Override
  public float[] embed(final CharSequence text) {
    if (text == null) {
      throw new IllegalArgumentException("text must not be null");
    }
    try {
      return run(new Tokens[] {encode(text)})[0];
    } catch (final OrtException e) {
      throw new EmbeddingException("Sentence vector inference failed.", e);
    }
  }

  /**
   * {@inheritDoc}
   *
   * <p>The inputs are tokenized up front and then distributed over inferences as the configured
   * {@link PaddingStrategy} dictates. Whichever strategy is in force, vector {@code i} is the
   * vector of input {@code i} and equals what {@link #embed(CharSequence)} returns for that input
   * on its own: a padded position is {@code 0} in the attention mask and so changes no output
   * vector. No inference exceeds {@value #MAX_BATCH_TOKEN_POSITIONS} token positions.</p>
   */
  @Override
  public float[][] embedAll(final List<? extends CharSequence> texts) {
    if (texts == null) {
      throw new IllegalArgumentException("texts must not be null");
    }
    final CharSequence[] checked = new CharSequence[texts.size()];
    for (int i = 0; i < checked.length; i++) {
      checked[i] = texts.get(i);
      if (checked[i] == null) {
        throw new IllegalArgumentException("texts[" + i + "] must not be null");
      }
    }
    final Tokens[] encoded = new Tokens[checked.length];
    for (int i = 0; i < checked.length; i++) {
      encoded[i] = encode(checked[i]);
    }
    final float[][] vectors = new float[checked.length][];
    try {
      for (final int[] group : batches(encoded)) {
        final Tokens[] batch = new Tokens[group.length];
        for (int b = 0; b < batch.length; b++) {
          batch[b] = encoded[group[b]];
        }
        final float[][] rows = run(batch);
        for (int b = 0; b < batch.length; b++) {
          vectors[group[b]] = rows[b];
        }
      }
    } catch (final OrtException e) {
      throw new EmbeddingException("Sentence vector inference failed.", e);
    }
    return vectors;
  }

  /**
   * Partitions the indices of a call into the batches that run as single inferences. Every index
   * appears in exactly one batch, so each input is embedded exactly once. No batch exceeds
   * {@value #MAX_BATCH_TOKEN_POSITIONS} token positions unless a single input is wider than that
   * on its own, in which case it forms a batch of one rather than being dropped.
   *
   * <p>Under {@link PaddingStrategy#EXACT_LENGTH} a batch holds the indices of one tokenized
   * length, in call order. Under {@link PaddingStrategy#LONGEST} the indices are ordered by
   * ascending tokenized length and cut into runs whose row count times longest row stays within
   * the bound, so one long input cannot pad out a batch of short ones. Under
   * {@link PaddingStrategy#MAX_LENGTH} every row is {@code maxLength} wide, so the runs are
   * simply consecutive groups of {@code MAX_BATCH_TOKEN_POSITIONS / maxLength} indices in call
   * order.</p>
   *
   * @param encoded The encodings of the call, in call order.
   * @return The batches, each an array of indices into {@code encoded}.
   */
  private List<int[]> batches(final Tokens[] encoded) {
    final List<int[]> groups = new ArrayList<>();
    if (padding == PaddingStrategy.EXACT_LENGTH) {
      final Map<Integer, List<Integer>> byLength = new LinkedHashMap<>();
      for (int i = 0; i < encoded.length; i++) {
        byLength.computeIfAbsent(encoded[i].ids().length, length -> new ArrayList<>()).add(i);
      }
      for (final List<Integer> group : byLength.values()) {
        // Every index in the group has the same tokenized length, so the first one gives it.
        addCapped(groups, group, encoded[group.get(0)].ids().length);
      }
      return groups;
    }
    if (padding == PaddingStrategy.MAX_LENGTH) {
      final List<Integer> all = new ArrayList<>(encoded.length);
      for (int i = 0; i < encoded.length; i++) {
        all.add(i);
      }
      addCapped(groups, all, maxLength);
      return groups;
    }
    final Integer[] order = new Integer[encoded.length];
    for (int i = 0; i < order.length; i++) {
      order[i] = i;
    }
    Arrays.sort(order, Comparator.comparingInt(i -> encoded[i].ids().length));
    final List<Integer> current = new ArrayList<>();
    int longest = 0;
    for (final int index : order) {
      final int width = Math.max(longest, encoded[index].ids().length);
      if (!current.isEmpty() && (long) (current.size() + 1) * width > MAX_BATCH_TOKEN_POSITIONS) {
        groups.add(toArray(current));
        current.clear();
        longest = 0;
      }
      current.add(index);
      longest = Math.max(longest, encoded[index].ids().length);
    }
    if (!current.isEmpty()) {
      groups.add(toArray(current));
    }
    return groups;
  }

  /**
   * Adds indices that all run at the same row width as batches of at most
   * {@value #MAX_BATCH_TOKEN_POSITIONS} token positions, in the order given. A width above the
   * bound yields batches of one row.
   *
   * @param groups The batches to add to.
   * @param indices The indices to split, in the order they must run.
   * @param width The padded row length every one of them runs at.
   */
  private static void addCapped(final List<int[]> groups, final List<Integer> indices,
      final int width) {
    final int rows = Math.max(1, MAX_BATCH_TOKEN_POSITIONS / Math.max(1, width));
    for (int from = 0; from < indices.size(); from += rows) {
      groups.add(toArray(indices.subList(from, Math.min(from + rows, indices.size()))));
    }
  }

  /**
   * {@return a new array of the given indices, in their list order}
   *
   * @param indices The indices to copy.
   */
  private static int[] toArray(final List<Integer> indices) {
    final int[] array = new int[indices.size()];
    for (int i = 0; i < array.length; i++) {
      array[i] = indices.get(i);
    }
    return array;
  }

  /**
   * {@return the shapes of the inferences {@link #embedAll(List)} would run for these inputs, each
   * as {@code {rows, padded row length}}, in the order they would run}
   *
   * <p>This is the batch plan itself, read without running the session. It exists so that tests
   * can assert how a {@link PaddingStrategy} shapes the tensors and how many times the session
   * runs, which the returned vectors alone cannot show: all three strategies return the same
   * vectors and differ only here. It is deliberately not public.</p>
   *
   * @param texts The inputs of the call. Must not be {@code null} or contain {@code null}.
   * @throws IllegalArgumentException Thrown if {@code texts} or any element is {@code null}.
   */
  final int[][] batchShapes(final List<? extends CharSequence> texts) {
    if (texts == null) {
      throw new IllegalArgumentException("texts must not be null");
    }
    final Tokens[] encoded = new Tokens[texts.size()];
    for (int i = 0; i < encoded.length; i++) {
      if (texts.get(i) == null) {
        throw new IllegalArgumentException("texts[" + i + "] must not be null");
      }
      encoded[i] = encode(texts.get(i));
    }
    final List<int[]> groups = batches(encoded);
    final int[][] shapes = new int[groups.size()][];
    for (int g = 0; g < shapes.length; g++) {
      final Tokens[] batch = new Tokens[groups.get(g).length];
      for (int b = 0; b < batch.length; b++) {
        batch[b] = encoded[groups.get(g)[b]];
      }
      shapes[g] = new int[] {batch.length, width(batch)};
    }
    return shapes;
  }

  /**
   * {@inheritDoc}
   *
   * <p>Read from the model's declared output shape, or from one inference at construction if
   * the model declares the hidden dimension dynamically.</p>
   */
  @Override
  public int dimension() {
    return dimension;
  }

  /**
   * Closes the ONNX session. An {@link OrtException} from the session is rethrown as the cause
   * of an {@link UncheckedIOException}.
   */
  @Override
  public void close() {
    try {
      super.close();
    } catch (final OrtException e) {
      throw new UncheckedIOException(new IOException("Cannot close the ONNX session.", e));
    }
  }

  /**
   * Encodes a text and truncates it to the maximum length, keeping the final {@code [SEP]}.
   *
   * @param text The text to encode. Must not be {@code null}.
   * @return The encoded text of at most the maximum length.
   */
  private Tokens encode(final CharSequence text) {
    final Tokens tokens = encodeTokens(text);
    final int length = tokens.ids().length;
    if (length <= maxLength) {
      return tokens;
    }
    final String[] pieces = new String[maxLength];
    final long[] ids = new long[maxLength];
    final long[] mask = new long[maxLength];
    final long[] types = new long[maxLength];
    System.arraycopy(tokens.tokens(), 0, pieces, 0, maxLength - 1);
    System.arraycopy(tokens.ids(), 0, ids, 0, maxLength - 1);
    System.arraycopy(tokens.mask(), 0, mask, 0, maxLength);
    System.arraycopy(tokens.types(), 0, types, 0, maxLength);
    pieces[maxLength - 1] = tokens.tokens()[length - 1];
    ids[maxLength - 1] = tokens.ids()[length - 1];
    return new Tokens(pieces, ids, mask, types);
  }

  /**
   * Runs one inference over a batch of encodings and returns one sentence vector per encoding, in
   * order.
   *
   * <p>Rows shorter than the batch's row width, which {@link #width(Tokens[])} decides from the
   * configured {@link PaddingStrategy}, are padded to it by
   * {@link #stage(Tokens[], int, LongBuffer, LongBuffer, LongBuffer)}: {@code input_ids} with the
   * vocabulary's padding token id, {@code attention_mask} with {@code 0} and {@code token_type_ids}
   * with {@code 0}. A padded position therefore changes no output vector, whether the sentence vector
   * comes from an in-graph {@code sentence_embedding} output that honors the mask or from
   * {@link #pool(FloatBuffer, int, int, long[])}, which reads only the positions the row's own mask
   * covers. Every row's vector equals the vector the row would get in a batch of its own, exactly on
   * the CPU provider and to about {@code 1.2e-4} absolute on a GPU one, where cuBLAS picks its
   * blocking from the tensor width; the worst deviation measured was {@code 5.15e-7} in one minus
   * cosine similarity. Anyone caching or persisting vectors produced on a GPU has to key the cache
   * by the batch shape too, or accept that tolerance.</p>
   *
   * <p>The output is read through a pinned output tensor whenever
   * {@link #pinnableOutputShape(int, int)} can name its shape, which is the ordinary case: the row
   * count and the row width are this method's own choices and the hidden size is
   * {@link #dimension()}. ONNX Runtime then writes the results into a direct buffer this class
   * already owns and the vectors are pooled straight out of it, with no copy and none of the nested
   * arrays a <code>{rows, width, hidden}</code> output would otherwise be materialized into, over
   * half of which a padded batch never reads. Where the shape cannot be named, which is the run that
   * discovers the hidden size and any model declaring an output dimension this class would pin
   * differently, the output is read through one flat copy instead and the pooling is identical.</p>
   *
   * @param batch The encodings, of any lengths.
   * @return The sentence vectors.
   * @throws OrtException Thrown if an error occurs during inference.
   */
  private float[][] run(final Tokens[] batch) throws OrtException {

    final int length = width(batch);
    // The tensors, the run and the release of every native handle belong to OnnxInference; what is
    // left here is what each row of the batch puts into the tensors and what this class makes of the
    // numbers that come back. The rows are written straight into the direct memory the inputs are
    // staged in, so the three flat long[rows * length] arrays this used to build, around 170 KB of
    // them at a batch of 128 rows padded to 56 positions, are not built and not copied out of. The
    // values are read inside the reader, while the output memory is still the run's, and what leaves
    // the reader is the pooled vectors, which are this class's own arrays.
    return inference.run(new long[] {batch.length, length},
        (ids, mask, types) -> stage(batch, length, ids, mask, types), outputName,
        pinnableOutputShape(batch.length, length),
        (values, shape) -> vectors(batch, values, shape));
  }

  /**
   * Writes one inference's rows into the three input buffers, row after row, padding each row out to
   * the width of the run.
   *
   * <p>Every position of all three buffers is written, which is what the buffers being reused across
   * inferences requires: a position left unwritten would hold whatever the previous inference of this
   * thread put there, and the model would embed that instead. The padded tail of a row is written too
   * rather than assumed to be zero already, including {@code token_type_ids}, which is zero for every
   * row of every batch this class stages and would therefore be the one input a shortcut looked safe
   * for. It is not safe: what makes it zero is this method, so skipping it would make the values of
   * one inference depend on what the inferences before it happened to write.</p>
   *
   * @param batch The encodings of the inference, in the order their rows are staged.
   * @param length The row width of the run, at least the longest row of {@code batch}.
   * @param ids The {@code input_ids} buffer, positioned at the first value to write.
   * @param mask The {@code attention_mask} buffer, positioned at the first value to write.
   * @param types The {@code token_type_ids} buffer, positioned at the first value to write.
   */
  private void stage(final Tokens[] batch, final int length, final LongBuffer ids,
      final LongBuffer mask, final LongBuffer types) {
    for (final Tokens row : batch) {
      final int rowLength = row.ids().length;
      ids.put(row.ids(), 0, rowLength);
      mask.put(row.mask(), 0, rowLength);
      types.put(row.types(), 0, rowLength);
      // A padded position carries the vocabulary's padding token id, a 0 attention mask so that no
      // output vector depends on it, and a 0 token type. Only the ids differ from 0, and only for a
      // vocabulary whose padding token is not id 0.
      for (int p = rowLength; p < length; p++) {
        ids.put(padTokenId);
        mask.put(0L);
        types.put(0L);
      }
    }
  }

  /**
   * {@return the shape to pin the output of one inference at, or {@code null} to read it unpinned}
   *
   * <p>The shape of the output of this model is <code>{rows, hidden}</code> for an in-graph pooled
   * output and <code>{rows, width, hidden}</code> for a token output. Both are known before the run:
   * {@code rows} and {@code width} are what {@link #run(Tokens[])} chose and {@code hidden} is
   * {@link #dimension()}. The candidate is then checked against every dimension the model declares
   * statically, so a model that fixes a dimension where this would pin a different one reads its
   * output unpinned rather than failing the run. {@code null} comes back while the hidden size is
   * still being discovered, and for any output rank other than 2 or 3.</p>
   *
   * @param rows The row count of the inference.
   * @param width The row width of the inference.
   */
  private long[] pinnableOutputShape(final int rows, final int width) {
    if (dimension <= 0 || rows <= 0 || width <= 0) {
      return null;
    }
    final long[] candidate;
    if (declaredOutputShape.length == POOLED_RANK) {
      candidate = new long[] {rows, dimension};
    } else if (declaredOutputShape.length == TOKEN_RANK) {
      candidate = new long[] {rows, width, dimension};
    } else {
      return null;
    }
    for (int d = 0; d < candidate.length; d++) {
      if (declaredOutputShape[d] > 0 && declaredOutputShape[d] != candidate[d]) {
        return null;
      }
    }
    return candidate;
  }

  /**
   * Pools one inference's output into one sentence vector per row, reading the flat values of the
   * output by index.
   *
   * @param batch The encodings of the inference, in the order their rows were staged.
   * @param values The values of the output, row major. Valid only for the duration of this call.
   * @param shape The shape of the output as the run reports it.
   * @return The sentence vectors, in the order of {@code batch}.
   *
   * @throws OrtException Thrown if the output does not have the shape this batch requires, which is
   *     a model-contract violation rather than an inference failure.
   */
  private float[][] vectors(final Tokens[] batch, final FloatBuffer values, final long[] shape)
      throws OrtException {
    final int rank = shape.length;
    if (rank != (pooledOutput ? POOLED_RANK : TOKEN_RANK)) {
      throw new OrtException("The model returned an output of rank " + rank + " where rank "
          + (pooledOutput ? POOLED_RANK : TOKEN_RANK) + " was selected at construction.");
    }
    if (shape[0] != batch.length) {
      throw new OrtException("The model returned " + shape[0] + " rows for a batch of "
          + batch.length + ".");
    }
    final int hidden = (int) shape[rank - 1];
    // The width of the output, which for a token output is the padded row width of the run, so a
    // row's own positions are the first mask.length of its own stride.
    final int width = pooledOutput ? 1 : (int) shape[1];
    final float[][] vectors = new float[batch.length][];
    for (int b = 0; b < batch.length; b++) {
      if (pooledOutput) {
        vectors[b] = slice(values, b * hidden, hidden);
      } else {
        final long[] rowMask = batch[b].mask();
        if (rowMask.length > width) {
          throw new OrtException("Row " + b + " has " + rowMask.length
              + " token positions but the output is only " + width + " wide.");
        }
        vectors[b] = pool(values, b * width * hidden, hidden, rowMask);
      }
      if (normalize) {
        scaleToUnitLength(vectors[b]);
      }
    }
    return vectors;
  }

  /**
   * {@return the row length of one inference, the second dimension of its tensors}
   *
   * <p>{@link PaddingStrategy#MAX_LENGTH} fixes it at the configured maximum whatever the rows
   * hold, so every inference has the same shape. The other strategies take the longest row of the
   * batch, which under {@link PaddingStrategy#EXACT_LENGTH} is the length of every row in it, so
   * nothing is padded.</p>
   *
   * @param batch The encodings of one inference, at least one.
   */
  private int width(final Tokens[] batch) {
    if (padding == PaddingStrategy.MAX_LENGTH) {
      return maxLength;
    }
    int longest = 0;
    for (final Tokens row : batch) {
      longest = Math.max(longest, row.ids().length);
    }
    return longest;
  }

  /**
   * Pools the token vectors of one input into its sentence vector, indexing the flat values of the
   * output rather than walking nested arrays.
   *
   * <p>Only the first {@code mask.length} positions of the row are read, so the trailing padded
   * positions of a batch row are ignored and a row pools to the same vector at any batch width. That
   * is the whole reason the read is flat: the positions past a row's own mask are the ones a padded
   * batch never looks at, and building an array per one of them is what made a batch of 128 rows
   * allocate 7168 arrays to read 3275 of them.</p>
   *
   * @param values The values of the output, row major.
   * @param base The index in {@code values} where this row's token vectors begin.
   * @param hidden The hidden size, which is the stride from one token vector to the next.
   * @param mask The attention mask of the input, without padding.
   * @return A new array holding the sentence vector.
   */
  private float[] pool(final FloatBuffer values, final int base, final int hidden,
      final long[] mask) {
    if (pooling == Pooling.CLS) {
      return slice(values, base, hidden);
    }
    final float[] sum = new float[hidden];
    int count = 0;
    for (int t = 0; t < mask.length; t++) {
      if (mask[t] != 0) {
        final int token = base + t * hidden;
        for (int d = 0; d < hidden; d++) {
          sum[d] += values.get(token + d);
        }
        count++;
      }
    }
    for (int d = 0; d < hidden; d++) {
      sum[d] /= Math.max(count, 1);
    }
    return sum;
  }

  /**
   * {@return {@code length} values of {@code values} from {@code offset}, as a new array}
   *
   * <p>Read with the absolute {@link FloatBuffer#get(int)} rather than a bulk get, because the
   * position of the buffer is the run's and not this class's to move: on the pinned path it is the
   * buffer ONNX Runtime wrote into and the next run of this thread reuses it as it stands.</p>
   *
   * @param values The values of the output, row major.
   * @param offset The index to read from.
   * @param length The number of values to read.
   */
  private static float[] slice(final FloatBuffer values, final int offset, final int length) {
    final float[] copy = new float[length];
    for (int i = 0; i < length; i++) {
      copy[i] = values.get(offset + i);
    }
    return copy;
  }

  /**
   * Scales a vector to unit length in place. A zero vector is left as it is.
   *
   * @param vector The vector to scale.
   */
  private void scaleToUnitLength(final float[] vector) {
    double squares = 0;
    for (final float value : vector) {
      squares += (double) value * value;
    }
    if (squares > 0) {
      final double norm = Math.sqrt(squares);
      for (int d = 0; d < vector.length; d++) {
        vector[d] = (float) (vector[d] / norm);
      }
    }
  }

  /**
   * Selects the output the sentence vector is read from: a {@code sentence_embedding} output of
   * rank 2 if present, otherwise the first output of rank 3, otherwise the first of rank 2.
   *
   * @param outputs The outputs of the model, in declaration order.
   * @return The name of the selected output.
   * @throws IllegalArgumentException Thrown if no output has rank 2 or 3.
   */
  private String selectOutput(final Map<String, NodeInfo> outputs) {
    if (rank(outputs.get(SENTENCE_EMBEDDING)) == POOLED_RANK) {
      return SENTENCE_EMBEDDING;
    }
    String pooled = null;
    for (final Map.Entry<String, NodeInfo> output : outputs.entrySet()) {
      final int rank = rank(output.getValue());
      if (rank == TOKEN_RANK) {
        return output.getKey();
      }
      if (rank == POOLED_RANK && pooled == null) {
        pooled = output.getKey();
      }
    }
    if (pooled == null) {
      throw new IllegalArgumentException(
          "The model has no output of shape [batch, hidden] or [batch, tokens, hidden].");
    }
    return pooled;
  }

  /**
   * {@return the rank of a float tensor output, or {@code -1} if the output is missing or not a
   * float tensor}
   *
   * @param output The output to check, or {@code null}.
   */
  private int rank(final NodeInfo output) {
    if (output != null && output.getInfo() instanceof TensorInfo tensor
        && tensor.type == OnnxJavaType.FLOAT) {
      return tensor.getShape().length;
    }
    return -1;
  }

}
