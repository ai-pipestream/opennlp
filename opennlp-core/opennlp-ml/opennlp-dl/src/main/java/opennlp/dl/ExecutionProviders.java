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

package opennlp.dl;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.TreeSet;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import opennlp.tools.util.ext.Providers;

/**
 * Resolves {@link ExecutionProviderRequest requests} to
 * {@link ExecutionProviderConfigurer configurers} and appends the execution providers they build to
 * a session's options, in the order requested.
 *
 * <p>ONNX Runtime keeps the execution providers of a session in the order they were appended and
 * runs a node on the first one that accepts it, so an ordered list is the model, not a single
 * choice: {@code [cuda, cpu]} means "CUDA where it can, CPU for the rest". Nothing here retries or
 * substitutes. A request that cannot be registered is reported, because ONNX Runtime does not fall
 * back to the CPU either, and a session that silently ran somewhere else than the caller asked for
 * is the failure this class exists to prevent.</p>
 *
 * <p>An id is resolved against the {@link ExecutionProviderConfigurer} SPI first and against the
 * two built-in configurers, {@value #CPU} and {@value #CUDA}, second. So an addon contributes a new
 * id by registering a configurer, and may also supersede a built-in id by registering a configurer
 * under it, which {@link Providers#allowedKey()} and {@link Providers#disabledKey()} let a
 * deployment control. The built-in pair needs no registration of its own and cannot be lost with a
 * service file, which matters for an uber-jar.</p>
 *
 * @see InferenceOptions#setExecutionProviders(List)
 * @since 3.0.0
 */
public final class ExecutionProviders {

  private static final Logger logger = LoggerFactory.getLogger(ExecutionProviders.class);

  /** The id of the built-in CPU configurer, {@link CpuExecutionProviderConfigurer}. */
  public static final String CPU = CpuExecutionProviderConfigurer.ID;

  /** The id of the built-in CUDA configurer, {@link CudaExecutionProviderConfigurer}. */
  public static final String CUDA = CudaExecutionProviderConfigurer.ID;

  /** How {@link #describe(List)} renders a list that asks for nothing. */
  private static final String NO_REQUEST =
      "none requested, so the ONNX Runtime default placement, which is the CPU";

  private static final String SEPARATOR = ", ";

  /**
   * The configurers {@code opennlp-dl} supplies itself. They are held here rather than in a
   * service file so that they are present however the jars were assembled. Both are immutable and
   * hold no native state, which is what the SPI requires of every configurer.
   */
  private static final Map<String, ExecutionProviderConfigurer> BUILT_IN =
      Map.of(CPU, new CpuExecutionProviderConfigurer(), CUDA, new CudaExecutionProviderConfigurer());

  /**
   * The execution provider ids this class classifies as an accelerator itself, which is the fallback
   * of {@link #runsOnAccelerator(List)} for a request whose configurer states no placement. That is
   * {@value #CUDA}, the one accelerator {@code opennlp-dl} can register through the base ONNX
   * Runtime API.
   *
   * <p>This set holds built-in ids and does not grow with the addons. An execution provider from an
   * addon answers for itself, through
   * {@link ExecutionProviderConfigurer#placement(opennlp.tools.util.ext.ProviderSpec)}, because the
   * placement of several of them is not in the id at all: ONNX Runtime registers OpenVINO with a
   * device type, so one id covers {@code GPU.0} and {@code CPU}. A configurer that states nothing,
   * which is every configurer written before that method existed, lands here, and an id that is not
   * in this set keeps the conservative answer rather than a guess.</p>
   */
  private static final Set<String> ACCELERATOR_IDS = Set.of(CUDA);

  private ExecutionProviders() {
  }

  /**
   * Resolves the execution providers an {@link InferenceOptions} asks for, applying the precedence
   * between the list and the deprecated GPU flag.
   *
   * <p>{@link InferenceOptions#setExecutionProviders(List)} wins whenever it is not empty, and
   * {@link InferenceOptions#setGpu(boolean)} is read only when it is empty. An empty list is the
   * state of an {@link InferenceOptions} that was never told anything about execution providers, so
   * it is the only state in which the deprecated flag can still be the caller's whole intent; once
   * a list is given, the flag is a leftover of an older configuration and is ignored rather than
   * silently prepended to a list whose order the caller chose.</p>
   *
   * @param inferenceOptions The options to read. Must not be {@code null}.
   * @return The requests to append, in order. Empty if nothing is requested, in which case ONNX
   *     Runtime places the session on the CPU by itself. Unmodifiable and never {@code null}.
   * @throws IllegalArgumentException Thrown if {@code inferenceOptions} is {@code null}.
   */
  public static List<ExecutionProviderRequest> resolve(final InferenceOptions inferenceOptions) {
    if (inferenceOptions == null) {
      throw new IllegalArgumentException("The inferenceOptions must not be null.");
    }
    final List<ExecutionProviderRequest> requested = inferenceOptions.getExecutionProviders();
    if (!requested.isEmpty()) {
      return requested;
    }
    return legacyGpuRequest(inferenceOptions);
  }

  /**
   * Maps the deprecated GPU flag to the request it stands for.
   *
   * @param inferenceOptions The options to read. Must not be {@code null}.
   * @return A single CUDA request on the configured device, or an empty list if the flag is not
   *     set. Unmodifiable and never {@code null}.
   */
  @SuppressWarnings({"deprecation", "removal"})
  private static List<ExecutionProviderRequest> legacyGpuRequest(
      final InferenceOptions inferenceOptions) {
    if (!inferenceOptions.isGpu()) {
      return List.of();
    }
    return List.of(ExecutionProviderRequest.of(CUDA,
        Map.of(CudaExecutionProviderConfigurer.DEVICE_ID_OPTION,
            Integer.toString(inferenceOptions.getGpuDeviceId()))));
  }

  /**
   * {@return whether the first of {@code requests} names an accelerator execution provider, which is
   * the test a setting derived from the placement of a session comes down to}
   *
   * <p>The first request answers the question because ONNX Runtime offers each node to the execution
   * providers in the order they were appended and passes on only what one rejects. A list of
   * {@code [cuda, cpu]} therefore runs an encoder graph on the GPU, and this returns {@code true}
   * for it, while {@code [cpu, cuda]} runs on the CPU, because the CPU execution provider accepts
   * any node and leaves the one behind it with none, and this returns {@code false}.</p>
   *
   * <p>The configurer of that first request answers the question, through
   * {@link ExecutionProviderConfigurer#placement(opennlp.tools.util.ext.ProviderSpec)}, and it is
   * handed the provider options of the request because the answer is not always in the id: the
   * OpenVINO execution provider is registered with a device type, so {@code GPU.0} and {@code CPU}
   * are one id on two placements. A configurer that states
   * {@link ExecutionProviderPlacement#UNSPECIFIED}, which is the default and therefore the answer of
   * every configurer written before that method existed, leaves the decision to the
   * {@code ACCELERATOR_IDS} of this class, and so does an id no configurer answers to at all.</p>
   *
   * <p>An empty list returns {@code false}: ONNX Runtime places such a session on the CPU by itself.
   * So does an unclassified id, which is the conservative answer rather than a guess.</p>
   *
   * <p>This resolves configurers, so it must not be called in a loop: like
   * {@link #addTo(OrtSession.SessionOptions, List)} it costs one service lookup, which belongs at the
   * construction of a component and not in a request.</p>
   *
   * @param requests The requests, in priority order, as {@link #resolve(InferenceOptions)} returns
   *     them. Must not be {@code null} or start with a {@code null} element.
   * @throws IllegalArgumentException Thrown if {@code requests} is {@code null} or its first element
   *     is {@code null}.
   * @see opennlp.dl.vectors.PaddingStrategy#defaultFor(List)
   * @since 3.0.0
   */
  public static boolean runsOnAccelerator(final List<ExecutionProviderRequest> requests) {
    if (requests == null) {
      throw new IllegalArgumentException("The requests must not be null.");
    }
    if (requests.isEmpty()) {
      return false;
    }
    final ExecutionProviderRequest first = requests.get(0);
    if (first == null) {
      throw new IllegalArgumentException("The requests must not hold a null element.");
    }
    final ExecutionProviderPlacement stated = placementOf(first);
    if (stated == ExecutionProviderPlacement.ACCELERATOR) {
      return true;
    }
    if (stated == ExecutionProviderPlacement.CPU) {
      return false;
    }
    return ACCELERATOR_IDS.contains(first.id());
  }

  /**
   * Asks the configurer of one request where its session would run.
   *
   * <p>A configurer that is not registered and not built in, one that returns {@code null} and one
   * that fails leave the placement {@link ExecutionProviderPlacement#UNSPECIFIED}. The last two are
   * bugs in an addon, and they are reported rather than thrown: the caller is deriving a default, and
   * a session that would have run is worth more than one that failed to start over a default. The
   * request is registered right afterwards, by {@link #addTo(OrtSession.SessionOptions, List)}, which
   * is where a broken configurer does fail the construction.</p>
   *
   * @param request The request. Must not be {@code null}.
   * @return The placement its configurer states. Never {@code null}.
   */
  private static ExecutionProviderPlacement placementOf(final ExecutionProviderRequest request) {
    final ExecutionProviderConfigurer configurer = Providers.of(ExecutionProviderConfigurer.class)
        .byName(request.id()).orElseGet(() -> BUILT_IN.get(request.id()));
    if (configurer == null) {
      return ExecutionProviderPlacement.UNSPECIFIED;
    }
    final ExecutionProviderPlacement placement;
    try {
      placement = configurer.placement(request.spec());
    } catch (final RuntimeException e) {
      logger.warn("The {} for the execution provider id '{}' failed to state a placement, so the "
          + "default derived from it stays the conservative one: {}",
          ExecutionProviderConfigurer.class.getSimpleName(), request.id(), e.toString());
      return ExecutionProviderPlacement.UNSPECIFIED;
    }
    if (placement == null) {
      logger.warn("The {} for the execution provider id '{}' stated no placement at all, which its "
          + "contract does not allow, so the default derived from it stays the conservative one.",
          ExecutionProviderConfigurer.class.getSimpleName(), request.id());
      return ExecutionProviderPlacement.UNSPECIFIED;
    }
    return placement;
  }

  /**
   * Appends the requested execution providers to {@code sessionOptions}, in the order of
   * {@code requests}.
   *
   * <p>The ONNX Runtime environment is initialized before the first execution provider is added,
   * even though the session is created later. Loading a shared execution provider library goes
   * through ONNX Runtime's default logger, which only exists once an {@link OrtEnvironment} has
   * been created, so on a JVM where nothing has touched ONNX Runtime yet {@code addCUDA} otherwise
   * fails with "Attempt to use DefaultLogger but none has been registered" and, on a second
   * attempt, with "Failed to load shared library". The environment is a process-wide singleton that
   * every component fetches anyway, so fetching it here costs nothing. It is fetched only when
   * there is something to append, so an empty list leaves the path of a plain CPU session exactly
   * as it was.</p>
   *
   * @param sessionOptions The session options to append to. Must not be {@code null}.
   * @param requests The requests, in priority order. Must not be {@code null} or hold a
   *     {@code null} element.
   * @throws IllegalArgumentException Thrown if an argument is {@code null}, if {@code requests}
   *     holds a {@code null} element, if no configurer is registered for the id of a request, or if
   *     a configurer rejects the provider options of its request.
   * @throws OrtException Thrown if ONNX Runtime cannot register a requested execution provider.
   *     ONNX Runtime does not fall back to the CPU in that case, and neither does this.
   * @throws UncheckedIOException Thrown if a configurer from an addon reads a resource while
   *     building its execution provider and that read fails. The built-in configurers read none.
   */
  public static void addTo(final OrtSession.SessionOptions sessionOptions,
                           final List<ExecutionProviderRequest> requests) throws OrtException {
    if (sessionOptions == null) {
      throw new IllegalArgumentException("The sessionOptions must not be null.");
    }
    if (requests == null) {
      throw new IllegalArgumentException("The requests must not be null.");
    }
    if (requests.isEmpty()) {
      return;
    }
    OrtEnvironment.getEnvironment();
    // A fresh lookup per call, as Providers requires: its instances read the class path of the
    // thread that creates them and are not meant to be cached in a static field. A session is
    // created rarely enough that one service lookup per session is not worth a cache that could
    // outlive the class loader it scanned.
    final Providers<ExecutionProviderConfigurer> providers =
        Providers.of(ExecutionProviderConfigurer.class);
    for (final ExecutionProviderRequest request : requests) {
      if (request == null) {
        throw new IllegalArgumentException("The requests must not hold a null element.");
      }
      final ExecutionProvider provider;
      try {
        provider = configurer(providers, request.id()).create(request.spec());
      } catch (final IOException e) {
        throw new UncheckedIOException(
            "The execution provider '" + request.id() + "' could not be built.", e);
      }
      if (provider == null) {
        throw new IllegalArgumentException("The configurer for the execution provider id '"
            + request.id() + "' returned no execution provider.");
      }
      provider.addTo(sessionOptions);
    }
  }

  /**
   * Renders a request list for a log entry or a message: the ids in order, each followed by the
   * names of its provider options. The option values are left out, as
   * {@link opennlp.tools.util.ext.ProviderSpec#toString()} leaves them out, since a provider option
   * can carry a credential.
   *
   * @param requests The requests, in priority order. Must not be {@code null}.
   * @return The rendering, which states that ONNX Runtime decides if {@code requests} is empty.
   *     Never {@code null}.
   * @throws IllegalArgumentException Thrown if {@code requests} is {@code null}.
   */
  public static String describe(final List<ExecutionProviderRequest> requests) {
    if (requests == null) {
      throw new IllegalArgumentException("The requests must not be null.");
    }
    if (requests.isEmpty()) {
      return NO_REQUEST;
    }
    final StringBuilder described = new StringBuilder();
    for (final ExecutionProviderRequest request : requests) {
      if (described.length() > 0) {
        described.append(SEPARATOR);
      }
      described.append(request);
    }
    return described.toString();
  }

  /**
   * Resolves the configurer for one id: a registered one if there is one, a built-in one
   * otherwise.
   *
   * @param providers The SPI lookup. Must not be {@code null}.
   * @param id The requested id. Must not be {@code null}.
   * @return The configurer. Never {@code null}.
   * @throws IllegalArgumentException Thrown if no configurer answers to {@code id}.
   */
  private static ExecutionProviderConfigurer configurer(
      final Providers<ExecutionProviderConfigurer> providers, final String id) {
    final Optional<ExecutionProviderConfigurer> registered = providers.byName(id);
    if (registered.isPresent()) {
      return registered.get();
    }
    final ExecutionProviderConfigurer builtIn = BUILT_IN.get(id);
    if (builtIn != null) {
      return builtIn;
    }
    throw new IllegalArgumentException("No " + ExecutionProviderConfigurer.class.getSimpleName()
        + " answers to the execution provider id '" + id + "'. These ids are known here: "
        + available(providers) + ". An id is case-sensitive, and an execution provider other than "
        + BUILT_IN.keySet().stream().sorted().toList() + " comes from an addon that registers a "
        + ExecutionProviderConfigurer.class.getSimpleName() + " in META-INF/services/"
        + ExecutionProviderConfigurer.class.getName() + ", so check that its jar is on the class "
        + "path and that " + providers.disabledKey() + " does not name it.");
  }

  /**
   * Lists the ids a request could use here, for the message of an unresolved id.
   *
   * @param providers The SPI lookup. Must not be {@code null}.
   * @return The built-in ids and the names of the registered configurers, the unavailable and the
   *     disabled ones included, sorted, each registered one followed by the class that provides
   *     it. Never {@code null}.
   */
  private static String available(final Providers<ExecutionProviderConfigurer> providers) {
    final Map<String, List<String>> ids = new LinkedHashMap<>();
    for (final String builtIn : new TreeSet<>(BUILT_IN.keySet())) {
      ids.put(builtIn, new ArrayList<>(List.of("built in")));
    }
    for (final ExecutionProviderConfigurer configurer : providers.installed()) {
      ids.computeIfAbsent(configurer.name(), name -> new ArrayList<>())
          .add(configurer.getClass().getName());
    }
    return ids.toString();
  }
}
