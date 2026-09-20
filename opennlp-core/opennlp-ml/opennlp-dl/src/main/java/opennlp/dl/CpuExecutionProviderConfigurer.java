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

import ai.onnxruntime.OrtEnvironment;

import opennlp.tools.util.ext.ProviderSpec;

/**
 * Registers the ONNX Runtime CPU execution provider under the id {@value #ID}. This is one of the
 * two configurers {@code opennlp-dl} supplies itself, because
 * {@link ai.onnxruntime.OrtSession.SessionOptions#addCPU(boolean)} is part of the base
 * {@code onnxruntime} API and needs no further artifact.
 *
 * <p>Requesting it is only needed to place the CPU explicitly or to turn its arena allocator off.
 * ONNX Runtime appends the CPU execution provider as the last fallback of every session on its
 * own, so an empty request list already runs on the CPU, and a list that ends in a request for
 * {@value #ID} says the same thing out loud.</p>
 *
 * <p>The only option is {@value #USE_ARENA_OPTION}. Unlike the option map of an accelerator, which
 * ONNX Runtime parses itself, this one is read here, so an unknown name is reported rather than
 * ignored.</p>
 *
 * @since 3.0.0
 */
public final class CpuExecutionProviderConfigurer implements ExecutionProviderConfigurer {

  /** The id this configurer answers to. */
  public static final String ID = "cpu";

  /**
   * The option that keeps the CPU arena allocator, {@code true} by default, which is ONNX
   * Runtime's own default, or {@code false} to allocate every buffer on its own. Turning the arena
   * off lowers the resident footprint of a process that holds many sessions and raises the
   * allocation cost of every inference.
   */
  public static final String USE_ARENA_OPTION = "use_arena";

  private static final String TRUE = "true";
  private static final String FALSE = "false";

  /** {@inheritDoc} */
  @Override
  public String name() {
    return ID;
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

  /** {@inheritDoc} */
  @Override
  public boolean supports(final ProviderSpec spec) {
    if (spec == null) {
      throw new IllegalArgumentException("spec must not be null");
    }
    return spec.location().isEmpty() && spec.hasOnlyOptions(USE_ARENA_OPTION);
  }

  /**
   * {@inheritDoc}
   * The returned {@link ExecutionProvider} appends the CPU execution provider with the arena
   * setting this spec asks for.
   *
   * @throws IllegalArgumentException Thrown if {@code spec} is {@code null}, carries a location,
   *     holds an option other than {@value #USE_ARENA_OPTION}, or gives that option a value that
   *     is neither {@code true} nor {@code false}.
   */
  @Override
  public ExecutionProvider create(final ProviderSpec spec) {
    if (spec == null) {
      throw new IllegalArgumentException("spec must not be null");
    }
    if (spec.location().isPresent()) {
      throw new IllegalArgumentException(
          "the " + ID + " execution provider takes no location: " + spec);
    }
    for (final String option : spec.options().keySet()) {
      if (!USE_ARENA_OPTION.equals(option)) {
        throw new IllegalArgumentException("the " + ID + " execution provider accepts only the "
            + "option '" + USE_ARENA_OPTION + "', not '" + option + "'");
      }
    }
    final String value = spec.option(USE_ARENA_OPTION, TRUE);
    if (!TRUE.equals(value) && !FALSE.equals(value)) {
      throw new IllegalArgumentException("the " + ID + " execution provider option '"
          + USE_ARENA_OPTION + "' must be " + TRUE + " or " + FALSE);
    }
    final boolean useArena = TRUE.equals(value);
    return sessionOptions -> {
      if (sessionOptions == null) {
        throw new IllegalArgumentException("sessionOptions must not be null");
      }
      sessionOptions.addCPU(useArena);
    };
  }
}
