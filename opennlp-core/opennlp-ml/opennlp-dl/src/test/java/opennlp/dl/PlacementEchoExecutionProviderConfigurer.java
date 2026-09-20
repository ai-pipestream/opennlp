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

import opennlp.tools.util.ext.ProviderSpec;

/**
 * A test-only {@link ExecutionProviderConfigurer} that states whichever
 * {@link ExecutionProviderPlacement} its {@value #PLACEMENT_OPTION} option names, registered the way
 * an addon registers one, in {@code META-INF/services/opennlp.dl.ExecutionProviderConfigurer} of this
 * module's test resources.
 *
 * <p>{@link EchoExecutionProviderConfigurer} covers the configurer that states nothing, which is
 * every configurer written before {@link ExecutionProviderConfigurer#placement(ProviderSpec)}
 * existed. This one covers the other side: an id nothing in {@code opennlp-dl} knows, classified as
 * an accelerator because its own configurer says so. That is the claim the SPI addition makes, and
 * it is what lets the accelerator group of {@link ExecutionProviders} stay at the built-in ids.</p>
 *
 * <p>{@value #NULL_PLACEMENT} and {@value #FAILING_PLACEMENT} are the two ways an addon can break
 * the contract of that method. They are here because the caller is deriving a default, so a broken
 * configurer must leave the default conservative rather than fail a construction.</p>
 *
 * <p>Like {@link EchoExecutionProviderConfigurer} it registers no execution provider, and writes a
 * session config entry instead, so a session configured with it is one a test can actually run.</p>
 */
public final class PlacementEchoExecutionProviderConfigurer
    implements ExecutionProviderConfigurer {

  /** The id this configurer answers to. It is in no accelerator group in {@code opennlp-dl}. */
  public static final String ID = "opennlp-test-placement";

  /** The option that names the placement to state, {@code accelerator} by default. */
  public static final String PLACEMENT_OPTION = "placement";

  /** The value of {@value #PLACEMENT_OPTION} that makes this return {@code null}. */
  public static final String NULL_PLACEMENT = "null";

  /** The value of {@value #PLACEMENT_OPTION} that makes this throw. */
  public static final String FAILING_PLACEMENT = "fail";

  /** The session config entry written in place of an execution provider. */
  public static final String KEY = "opennlp.test.placement.ran";

  /** {@inheritDoc} */
  @Override
  public String name() {
    return ID;
  }

  /** {@inheritDoc} */
  @Override
  public boolean supports(final ProviderSpec spec) {
    if (spec == null) {
      throw new IllegalArgumentException("spec must not be null");
    }
    return spec.location().isEmpty();
  }

  /**
   * {@inheritDoc}
   * The placement {@value #PLACEMENT_OPTION} names, {@link ExecutionProviderPlacement#ACCELERATOR}
   * where the option is absent, {@code null} for {@value #NULL_PLACEMENT} and a thrown
   * {@link IllegalStateException} for {@value #FAILING_PLACEMENT}.
   */
  @Override
  public ExecutionProviderPlacement placement(final ProviderSpec spec) {
    if (spec == null) {
      throw new IllegalArgumentException("spec must not be null");
    }
    final String stated =
        spec.option(PLACEMENT_OPTION, ExecutionProviderPlacement.ACCELERATOR.name());
    if (NULL_PLACEMENT.equals(stated)) {
      return null;
    }
    if (FAILING_PLACEMENT.equals(stated)) {
      throw new IllegalStateException("this configurer cannot state a placement");
    }
    return ExecutionProviderPlacement.valueOf(stated);
  }

  /** {@inheritDoc} */
  @Override
  public ExecutionProvider create(final ProviderSpec spec) {
    if (spec == null) {
      throw new IllegalArgumentException("spec must not be null");
    }
    return sessionOptions -> {
      if (sessionOptions == null) {
        throw new IllegalArgumentException("sessionOptions must not be null");
      }
      sessionOptions.addConfigEntry(KEY, ID);
    };
  }
}
