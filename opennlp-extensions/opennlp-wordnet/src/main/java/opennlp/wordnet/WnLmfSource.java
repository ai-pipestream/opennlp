/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
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

import java.io.IOException;
import java.io.InputStream;

/**
 * One resolved WN-LMF document: a diagnostic resource name plus a freshly opened stream. A
 * {@link WnLmfResolver} constructs one per {@link WnLmfResolver#resolve(WnLmfDependency)
 * resolve} call; the requesting {@link WnLmfReader} then owns the source and closes it exactly
 * once, whether reading succeeds or fails.
 *
 * <p>The stream is single-use: the reader consumes it once, and handing the same source out for
 * a second resolution fails loud, because the stream position would be undefined. Instances are
 * not thread safe; a source belongs to the single read that requested it.</p>
 *
 * @since 3.0.0
 */
public final class WnLmfSource implements AutoCloseable {

  private final String name;
  private final InputStream stream;
  private boolean consumed;

  /**
   * Creates a source over a freshly opened stream.
   *
   * @param name   The resource name used in error messages, typically a file name or URL.
   *               Must not be {@code null} or empty.
   * @param stream The document stream, opened for this source alone. Must not be {@code null}.
   * @throws IllegalArgumentException Thrown if {@code name} is {@code null} or empty or
   *         {@code stream} is {@code null}.
   */
  public WnLmfSource(String name, InputStream stream) {
    if (name == null || name.isEmpty()) {
      throw new IllegalArgumentException("Name must not be null or empty");
    }
    if (stream == null) {
      throw new IllegalArgumentException("Stream must not be null");
    }
    this.name = name;
    this.stream = stream;
  }

  /** {@return the resource name used in error messages} */
  public String name() {
    return name;
  }

  /**
   * Hands the stream to the reader, at most once.
   *
   * @return The document stream.
   * @throws IllegalStateException Thrown if the source was already consumed, which means a
   *         resolver returned the same source for more than one resolution.
   */
  InputStream consume() {
    if (consumed) {
      throw new IllegalStateException("Source " + name + " was already consumed; a resolver"
          + " must return a freshly opened source per resolve call");
    }
    consumed = true;
    return stream;
  }

  /**
   * Closes the underlying stream.
   *
   * @throws IOException Thrown if closing the stream fails.
   */
  @Override
  public void close() throws IOException {
    stream.close();
  }
}
