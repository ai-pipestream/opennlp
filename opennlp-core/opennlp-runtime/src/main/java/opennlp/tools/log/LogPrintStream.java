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


package opennlp.tools.log;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.CoderResult;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Locale;
import java.util.Objects;

import org.slf4j.Logger;
import org.slf4j.event.Level;

import opennlp.tools.commons.Internal;

/**
 * This class serves as an adapter for a {@link Logger} used within a {@link PrintStream}.
 * General-purpose formatted output is not supported; callers must pass complete text to the
 * {@code print}, {@code append}, or {@code println} methods.
 */
@Internal
public class LogPrintStream extends PrintStream {

  /**
   * Creates a {@link LogPrintStream} for the given {@link Logger}.
   *
   * @param logger must not be {@code null}
   * @throws IllegalArgumentException if {@code logger} is {@code null}
   */
  public LogPrintStream(Logger logger) {
    this(logger, Level.INFO);
  }

  /**
   * Creates a {@link LogPrintStream} for the given {@link Logger}, which logs at the specified
   * {@link Level level}.
   *
   * @param logger must not be {@code null}
   * @param level  must not be {@code null}
   * @throws IllegalArgumentException if {@code logger} or {@code level} is {@code null}
   */
  public LogPrintStream(Logger logger, Level level) {
    super(new LoggingOutputStream(requireArgument(logger, "logger"),
        requireArgument(level, "level")), false, StandardCharsets.UTF_8);
  }

  /**
   * @throws UnsupportedOperationException always; pass preformatted text to a print method
   */
  @Override
  public PrintStream printf(String format, Object... args) {
    throw unsupportedFormatting();
  }

  /**
   * @throws UnsupportedOperationException always; pass preformatted text to a print method
   */
  @Override
  public PrintStream printf(Locale locale, String format, Object... args) {
    throw unsupportedFormatting();
  }

  /**
   * @throws UnsupportedOperationException always; pass preformatted text to a print method
   */
  @Override
  public PrintStream format(String format, Object... args) {
    throw unsupportedFormatting();
  }

  /**
   * @throws UnsupportedOperationException always; pass preformatted text to a print method
   */
  @Override
  public PrintStream format(Locale locale, String format, Object... args) {
    throw unsupportedFormatting();
  }

  private static UnsupportedOperationException unsupportedFormatting() {
    return new UnsupportedOperationException("LogPrintStream does not support formatted output");
  }

  private static <T> T requireArgument(T value, String name) {
    if (value == null) {
      throw new IllegalArgumentException(name + " must not be null");
    }
    return value;
  }

  private static final class LoggingOutputStream extends OutputStream {

    private final Logger logger;
    private final Level level;
    private final ByteArrayOutputStream line = new ByteArrayOutputStream();

    private LoggingOutputStream(Logger logger, Level level) {
      this.logger = logger;
      this.level = level;
    }

    @Override
    public synchronized void write(int value) {
      int octet = value & 0xff;
      if (octet == '\n') {
        logLine(true, true);
      } else {
        line.write(octet);
      }
    }

    @Override
    public synchronized void write(byte[] bytes, int offset, int length) {
      if (bytes == null) {
        throw new IllegalArgumentException("bytes must not be null");
      }
      Objects.checkFromIndexSize(offset, length, bytes.length);
      for (int i = offset; i < offset + length; i++) {
        write(bytes[i] & 0xff);
      }
    }

    @Override
    public synchronized void flush() {
      if (line.size() > 0) {
        logLine(false, false);
      }
    }

    @Override
    public synchronized void close() throws IOException {
      if (line.size() > 0) {
        logLine(true, false);
      }
      line.close();
    }

    private void logLine(boolean endOfInput, boolean stripCarriageReturn) {
      byte[] bytes = line.toByteArray();
      int length = bytes.length;
      if (stripCarriageReturn && length > 0 && bytes[length - 1] == '\r') {
        length--;
      }

      CharsetDecoder decoder = StandardCharsets.UTF_8.newDecoder()
          .onMalformedInput(CodingErrorAction.REPLACE)
          .onUnmappableCharacter(CodingErrorAction.REPLACE);
      ByteBuffer input = ByteBuffer.wrap(bytes, 0, length);
      CharBuffer output = CharBuffer.allocate(length + 1);
      CoderResult result = decoder.decode(input, output, endOfInput);
      if (endOfInput) {
        decoder.flush(output);
      }
      if (result.isError()) {
        throw new IllegalStateException("UTF-8 decoder rejected replacement input");
      }

      String message = output.flip().toString();
      line.reset();
      line.writeBytes(Arrays.copyOfRange(bytes, input.position(), length));
      if (!endOfInput && message.isEmpty()) {
        return;
      }
      switch (level) {
        case TRACE:
          logger.trace(message);
          break;
        case DEBUG:
          logger.debug(message);
          break;
        case INFO:
          logger.info(message);
          break;
        case WARN:
          logger.warn(message);
          break;
        case ERROR:
          logger.error(message);
          break;
      }
    }
  }
}
