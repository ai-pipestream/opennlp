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
import java.nio.charset.StandardCharsets;
import java.util.Locale;

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
   */
  public LogPrintStream(Logger logger, Level level) {
    super(new LoggingOutputStream(requireArgument(logger, "logger"),
        requireArgument(level, "level")), false, StandardCharsets.UTF_8);
  }

  @Override
  public PrintStream printf(String format, Object... args) {
    throw unsupportedFormatting();
  }

  @Override
  public PrintStream printf(Locale locale, String format, Object... args) {
    throw unsupportedFormatting();
  }

  @Override
  public PrintStream format(String format, Object... args) {
    throw unsupportedFormatting();
  }

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
      if (value == '\n') {
        logLine();
      } else {
        line.write(value);
      }
    }

    @Override
    public synchronized void write(byte[] bytes, int offset, int length) {
      for (int i = offset; i < offset + length; i++) {
        write(bytes[i] & 0xff);
      }
    }

    @Override
    public synchronized void flush() {
      if (line.size() > 0) {
        logLine();
      }
    }

    @Override
    public synchronized void close() throws IOException {
      flush();
      line.close();
    }

    private void logLine() {
      byte[] bytes = line.toByteArray();
      int length = bytes.length;
      if (length > 0 && bytes[length - 1] == '\r') {
        length--;
      }
      String message = new String(bytes, 0, length, StandardCharsets.UTF_8);
      line.reset();
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
