/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
package opennlp.tools.log;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.slf4j.Marker;
import org.slf4j.event.Level;
import org.slf4j.helpers.AbstractLogger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class LogPrintStreamTest {

  @Test
  void logsPrintAndAppendContentByLine() {
    RecordingLogger logger = new RecordingLogger();
    try (LogPrintStream stream = new LogPrintStream(logger)) {
      stream.print("first");
      stream.append(" line").append('\n');
      stream.append("second\r\nthird\n");

      assertEquals(List.of("first line", "second", "third"), logger.messages(Level.INFO));
    }
  }

  @Test
  void decodesUtf8WhenACharacterIsSplitAcrossWrites() {
    byte[] bytes = "A😀B\n".getBytes(StandardCharsets.UTF_8);
    RecordingLogger logger = new RecordingLogger();
    try (LogPrintStream stream = new LogPrintStream(logger)) {
      stream.write(bytes, 0, 3);
      stream.write(bytes, 3, bytes.length - 3);

      assertEquals(List.of("A😀B"), logger.messages(Level.INFO));
    }
  }

  @Test
  void flushAndCloseLogIncompleteLines() {
    RecordingLogger logger = new RecordingLogger();
    LogPrintStream flushed = new LogPrintStream(logger);
    flushed.print("flushed");
    flushed.flush();
    flushed.close();

    LogPrintStream closed = new LogPrintStream(logger);
    closed.print("closed");
    closed.close();

    assertEquals(List.of("flushed", "closed"), logger.messages(Level.INFO));
  }

  @Test
  void logsAtConfiguredLevelAndPreservesEmptyLines() {
    RecordingLogger logger = new RecordingLogger();
    try (LogPrintStream stream = new LogPrintStream(logger, Level.WARN)) {
      stream.println();
      stream.println("warning");

      assertEquals(List.of("", "warning"), logger.messages(Level.WARN));
      assertEquals(List.of(), logger.messages(Level.INFO));
    }
  }

  @Test
  void rejectsNullConstructorArguments() {
    assertThrows(IllegalArgumentException.class, () -> new LogPrintStream(null));
    assertThrows(IllegalArgumentException.class,
        () -> new LogPrintStream(new RecordingLogger(), null));
  }

  @Test
  void rejectsGeneralFormattingEntryPoints() {
    try (LogPrintStream stream = new LogPrintStream(new RecordingLogger())) {
      assertThrows(UnsupportedOperationException.class, () -> stream.printf("%s", "value"));
      assertThrows(UnsupportedOperationException.class,
          () -> stream.printf(Locale.ROOT, "%s", "value"));
      assertThrows(UnsupportedOperationException.class, () -> stream.format("%s", "value"));
      assertThrows(UnsupportedOperationException.class,
          () -> stream.format(Locale.ROOT, "%s", "value"));
    }
  }

  private static final class RecordingLogger extends AbstractLogger {

    private static final long serialVersionUID = 1L;
    private final Map<Level, List<String>> messages = new EnumMap<>(Level.class);

    private RecordingLogger() {
      for (Level level : Level.values()) {
        messages.put(level, new ArrayList<>());
      }
    }

    private List<String> messages(Level level) {
      return messages.get(level);
    }

    @Override
    public String getName() {
      return "recording";
    }

    @Override
    public boolean isTraceEnabled() {
      return true;
    }

    @Override
    public boolean isTraceEnabled(Marker marker) {
      return true;
    }

    @Override
    public boolean isDebugEnabled() {
      return true;
    }

    @Override
    public boolean isDebugEnabled(Marker marker) {
      return true;
    }

    @Override
    public boolean isInfoEnabled() {
      return true;
    }

    @Override
    public boolean isInfoEnabled(Marker marker) {
      return true;
    }

    @Override
    public boolean isWarnEnabled() {
      return true;
    }

    @Override
    public boolean isWarnEnabled(Marker marker) {
      return true;
    }

    @Override
    public boolean isErrorEnabled() {
      return true;
    }

    @Override
    public boolean isErrorEnabled(Marker marker) {
      return true;
    }

    @Override
    protected String getFullyQualifiedCallerName() {
      return LogPrintStreamTest.class.getName();
    }

    @Override
    protected void handleNormalizedLoggingCall(Level level, Marker marker, String message,
        Object[] arguments, Throwable throwable) {
      messages.get(level).add(message);
    }
  }
}
