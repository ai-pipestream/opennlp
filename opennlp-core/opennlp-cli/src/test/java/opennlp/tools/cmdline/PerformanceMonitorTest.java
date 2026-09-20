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
package opennlp.tools.cmdline;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PerformanceMonitorTest {

  @Test
  void treatsPercentCharactersInUnitAsLiteralText() {
    ByteArrayOutputStream output = new ByteArrayOutputStream();
    PerformanceMonitor monitor = new PerformanceMonitor(
        new PrintStream(output, true, StandardCharsets.UTF_8), "items % literal");
    monitor.start();
    monitor.incrementCounter();

    assertDoesNotThrow(monitor::stopAndPrintFinalResult);
    assertTrue(output.toString(StandardCharsets.UTF_8).contains("items % literal"));
  }

  @Test
  void formatsOneDecimalWithoutTheGeneralFormatter() {
    assertEquals("1.3", PerformanceMonitor.formatOneDecimal(1.25));
    assertEquals("-1.3", PerformanceMonitor.formatOneDecimal(-1.25));
    assertEquals("NaN", PerformanceMonitor.formatOneDecimal(Double.NaN));
    assertEquals("Infinity", PerformanceMonitor.formatOneDecimal(Double.POSITIVE_INFINITY));
    assertEquals("-Infinity", PerformanceMonitor.formatOneDecimal(Double.NEGATIVE_INFINITY));
  }
}
