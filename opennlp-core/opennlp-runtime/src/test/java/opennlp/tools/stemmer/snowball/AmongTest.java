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
package opennlp.tools.stemmer.snowball;

import java.lang.invoke.MethodHandles;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;

class AmongTest {

  @Test
  void missingMethodDiagnosticRetainsClassMethodAndCause() throws IllegalAccessException {
    MethodHandles.Lookup lookup = MethodHandles.privateLookupIn(BrokenStemmer.class,
        MethodHandles.lookup());

    RuntimeException exception = assertThrows(RuntimeException.class,
        () -> new Among("suffix", -1, 1, "missing", lookup));

    assertEquals("Snowball program 'BrokenStemmer' is broken, cannot access method: "
        + "boolean missing()", exception.getMessage());
    assertInstanceOf(NoSuchMethodException.class, exception.getCause());
  }

  private static final class BrokenStemmer extends SnowballProgram {
    private static final long serialVersionUID = 1L;
  }
}
