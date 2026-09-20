/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
package opennlp.tools.util;

import java.net.URL;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;

import opennlp.tools.models.ModelType;

@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Thread)
public class DownloadLinkColdStartBenchmark {

  @Param({"normal", "markupHeavy", "longIndex"})
  private String workload;

  private DownloadLinkBenchmarkFixture fixture;
  private Map<String, Map<ModelType, URL>> parsed;

  @Setup(Level.Trial)
  public void setUp() throws Exception {
    fixture = DownloadLinkBenchmarkFixture.create(workload);
  }

  @Benchmark
  @BenchmarkMode(Mode.SingleShotTime)
  @Warmup(iterations = 0)
  @Measurement(iterations = 1)
  public Map<String, Map<ModelType, URL>> firstParserCall() throws Exception {
    parsed = new DownloadUtil.DownloadParser(fixture.indexUrl()).getAvailableModels();
    return parsed;
  }

  @TearDown(Level.Trial)
  public void tearDown() throws Exception {
    try {
      if (parsed == null) {
        throw new IllegalStateException("Cold-start benchmark did not execute");
      }
      fixture.validateParsedModels(parsed);
      System.out.println("DownloadLinkColdStartBenchmark fingerprint " + workload + "="
          + DownloadLinkBenchmarkFixture.fingerprint(parsed));
    } finally {
      fixture.close();
    }
  }
}
