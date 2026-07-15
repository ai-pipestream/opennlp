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

package opennlp.tools.money;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.math.BigDecimal;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.document.DocumentAnalyzer;
import opennlp.tools.document.DocumentAnnotator;
import opennlp.tools.document.LayerKey;
import opennlp.tools.document.Layers;
import opennlp.tools.geo.DocumentRegionAnnotator;
import opennlp.tools.geo.GazetteerEntry;
import opennlp.tools.geo.GeoPoint;
import opennlp.tools.geo.GeoResolution;
import opennlp.tools.geo.Geocoder;
import opennlp.tools.geo.RegionVote;
import opennlp.tools.util.Span;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Tests that the winning region picks the symbol table per document, and drives the
 * full chain from location entities to converted money. The rate fixture is
 * project-authored and synthetic.
 */
public class RegionAwareMoneyAnnotatorTest {

  private static Document withBallot(String text, String countryCode) {
    return Document.of(text).with(DocumentRegionAnnotator.REGIONS,
        List.of(new Annotation<>(new Span(0, text.length()),
            new RegionVote(countryCode, 1.0))));
  }

  @Test
  void testWinningRegionResolvesTheSymbol() {
    final Document document = new RegionAwareMoneyAnnotator()
        .annotate(withBallot("the deal is worth $5", "AU"));
    assertEquals("AUD",
        document.get(MoneyAnnotator.MONEY).get(0).value().currency());
  }

  @Test
  void testEmptyBallotFallsBackToTheDefaultTable() {
    final Document document = new RegionAwareMoneyAnnotator()
        .annotate(Document.of("costs $5")
            .with(DocumentRegionAnnotator.REGIONS, List.of()));
    assertEquals("USD",
        document.get(MoneyAnnotator.MONEY).get(0).value().currency());
  }

  @Test
  void testEndToEndFromEntitiesToConvertedMoney() throws IOException {
    // entities -> geocode -> region ballot -> region-aware money -> FX conversion
    final Geocoder geocoder = (text, mentions) -> {
      final List<GeoResolution> resolutions = new ArrayList<>();
      for (final Span mention : mentions) {
        resolutions.add(new GeoResolution(mention,
            new GazetteerEntry("test", "sydney", "Sydney", List.of(),
                new GeoPoint(-33.87, 151.21), "AU", List.of(), 5_300_000,
                GazetteerEntry.FEATURE_CLASS_CITY, Map.of()), 0.9));
      }
      return resolutions;
    };
    final EcbFxRates rates = EcbFxRates.load(new ByteArrayInputStream(String.join("\n",
        "Date,USD,AUD,", "2026-07-10,1.2000,1.8000,", "").getBytes()));

    final String text = "a Sydney startup raised $3 million this week";
    final DocumentAnnotator entities = new DocumentAnnotator() {

      @Override
      public Document annotate(Document document) {
        final int start = text.indexOf("Sydney");
        return document.with(Layers.ENTITIES, List.of(
            new Annotation<>(new Span(start, start + 6), "location")));
      }

      @Override
      public java.util.Set<LayerKey<?>> provides() {
        return java.util.Set.of(Layers.ENTITIES);
      }
    };

    final Document document = DocumentAnalyzer.builder()
        .add(entities)
        .add(new DocumentRegionAnnotator(geocoder))
        .add(new RegionAwareMoneyAnnotator())
        .add(new MoneyConversionAnnotator(rates, "USD", LocalDate.parse("2026-07-10")))
        .build()
        .analyze(text);

    final Annotation<MoneyAmount> identified =
        document.get(MoneyAnnotator.MONEY).get(0);
    assertEquals("AUD", identified.value().currency());
    assertEquals("$3 million",
        identified.span().getCoveredText(document.text()).toString());

    final Annotation<MoneyAmount> converted =
        document.get(MoneyConversionAnnotator.CONVERTED_MONEY).get(0);
    assertEquals("USD", converted.value().currency());
    // 3,000,000 AUD * (1.2 / 1.8) = 2,000,000 USD
    assertEquals(0, new BigDecimal("2000000").compareTo(converted.value().amount()));
    assertEquals(identified.span(), converted.span());
  }

  @Test
  void testUnknownCountryFallsBackToTheDefaultTable() {
    final Document document = new RegionAwareMoneyAnnotator()
        .annotate(withBallot("costs $5", "ZZ"));
    assertEquals("USD",
        document.get(MoneyAnnotator.MONEY).get(0).value().currency());
  }

  @Test
  void testNullDocumentThrows() {
    assertThrows(IllegalArgumentException.class,
        () -> new RegionAwareMoneyAnnotator().annotate(null));
  }
}
