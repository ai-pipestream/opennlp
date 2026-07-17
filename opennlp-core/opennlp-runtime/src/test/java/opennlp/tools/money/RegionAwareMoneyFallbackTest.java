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

import java.math.BigDecimal;
import java.util.List;

import org.junit.jupiter.api.Test;

import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.geo.DocumentRegionAnnotator;
import opennlp.tools.geo.Geocoder;
import opennlp.tools.geo.RegionVote;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Tests the fallback paths of region-aware money extraction: a pipeline-produced empty
 * ballot and a winning country whose currency symbol is spelled with letters both keep
 * the default symbol table, under which the ambiguous {@code $} sign denotes
 * {@code USD}.
 */
public class RegionAwareMoneyFallbackTest {

  /**
   * Builds a document that carries a region ballot with a single full-share vote for
   * one country.
   *
   * @param text The document text. Must not be {@code null}.
   * @param countryCode The winning ISO 3166-1 alpha-2 country code. Must not be
   *                    {@code null}.
   * @return A {@link Document} with a one-row region ballot. Never {@code null}.
   */
  private static Document withBallot(String text, String countryCode) {
    return Document.of(text).with(DocumentRegionAnnotator.REGIONS,
        List.of(Annotation.of(new RegionVote(countryCode, 1.0))));
  }

  /**
   * Verifies the empty-ballot fallback on a ballot the region annotator itself
   * produced: a document without location entities gets an empty region layer, so the
   * money annotator uses the default symbol table and identifies {@code $7} as
   * {@code USD}.
   */
  @Test
  void testPipelineProducedEmptyBallotUsesTheDefaultTable() {
    final Geocoder unreachable = (text, mentions) -> {
      throw new IllegalStateException("the geocoder must not be consulted");
    };
    final Document document = new RegionAwareMoneyAnnotator().annotate(
        new DocumentRegionAnnotator(unreachable).annotate(Document.of("the fee is $7")));

    final List<Annotation<MoneyAmount>> money = document.get(MoneyAnnotator.MONEY);
    assertEquals(1, money.size());
    assertEquals("USD", money.get(0).value().currency());
    assertEquals(0, new BigDecimal("7").compareTo(money.get(0).value().amount()));
    assertEquals("$7",
        money.get(0).span().getCoveredText(document.text()).toString());
  }

  /**
   * Verifies the letter-symbol fallback: Switzerland wins the ballot, but the Swiss
   * franc is conventionally written {@code CHF}, not a single currency-sign code point,
   * so the symbol table stays at the default and {@code $5} is identified as
   * {@code USD}, not remapped.
   */
  @Test
  void testLetterSymbolCountryKeepsTheDefaultTable() {
    final Document document = new RegionAwareMoneyAnnotator()
        .annotate(withBallot("the invoice totals $5", "CH"));

    final List<Annotation<MoneyAmount>> money = document.get(MoneyAnnotator.MONEY);
    assertEquals(1, money.size());
    assertEquals("USD", money.get(0).value().currency());
  }
}
