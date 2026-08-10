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

package opennlp.tools.pii;

import java.util.List;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

public class DevicePiiExtractorTest {

  private final DevicePiiExtractor extractor = new DevicePiiExtractor();

  @ParameterizedTest
  @ValueSource(strings = {
      "IMEI: 490154203237518",
      "imei # 490154203237518",
      "IMEI=35-693803-564380-9"})
  void testAcceptsLabeledImei(String text) {
    final List<PiiMention> mentions = extractor.extract(text);

    Assertions.assertEquals(1, mentions.size(), text);
    Assertions.assertEquals(PiiMention.TYPE_IMEI, mentions.get(0).type());
    Assertions.assertEquals("490154203237518", mentions.get(0).normalized());
  }

  @ParameterizedTest
  @ValueSource(strings = {
      "490154203237518",
      "IMEI: 490154203237519",
      "IMEI: 49015420323751",
      "IMEI: 4901542032375180",
      "IMEI was recorded as 490154203237518",
      "xIMEI: 490154203237518"})
  void testRejectsUnlabeledOrInvalidImei(String text) {
    Assertions.assertTrue(extractor.extract(text).isEmpty(), text);
  }

  @Test
  void testReportsExactNumberSpan() {
    final String text = "Device IMEI: 490154203237518.";
    final PiiMention mention = extractor.extract(text).get(0);

    Assertions.assertEquals("490154203237518",
        text.substring(mention.span().getStart(), mention.span().getEnd()));
  }

  @Test
  void testRejectsMissingText() {
    Assertions.assertThrows(IllegalArgumentException.class, () -> extractor.extract(null));
  }
}
