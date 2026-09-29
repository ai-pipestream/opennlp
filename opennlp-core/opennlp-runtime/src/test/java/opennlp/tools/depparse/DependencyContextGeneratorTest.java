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

package opennlp.tools.depparse;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Pins the features of {@link DependencyContextGenerator}. Trained models store these
 * strings, so a change to any of them makes existing models score differently.
 */
public class DependencyContextGeneratorTest {

  private static final String[] TOKENS = {"the", "dog", "barks", "today"};
  private static final String[] TAGS = {"DT", "NN", "VBZ", "RB"};

  @Test
  void testFeaturesOfAPartialParse() {
    // stack: root, dog, barks; buffer: today. "the" is the only dependent of "dog",
    // so it is both its leftmost and its rightmost dependent.
    final ArcStandardState state = new ArcStandardState(TOKENS.length);
    state.apply(Transition.SHIFT);
    state.apply(Transition.SHIFT);
    state.apply(Transition.leftArc("det"));
    state.apply(Transition.SHIFT);

    assertArrayEquals(new String[] {
        "s0w=barks", "s0t=VBZ", "s1w=dog", "s1t=NN", "s2t=*ROOT*",
        "b0w=today", "b0t=RB", "b1w=*NULL*", "b1t=*NULL*", "b2t=*NULL*",
        "s0wt=barks/VBZ", "s1wt=dog/NN", "b0wt=today/RB",
        "s0w,b0w=barks|today", "s0t,b0t=VBZ|RB", "s0w,b0t=barks|RB", "s0t,b0w=VBZ|today",
        "s0wt,b0t=barks/VBZ|RB", "s1t,s0t=NN|VBZ", "s1t,s0w=NN|barks", "s1w,s0t=dog|VBZ",
        "s1t,s0t,b0t=NN|VBZ|RB", "s0t,b0t,b1t=VBZ|RB|*NULL*", "s2t,s1t,s0t=*ROOT*|NN|VBZ",
        "s0lct=*NULL*", "s0rct=*NULL*", "s1lct=DT", "s1rct=DT",
        "s0lcl=*NULL*", "s0rcl=*NULL*", "s1rcl=det",
        "s1t,s1rct,s0t=NN|DT|VBZ", "s0t,s0lct,b0t=VBZ|*NULL*|RB",
        "s0deps=0", "s1deps=1", "dist=1", "dist,s0t,b0t=1|VBZ|RB"},
        new DependencyContextGenerator().getContext(state, TOKENS, TAGS));
  }

  @Test
  void testFeaturesOfTheStartConfiguration() {
    final ArcStandardState state = new ArcStandardState(1);
    assertArrayEquals(new String[] {
        "s0w=*ROOT*", "s0t=*ROOT*", "s1w=*NULL*", "s1t=*NULL*", "s2t=*NULL*",
        "b0w=Run", "b0t=VB", "b1w=*NULL*", "b1t=*NULL*", "b2t=*NULL*",
        "s0wt=*ROOT*/*ROOT*", "s1wt=*NULL*/*NULL*", "b0wt=Run/VB",
        "s0w,b0w=*ROOT*|Run", "s0t,b0t=*ROOT*|VB", "s0w,b0t=*ROOT*|VB", "s0t,b0w=*ROOT*|Run",
        "s0wt,b0t=*ROOT*/*ROOT*|VB", "s1t,s0t=*NULL*|*ROOT*", "s1t,s0w=*NULL*|*ROOT*",
        "s1w,s0t=*NULL*|*ROOT*", "s1t,s0t,b0t=*NULL*|*ROOT*|VB",
        "s0t,b0t,b1t=*ROOT*|VB|*NULL*", "s2t,s1t,s0t=*NULL*|*NULL*|*ROOT*",
        "s0lct=*NULL*", "s0rct=*NULL*", "s1lct=*NULL*", "s1rct=*NULL*",
        "s0lcl=*NULL*", "s0rcl=*NULL*", "s1rcl=*NULL*",
        "s1t,s1rct,s0t=*NULL*|*NULL*|*ROOT*", "s0t,s0lct,b0t=*ROOT*|*NULL*|VB",
        "s0deps=*NULL*", "s1deps=*NULL*", "dist=*NULL*", "dist,s0t,b0t=*NULL*|*ROOT*|VB"},
        new DependencyContextGenerator().getContext(state, new String[] {"Run"},
            new String[] {"VB"}));
  }

  /**
   * Pins the features once every token is on the stack: the buffer features and the
   * distance are the null marker.
   */
  @Test
  void testFeaturesWithAnEmptyBuffer() {
    final ArcStandardState state = new ArcStandardState(TOKENS.length);
    for (int i = 0; i < TOKENS.length; i++) {
      state.apply(Transition.SHIFT);
    }
    assertArrayEquals(new String[] {
        "s0w=today", "s0t=RB", "s1w=barks", "s1t=VBZ", "s2t=NN", "b0w=*NULL*", "b0t=*NULL*",
        "b1w=*NULL*", "b1t=*NULL*", "b2t=*NULL*", "s0wt=today/RB", "s1wt=barks/VBZ",
        "b0wt=*NULL*/*NULL*", "s0w,b0w=today|*NULL*", "s0t,b0t=RB|*NULL*",
        "s0w,b0t=today|*NULL*", "s0t,b0w=RB|*NULL*", "s0wt,b0t=today/RB|*NULL*",
        "s1t,s0t=VBZ|RB", "s1t,s0w=VBZ|today", "s1w,s0t=barks|RB", "s1t,s0t,b0t=VBZ|RB|*NULL*",
        "s0t,b0t,b1t=RB|*NULL*|*NULL*", "s2t,s1t,s0t=NN|VBZ|RB", "s0lct=*NULL*",
        "s0rct=*NULL*", "s1lct=*NULL*", "s1rct=*NULL*", "s0lcl=*NULL*", "s0rcl=*NULL*",
        "s1rcl=*NULL*", "s1t,s1rct,s0t=VBZ|*NULL*|RB", "s0t,s0lct,b0t=RB|*NULL*|*NULL*",
        "s0deps=0", "s1deps=0", "dist=*NULL*", "dist,s0t,b0t=*NULL*|RB|*NULL*"},
        new DependencyContextGenerator().getContext(state, TOKENS, TAGS));
  }

  /**
   * Pins the capped features: a stack top with three right dependents reports the
   * valency cap, and a buffer front four tokens away reports the distance bucket.
   */
  @Test
  void testFeaturesAtTheValencyAndDistanceCaps() {
    final String[] tokens = {"eat", "fish", "chips", "peas", "now"};
    final String[] tags = {"VB", "NN", "NNS", "NNS", "RB"};
    final ArcStandardState state = new ArcStandardState(tokens.length);
    state.apply(Transition.SHIFT);
    for (String label : new String[] {"obj", "conj", "conj"}) {
      state.apply(Transition.SHIFT);
      state.apply(Transition.rightArc(label));
    }
    assertArrayEquals(new String[] {
        "s0w=eat", "s0t=VB", "s1w=*ROOT*", "s1t=*ROOT*", "s2t=*NULL*", "b0w=now", "b0t=RB",
        "b1w=*NULL*", "b1t=*NULL*", "b2t=*NULL*", "s0wt=eat/VB", "s1wt=*ROOT*/*ROOT*",
        "b0wt=now/RB", "s0w,b0w=eat|now", "s0t,b0t=VB|RB", "s0w,b0t=eat|RB", "s0t,b0w=VB|now",
        "s0wt,b0t=eat/VB|RB", "s1t,s0t=*ROOT*|VB", "s1t,s0w=*ROOT*|eat", "s1w,s0t=*ROOT*|VB",
        "s1t,s0t,b0t=*ROOT*|VB|RB", "s0t,b0t,b1t=VB|RB|*NULL*", "s2t,s1t,s0t=*NULL*|*ROOT*|VB",
        "s0lct=NN", "s0rct=NNS", "s1lct=*NULL*", "s1rct=*NULL*", "s0lcl=obj", "s0rcl=conj",
        "s1rcl=*NULL*", "s1t,s1rct,s0t=*ROOT*|*NULL*|VB", "s0t,s0lct,b0t=VB|NN|RB", "s0deps=3",
        "s1deps=*NULL*", "dist=4+", "dist,s0t,b0t=4+|VB|RB"},
        new DependencyContextGenerator().getContext(state, tokens, tags));
  }

  @Test
  void testRejectsInvalidArguments() {
    final DependencyContextGenerator generator = new DependencyContextGenerator();
    final ArcStandardState state = new ArcStandardState(TOKENS.length);
    assertThrows(IllegalArgumentException.class,
        () -> generator.getContext(null, TOKENS, TAGS));
    assertThrows(IllegalArgumentException.class,
        () -> generator.getContext(state, null, TAGS));
    assertThrows(IllegalArgumentException.class,
        () -> generator.getContext(state, TOKENS, new String[] {"DT"}));
  }
}
