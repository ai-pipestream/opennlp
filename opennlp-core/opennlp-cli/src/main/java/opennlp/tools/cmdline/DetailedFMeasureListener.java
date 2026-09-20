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

package opennlp.tools.cmdline;

import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.SortedSet;
import java.util.TreeSet;

import opennlp.tools.commons.Internal;
import opennlp.tools.util.Span;
import opennlp.tools.util.eval.EvaluationMonitor;

/**
 * This listener will gather detailed information about the sample under evaluation and will
 * allow detailed FMeasure for each outcome.
 * <p>
 * <b>Note:</b> Do not use this class, internal use only!
 */
@Internal
public abstract class DetailedFMeasureListener<T> implements EvaluationMonitor<T> {

  private int samples = 0;
  private final Stats generalStats = new Stats();
  private final Map<String, Stats> statsForOutcome = new HashMap<>();

  protected abstract Span[] asSpanArray(T sample);

  @Override
  public void correctlyClassified(T reference, T prediction) {
    samples++;
    // add all true positives!
    Span[] spans = asSpanArray(reference);
    for (Span span : spans) {
      addTruePositive(span.getType());
    }
  }

  @Override
  public void misclassified(T reference, T prediction) {
    samples++;
    Span[] references = asSpanArray(reference);
    Span[] predictions = asSpanArray(prediction);

    Set<Span> refSet = new HashSet<>(Arrays.asList(references));
    Set<Span> predSet = new HashSet<>(Arrays.asList(predictions));

    for (Span ref : refSet) {
      if (predSet.contains(ref)) {
        addTruePositive(ref.getType());
      } else {
        addFalseNegative(ref.getType());
      }
    }

    for (Span pred : predSet) {
      if (!refSet.contains(pred)) {
        addFalsePositive(pred.getType());
      }
    }
  }

  private void addTruePositive(String type) {
    Stats s = initStatsForOutcomeAndGet(type);
    s.incrementTruePositive();
    s.incrementTarget();

    generalStats.incrementTruePositive();
    generalStats.incrementTarget();
  }

  private void addFalsePositive(String type) {
    Stats s = initStatsForOutcomeAndGet(type);
    s.incrementFalsePositive();
    generalStats.incrementFalsePositive();
  }

  private void addFalseNegative(String type) {
    Stats s = initStatsForOutcomeAndGet(type);
    s.incrementTarget();
    generalStats.incrementTarget();

  }

  private Stats initStatsForOutcomeAndGet(String type) {
    if (!statsForOutcome.containsKey(type)) {
      statsForOutcome.put(type, new Stats());
    }
    return statsForOutcome.get(type);
  }

  /**
   * Creates a report using locale-independent number formatting.
   *
   * @return The evaluation report.
   */
  public String createReport() {
    return createReport(Locale.ROOT);
  }

  /**
   * Creates a report whose percentages and integer counts use the requested locale.
   *
   * @param locale The locale used to format report numbers.
   * @return The evaluation report.
   * @throws IllegalArgumentException if {@code locale} is {@code null}.
   */
  public String createReport(Locale locale) {
    if (locale == null) {
      throw new IllegalArgumentException("locale must not be null");
    }
    DecimalFormat percentFormat = new DecimalFormat("0.00", DecimalFormatSymbols.getInstance(locale));
    percentFormat.setGroupingUsed(false);
    percentFormat.setRoundingMode(RoundingMode.HALF_UP);
    DecimalFormat integerFormat = new DecimalFormat("0", DecimalFormatSymbols.getInstance(locale));
    integerFormat.setGroupingUsed(false);
    StringBuilder ret = new StringBuilder();
    int tp = generalStats.getTruePositives();
    int found = generalStats.getFalsePositives() + tp;
    ret.append("Evaluated ").append(samples).append(" samples with ")
        .append(generalStats.getTarget()).append(" entities; found: ")
        .append(found).append(" entities; correct: ").append(tp).append(".\n");

    appendScores(ret, "TOTAL", generalStats, percentFormat, integerFormat, false);
    ret.append("\n");
    SortedSet<String> set = new TreeSet<>(new F1Comparator());
    set.addAll(statsForOutcome.keySet());
    for (String type : set) {

      appendScores(ret, type, statsForOutcome.get(type), percentFormat, integerFormat, true);
      ret.append("\n");
    }

    return ret.toString();
  }

  private void appendScores(StringBuilder report, String label, Stats stats,
      DecimalFormat percentFormat, DecimalFormat integerFormat, boolean includeCounts) {
    report.append(ReportLayout.right(label, 12)).append(": precision: ")
        .append(formatPercent(stats.getPrecisionScore(), percentFormat))
        .append(";  recall: ").append(formatPercent(stats.getRecallScore(), percentFormat))
        .append("; F1: ").append(formatPercent(stats.getFMeasure(), percentFormat)).append('.');
    if (includeCounts) {
      report.append(" [target: ")
          .append(ReportLayout.right(integerFormat.format(stats.getTarget()), 3))
          .append("; tp: ")
          .append(ReportLayout.right(integerFormat.format(stats.getTruePositives()), 3))
          .append("; fp: ")
          .append(ReportLayout.right(integerFormat.format(stats.getFalsePositives()), 3)).append(']');
    }
  }

  private String formatPercent(double score, DecimalFormat format) {
    double percent = zeroOrPositive(score * 100);
    String sign = percent >= 0 ? " " : "";
    return ReportLayout.right(sign + format.format(percent), 7) + "%";
  }

  @Override
  public String toString() {
    return createReport();
  }

  private double zeroOrPositive(double v) {
    if (v < 0) {
      return 0;
    }
    return v;
  }

  private class F1Comparator implements Comparator<String> {

    @Override
    public int compare(String o1, String o2) {
      if (o1.equals(o2))
        return 0;
      double t1 = 0;
      double t2 = 0;

      if (statsForOutcome.containsKey(o1))
        t1 += statsForOutcome.get(o1).getFMeasure();
      if (statsForOutcome.containsKey(o2))
        t2 += statsForOutcome.get(o2).getFMeasure();

      t1 = zeroOrPositive(t1);
      t2 = zeroOrPositive(t2);

      if (t1 + t2 > 0d) {
        if (t1 > t2)
          return -1;
        return 1;
      }
      return o1.compareTo(o2);
    }

  }

  /**
   * Holds the statistics.
   */
  private static class Stats {

    // maybe we could use FMeasure class, but it wouldn't allow us to get
    // details like total number of false positives and true positives.

    private int falsePositiveCounter = 0;
    private int truePositiveCounter = 0;
    private int targetCounter = 0;

    public void incrementFalsePositive() {
      falsePositiveCounter++;
    }

    public void incrementTruePositive() {
      truePositiveCounter++;
    }

    public void incrementTarget() {
      targetCounter++;
    }

    public int getFalsePositives() {
      return falsePositiveCounter;
    }

    public int getTruePositives() {
      return truePositiveCounter;
    }

    public int getTarget() {
      return targetCounter;
    }

    /**
     * Retrieves the arithmetic mean of the precision scores calculated for each
     * evaluated sample.
     *
     * @return the arithmetic mean of all precision scores
     */
    public double getPrecisionScore() {
      int tp = getTruePositives();
      int selected = tp + getFalsePositives();
      return selected > 0 ? (double) tp / (double) selected : 0;
    }

    /**
     * Retrieves the arithmetic mean of the recall score calculated for each
     * evaluated sample.
     *
     * @return the arithmetic mean of all recall scores
     */
    public double getRecallScore() {
      int target = getTarget();
      int tp = getTruePositives();
      return target > 0 ? (double) tp / (double) target : 0;
    }

    /**
     * Retrieves the f-measure score.
     * <p>
     * {@code f-measure = 2 * precision * recall / (precision + recall)}
     *
     * @return the f-measure or -1 if precision + recall <= 0
     */
    public double getFMeasure() {

      if (getPrecisionScore() + getRecallScore() > 0) {
        return 2 * (getPrecisionScore() * getRecallScore())
            / (getPrecisionScore() + getRecallScore());
      } else {
        // cannot divide by zero, return error code
        return -1;
      }
    }

  }

}
