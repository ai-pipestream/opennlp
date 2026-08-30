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

package opennlp.tools.coref;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.Serial;
import java.nio.file.Path;
import java.util.Map;
import java.util.Properties;

import opennlp.tools.ml.model.AbstractModel;
import opennlp.tools.ml.model.MaxentModel;
import opennlp.tools.util.InvalidFormatException;
import opennlp.tools.util.model.BaseModel;

/**
 * The {@link CorefAnnotator} ranking model: a classifier over anaphor and candidate
 * antecedent pairs with the outcomes link and apart, trained by {@link CorefTrainer}.
 *
 * @since 3.0.0
 */
public class CorefModel extends BaseModel {

  @Serial
  private static final long serialVersionUID = 1409953135923471757L;

  private static final String COMPONENT_NAME = "CorefAnnotator";

  private static final String COREF_MODEL_ENTRY_NAME = "coref.model";

  /** The manifest property that marks a model trained by ranking. */
  private static final String RANKING_PROPERTY = "coref.ranking";

  /**
   * Initializes a model from a trained pair classifier.
   *
   * @param languageCode The ISO language code of the training data.
   * @param pairModel The pair classifier. Must not be {@code null}.
   * @param manifestInfoEntries Additional manifest entries, or {@code null}.
   * @throws IllegalArgumentException Thrown if {@code pairModel} is {@code null}.
   */
  public CorefModel(String languageCode, MaxentModel pairModel,
      Map<String, String> manifestInfoEntries) {
    this(languageCode, pairModel, false, manifestInfoEntries);
  }

  /**
   * Initializes a model from a trained pair classifier or ranker.
   *
   * @param languageCode The ISO language code of the training data.
   * @param pairModel The pair model. Must not be {@code null}.
   * @param ranking Whether the model was trained by ranking, so a candidate is linked
   *                when it outscores the new-chain option rather than a threshold.
   * @param manifestInfoEntries Additional manifest entries, or {@code null}.
   * @throws IllegalArgumentException Thrown if {@code pairModel} is {@code null}.
   */
  public CorefModel(String languageCode, MaxentModel pairModel, boolean ranking,
      Map<String, String> manifestInfoEntries) {
    super(COMPONENT_NAME, languageCode, manifestInfoEntries);
    if (pairModel == null) {
      throw new IllegalArgumentException("pairModel must not be null");
    }
    artifactMap.put(COREF_MODEL_ENTRY_NAME, pairModel);
    ((Properties) artifactMap.get(MANIFEST_ENTRY))
        .setProperty(RANKING_PROPERTY, Boolean.toString(ranking));
    checkArtifactMap();
  }

  /** {@return whether the model was trained by ranking} */
  public boolean isRanking() {
    return Boolean.parseBoolean(getManifestProperty(RANKING_PROPERTY));
  }

  /**
   * Reads a model.
   *
   * @param in The stream to read from. Must not be {@code null}.
   * @throws IOException Thrown if reading fails or the model is invalid.
   */
  public CorefModel(InputStream in) throws IOException {
    super(COMPONENT_NAME, in);
  }

  /**
   * Reads a model.
   *
   * @param modelFile The file to read. Must not be {@code null}.
   * @throws IOException Thrown if reading fails or the model is invalid.
   */
  public CorefModel(File modelFile) throws IOException {
    super(COMPONENT_NAME, modelFile);
  }

  /**
   * Reads a model.
   *
   * @param modelPath The path to read. Must not be {@code null}.
   * @throws IOException Thrown if reading fails or the model is invalid.
   */
  public CorefModel(Path modelPath) throws IOException {
    super(COMPONENT_NAME, modelPath);
  }

  /** {@inheritDoc} Requires the pair classifier entry. */
  @Override
  protected void validateArtifactMap() throws InvalidFormatException {
    super.validateArtifactMap();
    if (!(artifactMap.get(COREF_MODEL_ENTRY_NAME) instanceof AbstractModel)) {
      throw new InvalidFormatException("Coreference model is incomplete!");
    }
  }

  /** {@return the pair classifier} */
  public MaxentModel getPairModel() {
    return (MaxentModel) artifactMap.get(COREF_MODEL_ENTRY_NAME);
  }
}
