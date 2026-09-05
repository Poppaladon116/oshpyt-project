import * as fs from "fs";
import { Tensor } from "./OshpytTensor";

type ValidatedTrainingData = {
  version: string;
  description: string;
  vocabulary: string[];
  vocabularySize: number;
  tokenToId: Record<string, number>;
  promptSequences: string[][];
  trainingSequences: string[][];
  indexedPromptSequences: number[][];
  indexedTrainingSequences: number[][];
  maxGenerationTokens: number;
};

const MODEL_DIR = "models/multi_v4_email_first";
const TRAINING_DATA_PATH =
  process.env.TRAINING_DATA_PATH || "components_v3.validated.json";

const HIDDEN_SIZE = 48;
const EPOCHS = 4000;
const LEARNING_RATE = 0.02;
const GRADIENT_CLIP = 5.0;
const NEGATIVE_SLOPE = 0.01;

function fail(message: string): never {
  throw new Error("train_rnn_v3: " + message);
}

function zeros(size: number): Float32Array {
  return new Float32Array(size);
}

function randomWeights(size: number, scale = 0.08): Float32Array {
  const values = new Float32Array(size);

  for (let i = 0; i < values.length; i++) {
   values[i] = (Math.random() * 2 - 1) * scale;
  }

  return values;
}

function oneHot(tokenId: number, vocabularySize: number): Float32Array {
  if (!Number.isInteger(tokenId) || tokenId < 0 || tokenId >= vocabularySize) {
    fail("token ID outside vocabulary: " + tokenId);
  }

  const result = zeros(vocabularySize);
  result[tokenId] = 1;
  return result;
}

function addArrays(
  left: Float32Array,
  right: Float32Array
): Float32Array {
  if (left.length !== right.length) {
    fail("array length mismatch.");
  }

  const result = new Float32Array(left.length);

  for (let i = 0; i < result.length; i++) {
    result[i] = left[i] + right[i];
  }

  return result;
}

function addInto(target: Float32Array, source: Float32Array): void {
  if (target.length !== source.length) {
    fail("gradient array length mismatch.");
  }

  for (let i = 0; i < target.length; i++) {
    target[i] += source[i];
  }
}

function subtractOneHot(
  probabilities: Float32Array,
  targetToken: number
): Float32Array {
  if (
    !Number.isInteger(targetToken) ||
    targetToken < 0 ||
    targetToken >= probabilities.length
  ) {
    fail("target token outside vocabulary: " + targetToken);
  }

  const result = new Float32Array(probabilities);
  result[targetToken] -= 1;
  return result;
}

function clipGradient(
  values: Float32Array,
  limit: number
): Float32Array {
  const result = new Float32Array(values.length);

  for (let i = 0; i < values.length; i++) {
    result[i] = Math.max(-limit, Math.min(limit, values[i]));
  }

  return result;
}

function leakyReluValues(values: Float32Array): Float32Array {
  const result = new Float32Array(values.length);

  for (let i = 0; i < values.length; i++) {
    result[i] = values[i] >= 0
      ? values[i]
      : values[i] * NEGATIVE_SLOPE;
  }

  return result;
}

function leakyReluGradient(
  gradient: Float32Array,
  preActivation: Float32Array
): Float32Array {
  if (gradient.length !== preActivation.length) {
    fail("Leaky ReLU gradient length mismatch.");
  }

  const result = new Float32Array(gradient.length);

  for (let i = 0; i < gradient.length; i++) {
    result[i] = gradient[i] * (
      preActivation[i] >= 0 ? 1 : NEGATIVE_SLOPE
    );
  }

  return result;
}

function argmax(values: Float32Array): number {
  let bestIndex = 0;
  let bestValue = -Infinity;

  for (let i = 0; i < values.length; i++) {
    if (values[i] > bestValue) {
      bestValue = values[i];
      bestIndex = i;
    }
  }

  return bestIndex;
}

function probabilityOf(probabilities: Float32Array, tokenId: number): number {
  return Math.max(probabilities[tokenId], 1e-12);
}

function makeRow(values: Float32Array): Tensor {
  return Tensor.fromArray(values, 1, values.length);
}

function destroyAll(tensors: Tensor[]): void {
  for (const tensor of tensors) {
    tensor.destroy();
  }
}

function saveTensor(filePath: string, tensor: Tensor): void {
  const values = tensor.download();

  fs.writeFileSync(
    filePath,
    Buffer.from(values.buffer, values.byteOffset, values.byteLength)
  );
}

function loadTrainingData(): ValidatedTrainingData {
  if (!fs.existsSync(TRAINING_DATA_PATH)) {
    fail("training data file not found: " + TRAINING_DATA_PATH);
  }

  let raw: unknown;

  try {
    raw = JSON.parse(fs.readFileSync(TRAINING_DATA_PATH, "utf8"));
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    fail("could not parse training data: " + message);
  }

  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
    fail("training data root must be an object.");
  }

  const data = raw as Partial<ValidatedTrainingData>;

  if (!Array.isArray(data.vocabulary) || data.vocabulary.length === 0) {
    fail("vocabulary must be a non-empty array.");
  }

  if (
    typeof data.vocabularySize !== "number" ||
    data.vocabularySize !== data.vocabulary.length
  ) {
    fail("vocabularySize must equal vocabulary.length.");
  }

  if (
    !Array.isArray(data.trainingSequences) ||
    !Array.isArray(data.indexedTrainingSequences) ||
    data.trainingSequences.length === 0 ||
    data.trainingSequences.length !== data.indexedTrainingSequences.length
  ) {
    fail("training sequences are missing or inconsistent.");
  }

  if (
    !Array.isArray(data.promptSequences) ||
    !Array.isArray(data.indexedPromptSequences) ||
    data.promptSequences.length !== data.trainingSequences.length ||
    data.indexedPromptSequences.length !== data.trainingSequences.length
  ) {
    fail("prompt sequences are missing or inconsistent.");
  }

  if (
    typeof data.maxGenerationTokens !== "number" ||
    !Number.isInteger(data.maxGenerationTokens) ||
    data.maxGenerationTokens < 2
  ) {
    fail("maxGenerationTokens is invalid.");
  }

  const vocabulary = data.vocabulary.map((token, index) => {
    if (typeof token !== "string" || token.trim() === "") {
      fail("vocabulary[" + index + "] is invalid.");
    }

    return token;
  });

  const trainingSequences = data.trainingSequences.map((sequence, index) => {
    if (!Array.isArray(sequence) || sequence.length < 2) {
      fail("trainingSequences[" + index + "] is invalid.");
    }

    return sequence.map((token, tokenIndex) => {
      if (typeof token !== "string" || token.trim() === "") {
        fail(
          "trainingSequences[" +
            index +
            "][" +
            tokenIndex +
            "] is invalid."
        );
      }

      return token;
    });
  });

  const promptSequences = data.promptSequences.map((sequence, index) => {
    if (!Array.isArray(sequence) || sequence.length === 0) {
      fail("promptSequences[" + index + "] is invalid.");
    }

    return sequence.map((token, tokenIndex) => {
      if (typeof token !== "string" || token.trim() === "") {
        fail(
          "promptSequences[" +
            index +
            "][" +
            tokenIndex +
            "] is invalid."
        );
      }

      return token;
    });
  });

  const indexedTrainingSequences = data.indexedTrainingSequences.map(
    (sequence, index) => {
      if (
        !Array.isArray(sequence) ||
        sequence.length !== trainingSequences[index].length
      ) {
        fail("indexedTrainingSequences[" + index + "] is invalid.");
      }

      return sequence.map((tokenId, tokenIndex) => {
        if (
          typeof tokenId !== "number" ||
          !Number.isInteger(tokenId) ||
          tokenId < 0 ||
          tokenId >= vocabulary.length
        ) {
          fail(
            "indexedTrainingSequences[" +
              index +
              "][" +
              tokenIndex +
              "] is invalid."
          );
        }

        if (vocabulary[tokenId] !== trainingSequences[index][tokenIndex]) {
          fail(
            "indexed training token does not match vocabulary at sequence " +
              index +
              ", token " +
              tokenIndex
          );
        }

        return tokenId;
      });
    }
  );

  const indexedPromptSequences = data.indexedPromptSequences.map(
    (sequence, index) => {
      if (
        !Array.isArray(sequence) ||
        sequence.length !== promptSequences[index].length
      ) {
        fail("indexedPromptSequences[" + index + "] is invalid.");
      }

      return sequence.map((tokenId, tokenIndex) => {
        if (
          typeof tokenId !== "number" ||
          !Number.isInteger(tokenId) ||
          tokenId < 0 ||
          tokenId >= vocabulary.length
        ) {
          fail(
            "indexedPromptSequences[" +
              index +
              "][" +
              tokenIndex +
              "] is invalid."
          );
        }

        if (vocabulary[tokenId] !== promptSequences[index][tokenIndex]) {
          fail(
            "indexed prompt token does not match vocabulary at sequence " +
              index +
              ", token " +
              tokenIndex
          );
        }

        return tokenId;
      });
    }
  );

  const tokenToId: Record<string, number> = {};

  vocabulary.forEach((token, index) => {
    tokenToId[token] = index;
  });

  return {
    version: typeof data.version === "string" ? data.version : "multi_v3",
    description:
      typeof data.description === "string"
        ? data.description
        : "Validated v3 training data",
    vocabulary,
    vocabularySize: vocabulary.length,
    tokenToId,
    promptSequences,
    trainingSequences,
    indexedPromptSequences,
    indexedTrainingSequences,
    maxGenerationTokens: data.maxGenerationTokens,
  };
}

function advanceHiddenState(
  tokenId: number,
  hidden: Tensor,
  Wx: Tensor,
  Wh: Tensor,
  vocabularySize: number
): Tensor {
  const input = makeRow(oneHot(tokenId, vocabularySize));
  const inputPart = Tensor.allocate(1, HIDDEN_SIZE);
  const memoryPart = Tensor.allocate(1, HIDDEN_SIZE);
  const nextHidden = Tensor.allocate(1, HIDDEN_SIZE);

  try {
    input.matmulTo(Wx, inputPart);
    hidden.matmulTo(Wh, memoryPart);

    const preActivation = addArrays(
      inputPart.download(),
      memoryPart.download()
    );

    nextHidden.update(leakyReluValues(preActivation));

    return nextHidden;
  } finally {
    input.destroy();
    inputPart.destroy();
    memoryPart.destroy();
  }
}

function generateFromPrompt(
  promptIds: number[],
  Wx: Tensor,
  Wh: Tensor,
  Wo: Tensor,
  vocabulary: string[],
  maxTokens: number
): string[] {
  const vocabularySize = vocabulary.length;
  const generated: string[] = [];
  let hidden = makeRow(zeros(HIDDEN_SIZE));

  try {
    for (const promptId of promptIds) {
      generated.push(vocabulary[promptId]);

      const nextHidden = advanceHiddenState(
        promptId,
        hidden,
        Wx,
        Wh,
        vocabularySize
      );

      hidden.destroy();
      hidden = nextHidden;
    }

    while (generated.length < maxTokens) {
      const output = Tensor.allocate(1, vocabularySize);

      try {
        hidden.matmulTo(Wo, output);
        output.softmax();

        const nextTokenId = argmax(output.download());
        generated.push(vocabulary[nextTokenId]);

        if (vocabulary[nextTokenId] === "COMPONENT_END") {
          break;
        }

        const nextHidden = advanceHiddenState(
          nextTokenId,
          hidden,
          Wx,
          Wh,
          vocabularySize
        );

        hidden.destroy();
        hidden = nextHidden;
      } finally {
        output.destroy();
      }
    }

    return generated;
  } finally {
    hidden.destroy();
  }
}

function main(): void {
  const data = loadTrainingData();
  const TOKENS = data.vocabulary;
  const VOCAB_SIZE = data.vocabularySize;
  const TRAINING_SEQUENCES = data.indexedTrainingSequences;

  console.log("--- OSHPYT RNN LAB: V3 MULTI-SEQUENCE BPTT ---");
  console.log("Dataset: " + TRAINING_DATA_PATH);
  console.log("Version: " + data.version);
  console.log("Examples: " + TRAINING_SEQUENCES.length);
  console.log("Vocabulary size: " + VOCAB_SIZE);
  console.log("Hidden size: " + HIDDEN_SIZE);
  console.log("Output directory: " + MODEL_DIR);

  const Wx = Tensor.fromArray(
    randomWeights(VOCAB_SIZE * HIDDEN_SIZE),
    VOCAB_SIZE,
    HIDDEN_SIZE
  );

  const Wh = Tensor.fromArray(
    randomWeights(HIDDEN_SIZE * HIDDEN_SIZE),
    HIDDEN_SIZE,
    HIDDEN_SIZE
  );

  const Wo = Tensor.fromArray(
    randomWeights(HIDDEN_SIZE * VOCAB_SIZE),
    HIDDEN_SIZE,
    VOCAB_SIZE
  );

  try {
    for (let epoch = 1; epoch <= EPOCHS; epoch++) {
      let totalLoss = 0;
      let correct = 0;
      let totalPredictions = 0;

      const dWxHost = zeros(VOCAB_SIZE * HIDDEN_SIZE);
      const dWhHost = zeros(HIDDEN_SIZE * HIDDEN_SIZE);
      const dWoHost = zeros(HIDDEN_SIZE * VOCAB_SIZE);

      for (const sequence of TRAINING_SEQUENCES) {
        const inputs: Tensor[] = [];
        const previousHiddenStates: Tensor[] = [];
        const preActivations: Tensor[] = [];
        const hiddenStates: Tensor[] = [makeRow(zeros(HIDDEN_SIZE))];
        const probabilities: Tensor[] = [];
        const targets: number[] = [];

        try {
          for (let step = 0; step < sequence.length - 1; step++) {
            const inputToken = sequence[step];
            const targetToken = sequence[step + 1];

            const input = makeRow(oneHot(inputToken, VOCAB_SIZE));
            const previousHidden = hiddenStates[step];
            const inputPart = Tensor.allocate(1, HIDDEN_SIZE);
            const memoryPart = Tensor.allocate(1, HIDDEN_SIZE);
            const preActivation = Tensor.allocate(1, HIDDEN_SIZE);
            const hidden = Tensor.allocate(1, HIDDEN_SIZE);
            const output = Tensor.allocate(1, VOCAB_SIZE);

            try {
              input.matmulTo(Wx, inputPart);
              previousHidden.matmulTo(Wh, memoryPart);

              preActivation.update(
                addArrays(inputPart.download(), memoryPart.download())
              );

              hidden.update(
                leakyReluValues(preActivation.download())
              );

              hidden.matmulTo(Wo, output);
              output.softmax();

              const outputValues = output.download();

              totalLoss -= Math.log(probabilityOf(outputValues, targetToken));
              totalPredictions++;

              if (argmax(outputValues) === targetToken) {
                correct++;
              }

              inputs.push(input);
              previousHiddenStates.push(previousHidden);
              preActivations.push(preActivation);
              hiddenStates.push(hidden);
              probabilities.push(output);
              targets.push(targetToken);
            } finally {
              inputPart.destroy();
              memoryPart.destroy();
            }
          }

          let futureHiddenGradient = zeros(HIDDEN_SIZE);

          for (let step = targets.length - 1; step >= 0; step--) {
            const outputGradient = makeRow(
              subtractOneHot(probabilities[step].download(), targets[step])
            );

            const dWoStep = Tensor.allocate(HIDDEN_SIZE, VOCAB_SIZE);
            const hiddenFromOutput = Tensor.allocate(1, HIDDEN_SIZE);
            const dPreActivation = Tensor.allocate(1, HIDDEN_SIZE);
            const dWxStep = Tensor.allocate(VOCAB_SIZE, HIDDEN_SIZE);
            const dWhStep = Tensor.allocate(HIDDEN_SIZE, HIDDEN_SIZE);
            const priorHiddenGradient = Tensor.allocate(1, HIDDEN_SIZE);

            try {
              dWoStep.backwardTo(hiddenStates[step + 1], outputGradient);
              addInto(dWoHost, dWoStep.download());

              hiddenFromOutput.gradInputTo(Wo, outputGradient);

              const combinedHiddenGradient = addArrays(
                hiddenFromOutput.download(),
                futureHiddenGradient
              );

              dPreActivation.update(
                leakyReluGradient(
                  combinedHiddenGradient,
                  preActivations[step].download()
                )
              );

              dWxStep.backwardTo(inputs[step], dPreActivation);
              addInto(dWxHost, dWxStep.download());

              dWhStep.backwardTo(
                previousHiddenStates[step],
                dPreActivation
              );
              addInto(dWhHost, dWhStep.download());

              priorHiddenGradient.gradInputTo(Wh, dPreActivation);
              futureHiddenGradient = priorHiddenGradient.download();
            } finally {
              outputGradient.destroy();
              dWoStep.destroy();
              hiddenFromOutput.destroy();
              dPreActivation.destroy();
              dWxStep.destroy();
              dWhStep.destroy();
              priorHiddenGradient.destroy();
            }
          }
        } finally {
          destroyAll(inputs);
          destroyAll(preActivations);
          destroyAll(probabilities);
          destroyAll(hiddenStates);
        }
      }

      const dWx = Tensor.fromArray(
        clipGradient(dWxHost, GRADIENT_CLIP),
        VOCAB_SIZE,
        HIDDEN_SIZE
      );

      const dWh = Tensor.fromArray(
        clipGradient(dWhHost, GRADIENT_CLIP),
        HIDDEN_SIZE,
        HIDDEN_SIZE
      );

      const dWo = Tensor.fromArray(
        clipGradient(dWoHost, GRADIENT_CLIP),
        HIDDEN_SIZE,
        VOCAB_SIZE
      );

      Wx.optimizerStep(dWx, LEARNING_RATE);
      Wh.optimizerStep(dWh, LEARNING_RATE);
      Wo.optimizerStep(dWo, LEARNING_RATE);

      dWx.destroy();
      dWh.destroy();
      dWo.destroy();

      if (epoch === 1 || epoch % 250 === 0 || epoch === EPOCHS) {
        const averageLoss = totalLoss / Math.max(1, totalPredictions);
        const accuracy = correct / Math.max(1, totalPredictions);

        console.log(
          "Epoch " +
            epoch +
            "/" +
            EPOCHS +
            " | loss " +
            averageLoss.toFixed(6) +
            " | token accuracy " +
            (accuracy * 100).toFixed(1) +
            "%"
        );
      }
    }

    console.log("");
    console.log("--- PROMPT VALIDATION ---");

    let passedPlans = 0;

    for (let index = 0; index < data.indexedPromptSequences.length; index++) {
      const promptIds = data.indexedPromptSequences[index];
      const expected = data.trainingSequences[index];

      const generated = generateFromPrompt(
        promptIds,
        Wx,
        Wh,
        Wo,
        TOKENS,
        data.maxGenerationTokens
      );

      const generatedWithStart = ["COMPONENT_START", ...generated];

      const matches =
        generatedWithStart.join("|") === expected.join("|");

      if (matches) {
        passedPlans++;
      }

      console.log(
        "Prompt:    " + data.promptSequences[index].join(" -> ")
      );
      console.log("Expected:  " + expected.join(" -> "));
      console.log(
        "Generated: " + generatedWithStart.join(" -> ")
      );
      console.log("Result:    " + (matches ? "PASS" : "FAIL"));
      console.log("");
    }

    console.log(
      "Prompt validation: " +
        passedPlans +
        "/" +
        data.indexedPromptSequences.length +
        " exact matches"
    );

       if (passedPlans !== data.indexedPromptSequences.length) {
      fail(
        "prompt validation failed: " +
          passedPlans +
          "/" +
          data.indexedPromptSequences.length +
          " exact matches; refusing to save checkpoint."
      );
    }

    const meta = {
      architecture: "single_layer_leaky_relu_rnn",
      activation: "leaky_relu",
      negativeSlope: NEGATIVE_SLOPE,
      vocabulary: TOKENS,
      vocabularySize: VOCAB_SIZE,
      hiddenSize: HIDDEN_SIZE,
      maxGenerationTokens: data.maxGenerationTokens,
      epochs: EPOCHS,
      learningRate: LEARNING_RATE,
      trainingSequences: data.trainingSequences,
      promptSequences: data.promptSequences,
      datasetVersion: data.version,
      datasetDescription: data.description,
      exactPromptValidation:
        passedPlans + "/" + data.indexedPromptSequences.length,
    };

    fs.mkdirSync(MODEL_DIR, { recursive: true });

    saveTensor(MODEL_DIR + "/rnn_Wx.bin", Wx);
    saveTensor(MODEL_DIR + "/rnn_Wh.bin", Wh);
    saveTensor(MODEL_DIR + "/rnn_Wo.bin", Wo);

    fs.writeFileSync(
      MODEL_DIR + "/rnn_meta.json",
      JSON.stringify(meta, null, 2) + "\n",
      "utf8"
    );

    console.log("");
    console.log("SUCCESS: V3 multi-sequence RNN checkpoint saved.");
    console.log("Saved: " + MODEL_DIR + "/rnn_Wx.bin");
    console.log("Saved: " + MODEL_DIR + "/rnn_Wh.bin");
    console.log("Saved: " + MODEL_DIR + "/rnn_Wo.bin");
    console.log("Saved: " + MODEL_DIR + "/rnn_meta.json");
  } finally {
    Wx.destroy();
    Wh.destroy();
    Wo.destroy();
  }
}

main();