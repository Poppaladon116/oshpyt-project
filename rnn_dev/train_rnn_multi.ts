import * as fs from "fs";
import { Tensor } from "./OshpytTensor";

const MODEL_DIR = process.env.MODEL_DIR ?? "models/multi_v2";

const TRAINING_SEQUENCES = [
  [
    "COMPONENT_START",
    "BUTTON",
    "COLOR_BLUE",
    "TEXT_SAVE",
    "COMPONENT_END",
  ],
  [
    "COMPONENT_START",
    "BUTTON",
    "COLOR_GREEN",
    "TEXT_SUBMIT",
    "COMPONENT_END",
  ],
  [
    "COMPONENT_START",
    "OVOID_HEAD",
    "COLOR_SKIN",
    "SHADE_SOFT",
    "COMPONENT_END",
  ],
  [
    "COMPONENT_START",
    "CONTACT_FORM",
    "FIELD_NAME",
    "FIELD_EMAIL",
    "BUTTON_SUBMIT",
    "COMPONENT_END",
  ],
  [
    "COMPONENT_START",
    "ANIMATION",
    "MOVE_RIGHT",
    "DURATION_SHORT",
    "COMPONENT_END",
  ],
  [
    "COMPONENT_START",
    "ANIMATION",
    "GROW",
    "DURATION_SHORT",
    "COMPONENT_END",
  ],
] as const;

const TEST_PROMPTS = [
  ["COMPONENT_START", "BUTTON", "COLOR_BLUE"],
  ["COMPONENT_START", "BUTTON", "COLOR_GREEN"],
  ["COMPONENT_START", "OVOID_HEAD"],
  ["COMPONENT_START", "CONTACT_FORM"],
  ["COMPONENT_START", "ANIMATION", "MOVE_RIGHT"],
  ["COMPONENT_START", "ANIMATION", "GROW"],
] as const;

const TOKENS = [...new Set(TRAINING_SEQUENCES.flat())];

const TOKEN_TO_ID = new Map<string, number>(
  TOKENS.map((token, index) => [token, index])
);

const TOKEN_SEQUENCES = TRAINING_SEQUENCES.map((sequence) =>
  sequence.map((token) => {
    const tokenId = TOKEN_TO_ID.get(token);

    if (tokenId == null) {
      throw new Error(`Vocabulary token is missing: ${token}`);
    }

    return tokenId;
  })
);

const VOCAB_SIZE = TOKENS.length;
const HIDDEN_SIZE = 48;

const EPOCHS = 4_000;
const LEARNING_RATE = 0.02;
const GRADIENT_CLIP = 5.0;
const MAX_GENERATION_TOKENS = 12;

function zeros(size: number): Float32Array {
  return new Float32Array(size);
}

function randomWeights(
  size: number,
  scale = 0.08
): Float32Array {
  const values = new Float32Array(size);

  for (let i = 0; i < values.length; i++) {
    values[i] = (Math.random() * 2 - 1) * scale;
  }

  return values;
}

function oneHot(tokenId: number): Float32Array {
  const values = zeros(VOCAB_SIZE);
  values[tokenId] = 1.0;
  return values;
}

function makeRow(values: Float32Array): Tensor {
  return Tensor.fromArray(values, 1, values.length);
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

function addArrays(
  left: Float32Array,
  right: Float32Array
): Float32Array {
  if (left.length !== right.length) {
    throw new Error("Array length mismatch during gradient addition.");
  }

  const result = new Float32Array(left.length);

  for (let i = 0; i < result.length; i++) {
    result[i] = left[i] + right[i];
  }

  return result;
}

function addInto(
  destination: Float32Array,
  source: Float32Array
): void {
  if (destination.length !== source.length) {
    throw new Error("Array length mismatch during accumulation.");
  }

  for (let i = 0; i < destination.length; i++) {
    destination[i] += source[i];
  }
}

function outputGradient(
  probabilities: Float32Array,
  targetToken: number
): Float32Array {
  const gradient = new Float32Array(probabilities);
  gradient[targetToken] -= 1.0;
  return gradient;
}

function clipGradient(
  values: Float32Array,
  limit: number
): Float32Array {
  const clipped = new Float32Array(values.length);

  for (let i = 0; i < values.length; i++) {
    clipped[i] = Math.max(-limit, Math.min(limit, values[i]));
  }

  return clipped;
}

function saveTensor(path: string, tensor: Tensor): void {
  const values = tensor.download();

  fs.writeFileSync(
    path,
    Buffer.from(
      values.buffer,
      values.byteOffset,
      values.byteLength
    )
  );
}

function destroyAll(tensors: Tensor[]): void {
  for (const tensor of tensors) {
    tensor.destroy();
  }
}

function getTokenId(token: string): number {
  const id = TOKEN_TO_ID.get(token);

  if (id == null) {
    throw new Error(`Unknown token in prompt: ${token}`);
  }

  return id;
}

function forwardToken(
  inputToken: number,
  hidden: Tensor,
  Wx: Tensor,
  Wh: Tensor,
  Wo: Tensor
): {
  nextHidden: Tensor;
  probabilities: Tensor;
  temporaries: Tensor[];
} {
  const input = makeRow(oneHot(inputToken));
  const inputPart = Tensor.allocate(1, HIDDEN_SIZE);
  const memoryPart = Tensor.allocate(1, HIDDEN_SIZE);
  const preActivation = Tensor.allocate(1, HIDDEN_SIZE);
  const nextHidden = Tensor.allocate(1, HIDDEN_SIZE);
  const probabilities = Tensor.allocate(1, VOCAB_SIZE);

  input.matmulTo(Wx, inputPart);
  hidden.matmulTo(Wh, memoryPart);

  preActivation.update(inputPart.download());
  preActivation.add(memoryPart);

  nextHidden.update(preActivation.download());
  nextHidden.leakyRelu();

  nextHidden.matmulTo(Wo, probabilities);
  probabilities.softmax();

  return {
    nextHidden,
    probabilities,
    temporaries: [
      input,
      inputPart,
      memoryPart,
      preActivation,
    ],
  };
}

function generateFromPrompt(
  prompt: readonly string[],
  Wx: Tensor,
  Wh: Tensor,
  Wo: Tensor
): string[] {
  let hidden = makeRow(zeros(HIDDEN_SIZE));
  const generated = [...prompt];

  try {
    /*
     * Prime recurrent state with every prompt token except its final token.
     * Then use the final prompt token to generate the first continuation.
     */
    for (let i = 0; i < prompt.length - 1; i++) {
      const step = forwardToken(
        getTokenId(prompt[i]),
        hidden,
        Wx,
        Wh,
        Wo
      );

      hidden.destroy();
      destroyAll(step.temporaries);
      step.probabilities.destroy();
      hidden = step.nextHidden;
    }

    let currentToken = getTokenId(prompt[prompt.length - 1]);

    for (
      let generatedCount = 0;
      generatedCount < MAX_GENERATION_TOKENS;
      generatedCount++
    ) {
      const step = forwardToken(
        currentToken,
        hidden,
        Wx,
        Wh,
        Wo
      );

      const nextTokenId = argmax(step.probabilities.download());
      const nextToken = TOKENS[nextTokenId];

      generated.push(nextToken);

      hidden.destroy();
      destroyAll(step.temporaries);
      step.probabilities.destroy();
      hidden = step.nextHidden;
      currentToken = nextTokenId;

      if (nextToken === "COMPONENT_END") {
        break;
      }
    }

    return generated;
  } finally {
    hidden.destroy();
  }
}

function main(): void {
  console.log("--- OSHPYT RNN LAB: MULTI-SEQUENCE BPTT ---");
  console.log(
    `Vocabulary: ${VOCAB_SIZE} tokens; hidden width: ${HIDDEN_SIZE}`
  );
  console.log(
    `Training sequences: ${TOKEN_SEQUENCES.length}`
  );

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
      let totalPredictions = 0;
      let totalCorrect = 0;

      for (const sequence of TOKEN_SEQUENCES) {
        const inputs: Tensor[] = [];
        const previousHiddenStates: Tensor[] = [];
        const preActivations: Tensor[] = [];
        const hiddenStates: Tensor[] = [
          makeRow(zeros(HIDDEN_SIZE)),
        ];
        const probabilities: Tensor[] = [];
        const targets: number[] = [];

        try {
          /*
           * Each independent example begins with h_0 = 0.
           */
          for (let step = 0; step < sequence.length - 1; step++) {
            const inputToken = sequence[step];
            const targetToken = sequence[step + 1];
            const input = makeRow(oneHot(inputToken));
            const previousHidden = hiddenStates[step];

            const inputPart = Tensor.allocate(1, HIDDEN_SIZE);
            const memoryPart = Tensor.allocate(1, HIDDEN_SIZE);
            const preActivation = Tensor.allocate(1, HIDDEN_SIZE);
            const hidden = Tensor.allocate(1, HIDDEN_SIZE);
            const output = Tensor.allocate(1, VOCAB_SIZE);

            input.matmulTo(Wx, inputPart);
            previousHidden.matmulTo(Wh, memoryPart);

            preActivation.update(inputPart.download());
            preActivation.add(memoryPart);

            hidden.update(preActivation.download());
            hidden.leakyRelu();

            hidden.matmulTo(Wo, output);
            output.softmax();

            const outputValues = output.download();

            totalLoss -= Math.log(
              Math.max(outputValues[targetToken], 1e-12)
            );

            totalPredictions++;

            if (argmax(outputValues) === targetToken) {
              totalCorrect++;
            }

            inputs.push(input);
            previousHiddenStates.push(previousHidden);
            preActivations.push(preActivation);
            hiddenStates.push(hidden);
            probabilities.push(output);
            targets.push(targetToken);

            inputPart.destroy();
            memoryPart.destroy();
          }

          const dWxHost = zeros(VOCAB_SIZE * HIDDEN_SIZE);
          const dWhHost = zeros(HIDDEN_SIZE * HIDDEN_SIZE);
          const dWoHost = zeros(HIDDEN_SIZE * VOCAB_SIZE);

          let futureHiddenGradient = zeros(HIDDEN_SIZE);

          for (let step = targets.length - 1; step >= 0; step--) {
            const dOutput = makeRow(
              outputGradient(
                probabilities[step].download(),
                targets[step]
              )
            );

            const dWoStep = Tensor.allocate(
              HIDDEN_SIZE,
              VOCAB_SIZE
            );

            dWoStep.backwardTo(
              hiddenStates[step + 1],
              dOutput
            );

            addInto(dWoHost, dWoStep.download());

            const hiddenFromOutput = Tensor.allocate(
              1,
              HIDDEN_SIZE
            );

            hiddenFromOutput.gradInputTo(Wo, dOutput);

            const dPreActivation = makeRow(
              addArrays(
                hiddenFromOutput.download(),
                futureHiddenGradient
              )
            );

            dPreActivation.leakyReluBackward(
              preActivations[step]
            );

            const dWxStep = Tensor.allocate(
              VOCAB_SIZE,
              HIDDEN_SIZE
            );

            dWxStep.backwardTo(inputs[step], dPreActivation);

            addInto(dWxHost, dWxStep.download());

            const dWhStep = Tensor.allocate(
              HIDDEN_SIZE,
              HIDDEN_SIZE
            );

            dWhStep.backwardTo(
              previousHiddenStates[step],
              dPreActivation
            );

            addInto(dWhHost, dWhStep.download());

            const priorHiddenGradient = Tensor.allocate(
              1,
              HIDDEN_SIZE
            );

            priorHiddenGradient.gradInputTo(
              Wh,
              dPreActivation
            );

            futureHiddenGradient = priorHiddenGradient.download();

            dOutput.destroy();
            dWoStep.destroy();
            hiddenFromOutput.destroy();
            dPreActivation.destroy();
            dWxStep.destroy();
            dWhStep.destroy();
            priorHiddenGradient.destroy();
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
        } finally {
          destroyAll(inputs);
          destroyAll(preActivations);
          destroyAll(probabilities);
          destroyAll(hiddenStates);
        }
      }

      if (epoch === 1 || epoch % 100 === 0) {
        const averageLoss = totalLoss / totalPredictions;
        const accuracy = (totalCorrect / totalPredictions) * 100;

        console.log(
          `Epoch ${epoch}/${EPOCHS} ` +
          `Loss=${averageLoss.toFixed(4)} ` +
          `Accuracy=${accuracy.toFixed(1)}%`
        );
      }
    }

    console.log("\n--- PROMPT VALIDATION ---");

    for (const prompt of TEST_PROMPTS) {
      const generated = generateFromPrompt(
        prompt,
        Wx,
        Wh,
        Wo
      );

      console.log(generated.join(" -> "));
    }

   fs.mkdirSync(MODEL_DIR, { recursive: true });

saveTensor(`${MODEL_DIR}/rnn_Wx.bin`, Wx);
saveTensor(`${MODEL_DIR}/rnn_Wh.bin`, Wh);
saveTensor(`${MODEL_DIR}/rnn_Wo.bin`, Wo);
    fs.writeFileSync(
      `${MODEL_DIR}/rnn_meta.json`,
      JSON.stringify(
        {
          architecture: "single_layer_leaky_relu_rnn",
          activation: "leaky_relu",
          negativeSlope: 0.01,
          vocabulary: TOKENS,
          vocabularySize: VOCAB_SIZE,
          hiddenSize: HIDDEN_SIZE,
          maxGenerationTokens: MAX_GENERATION_TOKENS,
          epochs: EPOCHS,
          learningRate: LEARNING_RATE,
          trainingSequences: TRAINING_SEQUENCES,
        },
        null,
        2
      )
    );

fs.mkdirSync(MODEL_DIR, { recursive: true });

    console.log("\nSUCCESS: Multi-sequence RNN weights saved.");
    console.log("SUCCESS: Multi-sequence RNN weights saved.");
console.log(`Saved: ${MODEL_DIR}/rnn_Wx.bin`);
console.log(`Saved: ${MODEL_DIR}/rnn_Wh.bin`);
console.log(`Saved: ${MODEL_DIR}/rnn_Wo.bin`);
console.log(`Saved: ${MODEL_DIR}/rnn_meta.json`);

  } finally {
    Wx.destroy();
    Wh.destroy();
    Wo.destroy();
  }
}

main();