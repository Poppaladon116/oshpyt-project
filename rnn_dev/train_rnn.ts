import * as fs from "fs";
import { Tensor } from "./OshpytTensor";

const MODEL_DIR = process.env.MODEL_DIR ?? "models/multi_v2";

const TOKENS = ["A", "B", "C", "D", "E", "COMPONENT_END"];
const TOKEN_TO_ID = new Map(
  TOKENS.map((token, index) => [token, index])
);

const sequence = TOKENS.map((token) => {
  const id = TOKEN_TO_ID.get(token);

  if (id == null) {
    throw new Error(`Missing token: ${token}`);
  }

  return id;
});

const VOCAB_SIZE = TOKENS.length;
const HIDDEN_SIZE = 32;

const EPOCHS = 2_000;
const LEARNING_RATE = 0.03;
const GRADIENT_CLIP = 5.0;

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

function oneHot(tokenId: number): Float32Array {
  const values = zeros(VOCAB_SIZE);
  values[tokenId] = 1.0;
  return values;
}

function addArrays(
  left: Float32Array,
  right: Float32Array
): Float32Array {
  if (left.length !== right.length) {
    throw new Error("Array add mismatch.");
  }

  const result = new Float32Array(left.length);

  for (let i = 0; i < result.length; i++) {
    result[i] = left[i] + right[i];
  }

  return result;
}

function subtractOneHot(
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
    clipped[i] = Math.max(
      -limit,
      Math.min(limit, values[i])
    );
  }

  return clipped;
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

function probabilityOf(
  probabilities: Float32Array,
  tokenId: number
): number {
  return Math.max(probabilities[tokenId], 1e-12);
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

function makeRow(data: Float32Array): Tensor {
  return Tensor.fromArray(data, 1, data.length);
}

function main(): void {
  console.log("--- OSHPYT RNN LAB: LEAKY-RELU BPTT ---");
  console.log(
    "OSHPYT: Training A -> B -> C -> D -> E -> COMPONENT_END"
  );

  /*
   * Tensor shape convention:
   *
   * input       = 1 x VOCAB_SIZE
   * hidden      = 1 x HIDDEN_SIZE
   * Wx          = VOCAB_SIZE x HIDDEN_SIZE
   * Wh          = HIDDEN_SIZE x HIDDEN_SIZE
   * Wo          = HIDDEN_SIZE x VOCAB_SIZE
   */

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
      /*
       * Forward-pass records.
       *
       * hiddenStates[0] is h_0, the all-zero state.
       * For step t, hiddenStates[t + 1] is h_t.
       */
      const inputs: Tensor[] = [];
      const previousHiddenStates: Tensor[] = [];
      const preActivations: Tensor[] = [];
      const hiddenStates: Tensor[] = [
        makeRow(zeros(HIDDEN_SIZE))
      ];
      const probabilities: Tensor[] = [];
      const targets: number[] = [];

      let totalLoss = 0;
      let correct = 0;

      try {
        for (let step = 0; step < sequence.length - 1; step++) {
          const inputToken = sequence[step];
          const targetToken = sequence[step + 1];

          const input = makeRow(oneHot(inputToken));
          const previousHidden = hiddenStates[step];

          const inputPart = Tensor.allocate(1, HIDDEN_SIZE);
          const memoryPart = Tensor.allocate(1, HIDDEN_SIZE);
          const preActivation = Tensor.allocate(1, HIDDEN_SIZE);

          input.matmulTo(Wx, inputPart);
          previousHidden.matmulTo(Wh, memoryPart);

          /*
           * preActivation = inputPart + memoryPart.
           * inputPart is then reused as the activated hidden state.
           */
          preActivation.update(inputPart.download());
          preActivation.add(memoryPart);

          const hidden = Tensor.allocate(1, HIDDEN_SIZE);
          hidden.update(preActivation.download());
          hidden.leakyRelu();

          const output = Tensor.allocate(1, VOCAB_SIZE);
          hidden.matmulTo(Wo, output);
          output.softmax();

          const outputValues = output.download();

          totalLoss -= Math.log(
            probabilityOf(outputValues, targetToken)
          );

          if (argmax(outputValues) === targetToken) {
            correct++;
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

        /*
         * Gradient accumulators. The current DLL API computes one matrix
         * gradient at a time, so we accumulate the values on the host.
         */
        const dWxHost = zeros(VOCAB_SIZE * HIDDEN_SIZE);
        const dWhHost = zeros(HIDDEN_SIZE * HIDDEN_SIZE);
        const dWoHost = zeros(HIDDEN_SIZE * VOCAB_SIZE);

        let futureHiddenGradient = zeros(HIDDEN_SIZE);

        for (let step = targets.length - 1; step >= 0; step--) {
          const outputGradientHost = subtractOneHot(
            probabilities[step].download(),
            targets[step]
          );

          const outputGradient = makeRow(outputGradientHost);

          const dWoStep = Tensor.allocate(
            HIDDEN_SIZE,
            VOCAB_SIZE
          );

          dWoStep.backwardTo(
            hiddenStates[step + 1],
            outputGradient
          );

          const dWoValues = dWoStep.download();

          for (let i = 0; i < dWoHost.length; i++) {
            dWoHost[i] += dWoValues[i];
          }

          /*
           * dL/dh from output: outputGradient * Wo^T.
           * Add gradient arriving from h_(t+1) through Wh.
           */
          const hiddenFromOutput = Tensor.allocate(
            1,
            HIDDEN_SIZE
          );

          hiddenFromOutput.gradInputTo(Wo, outputGradient);

          const combinedHiddenGradient = addArrays(
            hiddenFromOutput.download(),
            futureHiddenGradient
          );

          const preActivationGradient = makeRow(
            combinedHiddenGradient
          );

          /*
           * Applies the Leaky-ReLU derivative using the saved z_t.
           */
          preActivationGradient.leakyReluBackward(
            preActivations[step]
          );

          const dWxStep = Tensor.allocate(
            VOCAB_SIZE,
            HIDDEN_SIZE
          );

          dWxStep.backwardTo(
            inputs[step],
            preActivationGradient
          );

          const dWxValues = dWxStep.download();

          for (let i = 0; i < dWxHost.length; i++) {
            dWxHost[i] += dWxValues[i];
          }

          const dWhStep = Tensor.allocate(
            HIDDEN_SIZE,
            HIDDEN_SIZE
          );

          dWhStep.backwardTo(
            previousHiddenStates[step],
            preActivationGradient
          );

          const dWhValues = dWhStep.download();

          for (let i = 0; i < dWhHost.length; i++) {
            dWhHost[i] += dWhValues[i];
          }

          const priorHiddenGradient = Tensor.allocate(
            1,
            HIDDEN_SIZE
          );

          priorHiddenGradient.gradInputTo(
            Wh,
            preActivationGradient
          );

          futureHiddenGradient = priorHiddenGradient.download();

          outputGradient.destroy();
          dWoStep.destroy();
          hiddenFromOutput.destroy();
          preActivationGradient.destroy();
          dWxStep.destroy();
          dWhStep.destroy();
          priorHiddenGradient.destroy();
        }

        /*
         * One SGD update per complete sequence.
         */
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

        if (epoch === 1 || epoch % 100 === 0) {
          const averageLoss = totalLoss / targets.length;
          const accuracy = (correct / targets.length) * 100;

          console.log(
            `Epoch ${epoch}/${EPOCHS} ` +
            `Loss=${averageLoss.toFixed(4)} ` +
            `Accuracy=${accuracy.toFixed(1)}%`
          );
        }
      } finally {
        /*
         * Do not double-free hiddenStates[0]:
         * it is owned only by `hiddenStates`, not `previousHiddenStates`.
         */
        destroyAll(inputs);
        destroyAll(preActivations);
        destroyAll(probabilities);
        destroyAll(hiddenStates);
      }
    }

    /*
     * Greedy validation generation starting at token A.
     */
    let currentToken = TOKEN_TO_ID.get("A")!;

    const generated = ["A"];
    let hidden = makeRow(zeros(HIDDEN_SIZE));

    try {
      for (let step = 0; step < 8; step++) {
        const input = makeRow(oneHot(currentToken));
        const inputPart = Tensor.allocate(1, HIDDEN_SIZE);
        const memoryPart = Tensor.allocate(1, HIDDEN_SIZE);
        const preActivation = Tensor.allocate(1, HIDDEN_SIZE);
        const nextHidden = Tensor.allocate(1, HIDDEN_SIZE);
        const output = Tensor.allocate(1, VOCAB_SIZE);

        input.matmulTo(Wx, inputPart);
        hidden.matmulTo(Wh, memoryPart);

        preActivation.update(inputPart.download());
        preActivation.add(memoryPart);

        nextHidden.update(preActivation.download());
        nextHidden.leakyRelu();

        nextHidden.matmulTo(Wo, output);
        output.softmax();

        const nextToken = argmax(output.download());

        generated.push(TOKENS[nextToken]);

        input.destroy();
        inputPart.destroy();
        memoryPart.destroy();
        preActivation.destroy();
        output.destroy();
        hidden.destroy();

        hidden = nextHidden;
        currentToken = nextToken;

        if (TOKENS[nextToken] === "COMPONENT_END") {
          break;
        }
      }
    } finally {
      hidden.destroy();
    }

	
     fs.mkdirSync(MODEL_DIR, { recursive: true });

    saveTensor(`${MODEL_DIR}/rnn_Wx.bin`, Wx);
    saveTensor(`${MODEL_DIR}/rnn_Wh.bin`, Wh);
    saveTensor(`${MODEL_DIR}/rnn_Wo.bin`, Wo);

    
    console.log(`Generated: ${generated.join(" -> ")}`);
    console.log(
      `SUCCESS: model files saved to ${MODEL_DIR}`
    );
  } finally {
    Wx.destroy();
    Wh.destroy();
    Wo.destroy();
  }
}

main();