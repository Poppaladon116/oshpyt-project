import * as fs from "fs";
import * as path from "path";
import { OshpytTokenizer } from "./tokenizer";
import { Tensor } from "./OshpytTensor";

const SIZE = 1024;
const LAYER_COUNT = 18;

const GOAL_LOBE_SIZE = 512;
const HISTORY_OFFSET = 512;
const CLOCK_INDEX = SIZE - 1;
const MAX_USABLE_TOKEN_ID = 510;

const GOAL_STRENGTH = 15.0;
const HISTORY_STRENGTH = 10.0;
const CLOCK_STRENGTH = 1.0;

const LEARNING_RATE = 0.00005;
const EPOCHS = 100;

const MODEL_DIR = process.env.MODEL_DIR ?? process.cwd();
const DATA_PATH = path.join(process.cwd(), "data.txt");

type OutputMode = "html" | "svg" | "render" | "animation";

interface TrainingExample {
  trigger: string;
  mode: OutputMode;
  tokens: number[];
}

function destroyAll(tensors: Tensor[]): void {
  for (const tensor of tensors) {
    tensor.destroy();
  }
}

function validateTokenId(
  tokenId: number,
  label: string
): void {
  if (
    !Number.isInteger(tokenId) ||
    tokenId < 1 ||
    tokenId > MAX_USABLE_TOKEN_ID
  ) {
    throw new Error(
      `${label} has invalid ID ${tokenId}. ` +
      `Usable IDs are 1-${MAX_USABLE_TOKEN_ID}.`
    );
  }
}

function getModeFromTrigger(trigger: string): OutputMode {
  if (trigger.includes("_anim_")) {
    return "animation";
  }

  if (trigger.includes("_svg")) {
    return "svg";
  }

  if (
    trigger.includes("_draw_") ||
    trigger.includes("_dither_") ||
    trigger.includes("_shade_") ||
    trigger.includes("_head") ||
    trigger.includes("_sphere")
  ) {
    return "render";
  }

  return "html";
}

function getModeToken(mode: OutputMode): string {
  switch (mode) {
    case "animation":
      return "frame_1";
    case "render":
      return "render";
    case "svg":
      return "svg";
    default:
      return "html";
  }
}

function parseTrainingExamples(
  rawData: string,
  tokenizer: OshpytTokenizer
): TrainingExample[] {
  const chunks = rawData
    .toLowerCase()
    .split("component_end")
    .map((chunk) => chunk.trim())
    .filter((chunk) => chunk.length > 0);

  if (chunks.length < 2) {
    throw new Error(
      `Expected multiple records separated by component_end; found ${chunks.length}.`
    );
  }

  const examples: TrainingExample[] = [];

  for (const chunk of chunks) {
    const rawTokens = tokenizer.encode(chunk);

    if (rawTokens.length < 2) {
      continue;
    }

    const triggerText = tokenizer.decode([rawTokens[0]])
      .toLowerCase()
      .trim();

    if (!triggerText.startsWith("oshpyt_trigger_")) {
      throw new Error(
        `Every record must begin with oshpyt_trigger_*. Found: "${triggerText}".`
      );
    }

    const mode = getModeFromTrigger(triggerText);
    const modeId = tokenizer.vocab.get(getModeToken(mode));

    if (modeId == null) {
      throw new Error(
        `Missing required mode token "${getModeToken(mode)}" in vocabulary.`
      );
    }

    /*
     * Sequence format:
     * trigger -> mode -> actual output tokens -> component_end
     *
     * We deliberately insert mode after the trigger, even if the record
     * already starts with html/render/frame_1/etc. This creates a consistent,
     * unambiguous first prediction for every output family.
     */
    const tokens = [
      rawTokens[0],
      modeId,
      ...rawTokens.slice(1),
      tokenizer.vocab.get("component_end"),
    ].filter((id): id is number => id != null);

    if (tokens.length < 4) {
      continue;
    }

    for (const tokenId of tokens) {
      validateTokenId(tokenId, "Training token");
    }

    examples.push({
      trigger: triggerText,
      mode,
      tokens,
    });
  }

  if (examples.length < 2) {
    throw new Error("No usable training examples were parsed.");
  }

  return examples;
}

function createInput(
  goalToken: number,
  previousToken: number,
  position: number,
  totalSteps: number
): Float32Array {
  validateTokenId(goalToken, "Goal token");
  validateTokenId(previousToken, "Previous token");

  const input = new Float32Array(SIZE);

  input[goalToken] = GOAL_STRENGTH;
  input[HISTORY_OFFSET + previousToken] = HISTORY_STRENGTH;
  input[CLOCK_INDEX] =
    CLOCK_STRENGTH * position / Math.max(totalSteps, 1);

  return input;
}

function getTargetWeight(
  tokenText: string,
  position: number
): number {
  /*
   * The first output token and termination token are structurally important.
   * Binary 0/1 values occur many times in render examples, so they should
   * not outweigh all HTML/structure classes.
   */
  if (position === 1) {
    return 4.0;
  }

  if (
    tokenText === "html" ||
    tokenText === "svg" ||
    tokenText === "render" ||
    tokenText === "animation" ||
    tokenText === "frame_1" ||
    tokenText === "component_end"
  ) {
    return 3.0;
  }

  if (tokenText === "0" || tokenText === "1") {
    return 0.35;
  }

  return 1.0;
}

function createGradient(
  probabilities: Float32Array,
  targetToken: number,
  targetWeight: number
): Float32Array {
  const gradient = new Float32Array(probabilities.length);

  for (let index = 0; index < probabilities.length; index++) {
    gradient[index] = probabilities[index] * targetWeight;
  }

  gradient[targetToken] -= targetWeight;

  return gradient;
}

function initializeWeights(weights: Tensor[]): void {
  const limit = Math.sqrt(6.0 / SIZE);

  for (const weight of weights) {
    const values = new Float32Array(weight.size);

    for (let index = 0; index < values.length; index++) {
      values[index] = (Math.random() * 2.0 - 1.0) * limit;
    }

    weight.update(values);
  }
}

function saveModel(
  weights: Tensor[],
  epoch: number,
  tokenizer: OshpytTokenizer
): void {
  for (let layer = 0; layer < weights.length; layer++) {
    const values = weights[layer].download();

    const filePath = path.join(
      MODEL_DIR,
      `model_l${layer + 1}.bin`
    );

    fs.writeFileSync(
      filePath,
      Buffer.from(
        values.buffer,
        values.byteOffset,
        values.byteLength
      )
    );
  }

  fs.writeFileSync(
    path.join(MODEL_DIR, "model_meta.json"),
    JSON.stringify(
      {
        epoch,
        layerCount: LAYER_COUNT,
        size: SIZE,
        vocabularySize: tokenizer.vocab.size,
        format: "trigger_mode_previous_token_v1"
      },
      null,
      2
    )
  );
}

function getArgMax(
  values: Float32Array,
  vocabularySize: number
): number {
  const upperBound = Math.min(
    values.length,
    vocabularySize,
    MAX_USABLE_TOKEN_ID + 1
  );

  let bestId = 1;
  let bestValue = -Infinity;

  for (let index = 1; index < upperBound; index++) {
    if (values[index] > bestValue) {
      bestValue = values[index];
      bestId = index;
    }
  }

  return bestId;
}

async function main(): Promise<void> {
  console.log("--- OSHPYT 18-LAYER WIDE-BODY TRAINING ---");
  console.log("OSHPYT: Analyzing textbook for unique concepts...");

  if (!fs.existsSync(DATA_PATH)) {
    throw new Error(`Missing training data: ${DATA_PATH}`);
  }

  const rawData = fs.readFileSync(DATA_PATH, "utf-8")
    .toLowerCase();

  const tokenizer = new OshpytTokenizer();
  tokenizer.buildVocab(rawData);

  /*
   * Required reserved vocabulary values must exist before parsing.
   */
 for (const requiredToken of [
  "html",
  "svg",
  "render",
  "frame_1",
  "component_end",
]) {
    if (!tokenizer.vocab.has(requiredToken)) {
      throw new Error(
        `Vocabulary does not contain required token "${requiredToken}".`
      );
    }
  }

  if (tokenizer.vocab.size > MAX_USABLE_TOKEN_ID + 1) {
    throw new Error(
      `Vocabulary has ${tokenizer.vocab.size} entries. ` +
      `Maximum supported count is ${MAX_USABLE_TOKEN_ID + 1}.`
    );
  }

  const examples = parseTrainingExamples(rawData, tokenizer);

  console.log(
    `OSHPYT: ${tokenizer.vocab.size} tokens, ${examples.length} snippets.`
  );

  const weights = Array.from(
    { length: LAYER_COUNT },
    () => Tensor.allocate(SIZE, SIZE)
  );

  const activations = Array.from(
    { length: LAYER_COUNT + 1 },
    () => Tensor.allocate(1, SIZE)
  );

  const gradients = Array.from(
    { length: LAYER_COUNT + 1 },
    () => Tensor.allocate(1, SIZE)
  );

  const weightGradients = Array.from(
    { length: LAYER_COUNT },
    () => Tensor.allocate(SIZE, SIZE)
  );

  try {
    initializeWeights(weights);

    const startedAt = Date.now();

    for (let epoch = 1; epoch <= EPOCHS; epoch++) {
      let weightedLoss = 0.0;
      let totalWeight = 0.0;
      let correct = 0;
      let totalPredictions = 0;

      process.stdout.write(`Epoch ${epoch}/${EPOCHS} `);

      for (const example of examples) {
        const tokens = example.tokens;

        for (let position = 1; position < tokens.length; position++) {
          const goalToken = tokens[0];
          const previousToken = tokens[position - 1];
          const targetToken = tokens[position];

          activations[0].update(
            createInput(
              goalToken,
              previousToken,
              position,
              tokens.length - 1
            )
          );

          for (let layer = 0; layer < LAYER_COUNT; layer++) {
            activations[layer].matmulTo(
              weights[layer],
              activations[layer + 1]
            );

            if (layer < LAYER_COUNT - 1) {
              activations[layer + 1].relu();
            }
          }

          activations[LAYER_COUNT].softmax();

          const probabilities = activations[LAYER_COUNT].download();
          const targetText = tokenizer.decode([targetToken])
            .toLowerCase()
            .trim();

          const targetWeight = getTargetWeight(
            targetText,
            position
          );

          const safeProbability = Math.max(
            probabilities[targetToken],
            1e-12
          );

          weightedLoss +=
            -Math.log(safeProbability) * targetWeight;

          totalWeight += targetWeight;
          totalPredictions++;

          if (
            getArgMax(probabilities, tokenizer.vocab.size) ===
            targetToken
          ) {
            correct++;
          }

          gradients[LAYER_COUNT].update(
            createGradient(
              probabilities,
              targetToken,
              targetWeight
            )
          );

          for (let layer = LAYER_COUNT - 1; layer >= 0; layer--) {
            weightGradients[layer].backwardTo(
              activations[layer],
              gradients[layer + 1]
            );

            gradients[layer].gradInputTo(
              weights[layer],
              gradients[layer + 1]
            );

            if (layer > 0) {
              gradients[layer].reluBackwardFrom(
                activations[layer]
              );
            }
          }

          for (let layer = 0; layer < LAYER_COUNT; layer++) {
            weights[layer].optimizerStep(
              weightGradients[layer],
              LEARNING_RATE
            );
          }
        }

        process.stdout.write(".");
      }

     saveModel(weights, epoch, tokenizer);

      const loss =
        totalWeight > 0 ? weightedLoss / totalWeight : 0;

      const accuracy =
        totalPredictions > 0
          ? (correct / totalPredictions) * 100
          : 0;

      const minutes =
        (Date.now() - startedAt) / 1000 / 60;

      console.log(
        ` Loss=${loss.toFixed(4)}` +
        ` Accuracy=${accuracy.toFixed(2)}%` +
        ` Saved [${minutes.toFixed(1)} min]`
      );
    }

    console.log("OSHPYT: Training complete.");
  } finally {
    destroyAll([
      ...weights,
      ...activations,
      ...gradients,
      ...weightGradients,
    ]);
  }
}

main().catch((error) => {
  console.error(
    "CRITICAL TRAINING ERROR:",
    error instanceof Error ? error.message : error
  );

  process.exitCode = 1;
});