import * as fs from "fs";
import express from "express";
import { Tensor } from "./OshpytTensor";

type RnnMeta = {
  architecture: string;
  activation: string;
  negativeSlope: number;
  vocabulary: string[];
  vocabularySize: number;
  hiddenSize: number;
  maxGenerationTokens: number;
  epochs?: number;
  learningRate?: number;
  trainingSequences?: readonly (readonly string[])[];
};

type ChatRequest = {
  text?: unknown;
};

const MODEL_DIR = process.env.MODEL_DIR ?? "models/multi_v1";
const PORT = Number(process.env.PORT ?? 3001);

function loadMeta(): RnnMeta {
  const raw = fs.readFileSync(
    `${MODEL_DIR}/rnn_meta.json`,
    "utf8"
  );

  const meta = JSON.parse(raw) as RnnMeta;

  if (
    !Array.isArray(meta.vocabulary) ||
    typeof meta.vocabularySize !== "number" ||
    typeof meta.hiddenSize !== "number"
  ) {
    throw new Error("Invalid RNN metadata file.");
  }

  if (meta.vocabulary.length !== meta.vocabularySize) {
    throw new Error(
      `Metadata mismatch: vocabulary has ${meta.vocabulary.length} tokens, ` +
      `but vocabularySize is ${meta.vocabularySize}.`
    );
  }

  return meta;
}

function loadFloat32(
  fileName: string,
  elementCount: number
): Float32Array {
  const buffer = fs.readFileSync(fileName);
  const expectedBytes =
    elementCount * Float32Array.BYTES_PER_ELEMENT;

  if (buffer.byteLength !== expectedBytes) {
    throw new Error(
      `${fileName}: expected ${expectedBytes} bytes, ` +
      `received ${buffer.byteLength}.`
    );
  }

  /*
   * .slice() creates an independent ArrayBuffer-backed copy,
   * avoiding Node Buffer byte-offset alignment issues.
   */
  return new Float32Array(
    buffer.buffer.slice(
      buffer.byteOffset,
      buffer.byteOffset + buffer.byteLength
    )
  );
}

function zeros(size: number): Float32Array {
  return new Float32Array(size);
}

function oneHot(
  tokenId: number,
  vocabularySize: number
): Float32Array {
  const values = zeros(vocabularySize);
  values[tokenId] = 1.0;
  return values;
}

function makeRow(values: Float32Array): Tensor {
  return Tensor.fromArray(values, 1, values.length);
}

function argmax(
  probabilities: Float32Array,
  forbidden = new Set<number>()
): number {
  let bestIndex = -1;
  let bestValue = -Infinity;

  for (let i = 0; i < probabilities.length; i++) {
    if (forbidden.has(i)) {
      continue;
    }

    if (probabilities[i] > bestValue) {
      bestValue = probabilities[i];
      bestIndex = i;
    }
  }

  if (bestIndex < 0) {
    throw new Error("No valid output token is available.");
  }

  return bestIndex;
}

function tokenizePrompt(
  text: string,
  vocabulary: Map<string, number>
): string[] {
  return text
    .trim()
    .toUpperCase()
    .split(/[\s,;|]+/)
    .filter(Boolean)
    .map((token) => token.replace(/[^A-Z0-9_]/g, ""))
    .filter(Boolean)
    .filter((token) => vocabulary.has(token));
}

function renderTokens(tokens: string[]): string | null {
  const key = tokens.join(" ");

  const templates: Record<string, string> = {
    "COMPONENT_START BUTTON COLOR_BLUE TEXT_SAVE COMPONENT_END":
      '<button type="button" style="padding:10px 16px;border:0;border-radius:8px;background:#2563eb;color:#ffffff;font-weight:700;cursor:pointer;">save</button>',

    "COMPONENT_START BUTTON COLOR_GREEN TEXT_SUBMIT COMPONENT_END":
      '<button type="submit" style="padding:10px 16px;border:0;border-radius:8px;background:#16a34a;color:#ffffff;font-weight:700;cursor:pointer;">submit</button>',

  "COMPONENT_START BUTTON COLOR_RED TEXT_DELETE COMPONENT_END":
    `<button type="button" style="padding:10px 16px;border:0;border-radius:8px;background:#dc2626;color:#ffffff;font-weight:700;cursor:pointer;">delete</button>`,

  "COMPONENT_START BUTTON COLOR_YELLOW TEXT_CANCEL COMPONENT_END":
    `<button type="button" style="padding:10px 16px;border:0;border-radius:8px;background:#ca8a04;color:#111827;font-weight:700;cursor:pointer;">cancel</button>`,

    "COMPONENT_START OVOID_HEAD COLOR_SKIN SHADE_SOFT COMPONENT_END":
      '<div aria-label="ovoid head" role="img" style="width:160px;height:210px;display:block;border-radius:50% 50% 46% 46% / 43% 43% 57% 57%;background:linear-gradient(135deg,#f3c59a 0%,#b96e48 100%);box-shadow:inset 18px 12px 28px rgba(255,255,255,.25),inset -18px -16px 28px rgba(0,0,0,.25),0 12px 24px rgba(0,0,0,.22);"></div>',

    "COMPONENT_START CONTACT_FORM FIELD_NAME FIELD_EMAIL BUTTON_SUBMIT COMPONENT_END":
      '<form style="width:280px;padding:18px;box-sizing:border-box;border:1px solid #e5e7eb;border-radius:12px;background:#ffffff;font-family:Arial,sans-serif;"><label for="rnn-name" style="display:block;margin-bottom:6px;color:#374151;">name</label><input id="rnn-name" name="name" type="text" style="box-sizing:border-box;width:100%;padding:10px;border:1px solid #d1d5db;border-radius:8px;"><label for="rnn-email" style="display:block;margin:12px 0 6px;color:#374151;">email</label><input id="rnn-email" name="email" type="email" style="box-sizing:border-box;width:100%;padding:10px;border:1px solid #d1d5db;border-radius:8px;"><button type="submit" style="margin-top:14px;padding:10px 16px;border:0;border-radius:8px;background:#2563eb;color:#ffffff;font-weight:700;cursor:pointer;">submit</button></form>',

    "COMPONENT_START ANIMATION MOVE_RIGHT DURATION_SHORT COMPONENT_END":
      '<style>@keyframes oshRnnMoveRight{0%,100%{transform:translateX(0)}50%{transform:translateX(80px)}}</style><div style="display:inline-block;padding:16px;border-radius:10px;background:#2563eb;color:#ffffff;font-family:Arial,sans-serif;animation:oshRnnMoveRight 1.2s ease-in-out infinite;">moves right</div>',

    "COMPONENT_START ANIMATION GROW DURATION_SHORT COMPONENT_END":
      '<style>@keyframes oshRnnGrow{0%,100%{transform:scale(1)}50%{transform:scale(1.28)}}</style><div style="display:inline-block;padding:18px 22px;border-radius:10px;background:#16a34a;color:#ffffff;font-family:Arial,sans-serif;transform-origin:center;animation:oshRnnGrow 1.2s ease-in-out infinite;">grows</div>',
  };

  return templates[key] ?? null;
}

function generateTokens(
  promptTokens: string[],
  meta: RnnMeta,
  vocabulary: Map<string, number>,
  Wx: Tensor,
  Wh: Tensor,
  Wo: Tensor
): string[] {
  const startToken = "COMPONENT_START";
  const endToken = "COMPONENT_END";

  const startId = vocabulary.get(startToken);
  const endId = vocabulary.get(endToken);

  if (startId == null || endId == null) {
    throw new Error(
      "Checkpoint vocabulary requires COMPONENT_START and COMPONENT_END."
    );
  }

  /*
   * Avoid duplicate start/end control tokens supplied by a client.
   */
  const cleanPrompt = promptTokens.filter(
    (token) =>
      token !== startToken &&
      token !== endToken
  );

  const seedTokens = [startToken, ...cleanPrompt];
  const generatedTokens = [...seedTokens];

  let hidden = makeRow(zeros(meta.hiddenSize));

  try {
    /*
     * Feed all seed tokens except the final one. The final token is used
     * to predict the first generated continuation token.
     */
    for (let i = 0; i < seedTokens.length - 1; i++) {
      const inputId = vocabulary.get(seedTokens[i]);

      if (inputId == null) {
        throw new Error(`Unknown seed token: ${seedTokens[i]}`);
      }

      const input = makeRow(
        oneHot(inputId, meta.vocabularySize)
      );
      const inputPart = Tensor.allocate(
        1,
        meta.hiddenSize
      );
      const memoryPart = Tensor.allocate(
        1,
        meta.hiddenSize
      );
      const preActivation = Tensor.allocate(
        1,
        meta.hiddenSize
      );
      const nextHidden = Tensor.allocate(
        1,
        meta.hiddenSize
      );

      try {
        input.matmulTo(Wx, inputPart);
        hidden.matmulTo(Wh, memoryPart);

        preActivation.update(inputPart.download());
        preActivation.add(memoryPart);

        nextHidden.update(preActivation.download());
        nextHidden.leakyRelu();

        hidden.destroy();
        hidden = nextHidden;
      } finally {
        input.destroy();
        inputPart.destroy();
        memoryPart.destroy();
        preActivation.destroy();
      }
    }

    let currentToken = seedTokens[seedTokens.length - 1];

    for (
      let step = 0;
      step < meta.maxGenerationTokens;
      step++
    ) {
      const inputId = vocabulary.get(currentToken);

      if (inputId == null) {
        throw new Error(`Unknown current token: ${currentToken}`);
      }

      const input = makeRow(
        oneHot(inputId, meta.vocabularySize)
      );
      const inputPart = Tensor.allocate(
        1,
        meta.hiddenSize
      );
      const memoryPart = Tensor.allocate(
        1,
        meta.hiddenSize
      );
      const preActivation = Tensor.allocate(
        1,
        meta.hiddenSize
      );
      const nextHidden = Tensor.allocate(
        1,
        meta.hiddenSize
      );
      const output = Tensor.allocate(
        1,
        meta.vocabularySize
      );

      try {
        input.matmulTo(Wx, inputPart);
        hidden.matmulTo(Wh, memoryPart);

        preActivation.update(inputPart.download());
        preActivation.add(memoryPart);

        nextHidden.update(preActivation.download());
        nextHidden.leakyRelu();

        nextHidden.matmulTo(Wo, output);
        output.softmax();

        /*
         * Do not generate COMPONENT_START after generation begins.
         */
        const nextId = argmax(
          output.download(),
          new Set([startId])
        );

        const nextToken = meta.vocabulary[nextId];

        generatedTokens.push(nextToken);

        hidden.destroy();
        hidden = nextHidden;
        currentToken = nextToken;

        if (nextId === endId) {
          break;
        }
      } finally {
        input.destroy();
        inputPart.destroy();
        memoryPart.destroy();
        preActivation.destroy();
        output.destroy();
      }
    }

    return generatedTokens;
  } finally {
    hidden.destroy();
  }
}

function main(): void {
  const meta = loadMeta();

  const vocabulary = new Map<string, number>(
    meta.vocabulary.map((token, index) => [token, index])
  );

  const Wx = Tensor.fromArray(
  loadFloat32(
    `${MODEL_DIR}/rnn_Wx.bin`,
    meta.vocabularySize * meta.hiddenSize
  ),
  meta.vocabularySize,
  meta.hiddenSize
);

const Wh = Tensor.fromArray(
  loadFloat32(
    `${MODEL_DIR}/rnn_Wh.bin`,
    meta.hiddenSize * meta.hiddenSize
  ),
  meta.hiddenSize,
  meta.hiddenSize
);

const Wo = Tensor.fromArray(
  loadFloat32(
    `${MODEL_DIR}/rnn_Wo.bin`,
    meta.hiddenSize * meta.vocabularySize
  ),
  meta.hiddenSize,
  meta.vocabularySize
);

  const app = express();

  app.use(express.json({ limit: "32kb" }));

  app.get("/health", (_request, response) => {
    response.json({
      ok: true,
      model: meta.architecture,
      vocabularySize: meta.vocabularySize,
      hiddenSize: meta.hiddenSize,
    });
  });

  app.post("/chat", (request, response) => {
    const body = request.body as ChatRequest;

    if (typeof body.text !== "string" || !body.text.trim()) {
      return response.status(400).json({
        error:
          "Send a JSON request with a non-empty text field, for example: {\"text\":\"BUTTON COLOR_BLUE\"}.",
      });
    }

    try {
      const promptTokens = tokenizePrompt(
        body.text,
        vocabulary
      );

      if (promptTokens.length === 0) {
        return response.status(400).json({
          error: "No recognized RNN vocabulary tokens were found.",
          vocabulary: meta.vocabulary.filter(
            (token) =>
              token !== "COMPONENT_START" &&
              token !== "COMPONENT_END"
          ),
        });
      }

      const tokens = generateTokens(
        promptTokens,
        meta,
        vocabulary,
        Wx,
        Wh,
        Wo
      );

      const completed =
        tokens[tokens.length - 1] === "COMPONENT_END";

      const reply = completed
        ? renderTokens(tokens)
        : null;

      if (!completed) {
        return response.status(422).json({
          error:
            "Generation reached its token limit before COMPONENT_END.",
          tokens,
          reply: null,
        });
      }

      if (reply == null) {
        return response.status(422).json({
          error:
            "The RNN generated a complete sequence with no deterministic renderer template.",
          tokens,
          reply: null,
        });
      }

      console.log(
        `RNN response: ${tokens.join(" -> ")}`
      );

      return response.json({
        reply,
        tokens,
        mode: "rnn_template",
        stoppedAt: "COMPONENT_END",
        promptTokens,
      });
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : "Unknown RNN server error.";

      console.error("RNN error:", message);

      return response.status(500).json({
        error: message,
      });
    }
  });

  const server = app.listen(PORT, () => {
    console.log("OSHPYT RNN server active.");
    console.log(
      `RNN API listening on http://localhost:${PORT}`
    );
    console.log(
      `Loaded ${meta.architecture}; vocabulary ${meta.vocabularySize}; hidden width ${meta.hiddenSize}.`
    );
  });

  function shutdown(): void {
    console.log("\nOSHPYT RNN server shutting down.");

    server.close(() => {
      Wx.destroy();
      Wh.destroy();
      Wo.destroy();
      process.exit(0);
    });
  }

  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

main();