import * as fs from "fs";

type Example = {
  prompt: string[];
  tokens: string[];
};

type Dataset = {
  version: string;
  description: string;
  maxGenerationTokens: number;
  sequences: Example[];
};

const INPUT_PATH = process.env.DATASET_PATH || "components_v3.json";
const OUTPUT_PATH =
  process.env.OUTPUT_PATH || "components_v3.validated.json";

function fail(message: string): never {
  throw new Error("components_v3 validation failed: " + message);
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}

function cleanToken(value: string): string {
  return value.trim().toUpperCase();
}

function loadDataset(): Dataset {
  if (!fs.existsSync(INPUT_PATH)) {
    fail("input file not found: " + INPUT_PATH);
  }

  let raw: unknown;

  try {
    raw = JSON.parse(fs.readFileSync(INPUT_PATH, "utf8"));
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    fail("could not parse JSON: " + message);
  }

  if (!isObject(raw)) {
    fail("root must be a JSON object.");
  }

  const version = raw.version;
  const description = raw.description;
  const maxTokens = raw.maxGenerationTokens;
  const sourceSequences = raw.sequences;

  if (typeof version !== "string" || version.trim() === "") {
    fail('"version" must be a non-empty string.');
  }

  if (typeof description !== "string" || description.trim() === "") {
    fail('"description" must be a non-empty string.');
  }

  if (
    typeof maxTokens !== "number" ||
    !Number.isInteger(maxTokens) ||
    maxTokens < 2
  ) {
    fail('"maxGenerationTokens" must be an integer of at least 2.');
  }

  if (!Array.isArray(sourceSequences) || sourceSequences.length === 0) {
    fail('"sequences" must be a non-empty array.');
  }

  const sequences: Example[] = [];

  sourceSequences.forEach((item, index) => {
    if (!isObject(item)) {
      fail("sequences[" + index + "] must be an object.");
    }

    if (!isStringArray(item.prompt) || item.prompt.length === 0) {
      fail(
        "sequences[" + index + "].prompt must be a non-empty string array."
      );
    }

    if (!isStringArray(item.tokens) || item.tokens.length < 3) {
      fail(
        "sequences[" + index + "].tokens must have at least three strings."
      );
    }

    const prompt = item.prompt.map(cleanToken);
    const tokens = item.tokens.map(cleanToken);

    if (prompt.some((token) => token === "")) {
      fail("sequences[" + index + "].prompt contains an empty token.");
    }

    if (tokens.some((token) => token === "")) {
      fail("sequences[" + index + "].tokens contains an empty token.");
    }

    if (tokens[0] !== "COMPONENT_START") {
      fail("sequences[" + index + "] must begin with COMPONENT_START.");
    }

    if (tokens[tokens.length - 1] !== "COMPONENT_END") {
      fail("sequences[" + index + "] must end with COMPONENT_END.");
    }

    if (tokens.length > maxTokens) {
      fail(
        "sequences[" +
          index +
          "] has " +
          tokens.length +
          " tokens; maximum is " +
          maxTokens +
          "."
      );
    }

    sequences.push({ prompt, tokens });
  });

  return {
    version: version.trim(),
    description: description.trim(),
    maxGenerationTokens: maxTokens,
    sequences,
  };
}

function main(): void {
  const dataset = loadDataset();

  const vocabulary: string[] = [];
  const tokenToId: Record<string, number> = {};
  const seenTokens = new Set<string>();
  const seenPrompts = new Set<string>();

  for (let i = 0; i < dataset.sequences.length; i++) {
    const example = dataset.sequences[i];
    const promptKey = example.prompt.join(" ");

    if (seenPrompts.has(promptKey)) {
      fail("duplicate prompt sequence: " + promptKey);
    }

    seenPrompts.add(promptKey);

    for (const token of example.tokens) {
      if (!seenTokens.has(token)) {
        seenTokens.add(token);
        tokenToId[token] = vocabulary.length;
        vocabulary.push(token);
      }
    }
  }

  if (vocabulary.length === 0) {
    fail("generated vocabulary is empty.");
  }

  if (vocabulary[0] !== "COMPONENT_START") {
    fail("vocabulary must start with COMPONENT_START.");
  }

  if (tokenToId["COMPONENT_END"] === undefined) {
    fail("vocabulary must include COMPONENT_END.");
  }

  const promptSequences = dataset.sequences.map((example) => example.prompt);
  const trainingSequences = dataset.sequences.map((example) => example.tokens);

  const indexedPromptSequences = promptSequences.map((sequence, sequenceIndex) =>
    sequence.map((token, tokenIndex) => {
      const tokenId = tokenToId[token];

      if (tokenId === undefined) {
        fail(
          "prompt token missing from vocabulary at sequence " +
            sequenceIndex +
            ", token " +
            tokenIndex +
            ": " +
            token
        );
      }

      return tokenId;
    })
  );

  const indexedTrainingSequences = trainingSequences.map(
    (sequence, sequenceIndex) =>
      sequence.map((token, tokenIndex) => {
        const tokenId = tokenToId[token];

        if (tokenId === undefined) {
          fail(
            "training token missing from vocabulary at sequence " +
              sequenceIndex +
              ", token " +
              tokenIndex +
              ": " +
              token
          );
        }

        if (tokenId < 0 || tokenId >= vocabulary.length) {
          fail(
            "invalid token index " +
              tokenId +
              " at training sequence " +
              sequenceIndex +
              ", token " +
              tokenIndex
          );
        }

        return tokenId;
      })
  );

  const validated = {
    version: dataset.version,
    description: dataset.description,
    vocabulary,
    vocabularySize: vocabulary.length,
    tokenToId,
    promptSequences,
    trainingSequences,
    indexedPromptSequences,
    indexedTrainingSequences,
    maxGenerationTokens: dataset.maxGenerationTokens,
  };

  fs.writeFileSync(
    OUTPUT_PATH,
    JSON.stringify(validated, null, 2) + "\n",
    "utf8"
  );

  console.log("VALID: components dataset accepted.");
  console.log("Input: " + INPUT_PATH);
  console.log("Output: " + OUTPUT_PATH);
  console.log("Version: " + validated.version);
  console.log("Examples: " + validated.trainingSequences.length);
  console.log("Vocabulary size: " + validated.vocabularySize);
  console.log("Vocabulary: " + validated.vocabulary.join(" -> "));
}

main();