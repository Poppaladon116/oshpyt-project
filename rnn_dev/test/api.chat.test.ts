import { describe, expect, test } from "vitest";
import { apiCases } from "./fixtures/api-cases";

const baseUrl = process.env.API_BASE_URL ?? "http://localhost:3003";

type ChatResponse = {
  reply: string;
  tokens: string[];
  mode: string;
  stoppedAt: string;
  promptTokens: string[];
};

type ErrorResponse = {
  error: string;
  unknownTokens?: string[];
  reply?: unknown;
  tokens?: unknown;
  stoppedAt?: unknown;
};

async function postChat(body: unknown): Promise<Response> {
  return fetch(`${baseUrl}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(body)
  });
}

function expectNoInferencePayload(body: ErrorResponse): void {
  expect(body.reply).toBeUndefined();
  expect(body.tokens).toBeUndefined();
  expect(body.stoppedAt).toBeUndefined();
}

describe("POST /chat — v4 API regression", () => {
  test.each(apiCases)("$name", async (testCase) => {
    const response = await fetch(`${baseUrl}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        text: testCase.prompt
      })
    });

 expect(response.status).toBe(200);

    const body = (await response.json()) as ChatResponse;

    expect(body.mode).toBe("rnn_template");
    expect(body.stoppedAt).toBe("COMPONENT_END");
    expect(body.promptTokens).toEqual(testCase.prompt.split(" "));
    expect(body.tokens).toEqual(testCase.expectedTokens);
    expect(body.reply).toMatch(testCase.expectedHtml);

    expect(body.reply).not.toMatch(/<script\b/i);
    expect(body.reply).not.toMatch(/\son\w+\s*=/i);
  });
});

describe("POST /chat — input rejection", () => {
  test("rejects an unknown token", async () => {
    const response = await fetch(`${baseUrl}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        text: "BUTTON COLOR_PURPLE"
      })
    });

    expect(response.status).toBe(400);

    const body = (await response.json()) as {
      error: string;
      unknownTokens: string[];
    };

    expect(body.error).toBe("Unsupported prompt token(s).");
    expect(body.unknownTokens).toEqual(["COLOR_PURPLE"]);
  });

  test("rejects a prompt containing mixed valid and unknown tokens", async () => {
    const response = await fetch(`${baseUrl}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        text: "ALERT COLOR_RED UNEXPECTED_TOKEN"
      })
    });

    expect(response.status).toBe(400);

    const body = (await response.json()) as {
      unknownTokens: string[];
    };

    expect(body.unknownTokens).toEqual(["UNEXPECTED_TOKEN"]);
  });

  test("rejects an empty prompt", async () => {
    const response = await fetch(`${baseUrl}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        text: ""
      })
    });

    expect(response.status).toBe(400);

    const body = (await response.json()) as {
      error: string;
    };

    expect(body.error).toBe("Provide at least one supported prompt token.");
  });

  test("rejects a missing text field", async () => {
    const response = await postChat({});
    const body = (await response.json()) as ErrorResponse;

    expect(response.status).toBe(400);
    expect(body.error).toBe("Provide at least one supported prompt token.");
    expectNoInferencePayload(body);
  });

  test("rejects a non-string text field", async () => {
    const response = await postChat({ text: 42 });
    const body = (await response.json()) as ErrorResponse;

    expect(response.status).toBe(400);
    expect(body.error).toBe("Provide at least one supported prompt token.");
    expectNoInferencePayload(body);
  });

  test("rejects whitespace-only text", async () => {
    const response = await postChat({ text: " \t\r\n " });
    const body = (await response.json()) as ErrorResponse;

    expect(response.status).toBe(400);
    expect(body.error).toBe("Provide at least one supported prompt token.");
    expectNoInferencePayload(body);
  });

  test("rejects valid vocabulary tokens without a component trigger", async () => {
    const response = await postChat({
      text: "COLOR_BLUE TEXT_SAVE"
    });
    const body = (await response.json()) as ErrorResponse;

    expect(response.status).toBe(422);
    expect(body.error).toBe("No valid component trigger detected.");
    expectNoInferencePayload(body);
  });

  test("rejects and reports multiple unknown tokens in order", async () => {
    const response = await postChat({
      text: "BUTTON COLOR_PURPLE PAD_GIANT"
    });
    const body = (await response.json()) as ErrorResponse;

    expect(response.status).toBe(400);
    expect(body.error).toBe("Unsupported prompt token(s).");
    expect(body.unknownTokens).toEqual([
      "COLOR_PURPLE",
      "PAD_GIANT"
    ]);
    expectNoInferencePayload(body);
  });

  test("rejects a valid component with an unknown prefix", async () => {
    const response = await postChat({
      text: "HACK BUTTON COLOR_BLUE"
    });
    const body = (await response.json()) as ErrorResponse;

    expect(response.status).toBe(400);
    expect(body.error).toBe("Unsupported prompt token(s).");
    expect(body.unknownTokens).toEqual(["HACK"]);
    expectNoInferencePayload(body);
  });

  test("rejects markup and unsupported punctuation", async () => {
    const response = await postChat({
      text: "BUTTON COLOR_BLUE <script>alert(1)</script>"
    });
    const body = (await response.json()) as ErrorResponse;

    expect(response.status).toBe(400);
    expect(body.error).toBe("Unsupported prompt token(s).");
    expect(body.unknownTokens).toBeDefined();
    expect(body.unknownTokens!.length).toBeGreaterThan(0);
    expectNoInferencePayload(body);
  });
  test("rejects a valid but unsupported component plan", async () => {
  const response = await postChat({
    text: "BADGE COLOR_BLUE TEXT_SAVE"
  });
  const body = (await response.json()) as ErrorResponse;

  expect(response.status).toBe(422);
  expect(body.error).toBe("Prompt does not match a supported component plan.");
  expectNoInferencePayload(body);
  });
});