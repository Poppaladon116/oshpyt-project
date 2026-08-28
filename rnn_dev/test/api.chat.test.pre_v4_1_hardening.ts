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