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