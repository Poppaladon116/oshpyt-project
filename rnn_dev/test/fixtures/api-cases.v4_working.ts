export type ApiCase = {
  name: string;
  prompt: string;
  expectedTokens: string[];
  expectedHtml: RegExp;
};

export const apiCases: ApiCase[] = [
  {
    name: "blue save button",
    prompt: "BUTTON COLOR_BLUE",
    expectedTokens: [
      "COMPONENT_START",
      "BUTTON",
      "COLOR_BLUE",
      "TEXT_SAVE",
      "COMPONENT_END"
    ],
    expectedHtml: /<button[^>]*>save<\/button>/i
  },
  {
    name: "green submit button",
    prompt: "BUTTON COLOR_GREEN",
    expectedTokens: [
      "COMPONENT_START",
      "BUTTON",
      "COLOR_GREEN",
      "TEXT_SUBMIT",
      "COMPONENT_END"
    ],
    expectedHtml: /<button[^>]*>submit<\/button>/i
  },
  {
    name: "red delete button",
    prompt: "BUTTON COLOR_RED",
    expectedTokens: [
      "COMPONENT_START",
      "BUTTON",
      "COLOR_RED",
      "TEXT_DELETE",
      "COMPONENT_END"
    ],
    expectedHtml: /<button[^>]*>delete<\/button>/i
  },
  {
    name: "yellow cancel button",
    prompt: "BUTTON COLOR_YELLOW",
    expectedTokens: [
      "COMPONENT_START",
      "BUTTON",
      "COLOR_YELLOW",
      "TEXT_CANCEL",
      "COMPONENT_END"
    ],
    expectedHtml: /<button[^>]*>cancel<\/button>/i
  },
  {
    name: "yellow warning alert",
    prompt: "ALERT COLOR_YELLOW",
    expectedTokens: [
      "COMPONENT_START",
      "ALERT",
      "COLOR_YELLOW",
      "TEXT_WARNING",
      "COMPONENT_END"
    ],
    expectedHtml: /role="alert"[^>]*>warning<\/div>/i
  },
  {
    name: "red error alert",
    prompt: "ALERT COLOR_RED",
    expectedTokens: [
      "COMPONENT_START",
      "ALERT",
      "COLOR_RED",
      "TEXT_ERROR",
      "COMPONENT_END"
    ],
    expectedHtml: /role="alert"[^>]*>error<\/div>/i
  },
  {
    name: "ovoid head",
    prompt: "OVOID_HEAD COLOR_SKIN SHADE_SOFT",
    expectedTokens: [
      "COMPONENT_START",
      "OVOID_HEAD",
      "COLOR_SKIN",
      "SHADE_SOFT",
      "COMPONENT_END"
    ],
    expectedHtml: /border-radius:\s*50%/i
  },
  {
    name: "name and email contact form",
    prompt: "CONTACT_FORM FIELD_NAME FIELD_EMAIL",
    expectedTokens: [
      "COMPONENT_START",
      "CONTACT_FORM",
      "FIELD_NAME",
      "FIELD_EMAIL",
      "BUTTON_SUBMIT",
      "COMPONENT_END"
    ],
    expectedHtml: /name="email"/i
  },
  {
    name: "email and password contact form",
    prompt: "CONTACT_FORM FIELD_EMAIL FIELD_PASSWORD",
    expectedTokens: [
      "COMPONENT_START",
      "CONTACT_FORM",
      "FIELD_EMAIL",
      "FIELD_PASSWORD",
      "BUTTON_SUBMIT",
      "COMPONENT_END"
    ],
    expectedHtml: /type="password"/i
  },
  {
    name: "full contact form",
    prompt: "CONTACT_FORM FIELD_NAME FIELD_EMAIL FIELD_PASSWORD",
    expectedTokens: [
      "COMPONENT_START",
      "CONTACT_FORM",
      "FIELD_NAME",
      "FIELD_EMAIL",
      "FIELD_PASSWORD",
      "BUTTON_SUBMIT",
      "COMPONENT_END"
    ],
    expectedHtml: /name="email"[\s\S]*type="password"/i
  },
  {
    name: "move right animation",
    prompt: "ANIMATION MOVE_RIGHT",
    expectedTokens: [
      "COMPONENT_START",
      "ANIMATION",
      "MOVE_RIGHT",
      "DURATION_SHORT",
      "COMPONENT_END"
    ],
    expectedHtml: /@keyframes/i
  },
  {
    name: "grow animation",
    prompt: "ANIMATION GROW",
    expectedTokens: [
      "COMPONENT_START",
      "ANIMATION",
      "GROW",
      "DURATION_SHORT",
      "COMPONENT_END"
    ],
    expectedHtml: /scale/i
  },
  {
    name: "fade in animation",
    prompt: "ANIMATION FADE_IN",
    expectedTokens: [
      "COMPONENT_START",
      "ANIMATION",
      "FADE_IN",
      "DURATION_SHORT",
      "COMPONENT_END"
    ],
    expectedHtml: /opacity/i
  },
  {
    name: "spin animation",
    prompt: "ANIMATION SPIN",
    expectedTokens: [
      "COMPONENT_START",
      "ANIMATION",
      "SPIN",
      "DURATION_SHORT",
      "COMPONENT_END"
    ],
    expectedHtml: /rotate/i
  }
];