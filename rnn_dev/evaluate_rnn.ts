const cases = [
  {
    prompt: "BUTTON COLOR_BLUE",
    expected: [
      "COMPONENT_START",
      "BUTTON",
      "COLOR_BLUE",
      "TEXT_SAVE",
      "COMPONENT_END",
    ],
  },
  {
    prompt: "BUTTON COLOR_GREEN",
    expected: [
      "COMPONENT_START",
      "BUTTON",
      "COLOR_GREEN",
      "TEXT_SUBMIT",
      "COMPONENT_END",
    ],
  },
  {
    prompt: "OVOID_HEAD",
    expected: [
      "COMPONENT_START",
      "OVOID_HEAD",
      "COLOR_SKIN",
      "SHADE_SOFT",
      "COMPONENT_END",
    ],
  },
  {
    prompt: "CONTACT_FORM",
    expected: [
      "COMPONENT_START",
      "CONTACT_FORM",
      "FIELD_NAME",
      "FIELD_EMAIL",
      "BUTTON_SUBMIT",
      "COMPONENT_END",
    ],
  },
  {
    prompt: "ANIMATION MOVE_RIGHT",
    expected: [
      "COMPONENT_START",
      "ANIMATION",
      "MOVE_RIGHT",
      "DURATION_SHORT",
      "COMPONENT_END",
    ],
  },
  {
    prompt: "ANIMATION GROW",
    expected: [
      "COMPONENT_START",
      "ANIMATION",
      "GROW",
      "DURATION_SHORT",
      "COMPONENT_END",
    ],
  },
];