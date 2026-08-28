import express from "express";
import cors from "cors";
import * as fs from "fs";
import * as path from "path";
import { OshpytTokenizer } from "./tokenizer";
import { Tensor } from "./OshpytTensor";

const app = express();

app.use(cors());
app.use(express.json({ limit: "1mb" }));

const PORT = Number(process.env.PORT ?? 3000);
const MODEL_DIR = process.env.MODEL_DIR ?? process.cwd();

const SIZE = 1024;
const LAYER_COUNT = 18;

const HISTORY_OFFSET = 512;
const CLOCK_INDEX = SIZE - 1;
const MAX_USABLE_TOKEN_ID = 510;

const GOAL_STRENGTH = 15.0;
const HISTORY_STRENGTH = 10.0;
const CLOCK_STRENGTH = 1.0;

const MAX_GENERATION_TOKENS = 180;
const RECENT_TOKEN_BAN_COUNT = 8;

const MIN_OUTPUT_TOKENS: Record<OutputMode, number> = {
  html: 5,
  svg: 5,
  render: 8,
  animation: 12,
};

const FLOATS_PER_LAYER = SIZE * SIZE;
const BYTES_PER_LAYER =
  FLOATS_PER_LAYER * Float32Array.BYTES_PER_ELEMENT;

type OutputMode = "html" | "svg" | "render" | "animation";

const tokenizer = new OshpytTokenizer();
const layers: Tensor[] = [];

let modelReady = false;

const COMPONENT_TEMPLATES: Record<string, string> = {
  oshpyt_trigger_blue_button:
    '<button class="osh-btn osh-blue" style="padding:10px 16px;border:0;border-radius:8px;background:#2563eb;color:#fff;font-weight:700;cursor:pointer;">blue</button>',

  oshpyt_trigger_green_button:
    '<button class="osh-btn osh-green" style="padding:10px 16px;border:0;border-radius:8px;background:#16a34a;color:#fff;font-weight:700;cursor:pointer;">submit</button>',

  oshpyt_trigger_red_button:
    '<button class="osh-btn osh-red" style="padding:10px 16px;border:0;border-radius:8px;background:#dc2626;color:#fff;font-weight:700;cursor:pointer;">delete</button>',

  oshpyt_trigger_dark_button:
    '<button class="osh-btn osh-dark" style="padding:10px 16px;border:0;border-radius:8px;background:#111827;color:#fff;font-weight:700;cursor:pointer;">continue</button>',

  oshpyt_trigger_purple_button:
    '<button class="osh-btn osh-purple" style="padding:10px 16px;border:0;border-radius:8px;background:#7c3aed;color:#fff;font-weight:700;cursor:pointer;">launch</button>',

  oshpyt_trigger_outline_button:
    '<button class="osh-btn osh-outline" style="padding:10px 16px;border:2px solid #2563eb;border-radius:8px;background:transparent;color:#2563eb;font-weight:700;cursor:pointer;">learn more</button>',

  oshpyt_trigger_icon_button:
    '<button class="osh-icon-button" aria-label="Add" style="width:42px;height:42px;border:0;border-radius:50%;background:#111827;color:#fff;font-size:24px;cursor:pointer;">+</button>',

  oshpyt_trigger_glass_card:
    '<div class="osh-card osh-glass" style="width:220px;padding:20px;border:1px solid rgba(255,255,255,.5);border-radius:16px;background:rgba(255,255,255,.28);box-shadow:0 8px 24px rgba(0,0,0,.12);backdrop-filter:blur(8px);color:#111827;">glass card</div>',

  oshpyt_trigger_dark_card:
    '<div class="osh-card osh-dark-card" style="width:220px;padding:20px;border-radius:16px;background:#111827;color:#fff;box-shadow:0 8px 24px rgba(0,0,0,.22);">dark card</div>',

  oshpyt_trigger_profile_card:
    '<div class="osh-card osh-profile" style="width:240px;padding:20px;border:1px solid #e5e7eb;border-radius:16px;background:#fff;box-shadow:0 8px 24px rgba(0,0,0,.08);"><div style="width:48px;height:48px;border-radius:50%;background:#93c5fd;margin-bottom:10px;"></div><strong>profile</strong><div style="color:#6b7280;margin-top:4px;">designer</div></div>',

  oshpyt_trigger_stat_card:
    '<div class="osh-card osh-stat" style="width:220px;padding:20px;border-radius:16px;background:#fff;box-shadow:0 8px 24px rgba(0,0,0,.10);"><div style="color:#6b7280;">active users</div><strong style="display:block;margin-top:6px;font-size:32px;color:#111827;">1,250</strong></div>',

  oshpyt_trigger_neon_alert:
    '<div class="osh-alert osh-neon" style="width:300px;padding:14px;border:1px solid #22d3ee;border-radius:10px;background:#0f172a;color:#67e8f9;box-shadow:0 0 18px rgba(34,211,238,.55);">system error</div>',

  oshpyt_trigger_danger_alert:
    '<div class="osh-alert osh-danger" style="width:300px;padding:14px;border-radius:10px;background:#fef2f2;border:1px solid #fecaca;color:#991b1b;">unauthorized access</div>',

  oshpyt_trigger_success_alert:
    '<div class="osh-alert osh-success" style="width:300px;padding:14px;border-radius:10px;background:#f0fdf4;border:1px solid #bbf7d0;color:#166534;">saved successfully</div>',

  oshpyt_trigger_warning_alert:
    '<div class="osh-alert osh-warning" style="width:300px;padding:14px;border-radius:10px;background:#fffbeb;border:1px solid #fde68a;color:#92400e;">check input</div>',

  oshpyt_trigger_neon_terminal:
    '<div class="osh-terminal" style="width:320px;padding:16px;border-radius:10px;background:#07120a;color:#4ade80;font-family:monospace;box-shadow:0 0 18px rgba(74,222,128,.25);">> system online</div>',

  oshpyt_trigger_loading_state:
    '<div class="osh-loading" style="display:inline-flex;gap:8px;align-items:center;padding:12px 16px;border-radius:10px;background:#f3f4f6;color:#374151;"><span style="width:14px;height:14px;border:3px solid #9ca3af;border-top-color:#2563eb;border-radius:50%;display:inline-block;"></span>loading</div>',

  oshpyt_trigger_pulse_dot:
    '<div class="osh-pulse-dot" style="width:16px;height:16px;border-radius:50%;background:#22c55e;box-shadow:0 0 0 8px rgba(34,197,94,.18);"></div>',

  oshpyt_trigger_progress_bar:
    '<div class="osh-progress" style="width:260px;height:12px;border-radius:999px;background:#e5e7eb;overflow:hidden;"><div class="osh-progress-fill" style="width:68%;height:100%;border-radius:inherit;background:#2563eb;"></div></div>',

  oshpyt_trigger_user_table:
    '<table class="osh-table" style="border-collapse:collapse;min-width:280px;font-family:Arial,sans-serif;"><tr style="background:#111827;color:#fff;"><th style="padding:10px;text-align:left;">id</th><th style="padding:10px;text-align:left;">name</th></tr><tr><td style="padding:10px;border-bottom:1px solid #e5e7eb;">01</td><td style="padding:10px;border-bottom:1px solid #e5e7eb;">Avery</td></tr></table>',

  oshpyt_trigger_data_grid:
    '<div class="osh-grid" style="display:grid;grid-template-columns:repeat(2,90px);gap:8px;"><div style="padding:14px;border-radius:8px;background:#dbeafe;">a1</div><div style="padding:14px;border-radius:8px;background:#bfdbfe;">b1</div><div style="padding:14px;border-radius:8px;background:#bfdbfe;">a2</div><div style="padding:14px;border-radius:8px;background:#dbeafe;">b2</div></div>',

  oshpyt_trigger_contact_form:
    '<form class="osh-form" style="width:280px;padding:18px;border:1px solid #e5e7eb;border-radius:12px;background:#fff;"><label for="osh-name" style="display:block;margin-bottom:6px;color:#374151;">name</label><input id="osh-name" name="name" type="text" style="box-sizing:border-box;width:100%;padding:10px;border:1px solid #d1d5db;border-radius:8px;"><button type="submit" style="margin-top:12px;padding:10px 14px;border:0;border-radius:8px;background:#2563eb;color:#fff;font-weight:700;">submit</button></form>',

  oshpyt_trigger_sidebar_layout:
    '<div class="osh-layout" style="display:flex;width:380px;min-height:180px;border:1px solid #e5e7eb;border-radius:12px;overflow:hidden;"><aside style="width:100px;padding:16px;background:#111827;color:#fff;">nav</aside><main style="flex:1;padding:16px;background:#f9fafb;color:#111827;">content</main></div>',

  oshpyt_trigger_modern_nav:
    '<nav class="osh-nav" style="display:flex;justify-content:space-between;align-items:center;width:380px;padding:14px 18px;border-radius:12px;background:#fff;box-shadow:0 6px 18px rgba(0,0,0,.10);"><strong>logo</strong><span style="color:#6b7280;">menu</span></nav>',

  oshpyt_trigger_avatar_stack:
    '<div class="osh-avatar-stack" style="display:flex;padding-left:10px;"><div style="width:42px;height:42px;border:3px solid #fff;border-radius:50%;background:#f59e0b;"></div><div style="width:42px;height:42px;margin-left:-12px;border:3px solid #fff;border-radius:50%;background:#3b82f6;"></div><div style="width:42px;height:42px;margin-left:-12px;border:3px solid #fff;border-radius:50%;background:#22c55e;"></div></div>',

  oshpyt_trigger_chart_bars:
    '<div class="osh-chart-bars" style="display:flex;align-items:end;gap:10px;width:230px;height:140px;padding:16px;border-radius:12px;background:#f8fafc;"><div style="width:36px;height:52px;border-radius:6px 6px 0 0;background:#60a5fa;"></div><div style="width:36px;height:110px;border-radius:6px 6px 0 0;background:#2563eb;"></div><div style="width:36px;height:78px;border-radius:6px 6px 0 0;background:#93c5fd;"></div></div>',

  oshpyt_trigger_gold_header:
    '<h1 class="osh-gold-header" style="margin:0;font-family:Georgia,serif;font-size:42px;color:#b8860b;text-shadow:0 2px 8px rgba(184,134,11,.25);">premier</h1>',

  oshpyt_trigger_typography_hero:
    '<div style="max-width:480px;padding:20px;background:#0f172a;border-radius:16px;"><h1 class="osh-hero-text" style="margin:0;color:#fff;font-size:42px;line-height:1.05;font-family:Arial,sans-serif;">elegant design</h1></div>',

  oshpyt_trigger_floating_action:
    '<button class="osh-floating-action" aria-label="Add" style="width:56px;height:56px;border:0;border-radius:50%;background:#2563eb;color:#fff;font-size:30px;box-shadow:0 8px 18px rgba(37,99,235,.35);cursor:pointer;">+</button>',

  oshpyt_trigger_star_svg:
    '<svg class="osh-star" width="120" height="120" viewBox="0 0 24 24" fill="#fbbf24" stroke="#b45309" stroke-width="1.2"><polygon points="12,2 15,8 22,9 17,14 18,21 12,17 6,21 7,14 2,9 9,8"></polygon></svg>',

  oshpyt_trigger_ovoid_head:
    '<div class="osh-ovoid-head" style="width:160px;height:210px;display:block;border-radius:50% 50% 46% 46% / 43% 43% 57% 57%;background:linear-gradient(135deg,#f3c59a 0%,#b96e48 100%);box-shadow:inset 18px 12px 28px rgba(255,255,255,.25),inset -18px -16px 28px rgba(0,0,0,.25),0 12px 24px rgba(0,0,0,.22);"></div>',

  oshpyt_trigger_3d_sphere:
    '<div class="osh-sphere" style="width:160px;height:160px;border-radius:50%;background:radial-gradient(circle at 32% 28%,#fff 0%,#93c5fd 20%,#2563eb 55%,#172554 100%);box-shadow:inset -18px -20px 30px rgba(0,0,0,.35),0 16px 26px rgba(0,0,0,.22);"></div>',

  oshpyt_trigger_draw_square:
    '<div class="osh-square" style="width:150px;height:150px;border:5px solid #2563eb;background:#dbeafe;box-shadow:10px 10px 0 #93c5fd;"></div>',

  oshpyt_trigger_dither_art:
    '<pre class="osh-dither-art" style="display:inline-block;margin:0;padding:14px;border-radius:10px;background:#111827;color:#f9fafb;font:14px/1.1 monospace;">░░▒▒▓▓▒▒░░\n░▒▓████▓▒░\n▒▓██░░██▓▒\n░▒▓████▓▒░\n░░▒▒▓▓▒▒░░</pre>',

  oshpyt_trigger_shade_gradient:
    '<div class="osh-shade-gradient" style="width:300px;height:130px;border-radius:14px;background:linear-gradient(135deg,#111827 0%,#4f46e5 48%,#a78bfa 100%);box-shadow:0 10px 24px rgba(79,70,229,.28);"></div>',

  oshpyt_trigger_anim_pulse:
  '<style>@keyframes oshPulse{0%,100%{transform:scale(1);box-shadow:0 0 0 0 rgba(37,99,235,.55)}50%{transform:scale(1.16);box-shadow:0 0 0 24px rgba(37,99,235,0)}}</style><div style="width:80px;height:80px;border-radius:50%;background:#2563eb;animation:oshPulse 1.2s ease-in-out infinite;"></div>',

oshpyt_trigger_anim_left:
  '<style>@keyframes oshLeft{0%,100%{transform:translateX(80px)}50%{transform:translateX(0)}}</style><div style="width:180px;padding:16px;box-sizing:border-box;border-radius:10px;background:#2563eb;color:#fff;animation:oshLeft 1.2s ease-in-out infinite;">moves left</div>',

oshpyt_trigger_anim_right:
  '<style>@keyframes oshRight{0%,100%{transform:translateX(0)}50%{transform:translateX(80px)}}</style><div style="width:180px;padding:16px;box-sizing:border-box;border-radius:10px;background:#2563eb;color:#fff;animation:oshRight 1.2s ease-in-out infinite;">moves right</div>',

oshpyt_trigger_anim_down:
  '<style>@keyframes oshDown{0%,100%{transform:translateY(0)}50%{transform:translateY(55px)}}</style><div style="width:180px;padding:16px;box-sizing:border-box;border-radius:10px;background:#2563eb;color:#fff;animation:oshDown 1.2s ease-in-out infinite;">moves down</div>',

oshpyt_trigger_anim_grow:
  '<style>@keyframes oshGrow{0%,100%{transform:scale(1)}50%{transform:scale(1.28)}}</style><div style="width:130px;padding:22px;box-sizing:border-box;border-radius:10px;background:#16a34a;color:#fff;animation:oshGrow 1.2s ease-in-out infinite;transform-origin:center;">grows</div>',

oshpyt_trigger_anim_slide:
  '<style>@keyframes oshSlide{0%{transform:translateX(-90px);opacity:.2}45%,75%{transform:translateX(30px);opacity:1}100%{transform:translateX(220px);opacity:.2}}</style><div style="width:200px;padding:16px;box-sizing:border-box;border-radius:10px;background:#7c3aed;color:#fff;animation:oshSlide 1.8s ease-in-out infinite;">slides</div>',

oshpyt_trigger_anim_rotate:
  '<style>@keyframes oshRotate{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}</style><div style="width:110px;height:110px;padding-top:42px;box-sizing:border-box;border-radius:16px;background:#dc2626;color:#fff;text-align:center;animation:oshRotate 1.8s linear infinite;">rotate</div>',
};

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
      `Allowed IDs are 1-${MAX_USABLE_TOKEN_ID}.`
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
    trigger.includes("_ovoid_head") ||
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

function getTokenId(token: string): number {
  const tokenId = tokenizer.vocab.get(token);

  if (tokenId == null) {
    throw new Error(`Required token "${token}" is absent from vocab.json.`);
  }

  validateTokenId(tokenId, `Token "${token}"`);

  return tokenId;
}

function loadModel(): void {
  tokenizer.loadVocab();

  if (tokenizer.vocab.size > MAX_USABLE_TOKEN_ID + 1) {
    throw new Error(
      `Vocabulary has ${tokenizer.vocab.size} entries; maximum is ` +
      `${MAX_USABLE_TOKEN_ID + 1}.`
    );
  }

  for (const requiredToken of [
    "html",
    "svg",
    "render",
    "frame_1",
    "component_end"
  ]) {
    getTokenId(requiredToken);
  }

  const metadataPath = path.join(
    MODEL_DIR,
    "model_meta.json"
  );

  if (fs.existsSync(metadataPath)) {
    const metadata = JSON.parse(
      fs.readFileSync(metadataPath, "utf-8")
    );

    if (
      metadata.size !== SIZE ||
      metadata.layerCount !== LAYER_COUNT
    ) {
      throw new Error(
        "model_meta.json does not match server configuration."
      );
    }

    if (metadata.vocabularySize !== tokenizer.vocab.size) {
      throw new Error(
        `Vocabulary mismatch: model expects ${metadata.vocabularySize}; ` +
        `vocab.json has ${tokenizer.vocab.size}.`
      );
    }
  }

  console.log(
    `OSHPYT: Loading ${LAYER_COUNT} model layers from ${MODEL_DIR}...`
  );

  for (
    let layerNumber = 1;
    layerNumber <= LAYER_COUNT;
    layerNumber++
  ) {
    const filePath = path.join(
      MODEL_DIR,
      `model_l${layerNumber}.bin`
    );

    if (!fs.existsSync(filePath)) {
      throw new Error(`Missing model file: ${filePath}`);
    }

    const file = fs.readFileSync(filePath);

    if (file.byteLength !== BYTES_PER_LAYER) {
      throw new Error(
        `${filePath} has ${file.byteLength} bytes; expected ` +
        `${BYTES_PER_LAYER}.`
      );
    }

    const values = new Float32Array(
      file.buffer,
      file.byteOffset,
      FLOATS_PER_LAYER
    );

    layers.push(
      Tensor.fromArray(values, SIZE, SIZE)
    );
  }

  modelReady = true;

  console.log(
    `OSHPYT: Model ready. ${layers.length} layers loaded; ` +
    `vocabulary size ${tokenizer.vocab.size}.`
  );
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

function getTriggerFromText(text: string): number | undefined {
  const normalized = text.toLowerCase().trim();

  if (normalized.startsWith("oshpyt_trigger_")) {
    const directId = tokenizer.vocab.get(normalized);

    if (
      directId != null &&
      directId >= 1 &&
      directId <= MAX_USABLE_TOKEN_ID
    ) {
      return directId;
    }
  }

  let bestMatch:
    | { tokenId: number; phrase: string }
    | undefined;

  for (const [token, tokenId] of tokenizer.vocab.entries()) {
    if (
      !token.startsWith("oshpyt_trigger_") ||
      tokenId < 1 ||
      tokenId > MAX_USABLE_TOKEN_ID
    ) {
      continue;
    }

    const phrase = token
      .replace(/^oshpyt_trigger_/, "")
      .replace(/_/g, " ")
      .trim();

    if (
      phrase.length > 0 &&
      normalized.includes(phrase) &&
      (!bestMatch || phrase.length > bestMatch.phrase.length)
    ) {
      bestMatch = { tokenId, phrase };
    }
  }

  return bestMatch?.tokenId;
}

function getPermanentForbiddenIds(
  mode: OutputMode
): Set<number> {
  const forbidden = new Set<number>([0]);

  for (const [token, tokenId] of tokenizer.vocab.entries()) {
    if (token.startsWith("oshpyt_trigger_")) {
      forbidden.add(tokenId);
    }
  }

  /*
   * UI and SVG output must not begin generating binary map data.
   * Binary map and animation modes explicitly permit 0 and 1.
   */
  if (mode === "html" || mode === "svg") {
    const zeroId = tokenizer.vocab.get("0");
    const oneId = tokenizer.vocab.get("1");

    if (zeroId != null) forbidden.add(zeroId);
    if (oneId != null) forbidden.add(oneId);
  }

  return forbidden;
}

function chooseBestToken(
  probabilities: Float32Array,
  forbidden: ReadonlySet<number>
): number | undefined {
  const upperBound = Math.min(
    probabilities.length,
    tokenizer.vocab.size,
    MAX_USABLE_TOKEN_ID + 1
  );

  let bestId: number | undefined;
  let bestScore = -Infinity;

  for (let tokenId = 1; tokenId < upperBound; tokenId++) {
    if (forbidden.has(tokenId)) {
      continue;
    }

    const score = probabilities[tokenId];

    if (Number.isFinite(score) && score > bestScore) {
      bestScore = score;
      bestId = tokenId;
    }
  }

  return bestId;
}

function getTopCandidates(
  probabilities: Float32Array,
  forbidden: ReadonlySet<number>,
  count = 5
): Array<{
  token: string;
  tokenId: number;
  probability: number;
}> {
  const values: Array<{
    token: string;
    tokenId: number;
    probability: number;
  }> = [];

  const upperBound = Math.min(
    probabilities.length,
    tokenizer.vocab.size,
    MAX_USABLE_TOKEN_ID + 1
  );

  for (let tokenId = 1; tokenId < upperBound; tokenId++) {
    if (forbidden.has(tokenId)) {
      continue;
    }

    const probability = probabilities[tokenId];

    if (!Number.isFinite(probability)) {
      continue;
    }

    values.push({
      token: tokenizer.decode([tokenId]),
      tokenId,
      probability,
    });
  }

  values.sort((left, right) =>
    right.probability - left.probability
  );

  return values.slice(0, count);
}

function normalizeOutput(tokens: string[]): string {
  let output = tokens.join(" ");

  output = output.replace(/<\s+\/\s+/g, "</");
  output = output.replace(/<\s+/g, "<");
  output = output.replace(/\s+>/g, ">");
  output = output.replace(/\s+"/g, '"');
  output = output.replace(/"\s+/g, '" ');
  output = output.replace(/\s+;/g, ";");
  output = output.replace(/\s+:/g, ":");
  output = output.replace(/:\s+/g, ": ");
  output = output.replace(/\s+/g, " ");

  return output.trim();
}

function generate(
  goalToken: number
): {
  reply: string;
  mode: OutputMode;
  stoppedAt: string;
  tokensGenerated: number;
} {
  validateTokenId(goalToken, "Goal token");

  const trigger = tokenizer.decode([goalToken])
    .toLowerCase()
    .trim();

  const mode = getModeFromTrigger(trigger);
  const modeToken = getModeToken(mode);
  const modeId = getTokenId(modeToken);
  const endTokenId = getTokenId("component_end");

  /*
   * The first model prediction was trained as:
   * trigger -> mode.
   *
   * We add the known mode deterministically rather than asking the model
   * to rediscover it. The next prediction then begins actual component data.
   */
  const history = [goalToken, modeId];
  const outputTokens = [modeToken];

  let stoppedAt = "max_tokens";

  for (
    let outputStep = 1;
    outputStep <= MAX_GENERATION_TOKENS;
    outputStep++
  ) {
    const allocated: Tensor[] = [];

    try {
      const previousToken = history[history.length - 1];

      const input = Tensor.allocate(1, SIZE);
      allocated.push(input);

      input.update(
        createInput(
          goalToken,
          previousToken,
          outputStep + 1,
          MAX_GENERATION_TOKENS + 1
        )
      );

      let activation = input;

      for (
        let layerIndex = 0;
        layerIndex < LAYER_COUNT;
        layerIndex++
      ) {
        const next = Tensor.allocate(1, SIZE);
        allocated.push(next);

        activation.matmulTo(layers[layerIndex], next);

        if (layerIndex < LAYER_COUNT - 1) {
          next.relu();
        }

        activation = next;
      }

      activation.softmax();

      const probabilities = activation.download();

      const forbidden = getPermanentForbiddenIds(mode);

      for (const recentId of history.slice(-RECENT_TOKEN_BAN_COUNT)) {
        forbidden.add(recentId);
      }

      /*
       * Do not terminate before enough content exists for the output mode.
       */
      if (outputTokens.length < MIN_OUTPUT_TOKENS[mode]) {
        forbidden.add(endTokenId);
      } else {
        forbidden.delete(endTokenId);
      }

      /*
       * The mode is already supplied by the server; do not emit it again.
       */
      forbidden.add(modeId);

      const top = getTopCandidates(probabilities, forbidden);

      const nextId = chooseBestToken(probabilities, forbidden);

      if (nextId == null) {
        stoppedAt = "no_valid_token";
        break;
      }

      const nextToken = tokenizer.decode([nextId])
        .toLowerCase()
        .trim();

      console.log(
        `Step ${outputStep}: ${nextToken} ` +
        `(${probabilities[nextId].toFixed(6)})`
      );

      if (top.length > 0) {
        console.table(top);
      }

      if (nextId === endTokenId || nextToken === "component_end") {
        stoppedAt = "component_end";
        break;
      }

      history.push(nextId);
      outputTokens.push(nextToken);
    } finally {
      destroyAll(allocated);
    }
  }

  return {
    reply: normalizeOutput(outputTokens),
    mode,
    stoppedAt,
    tokensGenerated: outputTokens.length,
  };
}

app.get("/health", (_request, response) => {
  return response.status(modelReady ? 200 : 503).json({
    ready: modelReady,
    layersLoaded: layers.length,
    expectedLayers: LAYER_COUNT,
    vocabularySize: tokenizer.vocab.size,
    width: SIZE,
  });
});

app.get("/triggers", (_request, response) => {
  const triggers = Array.from(tokenizer.vocab.entries())
    .filter(
      ([token, tokenId]) =>
        token.startsWith("oshpyt_trigger_") &&
        tokenId >= 1 &&
        tokenId <= MAX_USABLE_TOKEN_ID
    )
    .map(([trigger, tokenId]) => ({
      trigger,
      tokenId,
      phrase: trigger
        .replace(/^oshpyt_trigger_/, "")
        .replace(/_/g, " "),
      mode: getModeFromTrigger(trigger),
    }))
    .sort((left, right) =>
      left.trigger.localeCompare(right.trigger)
    );

  return response.json({
    count: triggers.length,
    triggers,
  });
});

app.post("/chat", (request, response) => {
  const text = request.body?.text;

  if (typeof text !== "string" || text.trim().length === 0) {
    return response.status(400).json({
      error: 'Send JSON: { "text": "blue button" }',
    });
  }

  if (!modelReady) {
    return response.status(503).json({
      error: "Model is not ready.",
    });
  }

  const goalToken = getTriggerFromText(text);

  if (goalToken == null) {
    return response.status(404).json({
      error: "Unknown intent.",
      hint: "Use GET /triggers to list supported intents.",
    });
  }
  const trigger = tokenizer.decode([goalToken])
    .toLowerCase()
    .trim();

  const template = COMPONENT_TEMPLATES[trigger];

  if (template != null) {
  console.log(
    `Template response: ${trigger} (${getModeFromTrigger(trigger)})`
  );

  return response.json({
    reply: template,
    mode: getModeFromTrigger(trigger),
    stoppedAt: "template",
    tokensGenerated: 0,
    trigger,
  });
}
  try {
    console.log("\n--- OSHPYT GENERATION START ---");
    console.log(`Input: ${text}`);
    console.log(
      `Trigger: ${tokenizer.decode([goalToken])} (${goalToken})`
    );

    const result = generate(goalToken);

    console.log(`Stopped at: ${result.stoppedAt}`);
    console.log("--- OSHPYT GENERATION END ---\n");

    return response.json({
      reply: result.reply,
      mode: result.mode,
      stoppedAt: result.stoppedAt,
      tokensGenerated: result.tokensGenerated,
      trigger: tokenizer.decode([goalToken]),
    });
  } catch (error) {
    const message =
      error instanceof Error ? error.message : String(error);

    console.error("Generation error:", message);

    return response.status(500).json({
      error: "Generation failed.",
      detail: message,
    });
  }
});

function shutdown(): void {
  console.log("\nOSHPYT: Releasing native tensor memory.");
  destroyAll(layers);
  process.exit(0);
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);

try {
  loadModel();

  app.listen(PORT, () => {
    console.log(
      `OSHPYT API listening on http://localhost:${PORT}`
    );
  });
} catch (error) {
  const message =
    error instanceof Error ? error.message : String(error);

  console.error("FATAL SERVER ERROR:", message);
  destroyAll(layers);
  process.exitCode = 1;
}