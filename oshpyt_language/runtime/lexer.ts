export type TokenType =
  | "EOF"
  | "IDENT"
  | "STRING"
  | "NUMBER"
  | "EQ_EQ"
  | "BANG_EQ"
  | "LT_EQ"
  | "GT_EQ"
  | "LT"
  | "GT"
  | "PLUS"
  | "MINUS"
  | "STAR"
  | "SLASH"
  | "LPAREN"
  | "RPAREN"
  | "LBRACE"
  | "RBRACE"
  | "LBRACKET"
  | "RBRACKET"
  | "COMMA";

export type Token = {
  type: TokenType;
  value: string;
  line: number;
  col: number;
};

export class LexerError extends Error {}

export function lex(source: string): Token[] {
  const tokens: Token[] = [];
  let pos = 0;
  let line = 1;
  let col = 1;

  while (pos < source.length) {
    const char = source[pos];

    if (char === " " || char === "\t" || char === "\r") {
      pos += 1;
      col += 1;
      continue;
    }

    if (char === "\n") {
      pos += 1;
      line += 1;
      col = 1;
      continue;
    }

    if (char === "(") {
      tokens.push({ type: "LPAREN", value: char, line, col });
      pos += 1;
      col += 1;
      continue;
    }

    if (char === ")") {
      tokens.push({ type: "RPAREN", value: char, line, col });
      pos += 1;
      col += 1;
      continue;
    }

    if (char === "{") {
      tokens.push({ type: "LBRACE", value: char, line, col });
      pos += 1;
      col += 1;
      continue;
    }

    if (char === "}") {
      tokens.push({ type: "RBRACE", value: char, line, col });
      pos += 1;
      col += 1;
      continue;
    }

    if (char === "[") {
      tokens.push({ type: "LBRACKET", value: char, line, col });
      pos += 1;
      col += 1;
      continue;
    }

    if (char === "]") {
      tokens.push({ type: "RBRACKET", value: char, line, col });
      pos += 1;
      col += 1;
      continue;
    }

    if (char === ",") {
      tokens.push({ type: "COMMA", value: char, line, col });
      pos += 1;
      col += 1;
      continue;
    }

    if (char === "+") {
      tokens.push({ type: "PLUS", value: char, line, col });
      pos += 1;
      col += 1;
      continue;
    }

    if (char === "-") {
      tokens.push({ type: "MINUS", value: char, line, col });
      pos += 1;
      col += 1;
      continue;
    }

    if (char === "*") {
      tokens.push({ type: "STAR", value: char, line, col });
      pos += 1;
      col += 1;
      continue;
    }

    if (char === "/") {
      tokens.push({ type: "SLASH", value: char, line, col });
      pos += 1;
      col += 1;
      continue;
    }

    if (char === "=" && source[pos + 1] === "=") {
      tokens.push({ type: "EQ_EQ", value: "==", line, col });
      pos += 2;
      col += 2;
      continue;
    }

    if (char === "!" && source[pos + 1] === "=") {
      tokens.push({ type: "BANG_EQ", value: "!=", line, col });
      pos += 2;
      col += 2;
      continue;
    }

    if (char === "<") {
      if (source[pos + 1] === "=") {
        tokens.push({ type: "LT_EQ", value: "<=", line, col });
        pos += 2;
        col += 2;
      } else {
        tokens.push({ type: "LT", value: "<", line, col });
        pos += 1;
        col += 1;
      }

      continue;
    }

    if (char === ">") {
      if (source[pos + 1] === "=") {
        tokens.push({ type: "GT_EQ", value: ">=", line, col });
        pos += 2;
        col += 2;
      } else {
        tokens.push({ type: "GT", value: ">", line, col });
        pos += 1;
        col += 1;
      }

      continue;
    }

    if (char === '"') {
      const startLine = line;
      const startCol = col;

      pos += 1;
      col += 1;

      let value = "";

      while (pos < source.length && source[pos] !== '"') {
        if (source[pos] === "\n") {
          throw new LexerError(
            `Unterminated string beginning at ${startLine}:${startCol}.`
          );
        }

        value += source[pos];
        pos += 1;
        col += 1;
      }

      if (pos >= source.length) {
        throw new LexerError(
          `Unterminated string beginning at ${startLine}:${startCol}.`
        );
      }

      pos += 1;
      col += 1;

      tokens.push({
        type: "STRING",
        value,
        line: startLine,
        col: startCol,
      });

      continue;
    }

    if (/[0-9]/.test(char)) {
      const startLine = line;
      const startCol = col;
      let value = "";

      while (pos < source.length && /[0-9]/.test(source[pos])) {
        value += source[pos];
        pos += 1;
        col += 1;
      }

      if (source[pos] === ".") {
        value += source[pos];
        pos += 1;
        col += 1;

        if (pos >= source.length || !/[0-9]/.test(source[pos])) {
          throw new LexerError(
            `Expected digit after decimal point at ${line}:${col}.`
          );
        }

        while (pos < source.length && /[0-9]/.test(source[pos])) {
          value += source[pos];
          pos += 1;
          col += 1;
        }
      }

      tokens.push({
        type: "NUMBER",
        value,
        line: startLine,
        col: startCol,
      });

      continue;
    }

    if (/[A-Za-z_]/.test(char)) {
      const startLine = line;
      const startCol = col;
      let value = "";

      while (pos < source.length && /[A-Za-z0-9_]/.test(source[pos])) {
        value += source[pos];
        pos += 1;
        col += 1;
      }

      tokens.push({
        type: "IDENT",
        value,
        line: startLine,
        col: startCol,
      });

      continue;
    }

    throw new LexerError(`Unsupported character "${char}" at ${line}:${col}.`);
  }

  tokens.push({
    type: "EOF",
    value: "",
    line,
    col,
  });

  return tokens;
}