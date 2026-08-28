import type { Token, TokenType } from "./lexer";


export class ParserError extends Error {}


export type SourceLocation = {
  line: number;
  col: number;
};


export type BinaryExpr = {
  type: "Binary";
  operator:
    | "=="
    | "!="
    | "<"
    | "<="
    | ">"
    | ">="
    | "and"
    | "or"
    | "+"
    | "-"
    | "*"
    | "/";
  left: Expr;
  right: Expr;
  location: SourceLocation;
};


export type UnaryExpr = {
  type: "Unary";
  operator: "not" | "-";
  operand: Expr;
  location: SourceLocation;
};


export type ListLiteralExpr = {
  type: "ListLiteral";
  elements: Expr[];
  location: SourceLocation;
};


export type BuiltinCallExpr = {
  type: "BuiltinCall";
 name:
  | "Length"
  | "Get"
  | "Pop"
  | "ToString"
  | "ToNumber"
  | "TypeOf"
  | "Input"; 
  args: Expr[];
  location: SourceLocation;
};


export type FunctionCallExpr = {
  type: "FunctionCall";
  name: string;
  namespace?: string;
  args: Expr[];
  location: SourceLocation;
};


export type Expr =
  | {
      type: "Identifier";
      name: string;
      location: SourceLocation;
    }
  | {
      type: "StringLiteral";
      value: string;
      location: SourceLocation;
    }
  | {
      type: "NumberLiteral";
      value: number;
      location: SourceLocation;
    }
  | {
      type: "BooleanLiteral";
      value: boolean;
      location: SourceLocation;
    }
  | ListLiteralExpr
  | BuiltinCallExpr
  | FunctionCallExpr
  | BinaryExpr
  | UnaryExpr;


export type CreateNode = {
  type: "Create";
  name: string;
  typeName: "String" | "Number" | "Boolean" | "List";
  location: SourceLocation;
};


export type SetNode = {
  type: "Set";
  name: string;
  value: Expr;
  location: SourceLocation;
};


export type CallNode = {
  type: "Call";
  name: string;
  namespace?: string;
  args: Expr[];
  location: SourceLocation;
};


export type ReturnNode = {
  type: "Return";
  value: Expr;
  location: SourceLocation;
};


export type WhenNode = {
  type: "When";
  condition: Expr;
  thenBranch: StatementNode[];
  elseBranch?: StatementNode[];
  location: SourceLocation;
};


export type RepeatNode = {
  type: "Repeat";
  count: Expr;
  body: StatementNode[];
  location: SourceLocation;
};


export type WhileNode = {
  type: "While";
  condition: Expr;
  body: StatementNode[];
  location: SourceLocation;
};


export type BreakNode = {
  type: "Break";
  location: SourceLocation;
};


export type ContinueNode = {
  type: "Continue";
  location: SourceLocation;
};


export type StatementNode =
  | CallNode
  | CreateNode
  | SetNode
  | ReturnNode
  | WhenNode
  | RepeatNode
  | WhileNode
  | BreakNode
  | ContinueNode;


export type DefineNode = {
  type: "Define";
  name: string;
  params: string[];
  body: StatementNode[];
  location: SourceLocation;
};


export type ImportNode = {
  type: "Import";
  path: string;
  alias?: string;
  location: SourceLocation;
};


export type Node = ImportNode | DefineNode | StatementNode;


export class Parser {
  private pos = 0;
  private loopDepth = 0;
  private functionDepth = 0;


  constructor(private readonly tokens: Token[]) {}


 parse(): Node[] {
  const nodes: Node[] = [];


  while (!this.check("EOF")) {
    if (this.checkIdentifier("Import")) {
      nodes.push(this.parseImport());
    } else if (this.checkIdentifier("Define")) {
      nodes.push(this.parseDefine());
    } else {
      nodes.push(this.parseStatement());
    }
  }


  return nodes;
}


private parseImport(): ImportNode {
  const start = this.expectIdentifier("Import");
  const path = this.expect("STRING");

  let alias: string | undefined;

  if (this.checkIdentifier("as")) {
    this.advance();
    alias = this.expect("IDENT").value;
  }

  return {
    type: "Import",
    path: path.value,
    alias,
    location: this.location(start),
  };
}


  private parseDefine(): DefineNode {
    const start = this.expectIdentifier("Define");
    const name = this.expect("IDENT").value;


    this.expect("LPAREN");


    const params: string[] = [];


    if (!this.check("RPAREN")) {
      do {
        const parameter = this.expect("IDENT").value;


        if (params.includes(parameter)) {
          const token = this.peek();


          throw new ParserError(
            `Duplicate parameter "${parameter}" at ${token.line}:${token.col}.`
          );
        }


        params.push(parameter);
      } while (this.match("COMMA"));
    }


    this.expect("RPAREN");
    this.expect("LBRACE");


    this.functionDepth += 1;
    const body = this.parseStatementList();
    this.functionDepth -= 1;


    this.expect("RBRACE");


    return {
      type: "Define",
      name,
      params,
      body,
      location: this.location(start),
    };
  }


  private parseStatementList(): StatementNode[] {
    const body: StatementNode[] = [];


    while (!this.check("RBRACE")) {
      if (this.check("EOF")) {
        const token = this.peek();


        throw new ParserError(
          `Unterminated block at ${token.line}:${token.col}.`
        );
      }


      body.push(this.parseStatement());
    }


    return body;
  }


  private parseStatement(): StatementNode {
    if (this.checkIdentifier("Create")) {
      return this.parseCreate();
    }


    if (this.checkIdentifier("Set")) {
      return this.parseSet();
    }


    if (this.checkIdentifier("Call")) {
      return this.parseCall();
    }


    if (this.checkIdentifier("Return")) {
      return this.parseReturn();
    }


    if (this.checkIdentifier("When")) {
      return this.parseWhen();
    }


    if (this.checkIdentifier("Repeat")) {
      return this.parseRepeat();
    }


    if (this.checkIdentifier("While")) {
      return this.parseWhile();
    }


    if (this.checkIdentifier("Break")) {
      return this.parseBreak();
    }


    if (this.checkIdentifier("Continue")) {
      return this.parseContinue();
    }


    const token = this.peek();


    throw new ParserError(
      `Expected Create, Set, Call, Return, When, Repeat, While, Break, or Continue at ${token.line}:${token.col}, found "${token.value}".`
    );
  }


  private parseCreate(): CreateNode {
    const start = this.expectIdentifier("Create");
    const name = this.expect("IDENT").value;


    this.expectIdentifier("as");
    const typeToken = this.expect("IDENT");


    if (
      typeToken.value !== "String" &&
      typeToken.value !== "Number" &&
      typeToken.value !== "Boolean" &&
      typeToken.value !== "List"
    ) {
      throw new ParserError(
        `Expected type "String", "Number", "Boolean", or "List" at ${typeToken.line}:${typeToken.col}, found "${typeToken.value}".`
      );
    }


    return {
      type: "Create",
      name,
      typeName: typeToken.value,
      location: this.location(start),
    };
  }


  private parseSet(): SetNode {
    const start = this.expectIdentifier("Set");
    const name = this.expect("IDENT").value;


    this.expectIdentifier("to");


    return {
      type: "Set",
      name,
      value: this.parseExpression(),
      location: this.location(start),
    };
  }


  private parseCall(): CallNode {
  const start = this.expectIdentifier("Call");
  const firstName = this.expect("IDENT");

  let namespace: string | undefined;
  let name = firstName.value;

  if (this.match("DOT")) {
    namespace = firstName.value;
    name = this.expect("IDENT").value;
  }

  this.expect("LPAREN");
  const args = this.parseArguments();
  this.expect("RPAREN");

  return {
    type: "Call",
    name,
    namespace,
    args,
    location: this.location(start),
  };
}


  private parseReturn(): ReturnNode {
    const start = this.expectIdentifier("Return");


    if (this.functionDepth === 0) {
      throw new ParserError(
        `Return can only be used inside a function at ${start.line}:${start.col}.`
      );
    }


    return {
      type: "Return",
      value: this.parseExpression(),
      location: this.location(start),
    };
  }


  private parseWhen(): WhenNode {
    const start = this.expectIdentifier("When");
    const condition = this.parseExpression();


    this.expectIdentifier("then");
    this.expect("LBRACE");


    const thenBranch = this.parseStatementList();


    this.expect("RBRACE");


    let elseBranch: StatementNode[] | undefined;


    if (this.checkIdentifier("Otherwise")) {
      this.advance();
      this.expect("LBRACE");


      elseBranch = this.parseStatementList();


      this.expect("RBRACE");
    }


    return {
      type: "When",
      condition,
      thenBranch,
      elseBranch,
      location: this.location(start),
    };
  }


  private parseRepeat(): RepeatNode {
    const start = this.expectIdentifier("Repeat");
    const count = this.parseExpression();


    this.expectIdentifier("times");
    this.expect("LBRACE");


    this.loopDepth += 1;
    const body = this.parseStatementList();
    this.loopDepth -= 1;


    this.expect("RBRACE");


    return {
      type: "Repeat",
      count,
      body,
      location: this.location(start),
    };
  }


  private parseWhile(): WhileNode {
    const start = this.expectIdentifier("While");
    const condition = this.parseExpression();


    this.expect("LBRACE");


    this.loopDepth += 1;
    const body = this.parseStatementList();
    this.loopDepth -= 1;


    this.expect("RBRACE");


    return {
      type: "While",
      condition,
      body,
      location: this.location(start),
    };
  }


  private parseBreak(): BreakNode {
    const start = this.expectIdentifier("Break");


    if (this.loopDepth === 0) {
      throw new ParserError(
        `Break can only be used inside a While or Repeat loop at ${start.line}:${start.col}.`
      );
    }


    return {
      type: "Break",
      location: this.location(start),
    };
  }


  private parseContinue(): ContinueNode {
    const start = this.expectIdentifier("Continue");


    if (this.loopDepth === 0) {
      throw new ParserError(
        `Continue can only be used inside a While or Repeat loop at ${start.line}:${start.col}.`
      );
    }


    return {
      type: "Continue",
      location: this.location(start),
    };
  }


  private parseArguments(): Expr[] {
    const args: Expr[] = [];


    if (!this.check("RPAREN")) {
      do {
        args.push(this.parseExpression());
      } while (this.match("COMMA"));
    }


    return args;
  }


  private parseExpression(): Expr {
    return this.parseLogicalOr();
  }


  private parseLogicalOr(): Expr {
    let expr = this.parseLogicalAnd();


    while (this.checkIdentifier("or")) {
      const operator = this.advance();


      expr = {
        type: "Binary",
        operator: "or",
        left: expr,
        right: this.parseLogicalAnd(),
        location: this.location(operator),
      };
    }


    return expr;
  }


  private parseLogicalAnd(): Expr {
    let expr = this.parseLogicalNot();


    while (this.checkIdentifier("and")) {
      const operator = this.advance();


      expr = {
        type: "Binary",
        operator: "and",
        left: expr,
        right: this.parseLogicalNot(),
        location: this.location(operator),
      };
    }


    return expr;
  }


  private parseLogicalNot(): Expr {
    if (this.checkIdentifier("not")) {
      const operator = this.advance();


      return {
        type: "Unary",
        operator: "not",
        operand: this.parseLogicalNot(),
        location: this.location(operator),
      };
    }


    return this.parseEquality();
  }


  private parseEquality(): Expr {
    let expr = this.parseComparison();


    while (this.check("EQ_EQ") || this.check("BANG_EQ")) {
      const operator = this.advance();


      expr = {
        type: "Binary",
        operator: operator.value as "==" | "!=",
        left: expr,
        right: this.parseComparison(),
        location: this.location(operator),
      };
    }


    return expr;
  }


  private parseComparison(): Expr {
    let expr = this.parseAddSub();


    while (
      this.check("LT") ||
      this.check("LT_EQ") ||
      this.check("GT") ||
      this.check("GT_EQ")
    ) {
      const operator = this.advance();


      expr = {
        type: "Binary",
        operator: operator.value as "<" | "<=" | ">" | ">=",
        left: expr,
        right: this.parseAddSub(),
        location: this.location(operator),
      };
    }


    return expr;
  }


  private parseAddSub(): Expr {
    let expr = this.parseMulDiv();


    while (this.check("PLUS") || this.check("MINUS")) {
      const operator = this.advance();


      expr = {
        type: "Binary",
        operator: operator.value as "+" | "-",
        left: expr,
        right: this.parseMulDiv(),
        location: this.location(operator),
      };
    }


    return expr;
  }


  private parseMulDiv(): Expr {
    let expr = this.parseUnaryMinus();


    while (this.check("STAR") || this.check("SLASH")) {
      const operator = this.advance();


      expr = {
        type: "Binary",
        operator: operator.value as "*" | "/",
        left: expr,
        right: this.parseUnaryMinus(),
        location: this.location(operator),
      };
    }


    return expr;
  }


  private parseUnaryMinus(): Expr {
    if (this.check("MINUS")) {
      const operator = this.advance();


      return {
        type: "Unary",
        operator: "-",
        operand: this.parseUnaryMinus(),
        location: this.location(operator),
      };
    }


    return this.parsePrimary();
  }


  private parsePrimary(): Expr {
    const token = this.peek();


    if (token.type === "STRING") {
      this.advance();


      return {
        type: "StringLiteral",
        value: token.value,
        location: this.location(token),
      };
    }


    if (token.type === "NUMBER") {
      this.advance();


      return {
        type: "NumberLiteral",
        value: Number(token.value),
        location: this.location(token),
      };
    }


    if (token.type === "IDENT") {
      this.advance();


      if (token.value === "true" || token.value === "false") {
        return {
          type: "BooleanLiteral",
          value: token.value === "true",
          location: this.location(token),
        };
      }


      if (this.match("DOT")) {
  const functionName = this.expect("IDENT").value;

  if (!this.check("LPAREN")) {
    const next = this.peek();

    throw new ParserError(
      `Expected LPAREN after qualified name at ${next.line}:${next.col}, found ${next.type} "${next.value}".`
    );
  }

  return this.parseExpressionCall(token, token.value, functionName);
}

if (this.check("LPAREN")) {
  return this.parseExpressionCall(token);
}


      return {
        type: "Identifier",
        name: token.value,
        location: this.location(token),
      };
    }


    if (this.match("LBRACKET")) {
      return this.parseListLiteral(token);
    }


    if (this.match("LPAREN")) {
      const expression = this.parseExpression();
      this.expect("RPAREN");
      return expression;
    }


    throw new ParserError(
      `Expected expression at ${token.line}:${token.col}, found "${token.value}".`
    );
  }


  private parseExpressionCall(
  start: Token,
  namespace?: string,
  qualifiedName?: string
): BuiltinCallExpr | FunctionCallExpr {
  this.expect("LPAREN");
  const args = this.parseArguments();
  this.expect("RPAREN");

  const name = qualifiedName ?? start.value;

  if (
    namespace === undefined &&
    (name === "Length" ||
      name === "Get" ||
      name === "Pop" ||
      name === "ToString" ||
      name === "ToNumber" ||
      name === "TypeOf" ||
      name === "Input")
  ) {
    return {
      type: "BuiltinCall",
      name,
      args,
      location: this.location(start),
    };
  }

  return {
    type: "FunctionCall",
    name,
    namespace,
    args,
    location: this.location(start),
  };
}

  private parseListLiteral(start: Token): ListLiteralExpr {
    const elements: Expr[] = [];


    if (!this.check("RBRACKET")) {
      do {
        elements.push(this.parseExpression());
      } while (this.match("COMMA"));
    }


    this.expect("RBRACKET");


    return {
      type: "ListLiteral",
      elements,
      location: this.location(start),
    };
  }


  private peek(): Token {
    return this.tokens[this.pos];
  }


  private advance(): Token {
    const token = this.peek();
    this.pos += 1;
    return token;
  }


  private check(type: TokenType): boolean {
    return this.peek().type === type;
  }


  private checkIdentifier(value: string): boolean {
    const token = this.peek();
    return token.type === "IDENT" && token.value === value;
  }


  private match(type: TokenType): boolean {
    if (!this.check(type)) {
      return false;
    }


    this.advance();
    return true;
  }


  private expect(type: TokenType): Token {
    const token = this.peek();


    if (token.type !== type) {
      throw new ParserError(
        `Expected ${type} at ${token.line}:${token.col}, found ${token.type} "${token.value}".`
      );
    }


    return this.advance();
  }


  private expectIdentifier(value: string): Token {
    const token = this.peek();


    if (token.type !== "IDENT" || token.value !== value) {
      throw new ParserError(
        `Expected "${value}" at ${token.line}:${token.col}, found "${token.value}".`
      );
    }


    return this.advance();
  }


  private location(token: Token): SourceLocation {
    return {
      line: token.line,
      col: token.col,
    };
  }
}