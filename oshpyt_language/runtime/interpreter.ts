import type {
  BuiltinCallExpr,
  CallNode,
  DefineNode,
  Expr,
  FunctionCallExpr,
  Node,
  StatementNode,
} from "./parser";

export type RuntimeValue = string | number | boolean | RuntimeValue[];

export type InputProvider = (prompt: string) => Promise<string>;

export class RuntimeError extends Error {}

class BreakSignal extends Error {}

class ContinueSignal extends Error {}

class ReturnSignal extends Error {
  constructor(readonly value: RuntimeValue) {
    super();
  }
}

type VariableTypeName = "String" | "Number" | "Boolean" | "List" | "Any";

type VariableSlot = {
  typeName: VariableTypeName;
  value?: RuntimeValue;
  initialized: boolean;
};

class Environment {
  private readonly values = new Map<string, VariableSlot>();

  constructor(private readonly parent?: Environment) {}

  declare(
    name: string,
    typeName: "String" | "Number" | "Boolean" | "List"
  ): void {
    if (this.values.has(name)) {
      throw new RuntimeError(`Variable "${name}" already exists in this scope.`);
    }

    this.values.set(name, {
      typeName,
      initialized: false,
    });
  }

  define(
    name: string,
    value: RuntimeValue,
    typeName: VariableTypeName = "Any"
  ): void {
    if (this.values.has(name)) {
      throw new RuntimeError(`Variable "${name}" already exists in this scope.`);
    }

    this.values.set(name, {
      typeName,
      value: this.cloneValue(value),
      initialized: true,
    });
  }

  set(name: string, value: RuntimeValue): void {
    const slot = this.values.get(name);

    if (slot !== undefined) {
      this.checkType(slot.typeName, value, name);
      slot.value = this.cloneValue(value);
      slot.initialized = true;
      return;
    }

    if (this.parent) {
      this.parent.set(name, value);
      return;
    }

    throw new RuntimeError(`Undeclared variable "${name}".`);
  }

  get(name: string): RuntimeValue {
    const slot = this.values.get(name);

    if (slot !== undefined) {
      if (!slot.initialized || slot.value === undefined) {
        throw new RuntimeError(
          `Variable "${name}" is used before being initialized.`
        );
      }

      return this.cloneValue(slot.value);
    }

    if (this.parent) {
      return this.parent.get(name);
    }

    throw new RuntimeError(`Undefined variable "${name}".`);
  }

  getMutableList(name: string): RuntimeValue[] {
    const slot = this.values.get(name);

    if (slot !== undefined) {
      if (!slot.initialized || slot.value === undefined) {
        throw new RuntimeError(
          `Variable "${name}" is used before being initialized.`
        );
      }

      if (!Array.isArray(slot.value)) {
        throw new RuntimeError(`Variable "${name}" is not a List.`);
      }

      return slot.value;
    }

    if (this.parent) {
      return this.parent.getMutableList(name);
    }

    throw new RuntimeError(`Undefined variable "${name}".`);
  }

  private checkType(
    typeName: VariableTypeName,
    value: RuntimeValue,
    name: string
  ): void {
    if (typeName === "Any") {
      return;
    }

    const valid =
      (typeName === "String" && typeof value === "string") ||
      (typeName === "Number" && typeof value === "number") ||
      (typeName === "Boolean" && typeof value === "boolean") ||
      (typeName === "List" && Array.isArray(value));

    if (!valid) {
      throw new RuntimeError(
        `Type mismatch: variable "${name}" expects ${typeName}, received ${this.getTypeName(value)}.`
      );
    }
  }

  private getTypeName(
    value: RuntimeValue
  ): "String" | "Number" | "Boolean" | "List" {
    if (Array.isArray(value)) {
      return "List";
    }

    if (typeof value === "string") {
      return "String";
    }

    if (typeof value === "number") {
      return "Number";
    }

    return "Boolean";
  }

  private cloneValue(value: RuntimeValue): RuntimeValue {
    if (!Array.isArray(value)) {
      return value;
    }

    return value.map((item) => this.cloneValue(item));
  }
}

export class Interpreter {
  private static readonly MAX_WHILE_ITERATIONS = 10_000;

  private readonly functions = new Map<string, DefineNode>();
  private readonly global = new Environment();

  constructor(
    private readonly inputProvider: InputProvider = async () => {
      throw new RuntimeError(
        "Input is unavailable because no input provider was configured."
      );
    }
  ) {}

  register(nodes: Node[], sourceName = "<source>"): void {
    for (const node of nodes) {
      if (node.type !== "Define") {
        continue;
      }

      if (
        node.name === "Print" ||
        node.name === "Push" ||
        node.name === "Length" ||
        node.name === "Get" ||
        node.name === "Pop" ||
        node.name === "ToString" ||
        node.name === "ToNumber" ||
        node.name === "TypeOf" ||
        node.name === "Input"
      ) {
        throw new RuntimeError(
          `Cannot redefine reserved built-in "${node.name}" at ${sourceName}:${node.location.line}:${node.location.col}.`
        );
      }

      const existing = this.functions.get(node.name);

      if (existing) {
        throw new RuntimeError(
          `Duplicate function "${node.name}" at ${sourceName}:${node.location.line}:${node.location.col}; ` +
            `first defined at ${existing.location.line}:${existing.location.col}.`
        );
      }

      this.functions.set(node.name, node);
    }
  }

  async executeTopLevelCalls(nodes: Node[]): Promise<void> {
    for (const node of nodes) {
      if (node.type !== "Define") {
        await this.executeStatement(node, this.global);
      }
    }
  }

  private async executeBlock(
    statements: StatementNode[],
    parentEnvironment: Environment
  ): Promise<void> {
    const blockEnvironment = new Environment(parentEnvironment);

    for (const statement of statements) {
      await this.executeStatement(statement, blockEnvironment);
    }
  }

  private async executeStatement(
    node: StatementNode,
    environment: Environment
  ): Promise<void> {
    switch (node.type) {
      case "Create":
        environment.declare(node.name, node.typeName);
        return;

      case "Set":
        environment.set(node.name, await this.evaluate(node.value, environment));
        return;

      case "Call":
        await this.executeCall(node, environment);
        return;

      case "Return":
        throw new ReturnSignal(await this.evaluate(node.value, environment));

      case "Break":
        throw new BreakSignal();

      case "Continue":
        throw new ContinueSignal();

      case "When": {
        const condition = await this.evaluate(node.condition, environment);

        if (typeof condition !== "boolean") {
          throw new RuntimeError(
            `Condition in When statement must evaluate to Boolean at ${node.location.line}:${node.location.col}.`
          );
        }

        const branch = condition ? node.thenBranch : node.elseBranch;

        if (branch) {
          await this.executeBlock(branch, environment);
        }

        return;
      }

      case "Repeat": {
        const count = await this.evaluate(node.count, environment);

        if (typeof count !== "number") {
          throw new RuntimeError(
            `Repeat count must evaluate to Number at ${node.location.line}:${node.location.col}.`
          );
        }

        if (!Number.isInteger(count) || count < 0) {
          throw new RuntimeError(
            `Repeat count must be a non-negative integer at ${node.location.line}:${node.location.col}.`
          );
        }

        for (let index = 0; index < count; index += 1) {
          try {
            await this.executeBlock(node.body, environment);
          } catch (error) {
            if (error instanceof BreakSignal) {
              return;
            }

            if (error instanceof ContinueSignal) {
              continue;
            }

            throw error;
          }
        }

        return;
      }

      case "While": {
        let iterations = 0;

        while (true) {
          const condition = await this.evaluate(node.condition, environment);

          if (typeof condition !== "boolean") {
            throw new RuntimeError(
              `Condition in While statement must evaluate to Boolean at ${node.location.line}:${node.location.col}.`
            );
          }

          if (!condition) {
            return;
          }

          if (iterations >= Interpreter.MAX_WHILE_ITERATIONS) {
            throw new RuntimeError(
              `While loop exceeded the maximum of ${Interpreter.MAX_WHILE_ITERATIONS} iterations at ${node.location.line}:${node.location.col}.`
            );
          }

          iterations += 1;

          try {
            await this.executeBlock(node.body, environment);
          } catch (error) {
            if (error instanceof BreakSignal) {
              return;
            }

            if (error instanceof ContinueSignal) {
              continue;
            }

            throw error;
          }
        }
      }
    }
  }

  private async executeCall(
    node: CallNode,
    environment: Environment
  ): Promise<void> {
    if (node.name === "Print") {
      this.expectStatementArguments(node, 1);
      console.log(await this.evaluate(node.args[0], environment));
      return;
    }

    if (node.name === "Push") {
      this.expectStatementArguments(node, 2);

      const listArgument = node.args[0];

      if (listArgument.type !== "Identifier") {
        throw new RuntimeError(
          `Push expects a List variable as its first argument at ${node.location.line}:${node.location.col}.`
        );
      }

      const list = environment.getMutableList(listArgument.name);
      list.push(this.cloneValue(await this.evaluate(node.args[1], environment)));
      return;
    }

    await this.callUserFunction(
      node.name,
      node.args,
      node.location.line,
      node.location.col,
      environment
    );
  }

  private async evaluate(
    expression: Expr,
    environment: Environment
  ): Promise<RuntimeValue> {
    switch (expression.type) {
      case "StringLiteral":
      case "NumberLiteral":
      case "BooleanLiteral":
        return expression.value;

      case "ListLiteral": {
        const values: RuntimeValue[] = [];

        for (const element of expression.elements) {
          values.push(this.cloneValue(await this.evaluate(element, environment)));
        }

        return values;
      }

      case "BuiltinCall":
        return this.evaluateBuiltinCall(expression, environment);

      case "FunctionCall":
        return this.evaluateFunctionCall(expression, environment);

      case "Identifier":
        return environment.get(expression.name);

      case "Unary": {
        const operand = await this.evaluate(expression.operand, environment);

        if (expression.operator === "not") {
          if (typeof operand !== "boolean") {
            throw new RuntimeError(
              `Operator "not" expects Boolean operand at ${expression.location.line}:${expression.location.col}.`
            );
          }

          return !operand;
        }

        if (typeof operand !== "number") {
          throw new RuntimeError(
            `Operator "-" expects Number operand at ${expression.location.line}:${expression.location.col}.`
          );
        }

        return -operand;
      }

      case "Binary":
        return this.evaluateBinary(expression, environment);
    }
  }

  private async evaluateFunctionCall(
    expression: FunctionCallExpr,
    environment: Environment
  ): Promise<RuntimeValue> {
    const value = await this.callUserFunction(
      expression.name,
      expression.args,
      expression.location.line,
      expression.location.col,
      environment
    );

    if (value === undefined) {
      throw new RuntimeError(
        `Function "${expression.name}" did not Return a value at ${expression.location.line}:${expression.location.col}.`
      );
    }

    return value;
  }

  private async callUserFunction(
    name: string,
    args: Expr[],
    line: number,
    col: number,
    callerEnvironment: Environment
  ): Promise<RuntimeValue | undefined> {
    const definition = this.functions.get(name);

    if (!definition) {
      throw new RuntimeError(`Undefined function "${name}" at ${line}:${col}.`);
    }

    if (args.length !== definition.params.length) {
      throw new RuntimeError(
        `Function "${name}" expects ${definition.params.length} argument(s), received ${args.length} at ${line}:${col}.`
      );
    }

    const argumentValues: RuntimeValue[] = [];

    for (const argument of args) {
      argumentValues.push(await this.evaluate(argument, callerEnvironment));
    }

    const local = new Environment(callerEnvironment);

    for (let index = 0; index < definition.params.length; index += 1) {
      local.define(definition.params[index], argumentValues[index]);
    }

    try {
      for (const statement of definition.body) {
        await this.executeStatement(statement, local);
      }
    } catch (error) {
      if (error instanceof ReturnSignal) {
        return this.cloneValue(error.value);
      }

      throw error;
    }

    return undefined;
  }

  private async evaluateBuiltinCall(
    expression: BuiltinCallExpr,
    environment: Environment
  ): Promise<RuntimeValue> {
    if (expression.name === "Length") {
      this.expectExpressionArguments(expression, 1);

      const list = await this.evaluate(expression.args[0], environment);

      if (!Array.isArray(list)) {
        throw new RuntimeError(
          `Length expects a List argument at ${expression.location.line}:${expression.location.col}.`
        );
      }

      return list.length;
    }

    if (expression.name === "Get") {
      this.expectExpressionArguments(expression, 2);

      const list = await this.evaluate(expression.args[0], environment);

      if (!Array.isArray(list)) {
        throw new RuntimeError(
          `Get expects a List as its first argument at ${expression.location.line}:${expression.location.col}.`
        );
      }

      const index = await this.evaluateListIndex(
        expression.args[1],
        environment,
        expression
      );

      if (index >= list.length) {
        throw new RuntimeError(
          `List index ${index} is out of bounds at ${expression.location.line}:${expression.location.col}.`
        );
      }

      return this.cloneValue(list[index]);
    }

    if (expression.name === "Pop") {
      this.expectExpressionArguments(expression, 1);

      const listArgument = expression.args[0];

      if (listArgument.type !== "Identifier") {
        throw new RuntimeError(
          `Pop expects a List variable as its argument at ${expression.location.line}:${expression.location.col}.`
        );
      }

      const list = environment.getMutableList(listArgument.name);

      if (list.length === 0) {
        throw new RuntimeError(
          `Cannot Pop from an empty List at ${expression.location.line}:${expression.location.col}.`
        );
      }

      return this.cloneValue(list.pop() as RuntimeValue);
    }

    if (expression.name === "ToString") {
      this.expectExpressionArguments(expression, 1);
      return this.valueToString(
        await this.evaluate(expression.args[0], environment)
      );
    }

    if (expression.name === "ToNumber") {
      this.expectExpressionArguments(expression, 1);

      const value = await this.evaluate(expression.args[0], environment);

      if (typeof value !== "string") {
        throw new RuntimeError(
          `ToNumber expects a String argument at ${expression.location.line}:${expression.location.col}.`
        );
      }

      const normalized = value.trim();

      if (normalized === "" || !Number.isFinite(Number(normalized))) {
        throw new RuntimeError(
          `ToNumber could not convert "${value}" to Number at ${expression.location.line}:${expression.location.col}.`
        );
      }

      return Number(normalized);
    }

    if (expression.name === "TypeOf") {
      this.expectExpressionArguments(expression, 1);
      return this.getTypeName(
        await this.evaluate(expression.args[0], environment)
      );
    }

    if (expression.name === "Input") {
      this.expectExpressionArguments(expression, 1);

      const prompt = await this.evaluate(expression.args[0], environment);

      if (typeof prompt !== "string") {
        throw new RuntimeError(
          `Input expects a String prompt at ${expression.location.line}:${expression.location.col}.`
        );
      }

      return this.inputProvider(prompt);
    }

    throw new RuntimeError(
      `Unsupported built-in "${expression.name}" at ${expression.location.line}:${expression.location.col}.`
    );
  }

  private expectStatementArguments(node: CallNode, expected: number): void {
    if (node.args.length !== expected) {
      throw new RuntimeError(
        `${node.name} expects ${expected} argument${expected === 1 ? "" : "s"} at ${node.location.line}:${node.location.col}.`
      );
    }
  }

  private expectExpressionArguments(
    expression: BuiltinCallExpr,
    expected: number
  ): void {
    if (expression.args.length !== expected) {
      throw new RuntimeError(
        `${expression.name} expects ${expected} argument${expected === 1 ? "" : "s"} at ${expression.location.line}:${expression.location.col}.`
      );
    }
  }

  private async evaluateListIndex(
    expression: Expr,
    environment: Environment,
    builtin: BuiltinCallExpr
  ): Promise<number> {
    const value = await this.evaluate(expression, environment);

    if (typeof value !== "number" || !Number.isInteger(value) || value < 0) {
      throw new RuntimeError(
        `Get index must be a non-negative integer at ${builtin.location.line}:${builtin.location.col}.`
      );
    }

    return value;
  }

  private async evaluateBinary(
    expression: Extract<Expr, { type: "Binary" }>,
    environment: Environment
  ): Promise<RuntimeValue> {
    const left = await this.evaluate(expression.left, environment);

    if (expression.operator === "and") {
      if (typeof left !== "boolean") {
        throw new RuntimeError(
          `Operator "and" expects Boolean operands at ${expression.location.line}:${expression.location.col}.`
        );
      }

      if (!left) {
        return false;
      }

      const right = await this.evaluate(expression.right, environment);

      if (typeof right !== "boolean") {
        throw new RuntimeError(
          `Operator "and" expects Boolean operands at ${expression.location.line}:${expression.location.col}.`
        );
      }

      return right;
    }

    if (expression.operator === "or") {
      if (typeof left !== "boolean") {
        throw new RuntimeError(
          `Operator "or" expects Boolean operands at ${expression.location.line}:${expression.location.col}.`
        );
      }

      if (left) {
        return true;
      }

      const right = await this.evaluate(expression.right, environment);

      if (typeof right !== "boolean") {
        throw new RuntimeError(
          `Operator "or" expects Boolean operands at ${expression.location.line}:${expression.location.location.col}.`
        );
      }

      return right;
    }

    const right = await this.evaluate(expression.right, environment);

    if (expression.operator === "+") {
      if (typeof left === "number" && typeof right === "number") {
        return left + right;
      }

      if (typeof left === "string" && typeof right === "string") {
        return left + right;
      }

      throw new RuntimeError(
        `Operator "+" expects two Numbers or two Strings at ${expression.location.line}:${expression.location.col}.`
      );
    }

    if (
      expression.operator === "-" ||
      expression.operator === "*" ||
      expression.operator === "/"
    ) {
      if (typeof left !== "number" || typeof right !== "number") {
        throw new RuntimeError(
          `Operator "${expression.operator}" expects Number operands at ${expression.location.line}:${expression.location.col}.`
        );
      }

      switch (expression.operator) {
        case "-":
          return left - right;

        case "*":
          return left * right;

        case "/":
          if (right === 0) {
            throw new RuntimeError(
              `Division by zero at ${expression.location.line}:${expression.location.col}.`
            );
          }

          return left / right;
      }
    }

    if (expression.operator === "==") {
      return this.valuesEqual(left, right);
    }

    if (expression.operator === "!=") {
      return !this.valuesEqual(left, right);
    }

    if (typeof left !== "number" || typeof right !== "number") {
      throw new RuntimeError(
        `Operator "${expression.operator}" expects Number operands at ${expression.location.line}:${expression.location.col}.`
      );
    }

    switch (expression.operator) {
      case "<":
        return left < right;

      case "<=":
        return left <= right;

      case ">":
        return left > right;

      case ">=":
        return left >= right;

      default:
        throw new RuntimeError(
          `Unsupported operator "${expression.operator}".`
        );
    }
  }

  private valueToString(value: RuntimeValue): string {
    if (Array.isArray(value)) {
      return `[${value.map((item) => this.valueToString(item)).join(", ")}]`;
    }

    return String(value);
  }

  private getTypeName(value: RuntimeValue): "String" | "Number" | "Boolean" | "List" {
    if (Array.isArray(value)) {
      return "List";
    }

    if (typeof value === "string") {
      return "String";
    }

    if (typeof value === "number") {
      return "Number";
    }

    return "Boolean";
  }

  private cloneValue(value: RuntimeValue): RuntimeValue {
    if (!Array.isArray(value)) {
      return value;
    }

    return value.map((item) => this.cloneValue(item));
  }

  private valuesEqual(left: RuntimeValue, right: RuntimeValue): boolean {
    if (Array.isArray(left) || Array.isArray(right)) {
      if (!Array.isArray(left) || !Array.isArray(right)) {
        return false;
      }

      if (left.length !== right.length) {
        return false;
      }

      return left.every((value, index) =>
        this.valuesEqual(value, right[index])
      );
    }

    return left === right;
  }
}