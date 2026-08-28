import { readFile } from "node:fs/promises";
import { dirname, extname, resolve } from "node:path";

import { lex } from "./lexer";
import { Parser } from "./parser";
import { Interpreter, RuntimeError } from "./interpreter";
import type { ImportNode, Node } from "./parser";

export class ModuleRunner {
  private readonly loaded = new Set<string>();
  private readonly loading = new Set<string>();

  constructor(private readonly interpreter = new Interpreter()) {}

  async run(entryFile: string): Promise<void> {
    await this.load(resolve(entryFile));
  }

  private async load(filePath: string): Promise<void> {
    if (this.loaded.has(filePath)) {
      return;
    }

    if (this.loading.has(filePath)) {
      throw new RuntimeError(`Circular import detected: ${filePath}`);
    }

    if (extname(filePath) !== ".oshpyt") {
      throw new RuntimeError(`Imports must reference .oshpyt files: ${filePath}`);
    }

    this.loading.add(filePath);

    try {
      const source = await readFile(filePath, "utf8");
      const nodes = new Parser(lex(source)).parse();

      const imports = nodes.filter(
        (node): node is ImportNode => node.type === "Import"
      );

      for (const imported of imports) {
        const importedPath = resolve(dirname(filePath), imported.path);
        await this.load(importedPath);
      }

      const executableNodes = nodes.filter((node) => node.type !== "Import");

      this.interpreter.register(executableNodes, filePath);
      await this.interpreter.executeTopLevelCalls(executableNodes);

      this.loaded.add(filePath);
    } finally {
      this.loading.delete(filePath);
    }
  }
}