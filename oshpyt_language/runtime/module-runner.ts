import { readFile } from "node:fs/promises";
import { dirname, extname, resolve } from "node:path";

import { Interpreter, RuntimeError } from "./interpreter";
import { lex } from "./lexer";
import { Parser } from "./parser";
import type { ImportNode, Node } from "./parser";

export class ModuleRunner {
  private readonly loaded = new Set<string>();
  private readonly loading = new Set<string>();

  constructor(private readonly interpreter = new Interpreter()) {}

  async run(entryFile: string): Promise<void> {
    await this.load(resolve(entryFile), this.interpreter);
  }

  private async load(
    filePath: string,
    targetInterpreter: Interpreter
  ): Promise<void> {
    if (this.loaded.has(filePath)) {
      return;
    }

    if (this.loading.has(filePath)) {
      throw new RuntimeError(`Circular import detected: ${filePath}`);
    }

    if (extname(filePath) !== ".oshpyt") {
      throw new RuntimeError(
        `Imports must reference .oshpyt files: ${filePath}`
      );
    }

    this.loading.add(filePath);

    try {
      const source = await readFile(filePath, "utf8");
      const nodes = new Parser(lex(source)).parse();

      const imports = nodes.filter(
        (node): node is ImportNode => node.type === "Import"
      );

      for (const imported of imports) {
        await this.loadImport(filePath, imported, targetInterpreter);
      }

      const executableNodes: Node[] = nodes.filter(
        (node) => node.type !== "Import"
      );

      targetInterpreter.register(executableNodes, filePath);
      await targetInterpreter.executeTopLevelCalls(executableNodes);

      this.loaded.add(filePath);
    } finally {
      this.loading.delete(filePath);
    }
  }

  private async loadImport(
    importingFile: string,
    imported: ImportNode,
    targetInterpreter: Interpreter
  ): Promise<void> {
    const importedPath = resolve(dirname(importingFile), imported.path);

    if (imported.alias === undefined) {
      await this.load(importedPath, targetInterpreter);
      return;
    }

    if (this.loading.has(importedPath)) {
      throw new RuntimeError(`Circular import detected: ${importedPath}`);
    }

    if (extname(importedPath) !== ".oshpyt") {
      throw new RuntimeError(
        `Imports must reference .oshpyt files: ${importedPath}`
      );
    }

    const moduleInterpreter = new Interpreter();

    await this.load(importedPath, moduleInterpreter);

    targetInterpreter.registerNamespace(
      imported.alias,
      moduleInterpreter.getFunctions()
    );
  }
}