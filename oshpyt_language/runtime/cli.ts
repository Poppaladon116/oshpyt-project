import { readFileSync } from "node:fs";
import { createInterface } from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import { lex } from "./lexer";
import { Interpreter } from "./interpreter";
import { Parser } from "./parser";

type ProgramFile = {
  file: string;
  nodes: ReturnType<Parser["parse"]>;
};

function printUsage(): void {
  process.stderr.write(
    "Usage: npm run oshpyt:run -- <file1.oshpyt> [file2.oshpyt ...]\n"
  );
}

async function main(): Promise<void> {
  const files = process.argv.slice(2);

  if (files.length === 0) {
    printUsage();
    process.exitCode = 1;
    return;
  }

  const readline = createInterface({
    input,
    output,
  });

  try {
    const programs: ProgramFile[] = files.map((file) => {
      const source = readFileSync(file, "utf8");
      const nodes = new Parser(lex(source)).parse();

      return { file, nodes };
    });

    const interpreter = new Interpreter(async (prompt) =>
      readline.question(prompt)
    );

    for (const program of programs) {
      interpreter.register(program.nodes, program.file);
    }

    for (const program of programs) {
      await interpreter.executeTopLevelCalls(program.nodes);
    }
  } catch (error) {
    const message =
      error instanceof Error
        ? error.message
        : "Unknown Oshpyt runtime error.";

    process.stderr.write(`OSHPYT error: ${message}\n`);
    process.exitCode = 1;
  } finally {
    readline.close();
  }
}

void main();