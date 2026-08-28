import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { mkdir, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";

import { ModuleRunner } from "../module-runner";

const fixtureDirectory = join(process.cwd(), ".oshpyt-test-modules");

let logSpy: ReturnType<typeof vi.spyOn>;

beforeEach(async () => {
  await rm(fixtureDirectory, { recursive: true, force: true });
  await mkdir(fixtureDirectory, { recursive: true });

  logSpy = vi.spyOn(console, "log").mockImplementation(() => undefined);
});

afterEach(async () => {
  logSpy.mockRestore();
  await rm(fixtureDirectory, { recursive: true, force: true });
});

test("loads a function from an imported module", async () => {
  const libraryPath = join(fixtureDirectory, "greetings.oshpyt");
  const entryPath = join(fixtureDirectory, "main.oshpyt");

  await writeFile(
    libraryPath,
    `
Define Hello(name) {
  Call Print("Hello, " + name)
}
`.trim()
  );

  await writeFile(
    entryPath,
    `
Import "greetings.oshpyt"
Call Hello("Ada")
`.trim()
  );

  await new ModuleRunner().run(entryPath);

  expect(logSpy).toHaveBeenCalledWith("Hello, Ada");
});

test("calls an aliased module function as a statement", async () => {
  const libraryPath = join(fixtureDirectory, "greetings.oshpyt");
  const entryPath = join(fixtureDirectory, "main.oshpyt");

  await writeFile(
    libraryPath,
    `
Define Hello(name) {
  Call Print("Hello, " + name)
}
`.trim()
  );

  await writeFile(
    entryPath,
    `
Import "greetings.oshpyt" as Greetings
Call Greetings.Hello("Ada")
`.trim()
  );

  await new ModuleRunner().run(entryPath);

  expect(logSpy).toHaveBeenCalledTimes(1);
  expect(logSpy).toHaveBeenCalledWith("Hello, Ada");
});

test("uses an aliased module function in an expression", async () => {
  const libraryPath = join(fixtureDirectory, "math.oshpyt");
  const entryPath = join(fixtureDirectory, "main.oshpyt");

  await writeFile(
    libraryPath,
    `
Define Add(left, right) {
  Return left + right
}
`.trim()
  );

  await writeFile(
    entryPath,
    `
Import "math.oshpyt" as Math

Create answer as Number
Set answer to Math.Add(20, 22)

Call Print(answer)
`.trim()
  );

  await new ModuleRunner().run(entryPath);

  expect(logSpy).toHaveBeenCalledTimes(1);
  expect(logSpy).toHaveBeenCalledWith(42);
});

test("keeps same-named functions separate in namespaces", async () => {
  const numberLibraryPath = join(fixtureDirectory, "number-tools.oshpyt");
  const textLibraryPath = join(fixtureDirectory, "text-tools.oshpyt");
  const entryPath = join(fixtureDirectory, "main.oshpyt");

  await writeFile(
    numberLibraryPath,
    `
Define Describe() {
  Return "number tools"
}
`.trim()
  );

  await writeFile(
    textLibraryPath,
    `
Define Describe() {
  Return "text tools"
}
`.trim()
  );

  await writeFile(
    entryPath,
    `
Import "number-tools.oshpyt" as Numbers
Import "text-tools.oshpyt" as Text

Call Print(Numbers.Describe())
Call Print(Text.Describe())
`.trim()
  );

  await new ModuleRunner().run(entryPath);

  expect(logSpy).toHaveBeenCalledTimes(2);
  expect(logSpy).toHaveBeenNthCalledWith(1, "number tools");
  expect(logSpy).toHaveBeenNthCalledWith(2, "text tools");
});

test("allows an aliased module to call its own functions", async () => {
  const libraryPath = join(fixtureDirectory, "math.oshpyt");
  const entryPath = join(fixtureDirectory, "main.oshpyt");

  await writeFile(
    libraryPath,
    `
Define Double(value) {
  Return value * 2
}

Define Quadruple(value) {
  Return Double(Double(value))
}
`.trim()
  );

  await writeFile(
    entryPath,
    `
Import "math.oshpyt" as Math
Call Print(Math.Quadruple(21))
`.trim()
  );

  await new ModuleRunner().run(entryPath);

  expect(logSpy).toHaveBeenCalledTimes(1);
  expect(logSpy).toHaveBeenCalledWith(84);
});

test("rejects a duplicate import alias", async () => {
  const firstLibraryPath = join(fixtureDirectory, "first.oshpyt");
  const secondLibraryPath = join(fixtureDirectory, "second.oshpyt");
  const entryPath = join(fixtureDirectory, "main.oshpyt");

  await writeFile(
    firstLibraryPath,
    `
Define First() {
  Return 1
}
`.trim()
  );

  await writeFile(
    secondLibraryPath,
    `
Define Second() {
  Return 2
}
`.trim()
  );

  await writeFile(
    entryPath,
    `
Import "first.oshpyt" as Tools
Import "second.oshpyt" as Tools
`.trim()
  );

  await expect(new ModuleRunner().run(entryPath)).rejects.toThrow(
    'Import alias "Tools" already exists.'
  );
});

test("rejects an unknown namespace", async () => {
  const entryPath = join(fixtureDirectory, "main.oshpyt");

  await writeFile(
    entryPath,
    `
Call Missing.Add(1, 2)
`.trim()
  );

  await expect(new ModuleRunner().run(entryPath)).rejects.toThrow(
    'Undefined namespace "Missing"'
  );
});

test("rejects an unknown function in a known namespace", async () => {
  const libraryPath = join(fixtureDirectory, "math.oshpyt");
  const entryPath = join(fixtureDirectory, "main.oshpyt");

  await writeFile(
    libraryPath,
    `
Define Add(left, right) {
  Return left + right
}
`.trim()
  );

  await writeFile(
    entryPath,
    `
Import "math.oshpyt" as Math
Call Math.Multiply(2, 3)
`.trim()
  );

  await expect(new ModuleRunner().run(entryPath)).rejects.toThrow(
    'Undefined function "Math.Multiply"'
  );
});

test("rejects an aliased call with the wrong number of arguments", async () => {
  const libraryPath = join(fixtureDirectory, "math.oshpyt");
  const entryPath = join(fixtureDirectory, "main.oshpyt");

  await writeFile(
    libraryPath,
    `
Define Add(left, right) {
  Return left + right
}
`.trim()
  );

  await writeFile(
    entryPath,
    `
Import "math.oshpyt" as Math
Call Math.Add(1)
`.trim()
  );

  await expect(new ModuleRunner().run(entryPath)).rejects.toThrow(
    'Function "Math.Add" expects 2 argument(s), received 1'
  );
});

test("rejects a circular aliased import", async () => {
  const firstPath = join(fixtureDirectory, "first.oshpyt");
  const secondPath = join(fixtureDirectory, "second.oshpyt");

  await writeFile(
    firstPath,
    `
Import "second.oshpyt" as Second

Define First() {
  Return 1
}
`.trim()
  );

  await writeFile(
    secondPath,
    `
Import "first.oshpyt" as First

Define Second() {
  Return 2
}
`.trim()
  );

  await expect(new ModuleRunner().run(firstPath)).rejects.toThrow(
    "Circular import detected"
  );
});