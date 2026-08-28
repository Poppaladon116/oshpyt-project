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