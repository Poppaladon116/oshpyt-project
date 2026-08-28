import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { lex, LexerError } from "../lexer";
import { Parser } from "../parser";
import { Interpreter, RuntimeError } from "../interpreter";

function parse(source: string) {
  return new Parser(lex(source)).parse();
}

let interpreter: Interpreter;
let logSpy: ReturnType<typeof vi.spyOn>;

beforeEach(() => {
  interpreter = new Interpreter();
  logSpy = vi.spyOn(console, "log").mockImplementation(() => undefined);
});

afterEach(() => {
  logSpy.mockRestore();
});

test("defines and calls a function with a string argument", async () => {
  const program = parse(`
    Define Hello(name) {
      Call Print(name)
    }

    Call Hello("world")
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledWith("world");
});

test("allows a Call before its Define because definitions register first", async () => {
  const program = parse(`
    Call Hello("forward reference")

    Define Hello(name) {
      Call Print(name)
    }
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledWith("forward reference");
});

test("shares definitions between separate source programs", async () => {
  const definitions = parse(`
    Define Hello(name) {
      Call Print(name)
    }
  `);

  const calls = parse(`
    Call Hello("second file")
  `);

  interpreter.register(definitions, "first.oshpyt");
  interpreter.register(calls, "second.oshpyt");

  await interpreter.executeTopLevelCalls(definitions);
  await interpreter.executeTopLevelCalls(calls);

  expect(logSpy).toHaveBeenCalledWith("second file");
});

test("rejects duplicate function definitions", async () => {
  const first = parse(`
    Define Hello(name) {
      Call Print(name)
    }
  `);

  const second = parse(`
    Define Hello(value) {
      Call Print(value)
    }
  `);

  interpreter.register(first, "first.oshpyt");

  expect(() => interpreter.register(second, "second.oshpyt")).toThrow(
    /Duplicate function "Hello"/
  );
});

test("rejects redefining Print", async () => {
  const program = parse(`
    Define Print(value) {
      Call Print(value)
    }
  `);

  expect(() => interpreter.register(program, "bad.oshpyt")).toThrow(
    /Cannot redefine reserved built-in "Print"/
  );
});

test("rejects an undefined function", async () => {
  const program = parse(`
    Call MissingFunction("hello")
  `);

  interpreter.register(program);

  await expect(interpreter.executeTopLevelCalls(program)).rejects.toThrow(
    /Undefined function "MissingFunction"/
  );
});

test("rejects the wrong number of arguments", async () => {
  const program = parse(`
    Define Hello(name) {
      Call Print(name)
    }

    Call Hello()
  `);

  interpreter.register(program);

  await expect(interpreter.executeTopLevelCalls(program)).rejects.toThrow(
    /expects 1 argument/
  );
});

test("rejects an unterminated string", async () => {
  expect(() => lex(`Call Print("unfinished)`)).toThrow(LexerError);
});

test("rejects an unsupported character", async () => {
  expect(() => lex("Call Print(@)")).toThrow(LexerError);
});

test("global string variable", async () => {
  const program = parse(`
    Create greeting as String
    Set greeting to "Hello World"
    Call Print(greeting)
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledWith("Hello World");
});

test("global number variable", async () => {
  const program = parse(`
    Create count as Number
    Set count to 42
    Call Print(count)
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledWith(42);
});

test("local function variable", async () => {
  const program = parse(`
    Define Test() {
      Create localValue as String
      Set localValue to "inside function"
      Call Print(localValue)
    }

    Call Test()
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledWith("inside function");
});

test("local variable shadows a global variable", async () => {
  const program = parse(`
    Create message as String
    Set message to "global"

    Define Shadow() {
      Create message as String
      Set message to "local"
      Call Print(message)
    }

    Call Shadow()
    Call Print(message)
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenNthCalledWith(1, "local");
  expect(logSpy).toHaveBeenNthCalledWith(2, "global");
});

test("rejects duplicate Create in one scope", async () => {
  const program = parse(`
    Create duplicate as String
    Create duplicate as String
  `);

  interpreter.register(program);

  await expect(interpreter.executeTopLevelCalls(program)).rejects.toThrow(
    /Variable "duplicate" already exists in this scope/
  );
});

test("rejects Set for an undeclared variable", async () => {
  const program = parse(`
    Set undeclared to "value"
  `);

  interpreter.register(program);

  await expect(interpreter.executeTopLevelCalls(program)).rejects.toThrow(
    /Undeclared variable "undeclared"/
  );
});

test("rejects reading an uninitialized variable", async () => {
  const program = parse(`
    Create empty as String
    Call Print(empty)
  `);

  interpreter.register(program);

  await expect(interpreter.executeTopLevelCalls(program)).rejects.toThrow(
    /Variable "empty" is used before being initialized/
  );
});

test("rejects a type mismatch on Set", async () => {
  const program = parse(`
    Create num as Number
    Set num to "not a number"
  `);

  interpreter.register(program);

  await expect(interpreter.executeTopLevelCalls(program)).rejects.toThrow(
    /Type mismatch: variable "num" expects Number, received String/
  );
});

test("nested function calls can read a caller parameter", async () => {
  const program = parse(`
    Define Inner() {
      Call Print(name)
    }

    Define Outer(name) {
      Call Inner()
    }

    Call Outer("nested value")
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledWith("nested value");
});

test("supports Boolean variables and logical expressions", async () => {
  const program = parse(`
    Create enabled as Boolean
    Set enabled to true and not false
    Call Print(enabled)

    Create allowed as Boolean
    Set allowed to false or true
    Call Print(allowed)
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenNthCalledWith(1, true);
  expect(logSpy).toHaveBeenNthCalledWith(2, true);
});

test("supports equality and numeric comparisons", async () => {
  const program = parse(`
    Create equalStrings as Boolean
    Set equalStrings to "alpha" == "alpha"
    Call Print(equalStrings)

    Create differentNumbers as Boolean
    Set differentNumbers to 10 != 20
    Call Print(differentNumbers)

    Create atLeast as Boolean
    Set atLeast to 20 >= 10
    Call Print(atLeast)

    Create below as Boolean
    Set below to 4 < 9
    Call Print(below)
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenNthCalledWith(1, true);
  expect(logSpy).toHaveBeenNthCalledWith(2, true);
  expect(logSpy).toHaveBeenNthCalledWith(3, true);
  expect(logSpy).toHaveBeenNthCalledWith(4, true);
});

test("executes the true When branch", async () => {
  const program = parse(`
    Create score as Number
    Set score to 85

    When score >= 80 then {
      Call Print("pass")
    } Otherwise {
      Call Print("fail")
    }
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledTimes(1);
  expect(logSpy).toHaveBeenCalledWith("pass");
});

test("executes the Otherwise branch", async () => {
  const program = parse(`
    Create score as Number
    Set score to 45

    When score >= 80 then {
      Call Print("pass")
    } Otherwise {
      Call Print("fail")
    }
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledTimes(1);
  expect(logSpy).toHaveBeenCalledWith("fail");
});

test("supports nested When statements", async () => {
  const program = parse(`
    Create score as Number
    Set score to 90

    When score >= 80 then {
      Create honors as Boolean
      Set honors to score >= 90

      When honors then {
        Call Print("pass honors")
      } Otherwise {
        Call Print("pass")
      }
    } Otherwise {
      Call Print("fail")
    }
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledTimes(1);
  expect(logSpy).toHaveBeenCalledWith("pass honors");
});

test("supports When statements in functions", async () => {
  const program = parse(`
    Define Describe(value) {
      When value > 0 then {
        Call Print("positive")
      } Otherwise {
        Call Print("not positive")
      }
    }

    Call Describe(4)
    Call Describe(0)
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenNthCalledWith(1, "positive");
  expect(logSpy).toHaveBeenNthCalledWith(2, "not positive");
});

test("rejects a non-Boolean When condition", async () => {
  const program = parse(`
    When 123 then {
      Call Print("invalid")
    }
  `);

  interpreter.register(program);

  await expect(interpreter.executeTopLevelCalls(program)).rejects.toThrow(
    /Condition in When statement must evaluate to Boolean/
  );
});

test("rejects incompatible Boolean assignment", async () => {
  const program = parse(`
    Create enabled as Boolean
    Set enabled to "yes"
  `);

  interpreter.register(program);

  await expect(interpreter.executeTopLevelCalls(program)).rejects.toThrow(
    /Type mismatch: variable "enabled" expects Boolean, received String/
  );
});

test("rejects comparison of non-number operands", async () => {
  const program = parse(`
    Create result as Boolean
    Set result to "a" < "b"
  `);

  interpreter.register(program);

  await expect(interpreter.executeTopLevelCalls(program)).rejects.toThrow(
    /Operator "<" expects Number operands/
  );
});

test("rejects not with a non-Boolean operand", async () => {
  const program = parse(`
    Create result as Boolean
    Set result to not 123
  `);

  interpreter.register(program);

  await expect(interpreter.executeTopLevelCalls(program)).rejects.toThrow(
    /Operator "not" expects Boolean operand/
  );
});

test("rejects unterminated When blocks", async () => {
  expect(() =>
    parse(`
      When true then {
        Call Print("missing brace")
    `)
  ).toThrow(/Unterminated block/);
});

test("supports addition, subtraction, multiplication, and division", async () => {
  const program = parse(`
    Create sum as Number
    Set sum to 12 + 30
    Call Print(sum)

    Create difference as Number
    Set difference to 50 - 8
    Call Print(difference)

    Create product as Number
    Set product to 6 * 7
    Call Print(product)

    Create quotient as Number
    Set quotient to 84 / 2
    Call Print(quotient)
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenNthCalledWith(1, 42);
  expect(logSpy).toHaveBeenNthCalledWith(2, 42);
  expect(logSpy).toHaveBeenNthCalledWith(3, 42);
  expect(logSpy).toHaveBeenNthCalledWith(4, 42);
});

test("supports unary Number negation", async () => {
  const program = parse(`
    Create value as Number
    Set value to -15

    Create positive as Number
    Set positive to -value

    Call Print(value)
    Call Print(positive)
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenNthCalledWith(1, -15);
  expect(logSpy).toHaveBeenNthCalledWith(2, 15);
});

test("gives multiplication and division precedence over addition and subtraction", async () => {
  const program = parse(`
    Create result as Number
    Set result to 2 + 3 * 4
    Call Print(result)

    Create divisionFirst as Number
    Set divisionFirst to 20 - 12 / 3
    Call Print(divisionFirst)
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenNthCalledWith(1, 14);
  expect(logSpy).toHaveBeenNthCalledWith(2, 16);
});

test("parentheses override arithmetic precedence", async () => {
  const program = parse(`
    Create result as Number
    Set result to (2 + 3) * 4
    Call Print(result)
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledWith(20);
});

test("uses variables in arithmetic expressions", async () => {
  const program = parse(`
    Create first as Number
    Set first to 8

    Create second as Number
    Set second to 12

    Create sum as Number
    Set sum to first + second

    Create product as Number
    Set product to first * second

    Call Print(sum)
    Call Print(product)
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenNthCalledWith(1, 20);
  expect(logSpy).toHaveBeenNthCalledWith(2, 96);
});

test("uses arithmetic expressions in When conditions", async () => {
  const program = parse(`
    Create first as Number
    Set first to 10

    Create second as Number
    Set second to 5

    When first + second == 15 and first - second > 0 then {
      Call Print("condition met")
    } Otherwise {
      Call Print("condition failed")
    }
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledTimes(1);
  expect(logSpy).toHaveBeenCalledWith("condition met");
});

test("preserves not precedence below comparisons", async () => {
  const program = parse(`
    Create result as Boolean
    Set result to not (2 + 2 == 5)
    Call Print(result)
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledWith(true);
});

test("rejects arithmetic with non-Number operands", async () => {
  const program = parse(`
    Create text as String
    Set text to "hello"

    Create result as Number
    Set result to text + 1
  `);

  interpreter.register(program);

  await expect(interpreter.executeTopLevelCalls(program)).rejects.toThrow(
  /Operator "\+" expects two Numbers or two Strings/
  );
});

test("rejects unary negation of a non-Number operand", async () => {
  const program = parse(`
    Create result as Number
    Set result to -true
  `);

  interpreter.register(program);

  await expect(interpreter.executeTopLevelCalls(program)).rejects.toThrow(
    /Operator "-" expects Number operand/
  );
});

test("rejects division by zero", async () => {
  const program = parse(`
    Create result as Number
    Set result to 10 / 0
  `);

  interpreter.register(program);

  await expect(interpreter.executeTopLevelCalls(program)).rejects.toThrow(
    /Division by zero/
  );
});

test("repeats a block a fixed number of times", async () => {
  const program = parse(`
    Repeat 3 times {
      Call Print("tick")
    }
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledTimes(3);
  expect(logSpy).toHaveBeenNthCalledWith(1, "tick");
  expect(logSpy).toHaveBeenNthCalledWith(2, "tick");
  expect(logSpy).toHaveBeenNthCalledWith(3, "tick");
});

test("uses a Number variable as a Repeat count", async () => {
  const program = parse(`
    Create count as Number
    Set count to 2 + 2

    Repeat count times {
      Call Print("step")
    }
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledTimes(4);
  expect(logSpy).toHaveBeenNthCalledWith(1, "step");
  expect(logSpy).toHaveBeenNthCalledWith(4, "step");
});

test("updates variables inside a Repeat block", async () => {
  const program = parse(`
    Create count as Number
    Set count to 0

    Repeat 5 times {
      Set count to count + 1
    }

    Call Print(count)
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledWith(5);
});

test("supports When statements inside Repeat blocks", async () => {
  const program = parse(`
    Create count as Number
    Set count to 0

    Repeat 3 times {
      Set count to count + 1

      When count == 2 then {
        Call Print("middle")
      }
    }
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledTimes(1);
  expect(logSpy).toHaveBeenCalledWith("middle");
});

test("does not execute a Repeat block zero times", async () => {
  const program = parse(`
    Repeat 0 times {
      Call Print("never")
    }
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).not.toHaveBeenCalled();
});

test("rejects a non-Number Repeat count", async () => {
  const program = parse(`
    Repeat true times {
      Call Print("invalid")
    }
  `);

  interpreter.register(program);

  await expect(interpreter.executeTopLevelCalls(program)).rejects.toThrow(
    /Repeat count must evaluate to Number/
  );
});

test("rejects a negative Repeat count", async () => {
  const program = parse(`
    Repeat -1 times {
      Call Print("invalid")
    }
  `);

  interpreter.register(program);

  await expect(interpreter.executeTopLevelCalls(program)).rejects.toThrow(
    /Repeat count must be a non-negative integer/
  );
});

test("rejects a decimal Repeat count", async () => {
  const program = parse(`
    Repeat 2.5 times {
      Call Print("invalid")
    }
  `);

  interpreter.register(program);

  await expect(interpreter.executeTopLevelCalls(program)).rejects.toThrow(
    /Repeat count must be a non-negative integer/
  );
});

test("rejects a Repeat statement without times", async () => {
  expect(() =>
    parse(`
      Repeat 3 {
        Call Print("invalid")
      }
    `)
  ).toThrow(/Expected "times"/
  );
 });

test("executes a While block until its condition becomes false", async () => {
  const program = parse(`
    Create count as Number
    Set count to 0

    While count < 3 {
      Set count to count + 1
      Call Print(count)
    }
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledTimes(3);
  expect(logSpy).toHaveBeenNthCalledWith(1, 1);
  expect(logSpy).toHaveBeenNthCalledWith(2, 2);
  expect(logSpy).toHaveBeenNthCalledWith(3, 3);
});

test("does not execute a While block when its condition is initially false", async () => {
  const program = parse(`
    Create count as Number
    Set count to 5

    While count < 3 {
      Call Print("never")
    }

    Call Print(count)
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledTimes(1);
  expect(logSpy).toHaveBeenCalledWith(5);
});

test("supports nested While blocks", async () => {
  const program = parse(`
    Create outer as Number
    Create inner as Number

    Set outer to 0

    While outer < 2 {
      Set inner to 0

      While inner < 3 {
        Call Print(inner)
        Set inner to inner + 1
      }

      Set outer to outer + 1
    }
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledTimes(6);
  expect(logSpy).toHaveBeenNthCalledWith(1, 0);
  expect(logSpy).toHaveBeenNthCalledWith(2, 1);
  expect(logSpy).toHaveBeenNthCalledWith(3, 2);
  expect(logSpy).toHaveBeenNthCalledWith(4, 0);
  expect(logSpy).toHaveBeenNthCalledWith(5, 1);
  expect(logSpy).toHaveBeenNthCalledWith(6, 2);
});

test("supports While statements inside functions", async () => {
  const program = parse(`
    Define CountToThree() {
      Create count as Number
      Set count to 0

      While count < 3 {
        Set count to count + 1
      }

      Call Print(count)
    }

    Call CountToThree()
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledTimes(1);
  expect(logSpy).toHaveBeenCalledWith(3);
});

test("supports When statements inside While blocks", async () => {
  const program = parse(`
    Create count as Number
    Set count to 0

    While count < 4 {
      Set count to count + 1

      When count == 2 then {
        Call Print("middle")
      }
    }
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledTimes(1);
  expect(logSpy).toHaveBeenCalledWith("middle");
});

test("rejects a non-Boolean While condition", async () => {
  const program = parse(`
    While 123 {
      Call Print("invalid")
    }
  `);

  interpreter.register(program);

  await expect(interpreter.executeTopLevelCalls(program)).rejects.toThrow(
    /Condition in While statement must evaluate to Boolean/
  );
});

test("stops an infinite While loop at the safety limit", async () => {
  const program = parse(`
    While true {
    }
  `);

  interpreter.register(program);

  await expect(interpreter.executeTopLevelCalls(program)).rejects.toThrow(
    /While loop exceeded the maximum of 10000 iterations/
  );
});

test("Break exits a While loop", async () => {
  const program = parse(`
    Create count as Number
    Set count to 0

    While true {
      Set count to count + 1

      When count == 3 then {
        Break
      }

      Call Print(count)
    }

    Call Print(count)
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledTimes(3);
  expect(logSpy).toHaveBeenNthCalledWith(1, 1);
  expect(logSpy).toHaveBeenNthCalledWith(2, 2);
  expect(logSpy).toHaveBeenNthCalledWith(3, 3);
});

test("Continue skips the remainder of a While iteration", async () => {
  const program = parse(`
    Create count as Number
    Set count to 0

    While count < 5 {
      Set count to count + 1

      When count == 3 then {
        Continue
      }

      Call Print(count)
    }
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledTimes(4);
  expect(logSpy).toHaveBeenNthCalledWith(1, 1);
  expect(logSpy).toHaveBeenNthCalledWith(2, 2);
  expect(logSpy).toHaveBeenNthCalledWith(3, 4);
  expect(logSpy).toHaveBeenNthCalledWith(4, 5);
});

test("Break exits a Repeat loop", async () => {
  const program = parse(`
    Create count as Number
    Set count to 0

    Repeat 10 times {
      Set count to count + 1

      When count == 3 then {
        Break
      }

      Call Print(count)
    }

    Call Print(count)
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledTimes(3);
  expect(logSpy).toHaveBeenNthCalledWith(1, 1);
  expect(logSpy).toHaveBeenNthCalledWith(2, 2);
  expect(logSpy).toHaveBeenNthCalledWith(3, 3);
});

test("Continue skips the remainder of a Repeat iteration", async () => {
  const program = parse(`
    Create count as Number
    Set count to 0

    Repeat 5 times {
      Set count to count + 1

      When count == 3 then {
        Continue
      }

      Call Print(count)
    }
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledTimes(4);
  expect(logSpy).toHaveBeenNthCalledWith(1, 1);
  expect(logSpy).toHaveBeenNthCalledWith(2, 2);
  expect(logSpy).toHaveBeenNthCalledWith(3, 4);
  expect(logSpy).toHaveBeenNthCalledWith(4, 5);
});

test("Break affects only the nearest nested loop", async () => {
  const program = parse(`
    Create outer as Number
    Create inner as Number

    Set outer to 0

    While outer < 2 {
      Set inner to 0

      While true {
        Set inner to inner + 1

        When inner == 2 then {
          Break
        }

        Call Print(inner)
      }

      Set outer to outer + 1
    }

    Call Print(outer)
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledTimes(3);
  expect(logSpy).toHaveBeenNthCalledWith(1, 1);
  expect(logSpy).toHaveBeenNthCalledWith(2, 1);
  expect(logSpy).toHaveBeenNthCalledWith(3, 2);
});

test("Continue affects only the nearest nested loop", async () => {
  const program = parse(`
    Create outer as Number
    Create inner as Number

    Set outer to 0

    While outer < 2 {
      Set inner to 0

      While inner < 3 {
        Set inner to inner + 1

        When inner == 2 then {
          Continue
        }

        Call Print(inner)
      }

      Set outer to outer + 1
    }
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledTimes(4);
  expect(logSpy).toHaveBeenNthCalledWith(1, 1);
  expect(logSpy).toHaveBeenNthCalledWith(2, 3);
  expect(logSpy).toHaveBeenNthCalledWith(3, 1);
  expect(logSpy).toHaveBeenNthCalledWith(4, 3);
});

test("Break works inside a When branch within a function loop", async () => {
  const program = parse(`
    Define CountUntilThree() {
      Create count as Number
      Set count to 0

      While true {
        Set count to count + 1

        When count == 3 then {
          Break
        }
      }

      Call Print(count)
    }

    Call CountUntilThree()
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledTimes(1);
  expect(logSpy).toHaveBeenCalledWith(3);
});

test("rejects Break outside a loop", async () => {
  expect(() =>
    parse(`
      Break
    `)
  ).toThrow(/Break can only be used inside a While or Repeat loop/);
});

test("rejects Continue outside a loop", async () => {
  expect(() =>
    parse(`
      Continue
    `)
  ).toThrow(/Continue can only be used inside a While or Repeat loop/);
});

test("creates and prints a List literal", async () => {
  const program = parse(`
    Create numbers as List
    Set numbers to [10, 20, 30]
    Call Print(numbers)
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledWith([10, 20, 30]);
});

test("supports empty and nested List literals", async () => {
  const program = parse(`
    Create empty as List
    Set empty to []

    Create matrix as List
    Set matrix to [[1, 2], [3, 4]]

    Call Print(empty)
    Call Print(matrix)
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenNthCalledWith(1, []);
  expect(logSpy).toHaveBeenNthCalledWith(2, [
    [1, 2],
    [3, 4],
  ]);
});

test("evaluates expressions inside List literals", async () => {
  const program = parse(`
    Create base as Number
    Set base to 10

    Create values as List
    Set values to [base, base + 5, base * 2]

    Call Print(values)
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledWith([10, 15, 20]);
});

test("gets List length as an expression", async () => {
  const program = parse(`
    Create values as List
    Set values to ["a", "b", "c"]

    Create size as Number
    Set size to Length(values)

    Call Print(size)
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledWith(3);
});

test("gets a List element as an expression", async () => {
  const program = parse(`
    Create values as List
    Set values to [10, 20, 30]

    Create index as Number
    Set index to 1

    Create selected as Number
    Set selected to Get(values, index)

    Call Print(selected)
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledWith(20);
});

test("Push appends a value to a List", async () => {
  const program = parse(`
    Create values as List
    Set values to [1, 2]

    Call Push(values, 3)
    Call Print(values)
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledWith([1, 2, 3]);
});

test("Pop removes and returns the final List element", async () => {
  const program = parse(`
    Create values as List
    Set values to [1, 2, 3]

    Create removed as Number
    Set removed to Pop(values)

    Call Print(removed)
    Call Print(values)
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenNthCalledWith(1, 3);
  expect(logSpy).toHaveBeenNthCalledWith(2, [1, 2]);
});

test("supports List equality and inequality", async () => {
  const program = parse(`
    Create same as Boolean
    Set same to [1, [2, 3]] == [1, [2, 3]]

    Create different as Boolean
    Set different to [1, 2] != [1, 3]

    Call Print(same)
    Call Print(different)
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenNthCalledWith(1, true);
  expect(logSpy).toHaveBeenNthCalledWith(2, true);
});

test("copies List values during assignment", async () => {
  const program = parse(`
    Create first as List
    Set first to [1, 2]

    Create second as List
    Set second to first

    Call Push(second, 3)

    Call Print(first)
    Call Print(second)
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenNthCalledWith(1, [1, 2]);
  expect(logSpy).toHaveBeenNthCalledWith(2, [1, 2, 3]);
});

test("rejects assigning a non-List value to a List variable", async () => {
  const program = parse(`
    Create values as List
    Set values to 123
  `);

  interpreter.register(program);

  await expect(interpreter.executeTopLevelCalls(program)).rejects.toThrow(
    /Type mismatch: variable "values" expects List, received Number/
  );
});

test("rejects Length with a non-List argument", async () => {
  const program = parse(`
    Create result as Number
    Set result to Length(123)
  `);

  interpreter.register(program);

  await expect(interpreter.executeTopLevelCalls(program)).rejects.toThrow(
    /Length expects a List argument/
  );
});

test("rejects Get with an invalid List index", async () => {
  const program = parse(`
    Create values as List
    Set values to [10, 20]

    Create selected as Number
    Set selected to Get(values, -1)
  `);

  interpreter.register(program);

  await expect(interpreter.executeTopLevelCalls(program)).rejects.toThrow(
    /Get index must be a non-negative integer/
  );
});

test("rejects Get outside List bounds", async () => {
  const program = parse(`
    Create values as List
    Set values to [10, 20]

    Create selected as Number
    Set selected to Get(values, 2)
  `);

  interpreter.register(program);

  await expect(interpreter.executeTopLevelCalls(program)).rejects.toThrow(
    /List index 2 is out of bounds/
  );
});

test("rejects Push when its first argument is not a List variable", async () => {
  const program = parse(`
    Call Push([1, 2], 3)
  `);

  interpreter.register(program);

  await expect(interpreter.executeTopLevelCalls(program)).rejects.toThrow(
    /Push expects a List variable as its first argument/
  );
});

test("rejects Pop from an empty List", async () => {
  const program = parse(`
    Create values as List
    Set values to []

    Create removed as Number
    Set removed to Pop(values)
  `);

  interpreter.register(program);

  await expect(interpreter.executeTopLevelCalls(program)).rejects.toThrow(
    /Cannot Pop from an empty List/
  );
});

test("returns a value from a user-defined function", async () => {
  const program = parse(`
    Define Double(value) {
      Return value * 2
    }

    Create result as Number
    Set result to Double(21)

    Call Print(result)
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledWith(42);
});

test("supports function calls inside larger expressions", async () => {
  const program = parse(`
    Define AddOne(value) {
      Return value + 1
    }

    Create result as Number
    Set result to AddOne(4) * 2 + AddOne(1)

    Call Print(result)
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledWith(12);
});

test("supports a function returning a List", async () => {
  const program = parse(`
    Define MakePair(first, second) {
      Return [first, second]
    }

    Create pair as List
    Set pair to MakePair(10, 20)

    Call Print(pair)
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledWith([10, 20]);
});

test("supports early Return from a When block", async () => {
  const program = parse(`
    Define Absolute(value) {
      When value < 0 then {
        Return -value
      }

      Return value
    }

    Create negative as Number
    Set negative to Absolute(-7)

    Create positive as Number
    Set positive to Absolute(5)

    Call Print(negative)
    Call Print(positive)
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenNthCalledWith(1, 7);
  expect(logSpy).toHaveBeenNthCalledWith(2, 5);
});

test("supports Return from inside a loop", async () => {
  const program = parse(`
    Define FindFirstGreaterThan(values, limit) {
      Create index as Number
      Create current as Number

        Set index to 0

      While index < Length(values) {
       
        Set current to Get(values, index)

        When current > limit then {
          Return current
        }

        Set index to index + 1
      }

      Return -1
    }

    Create result as Number
    Set result to FindFirstGreaterThan([2, 4, 8, 10], 5)

    Call Print(result)
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledWith(8);
});

test("supports nested user-defined function calls", async () => {
  const program = parse(`
    Define Double(value) {
      Return value * 2
    }

    Define Quadruple(value) {
      Return Double(Double(value))
    }

    Create result as Number
    Set result to Quadruple(3)

    Call Print(result)
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledWith(12);
});

test("copies a List returned by a function", async () => {
  const program = parse(`
    Define MakeList() {
      Return [1, 2]
    }

    Create first as List
    Set first to MakeList()

    Create second as List
    Set second to MakeList()

    Call Push(second, 3)

    Call Print(first)
    Call Print(second)
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenNthCalledWith(1, [1, 2]);
  expect(logSpy).toHaveBeenNthCalledWith(2, [1, 2, 3]);
});

test("allows Call to discard a function Return value", async () => {
  const program = parse(`
    Define Announce() {
      Call Print("inside")
      Return 99
    }

    Call Announce()
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledTimes(1);
  expect(logSpy).toHaveBeenCalledWith("inside");
});

test("rejects Return outside a function", async () => {
  expect(() =>
    parse(`
      Return 42
    `)
  ).toThrow(/Return can only be used inside a function/);
});

test("rejects an expression call to an undefined function", async () => {
  const program = parse(`
    Create result as Number
    Set result to MissingFunction()
  `);

  interpreter.register(program);

  await expect(interpreter.executeTopLevelCalls(program)).rejects.toThrow(
    /Undefined function "MissingFunction"/
  );
});

test("rejects an expression call with incorrect argument count", async () => {
  const program = parse(`
    Define Add(left, right) {
      Return left + right
    }

    Create result as Number
    Set result to Add(1)
  `);

  interpreter.register(program);

  await expect(interpreter.executeTopLevelCalls(program)).rejects.toThrow(
    /Function "Add" expects 2 argument\(s\), received 1/
  );
});

test("rejects using a function without Return as an expression", async () => {
  const program = parse(`
    Define NoResult() {
      Call Print("ran")
    }

    Create result as Number
    Set result to NoResult()
  `);

  interpreter.register(program);

  await expect(interpreter.executeTopLevelCalls(program)).rejects.toThrow(
    /Function "NoResult" did not Return a value/
  );
});

test("creates a fresh block scope for every While iteration", async () => {
  const program = parse(`
    Create index as Number
    Set index to 0

    While index < 3 {
      Create current as Number
      Set current to index * 10

      Call Print(current)

      Set index to index + 1
    }
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledTimes(3);
  expect(logSpy).toHaveBeenNthCalledWith(1, 0);
  expect(logSpy).toHaveBeenNthCalledWith(2, 10);
  expect(logSpy).toHaveBeenNthCalledWith(3, 20);
});

test("creates a fresh block scope for every Repeat iteration", async () => {
  const program = parse(`
    Create count as Number
    Set count to 0

    Repeat 3 times {
      Create current as Number
      Set current to count + 1

      Call Print(current)

      Set count to count + 1
    }
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenNthCalledWith(1, 1);
  expect(logSpy).toHaveBeenNthCalledWith(2, 2);
  expect(logSpy).toHaveBeenNthCalledWith(3, 3);
});

test("does not expose a When block variable outside its block", async () => {
  const program = parse(`
    When true then {
      Create message as String
      Set message to "inside"
      Call Print(message)
    }

    Call Print(message)
  `);

  interpreter.register(program);

  await expect(interpreter.executeTopLevelCalls(program)).rejects.toThrow(
    /Undefined variable "message"/
  );

  expect(logSpy).toHaveBeenCalledWith("inside");
});

test("does not expose a While block variable after the loop", async () => {
  const program = parse(`
    Create count as Number
    Set count to 0

    While count < 1 {
      Create temporary as Number
      Set temporary to 99
      Set count to count + 1
    }

    Call Print(temporary)
  `);

  interpreter.register(program);

  await expect(interpreter.executeTopLevelCalls(program)).rejects.toThrow(
    /Undefined variable "temporary"/
  );
});

test("does not expose a Repeat block variable after the loop", async () => {
  const program = parse(`
    Repeat 1 times {
      Create temporary as String
      Set temporary to "hidden"
    }

    Call Print(temporary)
  `);

  interpreter.register(program);

  await expect(interpreter.executeTopLevelCalls(program)).rejects.toThrow(
    /Undefined variable "temporary"/
  );
});

test("allows a block to update a variable from its parent scope", async () => {
  const program = parse(`
    Create total as Number
    Set total to 0

    Repeat 4 times {
      Set total to total + 5
    }

    Call Print(total)
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledWith(20);
});

test("allows inner blocks to read outer variables", async () => {
  const program = parse(`
    Create base as Number
    Set base to 7

    When true then {
      Create result as Number
      Set result to base * 2
      Call Print(result)
    }
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledWith(14);
});

test("allows a nested block to shadow an outer variable", async () => {
  const program = parse(`
    Create value as Number
    Set value to 10

    When true then {
      Create value as Number
      Set value to 20
      Call Print(value)
    }

    Call Print(value)
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenNthCalledWith(1, 20);
  expect(logSpy).toHaveBeenNthCalledWith(2, 10);
});

test("allows Return to escape nested block scopes", async () => {
  const program = parse(`
    Define FindValue() {
      Repeat 3 times {
        Create value as Number
        Set value to 42
        Return value
      }

      Return 0
    }

    Create result as Number
    Set result to FindValue()

    Call Print(result)
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledWith(42);
});

test("concatenates two Strings with +", async () => {
  const program = parse(`
    Create greeting as String
    Set greeting to "Hello, " + "OSHPYT"

    Call Print(greeting)
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledWith("Hello, OSHPYT");
});

test("rejects string and number concatenation without ToString", async () => {
  const program = parse(`
    Create text as String
    Set text to "Value: " + 42
  `);

  interpreter.register(program);

  await expect(interpreter.executeTopLevelCalls(program)).rejects.toThrow(
    /Operator "\+" expects two Numbers or two Strings/
  );
});

test("converts Number, Boolean, and List values with ToString", async () => {
  const program = parse(`
    Create numberText as String
    Set numberText to ToString(42)

    Create booleanText as String
    Set booleanText to ToString(true)

    Create listText as String
    Set listText to ToString([1, [2, 3]])

    Call Print(numberText)
    Call Print(booleanText)
    Call Print(listText)
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenNthCalledWith(1, "42");
  expect(logSpy).toHaveBeenNthCalledWith(2, "true");
  expect(logSpy).toHaveBeenNthCalledWith(3, "[1, [2, 3]]");
});

test("converts a numeric String with ToNumber", async () => {
  const program = parse(`
    Create value as Number
    Set value to ToNumber(" 42.5 ")

    Call Print(value)
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledWith(42.5);
});

test("rejects ToNumber with invalid text", async () => {
  const program = parse(`
    Create value as Number
    Set value to ToNumber("not a number")
  `);

  interpreter.register(program);

  await expect(interpreter.executeTopLevelCalls(program)).rejects.toThrow(
    /ToNumber could not convert "not a number" to Number/
  );
});

test("rejects ToNumber with a non-String value", async () => {
  const program = parse(`
    Create value as Number
    Set value to ToNumber(42)
  `);

  interpreter.register(program);

  await expect(interpreter.executeTopLevelCalls(program)).rejects.toThrow(
    /ToNumber expects a String argument/
  );
});

test("returns runtime type names with TypeOf", async () => {
  const program = parse(`
    Call Print(TypeOf("text"))
    Call Print(TypeOf(42))
    Call Print(TypeOf(false))
    Call Print(TypeOf([1, 2]))
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenNthCalledWith(1, "String");
  expect(logSpy).toHaveBeenNthCalledWith(2, "Number");
  expect(logSpy).toHaveBeenNthCalledWith(3, "Boolean");
  expect(logSpy).toHaveBeenNthCalledWith(4, "List");
});

test("reads input through a configured input provider", async () => {
  const program = parse(`
    Create name as String
    Set name to Input("Name: ")

    Call Print("Hello, " + name)
  `);

  const inputProvider = async (prompt: string): Promise<string> => {
    expect(prompt).toBe("Name: ");
    return "Ada";
  };

  const inputInterpreter = new Interpreter(inputProvider);

  inputInterpreter.register(program);
  await inputInterpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledWith("Hello, Ada");
});

test("rejects Input with a non-String prompt", async () => {
  const program = parse(`
    Create response as String
    Set response to Input(123)
  `);

  const inputInterpreter = new Interpreter(async () => "unused");

  inputInterpreter.register(program);

  await expect(
    inputInterpreter.executeTopLevelCalls(program)
  ).rejects.toThrow(/Input expects a String prompt/);
});

test("supports Input inside a user-defined function", async () => {
  const program = parse(`
    Define AskName() {
      Return Input("Your name: ")
    }

    Create name as String
    Set name to AskName()

    Call Print("Hi, " + name)
  `);

  const inputInterpreter = new Interpreter(async (prompt) => {
    expect(prompt).toBe("Your name: ");
    return "Grace";
  });

  inputInterpreter.register(program);
  await inputInterpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledWith("Hi, Grace");
});

test("concatenates two Strings with +", async () => {
  const program = parse(`
    Create greeting as String
    Set greeting to "Hello, " + "OSHPYT"

    Call Print(greeting)
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledWith("Hello, OSHPYT");
});

test("rejects string and number concatenation without ToString", async () => {
  const program = parse(`
    Create text as String
    Set text to "Value: " + 42
  `);

  interpreter.register(program);

  await expect(interpreter.executeTopLevelCalls(program)).rejects.toThrow(
    /Operator "\+" expects two Numbers or two Strings/
  );
});

test("converts Number, Boolean, and List values with ToString", async () => {
  const program = parse(`
    Create numberText as String
    Set numberText to ToString(42)

    Create booleanText as String
    Set booleanText to ToString(true)

    Create listText as String
    Set listText to ToString([1, [2, 3]])

    Call Print(numberText)
    Call Print(booleanText)
    Call Print(listText)
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenNthCalledWith(1, "42");
  expect(logSpy).toHaveBeenNthCalledWith(2, "true");
  expect(logSpy).toHaveBeenNthCalledWith(3, "[1, [2, 3]]");
});

test("converts a numeric String with ToNumber", async () => {
  const program = parse(`
    Create value as Number
    Set value to ToNumber(" 42.5 ")

    Call Print(value)
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledWith(42.5);
});

test("rejects ToNumber with invalid text", async () => {
  const program = parse(`
    Create value as Number
    Set value to ToNumber("not a number")
  `);

  interpreter.register(program);

  await expect(interpreter.executeTopLevelCalls(program)).rejects.toThrow(
    /ToNumber could not convert "not a number" to Number/
  );
});

test("rejects ToNumber with a non-String value", async () => {
  const program = parse(`
    Create value as Number
    Set value to ToNumber(42)
  `);

  interpreter.register(program);

  await expect(interpreter.executeTopLevelCalls(program)).rejects.toThrow(
    /ToNumber expects a String argument/
  );
});

test("returns runtime type names with TypeOf", async () => {
  const program = parse(`
    Call Print(TypeOf("text"))
    Call Print(TypeOf(42))
    Call Print(TypeOf(false))
    Call Print(TypeOf([1, 2]))
  `);

  interpreter.register(program);
  await interpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenNthCalledWith(1, "String");
  expect(logSpy).toHaveBeenNthCalledWith(2, "Number");
  expect(logSpy).toHaveBeenNthCalledWith(3, "Boolean");
  expect(logSpy).toHaveBeenNthCalledWith(4, "List");
});

test("reads input through a configured input provider", async () => {
  const program = parse(`
    Create name as String
    Set name to Input("Name: ")

    Call Print("Hello, " + name)
  `);

  const inputProvider = async (prompt: string): Promise<string> => {
    expect(prompt).toBe("Name: ");
    return "Ada";
  };

  const inputInterpreter = new Interpreter(inputProvider);

  inputInterpreter.register(program);
  await inputInterpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledWith("Hello, Ada");
});

test("rejects Input with a non-String prompt", async () => {
  const program = parse(`
    Create response as String
    Set response to Input(123)
  `);

  const inputInterpreter = new Interpreter(async () => "unused");

  inputInterpreter.register(program);

  await expect(
    inputInterpreter.executeTopLevelCalls(program)
  ).rejects.toThrow(/Input expects a String prompt/);
});

test("supports Input inside a user-defined function", async () => {
  const program = parse(`
    Define AskName() {
      Return Input("Your name: ")
    }

    Create name as String
    Set name to AskName()

    Call Print("Hi, " + name)
  `);

  const inputInterpreter = new Interpreter(async (prompt) => {
    expect(prompt).toBe("Your name: ");
    return "Grace";
  });

  inputInterpreter.register(program);
  await inputInterpreter.executeTopLevelCalls(program);

  expect(logSpy).toHaveBeenCalledWith("Hi, Grace");
});

test("parses a top-level Import statement", async () => {
  const nodes = parse(`
    Import "modules/math.oshpyt"
  `);

  expect(nodes).toHaveLength(1);
  expect(nodes[0]).toMatchObject({
    type: "Import",
    path: "modules/math.oshpyt",
  });
});