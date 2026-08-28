import { Tensor } from './OshpytTensor';

function runPushTest() {
    console.log("--- OSHPYT CHALLENGE 1: FLUENT SYNTAX ---");

    // 1. Create Tensors
    const input = Tensor.fromArray([0.5, -1.2, 0.8, 2.0], 1, 4);
    const weights = Tensor.fromArray(new Array(16).fill(0.1), 4, 4);
    const bias = Tensor.fromArray(new Array(4).fill(0.01), 1, 4);

    // 2. THE CHAINED EXECUTION (The Goal!)
    // This looks like real AI code!
    const output = input
        .matmul(weights)
        .add(bias)
        .relu();

    console.log("OSHPYT: Chain complete. GPU has processed the layer.");
}

runPushTest();