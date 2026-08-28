import { Tensor } from './OshpytTensor';

function testSoftmax() {
    console.log("--- OSHPYT CHALLENGE 2: SOFTMAX DECISION ---");

    // Imagine these are scores for 3 words: [cat, dog, robot]
    // The model thinks it's likely "robot" (score 5.0)
    const scores = Tensor.fromArray([1.0, 2.0, 5.0], 1, 3);

    console.log("Scores before Softmax: [1.0, 2.0, 5.0]");

    // Turn scores into probabilities
    scores.softmax();

    console.log("OSHPYT has made a decision!");
    console.log("Check your console for the 'Parallel Softmax Applied' message.");
    
    // In a real LLM, [1.0, 2.0, 5.0] becomes roughly [0.01, 0.04, 0.95]
    // Meaning 95% chance it's "robot"!
}

testSoftmax();