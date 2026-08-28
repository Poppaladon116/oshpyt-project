import koffi from 'koffi';
import { OshpytTokenizer } from './tokenizer';

// --- 1. SETUP THE ENGINE ---
const lib = koffi.load('./oshpyt_engine.dll');
const oshpyt_matmul = lib.func('void oshpyt_matmul(float *A, float *B, float *C, int m, int n, int k)');
const load_weights = lib.func('void *load_weights_to_gpu(float *host_weights, int total_elements)');

async function runOshpyt() {
    console.log("--- OSHPYT LLM STARTING ---");

    // --- 2. TOKENIZATION ---
    const tokenizer = new OshpytTokenizer();
    const input = "Hello world";
    const tokens = tokenizer.encode(input);
    console.log(`Input: "${input}" -> Tokens: [${tokens}]`);

    // --- 3. LOADING WEIGHTS ---
    // We create some "dummy" weights (e.g., 128x128 matrix)
    const size = 128;
    const dummyWeights = new Float32Array(size * size).fill(0.01);
    const gpuWeightPtr = load_weights(dummyWeights, dummyWeights.length);

    // --- 4. THE FORWARD PASS (The "Thought") ---
    // We convert our tokens into a small matrix for the GPU to process
    const inputMatrix = new Float32Array(size * size).fill(0.5);
    const outputMatrix = new Float32Array(size * size);

    console.log("OSHPYT: Running Forward Pass on GPU...");
    
    // We call the math: Output = Input * Weights
    oshpyt_matmul(inputMatrix, gpuWeightPtr, outputMatrix, size, size, size);

    // --- 5. GENERATION ---
    // In a real LLM, we'd pick the highest number in outputMatrix
    // For now, let's simulate predicting the next word
    console.log("OSHPYT: Thought Process Complete.");
    const predictedToken = [995]; // Let's pretend it predicted "world"
    const result = tokenizer.decode(predictedToken);

    console.log(`Predicted next word: "${result}"`);
}

runOshpyt();