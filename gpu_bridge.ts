import koffi from 'koffi';

// 1. Load the DLL you compiled
const lib = koffi.load('./oshpyt_engine.dll');

// 2. Define the "oshpyt_matmul" function so TS knows how to call it
// It takes 3 pointers (the matrices) and 3 integers (the sizes)
const oshpyt_matmul = lib.func('void oshpyt_matmul(float *A, float *B, float *C, int m, int n, int k)');

/**
 * This is the function you will use in your language!
 */
export function runGPUMath(size: number) {
    // Create 3 arrays (Matrices) in memory
    const matrixA = new Float32Array(size * size).fill(1.5);
    const matrixB = new Float32Array(size * size).fill(2.0);
    const resultMatrix = new Float32Array(size * size);

    console.log(`OSHPYT: Sending ${size}x${size} matrix to GPU...`);

    // 3. CALL THE DLL MATH
    oshpyt_matmul(matrixA, matrixB, resultMatrix, size, size, size);

    console.log("OSHPYT: GPU is finished! First result value:", resultMatrix[0]);
    return resultMatrix;
}

// Test it
runGPUMath(128);