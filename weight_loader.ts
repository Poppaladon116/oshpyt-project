import * as fs from 'fs';
import koffi from 'koffi';

// 1. Setup the Bridge
const lib = koffi.load('./oshpyt_engine.dll');
// We use 'void *' because it's a memory address on the GPU
const load_weights = lib.func('void *load_weights_to_gpu(float *host_weights, int total_elements)');

export function loadModelWeights(filePath: string) {
    // 2. Read the binary file from Windows 11 disk
    const data = fs.readFileSync(filePath);
    
    // 3. Convert the buffer into a Float32 array (the format LLMs use)
    const weightData = new Float32Array(data.buffer, data.byteOffset, data.byteLength / 4);
    
    console.log(`OSHPYT: Reading ${filePath}... Found ${weightData.length} parameters.`);

    // 4. Send to GPU and get the VRAM address back
    const gpuPointer = load_weights(weightData, weightData.length);

    console.log("OSHPYT: Weights are now live in VRAM at address:", gpuPointer);
    return gpuPointer;
}

// --- TEST RUN ---
// First, let's create a "fake" weights file to test
const dummyWeights = new Float32Array([0.1, 0.5, -0.2, 0.8, 0.01]);
fs.writeFileSync('model_weights.bin', Buffer.from(dummyWeights.buffer));

// Now load it
loadModelWeights('model_weights.bin');