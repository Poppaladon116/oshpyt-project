import { OshpytTokenizer } from './tokenizer';
import { Tensor } from './OshpytTensor';
import { ModelStreamer } from './ModelStreamer';

async function startChat() {
    console.log("--- WELCOME TO OSHPYT AI (Inference Mode) ---");

    const tokenizer = new OshpytTokenizer();
    const streamer = new ModelStreamer("giant_model.bin"); // Our "Brain" on disk

    // 1. INPUT
    let inputPath = "Hello";
    console.log(`User: ${inputPath}`);

    // 2. GENERATION LOOP (Generate 5 words)
    for (let i = 0; i < 5; i++) {
        const tokens = tokenizer.encode(inputPath);
        
        // Load a layer from the SSD to the GPU
        const layer = streamer.loadLayer(0, 128); 
        const inputTensor = Tensor.fromArray(new Array(128).fill(0.5), 1, 128);

        // THE "THOUGHT" PROCESS
        // Math -> Activation -> Decision
        const prediction = inputTensor
            .matmul(layer)
            .relu()
            .softmax();

        // Predict a word (simulated)
        const nextWord = "world"; 
        inputPath += " " + nextWord;

        // Cleanup the GPU to keep VRAM fresh
        layer.destroy();
        inputTensor.destroy();
        prediction.destroy();
    }

    console.log(`OSHPYT AI: ${inputPath}`);
}

startChat();