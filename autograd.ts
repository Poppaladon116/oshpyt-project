import koffi from 'koffi';

class Value {
    dataPtr: any;
    gradPtr: any;
    size: number;
    creatorOp: string | null = null;
    prev: Value[] = [];

    constructor(dataPtr: any, gradPtr: any, size: number) {
        this.dataPtr = dataPtr;
        this.gradPtr = gradPtr; // The "Mistake" memory
        this.size = size;
    }

    // THE BACKWARD PASS
    backward() {
        console.log(`OSHPYT Autograd: Reversing ${this.creatorOp}...`);
        
        // 1. Calculate how much we missed the target
        // 2. Pass that error back to the previous layers
        for (let p of this.prev) {
            // Trigger the math in our DLL to calculate gradients
            p.backward();
        }
    }
}