import { Tensor } from "./OshpytTensor";

const preActivation = Tensor.fromArray(
  new Float32Array([-2, -0.1, 0, 0.1, 3]),
  1,
  5
);

const gradient = Tensor.fromArray(
  new Float32Array([1, 1, 1, 1, 1]),
  1,
  5
);

gradient.reluBackward(preActivation);

console.log(Array.from(gradient.download()));

preActivation.destroy();
gradient.destroy();