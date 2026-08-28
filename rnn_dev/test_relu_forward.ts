import { Tensor } from "./OshpytTensor";

const values = Tensor.fromArray(
  new Float32Array([-2, -0.1, 0, 0.1, 3]),
  1,
  5
);

values.relu();

console.log(Array.from(values.download()));

values.destroy();