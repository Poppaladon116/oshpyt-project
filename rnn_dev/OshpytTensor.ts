import koffi from "koffi";

const lib = koffi.load("./oshpyt_engine.dll");

const oshpyt_init = lib.func("void oshpyt_init()");
const oshpyt_set_point = lib.func(
  "void oshpyt_set_point(float *data, int size, int index, float value)"
);
const oshpyt_malloc = lib.func(
  "float *oshpyt_malloc(int total_elements)"
);
const oshpyt_upload = lib.func(
  "void oshpyt_upload(float *host_ptr, float *gpu_ptr, int size)"
);
const oshpyt_download = lib.func(
  "void oshpyt_download(void *gpu_ptr, float *host_ptr, int size)"
);
const oshpyt_matmul_to = lib.func(
  "void oshpyt_matmul_to(float *A, float *B, float *C, int m, int n, int k)"
);
const oshpyt_grad_input_to = lib.func(
  "void oshpyt_grad_input_to(float *grad_output, float *weights, float *grad_input, int m, int n, int k)"
);
const oshpyt_weight_grad_to = lib.func(
  "void oshpyt_weight_grad_to(float *input, float *grad_output, float *grad_weights, int m, int n, int k)"
);
const launch_sgd = lib.func(
  "void launch_sgd(float *w, float *g, int size, float lr)"
);
const oshpyt_relu = lib.func(
  "void oshpyt_relu(float *data, int size)"
);
const oshpyt_relu_backward = lib.func(
  "void oshpyt_relu_backward(float *pre_activation, float *gradient, int size)"
);
const oshpyt_add = lib.func(
  "void oshpyt_add(float *A, float *B, int size)"
);
const oshpyt_softmax = lib.func(
  "void oshpyt_softmax(float *data, int size)"
);
const oshpyt_free = lib.func(
  "void oshpyt_free(void *ptr)"
);

export class Tensor {
  ptr: any;
  rows: number;
  cols: number;
  size: number;

  constructor(ptr: any, rows: number, cols: number) {
    this.ptr = ptr;
    this.rows = rows;
    this.cols = cols;
    this.size = rows * cols;
  }

  static fromArray(
    data: Float32Array,
    rows: number,
    cols: number
  ): Tensor {
    if (data.length !== rows * cols) {
      throw new Error(
        `fromArray size mismatch: got ${data.length}, ` +
        `expected ${rows * cols}.`
      );
    }

    const ptr = oshpyt_malloc(rows * cols);

    if (!ptr) {
      throw new Error("oshpyt_malloc failed.");
    }

    oshpyt_upload(data, ptr, data.length);

    return new Tensor(ptr, rows, cols);
  }

  static allocate(rows: number, cols: number): Tensor {
    const ptr = oshpyt_malloc(rows * cols);

    if (!ptr) {
      throw new Error("oshpyt_malloc failed.");
    }

    return new Tensor(ptr, rows, cols);
  }

  update(data: Float32Array): void {
    if (data.length !== this.size) {
      throw new Error(
        `update size mismatch: got ${data.length}, ` +
        `expected ${this.size}.`
      );
    }

    oshpyt_upload(data, this.ptr, data.length);
  }

  setPoint(index: number, value = 1.0): void {
    if (index < 0 || index >= this.size) {
      throw new Error(
        `setPoint index ${index} is outside tensor size ${this.size}.`
      );
    }

    oshpyt_set_point(this.ptr, this.size, index, value);
  }

  add(other: Tensor): Tensor {
    if (this.size !== other.size) {
      throw new Error(
        `add size mismatch: ${this.size} vs ${other.size}.`
      );
    }

    oshpyt_add(this.ptr, other.ptr, this.size);

    return this;
  }

  matmulTo(weights: Tensor, target: Tensor): void {
    if (this.cols !== weights.rows) {
      throw new Error(
        `matmul mismatch: (${this.rows}x${this.cols}) * ` +
        `(${weights.rows}x${weights.cols}).`
      );
    }

    if (
      target.rows !== this.rows ||
      target.cols !== weights.cols
    ) {
      throw new Error(
        `matmul target must be ${this.rows}x${weights.cols}; got ` +
        `${target.rows}x${target.cols}.`
      );
    }

    oshpyt_matmul_to(
      this.ptr,
      weights.ptr,
      target.ptr,
      this.rows,
      weights.cols,
      this.cols
    );
  }

  backwardTo(input: Tensor, gradOutput: Tensor): void {
    if (
      input.rows !== gradOutput.rows ||
      this.rows !== input.cols ||
      this.cols !== gradOutput.cols
    ) {
      throw new Error(
        `weight gradient mismatch: input ${input.rows}x${input.cols}, ` +
        `grad ${gradOutput.rows}x${gradOutput.cols}, ` +
        `weights ${this.rows}x${this.cols}.`
      );
    }

    oshpyt_weight_grad_to(
      input.ptr,
      gradOutput.ptr,
      this.ptr,
      input.rows,
      gradOutput.cols,
      input.cols
    );
  }

  gradInputTo(weights: Tensor, gradOutput: Tensor): void {
    if (
      gradOutput.cols !== weights.cols ||
      this.rows !== gradOutput.rows ||
      this.cols !== weights.rows
    ) {
      throw new Error(
        `input gradient mismatch: grad ${gradOutput.rows}x${gradOutput.cols}, ` +
        `weights ${weights.rows}x${weights.cols}, ` +
        `target ${this.rows}x${this.cols}.`
      );
    }

    oshpyt_grad_input_to(
      gradOutput.ptr,
      weights.ptr,
      this.ptr,
      gradOutput.rows,
      weights.cols,
      weights.rows
    );
  }

  leakyRelu(): Tensor {
    oshpyt_relu(this.ptr, this.size);
    return this;
  }

  relu(): Tensor {
    return this.leakyRelu();
  }

  leakyReluBackward(preActivation: Tensor): Tensor {
    if (this.size !== preActivation.size) {
      throw new Error(
        `leakyReluBackward mismatch: ${this.size} vs ${preActivation.size}.`
      );
    }

    oshpyt_relu_backward(
      preActivation.ptr,
      this.ptr,
      this.size
    );

    return this;
  }

  reluBackward(preActivation: Tensor): Tensor {
    return this.leakyReluBackward(preActivation);
  }

  softmax(): Tensor {
    oshpyt_softmax(this.ptr, this.size);
    return this;
  }

  optimizerStep(
    gradient: Tensor,
    learningRate: number
  ): void {
    if (this.size !== gradient.size) {
      throw new Error(
        `optimizerStep mismatch: ${this.size} vs ${gradient.size}.`
      );
    }

    launch_sgd(
      this.ptr,
      gradient.ptr,
      this.size,
      learningRate
    );
  }

  download(): Float32Array {
    const host = new Float32Array(this.size);
    oshpyt_download(this.ptr, host, this.size);
    return host;
  }

  destroy(): void {
    if (this.ptr) {
      oshpyt_free(this.ptr);
      this.ptr = null;
    }
  }
}

oshpyt_init();