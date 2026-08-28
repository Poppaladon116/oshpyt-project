import koffi from "koffi";

const lib = koffi.load("./oshpyt_engine.dll");

const oshpyt_init = lib.func(
  "void oshpyt_init()"
);

const oshpyt_malloc = lib.func(
  "float *oshpyt_malloc(int total_elements)"
);

const oshpyt_free = lib.func(
  "void oshpyt_free(void *ptr)"
);

const oshpyt_zero = lib.func(
  "void oshpyt_zero(float *data, int size)"
);

const oshpyt_set_point = lib.func(
  "void oshpyt_set_point(float *data, int size, int index, float value)"
);

const oshpyt_upload = lib.func(
  "void oshpyt_upload(float *host_ptr, float *engine_ptr, int size)"
);

const oshpyt_download = lib.func(
  "void oshpyt_download(float *engine_ptr, float *host_ptr, int size)"
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
  "void launch_sgd(float *weights, float *gradients, int size, float learning_rate)"
);

const oshpyt_relu = lib.func(
  "void oshpyt_relu(float *data, int size)"
);

const oshpyt_relu_backward = lib.func(
  "void oshpyt_relu_backward(float *activation, float *gradient, int size)"
);

const oshpyt_softmax = lib.func(
  "void oshpyt_softmax(float *data, int size)"
);

export class Tensor {
  ptr: unknown;
  rows: number;
  cols: number;
  size: number;

  constructor(ptr: unknown, rows: number, cols: number) {
    if (!ptr) {
      throw new Error(`Tensor allocation failed for ${rows}x${cols}.`);
    }

    this.ptr = ptr;
    this.rows = rows;
    this.cols = cols;
    this.size = rows * cols;
  }

  static allocate(rows: number, cols: number): Tensor {
    if (
      !Number.isInteger(rows) ||
      !Number.isInteger(cols) ||
      rows <= 0 ||
      cols <= 0
    ) {
      throw new Error(`Invalid tensor shape: ${rows}x${cols}.`);
    }

    const totalElements = rows * cols;
    const ptr = oshpyt_malloc(totalElements);

    return new Tensor(ptr, rows, cols);
  }

  static fromArray(
    data: Float32Array,
    rows: number,
    cols: number
  ): Tensor {
    const expectedSize = rows * cols;

    if (data.length !== expectedSize) {
      throw new Error(
        `Tensor source mismatch: expected ${expectedSize}, got ${data.length}.`
      );
    }

    const tensor = Tensor.allocate(rows, cols);
    tensor.update(data);

    return tensor;
  }

  private assertAlive(): void {
    if (!this.ptr) {
      throw new Error("Tensor has already been destroyed.");
    }
  }

  zero(): void {
    this.assertAlive();
    oshpyt_zero(this.ptr, this.size);
  }

  update(data: Float32Array): void {
    this.assertAlive();

    if (data.length !== this.size) {
      throw new Error(
        `Tensor upload mismatch: expected ${this.size}, got ${data.length}.`
      );
    }

    oshpyt_upload(data, this.ptr, data.length);
  }

  setPoint(index: number, value = 1.0): void {
    this.assertAlive();

    if (
      !Number.isInteger(index) ||
      index < 0 ||
      index >= this.size
    ) {
      throw new RangeError(
        `Tensor index ${index} is outside 0-${this.size - 1}.`
      );
    }

    oshpyt_set_point(this.ptr, this.size, index, value);
  }

  matmulTo(weights: Tensor, target: Tensor): void {
    this.assertAlive();
    weights.assertAlive();
    target.assertAlive();

    if (this.cols !== weights.rows) {
      throw new Error(
        `Matmul mismatch: ${this.rows}x${this.cols} cannot multiply ` +
        `${weights.rows}x${weights.cols}.`
      );
    }

    if (
      target.rows !== this.rows ||
      target.cols !== weights.cols
    ) {
      throw new Error(
        `Matmul target mismatch: expected ${this.rows}x${weights.cols}, got ` +
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

  /*
   * Writes dW = input^T x gradOutput into this tensor.
   */
  backwardTo(input: Tensor, gradOutput: Tensor): void {
    this.assertAlive();
    input.assertAlive();
    gradOutput.assertAlive();

    if (
      input.rows !== gradOutput.rows ||
      this.rows !== input.cols ||
      this.cols !== gradOutput.cols
    ) {
      throw new Error(
        `Weight-gradient mismatch: input=${input.rows}x${input.cols}, ` +
        `gradOutput=${gradOutput.rows}x${gradOutput.cols}, ` +
        `gradWeights=${this.rows}x${this.cols}.`
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

  /*
   * Writes dInput = gradOutput x weights^T into this tensor.
   */
  gradInputTo(weights: Tensor, gradOutput: Tensor): void {
    this.assertAlive();
    weights.assertAlive();
    gradOutput.assertAlive();

    if (
      gradOutput.cols !== weights.cols ||
      this.rows !== gradOutput.rows ||
      this.cols !== weights.rows
    ) {
      throw new Error(
        `Input-gradient mismatch: gradOutput=${gradOutput.rows}x${gradOutput.cols}, ` +
        `weights=${weights.rows}x${weights.cols}, ` +
        `gradInput=${this.rows}x${this.cols}.`
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

  relu(): void {
    this.assertAlive();
    oshpyt_relu(this.ptr, this.size);
  }

  reluBackwardFrom(activation: Tensor): void {
    this.assertAlive();
    activation.assertAlive();

    if (this.size !== activation.size) {
      throw new Error(
        `Leaky-ReLU backward mismatch: gradient=${this.size}, activation=${activation.size}.`
      );
    }

    oshpyt_relu_backward(
      activation.ptr,
      this.ptr,
      this.size
    );
  }

  softmax(): void {
    this.assertAlive();
    oshpyt_softmax(this.ptr, this.size);
  }

  optimizerStep(gradient: Tensor, learningRate: number): void {
    this.assertAlive();
    gradient.assertAlive();

    if (this.size !== gradient.size) {
      throw new Error(
        `SGD mismatch: weights=${this.size}, gradients=${gradient.size}.`
      );
    }

    if (!Number.isFinite(learningRate) || learningRate <= 0) {
      throw new Error(`Invalid learning rate: ${learningRate}.`);
    }

    launch_sgd(
      this.ptr,
      gradient.ptr,
      this.size,
      learningRate
    );
  }

  download(): Float32Array {
    this.assertAlive();

    const hostData = new Float32Array(this.size);

    oshpyt_download(
      this.ptr,
      hostData,
      this.size
    );

    return hostData;
  }

  destroy(): void {
    if (this.ptr) {
      oshpyt_free(this.ptr);
      this.ptr = null;
    }
  }
}

oshpyt_init();