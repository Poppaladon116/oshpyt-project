#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

extern "C" {
__declspec(dllexport) void oshpyt_add(float* A, float* B, int size) {
    #pragma omp parallel for
    for(int i = 0; i < size; i++) A[i] += B[i];
}
    
__declspec(dllexport)
void oshpyt_init() {
    printf("OSHPYT: CPU Engine v2.1 Active.\n");
}

__declspec(dllexport)
float* oshpyt_malloc(int total_elements) {
    if (total_elements <= 0) {
        return nullptr;
    }

    return static_cast<float*>(
        malloc(static_cast<size_t>(total_elements) * sizeof(float))
    );
}

__declspec(dllexport)
void oshpyt_free(void* ptr) {
    if (ptr != nullptr) {
        free(ptr);
    }
}

__declspec(dllexport)
void oshpyt_zero(float* data, int size) {
    if (data != nullptr && size > 0) {
        memset(
            data,
            0,
            static_cast<size_t>(size) * sizeof(float)
        );
    }
}

__declspec(dllexport)
void oshpyt_set_point(
    float* data,
    int size,
    int index,
    float value
) {
    if (data != nullptr && index >= 0 && index < size) {
        data[index] = value;
    }
}

__declspec(dllexport)
void oshpyt_upload(
    const float* host_ptr,
    float* engine_ptr,
    int size
) {
    if (host_ptr != nullptr && engine_ptr != nullptr && size > 0) {
        memcpy(
            engine_ptr,
            host_ptr,
            static_cast<size_t>(size) * sizeof(float)
        );
    }
}

__declspec(dllexport)
void oshpyt_download(
    const float* engine_ptr,
    float* host_ptr,
    int size
) {
    if (engine_ptr != nullptr && host_ptr != nullptr && size > 0) {
        memcpy(
            host_ptr,
            engine_ptr,
            static_cast<size_t>(size) * sizeof(float)
        );
    }
}

/*
 * C[m, n] = A[m, k] x B[k, n]
 */
__declspec(dllexport)
void oshpyt_matmul_to(
    const float* A,
    const float* B,
    float* C,
    int m,
    int n,
    int k
) {
    if (
        A == nullptr ||
        B == nullptr ||
        C == nullptr ||
        m <= 0 ||
        n <= 0 ||
        k <= 0
    ) {
        return;
    }

    #pragma omp parallel for
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            float sum = 0.0f;

            for (int inner = 0; inner < k; inner++) {
                sum += A[row * k + inner] * B[inner * n + col];
            }

            C[row * n + col] = sum;
        }
    }
}

/*
 * gradInput[m, k] = gradOutput[m, n] x weights[k, n]^T
 */
__declspec(dllexport)
void oshpyt_grad_input_to(
    const float* grad_output,
    const float* weights,
    float* grad_input,
    int m,
    int n,
    int k
) {
    if (
        grad_output == nullptr ||
        weights == nullptr ||
        grad_input == nullptr ||
        m <= 0 ||
        n <= 0 ||
        k <= 0
    ) {
        return;
    }

    #pragma omp parallel for
    for (int row = 0; row < m; row++) {
        for (int input_col = 0; input_col < k; input_col++) {
            float sum = 0.0f;

            for (int output_col = 0; output_col < n; output_col++) {
                sum +=
                    grad_output[row * n + output_col] *
                    weights[input_col * n + output_col];
            }

            grad_input[row * k + input_col] = sum;
        }
    }
}

/*
 * gradWeights[k, n] = input[m, k]^T x gradOutput[m, n]
 */
__declspec(dllexport)
void oshpyt_weight_grad_to(
    const float* input,
    const float* grad_output,
    float* grad_weights,
    int m,
    int n,
    int k
) {
    if (
        input == nullptr ||
        grad_output == nullptr ||
        grad_weights == nullptr ||
        m <= 0 ||
        n <= 0 ||
        k <= 0
    ) {
        return;
    }

    #pragma omp parallel for
    for (int input_col = 0; input_col < k; input_col++) {
        for (int output_col = 0; output_col < n; output_col++) {
            float sum = 0.0f;

            for (int row = 0; row < m; row++) {
                sum +=
                    input[row * k + input_col] *
                    grad_output[row * n + output_col];
            }

            grad_weights[input_col * n + output_col] = sum;
        }
    }
}

__declspec(dllexport)
void launch_sgd(
    float* weights,
    const float* gradients,
    int size,
    float learning_rate
) {
    if (
        weights == nullptr ||
        gradients == nullptr ||
        size <= 0 ||
        !isfinite(learning_rate)
    ) {
        return;
    }

    #pragma omp parallel for
    for (int index = 0; index < size; index++) {
        weights[index] -= learning_rate * gradients[index];
    }
}

/*
 * Leaky ReLU:
 * output = input              when input > 0
 * output = 0.01 * input       when input <= 0
 */
__declspec(dllexport)
void oshpyt_relu(float* data, int size) {
    if (data == nullptr || size <= 0) {
        return;
    }

    constexpr float NEGATIVE_SLOPE = 0.01f;

    #pragma omp parallel for
    for (int index = 0; index < size; index++) {
        if (data[index] <= 0.0f) {
            data[index] *= NEGATIVE_SLOPE;
        }
    }
}

/*
 * In-place gradient update for Leaky ReLU.
 *
 * activation must contain the forward output AFTER oshpyt_relu().
 * For output > 0: derivative is 1.0
 * For output <= 0: derivative is 0.01
 */
extern "C" __declspec(dllexport)
void oshpyt_relu_backward(
    float* pre_activation,
    float* gradient,
    int size
) {
    for (int i = 0; i < size; ++i) {
        if (pre_activation[i] <= 0.0f) {
            gradient[i] = 0.0f;
        }
    }
}

    constexpr float NEGATIVE_SLOPE = 0.01f;

    #pragma omp parallel for
    for (int index = 0; index < size; index++) {
        gradient[index] *=
            activation[index] > 0.0f ? 1.0f : NEGATIVE_SLOPE;
    }
}

__declspec(dllexport)
void oshpyt_softmax(float* data, int size) {
    if (data == nullptr || size <= 0) {
        return;
    }

    float max_value = data[0];

    for (int index = 1; index < size; index++) {
        if (data[index] > max_value) {
            max_value = data[index];
        }
    }

    float sum = 0.0f;

    for (int index = 0; index < size; index++) {
        data[index] = expf(data[index] - max_value);
        sum += data[index];
    }

    if (!isfinite(sum) || sum <= 0.0f) {
        const float uniform_probability =
            1.0f / static_cast<float>(size);

        for (int index = 0; index < size; index++) {
            data[index] = uniform_probability;
        }

        return;
    }

    for (int index = 0; index < size; index++) {
        data[index] /= sum;
    }
}

}