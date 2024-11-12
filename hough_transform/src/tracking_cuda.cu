#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <chrono>
#include <iostream>
#include <TVector3.h>
#include "tracking_cuda.h"

__global__ void houghTransformKernel(int *hough_space, const float *x_data, const float *z_data, int data_size, int n_rho) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < data_size) {
        float x = x_data[index];
        float z = z_data[index];

        for (int theta = 0; theta <= 180; ++theta) {
            float radian = theta * M_PI / 180.0;
            int rho = static_cast<int>(round(z * cosf(radian) + x * sinf(radian)) + (n_rho - 1) / 2);
            atomicAdd(&hough_space[theta * n_rho + rho], 1);
        }
    }
}

std::vector<std::vector<int>> tracking_cuda(const std::vector<TVector3>& pos_container) {
    int max_iter = pos_container.size();
    std::vector<int> track_id_container(max_iter, -1);
    std::vector<std::vector<int>> indices(10);
    int track_id = 0;

    while (std::count(track_id_container.begin(), track_id_container.end(), -1) > 5 && track_id < 10) {
        auto start_time = std::chrono::high_resolution_clock::now();

        std::vector<float> host_x_data, host_z_data;
        float most_far_position = 0.0;
        for (int i = 0; i < max_iter; i++) {
            if (track_id_container[i] == -1) {
                host_x_data.push_back(pos_container[i].X());
                host_z_data.push_back(pos_container[i].Z());
                if (std::abs(pos_container[i].X()) > most_far_position || std::abs(pos_container[i].Z()) > most_far_position) {
                    most_far_position = std::max(std::abs(pos_container[i].X()), std::abs(pos_container[i].Z()));
                }
            }
        }

        int data_size = host_x_data.size();
        float *cuda_x_data, *cuda_z_data;
        int *cuda_hough_space;
        int n_rho = 2 * static_cast<int>(std::ceil(most_far_position * std::sqrt(2.0))) + 1;

        cudaMalloc(&cuda_x_data, data_size * sizeof(float));
        cudaMalloc(&cuda_z_data, data_size * sizeof(float));
        cudaMalloc(&cuda_hough_space, 181 * n_rho * sizeof(int));

        cudaMemcpy(cuda_x_data, host_x_data.data(), data_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_z_data, host_z_data.data(), data_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(cuda_hough_space, 0, 181 * n_rho * sizeof(int));

        int threadsPerBlock = 256;
        int blocksPerGrid = (data_size + threadsPerBlock - 1) / threadsPerBlock;

        houghTransformKernel<<<blocksPerGrid, threadsPerBlock>>>(cuda_hough_space, cuda_x_data, cuda_z_data, data_size, n_rho);

        std::vector<int> host_hough_space(181 * n_rho);
        cudaMemcpy(host_hough_space.data(), cuda_hough_space, 181 * n_rho * sizeof(int), cudaMemcpyDeviceToHost);

        cudaFree(cuda_x_data);
        cudaFree(cuda_z_data);
        cudaFree(cuda_hough_space);

        auto max_it = std::max_element(host_hough_space.begin(), host_hough_space.end());
        int max_index = std::distance(host_hough_space.begin(), max_it);
        int max_theta = max_index / n_rho;
        int max_rho = max_index % n_rho - static_cast<int>((n_rho - 1) / 2);

        double bin_diff;
        int max_diff = 4;
        for (int i = 0; i < max_iter; i++) {
            if (track_id_container[i] != -1) continue;
            bool within_circ = false;
            for (int theta = max_theta - max_diff; theta <= max_theta + max_diff; theta++) {
                double rho = std::cos(theta * M_PI / 180.0) * pos_container[i].Z() + std::sin(theta * M_PI / 180.0) * pos_container[i].X();
                double diff = std::abs(max_rho - rho) + std::abs(max_theta - theta);
                if (diff < max_diff) within_circ = true;
            }
            if (within_circ) {
                track_id_container[i] = track_id;
                indices[track_id].push_back(i);
            }
        }
        track_id++;
    }

    return indices;
}
