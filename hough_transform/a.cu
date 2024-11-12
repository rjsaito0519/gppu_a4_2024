#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <chrono>
#include <iostream>
#include <omp.h>
#include <TVector3.h>
#include "config.h"
#include "tracking_cuda.h"


// CUDAカーネルの定義
__global__ void houghTransformKernel(int *hough_space, const double *x_data, const double *z_data, int data_size, int n_rho) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < data_size) {
        double x = x_data[index];
        double z = z_data[index];

        for (int theta = 0; theta <= 180; ++theta) {
            double radian = theta * M_PI / 180.0;
            int rho = static_cast<int>(round(z * cosf(radian) + x * sinf(radian)) + (n_rho - 1) / 2);
            atomicAdd(&hough_space[theta * n_rho + rho], 1);
        }
    }
}

std::vector<std::vector<int>> tracking_cuda(const std::vector<TVector3>& pos_container, std::vector<int>& duration_container) {
    // --  prepare ----------
    int max_iter = pos_container.size();
    std::vector<int> track_id_container(max_iter, -1);
    std::vector<std::vector<int>> indices(10);
    int track_id = 0;

    while (std::count(track_id_container.begin(), track_id_container.end(), -1) > 5 && track_id < 10) {
        
        // -- prepare data -----
        std::vector<double> host_x_data, host_z_data;
        double most_far_position = 0.0;
        for (int i = 0; i < max_iter; i++) {
            if (track_id_container[i] == -1) {
                host_x_data.push_back(pos_container[i].X());
                host_z_data.push_back(pos_container[i].Z());
                if (std::abs(pos_container[i].X()) > most_far_position || std::abs(pos_container[i].Z()) > most_far_position) {
                    most_far_position = std::max(std::abs(pos_container[i].X()), std::abs(pos_container[i].Z()));
                }
            }
        }

        // Allocate CUDA device memory
        int data_size = host_x_data.size();
        double *cuda_x_data, *cuda_z_data;
        int *cuda_hough_space;
        int n_rho = 2 * static_cast<int>(std::ceil(most_far_position * std::sqrt(2.0))) + 1;
        cudaMalloc(&cuda_x_data, data_size * sizeof(double));
        cudaMalloc(&cuda_z_data, data_size * sizeof(double));
        cudaMalloc(&cuda_hough_space, 181 * n_rho * sizeof(int));

        // Copy data from host to device
        cudaMemcpy(cuda_x_data, host_x_data.data(), data_size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_z_data, host_z_data.data(), data_size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemset(cuda_hough_space, 0, 181 * n_rho * sizeof(int));

        // Launch the kernel
        int threads_per_block = conf.cuda_n_threads;
        int blocks_per_grid = (data_size + threads_per_block - 1) / threads_per_block;
        houghTransformKernel<<<blocks_per_grid, threads_per_block>>>(cuda_hough_space, cuda_x_data, cuda_z_data, data_size, n_rho);
        cudaDeviceSynchronize();

        // Copy the result to the host
        std::vector<int> host_hough_space(181 * n_rho);
        cudaMemcpy(host_hough_space.data(), cuda_hough_space, 181 * n_rho * sizeof(int), cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(cuda_x_data);
        cudaFree(cuda_z_data);
        cudaFree(cuda_hough_space);

        auto max_it = std::max_element(host_hough_space.begin(), host_hough_space.end());
        int max_index = std::distance(host_hough_space.begin(), max_it);
        int max_theta = max_index / n_rho;
        int max_rho = max_index % n_rho - static_cast<int>((n_rho - 1) / 2);


        auto start_time7 = std::chrono::high_resolution_clock::now();        
        // OpenMPを使用して最大値とインデックスを探索
        int max_value = -1;
        int max_index = -1;
        // Prepare local maximums as pairs (value, index) for each thread
        std::vector<std::pair<int, int>> local_max_pairs(omp_get_max_threads(), std::make_pair(-1, -1));

        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int local_max = -1;
            int local_index = -1;

            // Each thread finds the maximum value in its assigned range
            #pragma omp for nowait
            for (int i = 0; i < host_hough_space.size(); ++i) {
                if (host_hough_space[i] > local_max) {
                    local_max = host_hough_space[i];
                    local_index = i;
                }
            }

            // Store the local max value and index as a pair for this thread
            local_max_pairs[thread_id] = std::make_pair(local_max, local_index);
        }

        // Final aggregation to find the overall maximum and index
        for (const auto& local_pair : local_max_pairs) {
            if (local_pair.first > max_value) {
                max_value = local_pair.first;
                max_index = local_pair.second;
            }
            // If the maximum values are the same, keep the smaller index
            else if (local_pair.first == max_value && local_pair.second < max_index) {
                max_index = local_pair.second;
            }
        }

        int max_theta = max_index / n_rho;
        int max_rho   = max_index % n_rho - static_cast<int>((n_rho-1)/2);
        auto end_time7 = std::chrono::high_resolution_clock::now();
        auto duration7 = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time7 - start_time7).count();
        // std::cout << "max_element3: " << duration7 << " ns" << std::endl;
        // std::cout << "max_index, " << max_index << std::endl;


        // Event selection
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

        // 全体の計測
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
        std::cout << "total: " << duration << " ns" << std::endl;
        duration_container.push_back(duration);
    }

    return indices;
}
