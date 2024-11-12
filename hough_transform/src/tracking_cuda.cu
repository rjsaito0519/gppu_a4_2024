#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <chrono>
#include <iostream>
#include <omp.h>
#include <TVector3.h>
#include "tracking_cuda.h"


// CUDAカーネルの定義
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

std::vector<std::vector<int>> tracking_cuda(const std::vector<TVector3>& pos_container, std::vector<int>& duration_container) {
    // --  prepare ----------
    int max_iter = pos_container.size();
    std::vector<int> track_id_container(max_iter, -1);
    std::vector<std::vector<int>> indices(10);
    int track_id = 0;

    while (std::count(track_id_container.begin(), track_id_container.end(), -1) > 5 && track_id < 10) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // -- prepare data -----
        auto start_time0 = std::chrono::high_resolution_clock::now();
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
        auto end_time0 = std::chrono::high_resolution_clock::now();
        auto duration0 = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time0 - start_time0).count();
        std::cout << "data preparation: " << duration0 << " ns" << std::endl;

        // Allocate CUDA device memory
        int data_size = host_x_data.size();
        float *cuda_x_data, *cuda_z_data;
        int *cuda_hough_space;
        int n_rho = 2 * static_cast<int>(std::ceil(most_far_position * std::sqrt(2.0))) + 1;

        auto start_time1 = std::chrono::high_resolution_clock::now();
        cudaMalloc(&cuda_x_data, data_size * sizeof(float));
        cudaMalloc(&cuda_z_data, data_size * sizeof(float));
        cudaMalloc(&cuda_hough_space, 181 * n_rho * sizeof(int));
        auto end_time1 = std::chrono::high_resolution_clock::now();
        auto duration1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time1 - start_time1).count();
        std::cout << "cudaMalloc: " << duration1 << " ns" << std::endl;

        // Copy data from host to device
        auto start_time2 = std::chrono::high_resolution_clock::now();
        cudaMemcpy(cuda_x_data, host_x_data.data(), data_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_z_data, host_z_data.data(), data_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(cuda_hough_space, 0, 181 * n_rho * sizeof(int));
        auto end_time2 = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time2 - start_time2).count();
        std::cout << "cudaMemset: " << duration2 << " ns" << std::endl;

        // Launch the kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (data_size + threadsPerBlock - 1) / threadsPerBlock;

        auto start_time3 = std::chrono::high_resolution_clock::now();
        houghTransformKernel<<<blocksPerGrid, threadsPerBlock>>>(cuda_hough_space, cuda_x_data, cuda_z_data, data_size, n_rho);
        cudaDeviceSynchronize();
        auto end_time3 = std::chrono::high_resolution_clock::now();
        auto duration3 = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time3 - start_time3).count();
        std::cout << "houghTransformKernel: " << duration3 << " ns" << std::endl;

        // Copy the result to the host
        std::vector<int> host_hough_space(181 * n_rho);
        auto start_time4 = std::chrono::high_resolution_clock::now();
        cudaMemcpy(host_hough_space.data(), cuda_hough_space, 181 * n_rho * sizeof(int), cudaMemcpyDeviceToHost);
        auto end_time4 = std::chrono::high_resolution_clock::now();
        auto duration4 = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time4 - start_time4).count();
        std::cout << "cudaMemcpy: " << duration4 << " ns" << std::endl;

        // Free device memory
        cudaFree(cuda_x_data);
        cudaFree(cuda_z_data);
        cudaFree(cuda_hough_space);

        auto start_time5 = std::chrono::high_resolution_clock::now();
        auto max_it = std::max_element(host_hough_space.begin(), host_hough_space.end());
        int max_index = std::distance(host_hough_space.begin(), max_it);
        int max_theta = max_index / n_rho;
        int max_rho = max_index % n_rho - static_cast<int>((n_rho - 1) / 2);
        auto end_time5 = std::chrono::high_resolution_clock::now();
        auto duration5 = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time5 - start_time5).count();
        std::cout << "max_element1: " << duration5 << " ns" << std::endl;
        std::cout << "max_index, " << max_index << std::endl;



        auto start_time6 = std::chrono::high_resolution_clock::now();
        
        // OpenMPを使用して最大値とインデックスを探索
        int max_value = -1;
        max_index = -1;
        #pragma omp parallel
        {
            int local_max = -1;
            int local_index = -1;
            // 各スレッドで部分的に最大値を探す
            #pragma omp for nowait
            for (int i = 0; i < host_hough_space.size(); ++i) {
                if (host_hough_space[i] > local_max) {
                    local_max = host_hough_space[i];
                    local_index = i;
                }
            }

            // 最大値とインデックスの結果をcriticalセクションで更新
            #pragma omp critical
            {
                if (local_max > max_value) {
                    max_value = local_max;
                    max_index = local_index;
                }
                // 同じ最大値の場合、より小さいインデックスを保持
                else if (local_max == max_value && local_index < max_index) {
                    max_index = local_index;
                }
            }
        }
        int _max_theta = max_index / n_rho;
        int _max_rho   = max_index % n_rho - static_cast<int>((n_rho-1)/2);
        auto end_time6 = std::chrono::high_resolution_clock::now();
        auto duration6 = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time6 - start_time6).count();
        std::cout << "max_element2: " << duration6 << " ns" << std::endl;
        std::cout << "max_index, " << max_index << std::endl;


        auto start_time7 = std::chrono::high_resolution_clock::now();        
        // OpenMPを使用して最大値とインデックスを探索
        max_value = -1;
        max_index = -1;
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

        int _max_theta2 = max_index / n_rho;
        int _max_rho2   = max_index % n_rho - static_cast<int>((n_rho-1)/2);
        auto end_time7 = std::chrono::high_resolution_clock::now();
        auto duration7 = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time7 - start_time7).count();
        std::cout << "max_element3: " << duration7 << " ns" << std::endl;
        std::cout << "max_index, " << max_index << std::endl;


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
