#include <vector>
#include <cmath>
#include <chrono>
#include <iostream>
#include <omp.h>
#include "config.h"
#include <TVector3.h>

std::vector<std::vector<int>> tracking_openmp(const std::vector<TVector3>& pos_container, std::vector<int>& duration_container) {
    // -- Initial setup ----------
    int max_iter = pos_container.size();
    std::vector<int> track_id_container(max_iter, -1);
    std::vector<std::vector<int>> indices(10);
    int track_id = 0;
    int n_rho = 2 * static_cast<int>(std::ceil(250.0 * std::sqrt(2.0))) + 1;

    while (std::count(track_id_container.begin(), track_id_container.end(), -1) > 5 && track_id < 10) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Initialize Hough space and local spaces for each thread
        std::vector<int> hough_space(n_rho * 181, 0);
        std::vector<std::vector<int>> local_hough_spaces(omp_get_max_threads(), std::vector<int>(n_rho * 181, 0));

        // Parallel processing of Hough transform using OpenMP
        auto start_time1 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for
        for (int index = 0; index < max_iter; ++index) {
            int thread_id = omp_get_thread_num();
            if (track_id_container[index] == -1) {
                double x = pos_container[index].X();
                double z = pos_container[index].Z();
                for (int theta = 0; theta <= 180; ++theta) {
                    float radian = theta * M_PI / 180.0;
                    int rho = static_cast<int>(std::round(z * std::cos(radian) + x * std::sin(radian)) + (n_rho - 1) / 2);
                    local_hough_spaces[thread_id][theta * n_rho + rho] += 1;
                }
            }
        }
    
        // Aggregate local spaces using SIMD optimization
        #pragma omp parallel for
        for (int i = 0; i < n_rho * 181; ++i) {
            int sum = 0;
            #pragma omp simd reduction(+:sum)
            // #pragma omp parallel for reduction(+:sum)
            for (int t = 0; t < omp_get_max_threads(); ++t) {
                sum += local_hough_spaces[t][i];
            }
            hough_space[i] = sum;
        }
        auto end_time1 = std::chrono::high_resolution_clock::now();
        auto duration1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time1 - start_time1).count();

        
        auto start_time2 = std::chrono::high_resolution_clock::now();        
        // Find the maximum value
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
            for (int i = 0; i < hough_space.size(); ++i) {
                if (hough_space[i] > local_max) {
                    local_max = hough_space[i];
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
        auto end_time2 = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time2 - start_time2).count();

        // Event selection
        auto start_time3 = std::chrono::high_resolution_clock::now();        
        int max_diff = conf.hough_max_diff;
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
        auto end_time3 = std::chrono::high_resolution_clock::now();
        auto duration3 = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time3 - start_time3).count();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
        if (track_id == 0) {
            duration_container.push_back(duration1);
            duration_container.push_back(duration2);
            duration_container.push_back(duration3);
            duration_container.push_back(duration);
        }

        track_id++;
    }

    return indices;
}