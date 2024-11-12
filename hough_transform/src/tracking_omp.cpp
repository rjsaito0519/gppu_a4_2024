#include <vector>
#include <cmath>
#include <chrono>
#include <iostream>
#include <omp.h>
#include <TVector3.h>

std::vector<std::vector<int>> tracking_openmp(const std::vector<TVector3>& pos_container, std::vector<int>& duration_container) {
    // -- Initial setup ----------
    int max_iter = pos_container.size();
    std::vector<int> track_id_container(max_iter, -1);
    std::vector<std::vector<int>> indices(10);
    int track_id = 0;

    while (std::count(track_id_container.begin(), track_id_container.end(), -1) > 5 && track_id < 10) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // -- Data preparation ----- 
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
        int n_rho = 2 * static_cast<int>(std::ceil(most_far_position * std::sqrt(2.0))) + 1;

        // Initialize Hough space and local spaces for each thread
        std::vector<int> hough_space(n_rho * 181, 0);
        std::vector<std::vector<int>> local_hough_spaces(omp_get_max_threads(), std::vector<int>(n_rho * 181, 0));

        // Parallel processing of Hough transform using OpenMP
        auto start_time3 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for
        for (int index = 0; index < data_size; ++index) {
            int thread_id = omp_get_thread_num();
            float x = host_x_data[index];
            float z = host_z_data[index];

            for (int theta = 0; theta <= 180; ++theta) {
                float radian = theta * M_PI / 180.0;
                int rho = static_cast<int>(round(z * cosf(radian) + x * sinf(radian)) + (n_rho - 1) / 2);
                local_hough_spaces[thread_id][theta * n_rho + rho] += 1;
            }
        }
        auto end_time3 = std::chrono::high_resolution_clock::now();
        auto duration3 = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time3 - start_time3).count();
        std::cout << "Hough transform (OpenMP): " << duration3 << " ns" << std::endl;

        // Aggregate local spaces of each thread into the global Hough space
        for (int t = 0; t < omp_get_max_threads(); ++t) {
            for (int i = 0; i < n_rho * 181; ++i) {
                hough_space[i] += local_hough_spaces[t][i];
            }
        }

        // Find the maximum value
        int max_value = -1;
        int max_index = -1;

        #pragma omp parallel
        {
            int local_max = -1;
            int local_index = -1;

            // Each thread finds the maximum in its assigned part
            #pragma omp for nowait
            for (int i = 0; i < hough_space.size(); ++i) {
                if (hough_space[i] > local_max) {
                    local_max = hough_space[i];
                    local_index = i;
                }
            }

            // Update the global maximum value and index in a critical section
            #pragma omp critical
            {
                if (local_max > max_value) {
                    max_value = local_max;
                    max_index = local_index;
                }
                else if (local_max == max_value && local_index < max_index) {
                    max_index = local_index;
                }
            }
        }

        int max_theta = max_index / n_rho;
        int max_rho = max_index % n_rho - static_cast<int>((n_rho - 1) / 2);
        std::cout << "Max theta: " << max_theta << ", Max rho: " << max_rho << std::endl;

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

        // Measure the total time for this iteration
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
        std::cout << "Total iteration time (OpenMP): " << duration << " ns" << std::endl;
        duration_container.push_back(duration);
    }

    return indices;
}