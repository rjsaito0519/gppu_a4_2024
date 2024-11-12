#include <vector>
#include <cmath>
#include <chrono>
#include <iostream>
#include <TVector3.h>
#include <TH2D.h>
#include <TMath.h>
#include "tracking_cpu.h"

std::vector<std::vector<int>> tracking_cpu(const std::vector<TVector3>& pos_container, std::vector<int>& duration_container) {
    // -- prepare ----------
    int max_iter = pos_container.size();
    std::vector<int> track_id_container(max_iter, -1);
    std::vector<std::vector<int>> indices(10);
    int n_rho = 2 * static_cast<int>(std::ceil(250.0 * std::sqrt(2.0))) + 1;
    int track_id = 0;

    while (std::count(track_id_container.begin(), track_id_container.end(), -1) > 5 && track_id < 10) {
        auto start_time = std::chrono::high_resolution_clock::now();

        // -- prepare data -----
        TH2D h_hough("for_hough_transform", ";theta (deg.); r (mm)", 180, -0.5, 180.0 - 0.5, n_rho, -1.0 * std::ceil(250.0 * std::sqrt(2.0)), std::ceil(250.0 * std::sqrt(2.0)));
        for (int i = 0; i < max_iter; i++) {
            if (track_id_container[i] == -1) {
                for (int theta = 0; theta <= 180; theta++) {
                    double radian = theta * M_PI / 180.0;
                    double rho = pos_container[i].Z() * std::cos(radian) + pos_container[i].X() * std::sin(radian);
                    h_hough.Fill(theta, rho);
                }
            }
        }

        auto start_time1 = std::chrono::high_resolution_clock::now();
        int max_global_bin = h_hough.GetMaximumBin();
        auto end_time1 = std::chrono::high_resolution_clock::now();
        auto duration1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time1 - start_time1).count();
        // std::cout << "h_hough.GetMaximumBin(): " << duration1 << " ns" << std::endl;

        int max_x_bin, max_y_bin, max_z_bin;
        h_hough.GetBinXYZ(max_global_bin, max_x_bin, max_y_bin, max_z_bin);
        double max_theta = h_hough.GetXaxis()->GetBinCenter(max_x_bin);
        double max_rho = h_hough.GetYaxis()->GetBinCenter(max_y_bin);

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

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
        std::cout << "total (cpu): " << duration << " ns" << std::endl;
        duration_container.push_back(duration);
    }

    return indices;
}
