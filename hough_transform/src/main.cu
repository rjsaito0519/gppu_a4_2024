#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <random>
#include <cuda_runtime.h>
#include <TFile.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <TH1.h>
#include <TCanvas.h>
#include <TF1.h>
#include <TApplication.h>
#include <TMath.h>
#include <TBox.h>

#include "progress_bar.h"
#include "fit_tTpc.h"
#include "pad_helper.h"


// CUDAカーネルの定義
__global__ void houghTransformKernel(int *hough_space, const double *x_data, const double *z_data, int data_size, int n_rho) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < data_size) {
        double x = x_data[index];
        double z = z_data[index];

        for (int theta = 0; theta <= 180; ++theta) {
            double radian = theta * M_PI / 180.0;
            int rho = static_cast<int>( round(z*cosf(radian) + x*sinf(radian)) + (n_rho-1)/2 );
            if (rho >= 0 && rho < n_rho) {
                atomicAdd(&hough_space[theta * n_rho + rho], 1);
            } else {
                printf("x = %f, z = %f, theta = %d, n_rho = %d, rho = %d\n", x, z, theta, n_rho, rho);
            }
        }
    }
}

std::vector<int> tracking(const std::vector<TVector3>& positions)
{
    // --  prepare ----------
    int max_iter = positions.size();

    // --  search track ----------
    std::vector<int> track_id_container(max_iter, -1);
    int track_id = 0;
    while ( std::count(track_id_container.begin(), track_id_container.end(), -1) > 5 && track_id < 10) {

        // -- prepare data -----
        std::vector<double> host_x_data, host_z_data;
        double most_far_position = 0.0;
        for (int i = 0; i < max_iter; i++) if ( track_id_container[i] == -1 ) {
            host_x_data.push_back(positions[i].X());
            host_z_data.push_back(positions[i].Z());
            if (std::abs(positions[i].X()) > most_far_position || std::abs(positions[i].Z()) > most_far_position) {
                most_far_position = (std::abs(positions[i].X()) > std::abs(positions[i].Z())) ? std::abs(positions[i].X()) : std::abs(positions[i].Z());
            }
        }

        // Allocate CUDA device memory
        int data_size = host_x_data.size();
        double *cuda_x_data, *cuda_z_data;
        int *cuda_hough_space;
        int n_rho = 2*static_cast<int>(std::ceil(most_far_position*std::sqrt(2.0))) + 1; // |X|, |Z| maximum values are around 250. +1 mean rho = 0
        cudaMalloc(&cuda_x_data, data_size * sizeof(double));
        cudaMalloc(&cuda_z_data, data_size * sizeof(double));
        cudaMalloc(&cuda_hough_space, 181 * n_rho * sizeof(int));

        // Copy data from host to device
        cudaMemcpy(cuda_x_data, host_x_data.data(), data_size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_z_data, host_z_data.data(), data_size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemset(cuda_hough_space, 0, 181 * n_rho * sizeof(int));

        // Launch the kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (data_size + threadsPerBlock - 1) / threadsPerBlock;
        houghTransformKernel<<<blocksPerGrid, threadsPerBlock>>>(cuda_hough_space, cuda_x_data, cuda_z_data, data_size, n_rho);

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
        int max_rho   = max_index % n_rho - static_cast<int>((n_rho-1)/2);
        // std::cout << track_id << ": " << n_rho << ", " << *max_it << ", " << host_x_data.size() << ", " << max_index << ", " << max_theta << ", " << max_rho << std::endl;

        // -- event selection ----------
        double bin_diff;
        int max_diff = 4;
        for (int i = 0; i < max_iter; i++) {
            if ( track_id_container[i] != -1 ) continue;
            bool within_circ = false;
            for (int theta = max_theta-max_diff; theta <= max_theta+max_diff; theta++) {
                double rho = std::cos( theta*TMath::DegToRad() )*positions[i].Z() + std::sin( theta*TMath::DegToRad() )*positions[i].X();
                double diff = TMath::Abs( max_rho-rho ) + TMath::Abs( max_theta - theta );
                if ( diff < max_diff ) within_circ = true;
            }
            if (within_circ) track_id_container[i] = track_id;
        }
        track_id++;
    }

    return track_id_container;
}

int main(int argc, char** argv) {

    // -- check argument ------
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_root_file>" << std::endl;
        return 1;
    }

    // +----------------+
    // | load root file |
    // +----------------+
    const TString root_file_path = argv[1];

    auto *f = new TFile(root_file_path.Data());
    if (!f || f->IsZombie()) {
        std::cerr << "Error: Could not open file : " << root_file_path << std::endl;
        return 1;
    }
    TTreeReader reader("tpc", f);
    int total_entry = reader.GetEntries();
    TTreeReaderValue<int> runnum(reader, "runnum");
    TTreeReaderValue<int> evnum(reader, "evnum");
    TTreeReaderValue<int> nhTpc(reader, "nhTpc");
    TTreeReaderValue<std::vector<int>> layerTpc(reader, "layerTpc");
    TTreeReaderValue<std::vector<int>> rowTpc(reader, "rowTpc");
    TTreeReaderValue<std::vector<double>> deTpc(reader, "deTpc");
    TTreeReaderValue<std::vector<double>> tTpc(reader, "tTpc");
    
    // +-------------------+
    // | prepare histogram |
    // +-------------------+
    auto h_tdc = new TH1D("tdc", "tdc", 2000, 0.0, 200.0);
    // auto hdedx = new TH1D("dedx", "", 400, 0, 200);
    // auto hn = new TH1D("n", "", 500, 0, 500);
    // auto halpha = new TH1D("alpha", "", 500, -10, 10);
    
    // auto hdedx_vs_redchi = new TH2D("dedx_vs_redchi", "", 400, 0, 200, 1000, 0, 1000);
    // auto hdedx_vs_n = new TH2D("dedx_vs_n", "", 400, 0, 200, 500, 0, 500);


    // +-----------------------------------+
    // | fit tdc and determine time window |
    // +-----------------------------------+
    reader.Restart();
    while (reader.Next()){
        if (*nhTpc < 400) for (int i = 0; i < *nhTpc; i++) h_tdc->Fill( (*tTpc)[i] );
    }
    std::vector<double> tdc_fit_result = fit_tTpc(h_tdc);
    double min_tdc_gate = tdc_fit_result[1] - 3.0*tdc_fit_result[2];
    double max_tdc_gate = tdc_fit_result[1] + 3.0*tdc_fit_result[2];


    // +---------------------------------+
    // | detect track by hough transform |
    // +---------------------------------+
    int nhit_threshold = 0;
    double space_threshold = 30.;

    reader.Restart();
    while (reader.Next()) { displayProgressBar( *evnum+1, total_entry);
        std::vector<TVector3> position;
        std::vector<std::pair<int, double>> pad_and_de;
        
        // -- fill -----
        if (nhit_threshold < *nhTpc) for (int i = 0; i < *nhTpc; i++) {
            int pad = padHelper::getPadID((*layerTpc)[i], (*rowTpc)[i]);
            bool isNoisy = std::binary_search(padHelper::noisy_pad.begin(), padHelper::noisy_pad.end(), pad);
            if (isNoisy) continue;
            if ( min_tdc_gate < (*tTpc)[i] && (*tTpc)[i] < max_tdc_gate ) {
                pad_and_de.emplace_back( pad, (*deTpc)[i] );
                TVector3 pad_center_pos = padHelper::getPoint(pad);
                pad_center_pos.SetY( (*tTpc)[i] );
                position.push_back( pad_center_pos );
            }
        }
        if ( position.size() == 0) continue;
        
        // -- tracking and cal dedx -----
        std::vector<int> track_id_container = tracking(position);

    }

    // // 結果の一部を表示
    // std::cout << "Hough Space (一部表示):" << std::endl;
    // for (int i = 0; i < 10; ++i) {
    //     for (int j = 0; j < 10; ++j) {
    //         std::cout << hough_space[i * max_rho + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    return 0;
}