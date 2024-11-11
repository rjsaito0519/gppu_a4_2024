#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm> // std::max_element, std::count
#include <cuda_runtime.h> // CUDA runtime
#include <fstream> // std::ifstream
#include <cstdio> // std::remove
#include <chrono> // std::chrono
#include <TFile.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <TH1.h>
#include <TCanvas.h>
#include <TF1.h>
#include <TApplication.h>
#include <TMath.h>
#include <TGraph.h>
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
            atomicAdd(&hough_space[theta * n_rho + rho], 1);
        }
    }
}

std::vector<std::vector<int>> tracking(const std::vector<TVector3>& pos_container)
{
    // --  prepare ----------
    int max_iter = pos_container.size();

    // --  search track ----------
    std::vector<int> track_id_container(max_iter, -1);
    std::vector<std::vector<int>> indices(10);
    int track_id = 0;
    while ( std::count(track_id_container.begin(), track_id_container.end(), -1) > 5 && track_id < 10) {
        auto start_time = std::chrono::high_resolution_clock::now();
        // -- prepare data -----
        std::vector<double> host_x_data, host_z_data;
        double most_far_position = 0.0;
        for (int i = 0; i < max_iter; i++) if ( track_id_container[i] == -1 ) {
            host_x_data.push_back(pos_container[i].X());
            host_z_data.push_back(pos_container[i].Z());
            if (std::abs(pos_container[i].X()) > most_far_position || std::abs(pos_container[i].Z()) > most_far_position) {
                most_far_position = (std::abs(pos_container[i].X()) > std::abs(pos_container[i].Z())) ? std::abs(pos_container[i].X()) : std::abs(pos_container[i].Z());
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

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
        std::cout << duration << std::endl;

        auto max_it = std::max_element(host_hough_space.begin(), host_hough_space.end());
        int max_index = std::distance(host_hough_space.begin(), max_it);
        int max_theta = max_index / n_rho;
        int max_rho   = max_index % n_rho - static_cast<int>((n_rho-1)/2);

        // -- event selection ----------
        double bin_diff;
        int max_diff = 4;
        for (int i = 0; i < max_iter; i++) {
            if ( track_id_container[i] != -1 ) continue;
            bool within_circ = false;
            for (int theta = max_theta-max_diff; theta <= max_theta+max_diff; theta++) {
                double rho = std::cos( theta*TMath::DegToRad() )*pos_container[i].Z() + std::sin( theta*TMath::DegToRad() )*pos_container[i].X();
                double diff = TMath::Abs( max_rho-rho ) + TMath::Abs( max_theta - theta );
                if ( diff < max_diff ) within_circ = true;
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


    // +--------------------------+
    // | prepare output root file |
    // +--------------------------+
    TString save_name;
    int dot_index = root_file_path.Last('.');
    int sla_index = root_file_path.Last('/');
    for (int i = sla_index+1; i < dot_index; i++) save_name += root_file_path[i];
    TString output_path = Form("./results/%s_output.root", save_name.Data());
    if (std::ifstream(output_path.Data())) std::remove(output_path.Data());
    TFile fout(output_path.Data(), "create");
    TTree tracking_tree("tree", ""); 

    // -- prepare root file branch -----
    std::vector<int> pad_id;
    std::vector<double> pos_x, pos_y, pos_z, adc;
    double a_pos, b_pos, a_time, b_time, track_length, redchi2_pos, redchi2_time, sum_de, angle_pos, angle_time, dedx;
    int hit_num, evnum_buf;

    tracking_tree.Branch("evnum", &evnum_buf, "evnum/I");
    tracking_tree.Branch("nhit", &hit_num, "nhit/I");
    tracking_tree.Branch("a_pos", &a_pos, "a_pos/D");
    tracking_tree.Branch("b_pos", &b_pos, "b_pos/D");
    tracking_tree.Branch("a_time", &a_time, "a_time/D");
    tracking_tree.Branch("b_time", &b_time, "b_time/D");
    tracking_tree.Branch("sum_de", &sum_de, "desum/D");
    tracking_tree.Branch("length", &track_length, "length/D");
    tracking_tree.Branch("dedx", &dedx, "dedx/D");
    tracking_tree.Branch("redchi2_pos", &redchi2_pos, "redchi2_pos/D");
    tracking_tree.Branch("redchi2_time", &redchi2_time, "redchi2_time/D");
    tracking_tree.Branch("angle_pos", &angle_pos, "angle_pos/D");
    tracking_tree.Branch("angle_time", &angle_time, "angle_time/D");
    tracking_tree.Branch("adc", &adc);
    tracking_tree.Branch("pad_id", &pad_id);


    // +---------------------------------+
    // | detect track by hough transform |
    // +---------------------------------+
    int nhit_threshold = 0;
    double diff_threshold = 30.;
    reader.Restart();
    while (reader.Next()) { displayProgressBar( *evnum+1, total_entry);
        evnum_buf = *evnum;

        std::vector<TVector3> pos_container;
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
                pos_container.push_back( pad_center_pos );
            }
        }
        if ( pos_container.size() == 0) continue;
        
        // -- tracking and cal dedx -----
        // auto start_time = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<int>> indices = tracking(pos_container);
        // auto end_time = std::chrono::high_resolution_clock::now();
        // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        // std::cout << duration << std::endl;

        for (Int_t track_id = 0; track_id < 10; track_id++ ) {
            int hit_num = indices[track_id].size();
            if (hit_num == 0) continue;
            
            // -- initialize ---------
            pos_x.clear();
            pos_y.clear();
            pos_z.clear();
            adc.clear();
            pad_id.clear();
            track_length = 0.0, sum_de = 0.0;

            // -- check each track ---------
            for (const auto index : indices[track_id]) {
                pos_x.push_back( pos_container[index].X() );
                pos_y.push_back( pos_container[index].Y() );
                pos_z.push_back( pos_container[index].Z() );
                pad_id.push_back(pad_and_de[index].first);
                adc.push_back(pad_and_de[index].second);
                sum_de += pad_and_de[index].second;
            }

            // -- fit xy (time) plain ----------
            auto *g_time = new TGraph(pos_x.size(), &pos_x[0], &pos_y[0]);
            auto *f_time = new TF1("fit_f_time", "[0]*x + [1]", -250.0, 250.0);
            g_time->Fit("fit_f_time", "Q", "", -250.0, 250.0);
            a_time = f_time->GetParameter(0);
            b_time = f_time->GetParameter(1);
            redchi2_time = f_time->GetChisquare() / f_time->GetNDF();
            angle_time = TMath::ATan(a_time)*TMath::RadToDeg();

            // -- fit xz (position) plain ----------
            auto *g_pos = new TGraph(pos_x.size(), &pos_x[0], &pos_z[0]);
            auto *f_pos = new TF1( "fit_f_pos", "[0]*x + [1]", -250.0, 250.0);
            g_pos->Fit("fit_f_pos", "Q", "", -250.0, 250.0);
            a_pos = f_pos->GetParameter(0);
            b_pos = f_pos->GetParameter(1);
            redchi2_pos = f_pos->GetChisquare() / f_pos->GetNDF();
            angle_pos = TMath::ATan(a_pos)*TMath::RadToDeg();
            
            // -- calc. dedx ----------
            TVector3 pos_origin(-250.0, 0.0, -250.0*a_pos+b_pos);
            TVector3 vec_direc(1.0, 0.0, a_pos);

            std::vector<double> distance_from_origin;
            for (const auto index : indices[track_id]) distance_from_origin.push_back( vec_direc.Dot( pos_container[index] - pos_origin ) / vec_direc.Mag() );
            std::sort(distance_from_origin.begin(), distance_from_origin.end());
            for (Int_t i = 1, n = distance_from_origin.size(); i < n; i++) {
                Double_t diff = distance_from_origin[i]-distance_from_origin[i-1];
                if (diff < diff_threshold) track_length += diff;
            }
            dedx = sum_de/track_length;

            // -- fill branch -----
            tracking_tree.Fill();

            // -- delete gr ---------- 
            delete g_time;
            delete f_time;
            delete g_pos;
            delete f_pos;
        }
        // if (*evnum >= 500) break;
    }

    // +------------+
    // | Write data |
    // +------------+
    tracking_tree.Write();
    fout.Close(); 

    return 0;
}