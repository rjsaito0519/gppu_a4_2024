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
            atomicAdd(&hough_space[theta * n_rho + rho], 1);
        }
    }
}

std::vector<std:vector<int>> tracking(const std::vector<TVector3>& pos_container)
{
    // --  prepare ----------
    int max_iter = pos_container.size();

    // --  search track ----------
    std::vector<int> track_id_container(max_iter, -1);
    std::vector<std:vector<int>> indices(10);
    int track_id = 0;
    while ( std::count(track_id_container.begin(), track_id_container.end(), -1) > 5 && track_id < 10) {

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


        auto max_it = std::max_element(host_hough_space.begin(), host_hough_space.end());
        int max_index = std::distance(host_hough_space.begin(), max_it);
        int max_theta = max_index / n_rho;
        int max_rho   = max_index % n_rho - static_cast<int>((n_rho-1)/2);
        std::cout << track_id << ": " << n_rho << ", " << *max_it << ", " << host_x_data.size() << ", " << max_index << ", " << max_theta << ", " << max_rho << std::endl;

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
    double diff_threshold = 30.;

    // // -- prepare root file branch -----
    // std::vector<TVector3> positions, position_buf;
    // std::vector<std::pair<Int_t, Double_t>> pad_and_de;
    // std::vector<Int_t> track_ids, pad_id;
    // std::vector<Double_t> pos_x, pos_y, pos_z, distance_from_origin, adc;
    // Double_t a_pos, b_pos, a_time, b_time, track_length, redchi2_pos, redchi2_time, sum_de, angle_pos, angle_time, dedx;
    // Int_t hit_num, evnum_buf;
    // TVector3 pos_origin, vec_direc;

    // tracking_tree.Branch("totnum", &tot_num, "totnum/I");
    // tracking_tree.Branch("evnum", &evnum_buf, "evnum/I");
    // tracking_tree.Branch("nhit", &hit_num, "nhit/I");
    // tracking_tree.Branch("a_pos", &a_pos, "a_pos/D");
    // tracking_tree.Branch("b_pos", &b_pos, "b_pos/D");
    // tracking_tree.Branch("a_time", &a_time, "a_time/D");
    // tracking_tree.Branch("b_time", &b_time, "b_time/D");
    // tracking_tree.Branch("desum", &sum_de, "desum/D");
    // tracking_tree.Branch("length", &track_length, "length/D");
    // tracking_tree.Branch("dedx", &dedx, "dedx/D");
    // tracking_tree.Branch("redchi2_pos", &redchi2_pos, "redchi2_pos/D");
    // tracking_tree.Branch("redchi2_time", &redchi2_time, "redchi2_time/D");
    // tracking_tree.Branch("angle_pos", &angle_pos, "angle_pos/D");
    // tracking_tree.Branch("angle_time", &angle_time, "angle_time/D");
    // tracking_tree.Branch("adc", &adc);
    // tracking_tree.Branch("pad_id", &pad_id);

    reader.Restart();
    while (reader.Next()) { displayProgressBar( *evnum+1, total_entry);
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
        std::vector<std::vector<int>> indices = tracking(pos_container);
        for (Int_t track_id = 0; track_id < 10; track_id++ ) {
            int hit_num = indices[track_id].size();
            if (hit_num == 0) continue;
            
            // -- initialize ---------
            // pos_x.clear();
            // pos_y.clear();
            // pos_z.clear();
            // distance_from_origin.clear();
            // position_buf.clear();
            // adc.clear();
            // pad_id.clear();
            double track_length = 0.0, sum_de = 0.0;

            std::vector<double> pos_x, pos_y, pos_z;

            // -- check each track ---------
            for (const auto index : indices[track_id]) {
                pos_x.push_back( pos_container[index].X() );
                pos_y.push_back( pos_container[index].Y() );
                pos_z.push_back( pos_container[index].Z() );
                // position_buf.push_back( pos_container[i] );
                // pad_id.push_back(pad_and_de[i].first);
                // adc.push_back(pad_and_de[i].second);
                sum_de += pad_and_de[index].second;
            }

            // -- fit xy (time) plain ----------
            auto *g_time = new TGraph(pos_x.size(), &pos_x[0], &pos_y[0]);
            auto *f_time = new TF1("fit_f_time", "[0]*x + [1]", -250.0, 250.0);
            g_time->Fit("fit_f_time", "Q", "", -250.0, 250.0);
            double a_time = f_time->GetParameter(0);
            double b_time = f_time->GetParameter(1);
            double redchi2_time = f_time->GetChisquare() / f_time->GetNDF();
            double angle_time = TMath::ATan(a_time)*TMath::RadToDeg();

            // -- fit xz (position) plain ----------
            auto *g_pos = new TGraph(pos_x.size(), &pos_x[0], &pos_z[0]);
            auto *f_pos = new TF1( "fit_f_pos", "[0]*x + [1]", -250.0, 250.0);
            g_pos->Fit("fit_f_pos", "Q", "", -250.0, 250.0);
            double a_pos = f_pos->GetParameter(0);
            double b_pos = f_pos->GetParameter(1);
            double redchi2_pos = f_pos->GetChisquare() / f_pos->GetNDF();
            double angle_pos = TMath::ATan(a_pos)*TMath::RadToDeg();
            
            // -- calc. dedx ----------
            TVector3 pos_origin(-250.0, 0.0, -250.0*a_pos+b_pos);
            TVector3 vec_direc(1.0, 0.0, a_pos);

            for (const auto index : indices[track_id]) distance_from_origin.push_back( vec_direc.Dot( pos_container[index] - pos_origin ) / vec_direc.Mag() );
            std::sort(distance_from_origin.begin(), distance_from_origin.end());
            for (Int_t i = 1, n = distance_from_origin.size(); i < n; i++) {
                Double_t diff = distance_from_origin[i]-distance_from_origin[i-1];
                if (diff < diff_threshold) track_length += diff;
            }
            double dedx = sum_de/track_length;

            // // -- fill data to histo ----------
            // hdedx->Fill( sum_de/track_length );
            // hn->Fill( pos_x.size() );
            // halpha->Fill( a_pos );
            // hdedx_vs_redchi->Fill( sum_de/track_length, redchi2_pos );
            // hdedx_vs_n->Fill( sum_de/track_length, pos_x.size() );
            // tracking_tree.Fill();

            // -- delete gr ---------- 
            delete g_time;
            delete f_time;
            delete g_pos;
            delete f_pos;
        }
        if (*evnum >= 500) break;

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