#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm> // std::max_element, std::count
#include <cuda_runtime.h> // CUDA runtime
#include <fstream> // std::ifstream
#include <cstdio> // std::remove
#include <chrono> // std::chrono
#include <nlohmann/json.hpp>

#include <omp.h>

#include <TFile.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <TH1.h>
#include <TH2.h>
#include <TCanvas.h>
#include <TF1.h>
#include <TApplication.h>
#include <TMath.h>
#include <TGraph.h>
#include <TBox.h>
#include <TVector3.h>

#include "config.h"
#include "progress_bar.h"
#include "fit_tTpc.h"
#include "pad_helper.h"
#include "tracking_cuda.h"
#include "tracking_cpu.h"
#include "tracking_omp.h"


config conf;

void load_config(const std::string& config_file) {
    std::ifstream file(config_file);
    if (!file) {
        std::cerr << "Error: Could not open config file: " << config_file << std::endl;
        exit(1);
    }
    
    nlohmann::json json_config;
    file >> json_config;
    
    conf.root_file_path = TString(json_config.at("root_file_path").get<std::string>());
    conf.which_method   = json_config.at("which_method").get<std::string>();
    conf.omp_n_threads  = json_config.value("omp_n_threads", 4);
    conf.cuda_n_threads = json_config.value("cuda_n_threads", 256);
    conf.hough_max_diff = json_config.value("hough_max_diff", 5);
    conf.n_loop         = json_config.value("n_loop", 500);
    
    // 設定内容の確認
    std::cout << "Configuration loaded:" << std::endl;
    std::cout << "  Root file path   : " << conf.root_file_path << std::endl;
    std::cout << "  Method           : " << conf.which_method << std::endl;
    std::cout << "  OMP Threads      : " << conf.omp_n_threads << std::endl;
    std::cout << "  CUDA Threads     : " << conf.cuda_n_threads << std::endl;
    std::cout << "  hough_max_diff   : " << conf.hough_max_diff << std::endl;
    std::cout << "  n_loop           : " << conf.n_loop << std::endl;
}

int main(int argc, char** argv) {

    // -- load conf file ------
    std::string config_file = (argc > 1) ? argv[1] : "conf/config.json";
    load_config(config_file);


    // -- set omp n_thread -----
    omp_set_num_threads(conf.omp_n_threads);

    // +----------------+
    // | load root file |
    // +----------------+
    auto *f = new TFile(conf.root_file_path.Data());
    if (!f || f->IsZombie()) {
        std::cerr << "Error: Could not open file : " << conf.root_file_path << std::endl;
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


    // +-----------------------------------+
    // | fit tdc and determine time window |
    // +-----------------------------------+
    auto h_tdc = new TH1D("tdc", "tdc", 2000, 0.0, 200.0);
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
    int dot_index = conf.root_file_path.Last('.');
    int sla_index = conf.root_file_path.Last('/');
    for (int i = sla_index+1; i < dot_index; i++) save_name += conf.root_file_path[i];
    TString output_path = Form("./results/%s_output.root", save_name.Data());
    if (std::ifstream(output_path.Data())) std::remove(output_path.Data());
    TFile fout(output_path.Data(), "create");
    TTree tracking_tree("tree", ""); 

    // -- prepare root file branch -----
    std::vector<int> pad_id, duration_container;
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
    tracking_tree.Branch("duration", &duration_container);


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
        duration_container.clear();
        std::vector<std::vector<int>> indices;
        if      (conf.which_method == "cuda") indices = tracking_cuda(pos_container, duration_container);
        else if (conf.which_method == "cpu")  indices = tracking_cpu(pos_container, duration_container);
        else if (conf.which_method == "omp")  indices = tracking_openmp(pos_container, duration_container); 

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
        if (*evnum >= conf.n_loop) break;
    }
    std::cout << std::endl; // for progress bar

    // +------------+
    // | Write data |
    // +------------+
    tracking_tree.Write();
    fout.Close(); 

    return 0;
}