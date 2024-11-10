#include <iostream>
#include <sys/stat.h>
#include <fstream>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <utility>

#include "TFile.h"
#include "TTree.h"
#include "TEventList.h"
#include "TMath.h"
#include "TROOT.h"
#include "TApplication.h"
#include "TH1.h"
#include "TH2.h"
#include "TCanvas.h"
#include "TF1.h"
#include "TColor.h"
#include "TGraph.h"
#include "TStyle.h"
#include "TGaxis.h"
#include "TSpectrum.h"
#include "TPolyMarker.h"
#include "TTreeReader.h"
#include "TParticle.h"
#include "TLatex.h"
#include "TH2Poly.h"

#include "include/macro.hh"
#include "include/nagao_macro.hh"
#include "include/variable.hh"
#include "include/padHelper.hh"

static Double_t max_bin_diff = 3;
static auto *Hough_hist = new TH2D("2d_hough", ";theta (deg.); r (mm)", Li_theta_ndiv, Li_theta_min, Li_theta_max, Li_r_ndiv, Li_r_min, Li_r_max);

std::vector<Int_t> tracking(const std::vector<TVector3>& positions)
{
    // --  prepare ----------
    Int_t max_iter = positions.size();

    // --  search track ----------
    std::vector<Int_t> track_ids;
    track_ids.resize(max_iter, -1);
    Int_t track_id = 0;
    while ( std::count(track_ids.begin(), track_ids.end(), -1) > 5 && track_id < 10) {
        // -- make hough hist ----------
        Double_t theta, r;
        Hough_hist->Reset();
        for (Int_t i = 0; i < max_iter; i++) if ( track_ids[i] == -1 ) for(Int_t i_theta=0; i_theta < Li_theta_ndiv; i_theta++){
            theta = Li_theta_min + i_theta * (Li_theta_max-Li_theta_min)/Li_theta_ndiv;
            Hough_hist->Fill(theta, std::cos( theta*TMath::DegToRad() )*positions[i].Z() + std::sin( theta*TMath::DegToRad() )*positions[i].X() );
        }
        Int_t maxbin = Hough_hist->GetMaximumBin();
        Int_t mx,my,mz;
        Hough_hist->GetBinXYZ(maxbin, mx, my, mz);
        Double_t mtheta = Hough_hist->GetXaxis()->GetBinCenter(mx)*TMath::DegToRad();
        Double_t mr = Hough_hist->GetYaxis()->GetBinCenter(my);
        Double_t r_bin_width = Hough_hist->GetYaxis()->GetBinWidth(0);

        // -- event selection ----------
        Double_t bin_diff;
        Bool_t within_circ = false;
        for (Int_t i = 0; i < max_iter; i++) {
            if ( track_ids[i] != -1 ) continue;
            within_circ = false;
            for (Int_t bin_theta = mx-max_bin_diff; bin_theta <= mx+max_bin_diff; bin_theta++) {
                theta = Hough_hist->GetXaxis()->GetBinCenter(bin_theta);
                r     = std::cos( theta*TMath::DegToRad() )*positions[i].Z() + std::sin( theta*TMath::DegToRad() )*positions[i].X();
                bin_diff = TMath::Abs( mr-r )/r_bin_width + TMath::Abs( bin_theta-mx );
                if ( bin_diff < max_bin_diff ) within_circ = true;
            }
            if (within_circ) track_ids[i] = track_id;
        }
        track_id++;
    }

    return track_ids;
}

void analyze(TString path)
{
    // いろんな設定
    gROOT->GetColor(kOrange)->SetRGB(1.0, 0.4980392156862745, 0.054901960784313725);
    gROOT->GetColor(kBlue)->SetRGB(0.12156862745098039, 0.4666666666666667, 0.7058823529411765);
    gStyle->SetOptStat(0);
    gStyle->SetLabelSize(0.05, "XY");
    gStyle->SetTitleSize(1, "XY");
    gStyle->SetTitleFontSize(0.08);
    gROOT->GetColor(0)->SetAlpha(0.01);

    // -- load file ------------------------------------------------------------------
    auto *f = new TFile(path.Data());

    TTreeReader reader("tpc", f);
    Int_t tot_num = reader.GetEntries();
    TTreeReaderValue<Int_t> runnum(reader, "runnum");
    TTreeReaderValue<Int_t> evnum(reader, "evnum");
    TTreeReaderValue<Int_t> nhTpc(reader, "nhTpc");
    TTreeReaderValue<std::vector<int>> layerTpc(reader, "layerTpc");
    TTreeReaderValue<std::vector<int>> rowTpc(reader, "rowTpc");
    TTreeReaderValue<std::vector<double>> deTpc(reader, "deTpc");
    TTreeReaderValue<std::vector<double>> sigmaTpc(reader, "sigmaTpc");
    TTreeReaderValue<std::vector<double>> tTpc(reader, "tTpc");

    // -- prepare histogram ------------------------------------------------------------------
    auto htTpc = new TH1D("tdc", "", tdc_bin_num, tdc_min, tdc_max);
    auto hdedx = new TH1D("dedx", "", 400, 0, 200);
    auto hn = new TH1D("n", "", 500, 0, 500);
    auto halpha = new TH1D("alpha", "", 500, -10, 10);
    
    auto hdedx_vs_redchi = new TH2D("dedx_vs_redchi", "", 400, 0, 200, 1000, 0, 1000);
    auto hdedx_vs_n = new TH2D("dedx_vs_n", "", 400, 0, 200, 500, 0, 500);

    // -- fill ------------------------------------------------------------------
    Int_t tmp_nhit_threshold = 400;
    reader.Restart();
    while (reader.Next()){
        if (*nhTpc < tmp_nhit_threshold) for (Int_t i = 0; i < *nhTpc; i++) htTpc->Fill( (*tTpc)[i] );
    }

    // -- fit and make parameter ------------------------------------------------------------------
    // +------+
    // | tTpc |
    // +------+
    TCanvas *c_tTpc = new TCanvas("", "", 1000, 800);
    Double_t par[5];
    fit_tTpc(htTpc, par, c_tTpc, 1);
    // Double_t min_tTpc_gate = par[1] - tdc_n_sigma*par[2];
    // Double_t max_tTpc_gate = par[1] + tdc_n_sigma*par[2];
    Double_t min_tTpc_gate = 80.0;
    Double_t max_tTpc_gate = 90.0;

    // -- prepare making root file -----------------------------------------------------
    TString save_name;
    Int_t dot_index = path.Last('.');
    Int_t sla_index = path.Last('/');
    for (Int_t i = sla_index+1; i < dot_index; i++) save_name += path[i];
    system(Form("rm ./root_data/%s.root", save_name.Data()));
    TFile fout(Form("./root_data/%s.root", save_name.Data()), "create");
    TTree tracking_tree("tree", ""); 

    // -- refill and cal dEdx ------------------------------------------------------------------
    Int_t nhit_threshold = 0;
    Double_t space_threshold = 30.;

    std::vector<TVector3> positions, position_buf;
    std::vector<std::pair<Int_t, Double_t>> pad_and_de;
    std::vector<Int_t> track_ids, pad_id;
    std::vector<Double_t> pos_x, pos_y, pos_z, diff_from_origin, adc;
    Double_t a_pos, b_pos, a_time, b_time, track_length, redchi2_pos, redchi2_time, sum_de, angle_pos, angle_time, dedx;
    Int_t hit_num, evnum_buf;
    TVector3 pos_origin, direc_vec;

    tracking_tree.Branch("totnum", &tot_num, "totnum/I");
    tracking_tree.Branch("evnum", &evnum_buf, "evnum/I");
    tracking_tree.Branch("nhit", &hit_num, "nhit/I");
    tracking_tree.Branch("a_pos", &a_pos, "a_pos/D");
    tracking_tree.Branch("b_pos", &b_pos, "b_pos/D");
    tracking_tree.Branch("a_time", &a_time, "a_time/D");
    tracking_tree.Branch("b_time", &b_time, "b_time/D");
    tracking_tree.Branch("desum", &sum_de, "desum/D");
    tracking_tree.Branch("length", &track_length, "length/D");
    tracking_tree.Branch("dedx", &dedx, "dedx/D");
    tracking_tree.Branch("redchi2_pos", &redchi2_pos, "redchi2_pos/D");
    tracking_tree.Branch("redchi2_time", &redchi2_time, "redchi2_time/D");
    tracking_tree.Branch("angle_pos", &angle_pos, "angle_pos/D");
    tracking_tree.Branch("angle_time", &angle_time, "angle_time/D");
    tracking_tree.Branch("adc", &adc);
    tracking_tree.Branch("pad_id", &pad_id);

    Int_t pad;
    Bool_t isNoisy;
    reader.Restart();
    while (reader.Next()){
        evnum_buf = *evnum;
        pad_and_de.clear();
        positions.clear();
        // -- fill -----------------------
        if (nhit_threshold < *nhTpc) for (Int_t i = 0; i < *nhTpc; i++) {
            pad = padHelper::getPadID((*layerTpc)[i], (*rowTpc)[i]);
            isNoisy = std::binary_search(padHelper::noisy_pad.begin(), padHelper::noisy_pad.end(), pad);
            if (isNoisy) continue;
            if ( min_tTpc_gate < (*tTpc)[i] && (*tTpc)[i] < max_tTpc_gate ) { // normal
            // if ( min_tTpc_gate < (*tTpc)[i] && (*tTpc)[i] < max_tTpc_gate && pad > 1343) {  // w/o TGT region
                pad_and_de.emplace_back( pad, (*deTpc)[i] );
                TVector3 pad_center_pos = padHelper::getPoint(pad);
                pad_center_pos.SetY( (*tTpc)[i] );
                positions.push_back( pad_center_pos );
            }
        }
        if ( positions.size() == 0) continue;

        // -- tracking and cal dedx ----------
        track_ids.clear();
        track_ids = tracking(positions);
        for (Int_t track_id = 0; track_id <= *std::max_element(track_ids.begin(), track_ids.end()); track_id++ ) {
            hit_num = std::count(track_ids.begin(), track_ids.end(), track_id);
            // if ( hit_num < 15 || 200 < hit_num) continue;
            // if ( 200 < hit_num) continue;
            // -- initialize ---------
            pos_x.clear();
            pos_y.clear();
            pos_z.clear();
            diff_from_origin.clear();
            position_buf.clear();
            adc.clear();
            pad_id.clear();
            track_length = 0.0, sum_de = 0.0;

            // -- check each track ---------
            for (Int_t i = 0, n_id = track_ids.size(); i < n_id; i++) if ( track_ids[i] == track_id ) {
                pos_x.push_back( positions[i].X() );
                pos_y.push_back( positions[i].Y() );
                pos_z.push_back( positions[i].Z() );
                position_buf.push_back( positions[i] );
                pad_id.push_back(pad_and_de[i].first);
                adc.push_back(pad_and_de[i].second);
                sum_de += pad_and_de[i].second;
            }
            // -- fit xy plain ----------
            auto *gr_xy = new TGraph(pos_x.size(), &pos_x[0], &pos_y[0]);
            auto *finc_xy = new TF1( "fit_finc_xy", "[0]*x + [1]", DrawMinX, DrawMaxX);
            gr_xy->Fit("fit_finc_xy", "Q", "", DrawMinX, DrawMaxX);
            a_time = finc_xy->GetParameter(0);
            b_time = finc_xy->GetParameter(1);
            redchi2_time = finc_xy->GetChisquare() / finc_xy->GetNDF();
            angle_time = TMath::ATan(a_time)*TMath::RadToDeg();
            // -- fit xz plain ----------
            auto *gr_xz = new TGraph(pos_x.size(), &pos_x[0], &pos_z[0]);
            auto *finc_xz = new TF1( "fit_finc_xz", "[0]*x + [1]", DrawMinX, DrawMaxX);
            gr_xz->Fit("fit_finc_xz", "Q", "", DrawMinX, DrawMaxX);
            a_pos = finc_xz->GetParameter(0);
            b_pos = finc_xz->GetParameter(1);
            redchi2_pos = finc_xz->GetChisquare() / finc_xz->GetNDF();
            angle_pos = TMath::ATan(a_pos)*TMath::RadToDeg();
            // -- calc. dedx ----------
            pos_origin.SetXYZ(-270, 0, -270*a_pos+b_pos);
            direc_vec.SetXYZ(1, 0, a_pos);
            for (auto position : position_buf) diff_from_origin.push_back( direc_vec.Dot( position - pos_origin ) / direc_vec.Mag() );
            std::sort(diff_from_origin.begin(), diff_from_origin.end());
            for (Int_t i = 1, n_diff = diff_from_origin.size(); i < n_diff; i++) {
                Double_t tmp_diff = diff_from_origin[i]-diff_from_origin[i-1];
                if (tmp_diff < space_threshold) track_length += tmp_diff;
            }
            dedx = sum_de/track_length;
            // -- fill data to histo ----------
            hdedx->Fill( sum_de/track_length );
            hn->Fill( pos_x.size() );
            halpha->Fill( a_pos );
            hdedx_vs_redchi->Fill( sum_de/track_length, redchi2_pos );
            hdedx_vs_n->Fill( sum_de/track_length, pos_x.size() );
            tracking_tree.Fill();
            // -- delete gr ---------- 
            delete gr_xy;
            gr_xy = nullptr;
            delete finc_xy;
            finc_xy = nullptr;
            delete gr_xz;
            gr_xz = nullptr;
            delete finc_xz;
            finc_xz = nullptr;
        }
        if (*evnum%500 == 0) std::cout << "n/tot_num = " << *evnum << "/" << tot_num << std::endl;
        // if (*evnum >= 500) break;
    }

    tracking_tree.Write();
    fout.Close(); 

    TCanvas *c_dedx = new TCanvas("", "", 800, 800);
    c_dedx->cd(1);
    hdedx->Draw();

    TCanvas *c_n = new TCanvas("", "", 800, 800);
    c_n->cd(1);
    hn->Draw();

    TCanvas *c_alpha = new TCanvas("", "", 800, 800);
    c_alpha->cd(1);
    halpha->Draw();

    TCanvas *c_dedx_vs_redchi = new TCanvas("", "", 800, 800);
    c_dedx_vs_redchi->cd(1);
    hdedx_vs_redchi->Draw("colz");

    TCanvas *c_dedx_vs_n = new TCanvas("", "", 800, 800);
    c_dedx_vs_n->cd(1);
    hdedx_vs_n->Draw("colz");

    system(Form("python3 discord.py %s", path.Data()));
    std::cout << "finish" << std::endl;
}

Int_t main(int argc, char** argv) {
    

    TString path = argv[1];
    if (argc > 2) max_bin_diff = std::atoi(argv[2]);
    // TApplication *theApp = new TApplication("App", &argc, argv);    
    analyze(path);
    // theApp->Run();

    return 0;
}
