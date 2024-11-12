#include <vector>
#include <random>
#include <algorithm>

#include "TFile.h"
#include "TTreeReader.h"
#include "TCanvas.h"
#include "TMath.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TH2Poly.h"

#include "include/padHelper.hh"

static TString path;
static auto c_canvas_tpc = new TCanvas("Event Display", "TPC Event Display", 800, 800);

void TPC_pad_template(TH2Poly *h)
{
    Double_t X[5];
    Double_t Y[5];
    for (Int_t l=0; l<padHelper::NumOfLayersTPC; ++l) {
        Double_t pLength = padHelper::padParameter[l][5];
        Double_t st      = (180.-(360./padHelper::padParameter[l][3]) *
                          padHelper::padParameter[l][1]/2.);
        Double_t sTheta  = (-1+st/180.)*TMath::Pi();
        Double_t dTheta  = (360./padHelper::padParameter[l][3])/180.*TMath::Pi();
        Double_t cRad    = padHelper::padParameter[l][2];
        Int_t    nPad    = padHelper::padParameter[l][1];
        for (Int_t j=0; j<nPad; ++j) {
            X[1] = (cRad+(pLength/2.))*TMath::Cos(j*dTheta+sTheta);
            X[2] = (cRad+(pLength/2.))*TMath::Cos((j+1)*dTheta+sTheta);
            X[3] = (cRad-(pLength/2.))*TMath::Cos((j+1)*dTheta+sTheta);
            X[4] = (cRad-(pLength/2.))*TMath::Cos(j*dTheta+sTheta);
            X[0] = X[4];
            Y[1] = (cRad+(pLength/2.))*TMath::Sin(j*dTheta+sTheta);
            Y[2] = (cRad+(pLength/2.))*TMath::Sin((j+1)*dTheta+sTheta);
            Y[3] = (cRad-(pLength/2.))*TMath::Sin((j+1)*dTheta+sTheta);
            Y[4] = (cRad-(pLength/2.))*TMath::Sin(j*dTheta+sTheta);
            Y[0] = Y[4];
            for (Int_t k=0; k<5; ++k) X[k] += -143.;
            h->AddBin(5, X, Y);
        }
    }
    h->SetMaximum(0x1000);
}

void draw_track(Int_t n_rand)
{
    // +------------------------------------+
    // | Load ROOT file and set TTreeReader |
    // +------------------------------------+
    TFile *f = new TFile(path.Data());
    TTreeReader reader("tree", f);
    TTreeReaderValue<int> evnum(reader, "evnum");
    TTreeReaderValue<std::vector<int>> pad_id(reader, "pad_id");
    TTreeReaderValue<std::vector<double>> adc(reader, "adc");
    Int_t tot_num = reader.GetEntries();

    Int_t n = n_rand % tot_num;
    reader.Restart();
    reader.SetEntry(n);

    // +-------------------+
    // | Prepare histogram |
    // +-------------------+
    auto *h_tpc_adc2d = new TH2Poly("h_tpc_adc2d", Form("TPC ADC (evnum: %d);Z;X", n), -270.0, 270.0, -270.0, 270.0);
    TPC_pad_template(h_tpc_adc2d);

    // +---------------------+
    // | Fill event and draw |
    // +---------------------+
    for (Int_t i = 0; i < *nhTpc; i++) {
        h_tpc_adc2d->SetBinContent((*pad_id)[i]+1, (*adc)[i]);
    }

    // Draw the histogram
    c_canvas_tpc->cd(1)->SetLogz();
    h_tpc_adc2d->Draw("colz");

    // Close the ROOT file to release memory
    f->Close();
    delete f;
}

void event(Int_t n = -1)
{
    if (n == -1) {
        std::random_device rd;
        std::mt19937 gen(rd());
        Int_t n_rand = gen();
        while (n_rand < 0) n_rand = gen();
        draw_track(n_rand);
    } else {
        draw_track(n);
    }
}

void set_path(TString rootfile_path)
{
    path = rootfile_path;
}