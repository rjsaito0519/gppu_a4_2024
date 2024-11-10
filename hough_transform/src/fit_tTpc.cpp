#include "fit_tTpc.h"
#include <TF1.h>
#include <TCanvas.h>
#include <TMath.h>
#include <TBox.h>

std::vector<double> fit_tTpc(TH1D *h) {
    // -- fit ------
    double peak_pos = h->GetBinCenter( h->GetMaximumBin() );
    double range_min = peak_pos - 30.0;
    double range_max = peak_pos + 30.0;    
    TF1 *fit_f = new TF1(Form("gauss_%s", h->GetName()), "[0]*TMath::Gaus(x,[1],[2], true) + [3]", range_min, range_max);
    fit_f->SetParameter(1, peak_pos);
    fit_f->SetParameter(2, 5);
    fit_f->SetParameter(3, h->GetBinContent(static_cast<int>(range_min)) );
    fit_f->SetNpx(1000);
    fit_f->SetLineColor(kOrange);
    fit_f->SetLineWidth( 2 ); // 線の太さ変更
    h->Fit(fit_f, "0", "", range_min, range_max);
    std::vector<double> result;
    for (int i = 0; i < 4; i++) result.push_back(fit_f->GetParameter(i));

    // -- draw ------
    TCanvas *c = new TCanvas("", "", 800, 800);
    c->cd(1);
    h->GetXaxis()->SetRangeUser(result[1] - 5.0*result[2], result[1] + 5.0*result[2]);
    h->Draw();
    fit_f->Draw("same");

    // -- draw range ------
    double x1 = result[1] - 3.0 * result[2];
    double x2 = result[1] + 3.0 * result[2];
    double y1 = 0;
    double y2 = h->GetBinContent(h->GetMaximumBin());

    TBox *box = new TBox(x1, y1, x2, y2);
    box->SetFillColor(kBlue);
    box->SetFillStyle(3353);
    box->Draw("same");
    c->Update();

    // -- save and delete -----
    c->SaveAs("tdc.pdf");
    delete c;
    delete fit_f;
    delete box;

    return result;
}
