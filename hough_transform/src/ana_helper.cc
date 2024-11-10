

std::vector<Double_t> fit_tTpc(TH1D *h, TCanvas *c, Int_t n_c) {
    // -- fit ------
    Double_t peak_pos = h->GetBinCenter( h->GetMaximumBin() );
    Double_t range_min = peak_pos - 30.0;
    Double_t range_max = peak_pos + 30.0;    
    TF1 *fit_f = new TF1(Form("gauss_%s", h->GetName()), "[0]*TMath::Gaus(x,[1],[2], true) + [3]", range_min, range_max);
    fit_f->SetParameter(1, peak_pos);
    fit_f->SetParameter(2, 5);
    fit_f->SetParameter(3, h->GetBinContent(static_cast<Int_t>(range_min)) );
    fit_f->SetNpx(1000);
    fit_f->SetLineColor(kOrange);
    fit_f->SetLineWidth( 2 ); // 線の太さ変更
    h->Fit(fit_f, "0", "", range_min, range_max);
    std::vector<Double_t> result;
    for (Int_t i = 0; i < 4; i++) result.push_back(fit_f->GetParameter(i));

    // -- draw ------
    c->cd(n_c);
    h->GetXaxis()->SetRangeUser(result[1] - 5.0*result[2], result[1] + 5.0*result[2]);
    h->Draw();
    fit_f->Draw("same");

    // -- draw range ------
    Double_t x1 = result[1] - 3.0 * result[2];
    Double_t x2 = result[1] + 3.0 * result[2];
    Double_t y1 = 0;
    Double_t y2 = h->GetBinContent(h->GetMaximumBin());

    TBox *box = new TBox(x1, y1, x2, y2);
    box->SetFillColor(kBlue);
    box->SetFillStyle(3353);
    box->Draw("same");
    c->Update();

    return result;
}
