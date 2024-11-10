#include <iostream>
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


std::vector<double> fit_tTpc(TH1D *h, TCanvas *c, int n_c) {
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
    c->cd(n_c);
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

    return result;
}

// CUDAカーネルの定義
__global__ void houghTransformKernel(int *houghSpace, const int *xData, const int *yData, int dataSize, int maxRho) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dataSize) {
        int x = xData[index];
        int y = yData[index];
        for (int theta = 0; theta < 180; ++theta) {
            float radian = theta * M_PI / 180.0;
            int rho = (int)(x * cos(radian) + y * sin(radian));
            if (rho >= 0 && rho < maxRho) {
                atomicAdd(&houghSpace[theta * maxRho + rho], 1);
            }
        }
    }
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
    int tot_num = reader.GetEntries();
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
    TCanvas *c_tdc = new TCanvas("", "", 1000, 800);
    TApplication *theApp = new TApplication("App", &argc, argv);
    std::vector<double> tdc_fit_result = fit_tTpc(h_tdc, c_tdc, 1);
    theApp->Run();
    double min_tdc_gate = tdc_fit_result[1] - 3.0*tdc_fit_result[2];
    double max_tdc_gate = tdc_fit_result[1] + 3.0*tdc_fit_result[2];




    // データサイズの指定
    int dataSize = 1000; // 任意のデータサイズ
    std::vector<int> xData(dataSize);
    std::vector<int> yData(dataSize);

    // 乱数生成器の初期化
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, 1024); // 0から1024までの乱数

    // xDataとyDataに乱数を詰める
    for (int i = 0; i < dataSize; ++i) {
        xData[i] = dist(gen);
        yData[i] = dist(gen);
    }

    // CUDAデバイスメモリを確保
    int *d_xData, *d_yData, *d_houghSpace;
    int maxRho = (int)hypot(1024, 1024); // 仮の最大範囲。適宜調整してください。
    cudaMalloc(&d_xData, dataSize * sizeof(int));
    cudaMalloc(&d_yData, dataSize * sizeof(int));
    cudaMalloc(&d_houghSpace, 180 * maxRho * sizeof(int));

    // ホストからデバイスへデータをコピー
    cudaMemcpy(d_xData, xData.data(), dataSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_yData, yData.data(), dataSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_houghSpace, 0, 180 * maxRho * sizeof(int));

    // カーネルの起動
    int threadsPerBlock = 256;
    int blocksPerGrid = (dataSize + threadsPerBlock - 1) / threadsPerBlock;
    houghTransformKernel<<<blocksPerGrid, threadsPerBlock>>>(d_houghSpace, d_xData, d_yData, dataSize, maxRho);

    // 結果をホストにコピー
    std::vector<int> houghSpace(180 * maxRho);
    cudaMemcpy(houghSpace.data(), d_houghSpace, 180 * maxRho * sizeof(int), cudaMemcpyDeviceToHost);

    // 結果の一部を表示
    std::cout << "Hough Space (一部表示):" << std::endl;
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            std::cout << houghSpace[i * maxRho + j] << " ";
        }
        std::cout << std::endl;
    }

    // デバイスメモリを解放
    cudaFree(d_xData);
    cudaFree(d_yData);
    cudaFree(d_houghSpace);

    return 0;
}
