#include <iostream>
#include <vector>
#include <cmath>
#include <TFile.h>
#include <TTree.h>
#include <cuda_runtime.h>

#include <TFile.h>
#include <TTree.h>
#include <TEventList.h>
#include <TMath.h>
#include <TROOT.h>
#include <TApplication.h>
#include <TH1.h>
#include <TH2.h>
#include <TCanvas.h>
#include <TF1.h>
#include <TColor.h>
#include <TGraph.h>
#include <TStyle.h>
#include <TGaxis.h>
#include <TSpectrum.h>
#include <TPolyMarker.h>
#include <TTreeReader.h>
#include <TParticle.h>
#include <TLatex.h>
#include <TH2Poly.h>


#include "include/ana_helper.hh"

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


Int_t main(int argc, char** argv) {
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
        return;
    }
    TTreeReader reader("tpc", f);
    Int_t tot_num = reader.GetEntries();
    TTreeReaderValue<Int_t> runnum(reader, "runnum");
    TTreeReaderValue<Int_t> evnum(reader, "evnum");
    TTreeReaderValue<Int_t> nhTpc(reader, "nhTpc");
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
        if (*nhTpc < 400) for (Int_t i = 0; i < *nhTpc; i++) h_tdc->Fill( (*tTpc)[i] );
    }
    TCanvas *c_tdc = new TCanvas("", "", 1000, 800);
    std::vector<Double_t> tdc_fit_result = fit_tTpc(h_tdc, c_tdc, 1);
    Double_t min_tdc_gate = tdc_fit_result[1] - 3.0*tdc_fit_result[2];
    Double_t max_tdc_gate = tdc_fit_result[1] + 3.0*tdc_fit_result[2];




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
