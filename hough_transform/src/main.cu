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
        std::vector<TVector3> positions;
        std::vector<std::pair<int, double>> pad_and_de;
        
        // -- fill -----------------------
        if (nhit_threshold < *nhTpc) for (Int_t i = 0; i < *nhTpc; i++) {
            int pad = padHelper::getPadID((*layerTpc)[i], (*rowTpc)[i]);
            bool isNoisy = std::binary_search(padHelper::noisy_pad.begin(), padHelper::noisy_pad.end(), pad);
            if (isNoisy) continue;
            if ( min_tdc_gate < (*tTpc)[i] && (*tTpc)[i] < max_tdc_gate ) { // normal
            // if ( min_tTpc_gate < (*tTpc)[i] && (*tTpc)[i] < max_tTpc_gate && pad > 1343) {  // w/o TGT region
                pad_and_de.emplace_back( pad, (*deTpc)[i] );
                TVector3 pad_center_pos = padHelper::getPoint(pad);
                pad_center_pos.SetY( (*tTpc)[i] );
                positions.push_back( pad_center_pos );
            }
        }
        if ( positions.size() == 0) continue;
    }

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