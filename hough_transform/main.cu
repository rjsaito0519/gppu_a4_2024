#include <iostream>
#include <vector>
#include <cmath>
#include <TFile.h>
#include <TTree.h>
#include <cuda_runtime.h>

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


// データを読み込む関数
void loadData(const char* filename, std::vector<int>& xData, std::vector<int>& yData) {
    TFile *file = TFile::Open(filename);
    if (!file || file->IsZombie()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    TTree *tree = (TTree*)file->Get("tree"); // 適切にツリー名を設定
    if (!tree) {
        std::cerr << "Error: Could not find tree in the file" << std::endl;
        return;
    }

    int x, y;
    tree->SetBranchAddress("x", &x);
    tree->SetBranchAddress("y", &y);

    Long64_t nEntries = tree->GetEntries();
    for (Long64_t i = 0; i < nEntries; ++i) {
        tree->GetEntry(i);
        xData.push_back(x);
        yData.push_back(y);
    }

    file->Close();
}

int main(int argc, char** argv) {
    // 引数チェック
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_root_file>" << std::endl;
        return 1;
    }

    const char* rootFilePath = argv[1];

    // ROOTファイルからデータを読み込む
    std::vector<int> xData;
    std::vector<int> yData;
    loadData(rootFilePath, xData, yData);

    int dataSize = xData.size();
    if (dataSize == 0) {
        std::cerr << "Error: No data found in the ROOT file." << std::endl;
        return 1;
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
