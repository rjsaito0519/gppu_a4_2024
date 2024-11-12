#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <TString.h> // TStringのヘッダをインクルード

struct config {
    TString root_file_path;
    std::string which_method;
    int omp_n_threads;
    int cuda_n_threads;
};

extern config conf; // 他のファイルからアクセス可能にするために extern 宣言

#endif // CONFIG_H
