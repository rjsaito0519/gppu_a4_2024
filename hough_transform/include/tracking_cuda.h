#ifndef TRACKING_CUDA_H
#define TRACKING_CUDA_H

#include <vector>
#include <TVector3.h>

std::vector<std::vector<int>> tracking_cuda(const std::vector<TVector3>& pos_container);

#endif // TRACKING_CUDA_H
