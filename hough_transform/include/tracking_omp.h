#ifndef TRACKING_OMP_H
#define TRACKING_OMP_H

#include <vector>
#include <TVector3.h>

// Function declaration for tracking with OpenMP
std::vector<std::vector<int>> tracking_openmp(const std::vector<TVector3>& pos_container, std::vector<int>& duration_container);

#endif // TRACKING_OMP_H
