#ifndef TRACKING_CPU_H
#define TRACKING_CPU_H

#include <vector>
#include <TVector3.h>

std::vector<std::vector<int>> tracking_cpu(const std::vector<TVector3>& pos_container, std::vector<int>& duration_container);

#endif // TRACKING_CPU_H
