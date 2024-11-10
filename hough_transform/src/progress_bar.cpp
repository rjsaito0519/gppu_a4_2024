#include "progress_bar.h"
#include <iostream>
#include <iomanip>

void displayProgressBar(int current, int total) {
    static int lastPercent = -1;
    double progress = (double)current / total;
    int percent = static_cast<int>(progress * 100);

    // Update only when the percentage changes
    if (percent != lastPercent) {
        lastPercent = percent;
        int barWidth = 50;  // Width of the progress bar

        std::cout << "[";
        int pos = barWidth * progress;
        for (int i = 0; i < barWidth; ++i) {
            if (i < pos) {
                std::cout << "=";
            } else if (i == pos) {
                std::cout << ">";
            } else {
                std::cout << " ";
            }
        }
        std::cout << "] " << std::fixed << std::setprecision(1) << (progress * 100.0) << " %\r";
        std::cout.flush();  // Flush the output to display immediately
    }
}
