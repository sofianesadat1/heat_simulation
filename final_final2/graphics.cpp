#include "headers.h"


void updateGridTemperatures(float* d_data, std::vector<std::vector<float>>& grid, int nx, int ny) {
    std::vector<float> temp(nx * ny);
    cudaMemcpy(temp.data(), d_data, nx * ny * sizeof(float), cudaMemcpyDeviceToHost);
    for (int x = 0; x < nx; ++x) {
        for (int y = 0; y < ny; ++y) {
            grid[x][y] = temp[y * nx + x];
        }
    }
}

// Function to print a matrix
void printMatrix(const std::vector<std::vector<float>>& matrix) {
    for (int y = 0; y < matrix.size(); y++) {
        for (int x = 0; x < matrix[0].size(); x++) {
            printf("%f ", matrix[y][x]);
            
        }
        printf("\n");
        
    }
    
}