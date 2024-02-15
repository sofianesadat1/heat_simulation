#ifndef HEADERS_H
#define HEADERS_H

#include <chrono>
#include <iostream>
#include <GLFW/glfw3.h>
#include <cstdio>
#include <vector>
#include <cuda_gl_interop.h>

#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32

// Store temperature values for each grid cell
extern std::vector<std::vector<float>> gridTemperatures;

//bool global variables
extern bool isDrawing;
extern bool isSimulationRunning;

//grid dimension 
extern const int GRID_WIDTH;
extern const int GRID_HEIGHT;

//Cell size in the display window
extern const float CELL_WIDTH;
extern const float CELL_HEIGHT;



// Define minimum and maximum temperatures
extern const float minTemp; 
extern const float maxTemp;  

extern int circleRadius; // Initial circle radius
extern float drawingTemperature; 

// graphics
void updateGridTemperatures(float* d_data, std::vector<std::vector<float>>& grid, int nx, int ny);
void printMatrix(const std::vector<std::vector<float>>& matrix);


//Simulation.cu 
int __host__ __device__ getIndex(const int i, const int j, const int width);
__global__ void evolve_kernel(const float* Un, float* Unp1, const int nx, const int ny, const float dx2, const float dy2, const float dt, const float a);
void launchEvolveKernel(float* d_Un, float* d_Unp1, int nx, int ny, float dx2, float dy2, float dt, float a);

// Utilities
void drawCircleAtPosition(int gridX, int gridY);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) ;
void cursorPositionCallback(GLFWwindow* window, double xpos, double ypos) ;
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) ;
void createTextureAndCudaResource(int width, int height) ;
void drawGrid();
void copyGridToHostArray(const std::vector<std::vector<float>>& grid, float* h_Un, int nx, int ny);
void copyGridToDevice(const std::vector<std::vector<float>>& grid, float* d_data, int nx, int ny);


#endif