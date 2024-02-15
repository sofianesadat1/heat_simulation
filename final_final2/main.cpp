#include "headers.h"

bool isDrawing = false;
bool isSimulationRunning = false;

// Store temperature values for each grid cell
std::vector<std::vector<float>> gridTemperatures(GRID_WIDTH, std::vector<float>(GRID_HEIGHT, 0.0f));


const int GRID_WIDTH = 2000;
const int GRID_HEIGHT = 2000;


int circleRadius = GRID_HEIGHT / 10; // Initial circle radius
float drawingTemperature = 65.0f;    

const float CELL_WIDTH = 800.0f / GRID_WIDTH;
const float CELL_HEIGHT = 800.0f / GRID_HEIGHT;

// Define minimum and maximum temperatures
const float minTemp = 0.0f; 
const float maxTemp = 100.0f; 

// Global variable to track if the mouse button is held down
bool isMouseButtonDown = false;


int main() {


    const int nx = GRID_WIDTH;   // Width of the area
    const int ny = GRID_HEIGHT;   // Height of the area

    const float a = 0.5;     // Diffusion constant

    const float dx = 0.01;   // Horizontal grid spacing 
    const float dy = 0.01;   // Vertical grid spacing

    const float dx2 = dx*dx;
    const float dy2 = dy*dy;

    const float dt = dx2 * dy2 / (2.0 * a * (dx2 + dy2)); // Largest stable time step


    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return -1;
    }

    GLFWwindow* window = glfwCreateWindow(800, 600, "Heat Simulation", NULL, NULL);
    if (!window) {
        glfwTerminate();
        fprintf(stderr, "Failed to create GLFW window\n");
        return -1;
    }

    // Allocate host memory
    int numElements = nx * ny;
    float* h_Un = (float*)calloc(numElements, sizeof(float));
    if (!h_Un) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return -1;
    }

    // Allocate device memory
    float *d_Un, *d_Unp1;

    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void**)&d_Un, numElements * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for d_Un\n");
        free(h_Un);
        return -1;
    }
    cudaStatus = cudaMalloc((void**)&d_Unp1, numElements * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for d_Unp1\n");
        cudaFree(d_Un);
        free(h_Un);
        return -1;
    }


    // Copy data from host to device
    cudaMemcpy(d_Un, h_Un, numElements * sizeof(float), cudaMemcpyHostToDevice);        
    cudaMemcpy(d_Unp1, h_Un, numElements * sizeof(float), cudaMemcpyHostToDevice);

    // Create OpenGL texture and register with CUDA
    glfwMakeContextCurrent(window);

    dim3 numBlocks((nx + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (ny + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    dim3 threadsPerBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);


    glfwMakeContextCurrent(window);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetKeyCallback(window, keyCallback);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 800, 600, 0, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glfwSetCursorPosCallback(window, cursorPositionCallback);


// Main loop
while (!glfwWindowShouldClose(window)) {
    glClear(GL_COLOR_BUFFER_BIT);

    if (isSimulationRunning) {
        // Start timing
        auto start = std::chrono::high_resolution_clock::now();
        copyGridToDevice(gridTemperatures, d_Un, nx, ny) ;

        // Perform the simulation step
        launchEvolveKernel(d_Un, d_Unp1, nx, ny, dx*dx, dy*dy, dt, a);
        cudaDeviceSynchronize();


        // Update grid temperatures from CUDA device
        updateGridTemperatures(d_Unp1, gridTemperatures, nx, ny);

        // Swap pointers for next iteration
        std::swap(d_Un, d_Unp1);

        // End timing
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        std::cout << "Iteration took " << elapsed.count() << " ms\n";

        // isSimulationRunning = false;
    }

    // Draw the grid with the latest temperatures
    drawGrid();

    // Swap
    glfwSwapBuffers(window);
    glfwPollEvents();
}


// Cleanup
glfwTerminate();
cudaFree(d_Un);
cudaFree(d_Unp1);
free(h_Un);

return 0;
}