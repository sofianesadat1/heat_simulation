#include "headers.h"

// Function to draw the grid
void drawGrid() {
    float normalizationFactor = 1.0f / (maxTemp - minTemp); 

    glBegin(GL_QUADS); // Start

    for (int x = 0; x < GRID_WIDTH; ++x) {
        for (int y = 0; y < GRID_HEIGHT; ++y) {
            float temp = gridTemperatures[x][y];

            // Normalize the temperature value to [0, 1]
            float normalizedTemp = (temp - minTemp) * normalizationFactor;


            // Determine if we need to change the color
            static float lastR = -1.0f;


            // Set color only if it has changed
            if (normalizedTemp != lastR) {
                glColor3f(normalizedTemp, 0, 1-normalizedTemp);
                lastR = normalizedTemp;

            }

            // Specify the 4 vertices of the quad
            glVertex2f(x * CELL_WIDTH, y * CELL_HEIGHT);
            glVertex2f((x + 1) * CELL_WIDTH, y * CELL_HEIGHT);
            glVertex2f((x + 1) * CELL_WIDTH, (y + 1) * CELL_HEIGHT);
            glVertex2f(x * CELL_WIDTH, (y + 1) * CELL_HEIGHT);
        }
    }

    glEnd(); // Finish drawing quads
}

void drawCircleAtPosition(int gridX, int gridY) {
    for (int x = -circleRadius; x <= circleRadius; x++) {
        for (int y = -circleRadius; y <= circleRadius; y++) {
            if (x * x + y * y <= circleRadius * circleRadius) {
                int drawX = gridX + x;
                int drawY = gridY + y;
                if (drawX >= 0 && drawX < GRID_WIDTH && drawY >= 0 && drawY < GRID_HEIGHT) {
                    gridTemperatures[drawX][drawY] = drawingTemperature;
                }
            }
        }
    }
}
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (!isSimulationRunning) {
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            double xpos, ypos;
            glfwGetCursorPos(window, &xpos, &ypos);
            int gridX = static_cast<int>(xpos / CELL_WIDTH);
            int gridY = static_cast<int>(ypos / CELL_HEIGHT);

            if (action == GLFW_PRESS) {
                isDrawing = true;
                drawCircleAtPosition(gridX, gridY); // Draw circle on click
            } else if (action == GLFW_RELEASE) {
                isDrawing = false;
            }
        }
    }
}
// Key callback for starting the simulation
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    // if (key == GLFW_KEY_ENTER && action == GLFW_PRESS) {
    //     isSimulationRunning = true;
    // }
    switch (key) {
            case GLFW_KEY_ENTER : 
                isSimulationRunning = true;
            case GLFW_KEY_UP: 
                circleRadius++; // Increase radius
                break;
            case GLFW_KEY_DOWN:
                circleRadius = std::max(1, circleRadius - 1); // Decrease radius
                break;
            case GLFW_KEY_RIGHT:
                drawingTemperature += 5.0f; // Increase temperature
                break;
            case GLFW_KEY_LEFT:
                drawingTemperature = std::max(0.0f, drawingTemperature - 5.0f); // Decrease temperature
                break;
            default:
                break;
        }
}

void copyGridToHostArray(const std::vector<std::vector<float>>& grid, float* h_Un, int nx, int ny) {
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            int index = y * nx + x; // Calculate the flat index
            h_Un[index] = grid[x][y];
        }
    }
}

void copyGridToDevice(const std::vector<std::vector<float>>& grid, float* d_data, int nx, int ny) {
    std::vector<float> temp(nx * ny);

    for (int x = 0; x < nx; ++x) {
        for (int y = 0; y < ny; ++y) {
            temp[y * nx + x] = grid[x][y];
        }
    }

    cudaMemcpy(d_data, temp.data(), nx * ny * sizeof(float), cudaMemcpyHostToDevice);
}


void cursorPositionCallback(GLFWwindow* window, double xpos, double ypos) {
    if (!isSimulationRunning && isDrawing) {
        // Convert xpos and ypos to grid coordinates
        int gridX = static_cast<int>(xpos / CELL_WIDTH);
        int gridY = static_cast<int>(ypos / CELL_HEIGHT);

        int radius = circleRadius; 

        for (int x = -radius; x <= radius; x++) {
            for (int y = -radius; y <= radius; y++) {
                if (x * x + y * y <= radius * radius) {
                    int drawX = gridX + x;
                    int drawY = gridY + y;
                    if (drawX >= 0 && drawX < GRID_WIDTH && drawY >= 0 && drawY < GRID_HEIGHT) {
                        gridTemperatures[drawX][drawY] = drawingTemperature; // Set to max temperature
                    }
                }
            }
        }
    }
}