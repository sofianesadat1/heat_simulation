CC=g++
NVCC=nvcc
CFLAGS=-I/usr/local/cuda/include
LFLAGS=-L/usr/local/cuda/lib64 -lcudart -lglfw -lGL -lX11 -lpthread -lXrandr -lXi -ldl

# Object files
OBJ = main.o graphics.o utilities.o simulation.o

# Executable name
EXEC = final_executable

# Default target
all: $(EXEC)

# Link object files into a binary
$(EXEC): $(OBJ)
	$(CC) -o $@ $^ $(LFLAGS)

# Compile main.cpp to main.o
main.o: main.cpp
	$(CC) -c -o $@ $< $(CFLAGS)

# Compile graphics.cpp to graphics.o
graphics.o: graphics.cpp
	$(CC) -c -o $@ $< $(CFLAGS)

# Compile utilities.cpp to utilities.o
utilities.o: utilities.cpp
	$(CC) -c -o $@ $< $(CFLAGS)

# Compile simulation.cu to simulation.o
simulation.o: simulation.cu
	$(NVCC) -c -o $@ $< $(CFLAGS)

# Clean the build
clean:
	rm -f $(OBJ) $(EXEC)
