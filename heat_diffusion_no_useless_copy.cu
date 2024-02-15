#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "pngwriter.h"

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16



struct pixel_t {
    uint8_t red;
    uint8_t green;
    uint8_t blue;
};


static int heat_colormap[256][3] = {
    { 59,  76,  192 }, { 59,  76,  192 }, { 60,  78,  194 }, { 61,  80,  195 },
    { 62,  81,  197 }, { 64,  83,  198 }, { 65,  85,  200 }, { 66,  87,  201 },
    { 67,  88,  203 }, { 68,  90,  204 }, { 69,  92,  206 }, { 71,  93,  207 },
    { 72,  95,  209 }, { 73,  97,  210 }, { 74,  99,  211 }, { 75,  100, 213 },
    { 77,  102, 214 }, { 78,  104, 215 }, { 79,  105, 217 }, { 80,  107, 218 },
    { 82,  109, 219 }, { 83,  110, 221 }, { 84,  112, 222 }, { 85,  114, 223 },
    { 87,  115, 224 }, { 88,  117, 225 }, { 89,  119, 227 }, { 90,  120, 228 },
    { 92,  122, 229 }, { 93,  124, 230 }, { 94,  125, 231 }, { 96,  127, 232 },
    { 97,  129, 233 }, { 98,  130, 234 }, { 100, 132, 235 }, { 101, 133, 236 },
    { 102, 135, 237 }, { 103, 137, 238 }, { 105, 138, 239 }, { 106, 140, 240 },
    { 107, 141, 240 }, { 109, 143, 241 }, { 110, 144, 242 }, { 111, 146, 243 },
    { 113, 147, 244 }, { 114, 149, 244 }, { 116, 150, 245 }, { 117, 152, 246 },
    { 118, 153, 246 }, { 120, 155, 247 }, { 121, 156, 248 }, { 122, 157, 248 },
    { 124, 159, 249 }, { 125, 160, 249 }, { 127, 162, 250 }, { 128, 163, 250 },
    { 129, 164, 251 }, { 131, 166, 251 }, { 132, 167, 252 }, { 133, 168, 252 },
    { 135, 170, 252 }, { 136, 171, 253 }, { 138, 172, 253 }, { 139, 174, 253 },
    { 140, 175, 254 }, { 142, 176, 254 }, { 143, 177, 254 }, { 145, 179, 254 },
    { 146, 180, 254 }, { 147, 181, 255 }, { 149, 182, 255 }, { 150, 183, 255 },
    { 152, 185, 255 }, { 153, 186, 255 }, { 154, 187, 255 }, { 156, 188, 255 },
    { 157, 189, 255 }, { 158, 190, 255 }, { 160, 191, 255 }, { 161, 192, 255 },
    { 163, 193, 255 }, { 164, 194, 254 }, { 165, 195, 254 }, { 167, 196, 254 },
    { 168, 197, 254 }, { 169, 198, 254 }, { 171, 199, 253 }, { 172, 200, 253 },
    { 173, 201, 253 }, { 175, 202, 252 }, { 176, 203, 252 }, { 177, 203, 252 },
    { 179, 204, 251 }, { 180, 205, 251 }, { 181, 206, 250 }, { 183, 207, 250 },
    { 184, 207, 249 }, { 185, 208, 249 }, { 186, 209, 248 }, { 188, 209, 247 },
    { 189, 210, 247 }, { 190, 211, 246 }, { 191, 211, 246 }, { 193, 212, 245 },
    { 194, 213, 244 }, { 195, 213, 243 }, { 196, 214, 243 }, { 198, 214, 242 },
    { 199, 215, 241 }, { 200, 215, 240 }, { 201, 216, 239 }, { 202, 216, 239 },
    { 204, 217, 238 }, { 205, 217, 237 }, { 206, 217, 236 }, { 207, 218, 235 },
    { 208, 218, 234 }, { 209, 218, 233 }, { 210, 219, 232 }, { 211, 219, 231 },
    { 212, 219, 230 }, { 214, 220, 229 }, { 215, 220, 228 }, { 216, 220, 227 },
    { 217, 220, 225 }, { 218, 220, 224 }, { 219, 220, 223 }, { 220, 221, 222 },
    { 221, 221, 221 }, { 222, 220, 219 }, { 223, 220, 218 }, { 224, 219, 216 },
    { 225, 219, 215 }, { 226, 218, 214 }, { 227, 218, 212 }, { 228, 217, 211 },
    { 229, 216, 209 }, { 230, 216, 208 }, { 231, 215, 206 }, { 232, 215, 205 },
    { 233, 214, 203 }, { 233, 213, 202 }, { 234, 212, 200 }, { 235, 212, 199 },
    { 236, 211, 197 }, { 237, 210, 196 }, { 237, 209, 194 }, { 238, 208, 193 },
    { 239, 208, 191 }, { 239, 207, 190 }, { 240, 206, 188 }, { 240, 205, 187 },
    { 241, 204, 185 }, { 242, 203, 183 }, { 242, 202, 182 }, { 243, 201, 180 },
    { 243, 200, 179 }, { 243, 199, 177 }, { 244, 198, 176 }, { 244, 197, 174 },
    { 245, 196, 173 }, { 245, 195, 171 }, { 245, 194, 169 }, { 246, 193, 168 },
    { 246, 192, 166 }, { 246, 190, 165 }, { 246, 189, 163 }, { 247, 188, 161 },
    { 247, 187, 160 }, { 247, 186, 158 }, { 247, 184, 157 }, { 247, 183, 155 },
    { 247, 182, 153 }, { 247, 181, 152 }, { 247, 179, 150 }, { 247, 178, 149 },
    { 247, 177, 147 }, { 247, 175, 146 }, { 247, 174, 144 }, { 247, 172, 142 },
    { 247, 171, 141 }, { 247, 170, 139 }, { 247, 168, 138 }, { 247, 167, 136 },
    { 247, 165, 135 }, { 246, 164, 133 }, { 246, 162, 131 }, { 246, 161, 130 },
    { 246, 159, 128 }, { 245, 158, 127 }, { 245, 156, 125 }, { 245, 155, 124 },
    { 244, 153, 122 }, { 244, 151, 121 }, { 243, 150, 119 }, { 243, 148, 117 },
    { 242, 147, 116 }, { 242, 145, 114 }, { 241, 143, 113 }, { 241, 142, 111 },
    { 240, 140, 110 }, { 240, 138, 108 }, { 239, 136, 107 }, { 239, 135, 105 },
    { 238, 133, 104 }, { 237, 131, 102 }, { 237, 129, 101 }, { 236, 128, 99  },
    { 235, 126, 98  }, { 235, 124, 96  }, { 234, 122, 95  }, { 233, 120, 94  },
    { 232, 118, 92  }, { 231, 117, 91  }, { 230, 115, 89  }, { 230, 113, 88  },
    { 229, 111, 86  }, { 228, 109, 85  }, { 227, 107, 84  }, { 226, 105, 82  },
    { 225, 103, 81  }, { 224, 101, 79  }, { 223, 99,  78  }, { 222, 97,  77  },
    { 221, 95,  75  }, { 220, 93,  74  }, { 219, 91,  73  }, { 218, 89,  71  },
    { 217, 87,  70  }, { 215, 85,  69  }, { 214, 82,  67  }, { 213, 80,  66  },
    { 212, 78,  65  }, { 211, 76,  64  }, { 210, 74,  62  }, { 208, 71,  61  },
    { 207, 69,  60  }, { 206, 67,  59  }, { 204, 64,  57  }, { 203, 62,  56  },
    { 202, 59,  55  }, { 200, 57,  54  }, { 199, 54,  53  }, { 198, 52,  51  },
    { 196, 49,  50  }, { 195, 46,  49  }, { 193, 43,  48  }, { 192, 40,  47  },
    { 191, 37,  46  }, { 189, 34,  44  }, { 188, 30,  43  }, { 186, 26,  42  },
    { 185, 22,  41  }, { 183, 17,  40  }, { 182, 11,  39  }, { 180, 4,   38  },
};

void cmap(double value, const double scaling, const double offset,
          pixel_t * pix)
{
    int ival;

    ival = (int) (value * scaling + offset);
    if (ival < 0) {             
        pix->red = 0;
        pix->green = 0;
        pix->blue = 255;
    } else if (ival > 255) {
        pix->red = 255;         
        pix->green = 0;
        pix->blue = 0;
    } else {
        pix->red = heat_colormap[ival][0];
        pix->green = heat_colormap[ival][1];
        pix->blue = heat_colormap[ival][2];
    }
}


int save_png(float *data, const int height, const int width, const char *fname,
             const char lang)
{
    FILE *fp;
    png_structp pngstruct_ptr = NULL;
    png_infop pnginfo_ptr = NULL;
    png_byte **row_pointers = NULL;
    int i, j;

    
    int status = -1;

    int pixel_size = 3;
    int depth = 8;

    fp = fopen(fname, "wb");
    if (fp == NULL)
        goto fopen_failed;

    pngstruct_ptr =
        png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    if (pngstruct_ptr == NULL)
        goto pngstruct_create_failed;

    pnginfo_ptr = png_create_info_struct(pngstruct_ptr);

    if (pnginfo_ptr == NULL)
        goto pnginfo_create_failed;

    if (setjmp(png_jmpbuf(pngstruct_ptr)))
        goto setjmp_failed;

    png_set_IHDR(pngstruct_ptr, pnginfo_ptr, (size_t) width,
                 (size_t) height, depth, PNG_COLOR_TYPE_RGB,
                 PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

    row_pointers = (png_bytep*) png_malloc(pngstruct_ptr, height * sizeof(png_byte *));

    for (i = 0; i < height; i++) {
        png_bytep row = (png_bytep) png_malloc(pngstruct_ptr, sizeof(uint8_t) * width * pixel_size);

        row_pointers[i] = row;

        
        if (lang == 'c' || lang == 'C') {
            for (j = 0; j < width; j++) {
                pixel_t pixel;
                
                cmap(data[j + i * width], 2.55, 0.0, &pixel);
                *row++ = pixel.red;
                *row++ = pixel.green;
                *row++ = pixel.blue;
            }
        } else {
            for (j = 0; j < width; j++) {
                pixel_t pixel;
                
                cmap(data[i + j * height], 2.55, 0.0, &pixel);
                *row++ = pixel.red;
                *row++ = pixel.green;
                *row++ = pixel.blue;
            }
        }
    }

    png_init_io(pngstruct_ptr, fp);
    png_set_rows(pngstruct_ptr, pnginfo_ptr, row_pointers);
    png_write_png(pngstruct_ptr, pnginfo_ptr,
                  PNG_TRANSFORM_IDENTITY, NULL);

    status = 0;

    for (i = 0; i < height; i++) {
        png_free(pngstruct_ptr, row_pointers[i]);
    }
    png_free(pngstruct_ptr, row_pointers);

  setjmp_failed:
  pnginfo_create_failed:
    png_destroy_write_struct(&pngstruct_ptr, &pnginfo_ptr);
  pngstruct_create_failed:
    fclose(fp);
  fopen_failed:
    return status;
}



int __host__ __device__ getIndex(const int i, const int j, const int width)
{
    return i*width + j;
}

__global__ void evolve_kernel(const float* Un, float* Unp1, const int nx, const int ny, const float dx2, const float dy2, const float aTimesDt)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i > 0 && i < nx - 1)
    {
        int j = threadIdx.y + blockIdx.y*blockDim.y;
        if (j > 0 && j < ny - 1)
        {
            const int index = getIndex(i, j, ny);
            float uij = Un[index];
            float uim1j = Un[getIndex(i-1, j, ny)];
            float uijm1 = Un[getIndex(i, j-1, ny)];
            float uip1j = Un[getIndex(i+1, j, ny)];
            float uijp1 = Un[getIndex(i, j+1, ny)];

            // Explicit scheme
            Unp1[index] = uij + aTimesDt * ( (uim1j - 2.0*uij + uip1j)/dx2 + (uijm1 - 2.0*uij + uijp1)/dy2 );
        }
    }
}

int main()
{   int e = 7500;
    const int nx = e;   // Width of the area
    const int ny = e;   // Height of the area

    const float a = 0.5;     // Diffusion constant

    const float dx = 0.01;   // Horizontal grid spacing 
    const float dy = 0.01;   // Vertical grid spacing

    const float dx2 = dx*dx;
    const float dy2 = dy*dy;

    const float dt = dx2 * dy2 / (2.0 * a * (dx2 + dy2)); // Largest stable time step
    const int numSteps = 5000;                             // Number of time steps
    const int outputEvery = 5000;                          // How frequently to write output image

    int numElements = nx*ny;

    
    float* h_Un   = (float*)calloc(numElements, sizeof(float));

    
    float radius2 = (nx/6.0) * (nx/6.0);
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            int index = getIndex(i, j, ny);
            
            float ds2 = (i - nx/2) * (i - nx/2) + (j - ny/2)*(j - ny/2);
            if (ds2 < radius2)
            {
                h_Un[index] = 65.0;
            }
            else
            {
                h_Un[index] = 5.0;
            }
        }
    }

    float* d_Un;
    float* d_Unp1;

    cudaMalloc((void**)&d_Un, numElements*sizeof(float));
    cudaMalloc((void**)&d_Unp1, numElements*sizeof(float));

    cudaMemcpy(d_Un, h_Un, numElements*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Unp1, h_Un, numElements*sizeof(float), cudaMemcpyHostToDevice);

    dim3 numBlocks(nx/BLOCK_SIZE_X + 1, ny/BLOCK_SIZE_Y + 1);
    dim3 threadsPerBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    
    clock_t start = clock();

    
    for (int n = 0; n <= numSteps; n++)
    {
        evolve_kernel<<<numBlocks, threadsPerBlock>>>(d_Un, d_Unp1, nx, ny, dx2, dy2, a*dt);

        
        if (n % outputEvery == 0)
        {
            cudaMemcpy(h_Un, d_Un, numElements*sizeof(float), cudaMemcpyDeviceToHost);
            cudaError_t errorCode = cudaGetLastError();
            if (errorCode != cudaSuccess)
            {
                printf("Cuda error %d: %s\n", errorCode, cudaGetErrorString(errorCode));
                exit(0);
            }
            char filename[64];
            sprintf(filename, "heat_%04d.png", n);
            save_png(h_Un, nx, ny, filename, 'c');
        }

        std::swap(d_Un, d_Unp1);
    }

    
    clock_t finish = clock();
    printf("It took %f seconds\n", (double)(finish - start) / CLOCKS_PER_SEC);

    
    free(h_Un);

    cudaFree(d_Un);
    cudaFree(d_Unp1);
    
    return 0;
}