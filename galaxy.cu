#include <stdio.h>
#include <cuda.h>

// define stuff for the bins
#define BIN_WIDTH 0.25f
#define BIN_MIN 0.0f
#define BIN_MAX 180.0f
#define NUM_BINS (int)(BIN_MAX * (1.0f / BIN_WIDTH)) // 720

// define stuff for converting angles
#define PI 3.14159265359f
#define ARC_MIN_TO_RAD (PI / (60.0f * 180.0f))
#define RAD_TO_DEG (180.0f / PI)

// Store file content in a struct to make the program more simple
typedef struct {
    float *alphas, *deltas;
} Galaxies;

int getDevice(int deviceNo);
void calculate_histogram(int* histogram, int num_galaxies, Galaxies galaxies, Galaxies other);
void calculate_omegas(float* omegas, int* DD, int* DR, int* RR);
void write_histogram(const char *filename, int *histogram);
void write_omegas(const char *filename, float *omegas);
void print_histogram(const char* label, int *histogram, int bins_to_print);
void print_omegas(float* omegas, int bins_to_print);
long check_histogram_size(int *histogram);
int galaxiesFromFile(const char *filename, int *num_galaxies, Galaxies *galaxies);
__global__ void calculate_angles(float *alphas1, float *deltas1, float *alphas2, float *deltas2, int num_galaxies, int *gpu_hist);

int main(int argc, char **argv) {

    if (argc < 3) {
        printf("Usage: <real_data_file> <synthetic_data_file> <bins_to_print>\n");
        return(-1);
    }

    if (getDevice(0) != 0) {
        return(-1);
    }

    int num_galaxies;
    // Read files and store data in Galaxies classes.
    Galaxies real_galaxies;
    if (galaxiesFromFile(argv[1], &num_galaxies, &real_galaxies) != 0) {
        printf("Failed to read real galaxy data\n");
        return(-1);
    }

    Galaxies fake_galaxies;
    if (galaxiesFromFile(argv[2], &num_galaxies, &fake_galaxies) != 0) {
        printf("Failed to read fake galaxy data\n");
        return(-1);
    }

    // Allocate memory for all the histograms
    int* DD_hist = (int *) malloc(NUM_BINS*sizeof(int));
    int* DR_hist = (int *) malloc(NUM_BINS*sizeof(int));
    int* RR_hist = (int *) malloc(NUM_BINS*sizeof(int));

    // calculate the histograms based with the given data sets
    calculate_histogram(DD_hist, num_galaxies, real_galaxies, real_galaxies);
    calculate_histogram(DR_hist, num_galaxies, real_galaxies, fake_galaxies);
    calculate_histogram(RR_hist, num_galaxies, fake_galaxies, fake_galaxies);

    // Check that the histogram size is the expected size for each histogram
    long hist_sum;
    long expected = (long) num_galaxies * (long) num_galaxies;

    hist_sum = check_histogram_size(DD_hist);
    if (hist_sum != expected) {
        printf("Expected %ld DD histograms, got %ld\n", expected, hist_sum);
        printf("Exiting...\n");
        return(-1);
    }

    hist_sum = check_histogram_size(DR_hist);
    if (hist_sum != expected) {
        printf("Expected %ld DR histograms, got %ld\n", expected, hist_sum);
        printf("Exiting...\n");
        return(-1);
    }

    hist_sum = check_histogram_size(RR_hist);
    if (hist_sum != expected) {
        printf("Expected %ld RR histograms, got %ld\n", expected, hist_sum);
        printf("Exiting...\n");
        return(-1);
    }

    // Calculate omegas
    float* omegas = (float *) malloc(NUM_BINS * sizeof(float));
    calculate_omegas(omegas, DD_hist, DR_hist, RR_hist);

    // Write the histograms and the omega values to files
    write_histogram("dd_histogram.txt", DD_hist);
    write_histogram("dr_histogram.txt", DR_hist);
    write_histogram("rr_histogram.txt", RR_hist);
    write_omegas("omegas.txt", omegas);

    // Convert third arg to number and print that many lines. atoi gives 0 when the string is not a valid number.
    int bins_to_print = atoi(argv[3]);
    if (bins_to_print != 0) {
        print_histogram("DD", DD_hist, bins_to_print);
        print_histogram("DR", DR_hist, bins_to_print);
        print_histogram("RR", RR_hist, bins_to_print);
        print_omegas(omegas, bins_to_print);
    }

    // Free memory
    free(DD_hist);
    free(DR_hist);
    free(RR_hist);
    free(omegas);

    return(0);
}

__global__ void calculate_angles(float *alphas1, float *deltas1, float *alphas2, float *deltas2, int num_galaxies, int *gpu_hist) {
    int idx = blockDim.x*blockIdx.x + threadIdx.x; // Get id from cuda

    // Return early if the id is to
    if (idx >= num_galaxies) {
        return;
    }

    for (int i = 0; i < num_galaxies; i++){
        float angle = 0.0f;
        float alpha1 = alphas1[i];
        float alpha2 = alphas2[idx];
        float delta1 = deltas1[i];
        float delta2 = deltas2[idx];

        float x = sin(delta1) * sin(delta2) + cos(delta1) * cos(delta2) * cos(alpha1 - alpha2);
        angle = acosf(fmaxf(-1.0f, fminf(x, 1.0f))) * RAD_TO_DEG;

        int index = (int)(floor(angle * (1.0f / BIN_WIDTH))) % NUM_BINS;

        // Add atomically
        atomicAdd(&gpu_hist[index], 1);
    }
}

void calculate_histogram(int* histogram, int num_galaxies, Galaxies galaxies, Galaxies other) {

    // Declare angle arrays that will be accessible on GPU
    float *galaxy_gpu_alphas;
    float *galaxy_gpu_deltas;
    float *other_gpu_alphas;
    float *other_gpu_deltas;
    int *gpu_histogram;

    // Size of the data sets (galaxy arrays)
    float data_size = num_galaxies * sizeof(float);

    // Allocate memory on GPU for angle arrays
    cudaMalloc((void**) &galaxy_gpu_alphas, data_size);
    cudaMalloc((void**) &galaxy_gpu_deltas, data_size);
    cudaMalloc((void**) &other_gpu_alphas, data_size);
    cudaMalloc((void**) &other_gpu_deltas, data_size);
    cudaMalloc((void**) &gpu_histogram, NUM_BINS*sizeof(int));

    // Copy angles from CPU onto GPU
    cudaMemcpy(galaxy_gpu_alphas, galaxies.alphas, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(galaxy_gpu_deltas, galaxies.deltas, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(other_gpu_alphas, other.alphas, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(other_gpu_deltas, other.deltas, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_histogram, histogram, data_size, cudaMemcpyHostToDevice);

    // Calculate GPU related sizes. Adjust if needed.
    int warp_size = 32;
    int threads_in_block = 8 * warp_size;
    int blocks_in_grid = ceil((num_galaxies + threads_in_block) / threads_in_block);

    printf("Threads in block: %d\n", threads_in_block);
    printf("Blocks in grid: %d\n", blocks_in_grid);
    printf("Total number of threads: %d\n", threads_in_block*blocks_in_grid);

    // Define the grid size (blocks per grid)
    dim3 dim_grid(blocks_in_grid);

    // Define block size (threads per block)
    dim3 dim_block(threads_in_block);

    // Initialize the histogram as all zeros
    cudaMemset(gpu_histogram, 0, NUM_BINS * sizeof(int));

    // Calculate angles between galaxies[i] and every galaxy in other
    calculate_angles<<<dim_grid, dim_block>>>(galaxy_gpu_alphas, galaxy_gpu_deltas, other_gpu_alphas, other_gpu_deltas, num_galaxies, gpu_histogram);

    // Copy result histogram into CPU histogram
    cudaMemcpy(histogram, gpu_histogram, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);

    // Free all the memory we allocated on the GPU
    cudaFree(galaxy_gpu_alphas);
    cudaFree(galaxy_gpu_deltas);
    cudaFree(other_gpu_alphas);
    cudaFree(other_gpu_deltas);
    cudaFree(gpu_histogram);
}

void calculate_omegas(float* omegas, int* DD, int* DR, int* RR) {
    for(int i = 0; i < NUM_BINS; i++){
        if(RR[i] != 0.0f){
            omegas[i] = (DD[i] - 2.0f*DR[i] + RR[i]) / RR[i];
        }else{
            omegas[i] = 0.0f;
        }
    }
}

void print_omegas(float* omegas, int bins_to_print) {
    for (int i = 0; i < bins_to_print; i++){
        printf("omega [%d]: %f\n", i, omegas[i]);
    }
}

void print_histogram(const char* label, int *histogram, int bins_to_print) {
    printf("First %d %s:\n", bins_to_print, label);
    long long galaxies_counted = 0;
    // Print each bucket bin that has 1 or more galaxy-pair-angle in it.
    for (int i = 0; i < bins_to_print; i++) {
        float bucket_min = (float)i / (1.0f / BIN_WIDTH);
        float bucket_max = (float)i / (1.0f / BIN_WIDTH) + BIN_WIDTH;
        int bucket_value = histogram[i];

        galaxies_counted += histogram[i];
        printf("[%f, %f]: %d\n", bucket_min, bucket_max, bucket_value);
    }
}

void write_omegas(const char *filename, float *omegas) {
    FILE *file;
    file = fopen(filename, "w");

    for (int i = 0; i < NUM_BINS; i++) {
        fprintf(file, "%f\n", omegas[i]);
    }
    fclose(file);
}

void write_histogram(const char *filename, int *histogram) {
    FILE *file;
    file = fopen(filename, "w");

    for (int i = 0; i < NUM_BINS; i++) {
        fprintf(file, "%d\n", histogram[i]);
    }

    fclose(file);
}

long check_histogram_size(int *histogram) {
    long sum = 0;
    for (int i = 0; i < NUM_BINS; i++) {
        sum += histogram[i];
    }
    return sum;
}

int getDevice(int deviceNo) {

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Found %d CUDA devices\n",deviceCount);
    if ( deviceCount < 0 || deviceCount > 128 ) return(-1);
    int device;
    for (device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printf("Device %s        =          %d\n", deviceProp.name,device);
        printf("compute capability            =        %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("totalGlobalMemory             =       %.2lf GB\n", deviceProp.totalGlobalMem/1000000000.0);
        printf("l2CacheSize                   =   %8d B\n", deviceProp.l2CacheSize);
        printf("regsPerBlock                  =   %8d\n", deviceProp.regsPerBlock);
        printf("multiProcessorCount           =   %8d\n", deviceProp.multiProcessorCount);
        printf("maxThreadsPerMultiprocessor   =   %8d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("sharedMemPerBlock             =   %8d B\n", (int)deviceProp.sharedMemPerBlock);
        printf("warpSize                      =   %8d\n", deviceProp.warpSize);
        printf("clockRate                     =   %8.2lf MHz\n", deviceProp.clockRate/1000.0);
        printf("maxThreadsPerBlock            =   %8d\n", deviceProp.maxThreadsPerBlock);
        printf("asyncEngineCount              =   %8d\n", deviceProp.asyncEngineCount);
        printf("f to lf performance ratio     =   %8d\n", deviceProp.singleToDoublePrecisionPerfRatio);
        printf("maxGridSize                   =   %d x %d x %d\n",
               deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("maxThreadsDim in thread block =   %d x %d x %d\n",
               deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("concurrentKernels             =   ");

        if (deviceProp.concurrentKernels ==1 ) {
            printf("     yes\n");
        } else {
            printf("    no\n");
        }

        printf("deviceOverlap                 =   %8d\n", deviceProp.deviceOverlap);

        if(deviceProp.deviceOverlap == 1) {
            printf("Concurrently copy memory/execute kernel\n");
        }
    }

    cudaSetDevice(deviceNo);
    cudaGetDevice(&device);

    if (device != 0) {
        printf("Unable to set device 0, using %d instead", device);
    } else {
        printf("Using CUDA device %d\n\n", device);
    }

    return(0);
}

int galaxiesFromFile(const char *filename, int *num_galaxies, Galaxies *galaxies) {
    FILE *file = fopen(filename, "r");

    // Failed to open file, return early.
    if (!file) {
        printf("Error opening file %s\n", filename);
        return(-1);
    }

    // Read the first number into num_galaxies (100000 in our case).
    fscanf(file, "%d", num_galaxies);

    // Allocate memory for the alpha and delta arrays
    float *alphas = (float *)malloc(*num_galaxies * sizeof(float));
    float *deltas = (float *)malloc(*num_galaxies * sizeof(float));

    if (!alphas || !deltas) {
        printf("Memory allocation failed\n");
        fclose(file);
        return(-1);
    }

    // Loop through the file and scan the value into alpha and delta.
    // Then add the value to the array
    float alpha, delta;
    for (int i = 0; i < *num_galaxies; i++) {
        fscanf(file, "%f %f", &alpha, &delta);
        alphas[i] = alpha * ARC_MIN_TO_RAD;
        deltas[i] = delta * ARC_MIN_TO_RAD;
    }

    fclose(file);

    // Set the alpha and delta values for the array
    galaxies->alphas = alphas;
    galaxies->deltas = deltas;

    return(0);
}
