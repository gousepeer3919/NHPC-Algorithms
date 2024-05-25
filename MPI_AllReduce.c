#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define MAX_SIZE (1024 * 1024)
#define ITERATIONS 1000
#define WARMUP 10

int main(int argc, char **argv) {
    int size, rank;
    int src_rank = 0;
    int *src = NULL, *dst = NULL;
    int i;
    MPI_Status status;
    struct timeval start, end;
    float total_time = 0;

    // Allocate memory for source and destination arrays
    src = (int *)malloc(MAX_SIZE * sizeof(int));
    dst = (int *)malloc(MAX_SIZE * sizeof(int));

    // Check for memory allocation errors
    if (src == NULL || dst == NULL) {
        perror("Memory allocation failed\n");
        exit(0);
    }

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the size and rank of the MPI communicator
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Display information if the current rank is the source rank
    if (rank == src_rank)
        printf("Latency Benchmark - MPI All Reduce\n");

    // Initialize the source array with rank values
    for (i = 0; i < MAX_SIZE; i++) {
        src[i] = rank;
        // printf("Rank %d: Element at index %d has value %2d\n", rank, i, src[i]);
    }

    // Warm-up phase to ensure accurate measurements
    for (int current_size = 4; current_size <= MAX_SIZE; current_size *= 2) {
        for (i = 0; i < WARMUP; i++) {
            MPI_Allreduce(src, dst, current_size, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        }
    }

    // Benchmarking phase
    if (rank == src_rank)
        printf("%10s   %10s usecs\n", "Size(4*num of el)", "Latency (usecs)");

    for (int current_size = 1; current_size <= MAX_SIZE; current_size *= 2) {
        // Synchronize all ranks before starting measurements
        MPI_Barrier(MPI_COMM_WORLD);

        gettimeofday(&start, NULL);

        // Perform MPI All Reduce for the specified number of iterations
        for (i = 0; i < ITERATIONS; i++) {
            MPI_Allreduce(src, dst, current_size, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        }

        gettimeofday(&end, NULL);

        // Calculate and display the average time per iteration
        total_time = ((end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec)) / ITERATIONS;

        if (rank == src_rank) {
            printf("%10d   %10.2f\n", 4 * current_size, total_time);
        }
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}

