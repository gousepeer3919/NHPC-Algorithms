#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv) {

	int size, rank;
	int max_size;
	int* src = NULL, *orig_dst=NULL, *temp_buff=NULL, *dbt_dst=NULL;
	MPI_Status status[3];
	MPI_Request request[3];
	struct timeval start_time, end_time;
	float total_time = 0.0;
	int i, j;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	max_size = 1024;

	// allocate memory
	src = (int *)malloc(max_size * sizeof(int));
	orig_dst = (int *)malloc(max_size * sizeof(int));
	temp_buff = (int *)malloc(max_size * sizeof(int));
	dbt_dst = (int *)malloc(max_size * sizeof(int));
	if(src == NULL || orig_dst == NULL || temp_buff == NULL || dbt_dst == NULL) {
		perror("Cannot allocate memory\n");
		exit(0);
	}

	// initialize buffers
	for(i = 0; i < max_size; i++) {
		src[i] = rank + i;
		dbt_dst[i] = src[i];
		temp_buff[i] = 0;
	}
	
	// MPI AllReduce
	gettimeofday(&start_time, NULL);
	MPI_Allreduce(src, orig_dst, max_size, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	gettimeofday(&end_time, NULL);
	total_time = ((end_time.tv_sec - start_time.tv_sec) * 1000000.0 + (end_time.tv_usec - start_time.tv_usec));
	if(rank == 0) {
		printf("MPI_Allreduce (%d nodes, %d bytes message size) Time taken : %f (usecs)\n", size, max_size * 4, total_time);
	}

	// Double Binary Tree Algorithm (Works only for power of 2 nodes)
	
	// ========================= Reduce Scatter ================================

	MPI_Barrier(MPI_COMM_WORLD);
	gettimeofday(&start_time, NULL);

	int height = 0, peer = 0;
	int steps = (int) log2(size) - 1;
	int index = ((rank % 2 == 0) ? rank : rank + 1);	
	if(rank > 0) {
		while((index & (1 << height)) == 0)
			height++;	
	}

	// First step is to send data from leaf nodes
	index = ((rank % 2 == 0) ? rank : rank - 1);		
	peer = ((index % 4 == 0) ? rank + 1 : rank - 1);	

	MPI_Isend(dbt_dst, max_size, MPI_INT, peer, peer, MPI_COMM_WORLD, &request[0]);

	if(height == 1) {
		MPI_Irecv(dbt_dst, max_size, MPI_INT, rank - height, MPI_ANY_TAG, MPI_COMM_WORLD, &request[1]);
		MPI_Irecv(temp_buff, max_size, MPI_INT, rank + height, MPI_ANY_TAG, MPI_COMM_WORLD, &request[2]);
		MPI_Waitall(3, request, status);

		#pragma omp parallel for
		for(j = 0; j < max_size; j++) {
			dbt_dst[j] += temp_buff[j];
		}
	}

	else {
		MPI_Waitall(1, request, status);	
	}

	int child_offset = 0;
	index = ((rank % 2 == 0) ? rank : rank + 1);

	for(i = 1; i < steps; i++) {

		if(height == i) {
			peer = ((index - (1 << height)) % (1 << (2 + height))) == 0 ? rank + (1 << height) : rank - (1 << height);
			MPI_Isend(dbt_dst, max_size, MPI_INT, peer, peer, MPI_COMM_WORLD, &request[0]);
			MPI_Waitall(1, request, status);
		}

		if(height == i + 1) {
			child_offset = (1 << (height - 1));
			MPI_Irecv(dbt_dst, max_size, MPI_INT, rank - child_offset, MPI_ANY_TAG, MPI_COMM_WORLD, &request[0]);
			MPI_Irecv(temp_buff, max_size, MPI_INT, rank + child_offset, MPI_ANY_TAG, MPI_COMM_WORLD, &request[1]);
			MPI_Waitall(2, request, status);

			#pragma omp parallel for
			for(j = 0; j < max_size; j++) {
				dbt_dst[j] += temp_buff[j];
			}
		}
	}

	if(rank == (size / 2) || rank == (size / 2 - 1)) {
		MPI_Isend(dbt_dst, max_size, MPI_INT, 0, 0, MPI_COMM_WORLD, &request[0]);
		MPI_Waitall(1, request, status);
	}

	if(rank == 0) {
		MPI_Irecv(dbt_dst, max_size, MPI_INT, size / 2, MPI_ANY_TAG, MPI_COMM_WORLD, &request[0]);
		MPI_Irecv(temp_buff, max_size, MPI_INT, size / 2 - 1, MPI_ANY_TAG, MPI_COMM_WORLD, &request[1]);
		MPI_Waitall(2, request, status);

		#pragma omp parallel for
		for(j = 0; j < max_size; j++) {
			dbt_dst[j] += temp_buff[j];
		}
	}

	if(rank == 0) {
		MPI_Isend(dbt_dst, max_size, MPI_INT, size / 2, 0, MPI_COMM_WORLD, &request[0]);
		MPI_Isend(dbt_dst, max_size, MPI_INT, size / 2 - 1, 0, MPI_COMM_WORLD, &request[1]);
		MPI_Isend(dbt_dst, max_size, MPI_INT, size - 1, 0, MPI_COMM_WORLD, &request[2]);
		MPI_Waitall(3, request, status);
	}

	if(rank == (size / 2) || rank == (size / 2 - 1) || rank == size - 1) {
		MPI_Irecv(dbt_dst, max_size, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &request[0]);
		MPI_Waitall(1, request, status);
	}

	// ========================= All Gather ================================

	for(i = steps; i > 1; i--) {

		if(height == i) {
			child_offset = (1 << (height - 1));
			MPI_Isend(dbt_dst, max_size, MPI_INT, rank - child_offset, rank, MPI_COMM_WORLD, &request[0]);
			MPI_Isend(dbt_dst, max_size, MPI_INT, rank + child_offset, rank, MPI_COMM_WORLD, &request[1]);
			MPI_Waitall(2, request, status);
		}

		if(height == i - 1) {
			peer = ((index - (1 << height)) % (1 << (2 + height))) == 0 ? rank + (1 << height) : rank - (1 << height);
			MPI_Irecv(dbt_dst, max_size, MPI_INT, peer, MPI_ANY_TAG, MPI_COMM_WORLD, &request[0]);
			MPI_Waitall(1, request, status);
		}
	}

	// ========================== Timing and Correctness ==================================
	if(rank == 0) {
		gettimeofday(&end_time, NULL);
		total_time = ((end_time.tv_sec - start_time.tv_sec) * 1000000.0 + (end_time.tv_usec - start_time.tv_usec));
		printf("Double Binary Tree (%d nodes, %d bytes message size) Time taken : %f (usecs)\n", size, max_size * 4, total_time);
	}

	if(memcmp(orig_dst, dbt_dst, max_size*sizeof(int))) {
		printf("Rank : %d failed correctness\n", rank);
	}
	
	MPI_Finalize();
	return 0;
}

