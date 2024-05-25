#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

int main(int argc, char **argv) {
  int size, rank, send_to, receive_from;
  int *src = NULL, *tmp_dst = NULL, *ring_dst = NULL, *orig_dst = NULL;
  int max_size = 0;
  int i;

  struct timeval start, end;
  float total_time = 0;

  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  max_size = 4 * size;

  MPI_Request send_request[1];
  MPI_Request recv_request[1];
  MPI_Status status[1];

  printf("Allreduce benchmark\n");

  /* Allocate memory */
  src = (int *)malloc(max_size * sizeof(int));
  tmp_dst = (int *)malloc(max_size * sizeof(int));
  ring_dst = (int *)malloc(max_size * sizeof(int));
  orig_dst = (int *)malloc(max_size * sizeof(int));

  if (src == NULL || tmp_dst == NULL || ring_dst == NULL || orig_dst == NULL) {
    perror("Cannot allocate memory\n");
    exit(0);
  }

  /* Initialize the buffers */
  for (i = 0; i < max_size; i++) {
    src[i] = rank + i;
    printf("rank=%d src index=%d has value %2d\n", rank, i, src[i]);
  }

  /* MPI allreduce */
  MPI_Allreduce(src, orig_dst, max_size, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  /* Ring based allreduce algorithm */
  MPI_Barrier(MPI_COMM_WORLD);
  if (!rank)
    printf("Ring based allreduce algorithm\n");

  for (i = 0; i < max_size; i++) {
    ring_dst[i] = src[i];
    tmp_dst[i] = 0;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  for (i = 0; i < size - 1; i++) {
    send_to = (rank + 1) % size;
    receive_from = (rank - 1 + size) % size;

    MPI_Isend(ring_dst + ((rank - i + size) % size) * (max_size / size), max_size / size, MPI_INT, send_to, send_to, MPI_COMM_WORLD, &send_request[0]);
    MPI_Irecv(tmp_dst + ((rank - i - 1 + size) % size) * (max_size / size), max_size / size, MPI_INT, receive_from, rank, MPI_COMM_WORLD, &recv_request[0]);

    MPI_Waitall(1, send_request, status);
    MPI_Waitall(1, recv_request, status);

    int mem_start = ((rank - i - 1 + size) % size) * (max_size / size);
    for (int ms = mem_start; ms < mem_start + max_size / size; ms++) {
      ring_dst[ms] += tmp_dst[ms];
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  for (i = 0; i < size - 1; i++) {
    send_to = (rank + 1) % size;
    receive_from = (rank - 1 + size) % size;

    MPI_Isend(ring_dst + ((rank + 1 - i + size) % size) * (max_size / size), max_size / size, MPI_INT, send_to, send_to, MPI_COMM_WORLD, &send_request[0]);
    MPI_Irecv(ring_dst + ((rank + 1 - i - 1 + size) % size) * (max_size / size), max_size / size, MPI_INT, receive_from, rank, MPI_COMM_WORLD, &recv_request[0]);

    MPI_Waitall(1, send_request, status);
    MPI_Waitall(1, recv_request, status);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (memcmp(orig_dst, ring_dst, max_size * sizeof(int))) {
    printf("RING algorithm FAILED correctness\n");
  }

  MPI_Barrier(MPI_COMM_WORLD);

  for (int j = 0; j < size; j++) {
    if (j == rank) {
      printf("RING algorithm rank=%d ", rank);
      for (i = 0; i < max_size; i++) {
        printf("%d ", ring_dst[i]);
      }
      printf("\n");
    }
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  for (int j = 0; j < size; j++) {
    if (j == rank) {
      printf("MPI  algorithm rank=%d ", rank);
      for (i = 0; i < max_size; i++) {
        printf("%d ", orig_dst[i]);
      }
      printf("\n");
    }
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  MPI_Finalize();
  return 0;
}

