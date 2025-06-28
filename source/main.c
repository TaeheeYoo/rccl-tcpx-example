#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <rccl/rccl.h>

#define N 1024
#define NODES 2

#define HIP_CHECK(expression)                  				\
{                                              				\
    const hipError_t status = expression;      				\
    if(status != hipSuccess) {                  			\
        fprintf(stderr,"HIP error %d: %s: at %s:%d",			\
		       status, hipGetErrorString(status),		\
		       __FILE__, __LINE__);				\
    }									\
}

void show_comm_id(ncclUniqueId id)
{
	for (int i = 0; i < NCCL_UNIQUE_ID_BYTES; i++)
		printf("%x", id.internal[i]);

	putchar('\n');
}

void check_nccl(ncclResult_t result)
{
	if (result != ncclSuccess) {
		printf("RCCL error: %s\n", ncclGetErrorString(result));
		exit(1);
	}
}

int reuse_address(int fd)
{
	int sockopt = 1;

	if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, 
		       &sockopt, sizeof(sockopt)) == -1)
		return -1;

	return 0;
}

void send_unique_id(ncclUniqueId comm_id, const char *ipaddr)
{
	struct sockaddr_in sockaddr;
	int servfd, clntfd, retval;

	servfd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	if (servfd == -1)
		perror("socket() error: ");

	if (reuse_address(servfd) == -1)
		perror("reuse_address() error: ");

	memset(&sockaddr, 0x00, sizeof(struct sockaddr_in));

	sockaddr.sin_family = AF_INET;
	sockaddr.sin_addr.s_addr = inet_addr(ipaddr);
	sockaddr.sin_port = htons(4091);

	if (bind(servfd, (struct sockaddr *) &sockaddr, sizeof(sockaddr)) != 0)
		perror("bind() error: ");

	if (listen(servfd, 15) != 0)
		perror("listen() error: ");

	clntfd = accept(servfd, NULL, 0);
	if (clntfd == -1)
		perror("accept() error: ");

	retval = send(clntfd, &comm_id, sizeof(ncclUniqueId), 0);

	close(clntfd);

	close(servfd);
}

ncclUniqueId recv_unique_id(const char *ipaddr)
{
	struct sockaddr_in sockaddr;
	ncclUniqueId comm_id;
	int clntfd, retval;

	clntfd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	if (clntfd == -1)
		perror("socket() error: ");

	memset(&sockaddr, 0x00, sizeof(struct sockaddr_in));

	sockaddr.sin_family = AF_INET;
	sockaddr.sin_addr.s_addr = inet_addr(ipaddr);
	sockaddr.sin_port = htons(4091);

	retval = connect(clntfd, (struct sockaddr *) &sockaddr, sizeof(sockaddr));
	if (retval == -1)
		perror("accept() error");

	retval = recv(clntfd, &comm_id, sizeof(ncclUniqueId), 0);

	close(clntfd);

	return comm_id;
}

int main(int argc, char *argv[])
{
	ncclUniqueId comm_id;

	hipDeviceProp_t devProp;
	ncclComm_t comm;
	hipStream_t s;

	float *d_data, *h_data;
	int rank;

	if (argc != 3) {
		fprintf(stderr, "usage: %s <rank> <address>\n", argv[0]);
		exit(EXIT_FAILURE);
	}
	
	rank = strtol(argv[1], NULL, 10);
	if (rank == 0) {
		ncclGetUniqueId(&comm_id);
		send_unique_id(comm_id, argv[2]);
	} else {
		comm_id = recv_unique_id(argv[2]);
	}

	show_comm_id(comm_id);

	h_data = malloc(sizeof(float) * N);
	if (h_data == NULL) {
		fprintf(stderr, "failed to malloc(): %s", strerror(errno));
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < N; i++)
		h_data[i] = rank + 1.0f;

	HIP_CHECK(hipGetDeviceProperties(&devProp, 0));
	printf("System minor %d\n", devProp.minor);
	printf("System major %d\n", devProp.major);
	printf("Hip Device Prop Succeeded!\n");

	HIP_CHECK(hipStreamCreate(&s));

	HIP_CHECK(hipMalloc((void **) &d_data, N * sizeof(float)));

	check_nccl(ncclCommInitRank(&comm, NODES, comm_id, rank));

	HIP_CHECK(hipMemcpy(d_data, h_data, N * sizeof(float),
			    hipMemcpyHostToDevice));

	if (!rank) {
		check_nccl(ncclSend(d_data, N, ncclFloat, 1, comm, s));
	} else {
		check_nccl(ncclRecv(d_data, N, ncclFloat, 0, comm, s));
	}

	HIP_CHECK(hipMemcpy(h_data, d_data, N * sizeof(float),
		            hipMemcpyDeviceToHost));

	printf("Rank %d result: %f\n", rank, h_data[0]);

	check_nccl(ncclCommDestroy(comm));

	HIP_CHECK(hipFree(d_data));
	HIP_CHECK(hipStreamDestroy(s));

	return 0;
}
