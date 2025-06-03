#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <fstream>
#include <iostream>
#include <vector>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#define N 1024

#define HIP_CHECK(expression)                  \
{                                              \
    const hipError_t status = expression;      \
    if(status != hipSuccess){                  \
        std::cerr << "HIP error "              \
                  << status << ": "            \
                  << hipGetErrorString(status) \
                  << " at " << __FILE__ << ":" \
                  << __LINE__ << std::endl;    \
    }                                          \
}

#define show_comm_id(comm_id)					\
	do {							\
		for (int i = 0; i < NCCL_UNIQUE_ID_BYTES; i++)	\
			printf("%x", comm_id.internal[i]);	\
		printf("\n");					\
	} while(0)

void check_nccl(ncclResult_t result)
{
	if (result != ncclSuccess) {
		std::cerr << "RCCL error: " << ncclGetErrorString(result) << std::endl;
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

using namespace std;
int main(int argc, char *argv[])
{
	int size, rank = std::stoi(argv[1]);
	hipDeviceProp_t devProp;
	ncclUniqueId comm_id;
	ncclComm_t comm;
	float *d_data;
	hipStream_t s;

	size = 2;

	if (rank == 0) {
		ncclGetUniqueId(&comm_id);
		send_unique_id(comm_id, argv[2]);
	} else {
		comm_id = recv_unique_id(argv[2]);
	}
	show_comm_id(comm_id);

	/* TODO
	 * 0 to local rank
	 */
	HIP_CHECK(hipGetDeviceProperties(&devProp, 0));
	cout << " System minor " << devProp.minor << endl;
	cout << " System major " << devProp.major << endl;
	cout << " agent prop name " << devProp.name << endl;
	cout << "hip Device prop succeeded " << endl ;

	HIP_CHECK(hipStreamCreate(&s));

	HIP_CHECK(hipMalloc(&d_data, N * sizeof(float)));

	check_nccl(ncclCommInitRank(&comm, size, comm_id, rank));

	std::vector<float> h_data(N, rank + 1.0f);
	HIP_CHECK(hipMemcpy(d_data, h_data.data(), N * sizeof(float),
			    hipMemcpyHostToDevice));

#if 0
	check_nccl(ncclAllReduce(d_data, d_data, N, ncclFloat, ncclSum, comm, 0));
#else
	if (!rank) {
		check_nccl(ncclSend(d_data, N, ncclFloat, 1, comm, s));
	} else {
		check_nccl(ncclRecv(d_data, N, ncclFloat, 0, comm, s));
	}
#endif

	HIP_CHECK(hipMemcpy(h_data.data(), d_data, N * sizeof(float),
		            hipMemcpyDeviceToHost));
	std::cout << "Rank " << rank << " result: " << h_data[0] << std::endl;

	HIP_CHECK(hipFree(d_data));
	ncclCommDestroy(comm);

	return 0;
}

