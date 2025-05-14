#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <mpi.h>

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

void check_nccl(ncclResult_t result)
{
	if (result != ncclSuccess) {
		std::cerr << "RCCL error: " << ncclGetErrorString(result) << std::endl;
		exit(1);
	}
}

using namespace std;
int main(int argc, char *argv[])
{
	int size, rank = std::stoi(argv[1]);
	hipDeviceProp_t devProp;
	ncclUniqueId comm_id;
	ncclComm_t comm;
	float *d_data;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (rank == 0) {
		ncclGetUniqueId(&comm_id);
		std::ofstream out("nccl_unique_id.txt", std::ios::binary);
		out.write((char *)&comm_id, sizeof(comm_id));
		out.close();
	}

	if (rank != 0) {
		std::ifstream in("nccl_unique_id.txt", std::ios::binary);
		in.read((char *)&comm_id, sizeof(comm_id));
		in.close();
	}

	/* TODO
	 * 0 to local rank
	 */
	HIP_CHECK(hipGetDeviceProperties(&devProp, 0));
	cout << " System minor " << devProp.minor << endl;
	cout << " System major " << devProp.major << endl;
	cout << " agent prop name " << devProp.name << endl;
	cout << "hip Device prop succeeded " << endl ;

	HIP_CHECK(hipMalloc(&d_data, N * sizeof(float)));

	check_nccl(ncclCommInitRank(&comm, size, comm_id, rank));

	std::vector<float> h_data(N, rank + 1.0f);
	HIP_CHECK(hipMemcpy(d_data, h_data.data(), N * sizeof(float),
			    hipMemcpyHostToDevice));

	check_nccl(ncclAllReduce(d_data, d_data, N, ncclFloat, ncclSum, comm, 0));

	HIP_CHECK(hipMemcpy(h_data.data(), d_data, N * sizeof(float),
		            hipMemcpyDeviceToHost));
	std::cout << "Rank " << rank << " result: " << h_data[0] << std::endl;

	HIP_CHECK(hipFree(d_data));
	ncclCommDestroy(comm);

	MPI_Finalize();
	return 0;
}

