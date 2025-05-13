#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <fstream>
#include <iostream>
#include <vector>

#define N 1024  // 데이터 크기

void check_nccl(ncclResult_t result) {
	if (result != ncclSuccess) {
		std::cerr << "RCCL error: " << ncclGetErrorString(result) << std::endl;
		exit(1);
	}
}

using namespace std;
int main(int argc, char *argv[]) {
	int rank, size;
	size = 1;  // 예시로 2개의 GPU를 사용한다고 가정
	rank = std::stoi(argv[1]);  // rank는 커맨드라인 인자로 받음

	// RCCL을 위한 커뮤니케이터 및 리소스
	ncclComm_t comm;
	ncclUniqueId comm_id;

	// 첫 번째 노드에서만 comm_id를 생성하고, 나머지 노드에 전달
	if (rank == 0) {
		ncclGetUniqueId(&comm_id);
		std::ofstream out("nccl_unique_id.txt", std::ios::binary);
		out.write((char *)&comm_id, sizeof(comm_id));
		out.close();
	}

	// 다른 노드에서 comm_id를 읽어들임
	if (rank != 0) {
		std::ifstream in("nccl_unique_id.txt", std::ios::binary);
		in.read((char *)&comm_id, sizeof(comm_id));
		in.close();
	}

	// GPU 리소스 설정
	int device;
	hipSetDevice(rank);  // rank에 맞는 GPU 선택

	// GPU 메모리 할당
	float *d_data;
	hipMalloc(&d_data, N * sizeof(float));

	// RCCL 커뮤니케이션 초기화
	check_nccl(ncclCommInitRank(&comm, size, comm_id, rank));

	// 각 프로세스에서 데이터를 준비하고 RCCL로 전달
	std::vector<float> h_data(N, rank + 1.0f);  // rank 값으로 초기화
	hipMemcpy(d_data, h_data.data(), N * sizeof(float), hipMemcpyHostToDevice);

	// RCCL 리덕션 연산 예: 하나의 GPU에서 다른 GPU로 데이터 합치기
	check_nccl(ncclAllReduce(d_data, d_data, N, ncclFloat, ncclSum, comm, 0));

	// 결과 출력
	hipMemcpy(h_data.data(), d_data, N * sizeof(float), hipMemcpyDeviceToHost);
	std::cout << "Rank " << rank << " result: " << h_data[0] << std::endl;

	// 리소스 해제
	hipFree(d_data);
	ncclCommDestroy(comm);

	return 0;
}

