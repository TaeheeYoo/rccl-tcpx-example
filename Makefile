all:
	hipcc -o hip_rccl_example hip_rccl_example.cpp -lrccl -lmpi

