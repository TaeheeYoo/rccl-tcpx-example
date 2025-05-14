all:
	hipcc -o hip_rccl_example hip_rccl_example.cpp -L. -L /opt/rocm-6.4.0/lib -lrccl -lmpi

