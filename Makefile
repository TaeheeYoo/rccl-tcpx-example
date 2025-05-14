all:
	hipcc -o hip_rccl_example hip_rccl_example.cpp -I /usr/lib/x86_64-linux-gnu/openmpi/include -L. -L /opt/rocm-6.4.0/lib -lrccl -lmpi

