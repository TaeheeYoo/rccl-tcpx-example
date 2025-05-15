all:
	clang++ -o hip_rccl_example hip_rccl_example.cpp -D__HIP_PLATFORM_AMD__ -I. -I/opt/rocm-6.2.0/include -L. -L/opt/rocm-6.2.0/lib -lrccl -lamdhip64 -lstdc++ -lstdc++

