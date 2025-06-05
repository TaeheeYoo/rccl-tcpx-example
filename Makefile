-include build.mk

all:
	clang++ -o hip_rccl_example hip_rccl_example.cpp -D__HIP_PLATFORM_AMD__ -I. -I/opt/rocm-6.2.0/include -L. -L/opt/rocm-6.2.0/lib -lrccl -lamdhip64 -lstdc++ -lstdc++

run:
	NCCL_SOCKET_IFNAME=$(IFNAME)	\
	NCCL_IB_DISABLE=1				\
	NCCL_DEBUG=$(DEBUG_LEVEL)		\
	NCCL_NET_PLUGIN=$(PLUGIN_NAME)	\
	NCCL_TCPX_IFNAMES=$(IFNAME)		\
	LD_LIBRARY_PATH=$(shell pwd)	\
		./hip_rccl_example $(file < run.mk)
