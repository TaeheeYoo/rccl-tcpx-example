CXX := clang++

CPPFLAGS := -D__HIP_PLATFORM_AMD__ -I$(RCCL_ROOT)/include

LDLIBS := -lamdhip64 -lrccl
LDFLAGS := -L$(RCCL_ROOT)/lib
