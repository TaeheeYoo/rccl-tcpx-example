CC := clang++

LDLIBS := -lamdhip64 -lrccl
CPPFLAGS := -D__HIP_PLATFORM_AMD__
