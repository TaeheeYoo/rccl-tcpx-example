CC := clang++

CPPFLAGS := -D__HIP_PLATFORM_AMD__ -I/opt/rccl-gfx1030/include

LDLIBS := -lamdhip64 -lrccl
LDFLAGS := -L/opt/rccl-gfx1030/lib
