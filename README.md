# 1. Build
```bash
make all RCCL_ROOT=/opt/rccl-gfx1030/
```

# 2. Run
```bash
make run SERVER=0 ADDRESS=192.168.0.49 IFNAME=enp7s0	    \
		 LD_LIBRARY_PATH=$(pwd):/opt/rccl-gfx1030/lib		\
		 ADDITIONAL_ENV=HSA_OVERRIDE_GFX_VERSION=10.3.0
```

`librccl-net-tcpx.so` in the `LD_LIBRARY_PATH`.
