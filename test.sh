./hip_rccl_example 0 &
sleep 3
scp -P7674 ./nccl_unique_id.txt ap@192.168.1.4:/home/ap/
export NCCL_SOCKET_IFNAME=enp9s0f3np3
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
