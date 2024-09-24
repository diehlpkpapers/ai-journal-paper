
# Integration
- Used one variable per CUDA thread and sum each individual result on the host side (instead of using *atomicAdd* on a shared global variable)
