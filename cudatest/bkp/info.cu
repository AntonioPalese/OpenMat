#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>


void get_cuda_infos(int device_idx)
{
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device_idx);

    printf("device : %d, name : %s \n", device_idx, props.name);
    printf("device : %d, total memory : %f GB\n", device_idx, (float)(props.totalGlobalMem / pow(2, 32)));
    printf("device : %d, constant memory : %f GB\n", device_idx, (float)(props.totalConstMem / pow(2, 32)));

}


int main(int argc, char** argv)
{
    int device;

    if(argc == 2)
        device = atoi(argv[1]);
    else 
        device = 0;   
    
    get_cuda_infos(device);
}