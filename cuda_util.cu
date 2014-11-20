#include <string.h>
#include <ctype.h>
#ifndef _WIN32
#include <unistd.h>
#endif

#include "miner.h"

extern "C" char *device_name[8];
extern "C" int device_map[8];

extern "C" int cuda_num_devices()
{
    int version;
    cudaError_t err = cudaDriverGetVersion(&version);
    if (err != cudaSuccess)
    {
        applog(LOG_ERR, "Unable to query CUDA driver version! Is an nVidia driver installed?");
        exit(1);
    }

    int maj = version / 1000, min = version % 100; // same as in deviceQuery sample
    if (maj < 5 || (maj == 5 && min < 5))
    {
        applog(LOG_ERR, "Driver does not support CUDA %d.%d API! Update your nVidia driver!", 5, 5);
        exit(1);
    }

    int GPU_N;
    err = cudaGetDeviceCount(&GPU_N);
    if (err != cudaSuccess)
    {
        applog(LOG_ERR, "Unable to query number of CUDA devices! Is an nVidia driver installed?");
        exit(1);
    }
    return GPU_N;
}

extern "C" void cuda_devicenames()
{
    cudaError_t err;
    int GPU_N;
    err = cudaGetDeviceCount(&GPU_N);
    if (err != cudaSuccess)
    {
        applog(LOG_ERR, "Unable to query number of CUDA devices! Is an nVidia driver installed?");
        exit(1);
    }

    for (int i=0; i < GPU_N; i++)
    {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, device_map[i]);

        device_name[i] = strdup(props.name);
    }
}

static bool substringsearch(const char *haystack, const char *needle, int &match)
{
    int hlen = strlen(haystack);
    int nlen = strlen(needle);
    for (int i=0; i < hlen; ++i)
    {
        if (haystack[i] == ' ') continue;
        int j=0, x = 0;
        while(j < nlen)
        {
            if (haystack[i+x] == ' ') {++x; continue;}
            if (needle[j] == ' ') {++j; continue;}
            if (needle[j] == '#') return ++match == needle[j+1]-'0';
            if (tolower(haystack[i+x]) != tolower(needle[j])) break;
            ++j; ++x;
        }
        if (j == nlen) return true;
    }
    return false;
}

extern "C" int cuda_finddevice(char *name)
{
    int num = cuda_num_devices();
    int match = 0;
    for (int i=0; i < num; ++i)
    {
        cudaDeviceProp props;
        if (cudaGetDeviceProperties(&props, i) == cudaSuccess)
            if (substringsearch(props.name, name, match)) return i;
    }
    return -1;
}
