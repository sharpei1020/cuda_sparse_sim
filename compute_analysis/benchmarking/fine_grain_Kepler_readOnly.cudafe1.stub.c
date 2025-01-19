#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "fine_grain_Kepler_readOnly.fatbin.c"
extern void __device_stub__Z14global_latencyPKjiiPjS1_(const unsigned *__restrict__, int, int, unsigned *, unsigned *);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z14global_latencyPKjiiPjS1_(const unsigned *__restrict__ __par0, int __par1, int __par2, unsigned *__par3, unsigned *__par4){ const unsigned *__T2;
__cudaLaunchPrologue(5);__T2 = __par0;__cudaSetupArgSimple(__T2, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 12UL);__cudaSetupArgSimple(__par3, 16UL);__cudaSetupArgSimple(__par4, 24UL);__cudaLaunch(((char *)((void ( *)(const unsigned *__restrict__, int, int, unsigned *, unsigned *))global_latency)));}
# 154 "fine_grain_Kepler_readOnly.cu"
void global_latency( const unsigned *__restrict__ __cuda_0,int __cuda_1,int __cuda_2,unsigned *__cuda_3,unsigned *__cuda_4)
# 154 "fine_grain_Kepler_readOnly.cu"
{__device_stub__Z14global_latencyPKjiiPjS1_( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4);
# 192 "fine_grain_Kepler_readOnly.cu"
}
# 1 "fine_grain_Kepler_readOnly.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T3) {  __nv_dummy_param_ref(__T3); __nv_save_fatbinhandle_for_managed_rt(__T3); __cudaRegisterEntry(__T3, ((void ( *)(const unsigned *__restrict__, int, int, unsigned *, unsigned *))global_latency), _Z14global_latencyPKjiiPjS1_, (-1)); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
