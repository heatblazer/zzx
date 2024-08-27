#ifndef CONVACCEL_H
#define CONVACCEL_H

#ifdef NOGPU


struct gpu_kernel
{

    gpu_kernel() = default;

    ~gpu_kernel() = default;

    void conv3x3(unsigned int *pixels, unsigned int x, unsigned int y, unsigned int w, unsigned int h , float (&kern)[9])
    {
        (void)pixels; (void)x; (void)y;  (void) w; (void)h; (void)kern;
    }

    void conv5x5(unsigned int* pixels, unsigned long x, unsigned long  y ,unsigned long w, unsigned long h, float (&data)[3][25])
    {
        (void)pixels; (void)x; (void)y;  (void) w; (void)h; (void)data;

    }

    void warmer(void)
    {

    }
};
#else
//!TODO
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <CL/opencl.h>

struct gpu_kernel
{

    gpu_kernel();

    ~gpu_kernel();

    void conv3x3(unsigned int *pixels, unsigned int x, unsigned int y, unsigned int w, unsigned int h , float (&kern)[9]);

    void conv5x5(unsigned int* pixels, size_t x, size_t y ,size_t w, size_t h, float (&data)[3][25]);

    void warmer(void);

private:



    enum  eProgramSelect {
        Warmer,
        Convolve,
        Size
    };

    struct {
        cl_command_queue command_queue ;
        cl_context context ;
        cl_device_id device ;
        size_t global_work_size ;
        cl_kernel kernel ;
        cl_platform_id platform ;
        cl_program program ;
        cl_int compilerr ;
        cl_uint platforms;
        cl_uint numdevs ;
    } m_kernctx[eProgramSelect::Size];

};


#endif


#endif // CONVACCEL_H
