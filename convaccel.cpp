#include "convaccel.h"
#include <cstring>
#include <cstdio>

#ifdef NOGPU
#else
static const char* warmers = R"(

    __kernel void warmer(__global unsigned int *pixels, unsigned int p)
    {
        int x = get_global_id(0);
        pixels[x] = p;
    }
)";

static const char* src3x3= R"(
    __kernel void conv3x3(__global unsigned int *pixels, unsigned int w, unsigned int h,
    __global float* kern)
    {
        unsigned int MyMat[9];
        int x = get_global_id(0);
        int y = get_global_id(1);
        int k = get_global_id(2);
        int resr = 0, resg=0, resb=0;
        int r,g,b;
        unsigned int it=0;
        for(int i=0; i <3; i++) {
            for(int j=0; j < 3; j++) {
                r = ((pixels[(x+j) + w * (y+i)] >> 16) & 0xFFu) * kern[i * 3 +j];
                g = ((pixels[(x+j) + w * (y+i)] >> 8) & 0xFFu) * kern[i * 3 +j];
                b = ((pixels[(x+j) + w * (y+i)] >> 0) & 0xFFu) * kern[i * 3 +j];
                resr += r;
                resg += g;
                resb += b;
            }
        }
        MyMat[k] = (0xFF000000) | (resr & 0xFFu) << 16 | (resg & 0xFFu) << 8 | (resb & 0xFFu);
        resr=resg=resb=r=g=b=0;
        pixels[y * w + x] = MyMat[4];
//        if (x >= get_global_size(0) && y >= get_global_size(1)) return;
//        else {
//            barrier(CLK_LOCAL_MEM_FENCE);
//        }
    }
)";


gpu_kernel::gpu_kernel()
{
    memset(&m_kernctx, 0 , sizeof (m_kernctx));

}

gpu_kernel::~gpu_kernel()
{
    for(int i=0; i < eProgramSelect::Size; i++) {
        clReleaseKernel(m_kernctx[i].kernel);
        clReleaseCommandQueue(m_kernctx[i].command_queue);
        clReleaseProgram(m_kernctx[i].program);
        clReleaseContext(m_kernctx[i].context);
    }

}



void gpu_kernel::warmer(void)
{
    static unsigned int pixels[1 << 10] = {0};
    size_t dim = 1;
    size_t ws = (1 << 10);
    size_t global_offset[] = {0};
    size_t global_size[] = {1 << 10};
    size_t wsize = 0 ;
    size_t globwsize = ws *  sizeof(unsigned int);
    cl_mem bufferx;

    size_t log_size;
    char* program_log = 0;
    //CL_DEVICE_TYPE_CPU
    //CL_DEVICE_TYPE_ALL
    clGetPlatformIDs(1, &m_kernctx[eProgramSelect::Warmer].platform, &m_kernctx[eProgramSelect::Warmer].platforms);
    clGetDeviceIDs(m_kernctx[eProgramSelect::Warmer].platform, CL_DEVICE_TYPE_GPU, 1,
                   &m_kernctx[eProgramSelect::Warmer].device, &m_kernctx[eProgramSelect::Warmer].numdevs);
    m_kernctx[eProgramSelect::Warmer].context = clCreateContext(NULL, 1, &m_kernctx[eProgramSelect::Warmer].device, NULL, NULL, NULL);
    m_kernctx[eProgramSelect::Warmer].command_queue =     clCreateCommandQueue(m_kernctx[eProgramSelect::Warmer].context,
                                                                           m_kernctx[eProgramSelect::Warmer].device, 0, NULL);
    bufferx =   clCreateBuffer(m_kernctx[eProgramSelect::Warmer].context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                             globwsize , pixels, NULL);

    bufferx = clCreateBuffer(m_kernctx[eProgramSelect::Warmer].context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             sizeof(pixels), pixels , NULL);
    static const char* flags = "";
    m_kernctx[eProgramSelect::Warmer].program = clCreateProgramWithSource(m_kernctx[eProgramSelect::Warmer].context, 1, &warmers, NULL,
                                                                            &m_kernctx[eProgramSelect::Warmer].compilerr);
    if (m_kernctx[eProgramSelect::Warmer].compilerr) {
        return;
    }

    cl_int res = clBuildProgram(m_kernctx[eProgramSelect::Warmer].program, 1, &m_kernctx[eProgramSelect::Warmer].device, flags, NULL, NULL);
    if (res < 0) {
        clGetProgramBuildInfo(m_kernctx[eProgramSelect::Warmer].program, m_kernctx[eProgramSelect::Warmer].device, CL_PROGRAM_BUILD_LOG,
                              0, NULL, &log_size);
        program_log = (char*) calloc(log_size+1, sizeof(char));
        clGetProgramBuildInfo(m_kernctx[eProgramSelect::Warmer].program,
                              m_kernctx[eProgramSelect::Warmer].device, CL_PROGRAM_BUILD_LOG,
                              log_size+1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        return;
    }
    cl_int callres = 0;
    static unsigned int sp = 0;
    ++sp;
    m_kernctx[eProgramSelect::Warmer].kernel = clCreateKernel(m_kernctx[eProgramSelect::Warmer].program, "warmer", NULL);
    callres |= clSetKernelArg(m_kernctx[eProgramSelect::Warmer].kernel, 0, sizeof(cl_mem), &bufferx);
    callres |= clSetKernelArg(m_kernctx[eProgramSelect::Warmer].kernel, 1, sizeof(unsigned int), &sp);
    callres |= clGetKernelWorkGroupInfo(m_kernctx[eProgramSelect::Warmer].kernel,
                                        m_kernctx[eProgramSelect::Warmer].device,  CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &wsize, NULL);
    callres |= clEnqueueNDRangeKernel(m_kernctx[eProgramSelect::Warmer].command_queue,
                                      m_kernctx[eProgramSelect::Warmer].kernel,
                                      dim,
                                      global_offset,
                                      global_size,
                                      0,
                                      0,
                                      NULL,
                                      NULL);

        if (callres) {
            //handle errs
        } else {
            clFlush(m_kernctx[eProgramSelect::Warmer].command_queue);
            clFinish(m_kernctx[eProgramSelect::Warmer].command_queue);
            callres|= clEnqueueReadBuffer(m_kernctx[eProgramSelect::Warmer].command_queue, bufferx, CL_TRUE, 0,  globwsize, pixels, 0, NULL, NULL);
        }

    clReleaseMemObject(bufferx);

}


void gpu_kernel::conv3x3(unsigned int *pixels, unsigned int x, unsigned int y, unsigned int w, unsigned int h,
                         float (&kern)[9])
{
#ifndef HWACELL_ENABLED
    size_t dim = 3;
    size_t global_offset[] = {x, y, 0};
    size_t global_size[] = {w, h, 9};
    size_t wsize = 0 ;
    size_t globwsize = w * h * sizeof(unsigned int);
    cl_mem bufferx,  bufferkern;
    size_t log_size;
    char* program_log = 0;
    //CL_DEVICE_TYPE_CPU
    //CL_DEVICE_TYPE_ALL
    printf("x:%d y:%d w:%d h:%d size is (%lu)\r\n", x, y, w, h,globwsize);
    clGetPlatformIDs(1, &m_kernctx[eProgramSelect::Convolve].platform, &m_kernctx[eProgramSelect::Convolve].platforms);
    clGetDeviceIDs(m_kernctx[eProgramSelect::Convolve].platform, CL_DEVICE_TYPE_GPU, 1,
                   &m_kernctx[eProgramSelect::Convolve].device, &m_kernctx[eProgramSelect::Convolve].numdevs);
    m_kernctx[eProgramSelect::Convolve].context = clCreateContext(NULL, 1, &m_kernctx[eProgramSelect::Convolve].device, NULL, NULL, NULL);
    m_kernctx[eProgramSelect::Convolve].command_queue =     clCreateCommandQueue(m_kernctx[eProgramSelect::Convolve].context, m_kernctx[eProgramSelect::Convolve].device, 0, NULL);
    bufferx =   clCreateBuffer(m_kernctx[eProgramSelect::Convolve].context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                             globwsize , pixels, NULL);

    bufferkern = clCreateBuffer(m_kernctx[eProgramSelect::Convolve].context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(float) * 9 , kern , NULL);
    static const char* flags = "-cl-finite-math-only";
    m_kernctx[eProgramSelect::Convolve].program = clCreateProgramWithSource(m_kernctx[eProgramSelect::Convolve].context, 1, &src3x3, NULL,
                                                  &m_kernctx[eProgramSelect::Convolve].compilerr);
    if (m_kernctx[eProgramSelect::Convolve].compilerr) {
        return;
    }

    cl_int res = clBuildProgram(m_kernctx[eProgramSelect::Convolve].program, 1, &m_kernctx[eProgramSelect::Convolve].device, flags, NULL, NULL);
    if (res < 0) {
        clGetProgramBuildInfo(m_kernctx[eProgramSelect::Convolve].program, m_kernctx[eProgramSelect::Convolve].device, CL_PROGRAM_BUILD_LOG,
                              0, NULL, &log_size);
        program_log = (char*) calloc(log_size+1, sizeof(char));
        clGetProgramBuildInfo(m_kernctx[eProgramSelect::Convolve].program,
                              m_kernctx[eProgramSelect::Convolve].device, CL_PROGRAM_BUILD_LOG,
                              log_size+1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        return;
    }
    cl_int callres = 0;
    m_kernctx[eProgramSelect::Convolve].kernel = clCreateKernel(m_kernctx[eProgramSelect::Convolve].program, "conv3x3", NULL);
    callres |= clSetKernelArg(m_kernctx[eProgramSelect::Convolve].kernel, 0, sizeof(cl_mem), &bufferx);
    callres |= clSetKernelArg(m_kernctx[eProgramSelect::Convolve].kernel, 1, sizeof(unsigned int), &w);
    callres |= clSetKernelArg(m_kernctx[eProgramSelect::Convolve].kernel, 2, sizeof(unsigned int), &h);
    callres |= clSetKernelArg(m_kernctx[eProgramSelect::Convolve].kernel, 3, sizeof(cl_mem), &bufferkern);

    callres |= clGetKernelWorkGroupInfo(m_kernctx[eProgramSelect::Convolve].kernel,
                                        m_kernctx[eProgramSelect::Convolve].device,  CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &wsize, NULL);
    callres |= clEnqueueNDRangeKernel(m_kernctx[eProgramSelect::Convolve].command_queue,
                                      m_kernctx[eProgramSelect::Convolve].kernel,
                                      dim,
                                      global_offset,
                                      global_size,
                                      0,
                                      0,
                                      NULL,
                                      NULL);

    clFlush(m_kernctx[eProgramSelect::Convolve].command_queue);
    clFinish(m_kernctx[eProgramSelect::Convolve].command_queue);
    if (callres) {
        fprintf(stderr, "Error enqueue NDRangeKernel\r\n");
        goto cleanup;
    }
    callres|= clEnqueueReadBuffer(m_kernctx[eProgramSelect::Convolve].command_queue, bufferx, CL_TRUE, 0,  globwsize, pixels, 0, NULL, NULL);

    if (callres) {
        fprintf(stderr, "Failed to enqueue read buffer\r\n");
    } else {

    }
    cleanup:
    clReleaseMemObject(bufferx);
    clReleaseMemObject(bufferkern);

#else
#if 0
    unsigned int MyMat[3][9] ={{0}};
    for(unsigned int y=0; y < h; y++) {
        for (unsigned int x=0; x < w; x++) {

            //r
            for (int my=0; my < 3; my++) {
                for(int mx=0; mx < 3; mx++) {
                    unsigned int pix = pixels[(y+my) * w + (x + mx)];
                    MyMat[0][my * 3 + mx] = ((pix >> 16) & 0x000000FFu) ;
                }
            }
            //g
            for (int my=0; my < 3; my++) {
                for(int mx=0; mx < 3; mx++) {
                    unsigned int pix = pixels[(y+my) * w + (x + mx)];
                    MyMat[1][my * 3 + mx] = ((pix >> 8) & 0x000000FFu) ;
                }
            }
            //b
            for (int my=0; my < 3; my++) {
                for(int mx=0; mx < 3; mx++) {
                    unsigned int pix = pixels[(y+my) * w + (x + mx)];
                    MyMat[2][my * 3 + mx] = ((pix >> 0) & 0x000000FFu) ;
                }
            }
            unsigned int res = 0;
            unsigned int newpix =0;
            int shift = 16;
            for(int i=0; i < 3; i++) {
                for(int j=0; j < 9; j++) {
                    MyMat[i][j] *= kern[j];
                    res += MyMat[i][j];
                }
                newpix |= ((res & 0xFFu)<< shift);
                shift -= 8;
                res = 0;
            }
            pixels[x + w * y] = newpix;
        }
    }
#endif


    unsigned int MyMat[9];
    for(unsigned int y=0; y < h; y++) {
        for(unsigned int x=0; x < w; x++) {
        for(int i=0; i <3; i++) {
            unsigned char resr = 0, resg=0, resb=0;
            unsigned char r,g,b;

            for(int j=0; j < 3; j++) {
                r = kern[i*3+j] * ((pixels[(x+j) + w * (y+i)] >> 16) & 0xFFu);
                g = kern[i*3+j] * ((pixels[(x+j) + w * (y+i)] >> 8) & 0xFFu);
                b = kern[i*3+j] * ((pixels[(x+j) + w * (y+i)] >> 0) & 0xFFu);
                resr +=r;
                resg +=g;
                resb +=b;
                MyMat[i * 3 + j] = (0xFF000000u) | (resr & 0xFFu) << 16 | (resg & 0xFFu) << 8 | (resb & 0xFFu);
                }
            }
            pixels[y * w + x] = MyMat[4];
        }
    }
#endif
}

void gpu_kernel::conv5x5(unsigned int *pixels, size_t x, size_t y, size_t w, size_t h, float (&data)[3][25])
{
    (void)pixels; (void)x; (void)y;  (void) w; (void)h; (void)data;
}


#endif
