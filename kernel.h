#ifndef KERNEL_H
#define KERNEL_H
#include "defs.h"

struct kernel
{
    float val3x3[9];
    float val5x5[25];

    kernel();
    kernel(float (&data)[9]);

    FRGB operator * (FRGB (&data)[25]);
    FRGB operator * (FRGB (&data)[9]);
};


#endif // KERNEL_H
