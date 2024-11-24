#include "kernel.h"
#include <cstring>


kernel::kernel()
{
    memset(&val3x3, 0, sizeof(val3x3));
    memset(&val5x5, 0, sizeof(val5x5));
}

kernel::kernel(float (&data)[9]) {
    memcpy(val3x3, data, sizeof(data));
}


FRGB kernel::operator * (FRGB (&data)[25])
{
    FRGB out;
    memset(&out , 0, sizeof(out));
    float sum =0.0f;
    int res = 0;

    for(int i=0; i < 25; i++)
        sum += val5x5[i];
    for(int i=0; i < 3; i++) {
        for(int ii=0; ii < 25; ii++) {
            data[ii].rgb[i] *= val5x5[ii];
            res += data[ii].rgb[i];
        }
        out.rgb[i] = res;
        res = 0;

    }
    return out;
}

FRGB kernel::operator * (FRGB (&data)[9])
{
    FRGB out;
    memset(&out , 0, sizeof(out));
    float sum =0.0f;
    int res = 0;

    for(int i=0; i < 9; i++)
        sum += val3x3[i];
    for(int i=0; i < 3; i++) {
        for(int ii=0; ii < 9; ii++) {             // 1, 2, 3,       1, 1,1,
            data[ii].rgb[i] *= val3x3[ii];          // 1, 2, 3,   x   1, 1,1,
            res += data[ii].rgb[i];
        }
        out.rgb[i] = res;
        res = 0;

    }
    return out;
}

