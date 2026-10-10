#ifndef KERNEL_H
#define KERNEL_H
#include "defs.h"
#include <algorithm>
#include <cstring>

template <typename T>
struct ckernel
{
    T val3x3[9];
    T val5x5[25];

    ckernel()
    {
        memset(&val3x3, 0, sizeof(val3x3));
        memset(&val5x5, 0, sizeof(val5x5));
    }
    //template <typename U, std::size_t N>
    ckernel(const T (&data)[9])
    {
        memset(&val5x5, 0, sizeof(val5x5));
        memcpy(val3x3, data, sizeof(data));
    }

    FRGB operator *(FRGB (&data)[25])
    {
        FRGB out;
        memset(&out, 0, sizeof(out));

        for (int i = 0; i < 3; i++) {
            T sum = 0;
            for(int ii=0; ii < 25; ii++) {
                sum  += data[ii].rgb[i] * val5x5[ii];
            }
            out.rgb[i] = static_cast<unsigned int>(std::clamp<T>(sum, 0, 255));
        }
        return out;
    }
    FRGB operator * (FRGB (&data)[9])
    {
        FRGB out;
        memset(&out , 0, sizeof(out));
        for(int i=0; i < 3; i++) {
            T sum = 0;
            for(int ii=0; ii < 9; ii++) {             // 1, 2, 3,       1, 1,1,
                sum += data[ii].rgb[i] * val3x3[ii];          // 1, 2, 3,   x   1, 1,1,
            }
            out.rgb[i] = static_cast<unsigned int>(std::clamp<T>(sum, 0, 255));
        }
        return out;
    }

};


#endif // KERNEL_H
