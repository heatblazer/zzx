#ifndef DEFS_H
#define DEFS_H

typedef struct __attribute__((packed))
FRGB {

    unsigned int rgb[3];
} FRGB;


enum class eConvType
{
    GaussianBlur ,
    Sharper,
    IntensivSharper,
    Identity,
    Original,
    Custom3x3,
    Custom5x5,
    COUNT
};


struct kernel_t;

#endif // DEFS_H
