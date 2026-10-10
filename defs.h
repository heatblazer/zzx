#ifndef DEFS_H
#define DEFS_H
struct __attribute__((packed))
FRGB {
    unsigned int rgb[3];
} ;


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

enum class eMtScreenMode {
    LineBox,
    SpatialRect,
    COUNT
};

#endif // DEFS_H
