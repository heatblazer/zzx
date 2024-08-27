

--------------------------



[Host]
typedef struct __attribute__ ((packed)) _st_foo
{
    cl_int aaa;
    cl_int bbb;
     .....
    cl_int zzz;
}st_foo;

[Device]

typedef struct __attribute__ ((packed)) _st_foo
{
    int aaa;
    int bbb;
     .....
    int zzz;
};
