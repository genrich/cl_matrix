__kernel void uminus (
                      __global const double* src,
                      __global double*       dst)
{
    const size_t id = get_global_id (0);
    dst[id] = -src[id];
}

//______________________________________________________________________________

__kernel void transpose (
                         __global const double* src,
                         __global double*       dst)
{
    const size_t row  = get_global_id (0);
    const size_t col  = get_global_id (1);
    const size_t rows = get_global_size (0);
    const size_t cols = get_global_size (1);

    dst[row * cols + col] = src[col * rows + row];
}

//______________________________________________________________________________

__kernel void add (
                   __global const double* src1,
                   __global const double* src2,
                   __global double*       dst)
{
    const size_t id = get_global_id (0);
    dst[id] = src1[id] + src2[id];
}

//______________________________________________________________________________

__kernel void add_scalar (
                          __global const double* src,
                                   const double  scalar,
                          __global double*       dst)
{
    const size_t id = get_global_id (0);
    dst[id] = src[id] + scalar;
}

//______________________________________________________________________________

__kernel void sub (
                   __global const double* src1,
                   __global const double* src2,
                   __global double*       dst)
{
    const size_t id = get_global_id (0);
    dst[id] = src1[id] - src2[id];
}

//______________________________________________________________________________

__kernel void scalar_sub (
                                   const double  scalar,
                          __global const double* src,
                          __global double*       dst)
{
    const size_t id = get_global_id (0);
    dst[id] = scalar - src[id] ;
}

//______________________________________________________________________________

__kernel void mul_scalar (
                          __global const double* src,
                                   const double  scalar,
                          __global double*       dst)
{
    const size_t id = get_global_id (0);
    dst[id] = src[id] * scalar;
}

//______________________________________________________________________________

__kernel void scalar_div (
                                   const double  scalar,
                          __global const double* src,
                          __global double*       dst)
{
    const size_t id = get_global_id (0);
    dst[id] = scalar / src[id];
}

//______________________________________________________________________________

__kernel void el_mul (
                      __global const double* src1,
                      __global const double* src2,
                      __global double*       dst)
{
    const size_t id = get_global_id (0);
    dst[id] = src1[id] * src2[id];
}

//______________________________________________________________________________

__kernel void el_div (
                      __global const double* src1,
                      __global const double* src2,
                      __global double*       dst)
{
    const size_t id = get_global_id (0);
    dst[id] = src1[id] / src2[id];
}

//______________________________________________________________________________

__kernel void sigmoid (
                       __global const double* src,
                       __global double*       dst)
{
    const size_t id = get_global_id (0);
    dst[id] = 1 / (1 + exp (- src[id]));
}
