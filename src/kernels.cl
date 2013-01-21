__kernel void uminus (
                      __global const double* src,
                      __global double*       dst)
{
    const size_t id = get_global_id (0);
    dst[id] = -src[id];
}

__kernel void transpose (
                         __global const double* src,
                                  const int     rows,
                                  const int     cols,
                         __global double*       dst)
{
    const size_t id = get_global_id (0);
    const int row = id % rows;
    const int col = id / rows;
    dst[row * cols + col] = src[id];
}

__kernel void add (
                   __global const double* src1,
                   __global const double* src2,
                   __global double*       dst)
{
    const size_t id = get_global_id (0);
    dst[id] = src1[id] + src2[id];
}

__kernel void add_scalar (
                          __global const double* src,
                                   const double  scalar,
                          __global double*       dst)
{
    const size_t id = get_global_id (0);
    dst[id] = src[id] + scalar;
}

__kernel void sub (
                   __global const double* src1,
                   __global const double* src2,
                   __global double*       dst)
{
    const size_t id = get_global_id (0);
    dst[id] = src1[id] - src2[id];
}

__kernel void scalar_sub (
                                   const double  scalar,
                          __global const double* src,
                          __global double*       dst)
{
    const size_t id = get_global_id (0);
    dst[id] = scalar - src[id] ;
}

__kernel void mul_scalar (
                          __global const double* src,
                                   const double  scalar,
                          __global double*       dst)
{
    const size_t id = get_global_id (0);
    dst[id] = src[id] * scalar;
}

__kernel void scalar_div (
                                   const double  scalar,
                          __global const double* src,
                          __global double*       dst)
{
    const size_t id = get_global_id (0);
    dst[id] = scalar / src[id];
}

__kernel void el_mul (
                      __global const double* src1,
                      __global const double* src2,
                      __global double*       dst)
{
    const size_t id = get_global_id (0);
    dst[id] = src1[id] * src2[id];
}

__kernel void el_div (
                      __global const double* src1,
                      __global const double* src2,
                      __global double*       dst)
{
    const size_t id = get_global_id (0);
    dst[id] = src1[id] / src2[id];
}

__kernel void sigmoid (
                       __global const double* src,
                       __global double*       dst)
{
    const size_t id = get_global_id (0);
    dst[id] = 1 / (1 + exp (- src[id]));
}
