__kernel void init_zero (__global double* dst)
{
    const size_t id = get_global_id (0);
    dst[id] = 0;
}
//__________________________________________________________________________________________________

__kernel void uminus (__global const double* src,
                      __global double*       dst)
{
    const size_t id = get_global_id (0);
    dst[id] = -src[id];
}
//__________________________________________________________________________________________________

__kernel void transpose (__global const double* src,
                         __global double*       dst)
{
    const size_t row  = get_global_id (0);
    const size_t col  = get_global_id (1);
    const size_t rows = get_global_size (0);
    const size_t cols = get_global_size (1);

    dst[row * cols + col] = src[col * rows + row];
}
//__________________________________________________________________________________________________

__kernel void add (__global const double* src1,
                   __global const double* src2,
                   __global double*       dst)
{
    const size_t id = get_global_id (0);
    dst[id] = src1[id] + src2[id];
}
//__________________________________________________________________________________________________

__kernel void add_scalar (__global const double* src,
                                   const double  scalar,
                          __global double*       dst)
{
    const size_t id = get_global_id (0);
    dst[id] = src[id] + scalar;
}
//__________________________________________________________________________________________________

__kernel void sub (__global const double* src1,
                   __global const double* src2,
                   __global double*       dst)
{
    const size_t id = get_global_id (0);
    dst[id] = src1[id] - src2[id];
}
//__________________________________________________________________________________________________

__kernel void scalar_sub (         const double  scalar,
                          __global const double* src,
                          __global double*       dst)
{
    const size_t id = get_global_id (0);
    dst[id] = scalar - src[id] ;
}
//__________________________________________________________________________________________________

__kernel void mul_scalar (__global const double* src,
                                   const double  scalar,
                          __global double*       dst)
{
    const size_t id = get_global_id (0);
    dst[id] = src[id] * scalar;
}
//__________________________________________________________________________________________________

__kernel void scalar_div (         const double  scalar,
                          __global const double* src,
                          __global double*       dst)
{
    const size_t id = get_global_id (0);
    dst[id] = scalar / src[id];
}
//__________________________________________________________________________________________________

__kernel void el_mul (__global const double* src1,
                      __global const double* src2,
                      __global double*       dst)
{
    const size_t id = get_global_id (0);
    dst[id] = src1[id] * src2[id];
}
//__________________________________________________________________________________________________

__kernel void el_div (__global const double* src1,
                      __global const double* src2,
                      __global double*       dst)
{
    const size_t id = get_global_id (0);
    dst[id] = src1[id] / src2[id];
}
//__________________________________________________________________________________________________

__kernel void sigmoid (__global const double* src,
                       __global double*       dst)
{
    const size_t id = get_global_id (0);
    dst[id] = 1 / (1 + exp (- src[id]));
}
//__________________________________________________________________________________________________

__kernel __attribute__((vec_type_hint(double)))
         void sum_full_load (__global const double* src,
                                      const uint    iter,
                                      const ulong   count,
                             __local  double*       sums,
                             __global double*       dst)
{
    const size_t global_id       = get_global_id  (0);
    const size_t local_id        = get_local_id   (0);
    const size_t local_size      = get_local_size (0);
    const size_t group_id        = get_group_id   (0);
    const size_t group_size      = get_num_groups (0);
    const size_t half_local_size = local_size / 2;

    double sum = 0;
    for (uint j = 0; j < iter; j++)
    {
        const ulong id = global_id + j * group_size * local_size;
        sum += select ((double) 0, src[id], (long) (id < count));
    }
    sums [local_id] = sum;
    barrier (CLK_LOCAL_MEM_FENCE);

    for (uint j = half_local_size; j > 0; j >>= 1)
    {
        if (local_id < j)
            sums[local_id] += sums[local_id + j];

        barrier (CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0)
        dst[group_id] = sums[0];
}
//__________________________________________________________________________________________________

__kernel __attribute__((vec_type_hint(double)))
         void sum_comp_unit (__global const double* src,
                                      const ulong   count,
                             __local  double*       sums,
                             __global double*       dst)
{
    const size_t local_id        = get_local_id   (0);
    const size_t local_size      = get_local_size (0);
    const size_t half_local_size = local_size / 2;
    const ulong  id              = local_id;

    sums[local_id] = select ((double) 0, src[local_id], (long) (id < count));
    barrier (CLK_LOCAL_MEM_FENCE);

    for (uint j = half_local_size; j > 0; j >>= 1)
    {
        if (local_id < j)
            sums[local_id] += sums[local_id + j];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0)
        dst[0] = sums[0];
}
//__________________________________________________________________________________________________

__kernel __attribute__((vec_type_hint(double2)))
         void sum_full_load_2 (__global const double2* src,
                                        const uint     iter,
                                        const ulong    count,
                               __local  double2*       sums,
                               __global double2*       dst)
{
    const size_t global_id       = get_global_id  (0);
    const size_t local_id        = get_local_id   (0);
    const size_t local_size      = get_local_size (0);
    const size_t group_id        = get_group_id   (0);
    const size_t group_size      = get_num_groups (0);
    const size_t half_local_size = local_size / 2;

    double2 sum = 0;
    for (uint j = 0; j < iter; j++)
    {
        const ulong2 id = global_id + j * group_size * local_size;
        sum += select (0, src[id.s0], id < count);
    }
    sums [local_id] = sum;
    barrier (CLK_LOCAL_MEM_FENCE);

    for (uint j = half_local_size; j > 0; j >>= 1)
    {
        if (local_id < j)
            sums[local_id] += sums[local_id + j];

        barrier (CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0)
        dst[group_id] = sums[0];
}
//__________________________________________________________________________________________________

__kernel __attribute__((vec_type_hint(double2)))
         void sum_comp_unit_2 (__global const double2* src,
                                        const ulong    count,
                               __local  double2*       sums,
                               __global double*        dst)
{
    const size_t local_id        = get_local_id   (0);
    const size_t local_size      = get_local_size (0);
    const size_t half_local_size = local_size / 2;
    const ulong2 id              = local_id;

    sums[local_id] = select (0, src[local_id], id < count);
    barrier (CLK_LOCAL_MEM_FENCE);

    for (uint j = half_local_size; j > 0; j >>= 1)
    {
        if (local_id < j)
            sums[local_id] += sums[local_id + j];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0)
        dst[0] = sums[0].s0 + sums[0].s1;
}
