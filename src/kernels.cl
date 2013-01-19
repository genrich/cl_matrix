__kernel void el_mul (__global const double* src1,
                      __global const double* src2,
                      __global double* dst)
{
    size_t id = get_global_id(0);
    dst [id] = src1 [id] * src2 [id];
}

__kernel void sigmoid (__global const double* src,
                       __global double* dst)
{
    size_t id = get_global_id(0);
    dst [id] = 1 / (1 + exp (- src [id]));
}
