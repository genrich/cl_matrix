__kernel void sigmoid (__global const double* src, __global double* dst)
{
    size_t global_id = get_global_id(0);
    dst [global_id] = 1 / (1 + exp (- src [global_id]));
}
