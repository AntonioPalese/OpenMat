#include "ops/kernels/binary_op_macros.cuh"

namespace om 
{
    // Usage: name, expression using lhs(r,c) and rhs(r,c)
    DEFINE_BINARY_OP_KERNEL_AND_LAUNCH(add, lhs(r, c) + rhs(r, c))
    DEFINE_BINARY_OP_KERNEL_AND_LAUNCH(sub, lhs(r, c) - rhs(r, c))
    DEFINE_BINARY_OP_KERNEL_AND_LAUNCH(mul, lhs(r, c) * rhs(r, c))
    DEFINE_BINARY_OP_KERNEL_AND_LAUNCH(div, ( static_cast<double>(rhs(r, c)) != 0.0 ? lhs(r, c) / rhs(r, c) : INFINITY ))

    DEFINE_BINARY_OP_KERNEL_AND_LAUNCH_FRW_DEC(add)
    DEFINE_BINARY_OP_KERNEL_AND_LAUNCH_FRW_DEC(sub)
    DEFINE_BINARY_OP_KERNEL_AND_LAUNCH_FRW_DEC(mul)
    DEFINE_BINARY_OP_KERNEL_AND_LAUNCH_FRW_DEC(div)
}
