#include "ops/kernels/binary_op_macros.cuh"

namespace om 
{
    DEFINE_BINARY_OP_KERNEL_K1(add, lhs(x) + rhs(x))
    DEFINE_BINARY_OP_KERNEL_K2(add, lhs(y, x) + rhs(y, x))
    DEFINE_BINARY_OP_KERNEL_K3(add, lhs(z, y, x) + rhs(z, y, x))
    DEFINE_BINARY_OP_KERNEL_K4(add, lhs(n, c, h, w) + rhs(n, c, h, w))
    DEFINE_BINARY_OP_KERNEL_ND(add, lhs[offset] + rhs[offset])
    DEFINE_BINARY_OP_LAUNCH(add)
    DEFINE_BINARY_OP_LAUNCH_FRW_DEC(add)

    DEFINE_BINARY_OP_LAUNCH(sub)
    DEFINE_BINARY_OP_KERNEL_K1(sub, lhs(x) - rhs(x))
    DEFINE_BINARY_OP_KERNEL_K2(sub, lhs(y, x) - rhs(y, x))
    DEFINE_BINARY_OP_KERNEL_K3(sub, lhs(z, y, x) - rhs(z, y, x))
    DEFINE_BINARY_OP_KERNEL_K4(sub, lhs(n, c, h, w) - rhs(n, c, h, w))
    DEFINE_BINARY_OP_KERNEL_ND(sub, lhs[offset] - rhs[offset])
    DEFINE_BINARY_OP_LAUNCH_FRW_DEC(sub)


    DEFINE_BINARY_OP_LAUNCH(mul)
    DEFINE_BINARY_OP_KERNEL_K1(mul, lhs(x) * rhs(x))
    DEFINE_BINARY_OP_KERNEL_K2(mul, lhs(y, x) * rhs(y, x))
    DEFINE_BINARY_OP_KERNEL_K3(mul, lhs(z, y, x) * rhs(z, y, x))
    DEFINE_BINARY_OP_KERNEL_K4(mul, lhs(n, c, h, w) * rhs(n, c, h, w))
    DEFINE_BINARY_OP_KERNEL_ND(mul, lhs[offset] * rhs[offset])
    DEFINE_BINARY_OP_LAUNCH_FRW_DEC(mul)


    DEFINE_BINARY_OP_LAUNCH(div)
    DEFINE_BINARY_OP_KERNEL_K1(div, div_elem(lhs(x), rhs(x)))
    DEFINE_BINARY_OP_KERNEL_K2(div, div_elem(lhs(y, x), rhs(y, x)))
    DEFINE_BINARY_OP_KERNEL_K3(div, div_elem(lhs(z, y, x), rhs(z, y, x)))
    DEFINE_BINARY_OP_KERNEL_K4(div, div_elem(lhs(n, c, h, w), rhs(n, c, h, w)))
    DEFINE_BINARY_OP_KERNEL_ND(div, div_elem(lhs[offset], rhs[offset]))
    DEFINE_BINARY_OP_LAUNCH_FRW_DEC(div)
}
