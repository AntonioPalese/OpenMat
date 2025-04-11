#include "ops/kernels/unary_op_macros.cuh"

namespace om 
{
    // Usage: name, expression using lhs(r,c) and rhs(r,c)
    DEFINE_UNARY_OP_LAUNCH(add_k)
    DEFINE_UNARY_OP_KERNEL_K1(add_k, lhs(x) + value)
    DEFINE_UNARY_OP_KERNEL_K2(add_k, lhs(y, x) + value)
    DEFINE_UNARY_OP_KERNEL_K3(add_k, lhs(z, y, x) + value)
    DEFINE_UNARY_OP_KERNEL_K4(add_k, lhs(n, c, h, w) + value)
    DEFINE_UNARY_OP_KERNEL_ND(add_k, lhs[offset] + value)
    DEFINE_UNARY_OP_LAUNCH_FRW_DEC(add_k)

    DEFINE_UNARY_OP_LAUNCH(sub_k)
    DEFINE_UNARY_OP_KERNEL_K1(sub_k, lhs(x) - value)
    DEFINE_UNARY_OP_KERNEL_K2(sub_k, lhs(y, x) - value)
    DEFINE_UNARY_OP_KERNEL_K3(sub_k, lhs(z, y, x) - value)
    DEFINE_UNARY_OP_KERNEL_K4(sub_k, lhs(n, c, h, w) - value)
    DEFINE_UNARY_OP_KERNEL_ND(sub_k, lhs[offset] - value)
    DEFINE_UNARY_OP_LAUNCH_FRW_DEC(sub_k)


    DEFINE_UNARY_OP_LAUNCH(mul_k)
    DEFINE_UNARY_OP_KERNEL_K1(mul_k, lhs(x) * value)
    DEFINE_UNARY_OP_KERNEL_K2(mul_k, lhs(y, x) * value)
    DEFINE_UNARY_OP_KERNEL_K3(mul_k, lhs(z, y, x) * value)
    DEFINE_UNARY_OP_KERNEL_K4(mul_k, lhs(n, c, h, w) * value)
    DEFINE_UNARY_OP_KERNEL_ND(mul_k, lhs[offset] * value)
    DEFINE_UNARY_OP_LAUNCH_FRW_DEC(mul_k)


    DEFINE_UNARY_OP_LAUNCH(div_k)
    DEFINE_UNARY_OP_KERNEL_K1(div_k, ( static_cast<double>(value) != 0.0 ? lhs(x) / value : T{INFINITY} ))
    DEFINE_UNARY_OP_KERNEL_K2(div_k, ( static_cast<double>(value) != 0.0 ? lhs(y, x) / value : T{INFINITY} ))
    DEFINE_UNARY_OP_KERNEL_K3(div_k, ( static_cast<double>(value) != 0.0 ? lhs(z, y, x) / value : T{INFINITY} ))
    DEFINE_UNARY_OP_KERNEL_K4(div_k, ( static_cast<double>(value) != 0.0 ? lhs(n, c, h, w) / value : T{INFINITY} ))
    DEFINE_UNARY_OP_KERNEL_ND(div_k, ( static_cast<double>(value) != 0.0 ? lhs[offset] / value : T{INFINITY} ))
    DEFINE_UNARY_OP_LAUNCH_FRW_DEC(div_k)
}
