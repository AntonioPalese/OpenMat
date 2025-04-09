// #include "ops/kernels/binary_op_macros.cuh"

// namespace om 
// {
//     // Usage: name, expression using lhs(r,c) and rhs(r,c)
//     DEFINE_BINARY_OP_KERNEL_K1(add, lhs(x) + rhs(x))
//     DEFINE_BINARY_OP_KERNEL_K2(add, lhs(y, x) + rhs(y, x))
//     DEFINE_BINARY_OP_KERNEL_K3(add, lhs(z, y, x) + rhs(z, y, x))
//     DEFINE_BINARY_OP_KERNEL_K4(add, lhs(n, c, h, w) + rhs(n, c, h, w))
//     DEFINE_BINARY_OP_KERNEL_ND(add, lhs[offset] + rhs[offset])
//     DEFINE_BINARY_OP_LAUNCH(add)
//     //DEFINE_BINARY_OP_LAUNCH_FRW_DEC(add)

//     // DEFINE_BINARY_OP_LAUNCH(sub)
//     // DEFINE_BINARY_OP_KERNEL_K1(sub, lhs(x) - rhs(x))
//     // DEFINE_BINARY_OP_KERNEL_K2(sub, lhs(y, x) - rhs(y, x))
//     // DEFINE_BINARY_OP_KERNEL_K3(sub, lhs(z, y, x) - rhs(z, y, x))
//     // DEFINE_BINARY_OP_KERNEL_K4(sub, lhs(n, c, h, w) - rhs(n, c, h, w))
//     // DEFINE_BINARY_OP_KERNEL_ND(sub, lhs[offset] - rhs[offset])


//     // DEFINE_BINARY_OP_LAUNCH(mul)
//     // DEFINE_BINARY_OP_KERNEL_K1(mul, lhs(x) * rhs(x))
//     // DEFINE_BINARY_OP_KERNEL_K2(mul, lhs(y, x) * rhs(y, x))
//     // DEFINE_BINARY_OP_KERNEL_K3(mul, lhs(z, y, x) * rhs(z, y, x))
//     // DEFINE_BINARY_OP_KERNEL_K4(mul, lhs(n, c, h, w) * rhs(n, c, h, w))
//     // DEFINE_BINARY_OP_KERNEL_ND(mul, lhs[offset] * rhs[offset])


//     // DEFINE_BINARY_OP_LAUNCH(div)
//     // DEFINE_BINARY_OP_KERNEL_K1(div, ( static_cast<double>(rhs(x)) != 0.0 ? lhs(x) / rhs(x) : INFINITY ))
//     // DEFINE_BINARY_OP_KERNEL_K2(div, ( static_cast<double>(rhs(y, x)) != 0.0 ? lhs(y, x) / rhs(y, x) : INFINITY ))
//     // DEFINE_BINARY_OP_KERNEL_K3(div, ( static_cast<double>(rhs(z, y, x)) != 0.0 ? lhs(z, y, x) / rhs(z, y, x) : INFINITY ))
//     // DEFINE_BINARY_OP_KERNEL_K4(div, ( static_cast<double>(rhs(n, c, h, w)) != 0.0 ? lhs(n, c, h, w) / rhs(n, c, h, w) : INFINITY ))
//     // DEFINE_BINARY_OP_KERNEL_ND(div, ( static_cast<double>(rhs[offset]) != 0.0 ? lhs[offset] / rhs[offset] : INFINITY ))


//     // DEFINE_BINARY_OP_LAUNCH_FRW_DEC(add)
//     // DEFINE_BINARY_OP_LAUNCH_FRW_DEC(sub)
//     // DEFINE_BINARY_OP_LAUNCH_FRW_DEC(mul)
//     // DEFINE_BINARY_OP_LAUNCH_FRW_DEC(div)
// }
