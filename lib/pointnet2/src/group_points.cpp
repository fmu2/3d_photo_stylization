#include "group_points.h"
#include "utils.h"


void group_points_kernel_wrapper(
    const int b, 
    const int c, 
    const int n, 
    const int m, 
    const int s, 
    const float *points, 
    const int *idx, 
    float *out);


torch::Tensor group_points(
    torch::Tensor points, 
    torch::Tensor idx) {

    CHECK_CUDA(points);
    CHECK_CUDA(idx);

    CHECK_CONTIGUOUS(points);
    CHECK_CONTIGUOUS(idx);
    
    CHECK_IS_FLOAT(points);
    CHECK_IS_INT(idx);

    torch::Tensor out = torch::zeros(
        {points.size(0), points.size(1), idx.size(1), idx.size(2)}, 
        torch::device(points.device()).dtype(torch::ScalarType::Float));

    group_points_kernel_wrapper(
        points.size(0), 
        points.size(1), 
        points.size(2), 
        idx.size(1), 
        idx.size(2), 
        points.data_ptr <float>(), 
        idx.data_ptr <int>(), 
        out.data_ptr <float>());
    
    return out;
}


void group_points_grad_kernel_wrapper(
    const int b, 
    const int c, 
    const int n, 
    const int m, 
    const int s, 
    const float *grad_out, 
    const int *idx, 
    float *grad_points);


torch::Tensor group_points_grad(
    torch::Tensor grad_out, 
    torch::Tensor idx,
    const int n) {

    CHECK_CUDA(grad_out);
    CHECK_CUDA(idx);

    CHECK_CONTIGUOUS(grad_out);
    CHECK_CONTIGUOUS(idx);
    
    CHECK_IS_FLOAT(grad_out);
    CHECK_IS_INT(idx);

    torch::Tensor grad_points = torch::zeros(
        {grad_out.size(0), grad_out.size(1), n}, 
        torch::device(grad_out.device()).dtype(torch::ScalarType::Float));

    group_points_grad_kernel_wrapper(
        grad_out.size(0), 
        grad_out.size(1), 
        n, 
        grad_out.size(2), 
        grad_out.size(3), 
        grad_out.data_ptr <float>(), 
        idx.data_ptr <int>(), 
        grad_points.data_ptr <float>());
    
    return grad_points;
}