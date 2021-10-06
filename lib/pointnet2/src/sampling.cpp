#include "sampling.h"
#include "utils.h"


void gather_points_kernel_wrapper(
    const int b, 
    const int c, 
    const int n, 
    const int m, 
    const float *points, 
    const int *idx, 
    float *out);


torch::Tensor gather_points(
    torch::Tensor points, 
    torch::Tensor idx) {

    CHECK_CUDA(points);
    CHECK_CUDA(idx);

    CHECK_CONTIGUOUS(points);
    CHECK_CONTIGUOUS(idx);

    CHECK_IS_FLOAT(points);
    CHECK_IS_INT(idx);

    torch::Tensor out = torch::zeros(
        {points.size(0), points.size(1), idx.size(1)}, 
        torch::device(points.device()).dtype(torch::ScalarType::Float));

    gather_points_kernel_wrapper(
        points.size(0), 
        points.size(1), 
        points.size(2), 
        idx.size(1), 
        points.data_ptr <float>(), 
        idx.data_ptr <int>(), 
        out.data_ptr <float>());
    
    return out;
}


void gather_points_grad_kernel_wrapper(
    const int b,
    const int c,
    const int n,
    const int m,
    const float *grad_out,
    const int *idx,
    float *grad_points);


torch::Tensor gather_points_grad(
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

    gather_points_grad_kernel_wrapper(
        grad_out.size(0), 
        grad_out.size(1), 
        n, 
        grad_out.size(2), 
        grad_out.data_ptr <float>(), 
        idx.data_ptr <int>(), 
        grad_points.data_ptr <float>());
    
    return grad_points;
}


void furthest_point_sampling_kernel_wrapper(
    const int b, 
    const int n, 
    const int m, 
    const float *xyz, 
    float *temp, 
    int *idx);


torch::Tensor furthest_point_sampling(
    torch::Tensor xyz,
    const int m) {

    CHECK_CUDA(xyz);

    CHECK_CONTIGUOUS(xyz);
    
    CHECK_IS_FLOAT(xyz);

    torch::Tensor temp = torch::full(
        {xyz.size(0), xyz.size(1)}, 1000000.0,
        torch::device(xyz.device()).dtype(torch::ScalarType::Float));
    torch::Tensor idx = torch::zeros(
        {xyz.size(0), m}, 
        torch::device(xyz.device()).dtype(torch::ScalarType::Int));

    furthest_point_sampling_kernel_wrapper(
        xyz.size(0), 
        xyz.size(1), 
        m, 
        xyz.data_ptr <float>(), 
        temp.data_ptr <float>(), 
        idx.data_ptr <int>());
    
    return idx;
}