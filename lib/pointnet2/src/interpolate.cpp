#include "interpolate.h"
#include "utils.h"


void three_nn_kernel_wrapper(
    const int b, 
    const int n, 
    const int m, 
    const float *parent_xyz, 
    const float *child_xyz, 
    float *dist, 
    int *idx);


std::tuple< torch::Tensor, torch::Tensor > three_nn(
    torch::Tensor parent_xyz, 
    torch::Tensor child_xyz) {

    CHECK_CUDA(parent_xyz);
    CHECK_CUDA(child_xyz);

    CHECK_CONTIGUOUS(parent_xyz);
    CHECK_CONTIGUOUS(child_xyz);
    
    CHECK_IS_FLOAT(parent_xyz);
    CHECK_IS_FLOAT(child_xyz);

    torch::Tensor dist = torch::zeros(
        {parent_xyz.size(0), parent_xyz.size(1), 3}, 
        torch::device(parent_xyz.device()).dtype(torch::ScalarType::Float));
    torch::Tensor idx = torch::zeros(
        {parent_xyz.size(0), parent_xyz.size(1), 3}, 
        torch::device(parent_xyz.device()).dtype(torch::ScalarType::Int));

    three_nn_kernel_wrapper(
        parent_xyz.size(0), 
        parent_xyz.size(1), 
        child_xyz.size(1), 
        parent_xyz.data_ptr <float>(), 
        child_xyz.data_ptr <float>(), 
        dist.data_ptr <float>(), 
        idx.data_ptr <int>());

    return std::make_tuple(dist, idx);
}


void three_interpolate_kernel_wrapper(
    const int b, 
    const int c, 
    const int n, 
    const int m, 
    const float *points, 
    const int *idx, 
    const float *weight, 
    float *out);


torch::Tensor three_interpolate(
    torch::Tensor points,
    torch::Tensor idx,
    torch::Tensor weight) {

    CHECK_CUDA(points);
    CHECK_CUDA(idx);
    CHECK_CUDA(weight);

    CHECK_CONTIGUOUS(points);
    CHECK_CONTIGUOUS(idx);
    CHECK_CONTIGUOUS(weight);

    CHECK_IS_FLOAT(points);
    CHECK_IS_INT(idx);
    CHECK_IS_FLOAT(weight);

    torch::Tensor out = torch::zeros(
        {points.size(0), points.size(1), idx.size(1)}, 
        torch::device(points.device()).dtype(torch::ScalarType::Float));

    three_interpolate_kernel_wrapper(
        points.size(0), 
        points.size(1), 
        idx.size(1), 
        points.size(2), 
        points.data_ptr <float>(), 
        idx.data_ptr <int>(), 
        weight.data_ptr <float>(),
        out.data_ptr <float>());

    return out;
}


void three_interpolate_grad_kernel_wrapper(
    const int b, 
    const int c, 
    const int n, 
    const int m, 
    const float *grad_out, 
    const int *idx, 
    const float *weight, 
    float *grad_points);


torch::Tensor three_interpolate_grad(
    torch::Tensor grad_out,
    torch::Tensor idx,
    torch::Tensor weight,
    const int m) {

    CHECK_CUDA(grad_out);
    CHECK_CUDA(idx);
    CHECK_CUDA(weight);

    CHECK_CONTIGUOUS(grad_out);
    CHECK_CONTIGUOUS(idx);
    CHECK_CONTIGUOUS(weight);
    
    CHECK_IS_FLOAT(grad_out);
    CHECK_IS_INT(idx);
    CHECK_IS_FLOAT(weight);

    torch::Tensor grad_points = torch::zeros(
        {grad_out.size(0), grad_out.size(1), m}, 
        torch::device(grad_out.device()).dtype(torch::ScalarType::Float));

    three_interpolate_grad_kernel_wrapper(
        grad_out.size(0), 
        grad_out.size(1), 
        grad_out.size(2), 
        m, 
        grad_out.data_ptr <float>(), 
        idx.data_ptr <int>(), 
        weight.data_ptr <float>(), 
        grad_points.data_ptr <float>());

    return grad_points;
}