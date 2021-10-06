#include "ball_query.h"
#include "utils.h"


void ball_query_kernel_wrapper(
    const int b, 
    const int n, 
    const int m, 
    const float r, 
    const int s,
    const float *parent_xyz, 
    const float *child_xyz, 
    int *idx,
    bool *is_valid);


std::tuple< torch::Tensor, torch::Tensor > ball_query(
    torch::Tensor parent_xyz, 
    torch::Tensor child_xyz, 
    const float radius, 
    const int n_samples) {
    
    CHECK_CUDA(parent_xyz);
    CHECK_CUDA(child_xyz);

    CHECK_CONTIGUOUS(parent_xyz);
    CHECK_CONTIGUOUS(child_xyz);
    
    CHECK_IS_FLOAT(parent_xyz);
    CHECK_IS_FLOAT(child_xyz);

    torch::Tensor idx = torch::zeros(
        {child_xyz.size(0), child_xyz.size(1), n_samples}, 
        torch::device(child_xyz.device()).dtype(torch::ScalarType::Int));
    torch::Tensor is_valid = torch::zeros(
        {child_xyz.size(0), child_xyz.size(1), n_samples}, 
        torch::device(child_xyz.device()).dtype(torch::ScalarType::Bool));
    
    ball_query_kernel_wrapper(
        child_xyz.size(0), 
        parent_xyz.size(1), 
        child_xyz.size(1), 
        radius, 
        n_samples, 
        parent_xyz.data_ptr <float>(), 
        child_xyz.data_ptr <float>(), 
        idx.data_ptr <int>(),
        is_valid.data_ptr <bool>());

    return std::make_tuple(idx, is_valid);
}