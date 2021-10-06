#include "render.h"
#include "utils.h"


void rasterize_kernel_wrapper(
    const int b, 
    const int n, 
    const int h, 
    const int w, 
    const float *xyz, 
    float *z_buffer);


torch::Tensor rasterize(
    torch::Tensor xyz,
    const int h, 
    const int w) {

    CHECK_CUDA(xyz);

    CHECK_CONTIGUOUS(xyz);

    CHECK_IS_FLOAT(xyz);
    
    torch::Tensor z_buffer = torch::full(
        {xyz.size(0), h, w}, 10,
        torch::device(xyz.device()).dtype(torch::ScalarType::Float));

    rasterize_kernel_wrapper(
        xyz.size(0),
        xyz.size(1),
        h, 
        w, 
        xyz.data_ptr <float>(),
        z_buffer.data_ptr <float>());

    return z_buffer;
}


void refine_z_buffer_kernel_wrapper(
    const int b, 
    const int h, 
    const int w, 
    float *z_buffer);


void refine_z_buffer(
    torch::Tensor z_buffer) {

    CHECK_CUDA(z_buffer);

    CHECK_CONTIGUOUS(z_buffer);

    CHECK_IS_FLOAT(z_buffer);

    refine_z_buffer_kernel_wrapper(
        z_buffer.size(0),
        z_buffer.size(1),
        z_buffer.size(2),
        z_buffer.data_ptr <float>());
}


void splat_kernel_wrapper(
    const int b, 
    const int c,
    const int n,  
    const int h, 
    const int w,  
    const float *xyz, 
    const float *z_buffer, 
    const float *data, 
    float *out,
    bool *is_visible);


std::tuple< torch::Tensor, torch::Tensor > splat(
    torch::Tensor data,
    torch::Tensor xyz,
    torch::Tensor z_buffer) {

    CHECK_CUDA(data);
    CHECK_CUDA(xyz);
    CHECK_CUDA(z_buffer);

    CHECK_CONTIGUOUS(data);
    CHECK_CONTIGUOUS(xyz);
    CHECK_CONTIGUOUS(z_buffer);
    
    CHECK_IS_FLOAT(data);
    CHECK_IS_FLOAT(xyz);
    CHECK_IS_FLOAT(z_buffer);

    torch::Tensor out = torch::zeros(
        {data.size(0), data.size(1), z_buffer.size(1), z_buffer.size(2)},
        torch::device(data.device()).dtype(torch::ScalarType::Float));
    torch::Tensor is_visible = torch::zeros(
        {xyz.size(0), xyz.size(1)}, 
        torch::device(xyz.device()).dtype(torch::ScalarType::Bool));

    splat_kernel_wrapper(
        data.size(0),
        data.size(1),
        data.size(2),
        z_buffer.size(1),
        z_buffer.size(2),
        xyz.data_ptr <float>(),  
        z_buffer.data_ptr <float>(),
        data.data_ptr <float>(),
        out.data_ptr <float>(),
        is_visible.data_ptr <bool>());

    return std::make_tuple(out, is_visible);
}


void splat_grad_kernel_wrapper(
    const int b, 
    const int c, 
    const int n, 
    const int h, 
    const int w, 
    const float *xyz, 
    const float *z_buffer, 
    const float *grad_out, 
    float *grad_data);


torch::Tensor splat_grad(
    torch::Tensor grad_out,
    torch::Tensor xyz, 
    torch::Tensor z_buffer) {

    CHECK_CUDA(grad_out);
    CHECK_CUDA(xyz);
    CHECK_CUDA(z_buffer);

    CHECK_CONTIGUOUS(grad_out);
    CHECK_CONTIGUOUS(xyz);
    CHECK_CONTIGUOUS(z_buffer);

    CHECK_IS_FLOAT(grad_out);
    CHECK_IS_FLOAT(xyz);
    CHECK_IS_FLOAT(z_buffer);

    torch::Tensor grad_data = torch::zeros(
        {grad_out.size(0), grad_out.size(1), xyz.size(1)},
        torch::device(grad_out.device()).dtype(torch::ScalarType::Float));

    splat_grad_kernel_wrapper(
        grad_out.size(0),
        grad_out.size(1),
        xyz.size(1),
        grad_out.size(2),
        grad_out.size(3),
        xyz.data_ptr <float>(),
        z_buffer.data_ptr <float>(),
        grad_out.data_ptr <float>(),
        grad_data.data_ptr <float>());

    return grad_data;
}