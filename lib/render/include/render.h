#ifndef _RENDER_H
#define _RENDER_H

#pragma once
#include <torch/extension.h>
#include <utility>


torch::Tensor rasterize(
    torch::Tensor xyz,
    const int h, 
    const int w);

void refine_z_buffer(
    torch::Tensor z_buffer);

std::tuple< torch::Tensor, torch::Tensor > splat(
    torch::Tensor data,
    torch::Tensor xyz,
    torch::Tensor z_buffer);

torch::Tensor splat_grad(
    torch::Tensor grad_out,
    torch::Tensor xyz, 
    torch::Tensor z_buffer);

#endif