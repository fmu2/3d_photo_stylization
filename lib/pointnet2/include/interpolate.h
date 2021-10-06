#ifndef _INTERPOLATE_H
#define _INTERPOLATE_H

#pragma once
#include <torch/extension.h>
#include <utility>


std::tuple< torch::Tensor, torch::Tensor > three_nn(
    torch::Tensor parent_xyz, 
    torch::Tensor child_xyz);

torch::Tensor three_interpolate(
    torch::Tensor points,
    torch::Tensor idx,
    torch::Tensor weight);

torch::Tensor three_interpolate_grad(
    torch::Tensor grad_out,
    torch::Tensor idx,
    torch::Tensor weight,
    const int m);

#endif