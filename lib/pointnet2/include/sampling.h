#ifndef _SAMPLING_H
#define _SAMPLING_H

#pragma once
#include <torch/extension.h>
#include <utility>


torch::Tensor gather_points(
    torch::Tensor points, 
    torch::Tensor idx);

torch::Tensor gather_points_grad(
    torch::Tensor grad_out, 
    torch::Tensor idx, 
    const int n);

torch::Tensor furthest_point_sampling(
    torch::Tensor xyz,
    const int m);

#endif