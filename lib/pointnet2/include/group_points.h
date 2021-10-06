#ifndef _GROUP_POINTS_H
#define _GROUP_POINTS_H

#pragma once
#include <torch/extension.h>


torch::Tensor group_points(
    torch::Tensor points, 
    torch::Tensor idx);

torch::Tensor group_points_grad(
    torch::Tensor grad_out, 
    torch::Tensor idx,
    const int n);

#endif