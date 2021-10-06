#ifndef _BALL_QUERY_H
#define _BALL_QUERY_H

#pragma once
#include <torch/extension.h>
#include <utility>


std::tuple< torch::Tensor, torch::Tensor > ball_query(
    torch::Tensor parent_xyz,
    torch::Tensor child_xyz, 
    const float radius, 
    const int n_samples);

#endif