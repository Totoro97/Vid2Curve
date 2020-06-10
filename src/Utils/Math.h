//
// Created by aska on 2019/9/4.
//

#pragma once
#include <Eigen/Eigen>

#include <algorithm>

namespace Math {

double CrossProd(const Eigen::Vector2d& a, const Eigen::Vector2d& b);
double CrossProdAbs(const Eigen::Vector2d& a, const Eigen::Vector2d& b);
std::pair<Eigen::Vector3d, Eigen::Vector3d> GetOrthVectors(const Eigen::Vector3d& u);
}