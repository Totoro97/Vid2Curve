//
// Created by aska on 2019/9/4.
//

#include "Math.h"

namespace Math {

double CrossProd(const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
  return a(0) * b(1) - a(1) * b(0);
}

double CrossProdAbs(const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
  return std::abs(a(0) * b(1) - a(1) * b(0));
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> GetOrthVectors(const Eigen::Vector3d& u) {
  Eigen::Vector3d v, w;
  if (u(0) * u(0) + u(1) * u(1) > 1e-9) {
    v = Eigen::Vector3d(u(1), -u(0), 0.0).normalized();
  } else {
    v = Eigen::Vector3d(0.0, u(2), -u(1)).normalized();
  }
  w = u.cross(v).normalized();
  return {v, w};
}

}