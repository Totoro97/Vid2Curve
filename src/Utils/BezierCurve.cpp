//
// Created by aska on 2020/2/18.
//

#include "BezierCurve.h"

BezierCurve::BezierCurve(const std::vector<Eigen::Vector3d>& initial_points,
                         const std::vector<double>& errors,
                         double error_threshold,
                         double hope_dist) {
  points_p_.clear();
  points_t_.clear();
  hope_dist_ = hope_dist;
  BuildBezierCurve(initial_points, errors, error_threshold, 0, initial_points.size() - 1);
  CHECK_EQ(points_p_.size(), points_t_.size());
  CHECK_EQ(expressions_.size() + 1, initial_points.size());
  expressions_.emplace_back(points_p_.size() - 1, points_p_.size(), points_t_.size() - 1, 1.0);
  points_p_.emplace_back(initial_points.back());
}

void BezierCurve::BuildBezierCurve(const std::vector<Eigen::Vector3d> &initial_points,
                                   const std::vector<double> &errors,
                                   double error_threshold,
                                   int l_bound,
                                   int r_bound) {
  Eigen::Vector3d p1;
  if (IsSingleFittingOK(initial_points, errors, error_threshold, l_bound, r_bound, &p1)) {
    points_p_.emplace_back(initial_points[l_bound]);
    points_t_.emplace_back(p1);
    double len_sum = 0.0;
    for (int i = l_bound; i < r_bound; i++) {
      len_sum += (initial_points[i + 1] - initial_points[i]).norm();
    }
    double current_sum = 0.0;
    for (int i = l_bound; i < r_bound; i++) {
      expressions_.emplace_back(points_p_.size() - 1, points_p_.size(), points_t_.size() - 1, current_sum / len_sum);
      current_sum += (initial_points[i + 1] - initial_points[i]).norm();
    }
    return;
  }
  int mid = (l_bound + r_bound) / 2;
  BuildBezierCurve(initial_points, errors, error_threshold, l_bound, mid);
  BuildBezierCurve(initial_points, errors, error_threshold, mid, r_bound);
  return;
}

// Quadratic Bezier Curve
// B(t) = (1 - t)^2 * P0 + 2t(1 - t) * P1 + t^2 * P2
bool BezierCurve::IsSingleFittingOK(const std::vector<Eigen::Vector3d>& initial_points,
                                    const std::vector<double>& errors,
                                    double error_threshold,
                                    int l_bound,
                                    int r_bound,
                                    Eigen::Vector3d* estimated_p1) {
  CHECK_LT(l_bound, r_bound);
  if (l_bound + 1 >= r_bound) {
    if (estimated_p1 != nullptr) {
      *estimated_p1 = (initial_points[l_bound] + initial_points[r_bound]) * 0.5;
    }
    return true;
  }
  for (int i = l_bound + 1; i < r_bound; i++) {
    if (errors[i] > error_threshold) {
      return false;
    }
  }
  double len_sum = 0.0;
  for (int i = l_bound; i < r_bound; i++) {
    len_sum += (initial_points[i + 1] - initial_points[i]).norm();
  }
  double current_len = 0.0;
  const Eigen::Vector3d& p0 = initial_points[l_bound];
  const Eigen::Vector3d& p2 = initial_points[r_bound];
  Eigen::Vector3d A(0.0, 0.0, 0.0);
  Eigen::Vector3d B(0.0, 0.0, 0.0);
  Eigen::Vector3d C(0.0, 0.0, 0.0);
  // Find Best p1.
  for (int i = l_bound + 1; i < r_bound; i++) {
    current_len += (initial_points[i] - initial_points[i - 1]).norm();
    double t = current_len / len_sum;
    Eigen::Vector3d known_term = (1.0 - t) * (1.0 - t) * p0 + t * t * p2;
    double co = 2.0 * t * (1.0 - t);
    Eigen::Vector3d hope_residual = (initial_points[i] - known_term) / co;
    // (p1 - hope_residual)^2 ->
    for (int k = 0; k < 3; k++) {
      A(k) += 1.0;
      B(k) += - 2.0 * hope_residual(k);
      C(k) += hope_residual(k) * hope_residual(k);
    }
  }

  Eigen::Vector3d p1;
  for (int k = 0; k < 3; k++) {
    p1(k) = -0.5 * B(k) / A(k);
  }

  const double kErrorTolerance = hope_dist_ * 1.0;
  current_len = 0.0;
  for (int i = l_bound + 1; i < r_bound; i++) {
    current_len += (initial_points[i] - initial_points[i - 1]).norm();
    double t = current_len / len_sum;
    Eigen::Vector3d estimated_pt = (1.0 - t) * (1.0 - t) * p0 + 2.0 * (1.0 - t) * t * p1 + t * t * p2;
    double current_error = (estimated_pt - initial_points[i]).norm();
    if (current_error > kErrorTolerance) {
      return false;
    }
  }
  if (estimated_p1 != nullptr) {
    *estimated_p1 = p1;
  }
  return true;
}