//
// Created by aska on 2019/11/9.
//

#include "ICP.h"

ICP::ICP(std::vector<Eigen::Vector2d>& points, const std::vector<Eigen::Vector3d>& normals, const std::vector<double>& weights) :
    points_(points), quad_tree_(new QuadTree(points)), normals_(normals), weights_(weights) {
  n_points_ = points_.size();
  CHECK_EQ(n_points_, normals.size());
  CHECK_EQ(n_points_, weights.size());
}

void ICP::FitRT(const std::vector<Eigen::Vector3d>& world_points,
                Eigen::Matrix3d* R,
                Eigen::Vector3d* T,
                int max_iter_num) {
  double increament = 1e9;
  int cnt = 0;
  // LOG(INFO) << "Before: " << std::endl << *R << std::endl << T->transpose() << std::endl;
  while (cnt++ < max_iter_num && increament > 1e-4) {
    increament = FitRTSingleStep(world_points, R, T);
    // LOG(INFO) << "After: " << std::endl << *R << std::endl << T->transpose() << std::endl;
  }
  // LOG(INFO) << "After: " << std::endl << *R << std::endl << T->transpose() << std::endl;
  // LOG(INFO) << "weights_size!!!: " << weights_.size();
}

double ICP::FitRTSingleStep(const std::vector<Eigen::Vector3d>& world_points, Eigen::Matrix3d* R, Eigen::Vector3d* T) {
  Eigen::MatrixXd A(world_points.size() + 6, 6);
  A = Eigen::MatrixXd::Zero(world_points.size() + 6, 6);
  Eigen::VectorXd B(world_points.size() + 6);
  B = Eigen::VectorXd::Zero(world_points.size() + 6);

  // LOG(INFO) << "weight size:" << weights_.size();
  int n_world_points = world_points.size();
  int idx;
#pragma omp parallel for private(idx) default(none) shared(R, T, world_points, n_world_points, points_, A, B) // schedule (dynamic,2)
  for (idx = 0; idx < n_world_points; idx++) {
    const auto& world_pt = world_points[idx];
    Eigen::Vector3d pt_3d = *R * world_pt + *T;
    // CHECK(pt_3d(2) > 1e-9);
    Eigen::Vector2d pt_2d(pt_3d(0) / pt_3d(2), pt_3d(1) / pt_3d(2));
    // for (int i = 0; i < 2; i++) {
    // CHECK(pt_2d(i) > -100 && pt_2d(i) < 100);
    // }
    int nearest_idx = quad_tree_->NearestIdx(pt_2d);
    // CHECK(nearest_idx >= 0 && nearest_idx < weights_.size());
    // LOG(INFO) << "nearest_idx: " << nearest_idx;
    double weight = weights_[nearest_idx];
    // CHECK(weight > 0.5);
    double dx = points_[nearest_idx](0);
    double dy = points_[nearest_idx](1);
    double dz = 1.0;

    double nx = normals_[nearest_idx](0);
    double ny = normals_[nearest_idx](1);
    double nz = normals_[nearest_idx](2);

    // template point
    double sx = pt_3d(0);
    double sy = pt_3d(1);
    double sz = pt_3d(2);

    // setup least squares system
    A(idx, 0) = weight * (nz * sy - ny * sz);
    A(idx, 1) = weight * (nx * sz - nz * sx);
    A(idx, 2) = weight * (ny * sx - nx * sy);
    A(idx, 3) = weight * nx;
    A(idx, 4) = weight * ny;
    A(idx, 5) = weight * nz;
    B(idx) = weight * (nx * dx + ny * dy + nz * dz - nx * sx - ny * sy - nz * sz);
    // LOG(INFO) << A.block(idx, 0, 1, 6);
    // LOG(INFO) << B(idx);
  }

  // regularization
  double reg_weight = 0.1;
  int reg_idx = n_world_points - 1;
  for (int u = 0; u < 6; u++) {
    reg_idx++;
    A(reg_idx, u) = reg_weight;
    B(reg_idx) = 0.0;
  }

  // Eigen::MatrixXd A_;
  // A_ = A.transpose() * A;
  // LOG(INFO) << A_;
  // Eigen::VectorXd B_;
  // B_ = A.transpose() * B;
  // LOG(INFO) << B_;
  // LOG(INFO) << "Sollll";
  Eigen::VectorXd delta_p;
  delta_p = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(B);
  // delta_p = A.colPivHouseholderQr().solve(-B);
  // LOG(INFO) << "delta_p: " << delta_p.transpose();
  // rotation matrix
  Eigen::Matrix3d R_inc = Eigen::Matrix3d::Identity();
  R_inc(0, 1) = -delta_p(2);
  R_inc(1, 0) = +delta_p(2);
  R_inc(0, 2) = +delta_p(1);
  R_inc(2, 0) = -delta_p(1);
  R_inc(1, 2) = -delta_p(0);
  R_inc(2, 1) = +delta_p(0);

  Eigen::Vector3d T_inc(delta_p(3), delta_p(4), delta_p(5));
  Eigen::Matrix3d U;
  Eigen::Matrix3d V;
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(R_inc, Eigen::ComputeFullU | Eigen::ComputeFullV);
  U = svd.matrixU();
  V = svd.matrixV();
  // A = svd.singularValues();
  R_inc = U * V.transpose();

  if (R_inc.determinant() < 0) {
    Eigen::Matrix3d tmp = Eigen::Matrix3d::Identity();
    tmp(2, 2) = R_inc.determinant();
    R_inc = V * tmp * U.transpose();
  }
  *R = R_inc * *R;
  *T = R_inc * *T + T_inc;
  return std::max((R_inc - Eigen::Matrix3d::Identity()).norm(), T_inc.norm());
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> ICP::NearestRayNormal(const Eigen::Vector2d& point) {
  int idx = quad_tree_->NearestIdx(point);
  return { Eigen::Vector3d(points_[idx](0), points_[idx](1), 1.0), normals_[idx] };
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> ICP::RayNormalByIdx(int idx) {
  return { Eigen::Vector3d(points_[idx](0), points_[idx](1), 1.0), normals_[idx] };
}