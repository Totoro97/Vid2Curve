//
// Created by aska on 2020/1/1.
//

#include "MeshICP.h"
#include <iostream>

MeshICP::MeshICP(const std::vector<Eigen::Vector3d>& my_points,
                 const std::vector<std::tuple<int, int, int>>& my_faces,
                 const std::vector<Eigen::Vector3d>& gt_points,
                 const std::vector<std::tuple<int, int, int>>& gt_faces) :
                 my_points_(my_points), my_faces_(my_faces), gt_points_(gt_points), gt_faces_(gt_faces) {

  gt_normals_.clear();
  gt_normals_.resize(gt_points.size());
  gt_faces_at_points_.resize(gt_points.size());
  for (const auto &face : gt_faces_) {
    int a = std::get<0>(face);
    int b = std::get<1>(face);
    int c = std::get<2>(face);
    Eigen::Vector3d n = (gt_points_[b] - gt_points_[a]).cross(gt_points_[c] - gt_points_[a]).normalized();
    gt_normals_[a].emplace_back(n);
    gt_normals_[b].emplace_back(n);
    gt_normals_[c].emplace_back(n);
    gt_faces_at_points_[a].emplace_back(face);
    gt_faces_at_points_[b].emplace_back(face);
    gt_faces_at_points_[c].emplace_back(face);
  }

  octree_ = std::make_unique<OctreeNew>(gt_points_);
  R_ = Eigen::Matrix3d::Identity();
  T_ = Eigen::Vector3d::Zero();
  scale_ = 1.0;
}

// Find Best R, T, s, s.t. (X * R + T) * s == Y
void MeshICP::FitRT() {
  double rigid_increment = 1e9;
  double scale_increment = 1e9;
  int cnt = 0;
  // LOG(INFO) << "Before: " << std::endl << *R << std::endl << T->transpose() << std::endl;
  // while (cnt++ < 10 && rigid_increment > 1e-5 && scale_increment > 1e-5) {
  while (cnt++ < 10) {
    LOG(INFO) << "here~: " << cnt;
    rigid_increment = FitRTSingleStep();
    // scale_increment = FitScaleSingleStep();
    // LOG(INFO) << "After: " << std::endl << *R << std::endl << T->transpose() << std::endl;
  }
  /*
  cnt = 0;
  while (cnt++ < 50) {
    LOG(INFO) << "here~: " << cnt;
    rigid_increment = FitRTSingleStep();
    scale_increment = FitScaleSingleStep();
    // LOG(INFO) << "After: " << std::endl << *R << std::endl << T->transpose() << std::endl;
  }
  cnt = 0;
  while (cnt++ < 10) {
    LOG(INFO) << "here~: " << cnt;
    rigid_increment = FitRTSingleStep();
    // scale_increment = FitScaleSingleStep();
    // LOG(INFO) << "After: " << std::endl << *R << std::endl << T->transpose() << std::endl;
  }*/
}

double MeshICP::FitRTSingleStep() {
  int eq_idx = -1;
  // LOG(INFO) << "weight size:" << weights_.size()
  std::vector<int> nearest_indexes;
  std::vector<int> segments;
  int n_normals = 0;
  for (int idx = 0; idx < my_points_.size(); idx++) {
    const auto &pt = my_points_[idx];
    Eigen::Vector3d pt_3d = (R_ * pt + T_) * scale_;
    // CHECK(pt_3d(2) > 1e-9);
    // for (int i = 0; i < 2; i++) {
    // CHECK(pt_2d(i) > -100 && pt_2d(i) < 100);
    // }
    // std::cout << idx << std::endl;
    int nearest_idx = octree_->NearestIdx(pt_3d);
    nearest_indexes.emplace_back(nearest_idx);
    segments.emplace_back(n_normals);
    n_normals += gt_normals_[nearest_idx].size();
  }
  segments.emplace_back(n_normals);

  Eigen::MatrixXd A(n_normals, 6);
  A = Eigen::MatrixXd::Zero(n_normals, 6);
  Eigen::VectorXd B(n_normals);
  B = Eigen::VectorXd::Zero(n_normals);

  LOG(INFO) << "here~";
  int idx;
  int n_points = my_points_.size();
#pragma omp parallel for private(idx) default(none) shared(R_, T_, segments, n_points, my_points_, gt_points_, gt_normals_, nearest_indexes, A, B) // schedule (dynamic,2)
  for (idx = 0; idx < n_points; idx++) {
    // CHECK(nearest_idx >= 0 && nearest_idx < weights_.size());
    // LOG(INFO) << "nearest_idx: " << nearest_idx;
    // CHECK(weight > 0.5);
    int nearest_idx = nearest_indexes[idx];
    int l_bound = segments[idx];

    const Eigen::Vector3d pt_3d = R_ * my_points_[idx] + T_;
    // template point
    double sx = pt_3d(0);
    double sy = pt_3d(1);
    double sz = pt_3d(2);

    double dx = gt_points_[nearest_idx](0) / scale_;
    double dy = gt_points_[nearest_idx](1) / scale_;
    double dz = gt_points_[nearest_idx](2) / scale_;

    const auto& current_normals = gt_normals_[nearest_idx];
    for (int i_normal = 0; i_normal < current_normals.size(); i_normal++) {
      double nx = current_normals[i_normal](0);
      double ny = current_normals[i_normal](1);
      double nz = current_normals[i_normal](2);

      // double current_dis =
      //     (pt_3d - (gt_points_[nearest_idx] / scale_)).dot(current_normals[i_normal]);
      // double weight = current_dis * current_dis;
      double weight = 1.0;

      // setup least squares system
      A(l_bound + i_normal, 0) = (nz * sy - ny * sz) * weight;
      A(l_bound + i_normal, 1) = (nx * sz - nz * sx) * weight;
      A(l_bound + i_normal, 2) = (ny * sx - nx * sy) * weight;
      A(l_bound + i_normal, 3) = nx * weight;
      A(l_bound + i_normal, 4) = ny * weight;
      A(l_bound + i_normal, 5) = nz * weight;
      B(l_bound + i_normal) = (nx * dx + ny * dy + nz * dz - nx * sx - ny * sy - nz * sz);
    }
    // LOG(INFO) << A.block(idx, 0, 1, 6);
    // LOG(INFO) << B(idx);
  }

  // regularization
  // double reg_weight = 0.1;
  // int reg_idx = n_world_points - 1;
  // for (int u = 0; u < 6; u++) {
  //   reg_idx++;
  //   A(reg_idx, u) = reg_weight;
  //   B(reg_idx) = 0.0;
  // }

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
  R_ = R_inc * R_;
  T_ = R_inc * T_ + T_inc;
  return std::max((R_inc - Eigen::Matrix3d::Identity()).norm(), T_inc.norm());
}

double MeshICP::FitScaleSingleStep() {
  std::vector<int> nearest_indexes;
  std::vector<int> segments;
  int n_normals = 0;
  for (int idx = 0; idx < my_points_.size(); idx++) {
    const auto &pt = my_points_[idx];
    Eigen::Vector3d pt_3d = (R_ * pt + T_) * scale_;
    // CHECK(pt_3d(2) > 1e-9);
    // for (int i = 0; i < 2; i++) {
    // CHECK(pt_2d(i) > -100 && pt_2d(i) < 100);
    // }
    // std::cout << idx << std::endl;
    int nearest_idx = octree_->NearestIdx(pt_3d);
    nearest_indexes.emplace_back(nearest_idx);
    segments.emplace_back(n_normals);
    n_normals += gt_normals_[nearest_idx].size();
  }
  segments.emplace_back(n_normals);

  Eigen::MatrixXd A(n_normals, 1);
  A = Eigen::MatrixXd::Zero(n_normals, 1);
  Eigen::VectorXd B(n_normals);
  B = Eigen::VectorXd::Zero(n_normals);

  LOG(INFO) << "here~";
  int idx;
  int n_points = my_points_.size();
#pragma omp parallel for private(idx) default(none) shared(R_, T_, segments, n_points, my_points_, gt_points_, gt_normals_, nearest_indexes, A, B) // schedule (dynamic,2)
  for (idx = 0; idx < n_points; idx++) {
    // CHECK(nearest_idx >= 0 && nearest_idx < weights_.size());
    // LOG(INFO) << "nearest_idx: " << nearest_idx;
    // CHECK(weight > 0.5);
    int nearest_idx = nearest_indexes[idx];
    int l_bound = segments[idx];

    const Eigen::Vector3d pt_3d = (R_ * my_points_[idx] + T_) * scale_;
    double sx = pt_3d(0);
    double sy = pt_3d(1);
    double sz = pt_3d(2);

    double dx = gt_points_[nearest_idx](0);
    double dy = gt_points_[nearest_idx](1);
    double dz = gt_points_[nearest_idx](2);

    const auto& current_normals = gt_normals_[nearest_idx];
    for (int i_normal = 0; i_normal < current_normals.size(); i_normal++) {
      double nx = current_normals[i_normal](0);
      double ny = current_normals[i_normal](1);
      double nz = current_normals[i_normal](2);

      double current_dis = (pt_3d - gt_points_[nearest_idx]).dot(current_normals[i_normal]);
      double weight = current_dis * current_dis;
      // setup least squares system
      A(l_bound + i_normal, 0) = (nx * sx + ny * sy + nz * sz) * weight;
      B(l_bound + i_normal) = (nx * dx + ny * dy + nz * dz) * weight;
    }
  }

  Eigen::VectorXd s_solution;
  s_solution = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(B);
  // delta_p = A.colPivHouseholderQr().solve(-B);
  scale_ *= s_solution(0);
  LOG(INFO) << "new_scale: " << s_solution(0);
  return std::abs(s_solution(0) - 1.0);
}

MeshEvaluationResult MeshICP::Evaluate(std::vector<double>* points_distances) {
  if (points_distances != nullptr) {
    points_distances->clear();
  }
  double max_dis = 0.0;
  double ave_dis = 0.0;
  Eigen::Vector3d ma(-1e9, -1e9, -1e9);
  Eigen::Vector3d mi(1e9, 1e9, 1e9);
  for (int idx = 0; idx < my_points_.size(); idx++) {
    const auto &pt = my_points_[idx];
    Eigen::Vector3d pt_3d = (R_ * pt + T_) * scale_;
    for (int t = 0; t < 3; t++) {
      ma(t) = std::max(ma(t), pt_3d(t));
      mi(t) = std::min(mi(t), pt_3d(t));
    }
    // CHECK(pt_3d(2) > 1e-9);
    // for (int i = 0; i < 2; i++) {
    // CHECK(pt_2d(i) > -100 && pt_2d(i) < 100);
    // }
    // std::cout << idx << std::endl;
    int nearest_idx = octree_->NearestIdx(pt_3d);
    // double distance = 1e9;
    double distance = 1e9;
    for (int k = 0; k < gt_faces_at_points_[nearest_idx].size(); k++) {
      const auto& face = gt_faces_at_points_[nearest_idx][k];
      const auto& normal = gt_normals_[nearest_idx][k];
      Eigen::Vector3d a = gt_points_[std::get<0>(face)];
      Eigen::Vector3d b = gt_points_[std::get<1>(face)];
      Eigen::Vector3d c = gt_points_[std::get<2>(face)];
      Eigen::Vector3d pro_pt = (a - pt_3d).dot(normal) * (a - pt_3d) + pt_3d;
      Eigen::Vector3d A = (b - a).cross(pro_pt - a);
      Eigen::Vector3d B = (c - b).cross(pro_pt - b);
      Eigen::Vector3d C = (a - c).cross(pro_pt - c);
      if (A.dot(B) > -1e-9 && A.dot(C) > -1e-9 && B.dot(C) > -1e-9) {
        distance = std::min(distance, std::abs((pt_3d - a).dot(normal)));
      }
    }
    if (distance + 1.0 > 1e9) {
      distance = (pt_3d - gt_points_[nearest_idx]).norm();
    }

    // for (int k = 0; k < gt_normals_[nearest_idx].size(); k++) {
    //   distance = std::min(distance, std::abs((gt_points_[nearest_idx] - pt_3d).dot(gt_normals_[nearest_idx][k])));
    // }
    // distance /= gt_normals_[nearest_idx].size();

    if (points_distances != nullptr) {
      points_distances->emplace_back(distance);
    }
    max_dis = std::max(max_dis, distance);
    ave_dis += distance;
  }
  ave_dis /= my_points_.size();
  MeshEvaluationResult ret;
  ret.average_distance = ave_dis;
  ret.max_distance = max_dis;
  ret.global_scale = (ma - mi).norm();
  return ret;
}