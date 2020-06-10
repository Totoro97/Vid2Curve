//
// Created by aska on 2020/3/2.
//

#include "../Utils/Common.h"
// ceres solver
#include <ceres/ceres.h>
#include <ceres/rotation.h>

const double kViewTangErrorWeight = 0.5;

struct ViewProjectionError {
  ViewProjectionError(const Eigen::Vector3d& world_pt,
                      const Eigen::Vector2d& img_pt,
                      const Eigen::Vector2d& img_tang,
                      double weight,
                      double focal_length,
                      double width,
                      double height) {
    weight_ = std::max(kViewTangErrorWeight, weight);
    focal_length_ = focal_length;
    world_x_ = world_pt(0);
    world_y_ = world_pt(1);
    world_z_ = world_pt(2);
    p_x_ = img_pt(1) - width * 0.5;
    p_y_ = img_pt(0) - height * 0.5;
    dir_x_ = img_tang(1);
    dir_y_ = img_tang(0);
  }

  template <typename T>
  bool operator()(const T* const camera,
                  T* residuals) const {
    T point[3];
    point[0] = T(world_x_);
    point[1] = T(world_y_);
    point[2] = T(world_z_);
    T p[3];
    // camera[0,1,2] are the angle-axis rotation.
    ceres::AngleAxisRotatePoint(camera, point, p);
    // camera[3,4,5] are the translation.
    p[0] += camera[3]; p[1] += camera[4]; p[2] += camera[5];

    T bias_x = p[0] / p[2] * T(focal_length_) - T(p_x_);
    T bias_y = p[1] / p[2] * T(focal_length_) - T(p_y_);

    residuals[0] = (bias_x * T(dir_y_) - bias_y * T(dir_x_)) * T(weight_);
    residuals[1] = (bias_x * T(dir_x_) + bias_y * T(dir_y_)) * T(weight_ * kViewTangErrorWeight);
    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const Eigen::Vector3d& world_pt,
                                     const Eigen::Vector2d& img_pt,
                                     const Eigen::Vector2d& img_tang,
                                     double weight,
                                     double focal_length,
                                     double width,
                                     double height) {
    // LOG(INFO) << "weight: " << weight;
    return (new ceres::AutoDiffCostFunction<ViewProjectionError, 2, 6>(
        new ViewProjectionError(world_pt, img_pt, img_tang, weight, focal_length, width, height)));
  }

  double weight_;
  double focal_length_;
  double p_x_, p_y_;
  double dir_x_, dir_y_;
  double world_x_, world_y_, world_z_;
};
