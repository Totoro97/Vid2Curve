//
// Created by aska on 2019/9/4.
//

#include "Camera.h"

Camera::Camera(const Eigen::Matrix3d& intrinsic_mat,
               const Eigen::Matrix3d& R,
               const Eigen::Vector3d& T,
               const Eigen::Vector2d& distortion) {
  CHECK_NEAR(intrinsic_mat(2, 2), 1.0, 1e-5);
  focal_length_x_ = intrinsic_mat(0, 0);
  focal_length_y_ = intrinsic_mat(1, 1);
  half_width_ = intrinsic_mat(0, 2);
  half_height_ = intrinsic_mat(1, 2);
  R_ = R;
  T_ = T;
  distort_0_ = distortion(0);
  distort_1_ = distortion(1);
}

Camera::Camera(double focal_length, double width, double height) {
  focal_length_x_ = focal_length;
  focal_length_y_ = focal_length;
  half_width_ = width * 0.5;
  half_height_ = height * 0.5;
}

Eigen::Vector2d Camera::Cam2Image(const Eigen::Vector3d& point) {
  return { point(1) / point(2) * focal_length_y_ + half_height_,
           point(0) / point(2) * focal_length_x_ + half_width_ };
}

Eigen::Vector2d Camera::World2Image(const Eigen::Vector3d& point) {
  return Cam2Image(World2Cam(point));
}

Eigen::Vector3d Camera::World2Cam(const Eigen::Vector3d& point) {
  return R_ * point + T_;
}

Eigen::Vector3d Camera::Cam2World(const Eigen::Vector3d& point) {
  return R_.inverse() * (point - T_);
}

Eigen::Vector3d Camera::Image2Cam(const Eigen::Vector2d& point, double depth) {
  Eigen::Vector3d pt((point(1) - half_width_) / focal_length_x_, (point(0) - half_height_) / focal_length_y_, 1.0);
  return pt * depth;
}

Eigen::Vector3d Camera::Image2World(const Eigen::Vector2d& point, double depth) {
  return Cam2World(Image2Cam(point, depth));
}

double Camera::GetFocalLengthX() {
  return focal_length_x_;
}

double Camera::GetFocalLengthY() {
  return focal_length_y_;
}

void Camera::UpdateRT(const Eigen::Matrix3d &R, const Eigen::Vector3d &T) {
  R_ = R;
  T_ = T;
}

