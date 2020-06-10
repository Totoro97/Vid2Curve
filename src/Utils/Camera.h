//
// Created by aska on 2019/9/4.
//
#pragma once
#include "Common.h"

class Camera {
public:
  Camera(const Eigen::Matrix3d& intrinsic_mat,
         const Eigen::Matrix3d& R,
         const Eigen::Vector3d& T,
         const Eigen::Vector2d& distortion);
  Camera(double focal_length, double width, double height);

  Eigen::Vector2d Cam2Image(const Eigen::Vector3d& point);
  Eigen::Vector2d World2Image(const Eigen::Vector3d& point);
  Eigen::Vector3d World2Cam(const Eigen::Vector3d& point);
  Eigen::Vector3d Cam2World(const Eigen::Vector3d& point);
  Eigen::Vector3d Image2Cam(const Eigen::Vector2d& point, double depth = 1.0);
  Eigen::Vector3d Image2World(const Eigen::Vector2d& point, double depth = 1.0);
  double GetFocalLengthX();
  double GetFocalLengthY();

  void UpdateRT(const Eigen::Matrix3d& R, const Eigen::Vector3d& T);
private:
  double focal_length_x_ = 1.0;
  double focal_length_y_ = 1.0;
  double half_width_ = 0.0;
  double half_height_ = 0.0;
  double distort_0_ = 0.0;
  double distort_1_ = 0.0;
  Eigen::Matrix3d R_;
  Eigen::Vector3d T_;
};