//
// Created by aska on 2019/9/4.
//
#pragma once
#include "Model.h"
#include "CurveExtractor.h"
#include "CurveMatcher.h"
#include "../Utils/Common.h"

// ceres solver
#include <ceres/ceres.h>
#include <ceres/rotation.h>

const double kTangErrorWeight = 0.2;

// error setting for ceres.
struct ProjectionError {
  ProjectionError(const ImageLocalMatching& matching,
                  double weight,
                  double focal_length,
                  double width,
                  double height) {
    weight_ = std::max(kTangErrorWeight, weight);
    focal_length_ = focal_length;
    p_x_ = matching.p.o_(1) - width * 0.5;
    p_y_ = matching.p.o_(0) - height * 0.5;
    q_x_ = matching.q.o_(1) - width * 0.5;
    q_y_ = matching.q.o_(0) - height * 0.5;
    dir_x_ = matching.q.v_(1);
    dir_y_ = matching.q.v_(0);
  }

  template <typename T>
  bool operator()(const T* const camera,
                  const T* const depth,
                  T* residuals) const {
    T point[3];
    point[0] = T(p_x_) * *depth / T(focal_length_);
    point[1] = T(p_y_) * *depth / T(focal_length_);
    point[2] = *depth;
    T p[3];
    // camera[0,1,2] are the angle-axis rotation.
    ceres::AngleAxisRotatePoint(camera, point, p);
    // camera[3,4,5] are the translation.
    p[0] += camera[3]; p[1] += camera[4]; p[2] += camera[5];

    T bias_x = p[0] / p[2] * T(focal_length_) - T(q_x_);
    T bias_y = p[1] / p[2] * T(focal_length_) - T(q_y_);

    residuals[0] = bias_x * T(dir_y_) - bias_y * T(dir_x_);
    residuals[1] = (bias_x * T(dir_x_) + bias_y * T(dir_y_)) * T(weight_);
    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const ImageLocalMatching& matching,
                                     double weight,
                                     double focal_length,
                                     double width,
                                     double height) {
    // LOG(INFO) << "weight: " << weight;
    return (new ceres::AutoDiffCostFunction<ProjectionError, 2, 6, 1>(
        new ProjectionError(matching, weight, focal_length, width, height)));
  }

  double weight_;
  double focal_length_;
  double p_x_, p_y_;
  double q_x_, q_y_;
  double dir_x_, dir_y_;
};

struct SingleDepthError {
  explicit SingleDepthError(double hope_depth = 1.0, double weight = 1.0) : hope_depth_(hope_depth), weight_(weight) {}

  template <typename T>
  bool operator()(const T* const depth,
                  T* residuals) const {
    residuals[0] = (*depth - T(hope_depth_)) * T(weight_);
    return true;
  }

  static ceres::CostFunction* Create(double hope_depth = 1.0, double weight = 1.0) {
    return (new ceres::AutoDiffCostFunction<SingleDepthError, 1, 1>(new SingleDepthError(hope_depth, weight)));
  }

  double hope_depth_ = 1.0;
  double weight_ = 1.0;
};

struct RelativeDepthError {
  explicit RelativeDepthError(double weight = 1.0) : weight_(weight) {}

  template <typename T>
  bool operator()(const T* const depth_0,
                  const T* const depth_1,
                  T* residuals) const {
    residuals[0] = (*depth_0 - *depth_1) * T(weight_);
    return true;
  }

  static ceres::CostFunction* Create(double weight = 1.0) {
    return (new ceres::AutoDiffCostFunction<RelativeDepthError, 1, 1, 1>(new RelativeDepthError(weight)));
  }

  double weight_ = 1.0;
};

class Initializer {
public:
  Initializer(const std::vector<CurveExtractor*>& curve_extractors,
              double focal_length,
              double width,
              double height,
              double single_depth_error_weight,
              double relative_depth_error_weight);
  void GetInitialModelData(std::vector<std::unique_ptr<ModelData>>* model_states);

private:
  std::vector<CurveExtractor*> curve_extractors_;
  double focal_length_;
  double width_;
  double height_;
  double single_depth_error_weight_ = 0.01;
  double relative_depth_error_weight_ = 0.05;
};
