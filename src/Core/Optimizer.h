// This is new.
#include "../Utils/Common.h"
// ceres solver
#include <ceres/ceres.h>
#include <ceres/rotation.h>

// For ceres solver.

struct ModelToViewError {
  ModelToViewError(const Eigen::Matrix3d& R,
                   const Eigen::Vector3d& T,
                   double img_p_x,
                   double img_p_y,
                   double img_v_x,
                   double img_v_y,
                   double weight,
                   double tang_weight,
                   double focal_length,
                   double width,
                   double height) {
    weight_ = weight;
    tang_weight_ = tang_weight;
    focal_length_ = focal_length;
    p_x_ = img_p_y - width * 0.5;
    p_y_ = img_p_x - height * 0.5;
    v_x_ = img_v_y;
    v_y_ = img_v_x;

    // Camera poses.
    Eigen::AngleAxisd angle_axis(R);
    double angle = angle_axis.angle();
    Eigen::Vector3d axis = angle_axis.axis();
    for (int t = 0; t < 3; t++) {
      camera_[t] = axis(t) * angle;
      camera_[t + 3] = T(t);
    }
  }

  template <typename T>
  bool operator()(const T* const point_3d,
                  T* residuals) const {
    const T* const point = point_3d;
    T p[3];
    // camera[0,1,2] are the angle-axis rotation.
    T camera[6] = { T(camera_[0]), T(camera_[1]), T(camera_[2]), T(camera_[3]), T(camera_[4]), T(camera_[5]) };
    ceres::AngleAxisRotatePoint(camera, point, p);
    // camera[3,4,5] are the translation.
    p[0] += camera[3]; p[1] += camera[4]; p[2] += camera[5];

    T bias_x = p[0] / p[2] * T(focal_length_) - T(p_x_);
    T bias_y = p[1] / p[2] * T(focal_length_) - T(p_y_);

    residuals[0] = (bias_x * T(v_y_) - bias_y * T(v_x_)) * T(weight_) / T(focal_length_);
    residuals[1] = (bias_x * T(v_x_) + bias_y * T(v_y_)) * T(weight_) * T(tang_weight_) / T(focal_length_);
    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const Eigen::Matrix3d& R,
                                     const Eigen::Vector3d& T,
                                     double img_p_x,
                                     double img_p_y,
                                     double img_v_x,
                                     double img_v_y,
                                     double weight,
                                     double tang_weight,
                                     double focal_length,
                                     double width,
                                     double height) {
    // LOG(INFO) << "weight: " << weight;
    return (new ceres::AutoDiffCostFunction<ModelToViewError, 2, 3>(
        new ModelToViewError(R,
                             T,
                             img_p_x,
                             img_p_y,
                             img_v_x,
                             img_v_y,
                             weight,
                             tang_weight,
                             focal_length,
                             width,
                             height)));
  }

  double weight_;
  double tang_weight_;
  double focal_length_;
  double p_x_, p_y_;
  double v_x_, v_y_;
  double camera_[6];
};

struct SmoothnessError {
  SmoothnessError(double weight_0,
                  double weight_1) {
    weight_0_ = weight_0;
    weight_1_ = weight_1;
  }

  template <typename T>
  bool operator()(const T* const point_a,
                  const T* const point_b,
                  const T* const point_c,
                  T* residuals) const {
    residuals[0] =
        (point_a[0] - point_b[0]) * T(weight_0_) - (point_b[0] - point_c[0]) * T(weight_1_);
    residuals[1] =
        (point_a[1] - point_b[1]) * T(weight_0_) - (point_b[1] - point_c[1]) * T(weight_1_);
    residuals[2] =
        (point_a[2] - point_b[2]) * T(weight_0_) - (point_b[2] - point_c[2]) * T(weight_1_);
    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(double weight_0_,
                                     double weight_1_) {
    // LOG(INFO) << "weight: " << weight;
    return (new ceres::AutoDiffCostFunction<SmoothnessError, 3, 3, 3, 3>(
      new SmoothnessError(weight_0_, weight_1_)));
  }

  double weight_0_;
  double weight_1_;
};

struct ShrinkError {
  ShrinkError(double weight) {
    weight_ = weight;
  }

  template <typename T>
  bool operator()(const T* const point_a,
                  const T* const point_b,
                  T* residuals) const {
    residuals[0] =
        (point_a[0] - point_b[0]) * T(weight_);
    residuals[1] =
        (point_a[1] - point_b[1]) * T(weight_);
    residuals[2] =
        (point_a[2] - point_b[2]) * T(weight_);
    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(double weight) {
    // LOG(INFO) << "weight: " << weight;
    return (new ceres::AutoDiffCostFunction<ShrinkError, 3, 3, 3>(
        new ShrinkError(weight)));
  }

  double weight_;
};

struct FixError {
  FixError(double weight, const Eigen::Vector3d& pt) {
    weight_ = weight;
    x_ = pt(0);
    y_ = pt(1);
    z_ = pt(2);
  }

  template <typename T>
  bool operator()(const T* const point_a,
                  T* residuals) const {
    residuals[0] =
        (point_a[0] - T(x_)) * T(weight_);
    residuals[1] =
        (point_a[1] - T(y_)) * T(weight_);
    residuals[2] =
        (point_a[2] - T(z_)) * T(weight_);
    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(double weight, const Eigen::Vector3d& pt) {
    // LOG(INFO) << "weight: " << weight;
    return (new ceres::AutoDiffCostFunction<FixError, 3, 3>(
        new FixError(weight, pt)));
  }

  double weight_;
  double x_, y_, z_;
};

struct BezierModelToViewError {
  BezierModelToViewError(const Eigen::Matrix3d& R,
                         const Eigen::Vector3d& T,
                         double img_p_x,
                         double img_p_y,
                         double img_v_x,
                         double img_v_y,
                         double weight,
                         double tang_weight,
                         double focal_length,
                         double width,
                         double height,
                         double bezier_t) {
    weight_ = weight;
    tang_weight_ = tang_weight;
    focal_length_ = focal_length;
    p_x_ = img_p_y - width * 0.5;
    p_y_ = img_p_x - height * 0.5;
    v_x_ = img_v_x;
    v_y_ = img_v_y;
    bezier_t_ = bezier_t;

    // Camera poses.
    Eigen::AngleAxisd angle_axis(R);
    double angle = angle_axis.angle();
    Eigen::Vector3d axis = angle_axis.axis();
    for (int t = 0; t < 3; t++) {
      camera_[t] = axis(t) * angle;
      camera_[t + 3] = T(t);
    }
  }

  template <typename T>
  bool operator()(const T* const bezier_p0,
                  const T* const bezier_p1,
                  const T* const bezier_p2,
                  T* residuals) const {
    T point[3];
    point[0] = bezier_p0[0] * T((1.0 - bezier_t_) * (1.0 - bezier_t_)) +
               bezier_p1[0] * T(2.0 * (1.0 - bezier_t_) * bezier_t_) +
               bezier_p2[0] * T(bezier_t_ * bezier_t_);
    point[1] = bezier_p0[1] * T((1.0 - bezier_t_) * (1.0 - bezier_t_)) +
               bezier_p1[1] * T(2.0 * (1.0 - bezier_t_) * bezier_t_) +
               bezier_p2[1] * T(bezier_t_ * bezier_t_);
    point[2] = bezier_p0[2] * T((1.0 - bezier_t_) * (1.0 - bezier_t_)) +
               bezier_p1[2] * T(2.0 * (1.0 - bezier_t_) * bezier_t_) +
               bezier_p2[2] * T(bezier_t_ * bezier_t_);
    T p[3];
    // camera[0,1,2] are the angle-axis rotation.
    T camera[6] = { T(camera_[0]), T(camera_[1]), T(camera_[2]), T(camera_[3]), T(camera_[4]), T(camera_[5]) };
    ceres::AngleAxisRotatePoint(camera, point, p);
    // camera[3,4,5] are the translation.
    p[0] += camera[3]; p[1] += camera[4]; p[2] += camera[5];

    T bias_x = p[0] / p[2] * T(focal_length_) - T(p_x_);
    T bias_y = p[1] / p[2] * T(focal_length_) - T(p_y_);

    residuals[0] = (bias_x * T(v_y_) - bias_y * T(v_x_)) * T(weight_) / T(focal_length_);
    residuals[1] = (bias_x * T(v_x_) + bias_y * T(v_y_)) * T(weight_) * T(tang_weight_) / T(focal_length_);
    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const Eigen::Matrix3d& R,
                                     const Eigen::Vector3d& T,
                                     double img_p_x,
                                     double img_p_y,
                                     double img_v_x,
                                     double img_v_y,
                                     double weight,
                                     double tang_weight,
                                     double focal_length,
                                     double width,
                                     double height,
                                     double bezier_t) {
    // LOG(INFO) << "weight: " << weight;
    return (new ceres::AutoDiffCostFunction<BezierModelToViewError, 2, 3, 3, 3>(
        new BezierModelToViewError(R,
                                   T,
                                   img_p_x,
                                   img_p_y,
                                   img_v_x,
                                   img_v_y,
                                   weight,
                                   tang_weight,
                                   focal_length,
                                   width,
                                   height,
                                   bezier_t)));
  }

  double weight_;
  double tang_weight_;
  double focal_length_;
  double p_x_, p_y_;
  double v_x_, v_y_;
  double bezier_t_;
  double camera_[6];
};
