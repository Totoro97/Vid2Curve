//
// Created by aska on 2019/3/6.
//
#pragma once
#include "CurveExtractor.h"
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <vector>

using Vector3d = Eigen::Vector3d;
using Vector2d = Eigen::Vector2d;

// Local Core in image coordinate system.
struct ImageLocalMatching {
  ImageLocalMatching(const ImageDirPoint &p,
                     const ImageDirPoint &q) : p(p), q(q) {}
  ImageDirPoint p;
  ImageDirPoint q;
};

struct TangCostFunctor {
  TangCostFunctor(const Eigen::Vector3d &P,
                  const Eigen::Vector3d &Q,
                  const Eigen::Vector3d &N,
                  const Eigen::Vector3d &M) : P_(P), Q_(Q), N_(N), M_(M) {}
  bool operator() (const double *const parameters, double *residual) const {
    // residual[0] = (10 - parameters[0]) * (10 - parameters[0]) + parameters[0] * parameters[0];
    // return true;
    Eigen::Matrix3d R;
    R = Eigen::AngleAxisd(parameters[0], Vector3d::UnitX()) *
        Eigen::AngleAxisd(parameters[1], Vector3d::UnitY()) *
        Eigen::AngleAxisd(parameters[2], Vector3d::UnitZ());
    double x = parameters[3];
    double y = parameters[4];
    Eigen::Vector3d T(std::cos(x) * std::sin(y), std::cos(x) * std::cos(y), std::sin(x));
    residual[0] = (R * P_).cross(Q_).dot(T);
    return true;
    double t = (Q_ - T).dot(N_) / ((R * P_).dot(N_) + 1e-9);
    residual[0] = (t * R * P_ + T - Q_).dot(M_);
    return true;
  }

private:
  Eigen::Vector3d P_, Q_, N_, M_;
};

// Matcher should be more global.
class CurveMatcher {
public:
  // Methods
  CurveMatcher(int height, int width, double *prob_map_0, double *prob_map_1);
  CurveMatcher(CurveExtractor *extractor_0,
               CurveExtractor *extractor_1);
  ~CurveMatcher();

  void FindLocalCurveMatching(std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>>& matchings);
  std::pair<Eigen::Vector2d, double> FindMatching(Eigen::Vector2d pix);
  void EstimateRT(double focal_length,
                  const std::string& method,
                  std::vector<std::pair<Eigen::Matrix3d, Eigen::Vector3d>>* poses);
  void FindMatchings(const std::string &method, std::vector<ImageLocalMatching>* matchings,
                     bool show_debug_message = false);
  void FindMatchingsByOpticalFlow(std::vector<ImageLocalMatching>* matchings);
  void FindMatchingsByDP(std::vector<ImageLocalMatching>* matchings);
  bool ValidateCandiCurves();

  void ShowDebugInfo();

  int height_, width_;
  double *prob_map_0_, *prob_map_1_;
  CurveExtractor *extractor_0_, *extractor_1_;
  cv::Mat img_0_, img_1_;
  cv::Mat flow_img_;
};
