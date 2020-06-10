//
// Created by aska on 2020/2/18.
//
#include "Common.h"

struct BezierPointExpIdx {
  BezierPointExpIdx(int p0_idx = -1, int p1_idx = -1, int t_idx = -1, double t = -1) :
      p0_idx_(p0_idx), p1_idx_(p1_idx), t_idx_(t_idx), t_(t) {}
  int p0_idx_, p1_idx_;
  int t_idx_;
  double t_;
};

class BezierCurve {
public:
  BezierCurve(const std::vector<Eigen::Vector3d>& initial_points,
              const std::vector<double>& errors,
              double error_threshold,
              double hope_dist);
  std::vector<Eigen::Vector3d> points_p_;
  std::vector<Eigen::Vector3d> points_t_;
  std::vector<BezierPointExpIdx> expressions_;
  double hope_dist_ = 0.0;

private:
  void BuildBezierCurve(const std::vector<Eigen::Vector3d>& initial_points,
                        const std::vector<double>& errors,
                        double error_threshold,
                        int l_bound,
                        int r_bound);
  bool IsSingleFittingOK(const std::vector<Eigen::Vector3d>& initial_points,
                         const std::vector<double>& errors,
                         double error_threshold,
                         int l_bound,
                         int r_bound,
                         Eigen::Vector3d* estimated_p1 = nullptr);
};