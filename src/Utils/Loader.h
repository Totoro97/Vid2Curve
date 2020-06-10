//
// Created by aska on 2019/4/12.
//

#pragma once
#include <Eigen/Eigen>

using Vector3d = Eigen::Vector3d;
using Matrix3d = Eigen::Matrix3d;

class PointsLoader {
public:
  PointsLoader() = default;
  virtual Vector3d operator [] (int idx) { return { 0.0, 0.0, 0.0 }; }
  virtual int size() { return 0; }
};

class HelixPointsLoader : public PointsLoader {
public:
  HelixPointsLoader(const Vector3d &base,
                    const Vector3d &r,
                    const Vector3d &up,
                    int n_points, double ratio, double step_up, double step_ang):
    base_(base), r_(r), up_(up), n_points_(n_points), ratio_(ratio), step_up_(step_up), step_ang_(step_ang) {
  }
  ~HelixPointsLoader() = default;

  Vector3d operator [] (int idx) final {
    double ang = idx * step_ang_;
    double ang_bias = 1e9;
    const double pi = std::acos(-1.0);
    for (int i = 0; i < 6; i++) {
      ang_bias = std::min(ang_bias, std::abs(2.0 * pi / i - ang));
    }
    Eigen::Matrix3d T;
    T = Eigen::AngleAxisd(ang, up_ / up_.norm());
    Vector3d new_r = T * r_ * std::pow(ratio_, idx);
    Vector3d new_point = new_r + up_ * step_up_ * idx;
    return new_point + base_;
  }

  int size() final {
    return n_points_;
  }

  Vector3d base_;
  Vector3d r_;
  Vector3d up_;
  int n_points_;
  double ratio_;
  double step_up_;
  double step_ang_;
};

class SamplePointsLoader : public PointsLoader {
public:
  SamplePointsLoader(const std::vector<Vector3d> &base_points, int n_points):
    base_points_(base_points), n_points_(n_points) {
  }

  Vector3d operator [] (int idx) final {
    int step_len = n_points_ / (base_points_.size() - 1);
    int i = idx / step_len;
    double bias = (double) (idx % step_len) / (double) step_len;
    return base_points_[i + 1] * bias + base_points_[i] * (1.0 - bias);
  }

  int size() final {
    return n_points_;
  }
  std::vector<Vector3d> base_points_;
  int n_points_;
};

class CubicBezierLoader : public PointsLoader {
public:
  CubicBezierLoader(const std::vector<Vector3d> &base_points, const std::vector<Vector3d> &dirs, double s):
    base_points_(base_points), dirs_(dirs), s_(s) {
    UpdatePoints();
  }

  CubicBezierLoader(int rand_num, int n, double s) {
    std::srand(rand_num);
    auto RandLR = [](double L, double R) {
      return std::rand() / (double) RAND_MAX * (R - L) + L;
    };
    std::vector<Vector3d> pools;
    for (int i = 0; i < 2 * n; i++) {
      pools.emplace_back(RandLR(-1.0, 1.0), RandLR(-1.0, 1.0), RandLR(-1.0, 1.0));
    }

    for (int T = 100000; T > 0; T--) {
      int a = std::rand() % (n * 2);
      int b = std::rand() % (n * 2);
      double past_dis = 0.0;
      for (int i = 0; i + 1 < n * 2; i++) {
        past_dis += (pools[i] - pools[i + 1]).norm();
      }
      std::swap(pools[a], pools[b]);
      double new_dis = 0.0;
      for (int i = 0; i + 1 < n * 2; i++) {
        new_dis += (pools[i] - pools[i + 1]).norm();
      }
      if (past_dis < new_dis) {
        std::swap(pools[a], pools[b]);
      }
    }

    for (int i = 1; i < n; i++) {
      base_points_.emplace_back(pools[i * 2]);
      dirs_.emplace_back(pools[i * 2 + 1]);
    }
    s_ = s;
    UpdatePoints();
  }

  void UpdatePoints() {
    auto n = base_points_.size();
    for (int i = 0; i + 1 < n; i++) {
      auto p0 = base_points_[i];
      auto p1 = dirs_[i];
      auto p2 = base_points_[i + 1] * 2.0 - dirs_[i + 1];
      auto p3 = base_points_[i + 1];
      for (double t = 0.0; t < 1.0; t += s_) {
        double s = 1.0 - t;
        auto pt = s * s * s * p0 + 3 * s * s * t * p1 + 3 * s * t * t * p2 + t * t * t * p3;
        points_.emplace_back(pt);
      }
    }
    n_points_ = points_.size();
  }

  Vector3d operator [] (int idx) final {
    return points_[idx];
  };

  int size() final {
    return n_points_;
  }

  std::vector<Vector3d> base_points_;
  std::vector<Vector3d> points_;
  std::vector<Vector3d> dirs_;
  double s_;
  int n_points_;
};

class SphereLoader : public PointsLoader {
public:
  SphereLoader(const Vector3d& center,
               double r,
               int n_segments,
               int n_rings,
               int n_points_per_segment) {
    const double pi = std::acos(-1.0);
    for (int i = 0; i < n_segments; i++) {
      int n = n_points_per_segment * (n_rings + 1);
      double a = pi * 2.0 / n_segments * i;
      for (int j = 0; j <= n; j++) {
        double b = pi / n * j - 0.5 * pi;
        Vector3d p(std::cos(a) * std::cos(b), std::sin(a) * std::cos(b), std::sin(b));
        points_.emplace_back(p * r);
      }
    }

    for (int i = 1; i < n_rings; i++) {
      int n = n_points_per_segment * n_segments;
      double b = pi / n_rings * i - 0.5 * pi;
      for (int j = 0; j < n; j++) {
        double a = 2.0 * pi * j / n;
        Vector3d p(std::cos(a) * std::cos(b), std::sin(a) * std::cos(b), std::sin(b));
        points_.emplace_back(p * r);
      }
    }
  }

  Vector3d operator [] (int idx) final {
    return points_[idx];
  };

  int size() final {
    return points_.size();
  }

  std::vector<Vector3d> points_;
};

class GridLoader : public PointsLoader {
public:
  GridLoader(const Vector3d& center,
             double r,
             int n_segments,
             int n_points_per_segment,
             bool simple = false) {
    points_.clear();
    center_ = center;
    if (!simple) {
      for (int i = 0; i <= n_segments; i++) {
        double a = 2.0 * r / n_segments * i - r;
        for (int j = 0; j <= n_segments; j++) {
          double b = 2.0 * r / n_segments * j - r;
          for (int k = 0; k <= n_segments * n_points_per_segment; k++) {
            double c = 2.0 * r / (n_segments * n_points_per_segment) * k - r;
            points_.emplace_back(a, b, c);
            points_.emplace_back(a, c, b);
            points_.emplace_back(c, a, b);
          }
        }
      }
    }
    else {
      for (int i = 0; i <= n_segments; i++) {
        double a = 2.0 * r / n_segments * i - r;
        for (int j = 0; j <= n_segments * n_points_per_segment; j++) {
          double b = 2.0 * r / (n_segments * n_points_per_segment) * j - r;
          points_.emplace_back(a, b, -r);
          points_.emplace_back(a, b, r);
          points_.emplace_back(b, a, -r);
          points_.emplace_back(b, a, r);
          points_.emplace_back(a, -r, b);
          points_.emplace_back(a, r, b);
          points_.emplace_back(b, -r, a);
          points_.emplace_back(b, r, a);
          points_.emplace_back(-r, a, b);
          points_.emplace_back(r, a, b);
          points_.emplace_back(-r, b, a);
          points_.emplace_back(r, b, a);
        }
      }
    }
  }

  Vector3d operator [] (int idx) final {
    return points_[idx] + center_;
  };

  int size() final {
    return points_.size();
  }

  std::vector<Vector3d> points_;
  Vector3d center_;
};

// ------------------------------------------------------------------------------------------------

class TracesLoader {
public:
  TracesLoader() = default;
  virtual Matrix3d operator [] (int idx) {
    return Matrix3d();
  }
  virtual int size() { return 0; }
};

class RotateTracesLoader : public TracesLoader {
public:
  RotateTracesLoader(const Vector3d &base_pos,
                     const Vector3d &up, const Vector3d &look_at, double step_ang, int n_traces) :
                     base_pos_(base_pos), up_(up), look_at_(look_at), step_ang_(step_ang), n_traces_(n_traces) {
    up_ /= up.norm();
  }
  Matrix3d operator [] (int idx) final {
    auto RandLR = [](double L, double R) {
      return std::rand() / (double) RAND_MAX * (R - L) + L;
    };
    double ang = idx * step_ang_;
    Eigen::Matrix3d T;
    T = Eigen::AngleAxisd(ang, up_ / up_.norm());
    Vector3d bias = base_pos_ - look_at_;
    bias = T * bias + Eigen::Vector3d(RandLR(-0.00, 0.00), RandLR(-0.00, 0.00), RandLR(-0.00, 0.00));
    Vector3d pos = bias + look_at_;
    Matrix3d ret;
    ret.block(0, 0, 3, 1) = pos;
    ret.block(0, 1, 3, 1) = -bias / bias.norm();
    ret.block(0, 2, 3, 1) = up_ / up_.norm();
    return ret;
  }

  int size() final {
    return n_traces_;
  }

  Vector3d base_pos_;
  Vector3d up_;
  Vector3d look_at_;
  double step_ang_;
  int n_traces_;
};

class RandomTracesLoader : public TracesLoader {
public:
  RandomTracesLoader(const Vector3d &base_pos, const Vector3d &look_to, double step_ang, int n_traces) :
    base_pos_(base_pos), look_to_(look_to), step_ang_(step_ang), n_traces_(n_traces) {
    auto RandLR = [](double L, double R) {
      return std::rand() / (double) RAND_MAX * (R - L) + L;
    };
    Vector3d rot(RandLR(0.0, 1.0), RandLR(0.0, 1.0), RandLR(0.0, 1.0));
    rot /= rot.norm();
    Vector3d bias = look_to_- base_pos_;
    double d = bias.norm();
    bias /= bias.norm();
    Vector3d up;
    if (std::abs(bias(0)) > 1e-8 || std::abs(bias(1)) > 1e-8) {
      up = Eigen::Vector3d(bias(1), -bias(0), 0.0);
    }
    else {
      up = Eigen::Vector3d(bias(2), 0.0, -bias(0));
    }
    up /= up.norm();
    for (int i = 0; i < n_traces; i++) {
      Matrix3d pose;
      Eigen::Vector3d shake(RandLR(-1.0, 1.0), RandLR(-1.0, 1.0), RandLR(-1.0, 1.0));
      shake /= shake.norm();
      shake *= RandLR(0.005, 0.01);
      pose.block(0, 0, 3, 1) = look_to_ - bias * d + shake;
      pose.block(0, 1, 3, 1) = bias / bias.norm();
      pose.block(0, 2, 3, 1) = up / up.norm();
      poses_.emplace_back(pose);
      Eigen::Matrix3d T;
      T = Eigen::AngleAxisd(step_ang_, rot);
      bias = T * bias;
      up = T * up;
      bias /= bias.norm();
      up = up - bias * bias.dot(up);
      up /= up.norm();
      rot = Vector3d(RandLR(0.0, 1.0), RandLR(0.0, 1.0), RandLR(0.0, 1.0));
      rot /= rot.norm();
    }
  }

  Matrix3d operator [] (int idx) final {
    return poses_[idx];
  }

  int size() final {
    return n_traces_;
  }
  Vector3d base_pos_;
  Vector3d look_to_;
  double step_ang_;
  int n_traces_;
  std::vector<Matrix3d> poses_;
};

class TranslateTracesLoader : public TracesLoader {
public:
  TranslateTracesLoader(const Vector3d &base_pos,
                     const Vector3d &up, const Vector3d &look_at, const Vector3d &v, double step, int n_traces) :
    base_pos_(base_pos), up_(up), look_at_(look_at), v_(v), step_(step), n_traces_(n_traces) {
    v_ /= v_.norm();
    up_ /= up.norm();
  }
  Matrix3d operator [] (int idx) final {
    double len = idx * step_;
    Eigen::Matrix3d T;
    Vector3d pos = base_pos_ + v_ * len;
    Matrix3d ret;
    ret.block(0, 0, 3, 1) = pos;
    ret.block(0, 1, 3, 1) = look_at_ - base_pos_;
    ret.block(0, 2, 3, 1) = up_ / up_.norm();
    return ret;
  }

  int size() final {
    return n_traces_;
  }

  Vector3d base_pos_;
  Vector3d up_;
  Vector3d look_at_;
  Vector3d v_;
  double step_;
  int n_traces_;
};