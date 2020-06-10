//
// Created by aska on 2019/3/6.
//

#include "CurveMatcher.h"
#include "CurveExtractor.h"
#include "../Utils/Common.h"
#include "../Utils/QuadTree.h"
#include "../Utils/Utils.h"

#include <ceres/ceres.h>

#include <chrono>
#include <cstdlib>
#include <iostream>

CurveMatcher::CurveMatcher(
  CurveExtractor *extractor_0, CurveExtractor *extractor_1) {
  extractor_0_ = extractor_0;
  extractor_1_ = extractor_1;
  height_ = extractor_0_->height_;
  width_ = extractor_0_->width_;
}

CurveMatcher::CurveMatcher(int height, int width, double *prob_map_0, double *prob_map_1) :
  height_(height), width_(width), prob_map_0_(prob_map_0), prob_map_1_(prob_map_1) {}


CurveMatcher::~CurveMatcher() {}

std::pair<Eigen::Vector2d, double> CurveMatcher::FindMatching(Eigen::Vector2d pix) {
  auto FindNearest = [this](int base_a, int base_b, double *prob_map) {
    for (int r = 0; r <= 10; r++) {
      for (int x = std::max(0, base_a - r); x <= std::min(height_ - 1, base_a + r); x++) {
        double y_d = std::sqrt((double) (r * r - (base_a - x) * (base_a - x)));
        int y;
        for (int k = -1; k <= 1; k += 2) {
          y = base_b + (int) y_d * k;
          if (y < width_ && y >= 0 && prob_map[x * width_ + y] > 0.9) {
            return std::make_pair(x, y);
          }
        }
      }
    }
    return std::make_pair(-1, -1);
  };
  auto pr = FindNearest((int) std::round(pix(0)), (int) std::round(pix(1)), extractor_0_->prob_map_);
  int a = pr.first;
  int b = pr.second;
  if (a < 0 || a >= height_ || b < 0 || b >= width_) {
    return { { 0.0, 0.0 }, 0.0 };
  }
  else {
    float flow_a = flow_img_.at<cv::Vec2f>(a, b)[1];
    float flow_b = flow_img_.at<cv::Vec2f>(a, b)[0];
    int base_a = (int) std::round(flow_a) + a;
    int base_b = (int) std::round(flow_b) + b;
    auto pr = FindNearest(base_a, base_b, extractor_1_->prob_map_);
    int x = pr.first;
    int y = pr.second;
    if (x < 0 || x >= height_ || y < 0 || y >= width_) {
      return { { 0.0, 0.0 }, 0.0 };
    }
    else return { { x, y }, 1.0 };
  }
}

void CurveMatcher::FindMatchingsByOpticalFlow(std::vector<ImageLocalMatching> *matchings) {
  int height = extractor_0_->height_;
  int width = extractor_0_->width_;
  cv::Mat img_0(height, width, CV_8UC1);
  cv::Mat img_1(height, width, CV_8UC1);
  std::memset(img_0.data, 0, height * width * 1);
  std::memset(img_1.data, 0, height * width * 1);

  // Get images to compare.
  for (const auto &dir_point : extractor_0_->candi_points_) {
    int a = std::round(dir_point.o_(0));
    int b = std::round(dir_point.o_(1));
    if (a < height && a >= 0 && b < width && b >= 0) {
      for (int t = 0; t < 1; t++) {
        img_0.data[(a * width + b) + t] = 255;
      }
    }
  }
  for (const auto &dir_point : extractor_1_->candi_points_) {
    int a = std::round(dir_point.o_(0));
    int b = std::round(dir_point.o_(1));
    if (a < height && a >= 0 && b < width && b >= 0) {
      for (int t = 0; t < 1; t++) {
        img_1.data[(a * width + b) + t] = 255;
      }
    }
  }

  // Get rough points.
  Utils::GetOpticalFlow(img_0, img_1, &flow_img_, true);
  std::vector<ImageLocalMatching> rough_matchings;
  std::vector<Eigen::Vector2d> rough_points;

  // Offline scan.
  cv::Mat debug_image(img_0.rows, img_0.cols, CV_8UC3);
  std::memset(debug_image.data, 0, img_0.rows * img_0.cols * 3);
  for (int pt_i = 0; pt_i < extractor_0_->n_points_; pt_i++) {
    if (extractor_0_->near_junction_[pt_i]) {
      continue;
    }
    const auto& dir_point = extractor_0_->candi_points_[pt_i];
    int a = std::round(dir_point.o_(0));
    int b = std::round(dir_point.o_(1));
    if (a < 0 || a >= height || b < 0 || b >= width) {
      continue;
    }
    float flow_a = flow_img_.at<cv::Vec2f>(a, b)[1];
    float flow_b = flow_img_.at<cv::Vec2f>(a, b)[0];
    LOG(INFO) << "a, b" << a << " " << b << "..." << "flow_a,b" << flow_a << " " << flow_b;
    ImageDirPoint rough_matched_dir_point(
      Eigen::Vector2d(flow_a + dir_point.o_(0), flow_b + dir_point.o_(1)), dir_point.v_);
    rough_points.emplace_back(flow_a + dir_point.o_(0), flow_b + dir_point.o_(1));
    cv::circle(debug_image, cv::Point(flow_b + dir_point.o_(1), flow_a + dir_point.o_(0)), 0, cv::Scalar(255, 255, 255));
    rough_matchings.emplace_back(dir_point, rough_matched_dir_point);
  }
  for (const auto& pt : extractor_1_->points_) {
    cv::circle(debug_image, cv::Point(pt(1), pt(0)), 0, cv::Scalar(255, 0, 255));
  }
  cv::imshow("tmp", debug_image);
  cv::waitKey(500);
  cv::destroyAllWindows();
  std::vector<ImageDirPoint> nearest_points;
  extractor_1_->FindNearestPointsOffline(rough_points, &nearest_points);

  for (unsigned i = 0; i < rough_matchings.size(); i++) {
    rough_matchings[i].q = nearest_points[i];
  }
  *matchings = std::move(rough_matchings);
}

void CurveMatcher::FindMatchingsByDP(std::vector<ImageLocalMatching>* matchings) {
  int height = extractor_0_->height_;
  int width = extractor_0_->width_;
  cv::Mat img_0(height, width, CV_8UC1);
  cv::Mat img_1(height, width, CV_8UC1);
  std::memset(img_0.data, 0, height * width * 1);
  std::memset(img_1.data, 0, height * width * 1);

  // Get images to compare.
  for (const auto &dir_point : extractor_0_->candi_points_) {
    int a = std::round(dir_point.o_(0));
    int b = std::round(dir_point.o_(1));
    if (a < height && a >= 0 && b < width && b >= 0) {
      for (int t = 0; t < 1; t++) {
        img_0.data[(a * width + b) + t] = 255;
      }
    }
  }
  for (const auto &dir_point : extractor_1_->candi_points_) {
    int a = std::round(dir_point.o_(0));
    int b = std::round(dir_point.o_(1));
    if (a < height && a >= 0 && b < width && b >= 0) {
      for (int t = 0; t < 1; t++) {
        img_1.data[(a * width + b) + t] = 255;
      }
    }
  }

  // Get rough points.
  Utils::GetOpticalFlow(img_0, img_1, &flow_img_, true);


  auto quad_tree = std::make_unique<QuadTree>(extractor_1_->points_);
  matchings->clear();
  auto& paths = extractor_0_->paths_;
  // Find Candidates.
  const double kSearchingR = 10.0;

  std::vector<Eigen::Vector2d> rough_points;

  for (int u = 0; u < extractor_0_->n_points_; u++) {
    const Eigen::Vector2d& pt = extractor_0_->points_[u];
    int a = std::round(pt(0));
    int b = std::round(pt(1));
    a = std::min(std::max(a, 0), height - 1);
    b = std::min(std::max(b, 0), width - 1);
    double flow_a = flow_img_.at<cv::Vec2f>(a, b)[1];
    double flow_b = flow_img_.at<cv::Vec2f>(a, b)[0];
    rough_points.emplace_back(pt(0) + flow_a, pt(1) + flow_b);
  }
  auto SingleCost = [this, &rough_points](int u, int v) {
    return (rough_points[u] - extractor_1_->points_[v]).norm() * 1.0;
  };

  auto JointCost = [this](int u, int u_c, int v, int v_c) {
    return std::abs((extractor_0_->points_[u] - extractor_0_->points_[v]).norm() -
            (extractor_1_->points_[u_c]- extractor_1_->points_[v_c]).norm());
  };

  for (const auto& path : paths) {
    CHECK(!path.empty());
    std::vector<std::vector<int>> candidates(path.size());
    std::vector<std::vector<double>> f(path.size());
    std::vector<std::vector<int>> past_choice(path.size());
    for (int u = 0; u < path.size(); u++) {

      quad_tree->SearchingR(rough_points[path[u]], kSearchingR, &candidates[u]);
      if (candidates[u].empty()) {
        candidates[u].emplace_back(quad_tree->NearestIdx(rough_points[path[u]]));
      }
      f[u].resize(candidates[u].size(), 1e9);
      past_choice[u].resize(candidates[u].size(), -1);
    }
    for (int v = 0; v < candidates[0].size(); v++) {
      f[0][v] = SingleCost(path[0], candidates[0][v]);
    }
    for (int u = 1; u < path.size(); u++) {
      for (int u_c = 0; u_c < candidates[u].size(); u_c++) {
        for (int v_c = 0; v_c < candidates[u - 1].size(); v_c++) {
          double new_cost = f[u - 1][v_c] + SingleCost(path[u], candidates[u][u_c]) * 1.0 +
                            0.5 * JointCost(path[u], candidates[u][u_c], path[u - 1], candidates[u - 1][v_c]);
          if (new_cost < f[u][u_c]) {
            f[u][u_c] = new_cost;
            past_choice[u][u_c] = v_c;
          }
        }
      }
    }

    int c = -1;
    double final_cost = 1e10;
    for (int i = 0; i < f.back().size(); i++) {
      if (f.back()[i] < final_cost) {
        final_cost = f.back()[i];
        c = i;
      }
    }
    for (int u = path.size() - 1; u >= 0; u--) {
      CHECK(c >= 0);
      if (!extractor_0_->near_junction_[path[u]]) {
        matchings->emplace_back(extractor_0_->candi_points_[path[u]], extractor_1_->candi_points_[candidates[u][c]]);
      }
      c = past_choice[u][c];
    }
  }
}

void CurveMatcher::FindMatchings(
    const std::string& method, std::vector<ImageLocalMatching>*matchings, bool show_debug_message) {
  matchings->clear();
  if (method == "LOCAL_CURVE_DESCRIPTOR") {
    // TODO: FindLocalCurveMatching(...)
  }
  else if (method == "OPTICAL_FLOW") {
    FindMatchingsByOpticalFlow(matchings);
  }
  else if (method == "DP") {
    FindMatchingsByDP(matchings);
  }
  if (show_debug_message) {
    cv::Mat curve_img(height_, width_, CV_8UC3);
    std::memset(curve_img.data, 0, height_ * width_ * 3);
    for (const auto &pt : extractor_0_->candi_points_) {
      cv::rectangle(curve_img,
                    cv::Point(pt.o_(1), pt.o_(0)),
                    cv::Point(pt.o_(1), pt.o_(0)),
                    cv::Scalar(255, 255, 255),
                    1);
    }
    for (const auto &pt : extractor_1_->candi_points_) {
      cv::rectangle(curve_img,
                    cv::Point(pt.o_(1), pt.o_(0)),
                    cv::Point(pt.o_(1), pt.o_(0)),
                    cv::Scalar(255, 255, 0),
                    1);
    }
    cv::imshow("debug_matching", curve_img);
    cv::waitKey(-1);
    for (int i = 0; i < matchings->size(); i += 5) {
      const auto& matching = matchings->at(i);
      Eigen::Vector2d p = matching.p.o_;
      Eigen::Vector2d q = matching.q.o_;
      cv::line(curve_img,
               cv::Point(p(1), p(0)),
               cv::Point(q(1), q(0)),
               cv::Scalar(255, 0, 255),
               1);
    }
    cv::imshow("debug_matching", curve_img);
    cv::waitKey(-1);

  }
}

void CurveMatcher::EstimateRT(double focal_length,
                              const std::string& method,
                              std::vector<std::pair<Eigen::Matrix3d, Eigen::Vector3d>> *poses) {
  std::vector<ImageLocalMatching> matchings;
  FindMatchings("OPTICAL_FLOW", &matchings);

  if (method == "WEAKEN_TANG_MULTI") {
    using ceres::NumericDiffCostFunction;
    using ceres::CostFunction;
    using ceres::Problem;
    using ceres::Solver;
    using ceres::Solve;

    auto RandLR = [](double L, double R) {
      return double(std::rand()) / RAND_MAX * (R - L) + L;
    };

    for (int try_num = 0; try_num < 5; try_num++) {
      Problem problem;
      double parameters[5] = { 0.0, 0.0, 0.0, 0.0, 0.0 };
      for (int i = 3; i < 5; i++) {
        parameters[i] = RandLR(0.0, 0.0);
      }

      for (const auto &matching : matchings) {
        Eigen::Vector2d A = matching.p.o_;
        Eigen::Vector2d B = matching.q.o_;
        Eigen::Vector3d V1((A(1) - extractor_0_->width_ * 0.5) / focal_length,
                           (A(0) - extractor_0_->height_ * 0.5) / focal_length,
                           1.0);
        V1 /= V1.norm();
        Eigen::Vector3d V2((B(1) - extractor_1_->width_ * 0.5) / focal_length,
                           (B(0) - extractor_1_->height_ * 0.5) / focal_length,
                           1.0);
        V2 /= V2.norm();
        Eigen::Vector3d M(matching.q.v_(1), matching.q.v_(0), 0.0);
        M -= V2.dot(M) * V2;
        M /= M.norm();
        Eigen::Vector3d N = M.cross(V2);
        CostFunction *cost_function =
          new NumericDiffCostFunction<TangCostFunctor, ceres::NumericDiffMethodType::CENTRAL, 1, 5>(
            new TangCostFunctor(V1, V2, N, M));
        problem.AddResidualBlock(cost_function, nullptr, parameters);
      }

      Solver::Options options;
      options.max_num_iterations = 100;
      Solver::Summary summary;
      Solve(options, &problem, &summary);
      LOG(INFO) << summary.BriefReport();
      Eigen::Matrix3d R;
      R = Eigen::AngleAxisd(parameters[0], Vector3d::UnitX()) *
          Eigen::AngleAxisd(parameters[1], Vector3d::UnitY()) *
          Eigen::AngleAxisd(parameters[2], Vector3d::UnitZ());
      double x = parameters[3];
      double y = parameters[4];
      Eigen::Vector3d T(std::cos(x) * std::sin(y), std::cos(x) * std::cos(y), std::sin(x));
      if (summary.IsSolutionUsable()) {
        poses->emplace_back(R, T);
      }
    }
  } else if (method == "WEAKEN_TANG") {
    std::vector<Eigen::Matrix3d> Ms;
    std::vector<Eigen::Vector3d> V1s;
    std::vector<Eigen::Vector3d> V2s;
    int n_matchings = matchings.size();

    std::cout << "n_macthings: " << n_matchings << std::endl;
    for (const auto &matching : matchings) {
      Eigen::Vector2d A = matching.p.o_;
      Eigen::Vector2d B = matching.q.o_;
      auto pr_B = extractor_1_->CalcTangAndScore(B);
      B += pr_B.first * (std::rand() / double(RAND_MAX) - 0.5) * 0.5;

      Eigen::Vector3d V1((A(1) - extractor_0_->width_ * 0.5) / focal_length,
                         (A(0) - extractor_0_->height_ * 0.5) / focal_length,
                         1.0);
      V1s.emplace_back(V1);
      Eigen::Vector3d V2((B(1) - extractor_1_->width_ * 0.5) / focal_length,
                         (B(0) - extractor_1_->height_ * 0.5) / focal_length,
                         1.0);
      V2s.emplace_back(V2);
      Eigen::Vector3d tang(pr_B.first(1), pr_B.first(0), 0.0);
      Eigen::Vector3d V2_normed = V2 / V2.norm();
      tang -= V2_normed * tang.dot(V2_normed);
      tang /= tang.norm();
      Eigen::Vector3d nor = tang.cross(V2_normed);
      nor /= nor.norm();
      double score = 0.0;
      Eigen::Matrix3d M = Eigen::Matrix3d::Zero();
      M.block(0, 0, 1, 3) = tang.transpose() * (1.0 - score);
      M.block(1, 0, 1, 3) = nor.transpose();
      Ms.emplace_back(M);
    }

    Eigen::Vector3d T(0.0, 0.0, 0.0);
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();

    auto UpdateT = [n_matchings, &T, &R, &Ms, &V1s, &V2s, this]() {
      std::vector<Eigen::Vector3d> debug_points;

      debug_points.emplace_back(0.0, 0.0, 0.0);
      Eigen::Matrix3d C = Eigen::Matrix3d::Zero();
      for (int i = 0; i < n_matchings; i++) {
        auto X = Utils::GetLeftCrossProdMatrix(Ms[i] * R * V1s[i]) * Ms[i]; // X: D * 3;
        C += X.transpose() * X;
      }
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(C);
      T = solver.eigenvectors().block(0, 0, 3, 1);
      std::cout << "T: " << T.transpose() << std::endl;
      std::cout << "EigenValues: " << solver.eigenvalues().transpose() << std::endl;
      Utils::SavePointsAsPly("debug_points.ply", debug_points);
    };

    auto UpdateR = [n_matchings, &T, &R, &Ms, &V1s, &V2s, this]() {
      std::vector<cv::Point2d> P1s, P2s;
      for (int i = 0; i < n_matchings; i++) {
        Eigen::Matrix3d P = Utils::GetLeftCrossProdMatrix(Ms[i] * T) * Ms[i]; // P: 3 * 3;
        for (int t = 0; t < 3; t++) {
          Eigen::Vector3d P2 = P.block(t, 0, 1, 3).transpose();
          Eigen::Vector3d P1 = V1s[i];
          P1s.emplace_back(P1(0) / P1(2), P1(1) / P1(2));
          P2s.emplace_back(P2(0) / P2(2), P2(1) / P2(2));
        }
      }
      auto E_mat = cv::findEssentialMat(P1s, P2s);
      Eigen::Matrix3d E;
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          E(i, j) = E_mat.at<double>(i, j);
        }
      }
      Eigen::JacobiSVD<Eigen::Matrix3d> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
      auto U = svd.matrixU();
      auto V = svd.matrixV();
      R = U * V.transpose();
    };

    int max_iter_num = 10;
    for (int i = 0; i < max_iter_num; i++) {
      UpdateT();
      std::exit(0);
    }
    poses->emplace_back(R, T);
  }
  else if (method == "NAIVE") {
    std::vector<Eigen::Matrix3d> Ms;
    std::vector<Eigen::Matrix3d> Ps;
    std::vector<Eigen::Vector3d> V1s;
    std::vector<Eigen::Vector3d> V2s;
    int n_matchings = matchings.size();
    std::cout << "n_macthings: " << n_matchings << std::endl;
    for (const auto &matching : matchings) {
      Eigen::Vector2d A = matching.p.o_;
      Eigen::Vector2d B = matching.q.o_;
      auto pr_B = extractor_1_->CalcTangAndScore(B);

      Eigen::Vector3d V1((A(1) - extractor_0_->width_ * 0.5) / focal_length,
                         (A(0) - extractor_0_->height_ * 0.5) / focal_length,
                         1.0);
      V1s.emplace_back(V1);
      Eigen::Vector3d V2((B(1) - extractor_1_->width_ * 0.5) / focal_length,
                         (B(0) - extractor_1_->height_ * 0.5) / focal_length,
                         1.0);
      V2s.emplace_back(V2);
      Eigen::Vector3d tang(pr_B.first(1), pr_B.first(0), 0.0);
      tang /= tang.norm();
      double score = 0.0;
      Ms.emplace_back(Eigen::Matrix3d::Identity() - score * tang * tang.transpose());
      const auto &M = Ms.back();
      Eigen::Matrix3d P = (M.transpose() * Utils::GetLeftCrossProdMatrix(M * V2).transpose() * M).transpose();
      Ps.emplace_back(P);
    }

    Eigen::Vector3d T(0.0, 0.0, 0.0);
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();

    auto UpdateT = [n_matchings, &T, &R, &Ms, &Ps, &V1s, &V2s, this]() {
      std::vector<Eigen::Vector3d> debug_points;

      debug_points.emplace_back(0.0, 0.0, 0.0);
      Eigen::Matrix3d C = Eigen::Matrix3d::Zero();
      for (int i = 0; i < n_matchings; i++) {
        auto X = Ps[i] * R * V1s[i];
        debug_points.emplace_back(X);
        C += X * X.transpose();
      }
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(C);
      T = solver.eigenvectors().block(0, 0, 3, 1);
      std::cout << "T: " << T.transpose() << std::endl;
      std::cout << "EigenValues: " << solver.eigenvalues().transpose() << std::endl;
      Utils::SavePointsAsPly("debug_points.ply", debug_points);
    };
    auto UpdateR = [n_matchings, &T, &R, &Ms, &Ps, &V1s, &V2s, this]() {
      std::vector<cv::Point2d> P1s, P2s;
      for (int i = 0; i < n_matchings; i++) {
        auto P2 = Ps[i].transpose() * T;
        auto P1 = V1s[i];
        P1s.emplace_back(P1(0) / P1(2), P1(1) / P1(2));
        P2s.emplace_back(P2(0) / P2(2), P2(1) / P2(2));
      }
      auto E_mat = cv::findEssentialMat(P1s, P2s);
      Eigen::Matrix3d E;
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          E(i, j) = E_mat.at<double>(i, j);
        }
      }
      Eigen::JacobiSVD<Eigen::Matrix3d> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
      auto U = svd.matrixU();
      auto V = svd.matrixV();
      R = U * V.transpose();
    };

    // TODO: Hard code here.
    int max_iter_num = 10;
    for (int i = 0; i < max_iter_num; i++) {
      UpdateT();
      std::exit(0);
    }
    poses->emplace_back(R, T);
  }
}
