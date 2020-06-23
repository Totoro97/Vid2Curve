//
// Created by aska on 2019/3/12.
//
// This is new.

// #include "Algonoke/Algonoke.h"
#include "CurveExtractor.h"
#include "../Utils/Common.h"
#include "../Utils/Math.h"
#include "../Utils/Thinning.h"
#include "../Utils/Utils.h"

#include <algorithm>
#include <numeric>
#include <random>

// Deprecated.
CurveExtractor::CurveExtractor(double* prob_map, int height, int width): prob_map_(prob_map) {
  need_delete_prob_map_ = false;
  height_ = height;
  width_ = width;
  RunPca();
}

CurveExtractor::CurveExtractor(StreamerBase* streamer, PropertyTree* ptree) {
  ptree_ = ptree;
  std::string method = ptree->get<std::string>("CurveExtractor.ExtractionMethod");
  use_gpu_ = ptree->get<bool>("CurveExtractor.UseGPU");
  CHECK(!use_gpu_) << "Currently we don't support GPU.";
  height_ = streamer->Height();
  width_ = streamer->Width();
  seg_map_.resize(height_ * width_, 0);
  if (method == "NAIVE") {
    // Get prob map / intensity map.
    need_delete_prob_map_ = true;
    bool need_thinning = true;
    cv::Mat img(height_, width_, CV_8UC3);
    std::memcpy(img.data, streamer->CurrentFrame(), height_ * width_ * 3);
    if (need_thinning) {
      CHECK_EQ(streamer->Channels(), 3);
      cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
      cv::threshold(img, img, 200, 255, cv::THRESH_BINARY_INV);
      for (int a = 0; a < height_; a++) {
        for (int b = 0; b < width_; b++) {
          seg_map_[a * width_ + b] = img.data[a * width_ + b] ? 1 : 0;
        }
      }
      thinning(img, img);
      int c = img.channels();
      prob_map_ = new double[height_ * width_];
      for (int a = 0; a < height_; a++) {
        for (int b = 0; b < width_; b++) {
          if (img.data[(a * width_ + b) * c] == 0) {
            prob_map_[a * width_ + b] = 0.0;
          } else {
            prob_map_[a * width_ + b] = 1.0;
          }
        }
      }
    } else {
      prob_map_ = ConvertImg2ProbMap(img);
      for (int a = 0; a < height_; a++) {
        for (int b = 0; b < width_; b++) {
          seg_map_[a * width_ + b] = prob_map_[a * width_ + b] > 0.5 ? 1 : 0;
        }
      }
    }

    if (!use_gpu_) {
      RunPca();
    } else {
      RunPcaGPU();
    }
  } else if (method == "PCA_COLOR_DIFF") {
    RunPcaConsideringColor(streamer);
  } else if (method == "GRADIENT") {
    RunGradientBasedExtraction(streamer);
  } else {
    LOG(FATAL) << "Unsupported curve extraction method";
  }

  // When the control flow arrives here, pls make sure we have already obtained:
  // 1. points_
  // 2. tangs_
  // 3. tang_scores_
  // 4. candi_points_
  // 5. seg_map_
  CalcEstimateRadius();
  FilterTooThickPoints();
  CalcLinkInformation();
  CalcPaths();
  SmoothPoints();
}

CurveExtractor::~CurveExtractor() {
  if (need_delete_prob_map_) {
    delete (prob_map_);
  }
}


void CurveExtractor::FilterTooThickPoints() {
  // TODO
}

double CurveExtractor::EstimateR(double base_a, double base_b, double step_a, double step_b) {
  int base_a_int = std::round(base_a);
  int base_b_int = std::round(base_b);
  if (base_a_int < 0 || base_a_int >= height_ || base_b_int < 0 || base_b_int >= width_) {
    return 1e9; // Not valid;
  }
  int r = 1;
  while (true) {
    double a_try = base_a + r * step_a;
    double b_try = base_b + r * step_b;
    int new_a_int = int(std::round(a_try));
    int new_b_int = int(std::round(b_try));
    if (new_a_int < 0 || new_a_int >= height_ || new_b_int < 0 || new_b_int >= width_) {
      break;
    }
    if (!seg_map_[new_a_int * width_ + new_b_int]) {
      break;
    }
    r++;
  }
  return (double) r;
};

void CurveExtractor::CalcEstimateRadius() {
  CHECK_EQ(n_points_, tangs_.size());
  CHECK_EQ(n_points_, points_.size());
  CHECK_EQ(n_points_, candi_points_.size());
  estimated_rs_.resize(n_points_, 0.0);

  for (int i = 0; i < n_points_; i++) {
    const Eigen::Vector2d &pt = points_[i];
    Eigen::Vector2d initial_v(-tangs_[i](1), tangs_[i](0));
    initial_v /= initial_v.norm();
    estimated_rs_[i] = 1e9;
    int n_slices = 32;
    for (int t = 0; t < n_slices; t++) {
      double co = std::cos(kDoublePi * t / n_slices);
      double si = std::sin(kDoublePi * t / n_slices);
      Eigen::Vector2d v(co * initial_v(0) - si * initial_v(1), si * initial_v(0) + co * initial_v(1));
      double r_1 = EstimateR(pt(0), pt(1), v(0), v(1));
      double r_2 = EstimateR(pt(0), pt(1), -v(0), -v(1));
      estimated_rs_[i] = std::max(0.5, std::min(estimated_rs_[i], std::min(r_1, r_2) - 0.5));
    }
  }
  average_radius_ = std::accumulate(estimated_rs_.begin(), estimated_rs_.end(), 0.0) / estimated_rs_.size();
  std::vector<int> p(n_points_, 0);
  std::iota(p.begin(), p.end(), 0);
  std::sort(p.begin(), p.end(), [this](int a, int b) { return estimated_rs_[a] < estimated_rs_[b]; });
  thick_ratio_ = std::vector<double>(n_points_, 0.0);
  for (int i = 0; i < n_points_; i++) {
    thick_ratio_[p[i]] = double(i) / n_points_;
  }
}

double CurveExtractor::RadiusAt(double a, double b) {
  if (a < 1e-9 || a + 1 + 1e-9 > height_ || b < 1e-9 || b + 1 + 1e-9 > width_) {
    return -1.0;
  }
  return RadiusAtInt(std::round(a), std::round(b));
}

double CurveExtractor::RadiusAtInt(int a, int b) {
  int n_slices = 16;
  Eigen::Vector2d initial_v(1.0, 0.0);
  double ret_r = 1e5;
  for (int t = 0; t < n_slices; t++) {
    double co = std::cos(kDoublePi * t / n_slices);
    double si = std::sin(kDoublePi * t / n_slices);
    Eigen::Vector2d v(co * initial_v(0) - si * initial_v(1), si * initial_v(0) + co * initial_v(1));
    double r_1 = EstimateR(a, b, v(0), v(1));
    double r_2 = EstimateR(a, b, -v(0), -v(1));
    ret_r = std::max(0.5, std::min(ret_r, std::min(r_1, r_2) - 0.5));
  }
  return ret_r;
}
  
void CurveExtractor::RunPcaConsideringColor(StreamerBase* streamer) {
  CHECK(prob_map_ == nullptr);
  CHECK_EQ(streamer->Channels(), 3);

  int radius = ptree_->get<int>("CurveExtractor.Radius");
  if (use_gpu_) {
    // Data format of each single pixel: M[0, 0], M[0, 1], M[1, 0], M[1, 1], sum_bias.a, sum_bias.b, sum_w;
    const int kDataChannels = 7;
    auto out_data_pool = std::make_unique<double[]>(height_ * width_ * kDataChannels * sizeof(double));

    cv::Mat debug_img(height_, width_, CV_8UC1);
    for (int i = 0; i < height_; i++) {
      for (int j = 0; j < width_; j++) {
        int idx = i * width_ + j;
        Eigen::Matrix2d M;
        M << out_data_pool[idx * kDataChannels + 0], out_data_pool[idx * kDataChannels + 1],
             out_data_pool[idx * kDataChannels + 2], out_data_pool[idx * kDataChannels + 3];
        double l0 = 0, l1 = 0;
        Eigen::Vector2d v0, v1;
        CalcEigens(M, l0, l1, v0, v1);
        double score = l0 * l0 / (l0 * l0 + l1 * l1);
        double ave_bias_a = out_data_pool[idx * kDataChannels + 4] / out_data_pool[idx * kDataChannels + 6];
        double ave_bias_b = out_data_pool[idx * kDataChannels + 5] / out_data_pool[idx * kDataChannels + 6];
        double ave_bias = std::hypot(ave_bias_a, ave_bias_b);
        if (ave_bias < radius * 100 && score > 0.75) {
          debug_img.data[idx] = std::min(255.0, std::max(0.0, (score - 0.5) * 2.0 * 255.0));
        }
      }
    }

    cv::imwrite("debug.png", debug_img);
    LOG(FATAL) << "Crash here.";
  }
  else {
    LOG(FATAL) << "Haven't finished it.";
  }
}

double* CurveExtractor::ConvertImg2ProbMap(const cv::Mat &img) {
  int height = img.rows;
  int width = img.cols;
  int channels = img.channels();
  double ma = 1e-8;
  auto p_map = new double[height * width];
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int idx = i * width + j;
      double val = 0.0;
      for (int t = 0; t < channels; t++) {
        val += std::abs((double) img.data[idx * channels + t] - 0.0);
      }
      p_map[idx] = val;
      ma = std::max(ma, val);
    }
  }
  for (int i = 0; i < height * width; i++) {
    p_map[i] /= ma;
  }
  return p_map;
}

void CurveExtractor::RunPca(int trunc_r, double r) {
  candi_points_.clear();
  points_.clear();
  tangs_.clear();
  double *p_ptr = prob_map_;
  // The Mat img is for debug.
  cv::Mat img(height_, width_, CV_8UC1);
  std::memset(img.data, 0xFF, height_ * width_);

  for (int x = trunc_r; x + trunc_r < height_; x++) {
    for (int y = trunc_r; y + trunc_r < width_; y++) {
      // TODO: Hard code here.
      if (p_ptr[x * width_ + y] < 0.5) {
        continue;
      }
      Eigen::Matrix2d M;
      M << 1e-8, 1e-8, 1e-8, 1e-8;
      for (int a = x - trunc_r; a <= x + trunc_r; a++) {
        for (int b = y - trunc_r; b <= y + trunc_r; b++) {
          double d = std::hypot((double) a - x, (double) b - y);
          double p = p_ptr[a * width_ + b] * std::exp(-(0.5 * d * d / (r * r)));
          // double p = p_ptr[a * width_ + b];
          Eigen::Vector2d tmp(p * (double) (a - x), p * (double) (b - y));
          M += tmp * tmp.transpose();
        }
      }
      double l0 = 0, l1 = 0;
      Eigen::Vector2d v0, v1;
      CalcEigens(M, l0, l1, v0, v1);
      double score = l0 * l0 / (l0 * l0 + l1 * l1);
      if (score < 0.1 || std::isnan(score)) {
        continue;
      }
      points_.emplace_back((double) x, (double) y);
      tangs_.emplace_back(v0 / v0.norm());
      tang_scores_.emplace_back(score);
      candi_points_.emplace_back(Eigen::Vector2d(x, y), v0 / v0.norm(), score);

    }
  }

  n_points_ = points_.size();
  CHECK_EQ(n_points_, tangs_.size());
  CHECK_EQ(n_points_, candi_points_.size());

  // assign index.
  for (int i = 0; i < n_points_; i++) {
    candi_points_[i].idx = i;
  }
}

void CurveExtractor::RunPcaGPU(int trunc_r, double r) {
  auto out_data_pool = std::make_unique<double[]>(width_ * height_ * 4);
  // CudaPca(height_, width_, prob_map_, out_data_pool.get());
  for (int x = 0; x < height_; x++) {
    for (int y = 0; y < width_; y++) {
      int idx = x * width_ + y;
      if (prob_map_[idx] < 0.5) {
        continue;
      }
      Eigen::Matrix2d M;
      M << out_data_pool[idx * 4], out_data_pool[idx * 4 + 1], out_data_pool[idx * 4 + 2], out_data_pool[idx * 4 + 3];
      double l0 = 0, l1 = 0;
      Eigen::Vector2d v0, v1;
      CalcEigens(M, l0, l1, v0, v1);
      double score = l0 * l0 / (l0 * l0 + l1 * l1);
      if (score < 0.1 || std::isnan(score)) {
        continue;
      }
      points_.emplace_back((double) x, (double) y);
      tangs_.emplace_back(v0 / v0.norm());
      tang_scores_.emplace_back(score);
      candi_points_.emplace_back(Eigen::Vector2d(x, y), v0 / v0.norm(), score);
    }
  }

  n_points_ = points_.size();
  CHECK_EQ(n_points_, tangs_.size());
  CHECK_EQ(n_points_, candi_points_.size());

  for (int i = 0; i < n_points_; i++) {
    candi_points_[i].idx = i;
  }
}

// The method is based on the paper: https://cgl.ethz.ch/Downloads/Publications/Papers/2013/Nor13/Nor13.pdf
// Gradient, reference: https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html
void CurveExtractor::RunGradientBasedExtraction(StreamerBase* streamer) {
  CHECK_EQ(streamer->Channels(), 3);
  CHECK_GT(height_, 0);
  CHECK_GT(width_, 0);
  // Why image -> streamer -> image ???? What a stupid design......
  uchar* streamer_data = streamer->CurrentFrame();
  cv::Mat img(height_, width_, CV_8UC3);
  std::memcpy(img.data, streamer_data, height_ * width_ * 3);
  // Clear noise
  GaussianBlur(img, img, cv::Size(3, 3), 0, 0);
  cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
  cv::Mat grad_x, grad_y;
  cv::Sobel(img, grad_x, CV_16S, 1, 0, 1, 1, 0);
  cv::Sobel(img, grad_y, CV_16S, 0, 1, 1, 1, 0);
  // for debugging.
  if (false) {
    cv::Mat abs_grad_x;
    cv::convertScaleAbs(grad_x, abs_grad_x);
    cv::imwrite("debug_x.png", abs_grad_x);
    cv::Mat abs_grad_y;
    cv::convertScaleAbs(grad_y, abs_grad_y);
    cv::imwrite("debug_y.png", abs_grad_y);
  }

  // Get points with observable gradient. (Has noise).
  double max_grad = 0.0;
  for (int a = 0; a < height_; a++) {
    for (int b = 0; b < width_; b++) {
      double grad_a = grad_y.at<short>(a, b);
      double grad_b = grad_x.at<short>(a, b);
      Eigen::Vector2d current_grad(grad_a, grad_b);
      max_grad = std::max(max_grad, current_grad.norm());
    }
  }

  const double kGradRatioThreshold = 0.2;
  cv::Mat debug_img(height_, width_, CV_8UC3);
  std::memset(debug_img.data, 255, height_ * width_ * 3);
  std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> grad_points;
  for (int a = 0; a < height_; a++) {
    for (int b = 0; b < width_; b++) {
      double grad_a = grad_y.at<short>(a, b);
      double grad_b = grad_x.at<short>(a, b);
      Eigen::Vector2d current_grad(grad_a, grad_b);
      if (current_grad.norm() > kGradRatioThreshold * max_grad) {
        grad_points.emplace_back(Eigen::Vector2d(a, b), current_grad);
        cv::circle(debug_img, cv::Point(b, a), 0, cv::Scalar(255, 0, 255), -1);
      }
    }
  }

  const double kMovingStepLength = 0.5;
  int tmp_ticker = 0;
  while (++tmp_ticker < 500) {
    LOG(INFO) << "tmp_ticker: " << tmp_ticker;
    std::memset(debug_img.data, 255, height_ * width_ * 3);
    std::unordered_map<int, std::vector<int>> grids;
    auto Hash = [](const Eigen::Vector2d& pt) {
      return (int) std::round(pt(0)) * 233333 + std::round(pt(1));
    };
    for (int i = 0; i < grad_points.size(); i++) {
      grids[Hash(grad_points[i].first)].emplace_back(i);
    }
    std::vector<int> can_move(grad_points.size(), 1);
    for (int i = 0; i < grad_points.size(); i++) {
      const auto& pt = grad_points[i].first;
      const auto& grad = grad_points[i].second;
      for (int bias_a = -1; bias_a <= 1; bias_a++) {
        for (int bias_b = -1; bias_b <= 1; bias_b++) {
          for (int j : grids[Hash(pt + Eigen::Vector2d(bias_a, bias_b))]) {
            const auto& ano_pt = grad_points[j].first;
            const auto& ano_grad = grad_points[j].second;
            if ((pt - ano_pt).norm() < 1.0 && grad.dot(ano_grad) < -5e-1) {
              can_move[i] = 0;
            }
          }
        }
      }
    }
    for (int i = 0; i < grad_points.size(); i++) {
      if (!can_move[i]) {
        continue;
      }
      auto& pr = grad_points[i];
      pr.first -= pr.second.normalized() * kMovingStepLength;
    }
    for (const auto& pr : grad_points) {
      cv::circle(debug_img, cv::Point(pr.first(1), pr.first(0)), 0, cv::Scalar(255, 0, 255), -1);
    }
    cv::imshow("debug", debug_img);
    cv::waitKey(10);
  }
  cv::imwrite("clustered.png", debug_img);
  std::exit(0);

}

void CurveExtractor::FindNearestPointsOffline(const std::vector<Eigen::Vector2d> &query_points,
                                              std::vector<ImageDirPoint> *nearest_points,
                                              double searching_r) {
  nearest_points->resize(query_points.size());

  std::vector<unsigned> p(query_points.size());
  std::iota(p.begin(), p.end(), 0);
  auto cmp_p = [&query_points](unsigned a, unsigned b) {
    return query_points[a](0) < query_points[b](0);
  };
  std::sort(p.begin(), p.end(), cmp_p);

  std::vector<unsigned> q(candi_points_.size());
  std::iota(q.begin(), q.end(), 0);
  auto cmp_q = [this](unsigned a, unsigned b) {
    return candi_points_[a].o_(0) < candi_points_[b].o_(0);
  };
  std::sort(q.begin(), q.end(), cmp_q);


  auto ptr_l = q.begin();
  auto ptr_r = q.begin();
  for (auto p_idx : p) {
    const auto &query_point = query_points[p_idx];
    for (; ptr_l != q.end() && candi_points_[*ptr_l].o_(0) + searching_r < query_point(0); ptr_l++);
    for (; ptr_r != q.end() && candi_points_[*ptr_r].o_(0) <= query_point(0) + searching_r; ptr_r++);
    double dis = std::numeric_limits<double>::infinity();
    ImageDirPoint current_dir_point;
    for (auto iter = ptr_l; iter != ptr_r; iter++) {
      if ((candi_points_[*iter].o_ - query_point).norm() < dis) {
        current_dir_point = candi_points_[*iter];
        dis = (candi_points_[*iter].o_ - query_point).norm();
      }
    }
    (*nearest_points)[p_idx] = current_dir_point;
  }
}

void CurveExtractor::SmoothPoints() {
  bool use_average_smoothing = true;
  if (use_average_smoothing) {
    std::vector<Eigen::Vector2d> new_points = points_;
    CHECK(!paths_.empty());
    for (const auto &path : paths_) {
      if (path.size() < 5) {
        continue;
      }
      for (int i = 1; i + 1 < path.size(); i++) {
        new_points[path[i]] = 0.25 * points_[path[i - 1]] + 0.5 * points_[path[i]] + 0.25 * points_[path[i + 1]];
      }
    }
    points_ = new_points;
    for (int i = 0; i < n_points_; i++) {
      candi_points_[i].o_ = points_[i];
    }
  }
  else {
    std::vector<int> p;
    for (int i = 0; i < n_points_; i++) {
      p.push_back(i);
    }
    auto cmp = [this](int a, int b) {
      return points_[a](0) < points_[b](0);
    };
    std::sort(p.begin(), p.end(), cmp);
    int base = 0;
    // TODO: Hard code here.
    const double r = 4.0;
    std::vector<uint32_t> coord_r;
    std::vector<uint32_t> coord_c;
    std::vector<double> values;
    std::vector<double> B;
    auto AddValue = [&coord_r, &coord_c, &values](int r, int c, double val) {
      coord_r.push_back(r);
      coord_c.push_back(c);
      values.push_back(val);
    };
    int idx = -1;
    for (int i = 0; i < n_points_; i++) {
      int u = p[i];
      for (; points_[u](0) - points_[p[base]](0) > r; base++);
      std::pair<double, int> neighbor_0(-1e9, -1);
      std::pair<double, int> neighbor_1(1e9, -1);
      for (int j = base; j < n_points_ && points_[p[j]](0) - points_[u](0) < r; j++) {
        int v = p[j];
        if ((points_[v] - points_[u]).norm() > r) {
          continue;
        }
        Eigen::Vector2d bias = points_[v] - points_[u];
        double tang_dis = (std::abs(bias.dot(Eigen::Vector2d(tangs_[u](1), -tangs_[u](0)))) +
                           std::abs(bias.dot(Eigen::Vector2d(tangs_[v](1), -tangs_[v](0))))) * 0.5;
        double project_dis = bias.dot(tangs_[u]);
        if (tang_dis > 2.0) {
          continue;
        }
        if (project_dis < 0.0 && project_dis > neighbor_0.first) {
          neighbor_0 = std::make_pair(project_dis, v);
        }
        if (project_dis > 0.0 && project_dis < neighbor_1.first) {
          neighbor_1 = std::make_pair(project_dis, v);
        }
      }
      {
        const double solid_weight = 1.0 / tang_scores_[u];
        idx++;
        AddValue(idx, u * 2 + 0, solid_weight);
        B.push_back(points_[u](0) * solid_weight);
        idx++;
        AddValue(idx, u * 2 + 1, solid_weight);
        B.push_back(points_[u](1) * solid_weight);
      }
      {
        const double smooth_weight = 5.0;
        int a = neighbor_0.second;
        int b = neighbor_1.second;
        if (a == -1 || b == -1) {
          continue;
        }
        for (int t = 0; t < 2; t++) {
          idx++;
          AddValue(idx, u * 2 + t, smooth_weight * 2.0);
          AddValue(idx, a * 2 + t, -smooth_weight);
          AddValue(idx, b * 2 + t, -smooth_weight);
          B.push_back(0.0);
        }
      }
    }
    std::vector<double> solution(points_.size() * 2, 0.0);
    Utils::SolveLinearSqrLSQR((int) B.size(), n_points_ * 2, coord_r, coord_c, values, B, solution);
    for (int i = 0; i < n_points_; i++) {
      points_[i] = Eigen::Vector2d(solution[i * 2], solution[i * 2 + 1]);
    }
  }
}

void CurveExtractor::CalcLinkInformation() {
  // Offline searching.
  const double kSearchingR = 2.1;
  neighbors_.resize(n_points_);
  auto cmp_by_x = [this] (int a, int b) {
    return candi_points_[a].o_(0) < candi_points_[b].o_(0);
  };

  // LOG(INFO) << "n_points: " << n_points_;
  std::vector<int> indexes(n_points_);
  std::iota(indexes.begin(), indexes.end(), 0);
  std::sort(indexes.begin(), indexes.end(), cmp_by_x);

  auto iter_l = indexes.begin();
  auto iter_r = indexes.begin();

  for (auto iter = indexes.begin(); iter != indexes.end(); iter++) {
    for (; iter_l != indexes.end() &&
        candi_points_[*iter_l].o_(0) + kSearchingR < candi_points_[*iter].o_(0); iter_l++);
    for (; iter_r != indexes.end() &&
        candi_points_[*iter_r].o_(0) < candi_points_[*iter].o_(0) + kSearchingR; iter_r++);
    std::vector<std::tuple<int, int, double>> tmp_link_info;

    for (auto ano_iter = iter_l; ano_iter != iter_r; ano_iter++) {
      if (*iter == *ano_iter) {
        continue;
      }
      const auto& dir_point_0 = candi_points_[*iter];
      const auto& dir_point_1 = candi_points_[*ano_iter];
      Eigen::Vector2d bias = dir_point_1.o_ - dir_point_0.o_;
      if (bias.norm() > kSearchingR) {
        continue;
      }
      double dis_sqr = bias.squaredNorm();
      double weight = std::exp(-dis_sqr / (kSearchingR * kSearchingR) * 0.5);
      if (weight < 1e-9) {
        continue;
      }
      tmp_link_info.emplace_back(*iter, *ano_iter, weight);
      neighbors_[*iter].emplace_back(*ano_iter);
      neighbors_[*ano_iter].emplace_back(*iter);
    }

    auto cmp_by_score = [](const std::tuple<int, int, double>& a, const std::tuple<int, int, double>& b) {
      return std::get<2>(a) > std::get<2>(b);
    };

    std::sort(tmp_link_info.begin(), tmp_link_info.end(), cmp_by_score);
    const int kMaxNeighbors = 5;
    for (int i = 0; i < std::min(kMaxNeighbors, (int) tmp_link_info.size()); i++) {
      link_info_.emplace_back(tmp_link_info[i]);
    }
  }
}

void CurveExtractor::CalcPaths() {
  paths_.clear();
  std::vector<int> fa(n_points_, 0);
  std::iota(fa.begin(), fa.end(), 0);
  std::function<int(int)> FindRoot;
  FindRoot = [&fa, &FindRoot](int a) -> int {
    return fa[a] == a ? a : (fa[a] = FindRoot(fa[a]));
  };
  auto& edges = edges_;
  edges.clear();
  edges.resize(n_points_);
  std::vector<int> link_info_p(link_info_.size());
  std::iota(link_info_p.begin(), link_info_p.end(), 0);
  std::sort(link_info_p.begin(),
            link_info_p.end(),
            [this](int a, int b) { return std::get<2>(link_info_[a]) > std::get<2>(link_info_[b]); });
  for (int p : link_info_p) {
    const auto& current_edge = link_info_[p];
    int a = std::get<0>(current_edge);
    int b = std::get<1>(current_edge);
    if (FindRoot(a) == FindRoot(b)) {
      continue;
    }
    fa[fa[a]] = fa[b];
    edges[a].emplace_back(b);
    edges[b].emplace_back(a);
  }

  std::vector<int> f(n_points_, -1);
  for (int rt = 0; rt < n_points_; rt++) {
    if (f[rt] != -1) {
      continue;
    }
    if (edges[rt].empty()) {
      paths_.emplace_back();
      paths_.back().push_back(rt);
      continue;
    }
    if (edges[rt].size() > 1) {
      continue;
    }

    // BFS.
    std::vector<int> que = { rt };
    int L = 0;
    for (; L < que.size(); L++) {
      int u = que[L];
      for (const int v : edges[u]) {
        if (f[u] == v) {
          continue;
        }
        f[v] = u;
        que.emplace_back(v);
      }
    }
    std::reverse(que.begin(), que.end());
    for (int u : que) {
      if (f[u] == -1 || edges[u].size() == 2) {
        continue;
      }
      CHECK(edges[u].size() != 2);
      paths_.emplace_back();
      auto& current_path = paths_.back();
      int v = u;
      for (; v == u || (v != -1 && f[v] != -1 && edges[v].size() <= 2); v = f[v]) {
        current_path.emplace_back(v);
      }
      if (v >= 0) {
        current_path.emplace_back(v);
      }
    }
  }
  
  near_junction_ = std::vector<int>(n_points_, 0);
  int kNearingThreshold = 5;
  for (const auto& path : paths_) {
    for (int i = 0; i < kNearingThreshold && i < path.size(); i++) {
      near_junction_[path[i]] = 1;
      near_junction_[path[path.size() - 1 - i]] = 1;
    }
  }

  // Re-calculate neighbors: Only consider neighbors on curves.
  neighbors_.clear();
  neighbors_.resize(n_points_);
  for (const auto& path : paths_) {
    for (int k = 1; k <= 2; k++) {
      for (int a = 0; a + k < path.size(); a++) {
        int b = a + k;
        neighbors_[path[b]].emplace_back(path[a]);
        neighbors_[path[a]].emplace_back(path[b]);
      }
    }
  }
}

std::pair<Eigen::Vector2d, double> CurveExtractor::CalcTangAndScore(const Eigen::Vector2d &pix) {
  Eigen::Matrix2d M;
  M << 1e-8, 1e-8, 1e-8, 1e-8;
  int x = std::round(pix(0));
  int y = std::round(pix(1));
  const int trunc_r = 3;
  const double r = 4.0;
  double *p_ptr = prob_map_;
  for (int a = x - trunc_r; a <= x + trunc_r; a++) {
    for (int b = y - trunc_r; b <= y + trunc_r; b++) {
      if (a < 0 || a >= height_ || b < 0 || b >= width_) {
        continue;
      }
      double d = std::hypot((double) a - x, (double) b - y);
      double p = p_ptr[a * width_ + b] * std::exp(-(0.5 * d * d / (r * r)));
      // double p = p_ptr[a * width_ + b];
      Eigen::Vector2d tmp(p * (double) (a - x), p * (double) (b - y));
      M += tmp * tmp.transpose();
    }
  }
  double l0 = 0, l1 = 0;
  Eigen::Vector2d v0, v1;
  CalcEigens(M, l0, l1, v0, v1);
  double score = l0 * l0 / (l0 * l0 + l1 * l1);
  return { v0 / v0.norm(), score };
}

void CurveExtractor::CalcEigens(const Eigen::Matrix2d &C, double &l1, double &l2,
           Eigen::Vector2d &v1, Eigen::Vector2d &v2) {
  // reference: http://www.math.harvard.edu/archive/21b_fall_04/exhibits/2dmatrices/index.html
  double t = C(0, 0) + C(1, 1);
  double d = C.determinant();
  l1 = t * 0.5 + std::sqrt(t * t * 0.25 - d);
  l2 = t * 0.5 - std::sqrt(t * t * 0.25 - d);
  if (std::abs(C(1, 0)) > 1e-10) {
    v1 = Eigen::Vector2d(l1 - C(1, 1), C(1, 0));
    v2 = Eigen::Vector2d(l2 - C(1, 1), C(1, 1));
  }
  else if (std::abs(C(0, 1)) > 1e-10) {
    v1 = Eigen::Vector2d(C(0, 1), l1 - C(0, 0));
    v2 = Eigen::Vector2d(C(0, 1), l2 - C(0, 0));
  }
  else {
    v1 = Eigen::Vector2d(0.0, 1.0);
    v2 = Eigen::Vector2d(1.0, 0.0);
  }
}

void CurveExtractor::GetEdgePoints(std::vector<Eigen::Vector2d>* points) {
  points->clear();
  for (int a = 0; a < height_; a++) {
    for (int b = 0; b < width_; b++) {
      if (seg_map_[a * width_ + b] < 1) {
        continue;
      }
      bool is_edge = false;
      for (int x = std::max(0, a - 1); x <= std::min(height_ - 1, a + 1) && !is_edge; x++) {
        for (int y = std::max(0, b - 1); y <= std::min(width_ - 1, b + 1) && !is_edge; y++) {
          if (std::abs(a - x) + std::abs(y - b) > 1) {
            continue;
          }
          if (seg_map_[x * width_ + y] < 1) {
            is_edge = true;
            break;
          }
        }
      }
      if (is_edge) {
        points->emplace_back(a, b);
      }
    }
  }
}

cv::Mat CurveExtractor::GetSegImageWithNoise(double shaking_len,
                                             double shaking_ang,
                                             const std::vector<Eigen::Vector2d>& polluted_positions,
                                             double polluted_radius) {
  cv::Mat seg_img(height_, width_, CV_8UC3);
  std::memset(seg_img.data, 255, height_ * width_ * 3);
  for (int a = 0; a < height_; a++) {
    for (int b = 0; b < width_; b++) {
      if (seg_map_[a * width_ + b] > 0) {
        for (int k = 0; k < 3; k++) {
          seg_img.data[(a * width_ + b) * 3 + k] = 0;
        }
      }
    }
  }

  // Get shake noise.
  Eigen::Vector2d shaking_dir = Eigen::Vector2d(std::cos(shaking_ang), std::sin(shaking_ang));
  std::vector<Eigen::Vector2d> edge_points;
  GetEdgePoints(&edge_points);
  LOG(INFO) << "edge_points size: " << edge_points.size();
  LOG(INFO) << "shaking len: " << shaking_len;
  for (const auto& pt : edge_points) {
    double new_shaking_len = std::rand() % 2 == 0 ? shaking_len : shaking_len + 1.0;
    for (double t = 0.0; t < new_shaking_len; t += 0.3) {
      for (int k = -1; k <= 1; k += 2) {
        Eigen::Vector2d new_pt = pt + shaking_dir * t * k;
        int a = std::round(new_pt(0));
        int b = std::round(new_pt(1));
        if (a >= 0 && a < height_ && b >= 0 && b < width_) {
          for (int k = 0; k < 3; k++) {
            seg_img.data[(a * width_ + b) * 3 + k] = 0;
          }
        }
      }
    }
  }

  // Get polluted_noise.
  if (std::abs(polluted_radius) > 1e-3) {
    int pollution_color = polluted_radius > 0.0 ? 0 : 255;
    for (const auto &pt : polluted_positions) {
      int n_slices = 12;
      double ang_step = kDoublePi / n_slices;
      double initial_angle = std::rand() / double(RAND_MAX) * kDoublePi;
      double past_r = (std::rand() / double(RAND_MAX) * 0.7 + 0.3) * polluted_radius;
      double origin_r = past_r;
      std::vector<cv::Point> draw_points;
      for (int i = 0; i < n_slices; i++) {
        double new_r = i + 1 == n_slices ? origin_r : (std::rand() / double(RAND_MAX) * 0.7 + 0.3) * polluted_radius;
        double L = ang_step * i;
        double R = ang_step * (i + 1);
        for (double t = L; t < R; t += kDoublePi / 360.0) {
          double inter = (std::sin(((t - L) / (R - L) - 0.5) * kOnePi) + 1.0) * 0.5;
          double inter_r = new_r * inter + past_r * (1.0 - inter);
          draw_points.emplace_back(std::sin(t) * inter_r + pt(1), std::cos(t) * inter_r + pt(0));
        }
        past_r = new_r;
      }
      const cv::Point *ppt[1] = {draw_points.data()};
      int npt[] = {int(draw_points.size())};
      cv::fillPoly(seg_img, ppt, npt, 1, cv::Scalar(pollution_color, pollution_color, pollution_color));
    }
  }
  return seg_img;
}

cv::Mat CurveExtractor::GetGaussianNoiseSegmentation(double std_dev) {
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<> normal_d(0.0, std_dev);
  std::uniform_real_distribution<> uniform_alpha(0.0, kDoublePi);
  cv::Mat seg_img(height_, width_, CV_8UC3);
  std::memset(seg_img.data, 255, height_ * width_ * 3);
  for (int a = 0; a < height_; a++) {
    for (int b = 0; b < width_; b++) {
      if (seg_map_[a * width_ + b] > 0) {
        for (int k = 0; k < 3; k++) {
          seg_img.data[(a * width_ + b) * 3 + k] = 0;
        }
      }
    }
  }

  std::vector<Eigen::Vector2d> edge_points;
  GetEdgePoints(&edge_points);
  LOG(INFO) << "edge_points size: " << edge_points.size();

  for (const auto& pt : edge_points) {
    double new_shaking_len = normal_d(gen);
    int color = 0;
    if (new_shaking_len < 0) {
      color = 255;
      new_shaking_len = -new_shaking_len;
      if (new_shaking_len < 0.5) {
        continue;
      }
    }
    int a = std::round(pt(0));
    int b = std::round(pt(1));
    cv::circle(seg_img, cv::Point(b, a), new_shaking_len, cv::Scalar(color, color, color), -1);
  }
  return seg_img;
}

/*
void CurveExtractor::GenPollutingNoise(const std::vector<E>) {
}
*/

double CurveExtractor::ProbAt(double a, double b) {
  a = std::min(std::max(a, 0.0), height_ - 1.0);
  b = std::min(std::max(b, 0.0), width_ - 1.0);
  int a_low = std::floor(a + 1e-9);
  int a_up = std::min(a_low + 1, height_ - 1);
  int b_low = std::floor(b + 1e-9);
  int b_up = std::min(b_low + 1, width_ - 1);
  double a_bias = a - a_low;
  double b_bias = b - b_low;

  return prob_map_[a_low * width_ + b_low] * (1.0 - a_bias) * (1.0 - b_bias) +
         prob_map_[a_up * width_ + b_low] * a_bias * (1.0 - b_bias) +
         prob_map_[a_low * width_ + b_up] * (1.0 - a_bias) * b_bias +
         prob_map_[a_up * width_ + b_up] * a_bias * b_bias;
}
