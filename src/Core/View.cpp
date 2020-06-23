//
// Created by aska on 2019/4/21.
//
// This is new.

#include "View.h"
#include "ViewOptimizer.h"
#include "../Utils/Common.h"
#include "../Utils/KMeansClustering.h"
#include "../Utils/QuadTree.h"

#include <chrono>
#include <map>
#include <thread>

// Hard parameters.
const int kSearchingR = 10;  // Unit: pixel.

View::View(CurveExtractor *extractor, int time_stamp, double focal_length) :
    extractor_(extractor),
    time_stamp_(time_stamp),
    focal_length_(focal_length),
    M_(new double[extractor->points_.size() * 3]),
    camera_(focal_length, extractor->width_, extractor->height_) {
  n_points_ = (int) extractor_->points_.size();
  covered_.resize(n_points_);
  for (int i = 0; i < n_points_; i++) {
    M_[i * 3] =     (extractor_->points_[i](1) - extractor_->width_  * 0.5) / focal_length_;
    M_[i * 3 + 1] = (extractor_->points_[i](0) - extractor_->height_ * 0.5) / focal_length_;
    M_[i * 3 + 2] = 1.0;
  }
  R_ = Eigen::Matrix3d::Identity();
  T_ = Eigen::Vector3d::Zero();

  // Build ICP data structure.
  std::vector<Eigen::Vector2d> icp_points(n_points_);
  std::vector<Eigen::Vector3d> icp_normals(n_points_);
  std::vector<double> icp_weights(n_points_, 1.0);
  for (int i = 0; i < n_points_; i++) {
    icp_points[i](0) = (extractor_->points_[i](1) - extractor_->width_  * 0.5) / focal_length_;
    icp_points[i](1) = (extractor_->points_[i](0) - extractor_->height_ * 0.5) / focal_length_;

    Eigen::Vector3d cross_a(icp_points[i](0), icp_points[i](1), 1.0);
    Eigen::Vector3d cross_b(extractor_->tangs_[i](1), extractor_->tangs_[i](0), 0.0);
    Eigen::Vector3d crossed = cross_a.cross(cross_b);
    CHECK(std::abs(crossed.dot(cross_a)) < 1e-3);
    CHECK(std::abs(crossed.dot(cross_b)) < 1e-3);
    icp_normals[i] = crossed / crossed.norm();
  }
  icp_ = std::make_unique<ICP>(icp_points, icp_normals, icp_weights);

  // Build Quadtree.
  quad_tree_ = std::make_unique<QuadTree>(extractor_->points_);

  // Build SegmentQuadtree.
  std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> segments;
  std::vector<Eigen::Vector2d> tangs;
  for (const auto& path : extractor_->paths_) {
    CHECK(!path.empty());
    for (auto iter = path.begin(); std::next(iter) != path.end(); iter++) {
      segments.emplace_back(extractor_->points_[*iter], extractor_->points_[*std::next(iter)]);
      tangs.emplace_back(extractor_->tangs_[*iter]);
    }
  }
  segment_quad_tree_ = std::make_unique<SegmentQuadTree>(segments, tangs);

  BuildGridIndex();

  is_self_occluded_ = std::vector<int>(n_points_, 0);

  // View center
  std::vector<double> x(n_points_);
  std::vector<double> y(n_points_);
  for (int i = 0; i < n_points_; i++) {
    x[i] = extractor_->points_[i](0);
    y[i] = extractor_->points_[i](1);
  }
  int L = int(0.15 * n_points_);
  int R = int(0.85 * n_points_);
  std::sort(x.begin(), x.end());
  std::sort(y.begin(), y.end());
  view_center_(0) = (x[L] + x[R]) * 0.5;
  view_center_(1) = (y[L] + y[R]) * 0.5;
}

void View::ClearSelfOcclusionState() {
  is_self_occluded_.resize(extractor_->n_points_);
  std::fill(is_self_occluded_.begin(), is_self_occluded_.end(), 0);
}

void View::BuildGridIndex() {
  grids_.clear();
  grid_h_ = std::ceil(extractor_->height_ / kSearchingR) + 1;
  grid_w_ = std::ceil(extractor_->width_ / kSearchingR) + 1;

  for (int i = 0; i < extractor_->n_points_; i++) {
    Eigen::Vector2d& pt = extractor_->candi_points_[i].o_;
    unsigned a = pt(0) / kSearchingR + 1e-9;
    unsigned b = pt(1) / kSearchingR + 1e-9;
    if (grids_.find(a * grid_w_ + b) == grids_.end()) {
      grids_.emplace(a * grid_w_ + b, std::vector<int>());
    }
    grids_[a * grid_w_ + b].push_back(i);
  }
}

void View::FindPointsInFOV(const std::vector<Eigen::Vector3d>& points,
                           double ratio,
                           std::vector<int>* indexes) {
  indexes->clear();
  int height = extractor_->height_;
  int width = extractor_->width_;
  for (int i = 0; i < points.size(); i++) {
    Eigen::Vector2d pt = camera_.World2Image(points[i]);
    double current_ratio = std::max(std::abs(pt(0) / height - 0.5), std::abs(pt(1) / width - 0.5)) * 2.0;
    if (current_ratio < ratio) {
      indexes->emplace_back(i);
    }
  }
}

void View::UpdateWorldPoints(std::vector<Eigen::Vector3d> *points) {
  world_points_ = points;
  UpdateMatching();
}

void View::UpdateMatching() {
  return;
}

int View::Track(const std::vector<std::pair<Eigen::Matrix3d, Eigen::Vector3d>>& pose_candidates,
                const std::vector<Eigen::Vector3d>& points,
                const std::vector<int>& points_history,
                const std::vector<Eigen::Vector3d>& tangs,
                const std::vector<double>& tang_scores,
                IronTown* curve_network,
                SpanningTree* spanning_tree,
                OctreeNew* octree,
                Eigen::Vector2d initial_flow,
                TrackType track_type) {
  std::vector<int> tracking_indexes;
  const double kExpectedTrackingPoints = 500.0;
  double sampling_ratio = std::max(1.0, points.size() / kExpectedTrackingPoints);
  std::vector<std::vector<int>> paths;
  curve_network->GetPaths(&paths);
  for (const auto& path : paths) {
    if (path.size() > sampling_ratio) {
      for (double k = 1 + 1e-8; k < path.size(); k += sampling_ratio) {
        tracking_indexes.emplace_back(path[(int) k]);
      }
    }
    else {
      tracking_indexes.emplace_back(path[path.size() / 2]);
    }
  }

  int region_len = std::min(extractor_->width_, extractor_->height_) / 16;
  int w = (extractor_->width_  + region_len - 1) / region_len;
  int h = (extractor_->height_ + region_len - 1) / region_len;

  std::random_shuffle(tracking_indexes.begin(), tracking_indexes.end());
  for (const auto& pose_candidate : pose_candidates) {
    int how_good_tracking_is = 0;
    const int kMaxIterNum = 10;
    const int kMaxPointNeededPerRegion = 10;
    const int kRandomTryingTime = 10;
    for (int try_t = 0; try_t < kRandomTryingTime && how_good_tracking_is <= 0; try_t++) {
      this->UpdateRT(pose_candidate.first, pose_candidate.second);
      int iter_num = 0;
      do {
        std::vector<int> current_tracking_indexes;
        std::vector<int> current_matching_indexes;
        std::vector<int> region_counter(w * h, -1);
        bool use_nearest_matching = ((iter_num > 0) || track_type == TrackType::NAIVE);
        // bool use_nearest_matching = true;
        if (use_nearest_matching) {
          std::vector<Eigen::Vector2d> projected_points(tracking_indexes.size());
          for (int u = 0; u < tracking_indexes.size(); u++) {
            projected_points[u] = camera_.World2Image(points[tracking_indexes[u]]);
            const auto& pt = projected_points[u];
            if (pt(0) < 0.1 * extractor_->height_ || pt(0) > 0.9 * extractor_->height_ ||
                pt(1) < 0.1 * extractor_->width_ || pt(1) > 0.9 * extractor_->width_) {
              continue;
            }
            int region_idx = int(pt(0) / region_len) * w + int(pt(1) / region_len);
            region_counter[region_idx] = 0;
          }

          std::vector<int> hitted_regions;
          for (int region_i = 0; region_i < h * w; region_i++) {
            if (region_counter[region_i] >= 0) {
              hitted_regions.emplace_back(region_i);
            }
          }
          std::random_shuffle(hitted_regions.begin(), hitted_regions.end());
          const double kDropoutRatio = 0.4;
          int n_region_to_remove = hitted_regions.size() * kDropoutRatio;
          for (int region_i = 0; region_i < n_region_to_remove; region_i++) {
            region_counter[hitted_regions[region_i]] = -1;
          }
          for (int u = 0; u < tracking_indexes.size(); u++) {
            int tracking_idx = tracking_indexes[u];
            const Eigen::Vector2d& pt = projected_points[u];
            if (pt(0) < 0.1 * extractor_->height_ || pt(0) > 0.9 * extractor_->height_ ||
                pt(1) < 0.1 * extractor_->width_ || pt(1) > 0.9 * extractor_->width_) {
              continue;
            }
            int region_idx = int(pt(0) / region_len) * w + int(pt(1) / region_len);
            if (region_counter[region_idx] < 0 || region_counter[region_idx] >= kMaxPointNeededPerRegion) {
              continue;
            }
            region_counter[region_idx]++;
            int v = quad_tree_->NearestIdx(pt);
            current_tracking_indexes.emplace_back(tracking_idx);
            current_matching_indexes.emplace_back(v);
          }
        } else {
          this->CacheMatchings(
              points,
              tangs,
              tang_scores,
              spanning_tree->Edges(),
              2.0 / focal_length_,
              (track_type == TrackType::MATCHING_OF),
              initial_flow);
          current_tracking_indexes.clear();
          current_matching_indexes.clear();
          for (int tracking_idx : tracking_indexes) {
            if (matching_indexes_[tracking_idx] < 0) {
              continue;
            }
            current_tracking_indexes.emplace_back(tracking_idx);
            current_matching_indexes.emplace_back(matching_indexes_[tracking_idx]);
          }
        }
        how_good_tracking_is = this->TrackSingleStep(points, current_tracking_indexes, current_matching_indexes);
        // LOG(INFO) << "how good tracking is ? " << how_good_tracking_is << " overall points number: "
        //           << current_matching_indexes.size();
      } while (how_good_tracking_is < 2 && ++iter_num < kMaxIterNum);
    }
    if (how_good_tracking_is > 0) {
      return 0;
    }
  }
  return -1;
}

int View::TrackSingleStep(const std::vector<Eigen::Vector3d>& points,
                          const std::vector<int>& tracking_indexes,
                          const std::vector<int>& current_matching_indexes) {
  CHECK_EQ(tracking_indexes.size(), current_matching_indexes.size());
  ceres::Problem problem;
  double camera[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
  Eigen::AngleAxisd angle_axis(R_);
  double angle = angle_axis.angle();
  Eigen::Vector3d axis = angle_axis.axis();
  for (int t = 0; t < 3; t++) {
    camera[t] = axis(t) * angle;
    camera[t + 3] = T_(t);
  }
  for (int i = 0; i < tracking_indexes.size(); i++) {
    ceres::CostFunction* cost_function =
        ViewProjectionError::Create(points[tracking_indexes[i]],
                                    extractor_->points_[current_matching_indexes[i]],
                                    extractor_->tangs_[current_matching_indexes[i]],
                                    1.0,
                                    focal_length_,
                                    extractor_->width_,
                                    extractor_->height_);
    problem.AddResidualBlock(cost_function, nullptr, camera);
  }
  // Solve & Optimization.
  ceres::Solver::Options options;
  options.max_num_iterations = 1;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  Eigen::Vector3d R_axis(camera[0], camera[1], camera[2]);
  Eigen::Matrix3d new_R;
  new_R = Eigen::AngleAxisd(R_axis.norm(), R_axis.normalized());
  Eigen::Vector3d new_T(camera[3], camera[4], camera[5]);
  UpdateRT(new_R, new_T);

  int n_bad_points = 0;
  int n_good_points = 0;
  for (int i = 0; i < tracking_indexes.size(); i++) {
    double dis =
        (camera_.World2Image(points[tracking_indexes[i]]) - extractor_->points_[current_matching_indexes[i]]).norm();
    if (dis > 3.0) {
      n_bad_points++;
    }
    if (dis < 1.1) {
      n_good_points++;
    }
  }
  if (n_good_points > tracking_indexes.size() * 0.8) {
    return 2;
  }
  else if (n_bad_points < tracking_indexes.size() * 0.3) {
    return 1;
  }
  else return 0;
}

void View::UpdateRTRobust(const std::vector<Eigen::Vector3d> &points,
                          const std::vector<int>& points_history) {
  // True updating process.
  Eigen::Matrix3d initial_R = R_;
  Eigen::Vector3d initial_T = T_;
  UpdateRT(points, points_history);
  if (!BadTrackingResult(points)) {
    return;
  }
  bool okay = false;
  const double AngleStep = 0.02 * kOnePi;
  Eigen::Matrix3d best_R;
  Eigen::Vector3d best_T;
  double best_score = 0.0;
  for (int x = -1; x <= 1 && !okay; x++) {
    for (int y = -1; y <= 1 && !okay; y++) {
      for (int z = -1; z <= 1 && !okay; z++) {
        Eigen::Matrix3d R_inc;
        R_inc = Eigen::AngleAxisd(AngleStep * x, Eigen::Vector3d::UnitX())
                * Eigen::AngleAxisd(AngleStep * y, Eigen::Vector3d::UnitY())
                * Eigen::AngleAxisd(AngleStep * z, Eigen::Vector3d::UnitZ());
        UpdateRT(R_inc * initial_R, initial_T);
        UpdateRT(points, points_history);
        double score = CalcIOU(points);
        if (score > 0.65) {
          okay = true;
          best_R = R_;
          best_T = T_;
          break;
        }
        else if (score > best_score) {
          best_score = score;
          best_R = R_;
          best_T = T_;
        }
      }
    }
  }
  if (!okay) {
    UpdateRT(best_R, best_T);
  }
}

void View::UpdateRT(const std::vector<Eigen::Vector3d>& points,
                    const std::vector<int>& points_history,
                    bool show_demo,
                    int max_iter_num) {
  CHECK_EQ(points.size(), points_history.size());
  std::vector<double> P;
  P.resize(points.size() * 3, 0.0);
  for (int i = 0; i < points.size(); i++) {
    for (int t = 0; t < 3; t++) {
      P[i * 3 + t] = points[i](t);
    }
  }

  CHECK(!show_demo);
  if (!show_demo) {
    std::vector<Eigen::Vector3d> points_for_tracking;
    std::vector<int> fov_indexes;
    // if (MatchingCached()) {
    if (false) {
      const double kThreshold = 10.0 / focal_length_;
      int max_history = -1;
      for (const int history : points_history) {
        max_history = std::max(max_history, history);
      }
      int history_baseline = std::min(max_history / 2, 5);
      // LOG(INFO) << "------ history baseline: " << history_baseline;
      for (int i = 0; i < n_points_; i++) {
        const auto pt = extractor_->points_[i];
        if (pt(0) < extractor_->height_ * 0.1 || pt(0) > extractor_->height_ * 0.9 ||
            pt(1) < extractor_->width_  * 0.1 || pt(1) > extractor_->width_  * 0.9) {
          continue;
        }
        if (nearest_matched_idx_[i] > -1 && matched_std_dev_[i] < 10.0 / kThreshold) {
          int u = nearest_matched_idx_[i];
          if (points_history[u] >= history_baseline) {
            fov_indexes.emplace_back(u);
          }
        }
      }
      LOG(INFO) << "---" << " n_points: " << n_points_ << " fov indexes size: " << fov_indexes.size();
    }
    else {
      FindPointsInFOV(points, 0.9, &fov_indexes);
    }
    // std::random_shuffle(fov_indexes.begin(), fov_indexes.end());
    // fov_indexes.resize(std::min(std::max(int(fov_indexes.size() * 0.2), 100), int(fov_indexes.size())));
    for (int idx : fov_indexes) {
      points_for_tracking.emplace_back(points[idx]);
    }

    icp_->FitRT(points, &R_, &T_, max_iter_num);
  }

  camera_.UpdateRT(R_, T_);
}


void View::UpdateRTCeres(const std::vector<Eigen::Vector3d>& points,
                         const std::vector<int>& points_history,
                         int max_iter_num) {
  CHECK(MatchingCached());
  CHECK_EQ(points.size(), matching_pixels_.size());
  CHECK_EQ(points.size(), points_history.size());

  ceres::Problem problem;

  const double kThreshold = 5.0 / focal_length_;
  int max_history = -1;
  for (const int history : points_history) {
    max_history = std::max(max_history, history);
  }
  int history_baseline = std::min(max_history / 2, 5);

  double camera[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
  Eigen::AngleAxisd angle_axis(R_);
  double angle = angle_axis.angle();
  Eigen::Vector3d axis = angle_axis.axis();
  for (int t = 0; t < 3; t++) {
    camera[t] = axis(t) * angle;
    camera[t + 3] = T_(t);
  }
  std::vector<std::pair<double, int>> dis_idx;
  for (int i = 0; i < n_points_; i++) {
    if (nearest_matched_idx_[i] > -1 && !is_self_occluded_[i]) {
      int u = nearest_matched_idx_[i];
      if (points_history[u] >= history_baseline) {
        dis_idx.emplace_back((camera_.World2Image(points[u]) - extractor_->points_[i]).norm(), i);
      }
    }
  }
  std::random_shuffle(dis_idx.begin(), dis_idx.end());
  dis_idx.resize(std::max(100, int(dis_idx.size() * 0.3)));
  for (const auto& pr : dis_idx) {
    int i = pr.second;
    int u = nearest_matched_idx_[i];
    ceres::CostFunction* cost_function =
        ViewProjectionError::Create(points[u],
                                    extractor_->points_[i],
                                    extractor_->tangs_[i],
                                    1.0,
                                    focal_length_,
                                    extractor_->width_,
                                    extractor_->height_);
    problem.AddResidualBlock(cost_function, nullptr, camera);
  }
  // Solve & Optimization.
  ceres::Solver::Options options;
  options.max_num_iterations = max_iter_num;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  Eigen::Vector3d R_axis(camera[0], camera[1], camera[2]);
  Eigen::Matrix3d new_R;
  new_R = Eigen::AngleAxisd(R_axis.norm(), R_axis.normalized());
  Eigen::Vector3d new_T(camera[3], camera[4], camera[5]);
  UpdateRT(new_R, new_T);
}

void View::UpdateRT(const Eigen::Matrix3d& R, const Eigen::Vector3d& T) {
  R_ = R;
  T_ = T;
  camera_.UpdateRT(R, T);
}

bool View::BadTrackingResult(const std::vector<Eigen::Vector3d>& points) {
  return false;
  double IOU = CalcIOU(points);
  LOG(INFO) << "current IOU: " << IOU;
  return CalcIOU(points) < 0.5;
}

int View::GetNearestPointIdx(const Eigen::Vector3d& world_point) {
  Eigen::Vector2d pt = camera_.World2Image(world_point);
  return quad_tree_->NearestIdx(pt);
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> View::GetRayAndNormal(const Eigen::Vector3d &point) {
  Eigen::Vector3d n_point = R_ * point + T_;
  Eigen::Vector2d pt(n_point(0) / n_point(2), n_point(1) / n_point(2));
  return icp_->NearestRayNormal(pt);
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> View::GetWorldRayAndNormal(const Eigen::Vector3d &point) {
  // TODO;
  return { { 0, 0, 0 }, {0, 0, 0}};
}

Eigen::Vector2d View::GetTang(const Eigen::Vector3d &point) {
  auto pr = GetRayAndNormal(point);
  Eigen::Vector2d tang(pr.second(1), -pr.second(0));
  return tang / tang.norm();
}

bool View::MatchingCached() {
  return !matching_ray_normals_.empty();
}

void View::ClearMatchingCache() {
  matching_ray_normals_.clear();
  matching_radius_.clear();
}

void View::CacheMatchings(const std::vector<Eigen::Vector3d>& world_points,
                          const std::vector<Eigen::Vector3d>& /*tangs*/,
                          const std::vector<double>& /*tang_scores*/,
                          const std::vector<std::vector<std::pair<int, double>>>& edges,
                          double /*hope_dist*/,
                          bool use_initial_alignment,
                          Eigen::Vector2d initial_flow,
                          bool show_debug_info) {
  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& ray_normals = matching_ray_normals_;
  std::vector<double>& matching_radius = matching_radius_;
  ray_normals.clear();
  matching_radius.clear();
  int n_world_points = world_points.size();
  const auto &points = extractor_->points_;
  StopWatch stop_watch;
  // Get Neighbors;
  std::vector<std::vector<std::pair<int, double>>> neighbors(n_world_points);
  std::vector<Eigen::Vector2d> projected_points;
  std::vector<Eigen::Vector2d> projected_flow_points;

  for (int u = 0; u < n_world_points; u++) {
    projected_points.push_back(camera_.World2Image(world_points[u]));
  }

  {
    // Get aligned points.
    if (use_initial_alignment) {
      int height = extractor_->height_;
      int width = extractor_->width_;
      cv::Mat img_0(height, width, CV_8UC3);
      cv::Mat img_1(height, width, CV_8UC3);
      std::memset(img_0.data, 0, height * width * 3);
      std::memset(img_1.data, 0, height * width * 3);

      for (int u = 0; u < n_world_points; u++) {
        auto pt = R_ * world_points[u] + T_;
        double x = pt(0) / pt(2) * focal_length_ + extractor_->width_ * 0.5;
        double y = pt(1) / pt(2) * focal_length_ + extractor_->height_ * 0.5;
        int a = std::round(y + initial_flow(0));
        int b = std::round(x + initial_flow(1));
        if (a < height && a >= 0 && b < width && b >= 0) {
          for (int t = 0; t < 3; t++) {
            img_0.data[(a * width + b) * 3 + t] = 255;
          }
        }
      }

      for (const auto &pt : points) {
        int a = std::round(pt(0));
        int b = std::round(pt(1));
        if (a < height && a >= 0 && b < width && b >= 0) {
          for (int t = 0; t < 3; t++) {
            img_1.data[(a * width + b) * 3 + t] = 255;
          }
        }
      }

      cv::Mat flow_img_;
      Utils::GetOpticalFlow(img_0, img_1, &flow_img_, true);
      projected_flow_points.clear();
      for (const auto &pt : projected_points) {
        int a = std::round(pt(0) + initial_flow(0));
        int b = std::round(pt(1) + initial_flow(1));
        if (a < 0 || a >= height || b < 0 || b >= width) {
          projected_flow_points.emplace_back(pt + initial_flow);
          continue;
        }
        float flow_a = flow_img_.at<cv::Vec2f>(a, b)[1];
        float flow_b = flow_img_.at<cv::Vec2f>(a, b)[0];
        projected_flow_points.emplace_back(flow_a + pt(0) + initial_flow(0),
                                           flow_b + pt(1) + initial_flow(1));
      }
    }
    else {
      projected_flow_points = projected_points;
    }

    const std::vector<int> bias_a = { -1, -1, -1, 1, 1, 1, 0, 0, 0 };
    const std::vector<int> bias_b = { -1, 1, 0, -1, 1, 0, -1, 1, 0 };
    const int kForHashing = 2333333;
    // Neighbor searching.
    for (int u = 0; u < n_world_points; u++) {
      neighbors[u].clear();
      const auto &proj_flow_pt = projected_flow_points[u];
      int a_int = std::max(0.0, std::round(proj_flow_pt(0)));
      int b_int = std::max(0.0, std::round(proj_flow_pt(1)));
      auto cache_iter = single_pixel_cache_.find(a_int * kForHashing + b_int);
      // Hit cache.
      if (cache_iter != single_pixel_cache_.end()) {
        for (int idx : cache_iter->second) {
          neighbors[u].emplace_back(idx, (proj_flow_pt - points[idx]).norm());
        }
        CHECK(!neighbors[u].empty());
        continue;
      }

      const int kNumSlices = 12;
      std::vector<int> segmented_neighbors(kNumSlices, -1);

      bool use_quad_tree = true;
      if (!use_quad_tree) { // Use grid search.
        unsigned h = std::max((unsigned) (proj_flow_pt(0) / kSearchingR + 1e-9), 0u);
        unsigned w = std::max((unsigned) (proj_flow_pt(1) / kSearchingR + 1e-9), 0u);

        for (int j = 0; j < 9; j++) {
          int new_h = (int) h + bias_a[j];
          int new_w = (int) w + bias_b[j];
          if (new_h < 0 || new_h >= (int) grid_h_ || new_w < 0 || new_w >= (int) grid_w_) {
            continue;
          }
          if (!grids_.count(new_h * grid_w_ + new_w)) {
            continue;
          }
          const std::vector<int>& vec = grids_[new_h * grid_w_ + new_w];
          for (int v : vec) {
            Eigen::Vector2d bias = points[v] - proj_flow_pt;
            if (bias.norm() > kSearchingR) {
              continue;
            }
            double hope_slice = (std::atan2(bias(1), bias(0)) + kOnePi) / kDoublePi - 1e-9;
            CHECK_NEAR(hope_slice, 0.5, 0.5 + 1e-3);
            hope_slice *= kNumSlices;
            int idx = std::max(0.0, std::floor(hope_slice));
            if (segmented_neighbors[idx] == -1) {
              segmented_neighbors[idx] = v;
            }
            else if ((points[segmented_neighbors[idx]] - proj_flow_pt).norm() > bias.norm()) {
              segmented_neighbors[idx] = v;
            }
          }
        }
      }
      else { // Use quad tree.
        std::vector<int> neighbors_in_circle;
        quad_tree_->SearchingR(proj_flow_pt, kSearchingR, &neighbors_in_circle);
        for (int v : neighbors_in_circle) {
          Eigen::Vector2d bias = points[v] - proj_flow_pt;
          if (bias.norm() > kSearchingR) {
            continue;
          }
          double hope_slice = (std::atan2(bias(1), bias(0)) + kOnePi) / kDoublePi - 1e-9;
          CHECK_NEAR(hope_slice, 0.5, 0.5 + 1e-3);
          hope_slice *= kNumSlices;
          int idx = std::max(0.0, std::floor(hope_slice));
          if (segmented_neighbors[idx] == -1) {
            segmented_neighbors[idx] = v;
          }
          else if ((points[segmented_neighbors[idx]] - proj_flow_pt).norm() > bias.norm()) {
            segmented_neighbors[idx] = v;
          }
        }
      }
      std::vector<int> current_neighbors;
      for (int idx : segmented_neighbors) {
        if (idx >= 0) {
          current_neighbors.push_back(idx);
        }
      }

      if (current_neighbors.empty()) {
        int nearest_idx = quad_tree_->NearestIdx(proj_flow_pt);
        current_neighbors.push_back(nearest_idx);
      }
      // Get candidates.
      int K = std::min(16, (int) current_neighbors.size());
      // CHECK(K > 0);
      std::vector<unsigned int> labels;
      if (K == current_neighbors.size()) {
        labels.resize(K);
        for (int t = 0; t < K; t++) {
          labels[t] = t;
        }
      }
      else {
        LOG(FATAL) << "deprecated.";
        Eigen::MatrixXd pts(1, current_neighbors.size());
        for (int t = 0; t < current_neighbors.size(); t++) {
          auto bias = points[current_neighbors[t]] - proj_flow_pt;
          double ang = 0.0;
          if (bias.norm() > 1e-9) {
            ang = std::atan2(bias(1), bias(0));
          }
          pts(0, t) = ang;
        }
        KMeansClustering kmeans;
        kmeans.SetK(K);
        kmeans.SetPoints(pts);
        kmeans.SetInitMethod(KMeansClustering::KMEANSPP);
        kmeans.SetRandom(false); // for repeatable results
        kmeans.Cluster();
        labels = kmeans.GetLabels();
      }
      std::vector<std::pair<int, double> > candidates(K, std::make_pair(-1, 1e9));
      for (int idx = 0; idx < current_neighbors.size(); idx++) {
        const auto &pt = points[current_neighbors[idx]];
        double dis = (pt - proj_flow_pt).norm();
        int k = labels[idx];
        if (k >= 0 && candidates[k].second > dis) {
          candidates[k] = std::make_pair(current_neighbors[idx], dis);
        }
      }

      auto& vec = single_pixel_cache_[a_int * kForHashing + b_int];
      double dis_threshold = 1e9;
      for (const auto &candidate : candidates) {
        if (candidate.first != -1 && candidate.second < 2.5) {
          dis_threshold = std::min(dis_threshold, std::max(candidate.second, 1.0) * 10.0);
        }
      }
      for (const auto &candidate : candidates) {
        if (candidate.first != -1 && candidate.second < dis_threshold + 1e-9) {
          neighbors[u].push_back(candidate);
          vec.push_back(candidate.first);
        }
      }
    }
  }

  // DP.
  {
    const int K = 17;
    std::vector<int> fa(n_world_points, -1);
    std::vector<int> degrees(n_world_points);
    std::vector<int> que;
    std::vector<int> vis(n_world_points, 0);
    std::vector<std::vector<double> > f(n_world_points);
    std::vector<std::vector<int>> states(n_world_points);
    for (int u = 0; u < n_world_points; u++) {
      CHECK(!neighbors[u].empty());
      f[u].resize(neighbors[u].size(), 1e9);
      states[u].resize(neighbors[u].size(), -1);
      degrees[u] = edges[u].size();
      if (degrees[u] > 10) {
        std::cout << degrees[u] << std::endl;
      }
      if (degrees[u] <= 1) {
        que.push_back(u);
        vis[u] = 1;
      }
    }

    std::vector<std::pair<int, double>> for_dp;
    std::function<void(int, int, int, int, double, Eigen::Vector2d, const Eigen::Vector2d&)> DFS;
    DFS = [&DFS, &for_dp, &neighbors, &f, &points, &states, this]
        (int u, int i, int n, int state, double dis, Eigen::Vector2d current_bias, const Eigen::Vector2d& hope_bias) -> void {
      if (i == n) {
        for (int idx = 0; idx < neighbors[u].size(); idx++) {
          const auto &candidate = neighbors[u][idx];
          double laplacian_cost;
          if (n == 0 || n > 1) {
            laplacian_cost = 0.0;
          } else {
            Eigen::Vector2d final_bias = current_bias + points[candidate.first];
            const Eigen::Vector2d& tang = extractor_->tangs_[candidate.first];
            if (std::abs(final_bias(0) * tang(1) - final_bias(1) * tang(0)) < 2.0) {
              final_bias = tang * tang.dot(final_bias);
              laplacian_cost = (final_bias - hope_bias).norm();
            }
            else {
              laplacian_cost = (final_bias - hope_bias).norm();
            }
          }
          double val = dis + candidate.second * 1e-1 + laplacian_cost * 1.0;
          if (f[u][idx] > val) {
            f[u][idx] = val;
            states[u][idx] = state;
          }
        }
        return;
      }
      int v = for_dp[i].first;
      double weight = for_dp[i].second;
      for (int idx = 0; idx < neighbors[v].size(); idx++) {
        const auto &candidate = neighbors[v][idx];
        DFS(u, i + 1, n, state * K + idx, dis + f[v][idx], current_bias - weight * points[candidate.first], hope_bias);
      }
    };
    for (int i = 0; i < que.size(); i++) {
      int u = que[i];
      for_dp.clear();
      for (const auto &edge : edges[u]) {
        int v = edge.first;
        if (fa[v] == u) {
          for_dp.emplace_back(edge);
        } else {
          CHECK_EQ(fa[u], -1);
          fa[u] = v;
          if (--degrees[v] <= 1 && !vis[v]) {
            que.push_back(v);
            vis[v] = 1;
          }
        }
      }
      Eigen::Vector2d hope_bias(projected_points[u]);
      double sum_weight = 0.0;
      for (const auto &edge : for_dp) {
        sum_weight += edge.second;
      }
      if (for_dp.size() > 0) {
        for (auto &edge : for_dp) {
          edge.second /= sum_weight;
          hope_bias -= projected_points[edge.first] * edge.second;
        }
      }
      else {
        hope_bias = Eigen::Vector2d(0.0, 0.0);
      }
      DFS(u, 0, for_dp.size(), 0, 0.0, Eigen::Vector2d(0.0, 0.0), hope_bias);
    }
    CHECK_EQ(que.size(), n_world_points);
    std::vector<int> matching_idx(n_world_points, -1);
    for (int i = ((int) que.size()) - 1; i >= 0; i--) {
      int u = que[i];
      if (matching_idx[u] == -1) {
        double val = 1e9;
        for (int idx = 0; idx < neighbors[u].size(); idx++) {
          if (val > f[u][idx]) {
            val = f[u][idx];
            matching_idx[u] = idx;
          }
        }
      }
      int state = states[u][matching_idx[u]];
      for (auto iter = edges[u].rbegin(); iter != edges[u].rend(); iter++) {
        int v = iter->first;
        if (fa[u] == v) {
          continue;
        }
        matching_idx[v] = state % K;
        state /= K;
      }
    }

    ray_normals.clear();
    matching_indexes_.clear();
    matching_radius.clear();
    matching_pixels_.clear();
    matched_ = std::vector<int>(n_points_, 0);
    nearest_matched_idx_ = std::vector<int>(n_points_, -1);
    matched_std_dev_ = std::vector<double>(n_points_, 0.0);
    if (is_world_point_self_occluded_.size() != n_world_points) {
      is_world_point_self_occluded_ = std::vector<int>(n_world_points, 0);
    }
    std::vector<std::vector<int>> matched_world_indexes(n_points_);

    for (int u = 0; u < n_world_points; u++) {
      CHECK(matching_idx[u] != -1) << "Core Index is -1";
      if (WorldPointOutImageRange(world_points[u])) {
        ray_normals.emplace_back(Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(0.0, 0.0, 0.0));
        matching_indexes_.emplace_back(-1);
        matching_radius.emplace_back(-1.0);
        matching_pixels_.emplace_back(-1e9, -1e9);
      }
      else {
        int idx = neighbors[u][matching_idx[u]].first;
        // Please make sure that the permutation in ICP and extractor is the same.
        ray_normals.emplace_back(icp_->RayNormalByIdx(idx));
        matching_indexes_.emplace_back(idx);
        matching_radius.emplace_back(extractor_->estimated_rs_[idx]);
        matching_pixels_.emplace_back(extractor_->points_[idx]);
        matched_[idx] = 1;
        matched_world_indexes[idx].emplace_back(u);
      }
    }

    // Calc std dev. && nearest world point idx.
    for (int u = 0; u < n_points_; u++) {
      if (!matched_[u]) {
        matched_std_dev_[u] = 1e9;
        continue;
      }
      Eigen::Vector3d mean_pt = Eigen::Vector3d::Zero();
      std::vector<Eigen::Vector3d> current_matchted_points;
      double dis = 1e9;
      for (int v : matched_world_indexes[u]) {
        if ((projected_points[v] - extractor_->points_[u]).norm() < dis) {
          dis = (projected_points[v] - extractor_->points_[u]).norm();
          nearest_matched_idx_[u] = v;
        }
        current_matchted_points.emplace_back(world_points[v]);
      }
      for (int neighbor : extractor_->neighbors_[u]) {
        for (int v : matched_world_indexes[neighbor]) {
          current_matchted_points.emplace_back(world_points[v]);
        }
      }
      for (const auto& pt : current_matchted_points) {
        mean_pt += pt;
      }
      mean_pt /= (double) current_matchted_points.size();
      double std_dev = 0.0;
      for (const auto& pt : current_matchted_points) {
        std_dev += (mean_pt - pt).squaredNorm();
      }
      if (current_matchted_points.empty()) {
        std_dev = 0;
      } else {
        std_dev = std::sqrt(std_dev / current_matchted_points.size());
      }
      matched_std_dev_[u] = std_dev;
      is_self_occluded_[u] |= int(std_dev > 10.0 / focal_length_);
      if (is_self_occluded_[u]) {
        for (int v : matched_world_indexes[u]) {
          is_world_point_self_occluded_[v] |= 1;
        }
      }
    }
  }

  // Debug info
  if (show_debug_info) {
    int height = extractor_->height_;
    int width = extractor_->width_;
    cv::Mat img(height * 2, width * 2, CV_8UC3);
    std::memset(img.data, 255, height * width * 4 * 3);
    int cnt = 0;
    for (int u = 0; u < n_world_points; u++) {
      auto pt_0 = R_ * world_points[u] + T_;
      double x_0 = pt_0(0) / pt_0(2) * focal_length_ + extractor_->width_ * 0.5;
      double y_0 = pt_0(1) / pt_0(2) * focal_length_ + extractor_->height_ * 0.5;

      auto pt_1 = ray_normals[u].first;
      double x_1 = pt_1(0) / pt_1(2) * focal_length_ + extractor_->width_ * 0.5;
      double y_1 = pt_1(1) / pt_1(2) * focal_length_ + extractor_->height_ * 0.5;

      if (cnt++ % 10 == 0) {
        cv::line(img, cv::Point(x_0 * 2.0, y_0 * 2.0), cv::Point(x_1 * 2.0, y_1 * 2.0), cv::Scalar(200, 0, 200), 1.5);
      }
    }

    for (int u = 0; u < n_world_points; u++) {
      auto pt = R_ * world_points[u] + T_;
      double x = pt(0) / pt(2) * focal_length_ + extractor_->width_ * 0.5;
      double y = pt(1) / pt(2) * focal_length_ + extractor_->height_ * 0.5;
      cv::circle(img, cv::Point(x * 2.0, y * 2.0), 2.0, cv::Scalar(200, 200, 0), -1);
    }

    for (const auto &pt : points) {
      int a = std::round(pt(0));
      int b = std::round(pt(1));
      cv::circle(img, cv::Point(b * 2.0, a * 2.0), 2.0, cv::Scalar(0, 200, 200), -1);
    }

    cv::imshow("debug", img);
    cv::waitKey(-1);
  }
}

void View::CopyWorldPointMatching(int u, int new_n_points) {
  matching_ray_normals_.emplace_back(matching_ray_normals_[u]);
  matching_indexes_.emplace_back(matching_indexes_[u]);
  matching_radius_.emplace_back(matching_radius_[u]);
  matching_pixels_.emplace_back(matching_pixels_[u]);
  CHECK_EQ(matching_ray_normals_.size(), new_n_points);
  CHECK_EQ(matching_indexes_.size(), new_n_points);
  CHECK_EQ(matching_radius_.size(), new_n_points);
  CHECK_EQ(matching_pixels_.size(), new_n_points);
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> View::GetMatching(int world_point_idx) {
  return matching_ray_normals_[world_point_idx];
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> View::GetWorldRayByIdx(int idx) {
  Eigen::Vector3d p_world = camera_.Image2World(extractor_->points_[idx]);
  Eigen::Vector3d o_world = camera_.Cam2World(Eigen::Vector3d(0.0, 0.0, 0.0));
  return { o_world, (p_world - o_world).normalized() };
}

Eigen::Vector2d View::GetMatchingPixels(int world_point_idx) {
  return matching_pixels_[world_point_idx];
}

double View::GetMatchingRadius(int world_point_idx) {
  return matching_radius_[world_point_idx];
}

void View::OutVisualizationInfo(const std::string& dir_path) {
  std::ofstream my_file;
  my_file.open(dir_path + std::to_string(time_stamp_) + ".txt");
  my_file << R_ << std::endl << T_.transpose() << std::endl;
  my_file << extractor_->width_ << " " << extractor_->height_ << std::endl;
  my_file << focal_length_ << std::endl;
  my_file.close();
}

void View::OutDebugImage(const std::string& img_name,
                         const std::vector<Eigen::Vector3d>& points,
                         bool show_or_write,
                         int wait_time) {
  cv::Mat img = OutDebugImageCore(points);
  if (show_or_write) {
    cv::imshow(img_name, img);
    cv::waitKey(wait_time);
  }
  else {
    cv::imwrite(img_name, img);
  }
}

void View::OutDebugImage(const std::string& mark,
                         const std::vector<Eigen::Vector3d>& points,
                         GlobalDataPool* global_data_pool) {
#ifdef USE_GUI
  if (global_data_pool == nullptr) {
    return;
  }
  cv::Mat img = OutDebugImageCore(points);
  if (mark == "last") {
    global_data_pool->UpdateImageData(img);
  }
  else {
    LOG(FATAL) << "No such mark.";
  }
#endif
}

cv::Mat View::OutDebugImageCore(const std::vector<Eigen::Vector3d>& points) {
  cv::Mat img(extractor_->height_, extractor_->width_, CV_8UC3);
  std::memset(img.data, 0xFF, img.cols * img.rows * 3);
  for (const auto &point : extractor_->points_) {
    int a = std::round(point(0));
    int b = std::round(point(1));
    if (a < 0 || a >= img.rows || b < 0 || b >= img.cols) {
      continue;
    }
    cv::circle(img, cv::Point(b, a), 1, cv::Scalar(200, 0, 100), -1);
  }
  for (const auto &point : points) {
    Eigen::Vector3d pt = R_ * point + T_;
    if (pt(2) < 0.0) {
      LOG(INFO) << "Weird camera pose, view_idx: " << time_stamp_;
      continue;
    }
    int a = std::round(pt(1) / pt(2) * focal_length_ + 0.5 * img.rows);
    int b = std::round(pt(0) / pt(2) * focal_length_ + 0.5 * img.cols);
    if (a < 0 || a >= img.rows || b < 0 || b >= img.cols) {
      continue;
    }
    cv::circle(img, cv::Point(b, a), 1, cv::Scalar(200, 200, 0), -1);
  }

  if (emphasize_missing_paths_) {
    std::vector<std::vector<int>> missing_paths;
    GetMissingPaths(&missing_paths);
    for (const auto& path : missing_paths) {
      for (const int u : path) {
        int a = std::round(extractor_->points_[u](0));
        int b = std::round(extractor_->points_[u](1));
        if (a < 0 || a >= img.rows || b < 0 || b >= img.cols) {
          continue;
        }
        cv::circle(img, cv::Point(b, a), 1, cv::Scalar(0, 200, 200), -1);
      }
    }
  }

  for (int u = 0; u < n_points_; u++) {
    if (is_self_occluded_[u]) {
      int a = std::round(extractor_->points_[u](0));
      int b = std::round(extractor_->points_[u](1));
      if (a < 0 || a >= img.rows || b < 0 || b >= img.cols) {
        continue;
      }
      cv::circle(img, cv::Point(b, a), 1, cv::Scalar(0, 200, 0), -1);
    }
  }
  return img;
}

void View::GetMissingWorldRays(const std::vector<Eigen::Vector3d>& points,
                               std::vector<std::pair<Eigen::Vector3d, double>>* world_rays) {
  world_rays->clear();
  int height = extractor_->height_;
  int width = extractor_->width_;
  cv::Mat projected_img(height, width, CV_8UC1);
  std::memset(projected_img.data, 0, height * width);
  double median_depth = 0.0;
  for (const auto &point : points) {
    auto pt = R_ * point + T_;
    median_depth += pt(2);
    int a = std::round(pt(1) / pt(2) * focal_length_ + height * 0.5);
    int b = std::round(pt(0) / pt(2) * focal_length_ + width * 0.5);
    if (a < 0 || a >= height || b < 0 || b >= width) {
      continue;
    }
    projected_img.data[a * width + b] = 1;
  }

  for (int idx = 0; idx < n_points_; idx++) {
    const Eigen::Vector2d& point = extractor_->points_[idx];
    int a = std::round(point(0));
    int b = std::round(point(1));
    if (a < 0 || a >= height || b < 0 || b >= width) {
      continue;
    }
    if (extractor_->tang_scores_[idx] < 0.8 || extractor_->estimated_rs_[idx] > extractor_->average_radius_ * 2.0) {
      continue;
    }
    int r = 1;
    bool found = false;
    for (int i = std::max(0, a - r); i <= std::min(height - 1, a + r) && !found; i++) {
      for (int j = std::max(0, b - r); j <= std::min(width - 1, b + r) && !found; j++) {
        found = projected_img.data[i * width + j];
      }
    }
    if (!found) {
      Eigen::Vector3d ray((point(1) - 0.5 * width) / focal_length_, (point(0) - 0.5 * height) / focal_length_, 1.0);
      ray /= ray.norm();
      double radius = extractor_->estimated_rs_[idx];
      double tang_score = extractor_->tang_scores_[idx];
      world_rays->emplace_back(R_.inverse() * ray, 1.0);
    }
  }
}

void View::UpdateMissingQuadTree() {
  missing_quad_tree_.release();
  std::vector<std::vector<int>> missing_paths;
  GetMissingPaths(&missing_paths);
  if (missing_paths.empty()) {
    missing_quad_tree_.reset(nullptr);
    return;
  }
  std::vector<Eigen::Vector2d> missing_points;
  for (const auto& path : missing_paths) {
    for (int u : path) {
      missing_points.emplace_back(extractor_->points_[u]);
    }
  }
  missing_quad_tree_.reset(new QuadTree(missing_points));
}

void View::GetMissingPaths(std::vector<std::vector<int>>* missing_paths,
                           int too_short_path_threshold) {
  missing_paths->clear();
  const auto& initial_paths = extractor_->paths_;
  const int kMinMissingPathSize = too_short_path_threshold;
  for (const auto& path : initial_paths) {
    int last_matched = -1;
    for (int i = 0; i < path.size(); i++) {
      // CHECK_LT(path[i], matched_.size());
      if (matched_[path[i]]) {
        if (i - last_matched - 1 >= kMinMissingPathSize) {
          missing_paths->emplace_back();
          for (int j = last_matched + 1; j < i; j++) {
            missing_paths->back().emplace_back(path[j]);
          }
        }
        last_matched = i;
      }
    }
    if (path.size() - last_matched - 1 >= kMinMissingPathSize) {
      missing_paths->emplace_back();
      for (int j = last_matched + 1; j < path.size(); j++) {
        missing_paths->back().emplace_back(path[j]);
      }
    }
  }
}

void View::GetDepthIntersectionsByWorldRay(const Eigen::Vector3d& o,
                                           const Eigen::Vector3d& v,
                                           double d_min,
                                           double d_max,
                                           std::vector<std::pair<double, double>>* depth_intersections) {
  depth_intersections->clear();
  Eigen::Vector3d o_cam = camera_.World2Cam(o);
  Eigen::Vector3d v_cam = (camera_.World2Cam(o + v) - o_cam).normalized();
  if (std::abs(v_cam(0)) < 1e-3 || std::abs(v_cam(1)) < 1e-3) {
    return;
  }
  if (o_cam.norm() < 1e-3) {
    return;
  }
  Eigen::Vector2d o_img = camera_.Cam2Image(o_cam);
  Eigen::Vector2d v_img = (camera_.Cam2Image(o_cam + v_cam) - o_img).normalized();

  std::vector<std::pair<double, Eigen::Vector2d>> intersections;
  segment_quad_tree_->FindIntersections(o_img, v_img, -1e9, 1e9, &intersections);

  const double kHardRadius = 3.0;
  auto FindDByPoint = [&o_cam, &v_cam](const Eigen::Vector3d& pt) -> double {
    CHECK_NEAR(pt(2), 1.0, 1e-8);
    for (int k = 0; k < 2; k++) {
      // ---------------------------------------------------------------- //
      // o_cam(k) + v_cam(k) * t = pt(k) * (o_cam(2) + v_cam(2) * t)      //
      // (v_cam(k) - v_cam(2) * pt(k)) * t = pt(k) * o_cam(2) - o_cam(k)  //
      // -----------------------------------------------------------------//
      if (std::abs(v_cam(k) - v_cam(2) * pt(k)) > 1e-8) {
        // LOG(INFO) << "---------";
        // LOG(INFO) << pt(k) * o_cam(2) - o_cam(k) << " " << (v_cam(k) - v_cam(2) * pt(k));
        // LOG(INFO) << "---------";
        return (pt(k) * o_cam(2) - o_cam(k)) / (v_cam(k) - v_cam(2) * pt(k));
      }
    }
    return std::numeric_limits<double>::signaling_NaN();
  };
  for (const auto& pr : intersections) {
    double d_img = pr.first;

    double l_bound = FindDByPoint(camera_.Image2Cam(o_img + v_img * (d_img - kHardRadius), 1.0));
    double r_bound = FindDByPoint(camera_.Image2Cam(o_img + v_img * (d_img + kHardRadius), 1.0));
    if (l_bound > r_bound) {
      std::swap(l_bound, r_bound);
    }
    if (!std::isnan(l_bound) && !std::isnan(r_bound) &&
        l_bound + 1e-9 > d_min && std::max(l_bound, d_min) < std::min(r_bound, d_max)) {
      depth_intersections->emplace_back(std::max(l_bound, d_min), std::min(r_bound, d_max));
    }
  }
}

double View::CalcBidirectionProjectionError(const std::vector<Eigen::Vector3d>& points) {
  double error = 0.0;
  int height = extractor_->height_;
  int width = extractor_->width_;
  std::vector<Eigen::Vector2d> points_0, points_1;
  for (const auto &point : points) {
    points_0.emplace_back(camera_.World2Image(point));
  }
  points_1 = extractor_->points_;
  auto cmp = [](const Eigen::Vector2d &a, const Eigen::Vector2d &b) {
    return a(0) < b(0);
  };
  std::sort(points_0.begin(), points_0.end(), cmp);
  std::sort(points_1.begin(), points_1.end(), cmp);
  if (points_0.empty() || points_1.empty()) {
    return 1e9;
  }
  double r = 10.0;
  auto AveDis = [this, r](std::vector<Eigen::Vector2d> &points_0, std::vector<Eigen::Vector2d> &points_1) {
    int n_points_0 = points_0.size();
    int n_points_1 = points_1.size();
    double dis = 0.0;
    int L = 0;
    for (int u = 0; u < n_points_0; u++) {
      while (L < n_points_1 && points_0[u](0) - points_1[L](0) > r) {
        L++;
      }
      double min_dis = r;
      for (int v = L; v < n_points_1 && points_1[v](0) - points_0[u](0) < r; v++) {
        min_dis = std::min(min_dis, (points_1[v] - points_0[u]).norm());
      }
      dis += min_dis;
    }
    return dis / (double) n_points_0;
  };
  return (AveDis(points_0, points_1) + AveDis(points_1, points_0)) * 0.5;
}

double View::CalcIOU(const std::vector<Eigen::Vector3d> &points) {
  std::vector<int> fov_indexes;
  FindPointsInFOV(points, 1.0, &fov_indexes);
  std::set<std::pair<int, int>> st, st_pix;
  for (int idx : fov_indexes) {
    Eigen::Vector3d point(R_ * points[idx] + T_);
    int a = std::round((point(1) / point(2) * focal_length_ + extractor_->height_ * 0.5) * 0.2);
    int b = std::round((point(0) / point(2) * focal_length_ + extractor_->width_  * 0.5) * 0.2);
    st.emplace(a, b);
  }
  unsigned I = 0;
  unsigned U = st.size();
  for (const auto &dir_point : extractor_->candi_points_) {
    int a = std::round(dir_point.o_(0) * 0.2);
    int b = std::round(dir_point.o_(1) * 0.2);
    if (st_pix.count(std::make_pair(a, b))) {
      continue;
    }
    st_pix.emplace(a, b);
    if (st.count(std::make_pair(a, b))) {
      I++;
    } else {
      U++;
    }
  }
  return (double) I / (double) U;
}

void View::RefreshCoveredMap() {
  covered_ = matched_;
}

// Calc but not update.
bool View::SinglePointHitMap(const Eigen::Vector3d& world_point) {
  LOG(FATAL) << "Deprecated.";
}

// Calc and update.
double View::CalcCoveredPixelsRatio(const std::vector<Eigen::Vector3d>& world_points,
                                    const std::vector<int>& path_indexes) {
  if (path_indexes.size() <= 3) { // Too short
    return 0.0;
  }
  int cnt = 0;
  int in_view_cnt = 0;
  for (int idx : path_indexes) {
    if (matching_indexes_[idx] < 0) {
      continue;
    }
    in_view_cnt++;
    if (covered_[matching_indexes_[idx]]) {
      cnt++;
    }
  }
  if (in_view_cnt <= 0) {
    return -1.0;
  }
  for (int idx : path_indexes) {
    if (matching_indexes_[idx] >= 0) {
      covered_[matching_indexes_[idx]] = 0;
      for (const int v : extractor_->neighbors_[matching_indexes_[idx]]) {
        covered_[v] = 0;
      }
    }
  }
  return double(cnt) / double(in_view_cnt);
}

double View::AverageNearestProjectionError(const std::vector<Eigen::Vector3d>& points) {
  return 0;
}

bool View::WorldPointOutImageRange(const Eigen::Vector3d& point, double ratio) {
  Eigen::Vector2d pt = camera_.World2Image(point);
  return (pt(0) < extractor_->height_ * ratio
          || pt(0) > extractor_->height_ * (1.0 - ratio)
          || pt(1) < extractor_->width_ * ratio
          || pt(1) > extractor_->width_ * (1.0 - ratio));
}
