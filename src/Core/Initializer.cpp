//
// Created by aska on 2019/9/4.
//

#include "Initializer.h"

#include "../Utils/Utils.h"

Initializer::Initializer(
    const std::vector<CurveExtractor*> &curve_extractors,
    double focal_length,
    double width,
    double height,
    double single_depth_error_weight,
    double relative_depth_error_weight) :
    curve_extractors_(curve_extractors),
    focal_length_(focal_length),
    width_(width),
    height_(height),
    single_depth_error_weight_(single_depth_error_weight),
    relative_depth_error_weight_(relative_depth_error_weight) {
}

// Estimate camera pose candidates and initial points by several frames.
// Currently we only support two frames as input.
void Initializer::GetInitialModelData(std::vector<std::unique_ptr<ModelData>>* model_states) {
  model_states->clear();
  CHECK_EQ(curve_extractors_.size(), 2);
  CurveExtractor* extractor_0 = curve_extractors_[0];
  CurveExtractor* extractor_1 = curve_extractors_[1];
  auto curve_matcher = std::make_unique<CurveMatcher>(extractor_0, extractor_1);
  std::vector<ImageLocalMatching> matchings;
  curve_matcher->FindMatchings("DP", &matchings);
  CHECK(!matchings.empty());

  const double kTranslationBias = 0.01;
  double camera_poses[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
  const double kInitialDepth = 1.0;
  std::vector<double> depths(extractor_0->n_points_, kInitialDepth);
  auto RandLR = [](double L, double R) {
    return L + (double) rand() / (double) RAND_MAX * (R - L);
  };
  double average_trans = 0.0;
  for (int x = -1; x <= 1; x++) {
    for (int y = -1; y <= 1; y++) {
      for (int z = 0; z <= 0; z++) {
        // Initialization.
        for (int i = 0; i < extractor_0->n_points_; i++) {
          depths[i] = kInitialDepth;
        }
        // std::fill(depths.begin(), depths.end(), kInitialDepth * RandLR(0.9, 1.1));
        camera_poses[3] = x * kTranslationBias;
        camera_poses[4] = y * kTranslationBias;
        camera_poses[5] = z * kTranslationBias;
        for (int i = 0; i < 3; i++) {
          camera_poses[i] = 0.0;
        }

        // Problem registration.
        ceres::Problem problem;
        for (const auto& matching : matchings) {
          ceres::CostFunction* cost_function =
              ProjectionError::Create(matching, (matching.q.score_ - 1.0) * (-2.0), focal_length_, width_, height_);
          problem.AddResidualBlock(cost_function, nullptr, camera_poses, depths.data() + matching.p.idx);
        }

        double final_single_depth_error_weight =
            single_depth_error_weight_ / kInitialDepth * focal_length_ * matchings.size() / extractor_0->n_points_;
        for (int i = 0; i < extractor_0->n_points_; i++) {
          ceres::CostFunction* cost_function = SingleDepthError::Create(kInitialDepth, final_single_depth_error_weight);
          problem.AddResidualBlock(cost_function, nullptr, depths.data() + i);
        }

        double final_relative_depth_error_weight =
            relative_depth_error_weight_ / kInitialDepth * focal_length_ *
            matchings.size() / (double) extractor_0->n_points_;
        if (final_relative_depth_error_weight > 1e-9) {
          for (const auto& path : extractor_0->paths_) {
            for (int i = 0; i + 1 < path.size(); i++) {
              ceres::CostFunction *cost_function =
                  RelativeDepthError::Create(final_relative_depth_error_weight);
              problem.AddResidualBlock(cost_function,
                                       nullptr,
                                       depths.data() + path[i],
                                       depths.data() + path[i + 1]);
            }
          }
        }

        // Solve & Optimization.
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        LOG(INFO) << camera_poses[3] << " " << camera_poses[4] << " " << camera_poses[5];
        double ave_depth = 0.0;
        for (double depth : depths) {
          ave_depth += depth;
        }
        LOG(INFO) << "ave depth:" << ave_depth / depths.size();

        Eigen::Matrix3d R;
        Eigen::Vector3d angle_axis(camera_poses[0], camera_poses[1], camera_poses[2]);
        R = Eigen::AngleAxisd(angle_axis.norm(), angle_axis / angle_axis.norm());
        Eigen::Vector3d T(camera_poses[3], camera_poses[4], camera_poses[5]);
        double trans_len = T.norm() / (ave_depth / depths.size());
        LOG(INFO) << "trans len: " << trans_len;
        bool is_redundant = false;
        const double kMinDifferentTranslationThreshold = 0.01;
        for (const auto& model_state : *model_states) {
          if ((model_state->camera_poses.back().second - T).norm() < kMinDifferentTranslationThreshold) {
            is_redundant = true;
            break;
          }
        }
        if (is_redundant) {
          continue;
        }

        // Add to model state.
        average_trans += trans_len;
        model_states->emplace_back(new ModelData());
        ModelData* new_model_state = model_states->back().get();

        new_model_state->curve_extractors.emplace_back(extractor_0);
        new_model_state->curve_extractors.emplace_back(extractor_1);

        new_model_state->camera_poses.emplace_back(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
        new_model_state->camera_poses.emplace_back(R, T);

        CHECK_EQ(extractor_0->n_points_, depths.size());
        for (int i = 0; i < extractor_0->n_points_; i++) {
          new_model_state->points.emplace_back(
              Utils::ImageCoordToCamera(extractor_0->candi_points_[i].o_, height_, width_, focal_length_) * depths[i]);
        }
      }
    }
  }
  average_trans /= model_states->size();
  if (average_trans < 0.03) {
    model_states->clear();
  }
  LOG(INFO) << "Initialize: Totally " << model_states->size() << " pose candidates.";
}