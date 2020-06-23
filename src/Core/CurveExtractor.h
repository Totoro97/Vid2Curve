//
// Created by aska on 2019/3/12.
//
#pragma once
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include "../Utils/Streamer.h"

// std
#include <tuple>

enum SegNoiseType { SHAKING, POLLUTING };

struct ImageDirPoint {
public:
  ImageDirPoint() = default;
  ImageDirPoint(const Eigen::Vector2d &o, const Eigen::Vector2d &v, double score = 1.0, int idx = -1):
      o_(o), v_(v), score_(score) {}
  Eigen::Vector2d o_, v_;
  double score_ = 1.0;
  int idx = -1;
};

class CurveExtractor {

public:
  CurveExtractor(StreamerBase* streamer, PropertyTree* ptree);
  // Deprecated.
  CurveExtractor(double *prob_map, int height, int width);
  ~CurveExtractor();
  void CalcEstimateRadius();

  // Method 1: Naive.
  // TODO: Better default parameters.
  void RunPca(int trunc_r = 3, double r = 3.0);
  void RunPcaGPU(int trunc_r = 3, double r = 3.0);
  // Method 2: Color difference.
  void RunPcaConsideringColor(StreamerBase* streamer);
  // Method 3: https://cgl.ethz.ch/Downloads/Publications/Papers/2013/Nor13/Nor13.pdf
  void RunGradientBasedExtraction(StreamerBase* streamer);

  std::pair<Eigen::Vector2d, double> CalcTangAndScore(const Eigen::Vector2d &pix);

  void FindNearestPointsOffline(const std::vector<Eigen::Vector2d>& query_points,
                                std::vector<ImageDirPoint>* nearest_points,
                                double searching_r = 15.0);

  void CalcEigens(const Eigen::Matrix2d &C, double &l1, double &l2,
                  Eigen::Vector2d &v1, Eigen::Vector2d &v2);
  void SmoothPoints();
  void LinkAll();
  void FilterTooThickPoints();
  void CalcLinkInformation();
  void CalcPaths();
  void GetEdgePoints(std::vector<Eigen::Vector2d>* points);
  cv::Mat GetSegImageWithNoise(double shaking_len,
                               double shaking_ang,
                               const std::vector<Eigen::Vector2d>& polluted_positions,
                               double polluted_radius);
  cv::Mat GetGaussianNoiseSegmentation(double std_dev = 1.0);
  double RadiusAt(double a, double b);
  double RadiusAtInt(int a, int b);
  double* ConvertImg2ProbMap(const cv::Mat &img);
  double ProbAt(double a, double b);
  double EstimateR(double base_a, double base_b, double step_a, double step_b);

public:
  // Data.
  bool use_gpu_;
  bool need_delete_prob_map_;
  // TODO: Change it from ptr to std::vector<double>.
  double *prob_map_ = nullptr;
  std::vector<int> seg_map_;
  std::vector<ImageDirPoint> candi_points_;
  std::vector<Eigen::Vector2d> points_;
  std::vector<Eigen::Vector2d> tangs_;
  std::vector<double> tang_scores_;
  std::vector<double> estimated_rs_;
  double average_radius_;
  std::vector<std::tuple<int, int, double>> link_info_;
  std::vector<std::vector<int>> paths_;
  std::vector<std::vector<int>> edges_;
  std::vector<std::vector<int>> neighbors_;
  std::vector<int> near_junction_;
  std::vector<double> thick_ratio_;
  int height_, width_;
  int n_points_;
  int id_ = -1;
  PropertyTree* ptree_;

};

// ------------------------ Declaration for GPU methods ----------------------

void CudaPca(int height, int width, double* prob_map, double* out_data_pool);

void CudaPcaConsideringColor(int height, int width, int radius, uchar* img_data, double* out_data_pool);
