//
// Created by aska on 2019/9/26.
//

#pragma once
#include <mutex>
#include <vector>
#ifdef USE_GUI
#include <pangolin/pangolin.h>
#endif

#include "Common.h"

struct CameraMotionInfo {
  std::vector<Eigen::Matrix3d> K_invs_;
  std::vector<Eigen::Matrix4d> Wfs_;
  std::vector<std::pair<int, int>> width_and_heights_;
};

class GlobalDataPool {
public:
  GlobalDataPool();
#ifdef USE_GUI
  void DeleteModel(void* model_ptr);
  void DrawModelPoints(pangolin::GlBuffer* gl_buffer);
  void UpdateModelPoints(void* model_ptr, const std::vector<Eigen::Vector3d>& points);
  void DrawCameras();
  void UpdateCameras(void* model_ptr,
                     const std::vector<Eigen::Matrix3d>& K_invs,
                     const std::vector<Eigen::Matrix4d>& Wfs,
                     const std::vector<std::pair<int, int>>& width_and_heights);
  void DrawImage(pangolin::GlTexture* image_texture);
  void UpdateImageData(const cv::Mat& image);
  void DrawPlotting(pangolin::DataLog* log);
  void UpdatePlottingData(const std::vector<double>& plotting_data);

  int n_points();

public:
  std::map<void*, std::vector<float>> mp_model_points_;
  std::mutex model_points_mutex_;
  std::mutex last_frame_mutex_;
  std::vector<uchar> last_frame_;

  std::vector<double> plotting_data_;
  bool plotting_data_refreshed_ = false;
  std::mutex plotting_data_mutex_;

  std::mutex cameras_mutex_;
  std::map<void*, CameraMotionInfo> mp_cameras_;
#endif
};
