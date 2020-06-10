//
// Created by aska on 2019/9/11.
//
#pragma once

#include "Common.h"

class StreamerBase {
public:
  StreamerBase() = default;
  virtual ~StreamerBase() = default;
  virtual int Height() = 0;
  virtual int Width() = 0;
  virtual int Channels() = 0;
  virtual uchar* CurrentFrame() = 0;
  virtual uchar* CurrentPixelAt(int a, int b) = 0;
  virtual int SwitchToNextFrame() { return 0; }
  virtual void Reset() {};
};

class ImageStreamer : public StreamerBase {
public:
  ImageStreamer(const std::string& img_path, int n_images);
  ~ImageStreamer() final = default;
  int Height() final { return height_; }
  int Width() final { return width_; }
  int Channels() final { return channels_; }
  uchar* CurrentFrame() final;
  uchar* CurrentPixelAt(int a, int b) final;
  int SwitchToNextFrame() final;
  void Reset() final;
  std::string img_path_;
  int n_images_ = 0;
  int current_img_idx_ = -1;
  int idx_direction_ = 1;
  int height_;
  int width_;
  int channels_;
  cv::Mat img_;
};

class SingleImageStreamer : public StreamerBase {
public:
  SingleImageStreamer(const std::string& img_path);
  ~SingleImageStreamer() final = default;
  int Height() final { return height_; }
  int Width() final { return width_; }
  int Channels() final { return channels_; }
  uchar* CurrentFrame() final;
  uchar* CurrentPixelAt(int a, int b) final;
  std::string img_path_;
  int height_;
  int width_;
  int channels_;
  cv::Mat img_;
};