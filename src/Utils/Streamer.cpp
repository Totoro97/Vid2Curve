//
// Created by aska on 2019/9/11.
//

#include "Streamer.h"

// ------------------------------ ImageStreamer ---------------------------------------

ImageStreamer::ImageStreamer(const std::string& img_path, int n_images):
    n_images_(n_images), img_path_(img_path) {
  if (img_path_.back() != '/') {
    img_path_ += "/";
  }
  Reset();
  SwitchToNextFrame();
}

void ImageStreamer::Reset() {
  current_img_idx_ = -1;
  idx_direction_ = 1;
}

int ImageStreamer::SwitchToNextFrame() {
  if (current_img_idx_ + idx_direction_ >= n_images_) {
    idx_direction_ = -1;
  } else if (current_img_idx_ + idx_direction_ < 0) {
    idx_direction_ = 1;
  }
  current_img_idx_ += idx_direction_;
  LOG(INFO) << img_path_ + std::to_string(current_img_idx_) + ".png";
  img_ = cv::imread(img_path_ + std::to_string(current_img_idx_) + ".png");
  if (current_img_idx_ > 0) {
    CHECK_EQ(img_.rows, height_);
    CHECK_EQ(img_.cols, width_);
    CHECK_EQ(img_.channels(), channels_);
  } else {
    height_ = img_.rows;
    width_ = img_.cols;
    channels_ = img_.channels();
  }
  return 0;
}

uchar* ImageStreamer::CurrentPixelAt(int a, int b) {
  return ((uchar*) img_.data) + a * width_ + b;
}

uchar* ImageStreamer::CurrentFrame() {
  return (uchar*) img_.data;
}

// ------------------------------ SingleImageStreamer ---------------------------------------

SingleImageStreamer::SingleImageStreamer(const std::string& img_path) {
  img_ = cv::imread(img_path);
  height_ = img_.rows;
  width_ = img_.cols;
  channels_ = img_.channels();
}

uchar* SingleImageStreamer::CurrentPixelAt(int a, int b) {
  return ((uchar*) img_.data) + a * width_ + b;
}

uchar* SingleImageStreamer::CurrentFrame() {
  return (uchar*) img_.data;
}