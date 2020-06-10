//
// Created by aska on 2019/8/5.
//

#include "lsqrSparse.h"

void lsqrSparse::SetMatrix(const std::vector<unsigned> &coord_r,
                           const std::vector<unsigned> &coord_c,
                           const std::vector<double> &values) {
  coord_r_ = coord_r;
  coord_c_ = coord_c;
  values_ = values;
}

void lsqrSparse::Aprod1(unsigned int m, unsigned int n, const double *x, double *y) const {
  for (unsigned i = 0; i < coord_r_.size(); i++) {
    y[coord_r_[i]] += x[coord_c_[i]] * values_[i];
  }
}

void lsqrSparse::Aprod2(unsigned int m, unsigned int n, double *x, const double *y) const {
  for (unsigned i = 0; i < coord_r_.size(); i++) {
    x[coord_c_[i]] += y[coord_r_[i]] * values_[i];
  }
}
