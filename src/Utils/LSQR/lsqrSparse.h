//
// Created by aska on 2019/8/5.
//

#pragma once
#include "lsqrBase.h"

#include <vector>

class lsqrSparse : public lsqrBase {
public:
  lsqrSparse() = default;
  ~lsqrSparse() = default;
  void SetMatrix(const std::vector<unsigned> &coord_r,
                 const std::vector<unsigned> &coord_c,
                 const std::vector<double> &values);
  void Aprod1(unsigned int m, unsigned int n, const double * x, double * y ) const;
  void Aprod2(unsigned int m, unsigned int n, double * x, const double * y ) const;
private:
  std::vector<unsigned> coord_r_;
  std::vector<unsigned> coord_c_;
  std::vector<double> values_;
};
