//
// Created by aska on 19-2-1.
//
// This is new.

#include <sstream>
#include "Utils.h"
#include "Common.h"
#include "FastOpticalFlow/OFDIS.h"
#include "LSQR/lsqrSparse.h"
#include <unsupported/Eigen/Splines>
// #include <cnpy.h>

void Utils::SavePointsAsPly(std::string save_path, const std::vector<Eigen::Vector3d> &points) {
  std::ofstream my_file;
  my_file.open(save_path.c_str());
  my_file << "ply\nformat ascii 1.0\n";
  my_file << "element vertex " << points.size() << "\n";
  my_file << "property float32 x\nproperty float32 y\nproperty float32 z\n";
  my_file << "end_header\n";
  for (const auto &pt : points) {
    my_file << pt(0) << " " << pt(1) << " " << pt(2) << "\n";
  }
  my_file.close();
}

/*
void Utils::SavePointsAsJSON(std::string save_path, const std::vector<Eigen::Vector3d> &points) {
  std::ofstream my_file;
  my_file << "{\"points\":[";

}
*/

void Utils::SolveLinearSqrLSQR(int rows,
                               int cols,
                               const std::vector<uint32_t> &coord_r,
                               const std::vector<uint32_t> &coord_c,
                               const std::vector<double> &values,
                               const std::vector<double> &B,
                               std::vector<double> &solution) {
  CHECK_EQ(solution.size(), cols);
  std::vector<double> res_B;
  res_B = B;
  for (int i = 0; i < coord_r.size(); i++) {
    res_B[coord_r[i]] -= solution[coord_c[i]] * values[i];
  }
  std::vector<double> res_solution(cols, 0.0);
  lsqrSparse solver;
  solver.SetMaximumNumberOfIterations(cols * 4);
  solver.SetMatrix(coord_r, coord_c, values);
  solver.Solve(rows, cols, res_B.data(), res_solution.data());
  for (int i = 0; i < cols; i++) {
    solution[i] += res_solution[i];
  }
}

void Utils::SolveLinearSqrStable(int rows, int cols, const std::vector<uint32_t> &coord_r,
                                 const std::vector<uint32_t> &coord_c, const std::vector<double> &values,
                                 const std::vector<double> &B, std::vector<double> &solution){
  Eigen::SparseMatrix<double> A(rows, cols);
  for (int i = 0; i < coord_r.size(); i++) {
    A.insert(coord_r[i], coord_c[i]) = values[i];
  }
  Eigen::VectorXd B_(rows);
  for (int i = 0; i < rows; i++)
    B_(i) = B[i];

  Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double> > solver;
  solver.compute(A);
  if(solver.info() != Eigen::Success) {
    // decomposition failed
    std::cout << "Eigen sparse solver failed" << std::endl;
    std::exit(0);
  }
  Eigen::VectorXd X = solver.solve(B_);
  solution.clear();
  for (int i = 0; i < cols; i++) {
    solution.push_back(X(i));
  }
}

void Utils::GetOpticalFlow(const cv::Mat &img_0, const cv::Mat &img_1, cv::Mat* flow_img, bool blur) {
  StopWatch stop_watch;
  cv::Mat img_blur_0, img_blur_1;
  if (blur) {
    cv::GaussianBlur(img_0, img_blur_0, cv::Point(9, 9), 0);
    cv::GaussianBlur(img_1, img_blur_1, cv::Point(9, 9), 0);
  }
  else {
    img_blur_0 = img_0;
    img_blur_1 = img_1;
  }
  *flow_img = RunOFDIS(img_blur_0, img_blur_1);
}

Eigen::Matrix3d Utils::GetLeftCrossProdMatrix(const Eigen::Vector3d &X) {
  Eigen::Matrix3d M;
  M << 0.0, -X(2), X(1),
    X(2), 0.0, -X(0),
    -X(1), X(0), 0.0;
  return M;
}

Eigen::Matrix3d Utils::GetRightCrossProdMatrix(const Eigen::Vector3d &X) {
  return -GetLeftCrossProdMatrix(X);
}

void Utils::OutputTriMeshAsOBJ(const std::string& save_path,
                               const Eigen::MatrixXd& verts,
                               const Eigen::MatrixXi& faces) {
  std::ofstream my_file;
  my_file.open(save_path.c_str());
  for(int vi=0;vi<verts.rows();vi++){
    my_file <<"v " << verts(vi, 0) << " " << verts(vi, 1) << " " << verts(vi, 2)<< "\n";
  }
  for(int fi=0;fi<faces.rows(); fi++){
    my_file<<"f " <<faces(fi,0)+1 << " " << faces(fi, 1)+1 << " " << faces(fi, 2)+1 <<"\n";
  }
  my_file.close();
}

void Utils::OutputTriMeshAsOBJ(const std::string& save_path,
                               const std::vector<Eigen::Vector3d>& points,
                               const std::vector<std::tuple<int, int, int>>& faces) {
  std::ofstream my_file;
  my_file.open(save_path);
  for (const auto& pt : points) {
    my_file << "v " << pt(0) << " " << pt(1) << " " << pt(2) << "\n";
  }
  for (const auto& face : faces) {
    my_file << "f " << std::get<0>(face) + 1 << " " << std::get<1>(face) + 1 << " " << std::get<2>(face) + 1 << "\n";
  }
  my_file.close();
}

void Utils::OutputCurves(std::string save_path, const std::vector<std::vector<Eigen::Vector3d>>& curves) {
  std::ofstream my_file;
  my_file.open(save_path.c_str());
  for (const auto &curve : curves) {
    for (const auto &pt : curve) {
      my_file << pt(0) << " " << pt(1) << " " << pt(2) << "\n";
    }
    my_file << "---\n";
  }
  my_file.close();
}

void Utils::MergeTriMeshes(std::vector<Eigen::MatrixXd>& in_verts_list,
                           std::vector<Eigen::MatrixXi>& in_faces_list,
                           Eigen::MatrixXd& res_verts,
                           Eigen::MatrixXi& res_faces) {
  res_verts = in_verts_list[0];
  res_faces = in_faces_list[0];
  for(int i=1;i<in_verts_list.size();i++){
    Eigen::MatrixXd tmp_matv(res_verts.rows()+in_verts_list[i].rows(), 3);
    tmp_matv<< res_verts, in_verts_list[i];
    Eigen::MatrixXi tmp_matf(res_faces.rows()+in_faces_list[i].rows(), 3);
    tmp_matf<< res_faces, in_faces_list[i] + Eigen::MatrixXi::Ones(in_faces_list[i].rows(), 3) * res_verts.rows();
    res_faces = tmp_matf;
    res_verts = tmp_matv;
  }
}

void Utils::OutputCurvesAsOBJ(const std::string& save_path,
                              const std::vector<Eigen::Vector3d>& points,
                              const std::vector<std::pair<int, int>>& edges) {
  std::ofstream my_file;
  my_file.open(save_path);
  for (const auto& pt : points) {
    my_file << "v " << pt(0) << " " << pt(1) << " " << pt(2) << "\n";
  }
  for (const auto& pr : edges) {
    my_file << "l " << pr.first + 1 << " " << pr.second + 1 << "\n";
  }
  my_file.close();
}


Eigen::Vector3d Utils::ImageCoordToCamera(const Eigen::Vector2d &x, int height, int width, double focal_length) {
  return Eigen::Vector3d((x(1) - width  * 0.5) / focal_length,
                         (x(0) - height * 0.5) / focal_length,
                         1.0);
// double CompareTwoPointCloud(const std::vector<Eigen::Vector3d>& A, const std::vector<Eigen::Vector3d>& B) {
//   return 0.0;
// }
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> Utils::GetAABB(const std::vector<Eigen::Vector3d>& points) {
  Eigen::Vector3d mi( 1e9,  1e9,  1e9);
  Eigen::Vector3d ma(-1e9, -1e9, -1e9);
  for (const auto& pt : points) {
    for (int t = 0; t < 3; t++) {
      mi(t) = std::min(mi(t), pt(t));
      ma(t) = std::max(ma(t), pt(t));
    }
  }
  return { mi, ma };
}

std::pair<Eigen::Vector2d, Eigen::Vector2d> Utils::GetAABB(const std::vector<Eigen::Vector2d>& points) {
  Eigen::Vector2d mi( 1e9,  1e9);
  Eigen::Vector2d ma(-1e9, -1e9);
  for (const auto& pt : points) {
    for (int t = 0; t < 2; t++) {
      mi(t) = std::min(mi(t), pt(t));
      ma(t) = std::max(ma(t), pt(t));
    }
  }
  return { mi, ma };
}

void Utils::ReadOBJMesh(const std::string& file_name,
                        std::vector<Eigen::Vector3d>* points,
                        std::vector<std::tuple<int, int, int>>* faces) {
  points->clear();
  faces->clear();
  std::ifstream my_file(file_name);
  while (!my_file.eof()) {
    std::string line_str;
    std::getline(my_file, line_str);
    if (line_str[0] == '#') {
      continue;
    }
    else if (line_str[0] == 'v') {
      std::istringstream iss(line_str);
      std::string tmp;
      double x, y, z;
      iss >> tmp >> x >> y >> z;
      points->emplace_back(x, y, z);
    }
    else if (line_str[0] == 'f') {
      std::istringstream iss(line_str);
      std::string tmp;
      int a, b, c;
      iss >> tmp >> a >> b >> c;
      faces->emplace_back(a - 1, b - 1, c - 1);
    }
    else {
      continue;
    }
  }
  my_file.close();
}

void Utils::ReadOBJCurves(const std::string& file_name,
                          std::vector<Eigen::Vector3d>* points,
                          std::vector<std::pair<int, int>>* edges) {
  points->clear();
  edges->clear();
  std::ifstream my_file(file_name);
  while (!my_file.eof()) {
    std::string line_str;
    std::getline(my_file, line_str);
    if (line_str[0] == '#') {
      continue;
    }
    else if (line_str[0] == 'v') {
      std::istringstream iss(line_str);
      std::string tmp;
      double x, y, z;
      iss >> tmp >> x >> y >> z;
      points->emplace_back(x, y, z);
    }
    else if (line_str[0] == 'l') {
      std::istringstream iss(line_str);
      std::string tmp;
      int a, b;
      iss >> tmp >> a >> b;
      edges->emplace_back(a - 1, b - 1);
    }
    else {
      continue;
    }
  }
  my_file.close();
}

void Utils::SplineFitting(const std::vector<Eigen::Vector3d>& in_path,
                          double resample_step,
                          std::vector<Eigen::Vector3d>* out_path) {
  Eigen::MatrixXd points_mat(3, in_path.size());
  for (int i = 0; i < in_path.size(); i++) {
    points_mat(0, i) = in_path[i](0);
    points_mat(1, i) = in_path[i](1);
    points_mat(2, i) = in_path[i](2);
  }
  Eigen::Spline3d spline = Eigen::SplineFitting<Eigen::Spline3d>::Interpolate(points_mat, 1);
  Eigen::Spline3d::KnotVectorType chord_lengths; // knot parameters
  Eigen::ChordLengths(points_mat, chord_lengths);
  double min_chord_len = chord_lengths(0);
  double max_chord_len = chord_lengths(in_path.size() - 1);
  out_path->clear();
  for (float sample_len = min_chord_len; sample_len <= max_chord_len; sample_len += resample_step)
  {
    Eigen::Spline3d::PointType pt = spline(sample_len);
    out_path->emplace_back(pt(0), pt(1), pt(2));
  }
}

int Utils::QuadraticFitting(const std::vector<double> &xs,
                            const std::vector<double> &ys,
                            double *a,
                            double *b,
                            double *c) {
  // min(sum(ax^2 + bx + c - y)^2));
  CHECK_EQ(xs.size(), ys.size());
  CHECK(a != nullptr);
  CHECK(b != nullptr);
  CHECK(c != nullptr);
  const int n = xs.size();

  Eigen::MatrixXd A(n, 3);
  Eigen::VectorXd B(n);
  for (int i = 0; i < n; i++) {
    A(i, 0) = xs[i] * xs[i];
    A(i, 1) = xs[i];
    A(i, 2) = 1.0;
    B(i) = ys[i];
  }
  Eigen::VectorXd ans;
  ans = A.colPivHouseholderQr().solve(B);
  *a = ans(0);
  *b = ans(1);
  *c = ans(2);
  return 0;
}

int Utils::LinearFitting(const std::vector<double> &xs, const std::vector<double> &ys, double *a, double *b) {
  // min(sum(ax^2 + bx + c - y)^2));
  CHECK_EQ(xs.size(), ys.size());
  CHECK(a != nullptr);
  CHECK(b != nullptr);
  const int n = xs.size();

  Eigen::MatrixXd A(n, 2);
  Eigen::VectorXd B(n);
  for (int i = 0; i < n; i++) {
    A(i, 0) = xs[i];
    A(i, 1) = 1.0;
    B(i) = ys[i];
  }
  Eigen::VectorXd ans;
  ans = A.colPivHouseholderQr().solve(B);
  *a = ans(0);
  *b = ans(1);
  return 0;
}
