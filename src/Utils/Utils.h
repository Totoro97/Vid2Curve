//
// Created by aska on 19-2-1.
//

#pragma once
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

#include <string>
#include <vector>
#include <fstream>

namespace Utils {
  void SavePointsAsPly(std::string save_path, const std::vector<Eigen::Vector3d> &points);
  void SolveLinearSqrLSQR(int rows,
                          int cols,
                          const std::vector<uint32_t>& coord_r,
                          const std::vector<uint32_t>& coord_c,
                          const std::vector<double>& values,
                          const std::vector<double>& B,
                          std::vector<double> &solution);
  void SolveLinearSqrStable(int rows,
                            int cols,
                            const std::vector<uint32_t>& coord_r,
                            const std::vector<uint32_t>& coord_c,
                            const std::vector<double>& values,
                            const std::vector<double>& B,
                            std::vector<double>& solution);
  int QuadraticFitting(const std::vector<double>& xs, const std::vector<double>& ys, double *a, double *b, double* c);
  int LinearFitting(const std::vector<double>& xs, const std::vector<double> &ys, double *a, double *b);
  void SplineFitting(const std::vector<Eigen::Vector3d>& in_path,
                     double resample_step,
                     std::vector<Eigen::Vector3d>* out_path);
  void GetOpticalFlow(const cv::Mat &img_0, const cv::Mat &img_1, cv::Mat* flow_img, bool blur = true);
  Eigen::Matrix3d GetLeftCrossProdMatrix(const Eigen::Vector3d &X);
  Eigen::Matrix3d GetRightCrossProdMatrix(const Eigen::Vector3d &X);
  void OutputCurves(std::string save_path,
                    const std::vector<std::vector<Eigen::Vector3d> > &curves);
  void OutputCurvesAsOBJ(const std::string& save_path,
                         const std::vector<Eigen::Vector3d>& points,
                         const std::vector<std::pair<int, int>>& edges);
  void OutputTriMeshAsOBJ(const std::string& save_path,
                          const Eigen::MatrixXd& verts,
                          const Eigen::MatrixXi& faces); //nenglun
  void OutputTriMeshAsOBJ(const std::string& save_path,
                          const std::vector<Eigen::Vector3d>& points,
                          const std::vector<std::tuple<int, int, int>>& faces);
  void MergeTriMeshes(std::vector<Eigen::MatrixXd>& in_verts_list,
                      std::vector<Eigen::MatrixXi>& in_faces_list,
                      Eigen::MatrixXd& res_verts,
                      Eigen::MatrixXi& res_faces);//nenglun

  void ReadOBJMesh(const std::string& file_name,
                   std::vector<Eigen::Vector3d>* points,
                   std::vector<std::tuple<int, int, int>>* faces);
  void ReadOBJCurves(const std::string& file_name,
                          std::vector<Eigen::Vector3d>* points,
                          std::vector<std::pair<int, int>>* edges);
  
Eigen::Vector3d ImageCoordToCamera(const Eigen::Vector2d& x, int height, int width, double focal_length);
  // double CompareTwoPointCloud(const std::vector<Eigen::Vector3d>& A, const std::vector<Eigen::Vector3d>& B);
  std::pair<Eigen::Vector3d, Eigen::Vector3d> GetAABB(const std::vector<Eigen::Vector3d>& points);
  std::pair<Eigen::Vector2d, Eigen::Vector2d> GetAABB(const std::vector<Eigen::Vector2d>& points);
}
