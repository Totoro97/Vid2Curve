/*=========================================================================
 *
 *  Copyright David Doria 2012 daviddoria@gmail.com
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

/*
KMeans clustering is a method in which to form K (known) clusters of points from
an unorganized set of input points.
Data in this class is stored as an Eigen matrix, where the data points are column vectors.
That is, if we have P N-D points, the matrix is N rows by P columns.
Likewise, objects returned have the same structure (e.g. GetClusterCenters() returns an NxK matrix).
*/

#ifndef KMeansClustering_h
#define KMeansClustering_h

// STL
#include <vector>

// Eigen
#include <Eigen/Eigen>

class KMeansClustering
{
public:

  /** Constructor. */
  KMeansClustering();

  /** Set the number of clusters to find. */
  void SetK(const unsigned int k);

  /** Get the number of clusters to find. */
  unsigned int GetK();

  /** Get the cluster centers.*/
  Eigen::MatrixXd GetClusterCenters() const;

  /** Set the cluster centers.*/
  void SetClusterCenters(const Eigen::MatrixXd& clusterCenters);

  /** Get the point ids with a specified cluster membership. */
  std::vector<unsigned int> GetIndicesWithLabel(const unsigned int label) const;

  /** Get the points with a specified cluster membership. */
  Eigen::MatrixXd GetPointsWithLabel(const unsigned int label) const;

  /** If this function is called, the randomness is removed for repeatability for testing. */
  void SetRandom(const bool r);

  /** Set the points to cluster. */
  void SetPoints(const Eigen::MatrixXd& points);

  /** Get the cluster membership of every point. */
  std::vector<unsigned int> GetLabels() const;

  /** Set which initialization method to use. */
  void SetInitMethod(const int method);

  /** Choices of initialization methods */
  enum InitMethodEnum{RANDOM, KMEANSPP, MANUAL};

  /** Actually perform the clustering. */
  void Cluster();

  /** Compute the maximum likelihood estimate (MLE) of the variance.
      This assumes a spherical Gaussian model (as this is the basis of KMeans),
      which means the variance is a scalar, rather than a diagonal covariance matrix
      or a full covariance matrix.*/
  float ComputeMLEVariance() const;

  /** Compute the maximum likelihood estimate (MLE) of the variance of just one cluster. */
  float ComputeMLEVariance(const unsigned int clusterId) const;

  /** Compute the Bayesian Information Criterion as a measure of how well the model represents the data. */
  float ComputeBIC() const;

  /** Compute the Bayesian Information Criterion as a measure of how well the model represents the data. */
  float ComputeBIC(const unsigned int clusterId) const;

protected:

  /** Randomly initialize cluster centers. */
  void RandomInit();

  /** Initialize cluster centers using the KMeans++ algorithm. */
  void KMeansPPInit();

  /** Get the membership of 'queryPoint'. */
  unsigned int ClosestCluster(const Eigen::VectorXd& queryPoint);

  /** Get the id of the closest point to 'queryPoint'. */
  unsigned int ClosestPointIndex(const Eigen::VectorXd& queryPoint);

  /** Get the distance between 'queryPoint' and its closest point. */
  double ClosestPointDistance(const Eigen::VectorXd& queryPoint);

  /** Get the distance between 'queryPoint' and its closest point excluding 'excludedId'. */
  double ClosestPointDistanceExcludingId(const Eigen::VectorXd& queryPoint, const unsigned int excludedId);

  /** Get the distance between 'queryPoint' and its closest point excluding 'excludedIds'. */
  double ClosestPointDistanceExcludingIds(const Eigen::VectorXd& queryPoint, const std::vector<unsigned int> excludedIds);

  /** Based on the current cluster membership, compute the cluster centers. */
  void EstimateClusterCenters();

  /** Construct an array of the closest cluster center to each point. */
  void AssignLabels();

  /** Determine if the membership of any point has changed. */
  bool CheckChanged(const std::vector<unsigned int>& labels, const std::vector<unsigned int>& oldLabels);

  /** Get a random point inside the bounding box of the points. */
  Eigen::VectorXd GetRandomPointInBounds();

  /** Select a random index, with the probability of choosing an index weighted by the 'weights' vector. */
  unsigned int SelectWeightedIndex(const Eigen::VectorXd weights);

private:

  /** The label (cluster membership) of each point. */
  std::vector<unsigned int> Labels;

  /** Should the computation be random? If false, then it is repeatable (for testing). */
  bool Random = true;

  /** The initialization method to use. */
  int InitMethod = RANDOM;

  /** The number of clusters to find. */
  unsigned int K = 3;

  /** The points to cluster. */
  Eigen::MatrixXd Points;

  /** The current cluster centers. */
  Eigen::MatrixXd ClusterCenters;
};

#endif
