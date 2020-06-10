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

#include "KMeansClustering.h"

// STL
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>

// Submodules
// #include "Helpers/Helpers.h"

KMeansClustering::KMeansClustering()
{
}

void KMeansClustering::Cluster()
{
  if(this->Points.cols() < this->K)
  {
    std::stringstream ss;
    ss << "The number of points (" << this->Points.cols()
       << ") must be larger than the number of clusters (" << this->K << ")";
    throw std::runtime_error(ss.str());
  }

  // Seed a random number generator
  if(this->Random)
  {
    unsigned int t = time(NULL);
    srand48(t);
  }
  else
  {
    srand48(0);
  }

  if(this->InitMethod == RANDOM)
  {
    RandomInit();
  }
  else if(this->InitMethod == KMEANSPP) // http://en.wikipedia.org/wiki/K-means%2B%2B
  {
    KMeansPPInit();
  }
  else if(this->InitMethod == MANUAL)
  {
    // do nothing, the cluster centers should have been provided manually
  }
  else
  {
    throw std::runtime_error("An invalid initialization method has been specified!");
  }

  // We must store the labels at the previous iteration to determine whether any labels changed at each iteration.
  std::vector<unsigned int> oldLabels(this->Points.cols(), 0); // initialize to all zeros

  // Initialize the labels array
  this->Labels.resize(this->Points.cols());

  // The current iteration number
  int iter = 0;

  // Track whether any labels changed in the last iteration
  bool changed = true;
  do
  {
    AssignLabels();

    EstimateClusterCenters();

    changed = CheckChanged(this->Labels, oldLabels);

    // Save the old labels
    oldLabels = this->Labels;
    iter++;
  }while(changed);
  //}while(iter < 100); // You could use this stopping criteria to make kmeans run for a specified number of iterations

  // std::cout << "KMeans finished in " << iter << " iterations." << std::endl;
}

std::vector<unsigned int> KMeansClustering::GetIndicesWithLabel(const unsigned int label) const
{
  std::vector<unsigned int> pointsWithLabel;
  for(unsigned int i = 0; i < this->Labels.size(); i++)
  {
    if(this->Labels[i] == label)
    {
      pointsWithLabel.push_back(i);
    }
  }

  return pointsWithLabel;
}

Eigen::MatrixXd KMeansClustering::GetPointsWithLabel(const unsigned int label) const
{
  std::vector<unsigned int> indicesWithLabel = GetIndicesWithLabel(label);

  Eigen::MatrixXd pointsWithLabel(this->Points.rows(), indicesWithLabel.size());

  for(unsigned int i = 0; i < indicesWithLabel.size(); i++)
  {
    pointsWithLabel.col(i) = this->Points.col(indicesWithLabel[i]);
  }

  return pointsWithLabel;
}

unsigned int KMeansClustering::SelectWeightedIndex(const Eigen::VectorXd weights)
{
  // Ensure all weights are positive
  for(unsigned int i = 0; i < weights.size(); i++)
  {
    if(weights[i] < 0)
    {
      std::stringstream ss;
      ss << "weights[" << i << "] is " << weights[i] << " (must be positive!)";
      throw std::runtime_error(ss.str());
    }
  }

  //Helpers::Output(weights);

  // Sum
  double sum = weights.sum() + 1e-9;
  // std::cout << "sum: " << sum << std::endl;
  if(sum <= 0)
  {
    std::stringstream ss;
    ss << "Sum must be positive, but it is " << sum << "!";
    throw std::runtime_error(ss.str());
  }

  // Normalize
  Eigen::VectorXd normalizedWeights = weights.normalized();

  double randomValue = drand48();

  double runningTotal = 0.0;
  for(unsigned int i = 0; i < normalizedWeights.size(); i++)
  {
    runningTotal += normalizedWeights[i];
    if(randomValue < runningTotal)
    {
      return i;
    }
  }

  return normalizedWeights.size() - 1;
  std::cerr << "runningTotal: " << runningTotal << std::endl;
  std::cerr << "randomValue: " << randomValue << std::endl;
  throw std::runtime_error("KMeansClustering::SelectWeightedIndex() reached end, we should never get here.");

  return 0;
}

Eigen::VectorXd KMeansClustering::GetRandomPointInBounds()
{
  Eigen::VectorXd minVector = this->Points.rowwise().minCoeff();
  Eigen::VectorXd maxVector = this->Points.rowwise().maxCoeff();

  Eigen::VectorXd randomVector = Eigen::VectorXd::Zero(minVector.size());

  for(int i = 0; i < randomVector.size(); ++i)
  {
    float range = maxVector(i) - minVector(i);
    float randomValue = drand48() * range + minVector(i);
    randomVector(i) = randomValue;
  }

  return randomVector;
}

bool KMeansClustering::CheckChanged(const std::vector<unsigned int>& labels, const std::vector<unsigned int>& oldLabels)
{
  bool changed = false;
  for(unsigned int i = 0; i < labels.size(); i++)
  {
    if(labels[i] != oldLabels[i]) //if something changed
    {
      changed = true;
      break;
    }
  }
  return changed;
}

void KMeansClustering::AssignLabels()
{
  // Assign each point to the closest cluster
  for(unsigned int point = 0; point < this->Points.cols(); ++point)
  {
    unsigned int closestCluster = ClosestCluster(this->Points.col(point));
    this->Labels[point] = closestCluster;
  }
}

void KMeansClustering::EstimateClusterCenters()
{
  Eigen::MatrixXd oldCenters = this->ClusterCenters;

  for(unsigned int cluster = 0; cluster < this->K; ++cluster)
  {
    std::vector<unsigned int> indicesWithLabel = GetIndicesWithLabel(cluster);
    Eigen::MatrixXd classPoints(this->Points.rows(), indicesWithLabel.size());
    for(unsigned int point = 0; point < indicesWithLabel.size(); point++)
    {
      classPoints.col(point) = this->Points.col(indicesWithLabel[point]);
    }

    Eigen::VectorXd center;
    if(classPoints.cols() == 0)
    {
      center = oldCenters.col(cluster);
    }
    else
    {
      center = classPoints.rowwise().mean();
    }

    this->ClusterCenters.col(cluster) = center;
  }
}

unsigned int KMeansClustering::ClosestCluster(const Eigen::VectorXd& queryPoint)
{
  unsigned int closestCluster = 0;
  double minDist = std::numeric_limits<double>::max();
  for(unsigned int i = 0; i < this->ClusterCenters.cols(); ++i)
  {
    double dist = (this->ClusterCenters.col(i) - queryPoint).norm();
    if(dist < minDist)
    {
      minDist = dist;
      closestCluster = i;
    }
  }

  return closestCluster;
}

unsigned int KMeansClustering::ClosestPointIndex(const Eigen::VectorXd& queryPoint)
{
  unsigned int closestPoint = 0;
  double minDist = std::numeric_limits<double>::max();
  for(unsigned int i = 0; i < this->Points.cols(); i++)
  {
    //double dist = sqrt(vtkMath::Distance2BetweenPoints(points->GetPoint(i), queryPoint));
    double dist = (this->Points.col(i) - queryPoint).norm();
    if(dist < minDist)
    {
      minDist = dist;
      closestPoint = i;
    }
  }

  return closestPoint;
}

/*double KMeansClustering::ClosestPointDistanceExcludingId(const Eigen::VectorXd& queryPoint, const unsigned int excludedId)
{
  std::vector<unsigned int> excludedIds;
  excludedIds.push_back(excludedId);
  return ClosestPointDistanceExcludingIds(queryPoint, excludedIds);
}*/

/*double KMeansClustering::ClosestPointDistanceExcludingIds(const Eigen::VectorXd& queryPoint, const std::vector<unsigned int> excludedIds)
{
  double minDist = std::numeric_limits<double>::infinity();
  for(unsigned int pointId = 0; pointId < this->Points.cols(); ++pointId)
  {
    if(Helpers::Contains(excludedIds, pointId))
    {
      continue;
    }
    double dist = (this->Points.col(pointId) - queryPoint).norm();

    if(dist < minDist)
    {
      minDist = dist;
    }
  }
  return minDist;
}*/

/*double KMeansClustering::ClosestPointDistance(const Eigen::VectorXd& queryPoint)
{
  std::vector<unsigned int> excludedIds; // none
  return ClosestPointDistanceExcludingIds(queryPoint, excludedIds);
}*/

void KMeansClustering::RandomInit()
{
  this->ClusterCenters.resize(this->Points.rows(), this->K);

  // Completely randomly choose initial cluster centers
  for(unsigned int i = 0; i < this->K; i++)
  {
    Eigen::VectorXd p = GetRandomPointInBounds();

    this->ClusterCenters.col(i) = p;
  }
}

void KMeansClustering::KMeansPPInit()
{
  this->ClusterCenters.resize(this->Points.rows(), this->K);

  // Assign one center at random
  unsigned int randomId = rand() % this->Points.cols();
  Eigen::VectorXd p = this->Points.col(randomId);
  this->ClusterCenters.col(0) = p;

  // Assign the rest of the initial centers using a weighted probability of the distance to the nearest center
  Eigen::VectorXd weights(this->Points.cols());
  for(unsigned int cluster = 1; cluster < this->K; ++cluster) // Start at 1 because cluster 0 is already set
  {
    // Create weight vector
    for(unsigned int i = 0; i < this->Points.cols(); i++)
    {
      Eigen::VectorXd currentPoint = this->Points.col(i);
      unsigned int closestCluster = ClosestCluster(currentPoint);
      weights(i) = (this->ClusterCenters.col(closestCluster) - currentPoint).norm();
    }

    unsigned int selectedPointId = SelectWeightedIndex(weights);
    p = this->Points.col(selectedPointId);
    this->ClusterCenters.col(cluster) = p;
  }
}

void KMeansClustering::SetK(const unsigned int k)
{
  this->K = k;
}

unsigned int KMeansClustering::GetK()
{
  return this->K;
}

void KMeansClustering::SetRandom(const bool r)
{
  this->Random = r;
}

void KMeansClustering::SetInitMethod(const int method)
{
  this->InitMethod = method;
}

void KMeansClustering::SetPoints(const Eigen::MatrixXd& points)
{
  this->Points = points;
}

std::vector<unsigned int> KMeansClustering::GetLabels() const
{
  return this->Labels;
}

Eigen::MatrixXd KMeansClustering::GetClusterCenters() const
{
  return this->ClusterCenters;
}

void KMeansClustering::SetClusterCenters(const Eigen::MatrixXd& clusterCenters)
{
  this->ClusterCenters = clusterCenters;
}

float KMeansClustering::ComputeMLEVariance() const
{
  // \hat{\sigma}^2 = \frac{1}{n} \sum_{i=1}^n (x_i - \hat{x}_i)^2

  float variance = 0;
  for(unsigned int i = 0; i < this->K; ++i)
  {
    variance += ComputeMLEVariance(i);
  }

  return variance;
}

float KMeansClustering::ComputeMLEVariance(const unsigned int clusterId) const
{
  std::vector<unsigned int> indicesWithLabel = GetIndicesWithLabel(clusterId);

  if(indicesWithLabel.size() == 0)
  {
    return 0;
  }

  // \hat{\sigma}^2 = \frac{1}{n} \sum_{i=1}^n (x_i - \hat{x}_i)^2

  Eigen::VectorXd clusterCenter = this->ClusterCenters.col(clusterId);

  float variance = 0;
  for(unsigned int i = 0; i < indicesWithLabel.size(); ++i)
  {
    float error = (clusterCenter - this->Points.col(indicesWithLabel[i])).norm();
    variance += error;
  }

  variance /= static_cast<float>(indicesWithLabel.size());
  return variance;
}

float KMeansClustering::ComputeBIC() const
{
  // http://en.wikipedia.org/wiki/Bayesian_information_criterion
  // BIC = n ln(\hat{\sigma}^2) + k ln(n)
  // where n is the number of points, sigma is the MLE variance, and k is the number of free parameters, which in XMeans is
  // (K-1) + (M*K) + 1 (where K is the K in KMeans and M is the dimensionality of the points)
  float k = (this->K - 1.0f) + (this->Points.rows() * this->K) + 1.0f;

  float bic = this->Points.cols() * log(this->ComputeMLEVariance()) + k * log(this->Points.cols());
  return bic;
}

float KMeansClustering::ComputeBIC(const unsigned int clusterId) const
{
  // http://en.wikipedia.org/wiki/Bayesian_information_criterion
  // BIC = n ln(\hat{\sigma}^2) + k ln(n)
  // where n is the number of points, sigma is the MLE variance, and k is the number of free parameters, which in XMeans is
  // (K-1) + (M*K) + 1 (where K is the K in KMeans and M is the dimensionality of the points)

  std::vector<unsigned int> indicesWithlabel = GetIndicesWithLabel(clusterId);

  float k = (this->K - 1.0f) + (this->Points.rows() * this->K) + 1.0f;
  return static_cast<float>(indicesWithlabel.size()) * log(this->ComputeMLEVariance(clusterId)) + k *
                                                                                                  log(static_cast<float>(indicesWithlabel.size()));
}
