#ifndef SEMANTIC_VOXEL_GRID_H
#define SEMANTIC_VOXEL_GRID_H

#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <unordered_map>
#include <vector>

// Custom libraries
#include <octomap_utils/traversability_point.h>

// Define a structure to represent a point cloud
typedef TraversabilityPoint PointT;
typedef pcl::PointCloud<PointT> PointCloud;

/**
 * Traversability voxel sampling method.
 * @param input_cloud: Input point cloud to be downsampled.
 * @param output_cloud: Output downsampled point cloud.
 * @param leaf_size: Downsampling voxel size.
 */
void traversabilityVoxelGrid(const PointCloud &input_cloud, PointCloud &output_cloud, float leaf_size);

#endif