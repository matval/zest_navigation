#include <map>

#include <octomap_utils/trav_voxel_grid.h>

/**
 * Traversability voxel sampling method.
 * @param input_cloud: Input point cloud to be downsampled.
 * @param output_cloud: Output downsampled point cloud.
 * @param leaf_size: Downsampling voxel size.
 */
void traversabilityVoxelGrid(const PointCloud &input_cloud, PointCloud &output_cloud, float leaf_size) {
    // Create a hash map (unordered_map) to store the voxel data
    std::unordered_map<std::string, std::vector<PointT>> voxel_points;

    // Iterate through the input point cloud
    for (const PointT& point : input_cloud.points) {
        // Calculate voxel indices based on leaf size
        int voxel_x = static_cast<int>(point.x / leaf_size);
        int voxel_y = static_cast<int>(point.y / leaf_size);
        int voxel_z = static_cast<int>(point.z / leaf_size);

        // Create a unique voxel key as a string
        std::string voxel_key = std::to_string(voxel_x) + "_" + std::to_string(voxel_y) + "_" + std::to_string(voxel_z);

        // Add the point to the voxel
        voxel_points[voxel_key].push_back(point);
    }

    // Iterate through the voxel data to calculate averages
    for (const auto& kv : voxel_points) {
        const std::vector<PointT>& points_in_voxel = kv.second;
        std::string voxel_key = kv.first;

        // Calculate the average (x, y, z) coordinates, semantic_label, and instance_label within the voxel
        PointT average_point;
        average_point.x = 0.0;
        average_point.y = 0.0;
        average_point.z = 0.0;
        average_point.intensity = 0.0;
        average_point.traversability = 0.0;
        int total_points = points_in_voxel.size();

        for (const PointT& point : points_in_voxel) {
            average_point.x += point.x;
            average_point.y += point.y;
            average_point.z += point.z;
            average_point.intensity += point.intensity;
            average_point.traversability += point.traversability;
        }

        if (total_points > 0) {
            average_point.x /= total_points;
            average_point.y /= total_points;
            average_point.z /= total_points;
            average_point.intensity /= total_points;
            average_point.traversability /= total_points;
        }
        // Add the downsampled point to the output cloud
        output_cloud.push_back(average_point);
    }
}