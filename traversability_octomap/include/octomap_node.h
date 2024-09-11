#ifndef TRAVERSABILITY_NODE_H
#define TRAVERSABILITY_NODE_H

#include <ros/ros.h>
#include <tf2_ros/buffer.h>
#include <pcl_ros/transforms.h>
#include <tf2_ros/transform_listener.h>

#include <grid_map_ros/grid_map_ros.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <boost/math/interpolators/cubic_b_spline.hpp>

#include "bkioctomap.h"
#include "markerarray_pub.h"

#include <grid_map_msgs/GridMap.h>
#include <geometry_msgs/TransformStamped.h>

namespace std {
    template <>
    struct hash<grid_map::Index> {
        std::size_t operator()(const grid_map::Index& k) const {
            return std::hash<int>()(k.x()) ^ (std::hash<int>()(k.y()) << 1);
        }
    };

    template <>
    struct equal_to<grid_map::Index> {
        bool operator()(const grid_map::Index& lhs, const grid_map::Index& rhs) const {
            return lhs.x() == rhs.x() && lhs.y() == rhs.y();
        }
    };
}

class OctomapNode
{
public:
    OctomapNode();
    void spin();

private:
    void initParameters();
    void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud);
    void publishGridMap();
    double calculateCVaR(double mu, double sigma, double nu, double alpha);

    // Node handle
    ros::NodeHandle nh_;

    // Transform listener
    tf2_ros::Buffer tfBuffer_;
    tf2_ros::TransformListener tfListener_;

    // Publishers and subscribers
    ros::Publisher grid_map_pub_;
    ros::Subscriber point_cloud_sub_;

    // Map and publisher pointers
    std::unique_ptr<traversability_bki::TraversabilityBKIOctoMap> map_;
    std::unique_ptr<traversability_bki::MarkerArrayPub> marker_pub_;

    // Parameters
    float resolution_ = 0.1;
    int block_depth_ = 4;
    double sf2_ = 1.0;
    double ell_ = 1.0;
    float mean_prior_ = 0.5;
    float lambda_prior_ = 4.0;
    float beta_prior_ = 0.5;
    float prior_A_ = 1.0;
    float prior_B_ = 1.0;
    float var_thresh_ = 1.0;
    double free_thresh_ = 0.3;
    double occupied_thresh_ = 0.7;
    double ds_resolution_ = 0.2;
    double free_resolution_ = 0.5;
    double min_range_ = 0.0;
    double max_range_ = -1;
    float min_z_ = -3.0;
    float max_z_ = 3.0;
    float min_z_grid_ = -1.0;
    float max_z_grid_ = 1.0;
    float confidence_level_ = 0.95;     // Confidence level for the CVaR computation
    float nu_threshold_ = 40.0;         // Degrees of freedom for the Student's t-distribution
    std::string frame_id_;
    std::string pointcloud_topic_;
    std::string map_topic_;
    std::string grid_map_topic_;
    boost::math::cubic_b_spline<double> quantile_interp_;
    boost::math::cubic_b_spline<double> pdf_interp_;

    // Transformation and state variables
    tf2::Vector3 last_position_;
    tf2::Quaternion last_orientation_;
    bool first_ = true;
    double position_change_thresh_ = 0.1;
    double orientation_change_thresh_ = 0.2;
    bool updated_ = false;
};

#endif // TRAVERSABILITY_NODE_H