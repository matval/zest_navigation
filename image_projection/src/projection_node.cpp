#include <iostream>
#include <math.h>
#include <chrono> 
#include <armadillo>

#include <ros/ros.h>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/PointCloud2.h>

#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/range_image/range_image.h>
#include <pcl/range_image/range_image_spherical.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/filters/statistical_outlier_removal.h>

#include "octomap_utils/traversability_point.h"

// Using namespaces for convenience
using namespace Eigen;
using namespace sensor_msgs;
using namespace message_filters;
using namespace std;

// Typedef for point cloud type
typedef pcl::PointCloud<pcl::PointXYZI> PointCloud;

// Publisher for output point cloud
ros::Publisher pc_pub;

// Parameters for LIDAR and camera
double maxlen = 100.0;       // Maximum distance of the LIDAR
double minlen = 0.01;        // Minimum distance of the LIDAR
double max_FOV = 3.0;        // Max Field of View of the camera in radians
double min_FOV = 0.4;        // Min Field of View of the camera in radians

// Parameters for converting point cloud to image
double angular_resolution_x = 0.5f;
double angular_resolution_y = 2.1f;
double max_angle_width = 360.0f;
double max_angle_height = 180.0f;
double z_max = 100.0f;
double z_min = 100.0f;

double max_depth = 100.0;
double min_depth = 8.0;
double max_var = 50.0; 

double interpol_value = 20.0;

// Flag to enable or disable point cloud filtering
bool f_pc = true; 

// Input topics for image and point cloud
std::string imgTopic = "/camera/color/image_raw";
std::string pcTopic = "/velodyne_points";

// Matrices for LIDAR to camera calibration and camera intrinsics
// Eigen::MatrixXf Tlc(3,1); // Translation matrix lidar-camera
// Eigen::MatrixXf Rlc(3,3); // Rotation matrix lidar-camera
Eigen::MatrixXf Mc(3,4);  // Camera calibration matrix

// Parameters for range image
boost::shared_ptr<pcl::RangeImageSpherical> rangeImage;
pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::LASER_FRAME;

tf2_ros::Buffer tfBuffer(ros::Duration(100));

void callback(const boost::shared_ptr<const sensor_msgs::PointCloud2>& in_pc2, const sensor_msgs::CompressedImageConstPtr& in_image){
    /*
        @brief Callback function to process the incoming point cloud and image messages
        @param in_pc2: Incoming point cloud message
        @param in_image: Incoming image message
    */
    cv_bridge::CvImagePtr cv_ptr;
    try{
        cv_ptr = cv_bridge::toCvCopy(in_image, sensor_msgs::image_encodings::MONO8);
    }
    catch (cv_bridge::Exception& e){
        ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
    }

    // Convert the ROS point cloud message to PCL format
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(*in_pc2, pcl_pc2);
    PointCloud::Ptr msg_pointCloud(new PointCloud);
    pcl::fromPCLPointCloud2(pcl_pc2, *msg_pointCloud);

    // Check if the point cloud is valid
    if (msg_pointCloud == NULL) return;

    PointCloud::Ptr cloud_in(new PointCloud);
    PointCloud::Ptr cloud_out(new PointCloud);

    // Remove NaN points from the point cloud
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*msg_pointCloud, *cloud_in, indices);

    // Filter points by distance
    for (int i = 0; i < (int)cloud_in->points.size(); i++){
        double distance = sqrt(cloud_in->points[i].x * cloud_in->points[i].x + cloud_in->points[i].y * cloud_in->points[i].y);     
        if (distance < minlen || distance > maxlen)
            continue;        
        cloud_out->push_back(cloud_in->points[i]);     
    }  

    // Convert point cloud to range image
    Eigen::Affine3f sensorPose = (Eigen::Affine3f)Eigen::Translation3f(0.0f, 0.0f, 0.0f);
    rangeImage->pcl::RangeImage::createFromPointCloud(*cloud_out, pcl::deg2rad(angular_resolution_x), pcl::deg2rad(angular_resolution_y),
                                                        pcl::deg2rad(max_angle_width), pcl::deg2rad(max_angle_height),
                                                        sensorPose, coordinate_frame, 0.0f, 0.0f, 0);

    int cols_img = rangeImage->width;
    int rows_img = rangeImage->height;

    // Matrices for interpolation
    arma::mat Z;  // Interpolation of the range image
    arma::mat Zz; // Interpolation of heights in the image

    Z.zeros(rows_img, cols_img);         
    Zz.zeros(rows_img, cols_img);       

    Eigen::MatrixXf ZZei(rows_img, cols_img);

    // Populate the range and height matrices
    for (int i = 0; i < cols_img; ++i) {
        for (int j = 0; j < rows_img; ++j) {
            float r = rangeImage->getPoint(i, j).range;     
            float zz = rangeImage->getPoint(i, j).z; 

            if (std::isinf(r) || r < minlen || r > maxlen || std::isnan(zz))
                continue;
            Z.at(j, i) = r;   
            Zz.at(j, i) = zz;
        }
    }

    // Interpolation to create a smoother image
    arma::vec X = arma::regspace(1, Z.n_cols);  // X = horizontal spacing
    arma::vec Y = arma::regspace(1, Z.n_rows);  // Y = vertical spacing 
    arma::vec XI = arma::regspace(X.min(), 1.0, X.max()); // Magnify by approx 2
    arma::vec YI = arma::regspace(Y.min(), 1.0 / interpol_value, Y.max()); 
    arma::mat ZI_near;  
    arma::mat ZI;
    arma::mat ZzI;

    // Perform linear interpolation on the range and height data
    arma::interp2(X, Y, Z, XI, YI, ZI, "lineal");  
    arma::interp2(X, Y, Zz, XI, YI, ZzI, "lineal");  

    // Create a new point cloud from the interpolated range image
    PointCloud::Ptr point_cloud(new PointCloud);
    PointCloud::Ptr cloud(new PointCloud);
    point_cloud->width = ZI.n_cols; 
    point_cloud->height = ZI.n_rows;
    point_cloud->is_dense = false;
    point_cloud->points.resize(point_cloud->width * point_cloud->height);

    arma::mat Zout = ZI;

    // Filter interpolated elements based on variance and create a new range image
    for (uint i = 0; i < ZI.n_rows; i += 1) {       
        for (uint j = 0; j < ZI.n_cols; j += 1) {             
            if (ZI(i, j) == 0){
                if (i + interpol_value < ZI.n_rows) {
                    for (int k = 1; k <= interpol_value; k += 1) {
                        Zout(i + k, j) = 0;
                    }
                }

                if (i > interpol_value) {
                    for (int k = 1; k <= interpol_value; k += 1) {
                        Zout(i - k, j) = 0;
                    }
                }
            }
        }      
    }

    if (f_pc) {        
        // Filter by variance
        for (uint i = 0; i < ((ZI.n_rows - 1) / interpol_value); i += 1)       
            for (uint j = 0; j < ZI.n_cols - 5; j += 1) {
                double average = 0;
                double variance = 0;
                for (uint k = 0; k < interpol_value; k += 1)
                average += ZI((i * interpol_value) + k, j);
                average /= interpol_value;    
                for (uint l = 0; l < interpol_value; l++) 
                variance += pow((ZI((i * interpol_value) + l, j) - average), 2.0);  
                if (variance > max_var)
                for (uint m = 0; m < interpol_value; m++) 
                    Zout((i * interpol_value) + m, j) = 0;                 
            }
            
        ZI = Zout;
    }

    // Convert the range image back to a 3D point cloud
    int num_pc = 0; 
    for (uint i = 0; i < ZI.n_rows - interpol_value; i += 1) {       
        for (uint j = 0; j < ZI.n_cols; j += 1) {
            float ang = M_PI - ((2.0 * M_PI * j) / (ZI.n_cols));

            if (ang < min_FOV - M_PI / 2.0 || ang > max_FOV - M_PI / 2.0) 
                continue;

            if (!(Zout(i, j) == 0)) {  
                float pc_module = Zout(i, j);
                float pc_x = sqrt(pow(pc_module, 2) - pow(ZzI(i, j), 2)) * cos(ang);
                float pc_y = sqrt(pow(pc_module, 2) - pow(ZzI(i, j), 2)) * sin(ang);

                float ang_x_lidar = 0.6 * M_PI / 180.0;  

                // Transformation matrix between lidar and range image
                Eigen::MatrixXf Lidar_matrix(3, 3); 
                Eigen::MatrixXf result(3, 1);
                Lidar_matrix << cos(ang_x_lidar), 0, sin(ang_x_lidar),
                                0, 1, 0,
                                -sin(ang_x_lidar), 0, cos(ang_x_lidar);

                result << pc_x,
                        pc_y,
                        ZzI(i, j);
                
                result = Lidar_matrix * result;  // Rotation for correction

                point_cloud->points[num_pc].x = result(0);
                point_cloud->points[num_pc].y = result(1);
                point_cloud->points[num_pc].z = result(2);

                cloud->push_back(point_cloud->points[num_pc]); 

                num_pc++;
            }
        }
    }  

    // Prepare the point cloud for publishing
    PointCloud::Ptr P_out(new PointCloud);
    P_out = cloud;
    
    geometry_msgs::TransformStamped transformStamped;
    try {
        transformStamped = tfBuffer.lookupTransform(in_image->header.frame_id, in_pc2->header.frame_id, in_pc2->header.stamp, ros::Duration(5));
    }
    catch (tf2::TransformException &ex) {
        ROS_WARN("Projection transform failed: %s", ex.what());
        return;
    }

    // Extract rotation and translation
    Eigen::Quaternionf q(
        transformStamped.transform.rotation.w,
        transformStamped.transform.rotation.x, 
        transformStamped.transform.rotation.y,
        transformStamped.transform.rotation.z
    );

    Eigen::MatrixXf Rlc(3, 3);
    Rlc = q.toRotationMatrix();
    
    Eigen::MatrixXf Tlc(3,1);
    Tlc << transformStamped.transform.translation.x,
           transformStamped.transform.translation.y, 
           transformStamped.transform.translation.z;

    // Transformation matrix from LIDAR to camera
    Eigen::MatrixXf RTlc(4, 4);
    RTlc = Eigen::MatrixXf::Identity(4, 4);
    RTlc.block(0, 0, 3, 3) = Rlc;
    RTlc.block(0, 3, 3, 1) = Tlc;

    int size_inter_Lidar = (int)P_out->points.size();

    Eigen::MatrixXf Lidar_camera(3, size_inter_Lidar);
    Eigen::MatrixXf Lidar_cam(3, 1);
    Eigen::MatrixXf pc_matrix(4, 1);
    Eigen::MatrixXf pointCloud_matrix(4, size_inter_Lidar);

    unsigned int cols = cv_ptr->image.cols;
    unsigned int rows = cv_ptr->image.rows;

    // Create a point cloud with traversability data
    pcl::PointCloud<TraversabilityPoint>::Ptr pc_trav(new pcl::PointCloud<TraversabilityPoint>);
    for (int i = 0; i < size_inter_Lidar; i++) {
        pc_matrix(0, 0) = P_out->points[i].x;
        pc_matrix(1, 0) = P_out->points[i].y;
        pc_matrix(2, 0) = P_out->points[i].z;
        pc_matrix(3, 0) = 1.0;

        Lidar_cam = Mc * (RTlc * pc_matrix);

        uint px_data = (int) Lidar_cam(0, 0) / Lidar_cam(2, 0);
        uint py_data = (int) Lidar_cam(1, 0) / Lidar_cam(2, 0);
        
        if (px_data < 0.0 || px_data >= cols || py_data < 0.0 || py_data >= rows)
            continue;

        uint8_t trav = cv_ptr->image.at<uint8_t>(py_data, px_data);

        TraversabilityPoint point;
        point.x = P_out->points[i].x;
        point.y = P_out->points[i].y;
        point.z = P_out->points[i].z;
        point.intensity = P_out->points[i].intensity;
        point.traversability = static_cast<float>(trav) / 255.0;
        
        pc_trav->points.push_back(point);       
    }

    // Convert the traversability point cloud to ROS message
    // PointCloud2 out_pc2;
    // pcl::toROSMsg(*pc_trav, out_pc2);
    // out_pc2.header = in_pc2->header;
    pc_trav->is_dense = true;
    pc_trav->width = static_cast<uint32_t>(pc_trav->points.size());
    pc_trav->height = 1;
    pc_trav->header.frame_id = in_pc2->header.frame_id;
    pc_trav->header.stamp = in_pc2->header.stamp.toNSec() / 1e3;

    // Publish the traversability point cloud
    // pc_pub.publish(out_pc2);
    pc_pub.publish(pc_trav);
}

int main(int argc, char** argv) {
    // Initialize ROS node
    ros::init(argc, argv, "projection_node");
    ros::NodeHandle nh("~");

    // Load parameters from ROS parameter server
    nh.param<double>("maxlen", maxlen, maxlen);
    nh.param<double>("minlen", minlen, minlen);
    nh.param<double>("max_ang_FOV", max_FOV, max_FOV);
    nh.param<double>("min_ang_FOV", min_FOV, min_FOV);
    nh.param<double>("max_var", max_var, max_var);
    nh.param<double>("x_resolution", angular_resolution_x, angular_resolution_x);
    nh.param<double>("y_interpolation", interpol_value, interpol_value);
    nh.param<double>("ang_Y_resolution", angular_resolution_y, angular_resolution_y);
    nh.param<bool>("filter_output_pc", f_pc, f_pc);
    
    nh.param<std::string>("image_topic", imgTopic, imgTopic);
    nh.param<std::string>("pointcloud_topic", pcTopic, pcTopic);
    
    XmlRpc::XmlRpcValue param;

    // nh.getParam("tlc", param);
    // Tlc << (double) param[0],
    //        (double) param[1],
    //        (double) param[2];

    // nh.getParam("rlc", param);

    // Rlc << (double)param[0], (double)param[1], (double)param[2],
    //        (double)param[3], (double)param[4], (double)param[5],
    //        (double)param[6], (double)param[7], (double)param[8];

    nh.getParam("camera_matrix", param);

    Mc << (double)param[0], (double)param[1], (double)param[2], (double)param[3],
          (double)param[4], (double)param[5], (double)param[6], (double)param[7],
          (double)param[8], (double)param[9], (double)param[10], (double)param[11];

    // Subscribers for point cloud and image topics
    message_filters::Subscriber<PointCloud2> pc_sub(nh, pcTopic, 100);
    message_filters::Subscriber<CompressedImage> img_sub(nh, imgTopic, 100);

    // Synchronize the two subscribers
    typedef sync_policies::ApproximateTime<PointCloud2, CompressedImage> MySyncPolicy;
    Synchronizer<MySyncPolicy> sync(MySyncPolicy(100), pc_sub, img_sub);
    sync.registerCallback(boost::bind(&callback, _1, _2));
    rangeImage = boost::shared_ptr<pcl::RangeImageSpherical>(new pcl::RangeImageSpherical);

    tf2_ros::TransformListener tfListener(tfBuffer);

    // Publisher for the output point cloud
    // pc_pub = nh.advertise<PointCloud2>("traversability_cloud", 1);
    pc_pub = nh.advertise<PointCloud>("traversability_cloud", 1);

    // Start ROS event loop
    ros::spin();
}
