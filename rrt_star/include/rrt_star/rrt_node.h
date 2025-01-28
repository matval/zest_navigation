#ifndef RRT_NODE_H
#define RRT_NODE_H

#include <ros/ros.h>
#include <grid_map_ros/grid_map_ros.hpp>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <grid_map_msgs/GridMap.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <vector>

// OMPL includes
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/ProblemDefinition.h>
#include <ompl/base/spaces/SE2StateSpace.h>
#include <ompl/base/Planner.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>
#include <ompl/util/RandomNumbers.h>

namespace ob = ompl::base;
namespace og = ompl::geometric;

/**
 * @class RRTNode
 * @brief A ROS node for performing path planning using the RRT* algorithm integrated with OMPL.
 */
class RRTNode
{
public:
    /**
     * @brief Constructor for the RRTNode class.
     *        Initializes the ROS node, sets up subscribers and publishers, and initializes the OMPL planner.
     */
    RRTNode();

private:
    /**
     * @brief Main loop function for the RRTNode.
     *        Continuously processes incoming messages and performs path planning.
     */
    void spin();

    /**
     * @brief Initializes the OMPL planner.
     */
    void setupOMPL();

    /**
     * @brief Checks if a given state is valid in the environment (for obstacle/traversability).
     * @param state The OMPL state to check.
     * @return True if the state is valid, false otherwise.
     */
    bool isStateValid(const ob::State *state);

    /**
     * @brief Adds a branch to the RRT tree visualization marker using OMPL's planner data.
     * @param rrtTreeMarker The RRT tree marker.
     */
    void addBranchtoRRTTree(visualization_msgs::Marker &rrtTreeMarker);

    /**
     * @brief Sets the final path data for visualization using OMPL's PathGeometric.
     * @param path The OMPL geometric path to visualize.
     * @param finalPath The final path marker.
     */
    void setFinalPathMarker(const og::PathGeometric &path, visualization_msgs::Marker &finalPath);

    /**
     * @brief Publishes the planned path using OMPL's PathGeometric.
     * @param path The OMPL geometric path.
     */
    void publishPath(const og::PathGeometric &path);

    /**
     * @brief Callback function for the GridMap subscriber.
     *        Processes incoming GridMap messages, extracts traversability data, updates the environment.
     * 
     * @param msg Pointer to the incoming GridMap message.
     */
    void gridMapCallback(const grid_map_msgs::GridMap::ConstPtr& msg);

    /**
     * @brief Updates the boundary values based on the received GridMap data.
     */
    void updateBoundsFromGridMap();

    /**
     * @brief Callback function for the goal subscriber.
     *        Processes incoming goal messages and triggers the path planning process.
     * 
     * @param msg Pointer to the incoming goal message.
     */
    void goalCallback(const geometry_msgs::PoseStamped::ConstPtr& msg); 

    /**
     * @brief Callback function for the Odometry subscriber.
     *        Processes incoming Odometry messages, extracts the robot's current state.
     * 
     * @param msg Pointer to the incoming Odometry message.
     */
    void odomCallback(const nav_msgs::Odometry::ConstPtr& msg);

    /**
     * @brief Initializes the visualization markers for the RRT path planning process.
     * @param sourcePoint Marker for the start point.
     * @param goalPoint Marker for the goal point.
     * @param randomPoint Marker for the random point sampled by the RRT algorithm.
     * @param rrtTreeMarker Marker for the RRT tree.
     * @param finalPath Marker for the final planned path.
     */
    void initializeMarkers(
        visualization_msgs::Marker &sourcePoint,
        visualization_msgs::Marker &goalPoint,
        visualization_msgs::Marker &rrtTreeMarker,
        visualization_msgs::Marker &finalPath);

    ros::NodeHandle nh_;                       ///< ROS node handle for communication with the ROS system.

    ros::Publisher path_pub_;                  ///< Publisher for the planned path (nav_msgs::Path).
    ros::Publisher rrt_pub_;                   ///< Publisher for the RRT tree visualization (visualization_msgs::Marker).
    ros::Subscriber grid_map_sub_;             ///< Subscriber for receiving traversability data (grid_map_msgs::GridMap).
    ros::Subscriber odom_sub_;                 ///< Subscriber for receiving the robot's odometry data (nav_msgs::Odometry).
    ros::Subscriber goal_sub_;                 ///< Subscriber for receiving the goal position (geometry_msgs::PoseStamped).

    // Parameters for RRT path planning
    double start_x_;                           ///< X coordinate of the start position for the path.
    double start_y_;                           ///< Y coordinate of the start position for the path.
    double goal_x_;                            ///< X coordinate of the goal position for the path.
    double goal_y_;                            ///< Y coordinate of the goal position for the path.
    double min_x_;                             ///< Minimum X boundary for the RRT algorithm.
    double min_y_;                             ///< Minimum Y boundary for the RRT algorithm.
    double max_x_;                             ///< Maximum X boundary for the RRT algorithm.
    double max_y_;                             ///< Maximum Y boundary for the RRT algorithm.

    // Parameters for ROS topics and coordinate frames
    std::string grid_map_topic_;               ///< ROS topic name for receiving grid map data.
    std::string path_topic_;                   ///< ROS topic name for publishing the planned path.
    std::string frame_id_;                     ///< The coordinate frame ID for the published path.

    std::vector<double> robot_states_;         ///< The current state of the robot. [x, y, theta]
    grid_map::GridMap grid_map_;               ///< GridMap object representing the traversability map.
    bool map_received_;                        ///< Flag indicating whether a grid map has been received.
    bool states_received_;                     ///< Flag indicating whether the start position has been received.
    bool goal_received_;                       ///< Flag indicating whether the goal position has been received.
    double alpha1_;                            ///< Weight for the traversability term in the RRT* cost function.
    double alpha2_;                            ///< Weight for the lambda term in the RRT* cost function.
    double traversability_threshold_;          ///< Threshold value for traversability.
    double robot_length_;                      ///< Length of the robot.
    double robot_width_;                       ///< Width of the robot.
    int num_samples_;                          ///< Number of samples to take along the motion.
    double planning_frequency_;                ///< Frequency for triggering the planning process.
    double solver_time_;                       ///< Time limit for the planner to solve the problem.

    // OMPL-specific members
    std::shared_ptr<og::SimpleSetup> omplSimpleSetup_;  ///< OMPL SimpleSetup for planning with RRT*.
};

#endif // RRT_NODE_H
