#include <tf/tf.h> 
#include <geometry_msgs/PoseStamped.h>
#include <grid_map_ros/GridMapRosConverter.hpp>

#include <rrt_star/traversability_objective.h>
#include <rrt_star/rrt_node.h>

RRTNode::RRTNode() :
    nh_("~"), map_received_(false), states_received_(false), goal_received_(false),
    min_x_(0), max_x_(0), min_y_(0), max_y_(0)  
{
    std::string odom_topic, grid_map_topic, goal_topic, path_topic, planner_viz_topic;
    // Initialize parameters from the ROS parameter server
    nh_.param("odom_topic", odom_topic, std::string("/odom"));
    nh_.param("grid_map_topic", grid_map_topic, std::string("/traversability_map"));
    nh_.param("goal_topic", goal_topic, std::string("/move_base_simple/goal"));
    nh_.param("path_topic", path_topic, std::string("planned_path"));
    nh_.param("planner_viz_topic", planner_viz_topic, std::string("rrt_viz"));
    nh_.param("frame_id", frame_id_, std::string("map"));
    nh_.param("alpha1", alpha1_, 1.0);    
    nh_.param("alpha2", alpha2_, 1.0);           
    nh_.param("robot_length", robot_length_, 0.5); 
    nh_.param("robot_width", robot_width_, 0.3);   
    nh_.param("num_samples", num_samples_, 10);
    nh_.param("planning_frequency", planning_frequency_, 0.5);
    nh_.param("solver_time", solver_time_, 1.0);

    std::cout << "Odom topic: " << odom_topic << std::endl;
    std::cout << "Grid map topic: " << grid_map_topic << std::endl;

    // Set up the subscriber for the GridMap topic
    grid_map_sub_ = nh_.subscribe(grid_map_topic, 1, &RRTNode::gridMapCallback, this);
    odom_sub_ = nh_.subscribe(odom_topic, 1, &RRTNode::odomCallback, this);
    goal_sub_ = nh_.subscribe(goal_topic, 1, &RRTNode::goalCallback, this);

    // Set up the publishers for the planned path
    path_pub_ = nh_.advertise<nav_msgs::Path>(path_topic, 1);
	rrt_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(planner_viz_topic, 1);
    
    ROS_INFO("RRT Node initialized successfully.");

    // Wait for the GridMap data
    ROS_INFO("Waiting for GridMap, Odom, and Goal data...");
    while (ros::ok() && (!map_received_ || !states_received_ || !goal_received_)) {
        ros::Duration(1).sleep();
        ros::spinOnce();
    }
    ROS_INFO("GridMap, Odom, and Goal data received.");

    // Set up OMPL after receiving all necessary data
    setupOMPL();  

    // Start the main loop
    spin();
}

void RRTNode::setupOMPL() {
    ROS_INFO("Setting up OMPL with bounds: min_x = %f, max_x = %f, min_y = %f, max_y = %f", min_x_, max_x_, min_y_, max_y_);

    if (min_x_ >= max_x_ || min_y_ >= max_y_) {
        ROS_ERROR("Invalid bounds: min_x >= max_x or min_y >= max_y. OMPL cannot proceed.");
        throw std::runtime_error("Invalid bounds for state space.");
    }

    // Create the SE2 state space
    auto space = std::make_shared<ob::SE2StateSpace>();

    // Define the bounds for the state space
    ob::RealVectorBounds bounds(2);
    bounds.setLow(0, min_x_);
    bounds.setLow(1, min_y_);
    bounds.setHigh(0, max_x_);
    bounds.setHigh(1, max_y_);
    space->setBounds(bounds);

    // Create the SimpleSetup object and set the sampler
    omplSimpleSetup_ = std::make_shared<og::SimpleSetup>(space);

    // Set the state validity checker
    omplSimpleSetup_->setStateValidityChecker([this](const ob::State *state) {
        return this->isStateValid(state);
    });

    // Set RRT* as the planner
    auto planner = std::make_shared<og::RRTstar>(omplSimpleSetup_->getSpaceInformation());
    planner->setRange(6.0);  
    planner->setRewireFactor(3.0); 

    omplSimpleSetup_->setPlanner(planner);

    // OMPL static setup complete, but start and goal are dynamic
    ROS_INFO("OMPL setup complete.");
}

bool RRTNode::isStateValid(const ob::State* state) {
    const ob::SE2StateSpace::StateType* se2state = state->as<ob::SE2StateSpace::StateType>();
    double x = se2state->getX();
    double y = se2state->getY();

    if (x < min_x_ || x > max_x_ || y < min_y_ || y > max_y_) {
        ROS_WARN("State (x = %f, y = %f) is out of bounds!", x, y);
        return false;
    }

    return true;
}

void RRTNode::spin() {
    ros::Rate rate(planning_frequency_); 

    // ros::Time last_planning_time = ros::Time::now();

    while (ros::ok()) {
        ros::spinOnce();  

        // ros::Time current_time = ros::Time::now();
        // ros::Duration time_since_last_plan = current_time - last_planning_time;

        // Determine if it's time to replan based on frequency
        // bool time_to_replan = time_since_last_plan.toSec() >= (1.0 / planning_frequency_);

        // Wait for goal
        if (!goal_received_) {
            rate.sleep();
            continue;
        }

        // Update the last planning time if replanning due to frequency
        // if (time_to_replan) {
        //     last_planning_time = current_time;
        // }

        // Proceed with planning
        try {
            if (!map_received_) {
                ROS_WARN_THROTTLE(5, "Map data not received yet.");
                rate.sleep();
                continue;
            }

            if (!states_received_) {
                ROS_WARN_THROTTLE(5, "Robot state data not received yet.");
                rate.sleep();
                continue;
            }

            if (goal_x_ == std::numeric_limits<double>::infinity() || goal_y_ == std::numeric_limits<double>::infinity()) {
                ROS_WARN_THROTTLE(5, "Goal not set. Awaiting goal...");
                rate.sleep();
                continue;
            }
            // Clear the previous solution
            omplSimpleSetup_->clear();

            // Validate robot_states_
            if (robot_states_.size() < 3) {
                ROS_ERROR("robot_states_ does not contain enough elements.");
                rate.sleep();
                continue;
            }

            double startX = robot_states_[0];
            double startY = robot_states_[1];
            double startTheta = robot_states_[2];

            ROS_INFO("Robot start position: x = %f, y = %f, theta = %f", startX, startY, startTheta);

            // Initialize OMPL planner
            if (!omplSimpleSetup_) {
                ROS_ERROR("omplSimpleSetup_ is not initialized!");
                rate.sleep();
                continue;
            }

            // Set the start and goal states dynamically
            ob::ScopedState<ob::SE2StateSpace> start(omplSimpleSetup_->getSpaceInformation());
            start->setX(startX);
            start->setY(startY);
            start->setYaw(startTheta);

            ob::ScopedState<ob::SE2StateSpace> goal(omplSimpleSetup_->getSpaceInformation());
            goal->setX(goal_x_);
            goal->setY(goal_y_);
            goal->setYaw(0.0);  

            ROS_INFO("Goal position set: x = %f, y = %f", goal_x_, goal_y_);

            // Get the problem definition and set the start and goal states
            auto pdef = omplSimpleSetup_->getProblemDefinition();
            pdef->setStartAndGoalStates(start, goal);

            auto combined_obj = getCombinedObjective(
                omplSimpleSetup_->getSpaceInformation(),
                grid_map_,
                alpha1_,
                alpha2_,
                robot_length_,
                robot_width_,
                num_samples_,
                goal_x_,
                goal_y_);

            if (!combined_obj) {
                ROS_ERROR("Failed to create combined optimization objective.");
                rate.sleep();
                continue;
            }

            pdef->setOptimizationObjective(combined_obj);

            ROS_INFO("Starting OMPL planning...");
            // Solve using OMPL
            ompl::base::PlannerStatus solved = omplSimpleSetup_->solve(solver_time_);

            ROS_INFO("Done solving OMPL problem!");

            if (solved) {
                ROS_INFO("Path found, publishing...");

                // Get the solution path and simplify
                og::PathGeometric path = omplSimpleSetup_->getSolutionPath();
                path.interpolate();  // Smooth the path

                // Initialize markers
                visualization_msgs::MarkerArray markerArray;
                visualization_msgs::Marker sourcePoint, goalPoint, rrtTreeMarker, finalPath;
                initializeMarkers(sourcePoint, goalPoint, rrtTreeMarker, finalPath);

                // Set source and goal marker positions
                sourcePoint.pose.position.x = startX;
                sourcePoint.pose.position.y = startY;
                goalPoint.pose.position.x = goal_x_;
                goalPoint.pose.position.y = goal_y_;

                // Add source and goal markers to the marker array
                markerArray.markers.push_back(sourcePoint);
                markerArray.markers.push_back(goalPoint);

                // Add branches to the RRT tree
                addBranchtoRRTTree(rrtTreeMarker);
                markerArray.markers.push_back(rrtTreeMarker);

                // Set the final path marker
                setFinalPathMarker(path, finalPath);
                markerArray.markers.push_back(finalPath);

                // Publish the marker array
                rrt_pub_.publish(markerArray);

                // Publish the path as nav_msgs::Path
                publishPath(path);

                // Reset the goal_received_ flag
                goal_received_ = false;

            } else {
                ROS_WARN("No path found.");
            }
        }
        catch (const std::exception& e) {
            ROS_ERROR("Exception during planning: %s", e.what());
        }

        rate.sleep();  
    }
}

void RRTNode::addBranchtoRRTTree(visualization_msgs::Marker &rrtTreeMarker) {
    ROS_INFO("Adding branches to RRT tree...");
    // Clear previous points
    rrtTreeMarker.points.clear();

    // Create a PlannerData object and get planner data from OMPL
    ompl::base::PlannerData plannerData(omplSimpleSetup_->getSpaceInformation());
    omplSimpleSetup_->getPlannerData(plannerData);

    // Iterate through all vertices in the planner data
    for (unsigned int i = 0; i < plannerData.numVertices(); ++i) {
        // Get the list of edges for vertex i
        std::vector<unsigned int> edgeList;
        plannerData.getEdges(i, edgeList);

        const ob::SE2StateSpace::StateType *parentState = plannerData.getVertex(i).getState()->as<ob::SE2StateSpace::StateType>();

        geometry_msgs::Point parentPoint;
        parentPoint.x = parentState->getX();
        parentPoint.y = parentState->getY();
        parentPoint.z = 0;

        // Iterate through all edges (child vertices) connected to vertex i
        for (unsigned int j = 0; j < edgeList.size(); ++j) {
            unsigned int childIndex = edgeList[j];
            const ob::SE2StateSpace::StateType *childState = plannerData.getVertex(childIndex).getState()->as<ob::SE2StateSpace::StateType>();

            geometry_msgs::Point childPoint;
            childPoint.x = childState->getX();
            childPoint.y = childState->getY();
            childPoint.z = 0;

            // Add the parent and child points as a pair to form a line segment
            rrtTreeMarker.points.push_back(parentPoint);
            rrtTreeMarker.points.push_back(childPoint);
        }
    }
}

void RRTNode::setFinalPathMarker(const og::PathGeometric &path, visualization_msgs::Marker &finalPath) {
    ROS_INFO("Setting final path marker...");
    finalPath.points.clear();

    for (std::size_t i = 0; i < path.getStateCount(); ++i) {
        const ob::SE2StateSpace::StateType *se2state = path.getState(i)->as<ob::SE2StateSpace::StateType>();
        geometry_msgs::Point point;
        point.x = se2state->getX();
        point.y = se2state->getY();
        point.z = 0.1;

        finalPath.points.push_back(point);
    }
}

void RRTNode::publishPath(const og::PathGeometric &path) {
    nav_msgs::Path pathMsg;
    pathMsg.header.stamp = ros::Time::now();
    pathMsg.header.frame_id = frame_id_;

    for (std::size_t i = 0; i < path.getStateCount(); ++i) {
        const ob::SE2StateSpace::StateType *se2state = path.getState(i)->as<ob::SE2StateSpace::StateType>();

        geometry_msgs::PoseStamped pose;
        pose.header = pathMsg.header;
        pose.pose.position.x = se2state->getX();
        pose.pose.position.y = se2state->getY();
        pose.pose.orientation = tf::createQuaternionMsgFromYaw(se2state->getYaw());
        pathMsg.poses.push_back(pose);
    }

    path_pub_.publish(pathMsg);
}

void RRTNode::gridMapCallback(const grid_map_msgs::GridMap::ConstPtr& msg) {
    grid_map::GridMapRosConverter::fromMessage(*msg, grid_map_);

    if (grid_map_.exists("traversability")) {
        const auto& traversabilityData = grid_map_["traversability"];
        
        // Update boundary values
        updateBoundsFromGridMap();
    } else {
        ROS_WARN("Traversability layer not found in the grid map.");
    }

    map_received_ = true;
}

void RRTNode::updateBoundsFromGridMap() {
    grid_map::Position map_center = grid_map_.getPosition();
    grid_map::Length map_size = grid_map_.getLength();

    // Calculate new boundaries based on the grid map size, robot position, and goal position
    double new_min_x = map_center.x() - (map_size.x() / 2);
    double new_max_x = map_center.x() + (map_size.x() / 2);
    double new_min_y = map_center.y() - (map_size.y() / 2);
    double new_max_y = map_center.y() + (map_size.y() / 2);

    if (states_received_) {
        new_min_x = std::min(new_min_x, robot_states_[0] - 1.0);
        new_max_x = std::max(new_max_x, robot_states_[0] + 1.0);
        new_min_y = std::min(new_min_y, robot_states_[1] - 1.0);
        new_max_y = std::max(new_max_y, robot_states_[1] + 1.0);
    }
    if (goal_received_) {
        new_min_x = std::min(new_min_x, goal_x_ - 1.0);
        new_max_x = std::max(new_max_x, goal_x_ + 1.0);
        new_min_y = std::min(new_min_y, goal_y_ - 1.0);
        new_max_y = std::max(new_max_y, goal_y_ + 1.0);
    }

    // If this is the first time or the map has grown, update the boundaries
    bool is_first_setup = (min_x_ == 0 && max_x_ == 0 && min_y_ == 0 && max_y_ == 0);
    bool map_has_grown = (new_min_x < min_x_) || (new_max_x > max_x_) || (new_min_y < min_y_) || (new_max_y > max_y_);

    if (is_first_setup || map_has_grown) {
        // Update the boundaries
        min_x_ = new_min_x;
        max_x_ = new_max_x;
        min_y_ = new_min_y;
        max_y_ = new_max_y;

        // If OMPL is already set up, reconfigure the bounds
        if (omplSimpleSetup_) {
            auto space = omplSimpleSetup_->getStateSpace()->as<ob::SE2StateSpace>();
            ob::RealVectorBounds bounds(2);
            bounds.setLow(0, min_x_);
            bounds.setLow(1, min_y_);
            bounds.setHigh(0, max_x_);
            bounds.setHigh(1, max_y_);
            space->setBounds(bounds);
        }
    } 
}

void RRTNode::goalCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
    // Update the goal position with the one received from RViz
    goal_x_ = msg->pose.position.x;
    goal_y_ = msg->pose.position.y;

    ROS_INFO("New goal received: x = %f, y = %f", goal_x_, goal_y_);

    goal_received_ = true;
}

void RRTNode::odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    // Update the current orientation of the robot
    tf::Quaternion q(
        msg->pose.pose.orientation.x,
        msg->pose.pose.orientation.y,
        msg->pose.pose.orientation.z,
        msg->pose.pose.orientation.w
    );

    tf::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);

    // Update the current position of the robot
    robot_states_.clear();
    robot_states_.push_back(msg->pose.pose.position.x);
    robot_states_.push_back(msg->pose.pose.position.y);
    robot_states_.push_back(yaw);

    states_received_ = true;
}

void RRTNode::initializeMarkers(
    visualization_msgs::Marker &sourcePoint,
    visualization_msgs::Marker &goalPoint,
    visualization_msgs::Marker &rrtTreeMarker,
    visualization_msgs::Marker &finalPath)
{
    // Initialize headers
    std_msgs::Header header;
    header.frame_id = frame_id_;
    header.stamp = ros::Time::now();

    sourcePoint.header = goalPoint.header = rrtTreeMarker.header = finalPath.header = header;
    sourcePoint.ns = goalPoint.ns = rrtTreeMarker.ns = finalPath.ns = "path_planner";
    sourcePoint.action = goalPoint.action = rrtTreeMarker.action = finalPath.action = visualization_msgs::Marker::ADD;
    sourcePoint.pose.orientation.w = goalPoint.pose.orientation.w = rrtTreeMarker.pose.orientation.w = finalPath.pose.orientation.w = 1.0;

    // Assign unique IDs
    sourcePoint.id = 0;
    goalPoint.id = 1;
    rrtTreeMarker.id = 2;
    finalPath.id = 3;

    // Define types
    sourcePoint.type = goalPoint.type = visualization_msgs::Marker::SPHERE;
    rrtTreeMarker.type = visualization_msgs::Marker::LINE_LIST;
    finalPath.type = visualization_msgs::Marker::LINE_STRIP;

    // Set scales (increase line width for better visibility)
    sourcePoint.scale.x = sourcePoint.scale.y = sourcePoint.scale.z = 0.5;
    goalPoint.scale.x = goalPoint.scale.y = goalPoint.scale.z = 0.5;
    rrtTreeMarker.scale.x = 0.05; // Increase line width
    finalPath.scale.x = 0.1;

    // Set colors
    // Source Point - Bright Red
    sourcePoint.color.r = 1.0f;
    sourcePoint.color.g = 0.0f;
    sourcePoint.color.b = 0.0f;
    sourcePoint.color.a = 1.0f;

    // Goal Point - Bright Green
    goalPoint.color.r = 0.0f;
    goalPoint.color.g = 1.0f;
    goalPoint.color.b = 0.0f;
    goalPoint.color.a = 1.0f;

    // RRT Tree Marker - Bright Orange
    rrtTreeMarker.color.r = 1.0f;
    rrtTreeMarker.color.g = 0.5f;
    rrtTreeMarker.color.b = 0.0f;
    rrtTreeMarker.color.a = 1.0f;

    // Final Path - Bright Cyan with reduced opacity
    finalPath.color.r = 0.0f;
    finalPath.color.g = 1.0f;
    finalPath.color.b = 1.0f;
    finalPath.color.a = 0.5f; 

    // Set lifetime to keep the markers displayed until overwritten
    sourcePoint.lifetime = ros::Duration(0);
    goalPoint.lifetime = ros::Duration(0);
    rrtTreeMarker.lifetime = ros::Duration(0);
    finalPath.lifetime = ros::Duration(0);
}