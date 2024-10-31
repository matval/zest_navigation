#ifndef TRAVERSABILITY_OBJECTIVE_H
#define TRAVERSABILITY_OBJECTIVE_H

#include <ompl/base/OptimizationObjective.h>
#include <grid_map_core/GridMap.hpp>
#include <grid_map_core/Polygon.hpp>
#include <ompl/base/spaces/SE2StateSpace.h>
#include <Eigen/Geometry>  
#include <limits>         

namespace ob = ompl::base;

class TraversabilityObjective : public ob::OptimizationObjective {
public:
    TraversabilityObjective(
        const ob::SpaceInformationPtr &si,
        grid_map::GridMap &gridMap,
        double alpha1,             
        double alpha2,              
        double robot_length,
        double robot_width,
        unsigned int num_samples,        
        double goal_x,
        double goal_y)
        : ob::OptimizationObjective(si),
          grid_map_(gridMap),
          alpha1_(alpha1),
          alpha2_(alpha2),
          robot_length_(robot_length),
          robot_width_(robot_width),
          num_samples_(num_samples),
          goal_x_(goal_x),
          goal_y_(goal_y) {}

    ob::Cost motionCost(const ob::State *s1, const ob::State *s2) const override {
        std::vector<ob::State *> interp_states;

        // Interpolate between s1 and s2
        for (unsigned int i = 0; i <= num_samples_; ++i) {
            ob::State *interpolated_state = si_->allocState();
            si_->getStateSpace()->interpolate(s1, s2, static_cast<double>(i) / num_samples_, interpolated_state);
            interp_states.push_back(interpolated_state);
        }

        double total_cost = 0.0;

        for (const auto *state : interp_states) {
            total_cost += computeTraversabilityCost(state);
        }

        // Free the allocated states
        for (auto *state : interp_states) {
            si_->freeState(state);
        }

        return ob::Cost(total_cost);
    }

    ob::Cost stateCost(const ob::State *s) const override {
        const auto *se2state = s->as<ob::SE2StateSpace::StateType>();
        double x = se2state->getX();
        double y = se2state->getY();

        // Compute Euclidean distance to the goal
        double dx = x - goal_x_;
        double dy = y - goal_y_;
        double distance_to_goal = std::hypot(dx, dy);

        return ob::Cost(distance_to_goal);
    }

private:
    double computeTraversabilityCost(const ob::State *s) const {
        const auto *se2state = s->as<ob::SE2StateSpace::StateType>();
        double x = se2state->getX();
        double y = se2state->getY();
        double theta = se2state->getYaw();

        // Compute the robot's footprint polygon
        std::vector<grid_map::Position> footprint = computeFootprint(x, y, theta);

        // Get the minimum traversability within the footprint
        float min_traversability = getMinTraversability(footprint);

        // Get the minimum lambda within the footprint
        float min_lambda = getMinLambda(footprint);

        // Compute the cost using alpha1 and alpha2
        double cost = exp(-(alpha1_ * min_traversability + alpha2_ * min_lambda));
        // double cost = -(alpha1_ * min_traversability + alpha2_ * min_lambda);

        // std::cout << "cost: " << cost << std::endl;

        return cost;
    }

    std::vector<grid_map::Position> computeFootprint(double x, double y, double theta) const {
        // Half dimensions
        double half_length = robot_length_ / 2.0;
        double half_width = robot_width_ / 2.0;

        // Corners of the rectangle before rotation (relative to robot center)
        std::vector<Eigen::Vector2d> corners;
        corners.emplace_back(half_length, half_width);
        corners.emplace_back(half_length, -half_width);
        corners.emplace_back(-half_length, -half_width);
        corners.emplace_back(-half_length, half_width);

        // Rotation matrix
        Eigen::Rotation2Dd rot(theta);

        // Compute rotated and translated corners
        std::vector<grid_map::Position> footprint;
        for (const auto& corner : corners) {
            Eigen::Vector2d rotated_corner = rot * corner;
            footprint.emplace_back(x + rotated_corner.x(), y + rotated_corner.y());
        }

        return footprint;
    }

    float getMinTraversability(const std::vector<grid_map::Position>& footprint) const {
        return getMinValue(footprint, "traversability", 0.0f); 
    }

    float getMinLambda(const std::vector<grid_map::Position>& footprint) const {
        return getMinValue(footprint, "lambda", 0.0f);
    }

    float getMinValue(const std::vector<grid_map::Position>& footprint, const std::string& layer_name, float default_value) const {
        // Create a polygon from the footprint
        grid_map::Polygon polygon;
        for (const auto& point : footprint) {
            polygon.addVertex(point);
        }

        // Get the bounding box of the polygon
        grid_map::Position center;
        grid_map::Length length;
        polygon.getBoundingBox(center, length);

        // Compute min and max bounds
        grid_map::Position min_bound(center.x() - length.x() / 2.0, center.y() - length.y() / 2.0);
        grid_map::Position max_bound(center.x() + length.x() / 2.0, center.y() + length.y() / 2.0);

        // Grid map resolution
        double resolution = grid_map_.getResolution();

        // Initialize minimum value
        float min_value = std::numeric_limits<float>::max();

        // Iterate over the cells within the bounding box
        for (double x = min_bound.x(); x <= max_bound.x(); x += resolution) {
            for (double y = min_bound.y(); y <= max_bound.y(); y += resolution) {
                grid_map::Position pos(x, y);
                if (polygon.isInside(pos) && grid_map_.isInside(pos)) {
                    float value = grid_map_.atPosition(layer_name, pos);
                    if (value < min_value) {
                        min_value = value;
                    }
                }
            }
        }

        // Handle case where no value was found
        if (min_value == std::numeric_limits<float>::max()) {
            // Return default value if none found within footprint
            min_value = default_value;
        }

        return min_value;
    }

    grid_map::GridMap &grid_map_;
    double alpha1_;             // Weight for traversability
    double alpha2_;             // Weight for lambda
    double robot_length_;
    double robot_width_;
    unsigned int num_samples_;  // Number of samples along the motion
    double goal_x_;
    double goal_y_;
};

ob::OptimizationObjectivePtr getCombinedObjective(
    const ob::SpaceInformationPtr &si,
    grid_map::GridMap &grid_map,
    double alpha1,
    double alpha2,
    double robot_length,
    double robot_width,
    unsigned int num_samples,
    double goal_x,
    double goal_y) {

    auto traversabilityObj = std::make_shared<TraversabilityObjective>(
        si, grid_map, alpha1, alpha2, robot_length, robot_width, num_samples, goal_x, goal_y);

    return traversabilityObj;
}

#endif // TRAVERSABILITY_OBJECTIVE_H
