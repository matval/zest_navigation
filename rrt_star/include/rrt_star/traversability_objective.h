#ifndef TRAVERSABILITY_OBJECTIVE_H
#define TRAVERSABILITY_OBJECTIVE_H

#include <ompl/base/OptimizationObjective.h>
#include <ompl/base/objectives/StateCostIntegralObjective.h>
#include <grid_map_core/GridMap.hpp>
#include <grid_map_core/Polygon.hpp>
#include <ompl/base/spaces/SE2StateSpace.h>
#include <Eigen/Geometry>  
#include <limits>         

namespace ob = ompl::base;

class TraversabilityObjective : public ob::StateCostIntegralObjective {
public:
    TraversabilityObjective(
        const ob::SpaceInformationPtr &si,
        grid_map::GridMap &gridMap,
        double alpha1,             
        double alpha2,              
        double robot_length,
        double robot_width)
        : ob::StateCostIntegralObjective(si, true), grid_map_(gridMap), alpha1_(alpha1),
            alpha2_(alpha2), robot_length_(robot_length), robot_width_(robot_width) {

            // Check if the layers exist in the grid map
            if (!grid_map_.exists("traversability") || !grid_map_.exists("lambda")) {
                std::cerr << "One or both layers do not exist in the grid map." << std::endl;
            }

            // Get the maximum value of the "traversability" layer
            maxTraversability_ = grid_map_["traversability"].maxCoeff();

            // Get the maximum value of the "lambda" layer
            maxLambda_ = grid_map_["lambda"].maxCoeff();

            // Print the results
            std::cout << "Maximum value in 'traversability' layer: " << maxTraversability_ << std::endl;
            std::cout << "Maximum value in 'lambda' layer: " << maxLambda_ << std::endl;
        }

    ob::Cost stateCost(const ob::State *s) const override {
        return computeTraversabilityCost(s);
    }

private:
    ob::Cost computeTraversabilityCost(const ob::State *s) const {
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
        // double cost = exp(-alpha1_ * min_traversability) + exp(-alpha2_ * (min_lambda - 2.0));
        double cost = (maxTraversability_ - min_traversability) * alpha1_ + (maxLambda_ - min_lambda) * alpha2_;

        return ob::Cost(cost);
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
        float sum_value = 0.0;
        int count = 0;

        // Iterate over the cells within the bounding box
        for (double x = min_bound.x(); x <= max_bound.x(); x += resolution) {
            for (double y = min_bound.y(); y <= max_bound.y(); y += resolution) {
                grid_map::Position pos(x, y);
                if (polygon.isInside(pos) && grid_map_.isInside(pos)) {
                    float value = grid_map_.atPosition(layer_name, pos);

                    sum_value += value;
                    count += 1;
                    // if (value < min_value) {
                    //     min_value = value;
                    // }
                }
            }
        }

        // Handle case where no value was found
        // if (min_value == std::numeric_limits<float>::max()) {
        //     // Return default value if none found within footprint
        //     min_value = default_value;
        // }
        min_value = sum_value / count;

        return min_value;
    }

    grid_map::GridMap &grid_map_;
    double alpha1_;             // Weight for traversability
    double alpha2_;             // Weight for lambda
    double robot_length_;
    double robot_width_;
    double maxTraversability_;
    double maxLambda_;
};

ob::OptimizationObjectivePtr getCombinedObjective(
    const ob::SpaceInformationPtr &si,
    grid_map::GridMap &grid_map,
    double alpha1,
    double alpha2,
    double robot_length,
    double robot_width) {

    auto traversabilityObj = std::make_shared<TraversabilityObjective>(
        si, grid_map, alpha1, alpha2, robot_length, robot_width);

    return ob::OptimizationObjectivePtr(traversabilityObj);
}

#endif // TRAVERSABILITY_OBJECTIVE_H
