#include "octomap_node.h"

int main(int argc, char **argv) {
    ros::init(argc, argv, "octomap_node");

    OctomapNode node;
    node.spin();

    return 0;
}