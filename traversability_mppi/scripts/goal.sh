#!/bin/bash
# Usage: goal.sh [goal_topic]   (default: /move_base_simple/goal)
GOAL_TOPIC="${1:-/move_base_simple/goal}"

rostopic pub "$GOAL_TOPIC" geometry_msgs/PoseStamped "header:
  seq: 0
  stamp: {secs: 0, nsecs: 0}
  frame_id: 'odom'
pose:
  position:
      x: 7.115
      y: 4.198
      z: 0.0
  orientation:
      x: 0.0021
      y: -0.0016
      z: 0.049
      w: 0.998"
