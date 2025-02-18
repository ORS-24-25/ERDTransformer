# This message is intended for communicating Isaac Gym simulator state to ROS,
# where everything is a list since there may be multiple envs at once.

sensor_msgs/Image[] projection_matrix
sensor_msgs/Image[] view_matrix
sensor_msgs/Image[] segmentation
sensor_msgs/Image[] depth
sensor_msgs/Image[] points