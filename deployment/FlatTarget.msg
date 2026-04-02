# reference for polynomial trajectory tracking

std_msgs/Header header

uint8 type_mask
uint8 IGNORE_POS = 1    # Ignore Position
uint8 IGNORE_VEL = 2    # Ignore Velocity
uint8 IGNORE_ACC = 4    # Ignore Acc
uint8 IGNORE_JERK = 8   # Ignore JERK
uint8 IGNORE_SNAP = 16  # Ignore SNAP
uint8 IGNORE_YAW = 32   # ignore yaw Reference
uint8 IGNORE_YAW_RATE = 64   # ignore yaw_dot Reference

#uint8 IGNORE_SNAP = 1	# Position Velocity Acceleration Jerk Reference
#uint8 IGNORE_SNAP_JERK = 2	# Position Velocity Acceleration Reference
#uint8 IGNORE_SNAP_JERK_ACC = 4	# Position Reference
#uint8 IGNORE_YAW = 8	# ignore yaw Reference

geometry_msgs/Vector3 position
geometry_msgs/Vector3 velocity
geometry_msgs/Vector3 acceleration
geometry_msgs/Vector3 jerk
geometry_msgs/Vector3 snap

float32 yaw
float32 yaw_dot
