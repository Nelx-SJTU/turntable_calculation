import rospy
import time
from std_srvs.srv import SetBool, Trigger
from turntable_ros.srv import Int, Float


class TurntableController:
    def __init__(self):
        rospy.init_node('turntable_controller')

    def set_max360(self, limit):
        rospy.wait_for_service('turntable_max360')
        try:
            set_limit = rospy.ServiceProxy('turntable_max360', SetBool)
            response = set_limit(limit)
            print("Max360 set to", limit)
        except rospy.ServiceException as e:
            print("Service call failed:", e)

    def set_0_ref(self):
        rospy.wait_for_service('turntable_set_0_ref')
        try:
            set_ref = rospy.ServiceProxy('turntable_set_0_ref', Trigger)
            response = set_ref()
            print("0° reference set")
        except rospy.ServiceException as e:
            print("Service call failed:", e)

    def set_abs_position(self, position):
        rospy.wait_for_service('turntable_set_abs_position')
        try:
            set_pos = rospy.ServiceProxy('turntable_set_abs_position', Int)
            response = set_pos(position)
            print("Absolute position set to", position)
        except rospy.ServiceException as e:
            print("Service call failed:", e)

    def set_rel_position(self, position):
        rospy.wait_for_service('turntable_set_rel_position')
        try:
            set_pos = rospy.ServiceProxy('turntable_set_rel_position', Int)
            response = set_pos(position)
            print("Relative position set to", position)
        except rospy.ServiceException as e:
            print("Service call failed:", e)

    '''
    @Name: TurntableController.set_velocity
    @Description: Starts a rotation motion at a given rotation speed (in turns per second)
    @rosmessage_type: turntable_ros::Float
    '''
    def set_velocity(self, speed):
        rospy.wait_for_service('turntable_start_rotation')
        try:
            start_rot = rospy.ServiceProxy('turntable_start_rotation', Float)
            response = start_rot(speed)
            print("Rotation started at", speed, "turns per second")
        except rospy.ServiceException as e:
            print("Service call failed:", e)

    '''
    @Name: set_acceleration
    @Description: Sets the acceleration of the turntable, ranging from 1 to 10 (arbitrary unit) 
    @rosmessage_type: turntable_ros::Int
    '''
    def set_acceleration(self, acceleration):
        rospy.wait_for_service('turntable_set_acceleration')
        try:
            set_accel = rospy.ServiceProxy('turntable_set_acceleration', Int)
            response = set_accel(acceleration)
            print("Acceleration set to", acceleration)
        except rospy.ServiceException as e:
            print("Service call failed:", e)

    def stop_rotation(self):
        rospy.wait_for_service('turntable_stop_rotation')
        try:
            stop_rot = rospy.ServiceProxy('turntable_stop_rotation', Trigger)
            response = stop_rot()
            print("Rotation stopped")
        except rospy.ServiceException as e:
            print("Service call failed:", e)


if __name__ == "__main__":
    controller = TurntableController()

    start_time = time.time()

    # controller.set_max360(True)
    # controller.set_0_ref()

    # -- Test set position
    controller.set_abs_position(0)  # 0->180->360(0): counterclockwise
    # controller.set_rel_position(30)  # +: counterclockwise, -: clockwise

    # -- Test Rotation speed
    # controller.set_velocity(0.05)
    # print("Start rotation.")
    # while time.time() - start_time < 12:
    #     pass
    #
    # controller.stop_rotation()
    # print("Stop.")
