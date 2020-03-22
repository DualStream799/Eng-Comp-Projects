#! /usr/bin/env python
# -*- coding:utf-8 -*-

import rospy
from geometry_msgs.msg import Twist, Vector3
from math import pi

v = 0.3  # Velocidade linear
w = pi/8  # Velocidade angular

forward = Twist(Vector3(v,0,0), Vector3(0,0,0))
stop = Twist(Vector3(0,0,0), Vector3(0,0,0))
turn = Twist(Vector3(0,0,0), Vector3(0,0,w))

if __name__ == "__main__":
    rospy.init_node("roda_exemplo")
    pub = rospy.Publisher("cmd_vel", Twist, queue_size=3)
    
    try:
        while not rospy.is_shutdown():
            pub.publish(stop)
            rospy.sleep(4.0)
            pub.publish(forward)
            rospy.sleep(4.0)
            pub.publish(stop)
            rospy.sleep(4.0)
            pub.publish(turn)
            rospy.sleep(4.0)
            pub.publish(stop)
            rospy.sleep(4.0)
    except rospy.ROSInterruptException:
        print("Ocorreu uma exceção com o rospy")