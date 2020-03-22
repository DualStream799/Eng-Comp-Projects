#! /usr/bin/env python
# -*- coding:utf-8 -*-

import rospy
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import LaserScan
import numpy as np

ahead = 1.0
def distance_scan(data):
    global ahead
    readings = np.array(data.ranges).round(decimals=2)
    ahead = readings[0]

v = 0.2  # Velocidade linear
forward = Twist(Vector3(v,0,0), Vector3(0,0,0))
backward = Twist(Vector3(-v,0,0), Vector3(0,0,0))
stop = Twist(Vector3(0,0,0), Vector3(0,0,0))

if __name__ == "__main__":
    rospy.init_node("roda_exemplo")
    pub = rospy.Publisher("cmd_vel", Twist, queue_size=3)
    scan = rospy.Subscriber("/scan", LaserScan, distance_scan)

    try:
        while not rospy.is_shutdown():
            if ahead >= 1.02:
                pub.publish(forward)
                rospy.sleep(2.0)
            elif ahead <= 1.00:
                pub.publish(backward)
                rospy.sleep(2.0)

    except rospy.ROSInterruptException:
        print("Ocorreu uma exceção com o rospy")