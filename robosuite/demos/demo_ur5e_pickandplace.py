#!/usr/bin/env python

import numpy as np
import trimesh
import os
import copy
# from moveit_commander import move_group
# from moveit_commander.robot import RobotCommander

# import rospy
# import moveit_msgs.srv
# import moveit_msgs.msg
# import geometry_msgs.msg
# from moveit_msgs.msg import Grasp
# from math import pi, tau
# import moveit_commander
# import sys
# from moveit_msgs.msg import AttachedCollisionObject
# from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
# from moveit_msgs.msg import PlaceLocation
# from tf.transformations import quaternion_from_euler
import numpy as np
import robosuite as suite

from std_msgs.msg import Header
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState

from mujoco_py import cymj
from tf.transformations import quaternion_from_euler
from robosuite import load_controller_config
from mujoco_py import MjSim, MjViewer


config = load_controller_config(default_controller="JOINT_POSITION")


# create environment instance
env = suite.make(
    env_name="ur5e_pickandplace", # try with other tasks like "Stack" and "Door"
    robots="UR5e",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=True,
    controller_configs=config,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    render_camera=None
)

# reset the environment
env.reset()
#viewer = MjViewer(env.sim)


desired_positions = [(-0.4865757957823134, -1.7595564350026254, 2.4746010722084986, -2.261700043048569, -1.6014259813153882, -1.99319855190651), (-0.4054703427207998, -1.531573823535642, 2.3238865383867173, -2.346935685343888, -1.5914791494325642, -1.9336917610246314), (-0.32436488965928617, -1.3035912120686588, 2.173172004564936, -2.4321713276392067, -1.58153231754974, -1.8741849701427526), (-0.24325943659777255, -1.0756086006016754, 2.022457470743155, -2.517406969934526, -1.571585485666916, -1.814678179260874)]
i = 0
print(env.ros_objects_pos)
delta_q = np.zeros(7)
goal_reach = False
while True:

    
    while not goal_reach:


        #current_positions = env.robots[0]._joint_positions
        #delta_q[:6] = desired_positions[i]-current_positions
        #action =1.4*delta_q
        action=np.zeros(7)
        # for i in range(100):
        #     action[-1] = 0.5
        #     env.step(action)
        #     env.render()  # render on display
            
        
            

        # for i in range(100):
        #     while env.robots[0].sim.data.actuator_length[6] >= -0.026:
        #         action[-1] = -0.1
        #         env.step(action)
        #         env.render()  # render on display
        
                
        for i in range(100000):
            if i < 10:
                print(env.ros_objects_pos)
            action=np.zeros(7)
            env.step(action)
            env.render()  # render on display
            

        #obs, reward, done, info = env.step(action)  # take action in the environment
        #env.step(action)
        #env.render()  # render on display


        # goal_reach = True
        # for e in range(len(delta_q)):
        #     if np.abs(delta_q[e]) > 0.006:
        #         goal_reach = False
        #         break

        #print(delta_q)

    goal_reach = False

    if i < len(desired_positions)-1:
        i+=1