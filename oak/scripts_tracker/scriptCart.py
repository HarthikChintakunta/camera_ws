from SvayaAPI.CartesianPose import CartesianPose
from SvayaAPI.clientApi import SvayaApi
import pinocchio
import time
import sys
from pathlib import Path
import numpy as np
IP = "localhost"
robot = SvayaApi()

def error_callback(error_priority, errpr_msg, error_status):
    print("Error:", error_priority, errpr_msg, error_status)
    if error_priority == "info":
        print("Info:", errpr_msg,error_status)
    elif error_priority == "medium":
        print("Medium:", errpr_msg,error_status)
    elif error_priority == "high":
        print("High:", errpr_msg,error_status)


def initialize():
    print("Connecting to robot...")
    robot.initialize(IP,error_callback)
    print("Connected to robot.")
    input("enter to enable ...")
    robot.enableRobot()
    time.sleep(1)

def moveCart(cPos):
    robot.setJogSpeed(50)
    robot.moveToCartPose(cPos)
    time.sleep(1)
    
def moveTraj(poses):
    traj = []
    trajnc = []
    for pos in poses:
        traj.append(CartesianPose(pos))
        trajnc.append(pos)
        #robot.moveToCartPose(CartesianPose(pos),100.0,100.0,ROBOT_NAME)
        time.sleep(0.2)
    print("trajectory",trajnc)
    robot.moveTrajectory(traj,500.0,500.0)
    print("moving")
urdf_file = "urdf/SR_L10_resolved.urdf"
model = pinocchio.buildModelFromUrdf(urdf_file)
print(model.name)
data = model.createData()
    
if __name__ == "__main__":
    initialize()
    time.sleep(0.1)
    z_move = 10
    repeat = True
    robot.moveJoints([0,35,100,0,45,0],50.0,50.0)
    q = np.deg2rad(np.array([0,35,100,0,45,0]))
    pinocchio.forwardKinematics(model,data,q)
    for name, oMi in zip(model.names, data.oMi):
        print("{:<24} : {: .2f} {: .2f} {: .2f}".format(name, *oMi.translation.T.flat))

    J = pinocchio.computeJointJacobian(model,data,q,6)
    J_det = np.linalg.det(J)
    J_inv = np.linalg.inv(J)
    q_des = J_inv @ np.array([0,0,10,0,0,0])
    print(np.rad2deg(q_des))
    #print(J_det)
    time.sleep(5)
    cartPose = CartesianPose([733,-10,150,0,180,0])
    # robot.moveToCartPose(cartPose)
    poses = [[733,-10,150,0,180,0],[733,-10,550,0,180,0],[733,-450,550,0,180,0],[733,-450,150,0,180,0], [733,-10,150,0,180,0]] # For 10kg robot
    # poses = [[176,100,20,0,180,0],[176,-100,20,0,180,0],[376,-100,20,0,180,0],[376,100,20,0,180,0],[170,100,20,0,180,0]]
    while repeat:
        flag = ""
        try:
            #moveCart(cartPose)
            # time.sleep(0.5)
            # moveTraj(poses)
            # input("enter to continue \n")
            continue
                    
        except KeyboardInterrupt:
            print("exited")
            robot.close()