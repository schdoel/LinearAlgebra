# from _typeshed import SupportsAnext
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
import math
from matplotlib.animation import FuncAnimation, PillowWriter 

def set_object(R, T):
    # drawing
    for obj, mat, rot in zip(objs, data, R):
        n = len(mat[0])
        # rotation      
        mat = np.dot(rot, mat) + np.dot(T, np.ones((1,n)))
        # set the object    
        obj.set_data(mat[0], mat[1])
        obj.set_3d_properties(mat[2])
    return objs

# def set_obj(R, T):
#     # drawing
#     for o, mat, in zip(Aobjs, Adata):
#         n = len(mat[0])
#         # rotation      
#         mat = np.dot(R, mat) + np.dot(T, np.ones((1,n)))
#         # set the object    
#         o.set_data(mat[0], mat[1])
#         o.set_3d_properties(mat[2])
#     return Aobjs


def roll(i):
    phi = 2*i*math.pi/N
    # define the rotation matrix
    R = np.array([[1,             0,             0],
                  [0, math.cos(phi), -math.sin(phi)], 
                  [0, math.sin(phi), math.cos(phi)]])
    
    m = len(data)
    T = np.zeros((3,1))     # no translation
    return set_object(R, T)

def yaw(i):
    phi = 2*i*math.pi/N
    # define the rotation matrix
    R = np.array([[math.cos(phi), -math.sin(phi), 0], 
                  [math.sin(phi),  math.cos(phi), 0], 
                  [0,              0,             1]])
    
    m = len(data)
    T = np.zeros((3,1))     # no translation
    return set_object(R, T)

def pitch(i):
    phi = 2*i*math.pi/N
    # define the rotation matrix
    R = np.array([[ math.cos(phi), 0, -math.sin(phi)], 
                  [0,              1,             0],
                  [math.sin(phi), 0, math.cos(phi)]])
    
    T = np.zeros((3,1))     # no translation
    ax.text(10, 10, 10, str(phi))
    return set_object(R, T)

def myMovie_test(i):
    T = np.array([[xdata[i]], [ydata[i]], [zdata[i]]])
    # slip a circle into N equal angles
    phi = -2*math.pi*i/N
    
    R= np.array([[ math.cos(phi), 0, -math.sin(phi)], 
                    [0,              1,             0],
                    [math.sin(phi), 0, math.cos(phi)]])
    
    n = len(data[1][0])
    #translate to center
    data[1] = np.dot(np.eye(3), data[1]) - np.dot(D[1], np.ones((1,n)))
    #rotate
    data[1] = np.dot(R, data[1])
    #translate back to original point
    data[1] = np.dot(np.eye(3), data[1]) + np.dot(D[1], np.ones((1,n)))

    objs[1].set_data(data[1][0], data[1][1])
    objs[1].set_3d_properties(data[1][2])
    
    theta = -20*math.pi*i/N
    R= np.array([[math.cos(theta), -math.sin(theta), 0], 
                        [math.sin(theta),  math.cos(theta), 0], 
                        [0,              0,             1]])
    n = len(data[2][0])
    #translate to center
    data[2] = np.dot(np.eye(3), data[2]) - np.dot(D[2], np.ones((1,n)))
    #rotate
    data[2] = np.dot(R, data[2])
    #translate back to original point
    data[2] = np.dot(np.eye(3), data[2]) + np.dot(D[2], np.ones((1,n)))

    objs[2].set_data(data[2][0], data[2][1])
    objs[2].set_3d_properties(data[2][2])

    
    theta = -5*math.pi*i/N
    sintheta = np.sin(theta)*0.3
    costheta = np.cos(theta)
    # yaw
    # R = np.array([[ math.cos(sintheta), -math.sin(sintheta), 0], 
    #               [math.sin(sintheta), math.cos(sintheta), 0], 
    #               [0,              0,             1]])
    
    #roll
    # R = np.dot(R,
    # R = np.array([[1,             0,             0],
    #               [0, math.cos(np.cos(costheta)), -math.sin(np.cos(costheta))], 
    #               [0, math.sin(np.cos(costheta)), math.cos(np.cos(costheta))]])#)

   
    # add pitch
    theta = 2*math.pi*xdata[int(i+N/4)%N]/r/12
    R = np.array([[ math.cos(theta*.1), 0, -math.sin(theta*.1)], 
                            [0,              1,             0],
                            [math.sin(theta*.1), 0, math.cos(theta*.1)]])

    # add roll
    R = np.dot(R, np.array([[1,              0,             0],
                            [0, math.cos(-theta*.1), -math.sin(-theta*.1)], 
                            [0, math.sin(-theta*.1),  math.cos(-theta*.1)]]))
    
    R = np.dot(R,np.array([[ math.cos(sintheta), -math.sin(sintheta), 0], 
                  [math.sin(sintheta), math.cos(sintheta), 0], 
                  [0,              0,             1]]) )

    for obj, mat in zip(objs, data):
        n = len(mat[0])
        # rotation      
        mat = np.dot(R, mat) + np.dot(T, np.ones((1,n)))
        # set the object    
        obj.set_data(mat[0], mat[1])
        obj.set_3d_properties(mat[2])

    return objs

def myMovie_basic(i):
    T = np.array([[xdata[i]], [ydata[i]], [xdata[i]]])
    R = np.eye(3)
    return set_object(R, T)

def myMovieRollPitchYaw(i):
    """
    function for question 2
    """
    # T = np.array([[0], [0], [0]])
    T = np.array([[xdata[i]], [ydata[i]], [xdata[i]]])

    # slip a circle into N equal angles
    phi = -2*math.pi*i/N
    R=[]
    R = [np.array([[1, 0, 0], 
                   [0, 1, 0], 
                   [0, 0, 1]]) for i in range(7)]

    # add roll
    R = np.dot(R, np.array([[1,              0,             0],
                            [0, math.cos(-phi), -math.sin(-phi)], 
                            [0, math.sin(-phi),  math.cos(-phi)]]))

    # add pitch
    theta = 2*math.pi*xdata[int(i+N/4)%N]/r/12
    R = np.dot(R, np.array([[ math.cos(theta), 0, -math.sin(theta)], 
                            [0,              1,             0],
                            [math.sin(theta), 0, math.cos(theta)]]))

    #add yaw
    R = np.dot(R, np.array([[ math.cos(phi), -math.sin(phi), 0], 
                  [math.sin(phi), math.cos(phi), 0], 
                  [0,              0,             1]]))

    theta = -10*math.pi*i/N
    R[2] = np.dot(R[2], np.array([[math.cos(theta), -math.sin(theta), 0], 
                        [math.sin(theta),  math.cos(theta), 0], 
                        [0,              0,             1]]))

    return set_object(R, T)

def myMoviePitchRollYaw(i):
    """
    function for question 2
    """
    # T = np.array([[0], [0], [0]])
    T = np.array([[xdata[i]], [ydata[i]], [xdata[i]]])

    # yaw
    # slip a circle into N equal angles
    phi = -2*math.pi*i/N
    R=[]
    R = [np.array([[1, 0, 0], 
                   [0, 1, 0], 
                   [0, 0, 1]]) for i in range(7)]

    # add pitch
    theta = 2*math.pi*xdata[int(i+N/4)%N]/r/12
    R = np.dot(R, np.array([[ math.cos(theta), 0, -math.sin(theta)], 
                            [0,              1,             0],
                            [math.sin(theta), 0, math.cos(theta)]]))
    
    # # add roll
    R = np.dot(R, np.array([[1,              0,             0],
                            [0, math.cos(-phi), -math.sin(-phi)], 
                            [0, math.sin(-phi),  math.cos(-phi)]]))

    #add yaw
    R = np.dot(R, np.array([[ math.cos(phi), -math.sin(phi), 0], 
                  [math.sin(phi), math.cos(phi), 0], 
                  [0,              0,             1]]))

    theta = -10*math.pi*i/N
    R[2] = np.dot(R[2], np.array([[math.cos(theta), -math.sin(theta), 0], 
                        [math.sin(theta),  math.cos(theta), 0], 
                        [0,              0,             1]]))

    return set_object(R, T)



    return temp_objs


def Gimbal(i):
    """
    for q5
    """
    T = np.array([[0], [0], [0]])
    a = (math.pi/2)*((i-20)/30) if i>=20 and i<60 else 0 if i<20 else (math.pi/2)
    b= (math.pi/2)*(i/20) if i<20 else (math.pi/2)
    c=(math.pi/2)*((i-60)/30) if i>=60 else 0
    R = np.array([[1,             0,             0],
                  [0, math.cos(a), -math.sin(a)], 
                  [0, math.sin(a), math.cos(a)]])
    R = np.dot(R, np.array([[ math.cos(b), 0, -math.sin(b)], 
                  [0,              1,             0],
                  [math.sin(b), 0, math.cos(b)]]))
    R = np.dot(R, np.array([[math.cos(c), -math.sin(c), 0], 
                  [math.sin(c),  math.cos(c), 0], 
                  [0,              0,             1]]))

    for obj, mat in zip(objs, data):
        n = len(mat[0])
        # rotation      
        mat = np.dot(R, mat) + np.dot(T, np.ones((1,n)))
        # set the object    
        obj.set_data(mat[0], mat[1])
        obj.set_3d_properties(mat[2])

    return objs
# -------------- main program starts here ----------------#
N = 100
fig = plt.gcf()
ax = Axes3D(fig, xlim=(-25, 25), ylim=(-25, 25), zlim=(-25, 25))

#BODY
M1 = np.array([ [-5, -11, -13, -12, -11,  -3,  -1,   1,  3,   5,   7,   5,   -3,  -5], 
                [ 0,   0,   0,   0,   0,   0,   0,   0,  0,   0,   0,   0,    0,   0], 
                [ 0,   0,   2,   3,   1,   1,   3,   3,  1,   1,  -1,  -3,   -3,  -0] ])
d1 = np.array([[0],[0],[0]])

#TAIL PROPELER
M2 =  np.array([ [-11.5,  -12,   -12,  -11.5,  -11.5], 
                [    0,     0,     0,      0,      0], 
                [    0,     0,     4,      4,      0] ])
d2 =np.array([ [-11.750], [0], [2]])

#HUGE PROPELER
M3 = np.array([ [-7,  -7,  7,   7, -7], 
                [-1,   1,  1,  -1, -1],
                [ 3,   3,  3,   3,  3] ])
d3 = np.array([ [0], [0], [3]])
   
#BASE
M4 = np.array([ [-3,  5,   5,  -3, -3], 
                [ 2,  2, 1.8, 1.8,  2],
                [-5, -5,  -5,  -5, -5] ]) 
d4 = np.array([[2],[0],[-3]])

#BASE
M5 = np.array([ [ -3,   5,    5,   -3,  -3], 
                [ -2,  -2, -1.8, -1.8,  -2],
                [ -5,  -5,   -5,   -5,  -5] ])
d5 = np.array([[2],[0],[-3]])

#BASE LEG
M6 = np.array([ [ -1,  -1,  -1,  -1,  -1,    -1,    -1,    -1,    -1,    -1,   -1],  
                [ -2,  -2,   0,   2,   2,   1.8,   1.8,     0,  -1.8,  -1.8,   -2],
                [ -5,  -4,  -3,  -4,  -5,    -5,  -4.2,  -3.2,  -4.2,  -4.8,   -5] ]) 
d6 = np.array([[2],[0],[-3]])

#BASE LEG
M7 = np.array( [[ 3,  3,   3,  3,   3,   3,    3,    3,     3,     3,  3],  
               [ -2, -2,   0,  2,   2, 1.8,  1.8,    0,  -1.8,  -1.8, -2],
               [ -5, -4,  -3, -4,  -5,  -5, -4.2, -3.2,  -4.2,  -4.8, -5] ]) 
d7 = np.array([[2],[0],[-3]])
   
data = [M1, M2, M3, M4, M5, M6, M7]
D = [d1, d2, d3, d4, d5, d6, d7]

# create 3D objects list
O1, = ax.plot3D(M1[0], M1[1], M1[2])
O2, = ax.plot3D(M2[0], M2[1], M2[2])
O3, = ax.plot3D(M3[0], M3[1], M3[2])
O4, = ax.plot3D(M4[0], M4[1], M4[2])
O5, = ax.plot3D(M5[0], M5[1], M5[2])
O6, = ax.plot3D(M6[0], M6[1], M6[2])
O7, = ax.plot3D(M7[0], M7[1], M7[2])

objs = [O1, O2, O3, O4, O5, O6, O7,]


# trajectory data
t = np.arange(0,1,0.01)
r = 10
xdata = r*np.sin(2*math.pi*t)
ydata = r*np.cos(2*math.pi*t)
zdata = r*np.cos(2*math.pi*t)

# ani = FuncAnimation(fig, myMoviePitchRollYaw, frames=len(xdata), interval=100)
# ani.save('A2_PitchRollYaw.gif', writer='imagemagick', fps=30)

# ani = FuncAnimation(fig, myMovieRollPitchYaw, frames=len(xdata), interval=100)
# ani.save('A2_RollPitchYaw.gif', writer='imagemagick', fps=30)

ani = FuncAnimation(fig, Gimbal, frames=len(xdata), interval=100)
ani.save('A2_Gimbal.gif', writer='imagemagick', fps=30)

# ani = FuncAnimation(fig, myMovie_test, frames=len(xdata), interval=100)
# ani.save('A2_q1.gif', writer='imagemagick', fps=30)
plt.show()
