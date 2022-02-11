import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
import math
from matplotlib.animation import FuncAnimation, PillowWriter 

def gimballock(i):
    T = np.array([[0], [0], [0]])
    alpha= (math.pi/2)*((i-20)/30) if i>=20 and i<60 else 0 if i<20 else (math.pi/2)
    beta= (math.pi/2)*(i/20) if i<20 else (math.pi/2)
    gamma=(math.pi/2)*((i-60)/30) if i>=60 else 0
    R = rRoll(alpha)
    R = np.dot(R, rPitch(beta))
    R = np.dot(R, rYaw(gamma))

    temp_objs, temp_dat = my_set_object(myobjs, mydata, R, T)

    return temp_objs

def my_set_object(Tobjs, Tdata, R, T):
    # drawing
    for oo, mat in zip(Tobjs, Tdata):
        n = len(mat[0])
        # rotation 
        mat = np.dot(R, mat) + np.dot(T, np.ones((1,n)))
        # set the object       
        oo.set_data(mat[0], mat[1])  
        oo.set_3d_properties(mat[2])
    return Tobjs, Tdata

def my_set_object_id(Tobjs, Tdata, R, T, id):
    ''' my_set_object_id (Tobjs, Tdata, R, T, id):

    Return object and data.

    Move specific object's middle point to coordinate (0,0,0).
    Do some rotation and return it to its middle point.
    Thus, the rotation happens on it's middle point, not respecting on (0,0,0).

    param:
    Tobjs:  object
    Tdata:  data
    T:  middle point of the object
    R:  rotation matrix
    id: point to object index inside data
    '''
    id = id-1
    n = len(Tdata[id][0])

    Tdata[id] = np.dot(np.eye(3), Tdata[id]) - np.dot(T, np.ones((1,n)))
    Tdata[id] = np.dot(R, Tdata[id])
    Tdata[id] = np.dot(np.eye(3), Tdata[id]) + np.dot(T, np.ones((1,n)))
    
    Tobjs[id].set_data(Tdata[id][0], Tdata[id][1])  
    Tobjs[id].set_3d_properties(Tdata[id][2])

    return Tobjs, Tdata



def rPitch(t):
    return np.array([[ math.cos(t), 0, -math.sin(t)], 
                  [           0, 1,            0],
                  [ math.sin(t), 0,  math.cos(t)]])

def rYaw(t):
    return np.array([[ math.cos(t), -math.sin(t), 0], 
                  [ math.sin(t),  math.cos(t), 0], 
                  [           0,            0, 1]])

def rRoll(t):
    return np.array([[ 1,           0,            0],
                  [ 0, math.cos(t), -math.sin(t)], 
                  [ 0, math.sin(t),  math.cos(t)]])



def myMovie_ship(i):
    T = np.array([[xaxis[i]], [yaxis[i]], [zaxis[i]-10]])
    #2*math.pi*xdata[int(i+N/4)%N]/r/12
    theta = np.sin(math.pi*i)*0.5*(np.sin(8*math.pi*i)) #look like zaxis
    R = rPitch(2*math.pi*theta/r/5)
    alpha = np.sin(-4*math.pi*i/N)*0.5  
    R = np.dot(R, rYaw(alpha))
    beta = np.cos(-4*math.pi*i/N)*0.5
    R = np.dot(R, rRoll(beta))

    mid_point = np.array([[-7], [0], [-1]])

    temp_objs, temp_dat = my_set_object_id(myobjs,mydata,rRoll(i),mid_point,10)  
    temp_objs, temp_dat = my_set_object(temp_objs,temp_dat,R,T)
    return temp_objs

def rollpitch(i):
    T = np.array([[0], [0], [0]])
    R = rRoll(i/50 if i<50 else 1)
    R = np.dot(R, rPitch(i/50-1 if i>=50 else 0))

    temp_objs, temp_dat = my_set_object(myobjs, mydata, R, T)

    return temp_objs
def pitchroll(i):
    T = np.array([[0], [0], [0]])
    R = rPitch(i/50 if i<50 else 1)
    R = np.dot(R, rRoll(i/50-1 if i>=50 else 0))

    temp_objs, temp_dat = my_set_object(myobjs, mydata, R, T)

    return temp_objs

# def set_object(R, T):
#     # drawing
#     for oo, mat in zip(objs, data):
#         n = len(mat[0])
#         # rotation 
#         mat = np.dot(R, mat) + np.dot(T, np.ones((1,n)))
#         # set the object       
#         oo.set_data(mat[0], mat[1])  
#         oo.set_3d_properties(mat[2])
#     return objs

# def roll(i):
#     phi = 2*i*math.pi/N
#     # define the rotation matrix
#     R = np.array([[1,             0,             0],
#                   [0, math.cos(phi), -math.sin(phi)], 
#                   [0, math.sin(phi), math.cos(phi)]]);
    
#     m = len(data)
#     T = np.zeros((m,1))     # no translation
#     return set_object(R, T)

# def yaw(i):
#     phi = 2*i*math.pi/N
#     # define the rotation matrix
#     R = np.array([[math.cos(phi), -math.sin(phi), 0], 
#                   [math.sin(phi),  math.cos(phi), 0], 
#                   [0,              0,             1]]);
    
#     m = len(data)
#     T = np.zeros((m,1))     # no translation
#     return set_object(R, T)

# def pitch(i):
#     phi = 2*i*math.pi/N
#     # define the rotation matrix
#     R = np.array([[ math.cos(phi), 0, -math.sin(phi)], 
#                   [0,              1,             0],
#                   [math.sin(phi), 0, math.cos(phi)]]);
    
#     m = len(data)
#     T = np.zeros((m,1))     # no translation
#     ax.text(10, 10, 10, str(phi))
#     return set_object(R, T)

def myMovie_basic(i):
    T = np.array([[xaxis[i]], [yaxis[i]], [zaxis[i]]])
    #2*math.pi*xdata[int(i+N/4)%N]/r/12
    theta = np.sin(math.pi*i)*0.5*(np.sin(11*math.pi*i)) #look like zaxis
    R = rPitch(2*math.pi*theta/r/5)
    alpha = np.sin(-5*math.pi*i/N)*0.5
    R = np.dot(R, rYaw(alpha))
    beta = np.cos(-5*math.pi*i/N)*0.5
    R = np.dot(R, rRoll(beta))

    mid_point = np.array([[-7], [0], [-1]])

    temp_objs, temp_dat = my_set_object_id(myobjs,mydata,rRoll(i),mid_point,10)  
    temp_objs, temp_dat = my_set_object(temp_objs,temp_dat,R,T)
    return temp_objs


# def myMovie_bird(i):
#     T = np.array([[xaxis[i]], [yaxis[i]], [zaxis[i]+3]])
#     #2*math.pi*xdata[int(i+N/4)%N]/r/12
#     theta = np.sin(math.pi*i)*0.5*(np.sin(8*math.pi*i)) #look like zaxis
#     R = rPitch(2*math.pi*theta/r/5)
#     alpha = np.sin(-4*math.pi*i/N)*0.5
#     R = np.dot(R, rYaw(alpha))
#     beta = np.cos(-4*math.pi*i/N)*0.5
#     R = np.dot(R, rRoll(beta))

#     mid_point = np.array([[-7], [0], [-1]])

#     #temp_objs, temp_dat = my_set_object_id(objs,data,rRoll(i),mid_point,10)  
#     temp_objs, temp_dat = my_set_object(objs,data,R,T)
#     return temp_objs

# def myMovie(i):
#     T = np.array([[xdata[i]], [ydata[i]], [xdata[i]]])
#     # yaw
#     # slip a circle into N equal angles
#     phi = -2*math.pi*i/N
#     R = np.array([[ math.cos(phi), -math.sin(phi), 0], 
#                   [ math.sin(phi),  math.cos(phi), 0], 
#                   [ 0,              0,             1]])

#     # add pitch
#     theta = 2*math.pi*xdata[int(i+N/4)%N]/r/12
#     R = np.dot(R, np.array([[ math.cos(theta), 0, -math.sin(theta)], 
#                             [0,              1,             0],
#                             [math.sin(theta), 0, math.cos(theta)]]))
    
#     # add roll
#     R = np.dot(R, np.array([[1,             0,             0],
#                             [0, math.cos(-phi), -math.sin(-phi)], 
#                             [0, math.sin(-phi),  math.cos(-phi)]]))
#     return set_object(R, T)



# def myOwnMovie(i):
#     T = np.array([[xdata[i]], [ydata[i]], [xdata[i]]])
#     # yaw
#     # slip a circle into N equal angles
#     phi = -2*math.pi*i/N
#     R = np.array([[ math.cos(phi), -math.sin(phi), 0], 
#                   [math.sin(phi), math.cos(phi), 0], 
#                   [0,              0,             1]])

#     # add pitch
#     theta = 2*math.pi*xdata[int(i+N/4)%N]/r/12
#     R = np.dot(R, np.array([[ math.cos(theta), 0, -math.sin(theta)], 
#                             [0,              1,             0],
#                             [math.sin(theta), 0, math.cos(theta)]]))
    
#     # add roll
#     R = np.dot(R, np.array([[1,              0,             0],
#                             [0, math.cos(-phi), -math.sin(-phi)], 
#                             [0, math.sin(-phi),  math.cos(-phi)]]))
#     return set_object(R, T)

# -------------- main program starts here ----------------#
N = 100
fig = plt.gcf()
ax = Axes3D(fig, xlim=(-15, 15), ylim=(-15, 15), zlim=(-15, 15))


# # data matrix
# M1 = np.array([[-3, -3, -2, -2, 2, 3, 2, -3], 
#                 [0, 0, 0, 0, 0, 0, 0, 0], 
#                 [-.5, .5, .5, 0, .5, 0, -.5, -.5]])
# M2 = np.array([[-2.5, -2.5, -1.5, -1.5, -2.5], 
#                 [1, -1, -1, 1, 1], 
#                 [0, 0, 0, 0, 0]])
# M3 = np.array([[-.5, -.5, 1, 1, -.5], 
#                 [3, -3, -3, 3, 3],
#                 [0, 0, 0, 0, 0]])
# data = [M1, M2, M3]

# # create 3D objects list
# O1, = ax.plot3D(M1[0], M1[1], M1[2])
# O2, = ax.plot3D(M2[0], M2[1], M2[2])
# O3, = ax.plot3D(M3[0], M3[1], M3[2])
# objs = [O1, O2, O3]

# DATA MATRIX
# back haul
MY1 = np.array([[-6,-6,-7,-7,-6], 
                [ 1,-1,-2, 2, 1],
                [-1,-1, 1, 1,-1]])
# right-left back haul
MY2 = np.array([[-6,-7, 0,-1,-6],
                [-1,-2,-3,-1,-1],
                [-1, 1, 2,-1,-1]])
MY3 = np.array([[-6,-7, 0,-1,-6],
                [ 1, 2, 3, 1, 1],
                [-1, 1, 2,-1,-1]])
# right-left front haul
MY4 = np.array([[ 0,-1, 0, 3, 0],
                [ 0,-1,-3, 0, 0],
                [-1,-1, 2, 2,-1]])
MY5 = np.array([[ 0,-1, 0, 3, 0],
                [ 0, 1, 3, 0, 0],
                [-1,-1, 2, 2,-1]])
# bottom haul    
MY6 = np.array([[-6,-1, 0,-1,-6,-6],
                [-1,-1, 0, 1, 1,-1],
                [-1,-1,-1,-1,-1,-1]])
# roof room
MY7 = np.array([[-5,-5,-2,-2,-5],
                [ 2,-2,-2, 2, 2],
                [ 4, 4, 5 ,5, 4]])
# front-back room
MY8 = np.array([[-2,-2,-2,-2,-2],
                [ 1,-1,-2, 2, 1],
                [-1,-1, 5, 5,-1]])
MY9 = np.array([[-5,-5,-5,-5,-5],
                [ 1,-1,-2, 2, 1],
                [-1,-1, 4, 4,-1]])
# propeler
MY10 = np.array([[-7,-7,-7,-7,-7],
                [.2,-.2,-.2,.2,.2],
                [-2,-2,0,0,-2]])
mydata = [MY1,MY2,MY3,MY4,MY5,MY6,MY7,MY8,MY9,MY10]

# 3D OBJECTS LIST
OB1, = ax.plot3D(MY1[0], MY1[1], MY1[2])
OB2, = ax.plot3D(MY2[0], MY2[1], MY2[2])
OB3, = ax.plot3D(MY3[0], MY3[1], MY3[2])
OB4, = ax.plot3D(MY4[0], MY4[1], MY4[2])
OB5, = ax.plot3D(MY5[0], MY5[1], MY5[2])
OB6, = ax.plot3D(MY6[0], MY6[1], MY6[2])
OB7, = ax.plot3D(MY7[0], MY7[1], MY7[2])
OB8, = ax.plot3D(MY8[0], MY8[1], MY8[2])
OB9, = ax.plot3D(MY9[0], MY9[1], MY9[2])
OB10, = ax.plot3D(MY10[0], MY10[1], MY10[2])

myobjs = [OB1,OB2,OB3,OB4,OB5,OB6,OB7,OB8,OB9,OB10]

# trajectory data
t = np.arange(0,1,0.01)
r = 10
# xdata = r*np.sin(2*math.pi*t)
# ydata = r*np.cos(4*math.pi*t)
xaxis = r*np.sin(4*math.pi*t)*0.5
yaxis = r*np.sin(4*math.pi*t)
zaxis = r*np.sin(math.pi*t)*0.5*(np.sin(8*math.pi*t)+1)
# basic rotations
# ani = FuncAnimation(fig, roll, frames=N, interval=10)
# ani = FuncAnimation(fig, yaw, frames=N, interval=10)
# ani = FuncAnimation(fig, pitch, frames=N, interval=1000)
# ani = FuncAnimation(fig, myMovie, frames=len(xdata), interval=100)
# ani = FuncAnimation(fig, myMovie_bird, frames=100, interval=100)

# ani = FuncAnimation(fig, myMovie_ship, frames=100, interval=100)
# ani.save('ship.gif', writer='imagemagick', fps=30)

# ani1 = FuncAnimation(fig, rollpitch, frames=100, interval=100)
# ani1.save('ship-rollpitch.gif', writer='imagemagick', fps=30)

# ani2 = FuncAnimation(fig, pitchroll, frames=100, interval=100)
# ani2.save('ship-pitchroll.gif', writer='imagemagick', fps=30)

ani3 = FuncAnimation(fig, gimballock, frames=100, interval=100)
ani3.save('ship-gimball.gif', writer='imagemagick', fps=30)

plt.show()