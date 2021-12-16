import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

import numpy as np
from numpy.linalg import norm
from numpy import arctan, arcsin, arccos, cos, sin, dot, cross, matmul, arctan2

from PIL import Image
import os

import cv2

def drawCube():
    # Cube
    cube_size = 0.125
    # Red side
    glBegin(GL_POLYGON)
    glColor3f(1.0, 0.0, 0.0)
    glVertex3f(cube_size, 0, 0)
    glVertex3f(cube_size, 0, cube_size)
    glVertex3f(cube_size, cube_size, cube_size)
    glVertex3f(cube_size, cube_size, 0)
    glEnd()

    # White side
    glBegin(GL_POLYGON)
    glColor3f(1.0, 1.0, 1.0)
    glVertex3f(0, 0, cube_size)
    glVertex3f(cube_size, 0, cube_size)
    glVertex3f(cube_size, cube_size, cube_size)
    glVertex3f(0, cube_size, cube_size)
    glEnd()

    # Purple side
    glBegin(GL_POLYGON)
    glColor3f(1.0, 0.0, 1.0)
    glVertex3f(0, cube_size, 0)
    glVertex3f(0, cube_size, cube_size)
    glVertex3f(cube_size, cube_size, cube_size)
    glVertex3f(cube_size, cube_size, 0)
    glEnd()

    # Green side
    glBegin(GL_POLYGON)
    glColor3f(0.0, 1.0, 0.0)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, cube_size)
    glVertex3f(cube_size, 0, cube_size)
    glVertex3f(cube_size, 0, 0)
    glEnd()

    # Blue side
    glBegin(GL_POLYGON)
    glColor3f(0.0, 0.0, 1.0)
    glVertex3f(0, 0, 0)
    glVertex3f(cube_size, 0, 0)
    glVertex3f(cube_size, cube_size, 0)
    glVertex3f(0, cube_size, 0)
    glEnd()

    # Red side
    glBegin(GL_POLYGON)
    glColor3f(1.0, 0.0, 0.0)
    glVertex3f(0, 0, 0)
    glVertex3f(0, cube_size, 0)
    glVertex3f(0, cube_size, cube_size)
    glVertex3f(0, 0, cube_size)
    glEnd()


def drawMiniAxis():
    # Cube axis
    line_len = 0.25
    glBegin(GL_LINES)

    glColor3f(0, 1.0, 1.0)
    glVertex3f(0, 0, 0)
    glVertex3f(line_len, 0, 0)

    glColor3f(0.0, 1.0, 0.0)
    glVertex3f(0, 0, 0)
    glVertex3f(0, line_len, 0)

    glColor3f(1.0, 0.0, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, line_len)
    glEnd()


def linear_interpolation(t, tm, start, end):
    x, y, z = (1-t/tm) * start + (t/tm) * end
    return x, y, z


def lerp(t, tm, q_start, q_end):
    qt = (1-t/tm) * q_start + (t/tm) * q_end
    qt = qt / norm(qt)

    p, fi = Quaternion_AngleAxis(qt)
    A = Rodrigez(p, fi)
    angles = Matrix_Angles(A)
    # convert back to degrees
    angles = np.degrees(angles)

    return angles


def slerp(t, tm, q_start, q_end):

    cos_fi = dot(q_start, q_end)

    if cos_fi < 0:       # go shorted way around circle
        q_start = -q_start
        cos_fi = -cos_fi

    if cos_fi > 0.95:    #  too close -> use lerp
        return lerp(t, tm, q_start, q_end)

    fi = arccos(cos_fi)

    qt = (q_start * sin(fi*(1-t/tm))) / sin(fi) + (q_end * sin(fi*(t/tm))) / sin(fi)

    p, fi = Quaternion_AngleAxis(qt)
    A = Rodrigez(p, fi)
    angles = Matrix_Angles(A)
    # convert back to degrees
    angles = np.degrees(angles)

    return angles


def animation(t, tm, position_start, angles_start, position_end, angles_end, q_start, q_end):
    if t >= tm:
        t = tm

    # Translation - linear interpolation
    xt, yt, zt = linear_interpolation(t, tm, position_start, position_end)

    ### Angle interpolation

    # OPTION 1: LINEAR ANGLE INTERPOLATION
    # alfa, beta, gama = linear_interpolation(t, tm, angles_start, angles_end)

    # OPTION 2: LERP (LINEAR QUATERNION INTERPOLATION)
    # alfa, beta, gama = lerp(t, tm, q_start, q_end)

    # OPTION 3: SLERP
    alfa, beta, gama = slerp(t, tm, q_start, q_end)

    ###

    glPushMatrix()

    glTranslatef(xt, yt, zt)

    glRotatef(alfa, 1, 0, 0)
    glRotatef(beta, 0, 1, 0)
    glRotatef(gama, 0, 0, 1)

    drawCube()
    drawMiniAxis()

    glPopMatrix()


def OriginalAxis():
    glBegin(GL_LINES)

    glColor3f(1.0, 1.0, 0.0)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(2.0, 0.0, 0.0)

    glColor3f(1.0, 1.0, 0.0)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(0.0, 2.0, 0.0)

    glColor3f(1.0, 1.0, 0.0)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(0.0, 0.0, 2.0)
    glEnd()


def Euler_Angles(angles):
    alfa, beta, gama = angles

    Rx = np.array([[1, 0, 0],
                   [0, cos(alfa), -sin(alfa)],
                   [0, sin(alfa), cos(alfa)]])

    Ry = np.array([[cos(beta), 0, sin(beta)],
                   [0, 1, 0],
                   [-sin(beta), 0, cos(beta)]])

    Rz = np.array([[cos(gama), -sin(gama), 0],
                   [sin(gama), cos(gama), 0],
                   [0, 0, 1]])

    A = np.matmul(Rz, matmul(Ry, Rx))
    return np.array(A)


def Matrix_AxisAngle(A):
    # Calc p
    M = A - np.identity(3)

    # Check if rows are not same to the factor
    row1 = np.array(M[0] / M[0][0])
    row2 = np.array(M[1] / M[1][0])
    row3 = np.array(M[2] / M[2][0])

    if not np.array_equal(row1, row2):
        p = cross(row1, row2)
    else:
        p = cross(row1, row3)

    p = p / norm(p)

    # Calc normal vector on p
    if p[0] != 0:
        u = [-(p[1] + p[2]) / p[0], 1, 1]
    elif p[1] != 0:
        u = [1, -(p[0] + p[2]) / p[1], 1]
    elif p[2] != 0:
        u = [1, 1, -(p[0] + p[1]) / p[2]]

    u = u / norm(u)

    # Calc u projection
    up = matmul(A, u)

    # Calc angle
    fi = arccos(dot(u, up) / (norm(u) * norm(up)))

    # Check positive orientation
    if dot(cross(u, up), p) < 0:
        p = -p

    return np.array(p), fi


def AngleAxis_Quaternion(p, fi):
    w = cos(fi / 2)

    p = p / norm(p)
    x, y, z = sin(fi / 2) * p

    return np.array([x, y, z, w])


def Quaternion_AngleAxis(q):
    q = q / norm(q)

    if q[3] < 0:
        q = -q

    fi = 2 * arccos(q[3])

    if q[3] == 1:
        p = [1, 0, 0]
    else:
        p = [q[0], q[1], q[2]]
        p = p / norm(p)

    return np.array(p), fi


def Rodrigez(p, fi):
    px = np.array([[0, -p[2], p[1]],
                   [p[2], 0, -p[0]],
                   [-p[1], p[0], 0]])
    p = np.matrix(p)
    C1 = matmul(p.T, p)
    C2 = cos(fi) * (np.identity(3) - C1)
    C3 = sin(fi) * px

    A = C1 + C2 + C3
    return np.array(A)


def Matrix_Angles(A):
    if A[2][0] < 1:
        if A[2][0] > -1:  # unique solution
            alfa = arctan2(A[2][1], A[2][2])
            beta = arcsin(-A[2][0])
            gama = arctan2(A[1][0], A[0][0])
            #print('unique')
        else:  # not unique: case Ox3 = -Oz
            alfa = 0
            beta = np.pi / 2
            gama = arctan2(-A[0][1], A[1][1])
            #print('not unique')
    else:  # not unique: case Ox3 = Oz
        alfa = 0
        beta = -np.pi / 2
        gama = arctan2(-A[0][1], A[1][1])
        #print('not unique')

    return np.array([alfa, beta, gama])


def AnglesToQuaternion(angles):
    if angles.all() == 0:
        return np.array([0, 0, 0, 1])
    # convert to radians
    angles = np.radians(angles)
    A = Euler_Angles(angles)
    p, fi = Matrix_AxisAngle(A)
    q = AngleAxis_Quaternion(p, fi)
    return np.array(q)


def main():
    pygame.init()
    width, height = 800, 600
    display = (width, height)
    screen = pygame.display.set_mode(display, DOUBLEBUF|OPENGL|GL_DEPTH)

    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
    glEnable(GL_DEPTH_TEST)

    gluLookAt(3.0, 3.0, 3.0 - 4.5, 0.0, 0.0, -4.5, 0, 1, 0)
    glTranslatef(0.0, 0.0, -4.5)

    ANIMATION_LEN = 150
    WAIT_AFTER_END = 30

    POSITION_START = np.array([0, 0, 0])
    ANGLES_START = np.array([30, 30, 30])

    POSITION_END = np.array([1.5, 1.8, 1.7])
    ANGLES_END = np.array([45, 80, 60])

    q_start = AnglesToQuaternion(ANGLES_START)
    q_end = AnglesToQuaternion(ANGLES_END)

    timer = 0
    frames = []

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Open video output file:
    out = cv2.VideoWriter('videoout.mp4', fourcc, 20.0, (width, height))

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        animation(timer, ANIMATION_LEN, POSITION_START, ANGLES_START, POSITION_END, ANGLES_END, q_start, q_end)
        OriginalAxis()

        # Read frame:
        screenshot = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        # Convert from binary to cv2 numpy array:
        snapshot = Image.frombuffer("RGB", (width, height), screenshot, "raw", "RGB", 0, 0)
        snapshot = np.array(snapshot)
        snapshot = cv2.flip(snapshot, 0)
        snapshot = cv2.cvtColor(snapshot, cv2.COLOR_BGR2RGB)
        # write frame to video file:
        out.write(snapshot)

        pygame.display.flip()
        pygame.time.wait(10)
        timer += 1

        if timer >= ANIMATION_LEN + WAIT_AFTER_END:
            out.release()
            pygame.quit()
            quit()




main()