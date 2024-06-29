"""
define grids system
compute sensitivity of magnetic or gravity
forward anomaly
"""

import numpy as np
from numpy import arctan, log, sqrt

class grids:

    def __init__(self, 
                 x_measure_start, x_measure_step, x_measure_end,
                 y_measure_start, y_measure_step, y_measure_end,
                 x_model_start, x_model_step, x_model_end,
                 y_model_start, y_model_step, y_model_end,
                 z_model_start, z_model_step, z_model_end):

        self.x_measure = np.arange(x_measure_start, x_measure_end + 0.001, x_measure_step)
        self.y_measure = np.arange(y_measure_start, y_measure_end + 0.001, y_measure_step)
        self.x_model = np.arange(x_model_start, x_model_end + 0.001, x_model_step)
        self.y_model = np.arange(y_model_start, y_model_end + 0.001, y_model_step)
        self.z_model = np.arange(z_model_start, z_model_end + 0.001, z_model_step)
        self.z_model = self.z_model + 0.001

        self.x_measure /= 1000
        self.y_measure /= 1000
        self.x_model /= 1000
        self.y_model /= 1000
        self.z_model /= 1000

        self.measurenum = len(self.x_measure)*len(self.y_measure)
        self.modelnum = (len(self.x_model)-1)*(len(self.y_model)-1)*(len(self.z_model)-1)

def graCore(X, Y, x, y, z):
    r = sqrt((x - X)**2 + (y - Y)**2 +z**2)
    result = (x-X)*log(r+Y-y) + (y-Y)*log(r+X-x) - z*arctan(-(X-x)*(Y-y)/r/z)
    return result

def magCore(X, Y, x, y, z):
    r = sqrt((x - X)**2 + (y - Y)**2 +z**2)   
    result = arctan(-(X-x)*(Y-y)/r/z)
    return result

def computeSensitivity(Flag, grid):
    
    if Flag == 'gra':
        def Core(X, Y, x, y, z):
            return graCore(X, Y, x, y, z)
        weight = 6.67
    
    if Flag == 'mag':
        def Core(X, Y, x, y, z):
            return magCore(X, Y, x, y, z)
        weight = 10**2
    
    S_temp = np.zeros(( len(grid.y_measure), len(grid.x_measure), \
                        len(grid.x_model), len(grid.y_model), len(grid.z_model) ))

    X, Y = np.meshgrid(grid.x_measure, grid.y_measure)

    for i in range(len(grid.x_model)):
        for j in range(len(grid.y_model)):
            for k in range(len(grid.z_model)):

                S_temp[:, :, i, j, k] = \
                    Core(X, Y, grid.x_model[i], grid.y_model[j], grid.z_model[k])

    S = S_temp[:, :, 1:, 1:, 1:] \
      + S_temp[:, :, 0:-1, 0:-1, 1:] \
      + S_temp[:, :, 0:-1, 1:, 0:-1] \
      + S_temp[:, :, 1:, 0:-1, 0:-1] \
      - S_temp[:, :, 0:-1, 1:, 1:] \
      - S_temp[:, :, 1:, 0:-1, 1:] \
      - S_temp[:, :, 1:, 1:, 0:-1] \
      - S_temp[:, :, 0:-1, 0:-1, 0:-1]

    S = np.reshape(S, (grid.measurenum, grid.modelnum), order = 'F')

    S = S * weight

    return S

def computeNormSensitivity(Flag, grid):

    if Flag == 'gra':
        def Core(X, Y, x, y, z):
            return graCore(X, Y, x, y, z)
        weight = 6.67
    
    if Flag == 'mag':
        def Core(X, Y, x, y, z):
            return magCore(X, Y, x, y, z)
        weight = 10**2

    S_temp1 = np.zeros((len(grid.x_model), len(grid.y_model), len(grid.z_model)))

    for i in range(len(grid.x_model)):
        for j in range(len(grid.y_model)):
            for k in range(len(grid.z_model)):

                S_temp1[i, j, k] = \
                    Core(0, 0, grid.x_model[i], grid.y_model[j], grid.z_model[k])

    S_temp2 = S_temp1[1:, 1:, 1:] \
            + S_temp1[ 0:-1, 0:-1, 1:] \
            + S_temp1[0:-1, 1:, 0:-1] \
            + S_temp1[1:, 0:-1, 0:-1] \
            - S_temp1[0:-1, 1:, 1:] \
            - S_temp1[1:, 0:-1, 1:] \
            - S_temp1[1:, 1:, 0:-1] \
            - S_temp1[0:-1, 0:-1, 0:-1]        

    S_temp2_flipr = np.fliplr(S_temp2)
    S_flipr = np.concatenate((S_temp2_flipr, S_temp2), axis=1)
    del S_temp2_flipr, S_temp2

    S_flipud = np.flipud(S_flipr)
    S_temp = np.concatenate((S_flipud, S_flipr), axis=0)
    del S_flipud, S_flipr   

    xmid = len(grid.x_model) - 1
    ymid = len(grid.y_model) - 1
    xlen = 2 * xmid
    ylen = 2 * ymid

    count = 0

    S =  np.zeros(( (len(grid.x_model)-1) * (len(grid.y_model)-1) * (len(grid.z_model)-1), \
                          len(grid.x_model) * len(grid.y_model) ))

    for i in range(len(grid.x_model)):
        for j in range(len(grid.y_model)):
            
            S_single_temp = S_temp[(xmid - i):(xlen - i), (ymid - j):(ylen - j), :]

            S_single = np.reshape(S_single_temp, (grid.modelnum, 1),order='F')
            S[:, count] = S_single[:, 0]
            count += 1

    S = S.T
    S = S * weight

    return S

def graSensitivity(x_measure_start, x_measure_step, x_measure_end,
                   y_measure_start, y_measure_step, y_measure_end,
                   x_model_start, x_model_step, x_model_end,
                   y_model_start, y_model_step, y_model_end,
                   z_model_start, z_model_step, z_model_end):

    normgrids = False
    if int(x_measure_start) == int(x_model_start):
        if int(x_measure_step) == int(x_model_step):
            if int(x_measure_end) == int(x_model_end):
                if int(y_measure_start) == int(y_model_start):
                    if int(y_measure_step) == int(y_model_step):
                        if int(y_measure_end) == int(y_model_end):
                            normgrids = True

    grid = grids(x_measure_start, x_measure_step, x_measure_end,
                 y_measure_start, y_measure_step, y_measure_end,
                 x_model_start, x_model_step, x_model_end,
                 y_model_start, y_model_step, y_model_end,
                 z_model_start, z_model_step, z_model_end)


    if normgrids == True:
        print('normal grids')
        return computeNormSensitivity('gra', grid)
    if normgrids == False:
        print('not normal grids')
        return computeSensitivity('gra', grid)


def magSensitivity(x_measure_start, x_measure_step, x_measure_end,
                   y_measure_start, y_measure_step, y_measure_end,
                   x_model_start, x_model_step, x_model_end,
                   y_model_start, y_model_step, y_model_end,
                   z_model_start, z_model_step, z_model_end):

    normgrids = False
    if int(x_measure_start) == int(x_model_start):
        if int(x_measure_step) == int(x_model_step):
            if int(x_measure_end) == int(x_model_end):
                if int(y_measure_start) == int(y_model_start):
                    if int(y_measure_step) == int(y_model_step):
                        if int(y_measure_end) == int(y_model_end):
                            normgrids = True

    grid = grids(x_measure_start, x_measure_step, x_measure_end,
                 y_measure_start, y_measure_step, y_measure_end,
                 x_model_start, x_model_step, x_model_end,
                 y_model_start, y_model_step, y_model_end,
                 z_model_start, z_model_step, z_model_end)


    if normgrids == True:
        print('normal grids')
        return computeNormSensitivity('mag', grid)
    if normgrids == False:
        print('not normal grids')
        return computeSensitivity('mag', grid)
                                    

class GraModel():

    def __init__(self,
                 x_measure_start, x_measure_step, x_measure_end,
                 y_measure_start, y_measure_step, y_measure_end,
                 x_model_start, x_model_step, x_model_end,
                 y_model_start, y_model_step, y_model_end,
                 z_model_start, z_model_step, z_model_end):

        self.grid = grids(x_measure_start, x_measure_step, x_measure_end,
                          y_measure_start, y_measure_step, y_measure_end,
                          x_model_start, x_model_step, x_model_end,
                          y_model_start, y_model_step, y_model_end,
                          z_model_start, z_model_step, z_model_end)
    
        self.property = np.zeros((len(self.grid.x_model) - 1, len(self.grid.y_model) - 1, len(self.grid.z_model) - 1))

        self.sensitivity = graSensitivity(x_measure_start, x_measure_step, x_measure_end,
                                          y_measure_start, y_measure_step, y_measure_end,
                                          x_model_start, x_model_step, x_model_end,
                                          y_model_start, y_model_step, y_model_end,
                                          z_model_start, z_model_step, z_model_end)
            
    def forward(self):

        self.property_vector = np.reshape(self.property, (self.grid.modelnum,), order='F')
        self.anomaly_vector = self.sensitivity.dot(self.property_vector)
        self.anomaly = np.reshape(self.anomaly_vector, (len(self.grid.x_measure), len(self.grid.y_measure)))

    def savedata(self, path):

        count = 0
        data = np.zeros((self.grid.measurenum, 3))
        for i in range(len(self.grid.y_measure)):
            for j in range(len(self.grid.x_measure)):

                data[count, 1] = self.grid.y_measure[i]*1000
                data[count, 0] = self.grid.x_measure[j]*1000
                data[count, 2] = self.anomaly[j, i]
                count += 1
        
        path = str(path)+'.dat'
        np.savetxt(path, data)


class MagModel():

    def __init__(self,
                 x_measure_start, x_measure_step, x_measure_end,
                 y_measure_start, y_measure_step, y_measure_end,
                 x_model_start, x_model_step, x_model_end,
                 y_model_start, y_model_step, y_model_end,
                 z_model_start, z_model_step, z_model_end):

        self.grid = grids(x_measure_start, x_measure_step, x_measure_end,
                          y_measure_start, y_measure_step, y_measure_end,
                          x_model_start, x_model_step, x_model_end,
                          y_model_start, y_model_step, y_model_end,
                          z_model_start, z_model_step, z_model_end)
    
        self.property = np.zeros((len(self.grid.x_model) - 1, len(self.grid.y_model) - 1, len(self.grid.z_model) - 1))

        self.sensitivity = magSensitivity(x_measure_start, x_measure_step, x_measure_end,
                                          y_measure_start, y_measure_step, y_measure_end,
                                          x_model_start, x_model_step, x_model_end,
                                          y_model_start, y_model_step, y_model_end,
                                          z_model_start, z_model_step, z_model_end)
            
    def forward(self):

        self.property_vector = np.reshape(self.property, (self.grid.modelnum,), order='F')
        self.anomaly_vector = self.sensitivity.dot(self.property_vector)
        self.anomaly = np.reshape(self.anomaly_vector, (len(self.grid.x_measure), len(self.grid.y_measure)))

    def savedata(self, path):

        count = 0
        data = np.zeros((self.grid.measurenum, 3))
        for i in range(len(self.grid.y_measure)):
            for j in range(len(self.grid.x_measure)):

                data[count, 1] = self.grid.y_measure[i]*1000
                data[count, 0] = self.grid.x_measure[j]*1000
                data[count, 2] = self.anomaly[j, i]
                count += 1
        
        path = str(path)+'.dat'
        np.savetxt(path, data)
