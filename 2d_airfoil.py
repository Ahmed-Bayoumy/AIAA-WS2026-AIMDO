# ------------------------------------------------------------------------------------#
#  2-D airfoil parameterization and transonic flow simulation                         #
#                                                                                     #
#  Author: Ahmed H. Bayoumy                                                           #
#  email: ahmed.bayoumy@mail.mcgill.ca                                                #
#                                                                                     #
#  This program is free software: you can redistribute it and/or modify it under the  #
#  terms of the GNU Lesser General Public License as published by the Free Software   #
#  Foundation, either version 3 of the License, or (at your option) any later         #
#  version.                                                                           #
#                                                                                     #
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY    #
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A    #
#  PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.   #
#                                                                                     #
#  You should have received a copy of the GNU Lesser General Public License along     #
#  with this program. If not, see <http://www.gnu.org/licenses/>.                     #
#                                                                                     #
#  You can find information on simple_mads at                                         #
#  https://github.com/Ahmed-Bayoumy/OMADS/tests/CFD/airfoil                 #
# ------------------------------------------------------------------------------------#

from cmath import pi
from logging import warn
import shutil
import sys
import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, List
from abc import ABC
import pandas as pd
from matplotlib import pyplot as plt
import os
from shutil import copy
from subprocess import CalledProcessError, Popen, PIPE
import copy
from numpy import linspace, zeros, ones, sin, cos, arctan, pi
import subprocess
import scipy.integrate as integrate
from scipy.optimize import minimize, Bounds
import OMADS
from sys import platform
try:
  import torch  # noqa: F401
except Exception:
  torch = None

from surrogate_ensemble import BayesianEnsembleSurrogateSingleConstraint

import sys

PARAVIEW_SITE_PACKAGES = r"C:\SU2\ParaView-6.1.1-Windows-Python3.12-msvc2017-AMD64\bin\Lib\site-packages"
PARAVIEW_PATH = r"C:\SU2\ParaView-6.1.1-Windows-Python3.12-msvc2017-AMD64\bin"

if PARAVIEW_SITE_PACKAGES not in sys.path:
  sys.path.insert(0, PARAVIEW_SITE_PACKAGES)
if PARAVIEW_PATH not in sys.path:
  sys.path.append(PARAVIEW_PATH)

try:
  import paraview
except Exception as exc:
  paraview = None
  warn(f"ParaView Python modules are unavailable ({exc}). Visualization helpers that depend on ParaView are disabled.")

@dataclass
class airfoil_generator:
  """ Class for generating airfoil 2d geometries """
  chord: float = 1.
  _nCPs: int = 3
  nPts: int = 3
  degree_t: int = 4
  degree_b: int = 4
  params: List[float] = None
  te_gap: float = 0.001263
  _CPt: List[float] = None
  _CPb: List[float] = None
  _t: List[float] = None
  _dim: int = 2
  xt: List = None
  xb: List = None
  yt: List = None
  yb: List = None
  thickness: List = None
  xs: List = None
  ys: List = None
  Pxt: List = None
  Pyt: List = None
  Pxb: List = None
  Pyb: List = None
  cpx: List = None
  cpu: List = None
  cpl: List = None
  Xu: List = None
  Zu: List = None
  Xl: List = None
  Zl: List = None
  num_pts: int = 64
  CSA: float = 0.
  CPmax: float = None

  def __init__(self, chord = 1, nPts = 3, degree_t = 4, degree_b = 4, params = None, tegap = 0.001263):
    self.te_gap = tegap
    self.chord = chord
    self.nPts = nPts
    self.degree_t = degree_t
    self.degree_b = degree_b
    self.params = params

  def generate(self):
    self.Pxt = np.zeros((self.nPts+2))
    self.Pyt = np.zeros((self.nPts+2))
    self.Pxb = np.zeros((self.nPts+2))
    self.Pyb = np.zeros((self.nPts+2))

    for i in range(self.nPts-1):
      dob_i = (2*(i+1))-1
      # x-coordinates of top
      self.Pxt[i+2] = self.params[dob_i]
      # x-coordinates of bottom
      self.Pxb[i+2] = self.params[dob_i-1+2*self.nPts]
      # y-coordinates of top
      self.Pyt[i+1] = self.params[dob_i-1]
      # y-coordinates of bottom
      self.Pyb[i+1] = self.params[dob_i-2+2*self.nPts]

    #set last y-points
    self.Pyt[self.nPts]=self.params[2*(self.nPts)-2]
    self.Pyb[self.nPts]=self.params[4*(self.nPts)-3]
    #set trailing edge points
    self.Pxt[self.nPts+1]=1
    self.Pxb[self.nPts+1]=1
    self.Pyt[self.nPts+1]=0.5* self.te_gap
    self.Pyb[self.nPts+1]=-0.5* self.te_gap
    
    self._CPt = []
    self._CPb = []
    for i in range(len(self.Pxt)):
      self._CPt.append([self.Pxt[i], self.Pyt[i], 0.])
      self._CPb.append([self.Pxb[i], self.Pyb[i], 0.])

    # top x and y values
    self.b_spline_f(self.degree_t, self._CPt, self.num_pts, "top")

    # bottom x and y values
    self.b_spline_f(self.degree_b, self._CPb, self.num_pts, "bottom")

    xt = self.xt
    xb = [x for x in self.xb]
    yt = self.yt
    yb = [y for y in self.yb]

    self.thickness = np.zeros((self.num_pts, 1))

    for i in range(1,self.num_pts-1):
      #find nearest x-location on bottom surface
      #first check bottom point with the same index, and see if points adjacent are nearer or closer to to xt
      dist0=abs(xt[i]-xb[i])             #point with same index
      dist_right=abs(xt[i]-xb[i+1])		   #point with index nearer to TE
      dist_left=abs(xt[i]-xb[i-1])		     #point with index nearer LE
      if dist0<dist_right and dist0<dist_left:
        self.thickness[i] = yt[i] - yb[i]
      elif dist0>dist_right and dist0<dist_left:
        j = i + 1
        while j<99 and abs(xt[i]-xb[j+1])<abs(xt[i]-xb[j]):
          j+=1
        if j == 148 and abs(xt[i]-xb[j+1])<abs(xt[i]-xb[j]):
          j+=1
        self.thickness[i] = yt[i] -yb[j]
      elif dist0<dist_right and dist0 > dist_left:
        j=i-1
        while j>0 and abs(xt[i]-xb[j-1])<abs(xt[i]-xb[j]):
          j-=1
        if j == 1 and abs(xt[i] - xb[j-1])<abs(xt[i]-xb[j]):
          j-=1
        self.thickness[i]=yt[i]-yb[j]

    self.thickness[self.num_pts-1] = self.te_gap

    # max airfoil thickness
    t_max = max(self.thickness)
    # min airfoil thickness
    t_min = min(self.thickness)

    # Trailing edge wedge angle
    sl_bot = (yb[self.num_pts-4]-yb[self.num_pts-1])/(xb[self.num_pts-5]-xb[self.num_pts-1])
    sl_top = (yt[self.num_pts-4]-yt[self.num_pts-1])/(xt[self.num_pts-5]-xt[self.num_pts-1])

    te_ang = (np.arctan(sl_bot)-np.arctan(sl_top))*180/np.pi
    self.Xu = copy.deepcopy(xt)
    self.Zu = copy.deepcopy(yt)
    self.Xl = copy.deepcopy(xb)
    self.Zl = copy.deepcopy(yb)
    self.xs: List = np.flip(xt).tolist()
    self.ys: List = np.flip(yt).tolist()
    [self.xs.append(xb[i]) for i in range(len(xb))]
    [self.ys.append(yb[i]) for i in range(len(yb))]
    out = []
    for i in range(len(self.xs)):
      out.append([self.xs[i], self.ys[i]])
    
    return self.xs,self.ys,t_max,t_min,te_ang

  def visualize(self):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    
    ax1.set_xlim([0,1])
    ax1.set_ylim([-0.3, 0.3])
    ax1.plot(self.xs, self.ys)
    ax1.plot(self.Pxt, self.Pyt, '--o')
    ax1.plot(self.Pxb, self.Pyb, '--o')
    ax2.set_xlim([0,1])
    if self.cpl is not None and self.cpx is not None and self.cpu is not None:
      ax2.plot(self.cpx, self.cpl, '-g')
      ax2.plot(self.cpx, self.cpu, '-')
    plt.ion()
    plt.show()
    plt.pause(0.1)
    # ax1.cla()
    # ax2.cla()
    


  def b_spline_f(self, d, p, n_pts, side: str):
    # number of control points - 1
    n = len(p)-1
    type = 1

    if len(p[0]) < 2 :
      raise IOError('Control points should have at least x and y components.')
    elif len(p[0])-1 > 2:
      raise IOError('Incorrect control points definition. P should have as many rows as the spatial dimensions (either 2 or 3).')

    if d > n:
      raise IOError('Degree must be less than the number of control points.')
    
    if d <1:
      raise IOError('Degree must be an integer 1 or greater.')
    self._dim = len(p[0])
    #Create knots vector
    knots=np.zeros(d+n+2)
    for i in range(d+n+2):
        if i<d+2-1:
            knots[i]=0
        elif d+2-1<=i and i<n+2-1:
            knots[i]=(i+1-d-1)/(n+1-d)
        else:
            knots[i]=1

    #orm t vector
    if type==1:  #linear spacing
        self.t=np.linspace(0,1,n_pts)
    else:        #log spacing
        self.t=np.logspace(-1,np.log10(1.1),n_pts)-0.1
    
    Q = np.zeros((self._dim, n_pts))
    N = np.zeros(n+1)

    for l in range(n_pts):
      l
      for i in range(-1, n):
        v = self.basis_fun(d, i+1, knots, self.t[l])
        N[i+1] = v
        #COMPLETE: Check the rest of this loop
        st = [list(x) for x in zip(*p)]
        s = [x[i+1]*N[i+1] for x in st]
        # 
        Q[:, l] = np.add(Q[:, l], s)
    
    if side == "top":
      self.xt = Q[0,:]
      self.yt = Q[1, :]
    else:
      self.xb = Q[0,:]
      self.yb = Q[1, :]
    

  def basis_fun(self, j, i, knots, t) -> int:
    m = len(knots)
    val: int = 0
    k = knots
    if j == 0:
      if k[i] <= t and t < k[i+1]:
        val = 1
      elif k[i] <= t and t == k[i+1] and k[i+1] == 1:
        val = 1 
      else:
        val = 0
    else:
      if k[i] < k[i+j]:
        val = (t-k[i])/(k[i+j]-k[i]) * self.basis_fun(j-1, i, k, t)
      else:
        val = 0
      if i < m:
        if k[i+1] < k[i+j+1]:
          val += (k[i+j+1]-t)/(k[i+j+1]-k[i+1])*self.basis_fun(j-1, i+1, k, t)
    return val
  
  def calc_cross_sectional_area(self, x1, y1, x2, y2):
    a1 = integrate.simpson(np.add(y1, 10), x1)
    a2 = integrate.simpson(np.add(y2, 10), x2)
    self.CSA = abs(a2-a1)

  def create_TSFOIL_sim_files(self, initFile:str = 'Initial.inp', id: int = 0):
    # COMPLETE: Prepare required input files to run both TSFOIL and OPENFOAM CFD simulations
    I = np.ones((len(self.xs)))
    x1 = np.flip([x for x in self.xs[0:int(len(self.xs)/2)]]).tolist()
    x2 = [x for x in self.xs[int(len(self.xs)/2):]]
    y1 = np.flip([y for y in self.ys[0:int(len(self.ys)/2)]]).tolist()
    y2 = [y for y in self.ys[int(len(self.ys)/2):]]
    y1[-1] = 0.
    y2[-1] = 0.
    x: List = x1
    x+=x2
    y: List = y1
    y+=y2
    L = [x[int(len(x)/2):], y[int(len(y)/2):]]
    U = [x[0:int(len(x)/2)], y[0:int(len(y)/2)]]
    self.calc_cross_sectional_area(x[0:int(len(x)/2)], y[0:int(len(y)/2)], x[int(len(x)/2):], y[int(len(y)/2):])
    src_f = os.path.join(os.getcwd(), 'CFD/TSFOIL/'+initFile) 
    dest_f = os.path.join(os.getcwd(), f'CFD/TSFOIL/Design{id}.inp') 

    shutil.copyfile(src_f, dest_f)
    with open(dest_f, 'a') as f:
      f.write('\n')
      for i in range(len(U[0])):
        f.write(f'   {U[0][i]:1.5f}   {U[1][i]:1.5f} \n')
      f.write(' LOWER SURFACE \n')
      for i in range(len(L[0])):
        f.write(f'   {L[0][i]:1.5f}   {L[1][i]:1.5f} \n')
  
  def run(self, id: int = 0):
    self.old = os.getcwd()
    os.chdir(os.path.join(self.old, 'CFD/TSFOIL'))
    cmd = './tsfoil'
    input_data = os.linesep.join(['default', f'Design{id}.out', f'Design{id}.inp', os.linesep])
    FNULL = open(os.devnull, 'w')
    p = Popen(cmd, stdin=PIPE, bufsize=0, stdout=FNULL, stderr=subprocess.STDOUT)
    p.communicate(input_data.encode('ascii'))
    self.error = False
    if p.returncode != 0:
      # raise CalledProcessError(p.returncode, cmd)
      warn('This is an unattainable point.')
      self.error = True
  
  def get_responses(self, id:int = 0, file_name: str = 'Design0', solver: str = "TSFOIL"):
    if solver == "TSFOIL":
      
      dest_f = os.path.join(os.getcwd(), file_name) 
      f = open(dest_f, 'r')
      h = 0
      index = []
      iBlock = -1
      table = []
      get = False
      for line in f:
        iBlock += 1
        temp = line
        if get and temp.strip().split('   ')[0] != 'AIRFOIL TRAILING EDGE' and temp.strip().split('   ')[0] != 'AIRFOIL LEADING EDGE' and line != '0\n' and line != '0' and line != '\n':
          table.append([float(x) for x in line.strip().split(' ') if x.lstrip('-').replace('.','',1).isdigit()])
        if line == '   I        X          CP          M1              CP          M1\n' or line == '   I        X          CP          M1              CP          M1\n':
          table = []
          get = True
        if line == '0\n':
          get = False
      
      getCL = False
      getCD = False
      f.close()
      f = open(dest_f, 'r')
      CL = []
      CD = []
      k = 0
      for line in f:
        temp = line
        if getCL == True:
          if k == 0:
            CL.append([float(x) for x in line.strip().split(' ') if x.lstrip('-').replace('.','',1).isdigit()][1])
          else:
            CL.append([float(x) for x in line.strip().split(' ') if x.lstrip('-').replace('.','',1).isdigit()][0])
          k += 1
          if k==3:
            getCL = False

        if temp.strip().split('                     ')[0] == 'FINAL MESH':
          getCL = True
      if temp.strip().split(' ')[0] == '0DRAG' and line != '0\n' and line != '0' and line != '\n':
        getCD = True
      if getCD == True:
        CD = [float(x) for x in line.strip().split(' ') if x.lstrip('-').replace('.','',1).isdigit()]
        getCD = False

      f.close()
      self.cpx = [s[1] for s in table]
      self.cpl = [s[3] for s in table]
      self.cpu = [s[5] for s in table]
      self.CL = copy.deepcopy(CL[0])
      self.CPmax = copy.deepcopy(CL[2])
      self.CD = copy.deepcopy(CD[0])
      shutil.move(os.path.join(os.getcwd(), file_name), f'../../../post/Designs/Design{id}.out')
      shutil.move(os.path.join(os.getcwd(), f'Design{id}.inp'), f'../../../post/Designs/Design{id}.inp')
      os.chdir(self.old)
      return self.cpx, self.cpl, self.cpu, self.CL, self.CPmax, self.CD
    elif solver == "SU2":
      f = os.path.join(os.getcwd(), file_name) 
      data = pd.read_csv(f, header=None)
      self.CL = data.iloc[-1, 7]
      self.CD = data.iloc[-1, 6]
      self.LDR = data.iloc[-1, 8]
      # os.chdir(self.old)
      return self.CL, self.CD, self.LDR
    else:
      # os.chdir(self.old)
      raise IOError(f'The flow solver {solver} is not supported! We only support SU2 and TSFOIL.')
    

  def gen_OF_blockmeshdict_from_coords(self, alpha_deg=4):
    """ This function generates an OpenFoam block mesh file from the airfoil coordinates """
    #TODO: This routine is ready for use, but current numerical setup takes too long so will revisit to make it simpler and faster 
    Xu = copy.deepcopy(self.Xu)
    Zu = copy.deepcopy(self.Zu)
    Xl = copy.deepcopy(self.Xl)
    Zl = copy.deepcopy(self.Zl)
    thickness = copy.deepcopy(self.thickness)

    alpha = np.deg2rad(alpha_deg)  # Angle of attack (in radians)
    c = 1.
    

    # Mesh dimensions
    scale = 1            # Scaling factor
    H = 8                # *Half* height of channel
    W = 0.5              # *Half* depth of foil (y-direction)
    D = 16               # Length of downstream section

    # Mesh resolution parameters
    Ni = 400             # Number of interpolation points along the foil
    Nx = 250             # Number of mesh cells along the foil
    ND = 150             # Number of cells in the downstream direction
    NT = 100             # Number of cells the transverse direction
    NW = 1               # Number of cells in the y-direction (along the foil axis)

    # Expansion rates
    ExpT = 500           # Expansion rate in transverse direction
    ExpD = 100           # Expansion rate in the downstream direction
    ExpArc = 50          # Expansion rate along the inlet arc

    # Create a vector with x-coordinates, camber and thickness
    beta = linspace(0, pi, Ni)
    x = c*(0.5*(1 - cos(beta)))
    # plt.plot(Xu, Zu)
    # plt.plot(Xl, Zl)
    # plt.show()
    # Rotate foil to reach specified angle of attack
    upper = np.matrix([[cos(alpha), sin(alpha)],
                       [-sin(alpha), cos(alpha)]])*np.vstack((self.Xu, Zu))
    lower = np.matrix([[cos(alpha), sin(alpha)],
                      [-sin(alpha), cos(alpha)]])*np.vstack((Xl, Zl))

    Xu = upper[0, :].conj().transpose()
    Zu = upper[1, :].conj().transpose()
    Xl = lower[0, :].conj().transpose()
    Zl = lower[1, :].conj().transpose()

    


    # Otherwise use location of max. thickness
    C_max_idx = np.where(thickness == max(thickness))[0][0]


    # Move point of mesh "nose"
    NoseX = (-H + Xu[C_max_idx])*cos(alpha)
    NoseZ = -(-H + Xu[C_max_idx])*sin(alpha)


    # Calculate the location of the vertices on the positive y-axis and put them in a matrix
    nv = 12
    vertices = zeros((nv, 3))

    vertices[0, :] = [NoseX[0], W, NoseZ[0]]
    vertices[1, :] = [Xu[C_max_idx], W, H]
    vertices[2, :] = [Xu[-1], W, H]
    vertices[3, :] = [D, W, H]
    vertices[4, :] = [Xu[0], W, Zu[0]]
    vertices[5, :] = [Xu[C_max_idx], W, Zu[C_max_idx]]
    vertices[6, :] = [Xl[C_max_idx], W, Zl[C_max_idx]]
    vertices[7, :] = [Xu[-1], W, Zu[-1]]
    vertices[8, :] = [D, W, Zu[-1]]
    vertices[9, :] = [Xl[C_max_idx], W, -H]
    vertices[10, :] = [Xu[-1], W, -H]
    vertices[11, :] = [D, W, -H]

    fig = plt.figure()
    ax = plt.axes()


    # ax.plot3D(vertices[:,0], vertices[:,1], vertices[:,2], '--r')
    # ax.view_init(0, -90)
    # [ax.text(vertices[i,0], vertices[i,1], vertices[i,2], str(i)) for i in range(nv)]

    # Create vertices for other side (negative y-axis)
    vertices2 = vertices.copy()
    vertices2[:, 1] *= -1

    # ax.plot3D(vertices2[:,0], vertices2[:,1], vertices2[:,2], '-b')
    # ax.view_init(0, -90)
    # [ax.text(vertices2[i,0], vertices2[i,1], vertices2[i,2], str(i)) for i in range(nv, 2*nv)]
    vertices = np.vstack((vertices, vertices2))




    # Edge 4-5 and 16-17
    pts1 = np.concatenate([Xu[1:C_max_idx], W*ones(np.shape(Xu[1:C_max_idx])),
                           Zu[1:C_max_idx]], axis=1)
    
    pts5 = np.concatenate([pts1[:, 0], -pts1[:, 1], pts1[:, 2]], axis=1)
    


    # Edge 5-7 and 17-19
    pts2 = np.concatenate([Xu[C_max_idx + 1:Ni - 1],
                           W*ones(np.shape(Xu[C_max_idx + 1:Ni - 1])),
                           Zu[C_max_idx + 1:Ni - 1]], axis=1)
    

    pts6 = np.concatenate([pts2[:, 0], -pts2[:, 1], pts2[:, 2]], axis=1)
    


    # Edge 4-6 and 16-18
    pts3 = np.concatenate([Xl[1:C_max_idx], W*ones(np.shape(Xl[1:C_max_idx])),
                           Zl[1:C_max_idx]], axis=1)
    

    pts7 = np.concatenate([pts3[:, 0], -pts3[:, 1], pts3[:, 2]], axis=1)
    


    # Edge 6-7 and 18-19
    pts4 = np.concatenate([Xl[C_max_idx + 1:Ni - 1],
                          W*ones(np.shape(Xl[C_max_idx + 1:Ni - 1])),
                          Zl[C_max_idx + 1:Ni - 1]], axis=1)
    

    pts8 = np.concatenate([pts4[:, 0], -pts4[:, 1], pts4[:, 2]], axis=1)
    


    # Edge 0-1 and 12-13
    pts9 = np.array([-H*cos(pi/4) + Xu[C_max_idx, 0], W, H*sin(pi/4)])
    pts11 = np.array([pts9[0], -pts9[1], pts9[2]])

    # Edge 0-9 and 12-21
    pts10 = np.array([-H*cos(pi/4) + Xu[C_max_idx, 0], W, -H*sin(pi/4)])
    pts12 = np.array([pts10[0], -pts10[1], pts10[2]])


    ax.plot(np.array(pts1)[:,0], np.array(pts1)[:,2], '--r')
    ax.plot(np.array(pts5)[:,0], np.array(pts5)[:,2], '--b')                       
    ax.plot(np.array(pts2)[:,0], np.array(pts2)[:,2], '--r')
    ax.plot(np.array(pts6)[:,0], np.array(pts6)[:,2], '--b')
    ax.plot(np.array(pts3)[:,0], np.array(pts3)[:,2], '--r')
    ax.plot(np.array(pts7)[:,0], np.array(pts7)[:,2], '--b')
    ax.plot(np.array(pts4)[:,0], np.array(pts4)[:,2], '--r')
    ax.plot(np.array(pts8)[:,0], np.array(pts8)[:,2], '--b')

    ax.plot(np.array(pts9)[0], np.array(pts9)[2], '*r')
    ax.plot(np.array(pts11)[0], np.array(pts11)[2], 'ob')

    ax.plot(np.array(pts10)[0], np.array(pts10)[2], '*r')
    ax.plot(np.array(pts12)[0], np.array(pts12)[2], 'ob')

    # Calculate number of mesh points along 4-5 and 4-6
    #N_leading = (C_max_idx/Ni)*Nx
    N_leading = int((x[C_max_idx]/c)*Nx)

    # Calculate number of mesh points along 5-7 and 6-7
    N_trailing = Nx - N_leading

    # Open file
    print(os.getcwd())
    f = open("CFD/openfoam_sim/system/blockMeshDict", "w")


    # Write file
    f.write("/*--------------------------------*- C++ -*----------------------------------*\\ \n")
    f.write("| =========                 |                                                 | \n")
    f.write("| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           | \n")
    f.write("|  \\\\    /   O peration     | Version:  3.0.x                                 | \n")
    f.write("|   \\\\  /    A nd           | Web:      www.OpenFOAM.com                      | \n")
    f.write("|    \\\\/     M anipulation  |                                                 | \n")
    f.write("\\*---------------------------------------------------------------------------*/ \n")
    f.write("FoamFile                                                                        \n")
    f.write("{                                                                               \n")
    f.write("    version     2.0;                                                            \n")
    f.write("    format      ascii;                                                          \n")
    f.write("    class       dictionary;                                                     \n")
    f.write("    object      blockMeshDict;                                                  \n")
    f.write("}                                                                               \n")
    f.write("// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * // \n")
    f.write("\n")
    f.write("convertToMeters %f; \n" % scale)
    f.write("\n")
    f.write("vertices \n")
    f.write("( \n")
    for vertex in vertices:
        f.write("    (%f %f %f)\n" % tuple(vertex))
    f.write("); \n")
    f.write("\n")
    f.write("blocks \n")
    f.write("( \n")
    f.write("    hex (4 5 1 0 16 17 13 12)     (%i %i %i) edgeGrading (1 %f %f 1 %f %f %f %f 1 1 1 1) \n" % (N_leading, NT, NW, 1/ExpArc, 1/ExpArc, ExpT, ExpT, ExpT, ExpT))
    f.write("    hex (5 7 2 1 17 19 14 13)     (%i %i %i) simpleGrading (1 %f 1) \n" % (N_trailing, NT, NW, ExpT))
    f.write("    hex (7 8 3 2 19 20 15 14)     (%i %i %i) simpleGrading (%f %f 1) \n" % (ND, NT, NW, ExpD, ExpT))
    f.write("    hex (16 18 21 12 4 6 9 0)     (%i %i %i) edgeGrading (1 %f %f 1 %f %f %f %f 1 1 1 1) \n" % (N_leading, NT, NW, 1/ExpArc, 1/ExpArc, ExpT, ExpT, ExpT, ExpT))
    f.write("    hex (18 19 22 21 6 7 10 9)    (%i %i %i) simpleGrading (1 %f 1) \n" % (N_trailing, NT, NW, ExpT))
    f.write("    hex (19 20 23 22 7 8 11 10)   (%i %i %i) simpleGrading (%f %f 1) \n" % (ND, NT, NW, ExpD, ExpT))

    f.write("); \n")
    f.write("\n")
    f.write("edges \n")
    f.write("( \n")

    f.write("    spline 4 5 \n")
    f.write("        ( \n")
    for pt in np.array(pts1):
        f.write("            (%f %f %f) \n" % tuple(pt))
    f.write("        ) \n")

    f.write("    spline 5 7 \n")
    f.write("        ( \n")
    for pt in np.array(pts2):
        f.write("            (%f %f %f)\n" % tuple(pt))
    f.write("        ) \n")

    f.write("    spline 4 6 \n")
    f.write("        ( \n")
    for pt in np.array(pts3):
        f.write("            (%f %f %f)\n" % tuple(pt))
    f.write("        ) \n")

    f.write("    spline 6 7 \n")
    f.write("        ( \n")
    for pt in np.array(pts4):
        f.write("            (%f %f %f)\n" % tuple(pt))
    f.write("        ) \n")

    f.write("    spline 16 17 \n")
    f.write("        ( \n")
    for pt in np.array(pts5):
        f.write("            (%f %f %f)\n" % tuple(pt))
    f.write("        ) \n")

    f.write("    spline 17 19 \n")
    f.write("        ( \n")
    for pt in np.array(pts6):
        f.write("            (%f %f %f)\n" % tuple(pt))
    f.write("        ) \n")

    f.write("    spline 16 18 \n")
    f.write("        ( \n")
    for pt in np.array(pts7):
        f.write("            (%f %f %f)\n" % tuple(pt))
    f.write("        ) \n")

    f.write("    spline 18 19 \n")
    f.write("        ( \n")
    for pt in np.array(pts8):
        f.write("            (%f %f %f)\n" % tuple(pt))
    f.write("        ) \n")

    f.write("    arc 0 1 (%f %f %f) \n" % tuple(pts9))
    f.write("    arc 0 9 (%f %f %f) \n" % tuple(pts10))
    f.write("    arc 12 13 (%f %f %f) \n" % tuple(pts11))
    f.write("    arc 12 21 (%f %f %f) \n" % tuple(pts12))

    f.write("); \n")
    f.write("\n")
    f.write("boundary \n")
    f.write("( \n")

    f.write("    inlet \n")
    f.write("    { \n")
    f.write("        type patch; \n")
    f.write("        faces \n")
    f.write("        ( \n")
    f.write("            (1 0 12 13) \n")
    f.write("            (0 9 21 12) \n")
    f.write("        ); \n")
    f.write("    } \n")
    f.write("\n")

    f.write("    outlet \n")
    f.write("    { \n")
    f.write("        type patch; \n")
    f.write("        faces \n")
    f.write("        ( \n")
    f.write("            (11 8 20 23) \n")
    f.write("            (8 3 15 20) \n")
    f.write("        ); \n")
    f.write("    } \n")
    f.write("\n")

    f.write("    topAndBottom \n")
    f.write("    { \n")
    f.write("        type patch; \n")
    f.write("        faces \n")
    f.write("        ( \n")
    f.write("            (3 2 14 15) \n")
    f.write("            (2 1 13 14) \n")
    f.write("            (9 10 22 21) \n")
    f.write("            (10 11 23 22) \n")
    f.write("        ); \n")
    f.write("    } \n")
    f.write("\n")

    f.write("    airfoil \n")
    f.write("    { \n")
    f.write("        type wall; \n")
    f.write("        faces \n")
    f.write("        ( \n")
    f.write("            (5 4 16 17) \n")
    f.write("            (7 5 17 19) \n")
    f.write("            (4 6 18 16) \n")
    f.write("            (6 7 19 18) \n")
    f.write("        ); \n")
    f.write("    } \n")
    f.write("); \n")
    f.write(" \n")
    f.write("mergePatchPairs \n")
    f.write("( \n")
    f.write("); \n")
    f.write(" \n")
    f.write("// ************************************************************************* // \n")

    # Close file
    f.close()

  def gmsh_generator(self, path:str):
    """ This is the generator of the 2D spatial domain mesh from the airfoil 2d Cartesian points"""
    #COMPLETE: Tested and ready for use

    file = open(os.path.join(os.getcwd(), path + '/airfoil.geo'), 'w')
    for i in range(len(self.xt)):

        point = 'Point(' + str(i+1) + ') = '
        coords = '{' + str(self.xt[len(self.xt)-i-1]) + ', ' + str(self.yt[i]) + ', ' + '0};' + '\n'

        file.write(point+coords)

    for i in range(len(self.xb)):

        point = 'Point(' + str(i+1+len(self.xt)) + ') = '
        coords = '{' + str(self.xb[i]) + ', ' + str(self.yb[i]) + ', ' + '0};' + '\n'

        file.write(point+coords)

    Line = 'Line(1) = {'

    for i in range(len(self.xt) + len(self.xb)):
        if not(i == len(self.xt) + len(self.xb) - 1):
            Line = Line + str(i+1) + ', '
        else:
            Line = Line + str(i+1) + ', 1};'

    file.write(Line)
    file.close()
    old = os.getcwd()
    su2_dir = os.path.join(old, path)
    mesh_geo = os.path.join(su2_dir, 'meshBlocks_2d.geo')
    su2_mesh = os.path.join(su2_dir, 'airfoil.su2')
    try:
      os.chdir(su2_dir)
      try:
        import gmsh
      except ImportError as exc:
        raise ImportError("gmsh Python package is required. Install with: pip install gmsh") from exc

      try:
        gmsh.initialize()
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.open(mesh_geo)
        gmsh.model.mesh.generate(2)
        gmsh.write(su2_mesh)
      finally:
        try:
          gmsh.finalize()
        except Exception:
          pass

      # Remove the unnecessary Fluid mark from the mesh file
      with open(su2_mesh, 'r+') as f:
        lines = f.readlines()
        # move file pointer to the beginning of a file
        f.seek(0)
        # truncate the file
        f.truncate()
        for i in range(len(lines)):
            if lines[i].startswith('NMARK= 3'):
                lines[i] = f'NMARK= 2 \n'
            elif lines[i].startswith("MARKER_TAG= Fluid"):
                break
        f.writelines(lines[0:i])
        f.close()
    finally:
      os.chdir(old)
    # if p.returncode != 0:
    #   raise CalledProcessError(p.returncode, cmd)    

  def create_openfoam_sim_files(self, alpha_deg=4):
    """"""
    self.gen_OF_blockmeshdict_from_coords(alpha_deg)
    

  def plot_cp(self):
    plt.plot(self.cpx, self.cpl)
    plt.plot(self.cpx, self.cpu)
    plt.title('Cp')
    plt.xlim([0,1])
    plt.show()

  def load_force_coeffs_single(self, time_dir):
    """Load force coefficients into a DataFrame for the given time directory."""
    fpath = "postProcessing/forceCoeffs/{}/forceCoeffs.dat".format(time_dir)
    data = np.loadtxt(fpath, skiprows=9)
    df = pd.DataFrame()
    df["time"] = data[:, 0]
    df["cl"] = data[:, 3]
    df["cd"] = data[:, 2]
    df["cm"] = data[:, 1]
    return df


  def load_force_coeffs(self, steady=False):
      """Load force coefficients from file.

      If steady, the file from the `0` directory is used, and the last values are
      returned. Otherwise, arrays are loaded from the latest file.
      """
      if steady:
          return self.load_force_coeffs_single("0")
      else:
          df = pd.DataFrame()
          time_dirs = sorted(os.listdir("postProcessing/forceCoeffs"))
          for t in time_dirs:
              df = df.append(self.load_force_coeffs_single(t), ignore_index=True)
      return df.sort_values(by="time")

  def runSU2CFD(self, path, nproc: int = 3):
    old = os.getcwd()
    os.chdir(os.path.join(old, path))
    try:
      def _read_cfg_value(cfg_lines, key: str):
        key_up = key.upper()
        for line in cfg_lines:
          stripped = line.strip()
          if not stripped or stripped.startswith('%') or '=' not in stripped:
            continue
          lhs, rhs = stripped.split('=', 1)
          if lhs.strip().upper() == key_up:
            return rhs.strip().split('%', 1)[0].strip().upper()
        return None

      def _set_cfg_value(cfg_lines, key: str, value: str):
        key_up = key.upper()
        out_lines = []
        found = False
        for line in cfg_lines:
          stripped = line.strip()
          if not stripped or stripped.startswith('%') or '=' not in stripped:
            out_lines.append(line)
            continue
          lhs, _ = stripped.split('=', 1)
          if lhs.strip().upper() == key_up:
            out_lines.append(f"{key}= {value}\n")
            found = True
          else:
            out_lines.append(line)
        if not found:
          out_lines.append(f"{key}= {value}\n")
        return out_lines

      cfg_file = "airfoil.cfg"
      with open(cfg_file, 'r') as f:
        cfg_lines = f.readlines()

      conv_flow = _read_cfg_value(cfg_lines, "CONV_NUM_METHOD_FLOW")
      muscl_flow = _read_cfg_value(cfg_lines, "MUSCL_FLOW")
      conv_adjflow = _read_cfg_value(cfg_lines, "CONV_NUM_METHOD_ADJFLOW")
      muscl_adjflow = _read_cfg_value(cfg_lines, "MUSCL_ADJFLOW")
      cfg_updated = False
      centered_flow_schemes = {"JST", "LAX-FRIEDRICH", "LAX_FRIEDRICH", "CUSP"}
      if conv_flow in centered_flow_schemes and muscl_flow == "YES":
        warn(f"{conv_flow} is a centered scheme; forcing MUSCL_FLOW= NO.")
        cfg_lines = _set_cfg_value(cfg_lines, "MUSCL_FLOW", "NO")
        cfg_updated = True
      centered_adjflow_schemes = {"JST", "LAX-FRIEDRICH", "LAX_FRIEDRICH"}
      if conv_adjflow in centered_adjflow_schemes and muscl_adjflow == "YES":
        warn(f"{conv_adjflow} is a centered adjoint scheme; forcing MUSCL_ADJFLOW= NO.")
        cfg_lines = _set_cfg_value(cfg_lines, "MUSCL_ADJFLOW", "NO")
        cfg_updated = True

      if cfg_updated:
        with open(cfg_file, 'w') as f:
          f.writelines(cfg_lines)

      su2_bin = r"C:\SU2\bin"
      env = os.environ.copy()
      path_parts = env.get("PATH", "").split(os.pathsep) if env.get("PATH") else []
      if su2_bin not in path_parts:
        env["PATH"] = su2_bin + os.pathsep + env.get("PATH", "")
      # Keep process-level PATH updated for subsequent calls in the same Python session.
      os.environ["PATH"] = env["PATH"]

      mpi_exec = shutil.which("mpiexec", path=env["PATH"]) or shutil.which("mpirun", path=env["PATH"])
      su2_exec = shutil.which("SU2_CFD", path=env["PATH"]) or shutil.which("SU2_CFD.exe", path=env["PATH"])

      if su2_exec is None:
        raise FileNotFoundError("SU2_CFD executable not found in PATH. Install SU2 and add it to PATH.")

      if mpi_exec is not None:
        cmd = [mpi_exec, "-n", str(nproc), su2_exec, "airfoil.cfg"]
      else:
        warn("MPI launcher not found. Running SU2_CFD in serial mode.")
        cmd = [su2_exec, "airfoil.cfg"]

      def _run_solver(run_cmd):
        return subprocess.run(
          run_cmd,
          shell=False,
          env=env,
          capture_output=True,
          text=True,
          errors="replace"
        )

      p = _run_solver(cmd)
      if p.returncode != 0 and mpi_exec is not None and cmd[0] == mpi_exec:
        warn("MPI SU2 run failed. Retrying in serial mode for diagnosis/recovery.")
        cmd = [su2_exec, "airfoil.cfg"]
        p = _run_solver(cmd)

      if p.returncode != 0:
        history_tail = ""
        history_file = "history.csv"
        if os.path.exists(history_file):
          with open(history_file, "r") as hf:
            history_tail = "".join(hf.readlines()[-15:])

        stderr_tail = "\n".join(p.stderr.splitlines()[-20:]) if p.stderr else ""
        stdout_tail = "\n".join(p.stdout.splitlines()[-20:]) if p.stdout else ""
        raise RuntimeError(
          "SU2_CFD failed.\n"
          f"Command: {' '.join(cmd)}\n"
          f"Return code: {p.returncode}\n"
          f"STDERR tail:\n{stderr_tail}\n"
          f"STDOUT tail:\n{stdout_tail}\n"
          f"history.csv tail:\n{history_tail}"
        )
    finally:
      os.chdir(old)

iter = 0
su2_surrogate = BayesianEnsembleSurrogateSingleConstraint(
  csv_path=os.path.join(os.getcwd(), "CFD/airfoil/post/Transonic-airfoil_run/su2_eval_log.csv"),
  constraint_limit=1.7,
  metric_name="abs_cd",
  initial_designs=20,
  retrain_every_simulated=5,
  ensemble_size=5,
  hidden_size=64,
  epochs=250,
  lr=1e-3,
  max_predicted_streak=4,
  improve_margin=1e-4,
  seed=0,
)


def _bootstrap_su2_surrogate_from_csv():
  csv_path = su2_surrogate.csv_path
  if not os.path.exists(csv_path):
    return

  try:
    hist = pd.read_csv(csv_path)
  except Exception as exc:
    warn(f"Unable to read surrogate history CSV at {csv_path}: {exc}")
    return

  if hist.empty:
    return

  required = {"source", "objective_min", "feasible", "c1"}
  if not required.issubset(set(hist.columns)):
    warn("Existing surrogate history CSV has incompatible schema; starting fresh in-memory history.")
    return

  su2_surrogate.df = hist.copy()
  su2_surrogate.records = hist.to_dict(orient="records")

  sim_df = su2_surrogate.df[su2_surrogate.df["source"] == "simulated"].copy()
  su2_surrogate.total_simulated = int(len(sim_df))
  su2_surrogate.simulated_since_retrain = int(len(sim_df) % max(1, su2_surrogate.retrain_every_simulated))
  su2_surrogate.predicted_streak = 0

  feasible_sim = sim_df[sim_df["feasible"] == True]
  if not feasible_sim.empty:
    su2_surrogate.best_feasible_objective = float(feasible_sim["objective_min"].min())
  else:
    su2_surrogate.best_feasible_objective = np.inf

  if torch is None:
    warn("PyTorch is unavailable: surrogate remains in simulation-only mode.")
    return

  if su2_surrogate.total_simulated >= su2_surrogate.initial_designs:
    try:
      su2_surrogate._fit_from_simulated()
      su2_surrogate.trained = True
    except Exception as exc:
      warn(f"Surrogate warm-start training from CSV failed: {exc}")
      su2_surrogate.trained = False


_bootstrap_su2_surrogate_from_csv()


def _simulate_tsf_outputs(x: List[float], local_iter: int):
  design = airfoil_generator(tegap=0.001263,
  degree_t=3,
  degree_b=3,
  chord = 1,
  nPts = 2,
  params = copy.deepcopy(x)
  )
  design.generate()
  design.create_TSFOIL_sim_files(id=local_iter)
  design.run(id=local_iter)
  design.get_responses(local_iter, f'Design{local_iter}.out', 'TSFOIL')
  design.visualize()

  CL = copy.deepcopy(design.CL)
  CSA = copy.deepcopy(design.CSA)
  CD = copy.deepcopy(design.CD)

  if design.error:
    CL = np.inf
    CSA = np.inf
    CD = np.inf

  del design
  return -float(CL), float(CSA), float(abs(CD))


def evaluateTSF(x):
  global iter
  iter += 1
  objective, csa, abs_cd = _simulate_tsf_outputs(x, iter)
  return [objective, [0.075-csa, abs_cd-0.006]]

def postprocess_flow(file: str, vis:str, iter: int, prop: str):
  if paraview is None:
    warn("postprocess_flow skipped: ParaView Python modules are not available.")
    return
  from paraview.simple import (
    XMLUnstructuredGridReader, SetActiveSource, GetActiveViewOrCreate, Show,
    GetMaterialLibrary, ColorBy, GetColorTransferFunction, GetOpacityTransferFunction,
    GetTransferFunction2D, GetLayout, SaveScreenshot, Delete
  )
  #### disable automatic camera reset on 'Show'
  paraview.simple._DisableFirstRenderCameraReset()

  # create a new 'XML Unstructured Grid Reader'
  flowvtu = XMLUnstructuredGridReader(registrationName='flow.vtu', FileName=[file])
  flowvtu.PointArrayStatus = ['Density', 'Momentum', 'Energy', 'Pressure', 'Temperature', 'Mach', 'Pressure_Coefficient']

  # set active source
  SetActiveSource(flowvtu)

  # get active view
  renderView1 = GetActiveViewOrCreate('RenderView')

  # show data in view
  flowvtuDisplay = Show(flowvtu, renderView1, 'UnstructuredGridRepresentation')

  # trace defaults for the display properties.
  flowvtuDisplay.Representation = 'Surface'
  flowvtuDisplay.ColorArrayName = [None, '']
  flowvtuDisplay.SelectTCoordArray = 'None'
  flowvtuDisplay.SelectNormalArray = 'None'
  flowvtuDisplay.SelectTangentArray = 'None'
  flowvtuDisplay.OSPRayScaleArray = 'Density'
  flowvtuDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
  flowvtuDisplay.SelectOrientationVectors = 'Momentum'
  flowvtuDisplay.ScaleFactor = 1.4030545711517335
  flowvtuDisplay.SelectScaleArray = 'Density'
  flowvtuDisplay.GlyphType = 'Arrow'
  flowvtuDisplay.GlyphTableIndexArray = 'Density'
  flowvtuDisplay.GaussianRadius = 0.07015272855758667
  flowvtuDisplay.SetScaleArray = ['POINTS', 'Density']
  flowvtuDisplay.ScaleTransferFunction = 'PiecewiseFunction'
  flowvtuDisplay.OpacityArray = ['POINTS', 'Density']
  flowvtuDisplay.OpacityTransferFunction = 'PiecewiseFunction'
  flowvtuDisplay.DataAxesGrid = 'GridAxesRepresentation'
  flowvtuDisplay.PolarAxes = 'PolarAxesRepresentation'
  flowvtuDisplay.ScalarOpacityUnitDistance = 0.5092996065907118
  flowvtuDisplay.OpacityArrayName = ['POINTS', 'Density']
  flowvtuDisplay.SelectInputVectors = ['POINTS', 'Momentum']
  flowvtuDisplay.WriteLog = ''

  # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
  flowvtuDisplay.ScaleTransferFunction.Points = [0.8978183269500732, 0.0, 0.5, 0.0, 1.7400438785552979, 1.0, 0.5, 0.0]

  # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
  flowvtuDisplay.OpacityTransferFunction.Points = [0.8978183269500732, 0.0, 0.5, 0.0, 1.7400438785552979, 1.0, 0.5, 0.0]

  #changing interaction mode based on data extents
  renderView1.InteractionMode = '2D'
  renderView1.CameraPosition = [2.984727144241333, 0.0, 10000.0]
  renderView1.CameraFocalPoint = [2.984727144241333, 0.0, 0.0]

  # get the material library
  materialLibrary1 = GetMaterialLibrary()

  # reset view to fit data
  renderView1.ResetCamera(False)

  # set scalar coloring
  ColorBy(flowvtuDisplay, ('POINTS', prop))

  # rescale color and/or opacity maps used to include current data range
  flowvtuDisplay.RescaleTransferFunctionToDataRange(True, False)

  # show color bar/color legend
  flowvtuDisplay.SetScalarBarVisibility(renderView1, True)

  # get color transfer function/color map for 'Mach'
  machLUT = GetColorTransferFunction(prop)

  # get opacity transfer function/opacity map for 'Mach'
  machPWF = GetOpacityTransferFunction(prop)

  # get 2D transfer function for 'Mach'
  machTF2D = GetTransferFunction2D(prop)

  # Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
  machLUT.ApplyPreset('Rainbow Uniform', True)

  # get layout
  layout1 = GetLayout()

  # layout/tab size in pixels
  layout1.SetSize(1630, 542)

  # current camera placement for renderView1
  renderView1.InteractionMode = '2D'
  renderView1.CameraPosition = [0.8660932449657379, 0.10319951299442405, 31.201420484456136]
  renderView1.CameraFocalPoint = [0.8660932449657379, 0.10319951299442405, 0.0]
  renderView1.CameraParallelScale = 1.2534398953829322

  # save screenshot
  SaveScreenshot(f'{vis}/Design{iter}_{prop}.png', renderView1, ImageResolution=[1630, 542])

  Delete(renderView1)
  del renderView1
  Delete(layout1)
  del layout1

  

#   #================================================================
#   # addendum: following script captures some of the application
#   # state to faithfully reproduce the visualization during playback
#   #================================================================

#   #--------------------------------
#   # saving layout sizes for layouts

def _simulate_su2_outputs(x: List[float]):
  global iter
  design = airfoil_generator(tegap=0.001263,
  degree_t=3,
  degree_b=3,
  chord = 1,
  nPts = 3,
  params = copy.deepcopy(x)
  )
  design.generate()
  design.gmsh_generator(os.path.relpath('CFD/SU2'))
  design.runSU2CFD(os.path.relpath('CFD/SU2'))
  L, D, LD = design.get_responses(file_name=os.path.relpath('CFD/SU2/history.csv'), solver='SU2')
  if paraview is not None and os.path.exists("CFD/SU2/flow.vtu"):
    postprocess_flow("CFD/SU2/flow.vtu", "CFD/SU2/post/vis", iter, 'Mach')
    postprocess_flow("CFD/SU2/flow.vtu", "CFD/SU2/post/vis", iter, 'Pressure')

  print(f'L= {L}, D= {D}, LD= {LD}')
  del design
  return -float(L), abs(float(D))

def evaluateSU2(x):
  # TODO: Implement a file management system to contain the simulation files of sampled dataset
  global iter
  iter += 1

  y = su2_surrogate.evaluate(x, _simulate_su2_outputs)
  row_idx = su2_surrogate.df.index[-1]
  last_row = su2_surrogate.df.iloc[-1]
  source = last_row["source"]
  pinf = last_row["prob_infeasible"]
  pni = last_row["prob_no_improve"]

  # Fallback probabilities avoid persistent NaN values when model is not ready.
  if pd.isna(pinf):
    pinf = 1.0 if y[1][0] > 0.0 else 0.0

  if pd.isna(pni):
    sim_df = su2_surrogate.df[su2_surrogate.df["source"] == "simulated"]
    if not sim_df.empty:
      feasible_sim = sim_df[sim_df["feasible"] == True]
      if not feasible_sim.empty:
        ref_obj = float(feasible_sim["objective_min"].min())
      else:
        # If no feasible point exists yet, compare against best observed objective.
        ref_obj = float(sim_df["objective_min"].min())
      pni = 1.0 if y[0] >= (ref_obj - su2_surrogate.improve_margin) else 0.0
    else:
      pni = np.nan

  su2_surrogate.df.loc[row_idx, "prob_infeasible"] = pinf
  su2_surrogate.df.loc[row_idx, "prob_no_improve"] = pni
  su2_surrogate.records[-1]["prob_infeasible"] = float(pinf) if pd.notna(pinf) else np.nan
  su2_surrogate.records[-1]["prob_no_improve"] = float(pni) if pd.notna(pni) else np.nan
  su2_surrogate.df.to_csv(su2_surrogate.csv_path, index=False)

  print(
    f"SU2 eval {iter}: source={source}, obj={y[0]:.6f}, c1={y[1][0]:.6f}, "
    f"P(infeasible)={pinf if pd.notna(pinf) else float('nan'):.4f}, "
    f"P(no-improve)={pni if pd.notna(pni) else float('nan'):.4f}"
  )
  return y


  
# if __name__ == "__main__":
#   x = [0.05,  0.3,0.08,  0.5,0.075,  -0.05,0.3,   -0.075,0.5,   -0.06]
#   evaluateSU2(x)
#   a=0


if __name__ == "__main__":
  # COMPLETE: Ready for use
  design = airfoil_generator(tegap=0.001263,
  degree_t=3,
  degree_b=3,
  chord = 1,
  nPts = 2,
  #        [yt1,  xt2, yt2,  xt3,yt3,    yb1,xb2,     yb2,xb3,    yb3  ]
  # params = [0.05,  0.3,0.08,  0.5,0.075,  -0.05,0.35,   0.02,0.5,   -0.004]
  # ------- CPs on upper camber/ Cps on lower camber/ TE thickness
  params = [0.05,  0.3,0.08,  0.5,0.075,  -0.05,0.3,   -0.075,0.5,   -0.06]
  )

  
  fun : Callable = evaluateSU2
  eval = {"blackbox": fun}
  param = {"name": "Transonic-airfoil",
              "baseline":  [0.05, 0.3, 0.08,   0.5, 0.075, -0.05,    0.3,   -0.075,0.5,   -0.06],
              "lb":        [0.05, 0.2, 0.05,   0.1,   0.05, -0.1,    0.2,    -0.09, 0.4, -0.07],
              "ub":        [0.1,  0.7,  0.3,   0.7,   0.3,   -0.04,    0.5,    -0.05, 0.6, -0.05],
              "var_names": ['d1','d2','d3',  'd4','d5', 'd6',    'd7'],
              "scaling": 1,
              "post_dir": "CFD/airfoil/post"} # You need to define an absolute path to be able to get the MADS.csv results file
  options = {
        "seed": 0,
        "budget": 200,
        "tol": 1E-3,
        "psize_init": 1.,
        "display": True,
        "opportunistic": False,
        "check_cache": True,
        "store_cache": True,
        "collect_y": False,
        "rich_direction": True,
        "precision": "high",
        "save_results": True,
        "save_coordinates": False,
        "save_all_best": False,
        "parallel_mode": False
      }

  data = {"evaluator": eval, "param": param, "options":options}

  out = {}
  out = OMADS.poll.main(data)
  print(out)


  # design.generate()
  # # design.visualize()
  # # design.gmsh_generator(os.path.abspath('CFD/SU2'))
  # # design.runSU2CFD(os.path.abspath('CFD/SU2'))
  # # design.get_responses(os.path.abspath('CFD/SU2/history.csv'), 'SU2')
  # #TODO: install the SML using setup wheel
  # # TODO: include this test in the main RAF file
  # # TODO: Fix RAF bugs
  # design.create_TSFOIL_sim_files()
  # # design.create_openfoam_sim_files(4)
  # design.run()
  # design.get_responses(f'Design{0}.out', 'TSFOIL')
  # design.plot_cp()
