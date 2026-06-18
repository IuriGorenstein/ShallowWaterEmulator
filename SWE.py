import numpy as np                  # Numpy
import matplotlib.pyplot as plt     # Figures
import matplotlib.colors            # Colors
from prettytable import PrettyTable # Tables
import sympy as sp                  # Sympy
import torch, gc, argparse
from tools import *
from scipy.ndimage import convolve

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
cpu = torch.device("cpu")   
#Domain definition
#---------------------
#global constants
erad = 6371220 #Earth Radius - global constant!
oneday = 3600*24  # One day in seconds
omega = 7.292e-05 # Earth rate of rotation
grav = 9.80616    # gravity constantg
 # Units are associated with Earth dimensions

def print_cudamem():
    if torch.cuda.is_available():
        torch.set_default_device(device)
        torch.cuda.empty_cache()
        print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
        print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
        usage,available = torch.cuda.mem_get_info()
        print("Total memory still available for usage: %fGB"%(usage/1024/1024/1024))
        print("Total memory available: %fGB"%(available/1024/1024/1024))
        
class Domain2D:
  def __init__(self, t0=0.0, T=oneday, xi=-np.pi*erad, xf=np.pi*erad, yi=-np.pi*erad, yf=np.pi*erad, dt=60.0, dx=2*np.pi*erad/100.0, dy=2*np.pi*erad/100.0,
               nt=None, mx=None, my=None,CROCO=False):

    self.t0 = t0     # Initial time (seconds)
    self.T = T       # Final time (seconds)
    self.xi = xi     # Leftmost position in space (meters)
    self.xf = xf     # Rightmost position in space (meters)
    self.yi = yi     # Bottom position in space (meters)
    self.yf = yf     # Top position in space (meters)
    self.Lx = xf-xi  # Length of x range
    self.Ly = yf-yi  # Length of y range

    #If Earth scale, add factor for plotting in 1000x km
    self.to_kkm = 1.0
    if self.Lx > 1000:
      self.to_kkm = 1.0/(1000*1000)

    #Time discretization
    if nt==None:
      nt = (T-t0)/dt
      if not nt.is_integer():
        print( "Warning! This dt does not allow integer number of timesteps!! Original dt: ", dt)
        dt = (T-t0)//(int((T-t0)/dt))
        print("   new dt:", dt)
    else:
      dt = (T-t0)/nt
      print("Calculating dt based on given nt. dt = ", dt)

    self.nt = int((T-t0)/dt)                # Number of timesteps
    self.t = np.linspace(t0, T, self.nt+1)  # Discrete times (includes endpoints!)

    # Space discretization (x) - j indexing
    if mx==None:
      mx = (xf-xi)/dx
      if not mx.is_integer():
        print("Warning! This dx does not allow integer number of space volumes!! Original dx=", dx)
        dx = (xf-xi)/(int((xf-xi)/dx))
        print("   new dx=", dx)
    else:
      dx = (xf-xi)/mx
      print("Calculating dx based on given mx. dx = ", dx)

    self.mx = int((xf-xi)/dx)                 # Number of points in space (x)
    self.x = np.linspace(xi, xf-dx, self.mx)  # Points in space (x)
    self.x_half = np.linspace(xi+dx/2.0, xf-dx/2.0, self.mx)  # Half points in space (x)
    self.indj = np.arange(0, self.mx, 1)                #Indexing range for j
    self.indj_p1 = np.arange(1, self.mx+1, 1)%self.mx   #Indexing range for j+1 (circular)
    self.indj_m1 = np.arange(-1, self.mx-1, 1)%self.mx   #Indexing range for j-1 (circular)

import numpy as np                  # Numpy
import matplotlib.pyplot as plt     # Figures
import matplotlib.colors            # Colors
from prettytable import PrettyTable # Tables
import sympy as sp                  # Sympy
import torch, gc, argparse
from tools import *
from model import *
from scipy.ndimage import convolve

#Domain definition
#---------------------
#global constants
erad = 6371220 #Earth Radius - global constant!
oneday = 3600*24  # One day in seconds
omega = 7.292e-05 # Earth rate of rotation
grav = 9.80616    # gravity constantg
 # Units are associated with Earth dimensions

def print_cudamem():
    if torch.cuda.is_available():
        torch.set_default_device(device)
        torch.cuda.empty_cache()
        print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
        print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
        usage,available = torch.cuda.mem_get_info()
        print("Total memory still available for usage: %fGB"%(usage/1024/1024/1024))
        print("Total memory available: %fGB"%(available/1024/1024/1024))
        
class Domain2D:
  def __init__(self, t0=0.0, T=oneday, xi=-np.pi*erad, xf=np.pi*erad, yi=-np.pi*erad, yf=np.pi*erad, dt=60.0, dx=2*np.pi*erad/100.0, dy=2*np.pi*erad/100.0,
               nt=None, mx=None, my=None,CROCO=False):

    self.t0 = t0     # Initial time (seconds)
    self.T = T       # Final time (seconds)
    self.xi = xi     # Leftmost position in space (meters)
    self.xf = xf     # Rightmost position in space (meters)
    self.yi = yi     # Bottom position in space (meters)
    self.yf = yf     # Top position in space (meters)
    self.Lx = xf-xi  # Length of x range
    self.Ly = yf-yi  # Length of y range

    #If Earth scale, add factor for plotting in 1000x km
    self.to_kkm = 1.0
    if self.Lx > 1000:
      self.to_kkm = 1.0/(1000*1000)

    #Time discretization
    if nt==None:
      nt = (T-t0)/dt
      if not nt.is_integer():
        print( "Warning! This dt does not allow integer number of timesteps!! Original dt: ", dt)
        dt = (T-t0)//(int((T-t0)/dt))
        print("   new dt:", dt)
    else:
      dt = (T-t0)/nt
      print("Calculating dt based on given nt. dt = ", dt)

    self.nt = int((T-t0)/dt)                # Number of timesteps
    self.t = np.linspace(t0, T, self.nt+1)  # Discrete times (includes endpoints!)

    # Space discretization (x) - j indexing
    if mx==None:
      mx = (xf-xi)/dx
      if not mx.is_integer():
        print("Warning! This dx does not allow integer number of space volumes!! Original dx=", dx)
        dx = (xf-xi)/(int((xf-xi)/dx))
        print("   new dx=", dx)
    else:
      dx = (xf-xi)/mx
      print("Calculating dx based on given mx. dx = ", dx)

    self.mx = int((xf-xi)/dx)                 # Number of points in space (x)
    self.x = np.linspace(xi, xf-dx, self.mx)  # Points in space (x)
    self.x_half = np.linspace(xi+dx/2.0, xf-dx/2.0, self.mx)  # Half points in space (x)
    self.indj = np.arange(0, self.mx, 1)                #Indexing range for j
    self.indj_p1 = np.arange(1, self.mx+1, 1)%self.mx   #Indexing range for j+1 (circular)
    self.indj_m1 = np.arange(-1, self.mx-1, 1)%self.mx   #Indexing range for j-1 (circular)

    # Space discretization (y) - i indexing
    if my==None:
      my = (yf-yi)/dy
      if not my.is_integer():
        print("Warning! This dy does not allow integer number of space volumes!! Original dx=", dy)
        dy = (yf-yi)/(int((yf-yi)/dy))
        print("   new dy=", dy)
    else:
      dy = (yf-yi)/my
      print("Calculating dy based on given my. dy = ", dy)

    self.my = int((yf-yi)/dy)                 # Number of points in space (y)
    self.y = np.linspace(yi, yf-dy, self.my)  # Points in space (y)
    self.y_half = np.linspace(yi+dy/2.0, yf-dy/2.0, self.my)  # Half points in space (y)

    self.indi = np.arange(0, self.my, 1)  # Indexing range for i
    self.indi_p1 = (self.indi+1)%self.my  # Indexing range for i+1 (circular)
    self.indi_m1 = (self.indi-1)%self.my  # Indexing range for i-1 (circular)

    self.dt = dt   # Timestep (sec)
    self.dx = dx   # Space step (meters) - x
    self.dy = dy   # Space step (meters) - y

    # 2D grids for staggering - used for ploting later
    # Notice the y-axis is flipped!!!!
    # ---------------------------
    #
    #   y_my
    #   |
    #   yi          (xj, yi)
    #   |
    #   |
    #   y0
    #  i/j x0------- xj ------- x_mx
    #
    # Xh, Yh : x,y poins for depth/height h - centers/primal grid (j, i)
    if CROCO:
      # dont flip...
      flippedy = self.y
      flippedhy=self.y_half
      self.flipped = False  # Flip dummy
    else:
      flippedy = self.y[::-1]
      flippedhy=self.y_half[::-1]
      self.flipped = True  # Flip dummy

    self.Xh, self.Yh = np.meshgrid(self.x, flippedy, indexing='xy')
    self.h_ext = ((self.x[0]-self.dx/2)*self.to_kkm, (self.x[-1]+self.dx/2)*self.to_kkm, (self.y[0]-self.dy/2)*self.to_kkm, (self.y[-1]+self.dy/2)*self.to_kkm)

    # Xu, Yu : x,y poins for zonal velocity u - East/West Edges (j+1/2, i)
    self.Xu, self.Yu = np.meshgrid(self.x_half, flippedy, indexing='xy')
    self.u_ext = ((self.x[0])*self.to_kkm, (self.x[-1]+self.dx)*self.to_kkm, (self.y[0]-self.dy/2)*self.to_kkm, (self.y[-1]+self.dy/2)*self.to_kkm)
    #self.u_ext = (self.x_half[0]*self.to_kkm, self.x_half[-1]*self.to_kkm, self.y[0]*self.to_kkm, self.y[-1]*self.to_kkm)

    # Xv, Yv : x,y poins for meridional velocity v - North/South Edges (j, i+1/2)
    self.Xv, self.Yv = np.meshgrid(self.x, flippedhy, indexing='xy')
    self.v_ext = ((self.x[0]-self.dx/2)*self.to_kkm, (self.x[-1]+self.dx/2)*self.to_kkm, (self.y[0])*self.to_kkm, (self.y[-1]+self.dy)*self.to_kkm)
    #self.v_ext = (self.x[0]*self.to_kkm, self.x[-1]*self.to_kkm, self.y_half[0]*self.to_kkm, self.y_half[-1]*self.to_kkm)

    # Xq, Yq : x,y poins for vorticity q - Dual cells (corners) (j+1/2, i+1/2)
    self.Xq, self.Yq = np.meshgrid(self.x_half, flippedhy, indexing='xy')
    self.q_ext = ((self.x[0])*self.to_kkm, (self.x[-1]+self.dx)*self.to_kkm, (self.y[0])*self.to_kkm, (self.y[-1]+self.dy)*self.to_kkm)
    #self.q_ext = (self.x_half[0]*self.to_kkm, self.x_half[-1]*self.to_kkm, self.y_half[0]*self.to_kkm, self.y_half[-1]*self.to_kkm)

    #Display variables
    if False:
      print("Domain set up:")
      print(" > dt = ", self.dt)
      print(" > nt = ", self.nt)
      print(" > T-T0 = ", self.T-self.t0)
      print(" > dx = ", self.dx)
      print(" > mx = ", self.mx)
      print(" > Lx = ", self.Lx)
      print(" > dy = ", self.dy)
      print(" > my = ", self.my)
      print(" > Ly = ", self.Ly)
      print(" > x e y H = ", self.Xh.shape,self.Yh.shape)
      print(" > x e y U = ", self.Xu.shape,self.Yu.shape)
      print(" > x e y V = ", self.Xv.shape,self.Yv.shape)

class Domain2D_C:
  def __init__(self, t0=0.0, T=oneday, xi=-np.pi*erad, xf=np.pi*erad, yi=-np.pi*erad, yf=np.pi*erad, dt=60.0, dx=2*np.pi*erad/100.0, dy=2*np.pi*erad/100.0,
               nt=None, mx=None, my=None):

    self.t0 = t0     # Initial time (seconds)
    self.T = T       # Final time (seconds)
    self.xi = xi     # Leftmost position in space (meters)
    self.xf = xf     # Rightmost position in space (meters)
    self.yi = yi     # Bottom position in space (meters)
    self.yf = yf     # Top position in space (meters)
    self.Lx = xf-xi  # Length of x range
    self.Ly = yf-yi  # Length of y range

    #If Earth scale, add factor for plotting in 1000x km
    self.to_kkm = 1.0
    if self.Lx > 1000:
      self.to_kkm = 1.0/(1000*1000)

    #Time discretization
    if nt==None:
      nt = (T-t0)/dt
      if not nt.is_integer():
        print( "Warning! This dt does not allow integer number of timesteps!! Original dt: ", dt)
        dt = (T-t0)//(int((T-t0)/dt))
        print("   new dt:", dt)
    else:
      dt = (T-t0)/nt
      print("Calculating dt based on given nt. dt = ", dt)

    self.nt = int((T-t0)/dt)                # Number of timesteps
    self.t = np.linspace(t0, T, self.nt+1)  # Discrete times (includes endpoints!)

    # Space discretization (x) - j indexing
    if mx==None:
      mx = (xf-xi)/dx
      if not mx.is_integer():
        print("Warning! This dx does not allow integer number of space volumes!! Original dx=", dx)
        dx = (xf-xi)/(int((xf-xi)/dx))
        print("   new dx=", dx)
    else:
      dx = (xf-xi)/mx
      print("Calculating dx based on given mx. dx = ", dx)

    self.mx = int((xf-xi)/dx)                 # Number of points in space (x)
    self.x = np.linspace(xi, xf-dx, self.mx)  # Points in space (x)
    self.x_half = np.linspace(xi+dx/2.0, xf-dx/2.0, self.mx-1)
    self.indj = np.arange(0, self.mx, 1)                #Indexing range for j
    self.indj_p1 = np.arange(1, self.mx+1, 1)%self.mx   #Indexing range for j+1 (circular)
    self.indj_m1 = np.arange(-1, self.mx-1, 1)%self.mx   #Indexing range for j-1 (circular)

    # Space discretization (y) - i indexing
    if my==None:
      my = (yf-yi)/dy
      if not my.is_integer():
        print("Warning! This dy does not allow integer number of space volumes!! Original dx=", dy)
        dy = (yf-yi)/(int((yf-yi)/dy))
        print("   new dy=", dy)
    else:
      dy = (yf-yi)/my
      print("Calculating dy based on given my. dy = ", dy)

    self.my = int((yf-yi)/dy)                 # Number of points in space (y)
    self.y = np.linspace(yi, yf-dy, self.my)  # Points in space (y)
    self.y_half = np.linspace(yi+dy/2.0, yf-dy/2.0, self.my-1)
    self.indi = np.arange(0, self.my, 1)  # Indexing range for i
    self.indi_p1 = (self.indi+1)%self.my  # Indexing range for i+1 (circular)
    self.indi_m1 = (self.indi-1)%self.my  # Indexing range for i-1 (circular)

    self.dt = dt   # Timestep (sec)
    self.dx = dx   # Space step (meters) - x
    self.dy = dy   # Space step (meters) - y

    # 2D grids for staggering - used for ploting later
    # Notice the y-axis is flipped!!!!
    # ---------------------------
    #
    #   y_my
    #   |
    #   yi          (xj, yi)
    #   |
    #   |
    #   y0
    #  i/j x0------- xj ------- x_mx
    #
    # Xh, Yh : x,y poins for depth/height h - centers/primal grid (j, i)
    self.Xh, self.Yh = np.meshgrid(self.x, self.y, indexing='xy')
    self.h_ext = ((self.x[0]-self.dx/2)*self.to_kkm, (self.x[-1]+self.dx/2)*self.to_kkm, (self.y[0]-self.dy/2)*self.to_kkm, (self.y[-1]+self.dy/2)*self.to_kkm)

    # Xu, Yu : x,y poins for zonal velocity u - East/West Edges (j+1/2, i)
    self.Xu, self.Yu = np.meshgrid(self.x_half, self.y, indexing='xy')
    self.u_ext = ((self.x[0])*self.to_kkm, (self.x[-1]+self.dx)*self.to_kkm, (self.y[0]-self.dy/2)*self.to_kkm, (self.y[-1]+self.dy/2)*self.to_kkm)
    #self.u_ext = (self.x_half[0]*self.to_kkm, self.x_half[-1]*self.to_kkm, self.y[0]*self.to_kkm, self.y[-1]*self.to_kkm)

    # Xv, Yv : x,y poins for meridional velocity v - North/South Edges (j, i+1/2)
    self.Xv, self.Yv = np.meshgrid(self.x, self.y_half, indexing='xy')
    self.v_ext = ((self.x[0]-self.dx/2)*self.to_kkm, (self.x[-1]+self.dx/2)*self.to_kkm, (self.y[0])*self.to_kkm, (self.y[-1]+self.dy)*self.to_kkm)
    #self.v_ext = (self.x[0]*self.to_kkm, self.x[-1]*self.to_kkm, self.y_half[0]*self.to_kkm, self.y_half[-1]*self.to_kkm)

    # Xq, Yq : x,y poins for vorticity q - Dual cells (corners) (j+1/2, i+1/2)
    self.Xq, self.Yq = np.meshgrid(self.x_half, self.y_half, indexing='xy')
    self.q_ext = ((self.x[0])*self.to_kkm, (self.x[-1]+self.dx)*self.to_kkm, (self.y[0])*self.to_kkm, (self.y[-1]+self.dy)*self.to_kkm)
    #self.q_ext = (self.x_half[0]*self.to_kkm, self.x_half[-1]*self.to_kkm, self.y_half[0]*self.to_kkm, self.y_half[-1]*self.to_kkm)

    #Display variables
    if False:
      print("Domain set up:")
      print(" > dt = ", self.dt)
      print(" > nt = ", self.nt)
      print(" > T-T0 = ", self.T-self.t0)
      print(" > dx = ", self.dx)
      print(" > mx = ", self.mx)
      print(" > Lx = ", self.Lx)
      print(" > dy = ", self.dy)
      print(" > my = ", self.my)
      print(" > Ly = ", self.Ly)
      print(" > x e y H = ", self.Xh.shape,self.Yh.shape)
      print(" > x e y U = ", self.Xu.shape,self.Yu.shape)
      print(" > x e y V = ", self.Xv.shape,self.Yv.shape)

# SWE problem definition
#-----------------------

class SWE_2D:
  def __init__(self, hbar = 10000.0, f = 2*omega, g = grav, ini = 0, dom = Domain2D(),
               Topo=0,random=False,noise_amp=1e-3, seed=None):

    self.random = random
    self.noise_amp = noise_amp

    if seed is not None:
        np.random.seed(seed)
    self.seed = seed

    self.sinfactor = 81
    self.hbar = hbar # Mean depth/height (meters)
    self.f = f       # Coriolis frequency - constant - f-plane (1/s)
    self.g = g       # Gravity acceleration (m/s^2)

    self.dom = dom    # Domain instance
    self.ini = ini    # Initial condition
                      #        0 - Constant wind for advection of hill
                      #        1 - Geostrophic balance
                      #        2 - Unstable jet - steady - See Peixoto & Schreiber 2019 SIAM paper
                      #        3 - Unstable jet - with perturbation - See Peixoto & Schreiber 2019 SIAM paper
                      #        4 - Playground
      
    self.H0 = 100.0   # Constant for initial contions
    self.U0 = 40.0 # Constant for initial contions
    self.eta_b = self.topo(Topo) #Topography (pre-calculated)
    #Display variables
    if False:
      print("Domain set up:")
      print(" > f = ", self.f)
      print(" > g = ", self.g)
      print(" > ini = ", self.ini)
        
  def _add_noise(self, field, scale=1.0, mask = False, smooth = True):
    """
    Add zero-mean Gaussian noise to a field.
    Noise amplitude is relative to scale.
    """
    noise = self.noise_amp * scale * np.random.randn(*field.shape)
    if mask:
        mask = np.abs(field) > scale
        noise = noise*mask
        
    if smooth:
        # Gaussian 3x3 Kernel
        KERNEL_3x3 = np.array([[1, 2, 1],
                               [2, 4, 2],
                               [1, 2, 1]
                            ], dtype=float) / 16.0
        # Neutral 3x3 Kernel
       #KERNEL_3x3 = np.array([[1, 1, 1],
       #                       [1, 1, 1],
       #                       [1, 1, 1]
       #                      ], dtype=float) / 9.0

        noise = convolve(noise,KERNEL_3x3,mode="reflect")
        
    return field + noise
      
  def u0(self):  #Initial condition in u
    if self.ini == 0:
      # Constant
      u0 = 2*np.pi*erad/(12*oneday) #Full rotation in 12 days
      return u0*np.ones_like(self.dom.Xu)

    elif self.ini == 1:
      #Geostrophic balance
      return -(self.g/self.f)*self.H0*np.cos(2*np.pi*self.dom.Yu/(self.dom.Ly))*2*np.pi/(self.dom.Ly)

    elif self.ini == 2 or self.ini == 3:
      #Unstable jet - see Peixoto & Schreiber 2019 SIAM paper
      u = self.U0*np.power(np.sin(2*np.pi*self.dom.Yu/(self.dom.Ly)), self.sinfactor)
        
      if self.random:
            u = self._add_noise(u, scale=self.U0, mask=True)

      return u

    elif self.ini == 4:
      #Rest
      return self.U0*np.power(np.sin(2*np.pi*self.dom.Yu/(self.dom.Ly)), 17)
      #np.zeros_like(self.dom.Xu)

    else:
      #Rest
      return np.zeros_like(self.dom.Xu)

  def v0(self):  #Initial condition in v

    if self.ini == 0:
      #rest
      return np.zeros_like(self.dom.Xv)

    elif self.ini == 1:
      #rest
      return np.zeros_like(self.dom.Xv)

    elif self.ini == 2 or self.ini == 3:
      #rest - see Peixoto & Schreiber 2019 SIAM paper
      v = np.zeros_like(self.dom.Xv)
      if self.random:
        v = self._add_noise(v, scale=self.U0)

      return v

    if self.ini == 4:
      #rest
      return np.zeros_like(self.dom.Xv)

    else:
      #rest
      return np.zeros_like(self.dom.Xv)

  def h0(self):  #Initial condition in h

    if self.ini == 0:
      #Bump
      k = 1000
      p1_x = 0.10*self.dom.Lx+self.dom.xi
      p1_y = 0.70*self.dom.Ly+self.dom.yi

      d1 = (self.dom.Xh-p1_x)**2/(self.dom.Lx**2) + (self.dom.Yh-p1_y)**2/(self.dom.Ly**2)

      bump = 0.1*self.hbar*(np.exp(-k*d1))

      return self.hbar*np.ones_like(self.dom.Yh) + bump

    elif self.ini == 1:
      #Geostrophic balance
      return self.H0*np.sin(2*np.pi*self.dom.Yh/(self.dom.Ly)) + self.hbar*np.ones_like(self.dom.Yh)

    elif self.ini == 2 or self.ini == 3:
      # Unstable jet - see Peixoto & Schreiber 2019 SIAM paper
      z = sp.Symbol('z')
      y = sp.Symbol('y')
      h_f = sp.integrate(sp.sin(2.0*sp.pi*z/self.dom.Ly)**(self.sinfactor), (z, self.dom.yi, y))
      h_f = sp.lambdify(y, h_f)

      pert = np.zeros_like(self.dom.Yh)

      if self.ini == 3:
        #add perturbation
        hpert = 0.02
        if self.hbar == 0:
          hpert=self.eta_b*hpert
        else:
          hpert*=self.hbar
        k = 1000.0
        p1_x = 0.15*self.dom.Lx+self.dom.xi
        p1_x = 0.65*self.dom.Lx+self.dom.xi

        p1_y = 0.75*self.dom.Ly+self.dom.yi
        p2_x = 0.85*self.dom.Lx+self.dom.xi
        p2_x = 0.35*self.dom.Lx+self.dom.xi

        p2_y = 0.25*self.dom.Ly+self.dom.yi
        d1 = (self.dom.Xh-p1_x)**2/(self.dom.Lx**2) + (self.dom.Yh-p1_y)**2/(self.dom.Ly**2)
        d2 = (self.dom.Xh-p2_x)**2/(self.dom.Lx**2) + (self.dom.Yh-p2_y)**2/(self.dom.Ly**2)
        pert = hpert*(np.exp(-k*d1)+np.exp(-k*d2))
        h = self.hbar*np.ones_like(self.dom.Yh) - (self.U0*self.f/self.g)*h_f(self.dom.Yh) - pert
          
        if self.random:
          h = self._add_noise(h, scale=self.hbar)
            
        return h
          
      return self.hbar*np.ones_like(self.dom.Yh) - (self.U0*self.f/self.g)*h_f(self.dom.Yh) - pert
              
        #return self.H0*np.sin(2*np.pi*self.dom.Yh/(self.dom.Ly)) + self.hbar*np.ones_like(self.dom.Yh)
    
    elif self.ini == 4:
      # 2 Bumps
      k = 1000
      p1_x = 0.10*self.dom.Lx+self.dom.xi
      p1_y = 0.70*self.dom.Ly+self.dom.yi
      p2_x = 0.55*self.dom.Lx+self.dom.xi
      p2_y = 0.25*self.dom.Ly+self.dom.yi
      d1 = (self.dom.Xh-p1_x)**2/(self.dom.Lx**2) + (self.dom.Yh-p1_y)**2/(self.dom.Ly**2)
      d2 = (self.dom.Xh-p2_x)**2/(self.dom.Lx**2) + (self.dom.Yh-p2_y)**2/(self.dom.Ly**2)
      pert = 0.1*self.hbar*(np.exp(-k*d1)+np.exp(-k*d2))

      return self.hbar*np.ones_like(self.dom.Yh) + pert

    elif self.ini == 5:
        # Centered cosine bell (smooth, zero at edges)
    
        # Domain center
        xc = self.dom.xi + 0.5 * self.dom.Lx
        yc = self.dom.yi + 0.5 * self.dom.Ly
    
        # Radial distance from center (normalized)
        r = np.sqrt(
            ((self.dom.Xh - xc) / (0.5 * self.dom.Lx))**2 +
            ((self.dom.Yh - yc) / (0.5 * self.dom.Ly))**2
        )
    
        # Radius of support (r <= 1)
        bell = np.zeros_like(r)
    
        mask = r <= 1.0
        p = 10  # sharper
        bell[mask] = (0.5 * (1.0 + np.cos(np.pi * r[mask])))**p
    
        # Amplitude
        A = 0.1 * self.hbar
        
        return self.hbar * np.ones_like(self.dom.Yh) + A * bell

    else:
      # constant
      return self.hbar*np.ones_like(self.dom.Xh)

  def topo(self,Topo=0): #Topography
    if self.ini == 0 :
      return np.ones_like(self.dom.Xh)*Topo

    elif self.ini == 1:
      return np.ones_like(self.dom.Xh)*Topo

    else:
      return np.ones_like(self.dom.Xh)*Topo
        
def plot2D_Zetapanel(Hae, Uae,Vae,
                     Hf,Uf,Vf,Hb,Ub,Vb,
                     dom = Domain2D(),file=None,name='no_name'):
  fontsize=14
  figsize=(14, 10)
  fig, axs = plt.subplots(3,3,figsize=figsize)
  
  if Hb is None:
      Hb,Ub,Vb = Hae-Hf, Uae-Uf,Vae-Vf
        #Plot Target
      axs[0,0].set_title("Zeta Target", fontsize=fontsize)
      Hmin = np.amin(Hae)
      Hmax = np.amax(Hae)
      Href= Hmax - Hmin
      im = axs[0,0].imshow(Hae, interpolation='nearest', extent=dom.u_ext, aspect='auto', vmin=Hmin, vmax=Hmax, cmap=plt.get_cmap('seismic'))
      cbar = plt.colorbar(im)
      cbar.set_label('m', rotation=270, labelpad=+20, size=fontsize)
      axs[0,0].set_ylabel("y (1000 km)", fontsize=fontsize)
        
      axs[1,0].set_title("U Target", fontsize=fontsize)
      cmin = np.amin(Uae)
      cmax = np.amax(Uae)
      Uref=max(abs(cmin),abs(cmax))
      im = axs[1,0].imshow(Uae, interpolation='nearest', extent=dom.u_ext, aspect='auto', vmin=-Uref, vmax=+Uref, cmap=plt.get_cmap('seismic'))
      cbar = plt.colorbar(im)
      cbar.set_label('m', rotation=270, labelpad=+20, size=fontsize)
      axs[1,0].set_ylabel("y (1000 km)", fontsize=fontsize)
        
      axs[2,0].set_title("V Target", fontsize=fontsize)
      cmin = np.amin(Vae)
      cmax = np.amax(Vae)
      Vref=max(abs(cmin),abs(cmax))
      im = axs[2,0].imshow(Vae, interpolation='nearest', extent=dom.u_ext, aspect='auto', vmin=-Vref, vmax=+Vref, cmap=plt.get_cmap('seismic'))
      cbar = plt.colorbar(im)
      cbar.set_label('m', rotation=270, labelpad=+20, size=fontsize)
      axs[2,0].set_ylabel("y (1000 km)", fontsize=fontsize)
    
      #Plot Prediction
      axs[0,1].set_title("Zeta Forward", fontsize=fontsize)
      im = axs[0,1].imshow(Hf, interpolation='nearest', extent=dom.u_ext, aspect='auto', vmin=Hmin, vmax=Hmax, cmap=plt.get_cmap('seismic'))
      cbar = plt.colorbar(im)
      cbar.set_label('m', rotation=270, labelpad=+20, size=fontsize)
      axs[0,1].set_ylabel("y (1000 km)", fontsize=fontsize)
        
      axs[1,1].set_title("Ubar Forward", fontsize=fontsize)
      im = axs[1,1].imshow(Uf, interpolation='nearest', extent=dom.u_ext, aspect='auto', vmin=-Uref, vmax=+Uref, cmap=plt.get_cmap('seismic'))
      cbar = plt.colorbar(im)
      cbar.set_label('m', rotation=270, labelpad=+20, size=fontsize)
      axs[1,1].set_ylabel("y (1000 km)", fontsize=fontsize)
        
      axs[2,1].set_title("Vbar Forward", fontsize=fontsize)
      im = axs[2,1].imshow(Vf, interpolation='nearest', extent=dom.u_ext, aspect='auto', vmin=-Vref, vmax=+Vref, cmap=plt.get_cmap('seismic'))
      cbar = plt.colorbar(im)
      cbar.set_label('m', rotation=270, labelpad=+20, size=fontsize)
      axs[2,1].set_ylabel("y (1000 km)", fontsize=fontsize)
    
      #Plot Bias
      axs[0,2].set_title("Zeta Bias", fontsize=fontsize)
      im = axs[0,2].imshow(Hb, interpolation='nearest', extent=dom.u_ext, aspect='auto', vmin=-Href, vmax=Href, cmap=plt.get_cmap('seismic'))
      cbar = plt.colorbar(im)
      cbar.set_label('m', rotation=270, labelpad=+20, size=fontsize)
      axs[0,2].set_ylabel("y (1000 km)", fontsize=fontsize)
        
      axs[1,2].set_title("Ubar Bias", fontsize=fontsize)
      im = axs[1,2].imshow(Ub, interpolation='nearest', extent=dom.u_ext, aspect='auto', vmin=-Uref, vmax=+Uref, cmap=plt.get_cmap('seismic'))
      cbar = plt.colorbar(im)
      cbar.set_label('m', rotation=270, labelpad=+20, size=fontsize)
      axs[1,2].set_ylabel("y (1000 km)", fontsize=fontsize)
        
      axs[2,2].set_title("Vbar Bias", fontsize=fontsize)
      im = axs[2,2].imshow(Vb, interpolation='nearest', extent=dom.u_ext, aspect='auto', vmin=-Vref, vmax=+Vref, cmap=plt.get_cmap('seismic'))
      cbar = plt.colorbar(im)
      cbar.set_label('m', rotation=270, labelpad=+20, size=fontsize)
      axs[2,2].set_ylabel("y (1000 km)", fontsize=fontsize)
    
      fig.tight_layout()
      if file is None:
          plt.show()
      else:
          plt.savefig(file+name+'.png')
          plt.close()
          
      return fig

  #Plot H
  axs[0,0].set_title("Zeta AE Forward", fontsize=fontsize)
  cmin = np.amin(Hae)
  cmax = np.amax(Hae)
  im = axs[0,0].imshow(Hae, interpolation='nearest', extent=dom.u_ext, aspect='auto', vmin=cmin, vmax=cmax, cmap=plt.get_cmap('seismic'))
  cbar = plt.colorbar(im)
  cbar.set_label('m', rotation=270, labelpad=+20, size=fontsize)
  axs[0,0].set_ylabel("y (1000 km)", fontsize=fontsize)
    
  axs[1,0].set_title("Zeta Forward", fontsize=fontsize)
  im = axs[1,0].imshow(Hf, interpolation='nearest', extent=dom.u_ext, aspect='auto', vmin=cmin, vmax=cmax, cmap=plt.get_cmap('seismic'))
  cbar = plt.colorbar(im)
  cbar.set_label('m', rotation=270, labelpad=+20, size=fontsize)
  axs[1,0].set_ylabel("y (1000 km)", fontsize=fontsize)
    
  axs[2,0].set_title("Zeta Backward", fontsize=fontsize)
  im = axs[2,0].imshow(Hb, interpolation='nearest', extent=dom.u_ext, aspect='auto', vmin=cmin, vmax=cmax, cmap=plt.get_cmap('seismic'))
  cbar = plt.colorbar(im)
  cbar.set_label('m', rotation=270, labelpad=+20, size=fontsize)
  axs[2,0].set_ylabel("y (1000 km)", fontsize=fontsize)

  #Plot U
  axs[0,1].set_title("Ubar AE Forward", fontsize=fontsize)
  cmin = np.amin(Uae)
  cmax = np.amax(Uae)
  cref=max(abs(cmin),abs(cmax))
  im = axs[0,1].imshow(Uae, interpolation='nearest', extent=dom.u_ext, aspect='auto', vmin=-cref, vmax=+cref, cmap=plt.get_cmap('seismic'))
  cbar = plt.colorbar(im)
  cbar.set_label('m', rotation=270, labelpad=+20, size=fontsize)
  axs[0,1].set_ylabel("y (1000 km)", fontsize=fontsize)
    
  axs[1,1].set_title("Ubar Forward", fontsize=fontsize)
  im = axs[1,1].imshow(Uf, interpolation='nearest', extent=dom.u_ext, aspect='auto', vmin=-cref, vmax=+cref, cmap=plt.get_cmap('seismic'))
  cbar = plt.colorbar(im)
  cbar.set_label('m', rotation=270, labelpad=+20, size=fontsize)
  axs[1,1].set_ylabel("y (1000 km)", fontsize=fontsize)
    
  axs[2,1].set_title("Ubar Backward", fontsize=fontsize)
  im = axs[2,1].imshow(Ub, interpolation='nearest', extent=dom.u_ext, aspect='auto', vmin=-cref, vmax=+cref, cmap=plt.get_cmap('seismic'))
  cbar = plt.colorbar(im)
  cbar.set_label('m', rotation=270, labelpad=+20, size=fontsize)
  axs[2,1].set_ylabel("y (1000 km)", fontsize=fontsize)

  #Plot V
  axs[0,2].set_title("Vbar AE Forward", fontsize=fontsize)
  cmin = np.amin(Vae)
  cmax = np.amax(Vae)
  cref=max(abs(cmin),abs(cmax))
  im = axs[0,2].imshow(Vae, interpolation='nearest', extent=dom.u_ext, aspect='auto', vmin=-cref, vmax=+cref, cmap=plt.get_cmap('seismic'))
  cbar = plt.colorbar(im)
  cbar.set_label('m', rotation=270, labelpad=+20, size=fontsize)
  axs[0,2].set_ylabel("y (1000 km)", fontsize=fontsize)
    
  axs[1,2].set_title("Vbar Forward", fontsize=fontsize)
  im = axs[1,2].imshow(Vf, interpolation='nearest', extent=dom.u_ext, aspect='auto', vmin=-cref, vmax=+cref, cmap=plt.get_cmap('seismic'))
  cbar = plt.colorbar(im)
  cbar.set_label('m', rotation=270, labelpad=+20, size=fontsize)
  axs[1,2].set_ylabel("y (1000 km)", fontsize=fontsize)
    
  axs[2,2].set_title("Vbar Backward", fontsize=fontsize)
  im = axs[2,2].imshow(Vb, interpolation='nearest', extent=dom.u_ext, aspect='auto', vmin=-cref, vmax=+cref, cmap=plt.get_cmap('seismic'))
  cbar = plt.colorbar(im)
  cbar.set_label('m', rotation=270, labelpad=+20, size=fontsize)
  axs[2,2].set_ylabel("y (1000 km)", fontsize=fontsize)

  fig.tight_layout()
  if file is None:
      plt.show()
  else:
      plt.savefig(file+name+'.png')
      plt.close()
      
  return fig
    
def plot2D_panel(u, v, h, z, dom = Domain2D(),file=None,name='no_name'):

  # Important remark: matplotlib imshow plots the matrix as we see it printed on the screen
  #      so that is why we reverted the Y axis in the definition of our 2D grid, so that the initial Y is at the bottom
  #      aligned with our indexing (j,i) that starts in the lower corner of the plane

  fontsize=14
  figsize=(14, 10)
  fig, axs = plt.subplots(2,2,figsize=figsize)
  CMAP = "PuOr_r"
    
  #Plot u (edges)
  axs[0,0].set_title("Zonal Velocity (u)", fontsize=fontsize)
  cmin = np.amin(u)
  cmax = np.amax(u)
  cref=max(abs(cmin),abs(cmax))
  im = axs[0,0].imshow(u, interpolation='nearest', extent=dom.u_ext, aspect='auto', vmin=-cref, vmax=+cref, cmap=CMAP)#plt.get_cmap('seismic'))
  cbar = plt.colorbar(im)
  if dom.Lx > 1000: #Earth scale
    cbar.set_label('m/s', rotation=270, labelpad=+20, size=fontsize)
    axs[0,0].set_ylabel("y (1000 km)", fontsize=fontsize)

  # Plot v (edges)
  axs[0,1].set_title("Meridional Velocity (v)", fontsize=fontsize)
  cmin = np.amin(v)
  cmax = np.amax(v)
  cref=max(abs(cmin),abs(cmax))
  im = axs[0,1].imshow(v, interpolation='nearest', extent=dom.v_ext, aspect='auto', vmin=-cref, vmax=+cref, cmap=CMAP)#plt.get_cmap('seismic'))
  cbar = plt.colorbar(im)
  if dom.Lx > 1000: #Earth scale
    cbar.set_label('m/s', rotation=270, labelpad=+20, size=fontsize)

  #Plot h
  axs[1,0].set_title("Height/Depth ($\zeta$)", fontsize=fontsize)
  if dom.Lx > 1000: #Earth scale
    im = axs[1,0].imshow(h/1000, interpolation='nearest', extent=dom.h_ext, aspect='auto', cmap=CMAP)#cmap=plt.get_cmap('jet'))
    axs[1,0].set_xlabel("x (1000 km)", fontsize=fontsize)
    axs[1,0].set_ylabel("y (1000 km)", fontsize=fontsize)
    cbar = plt.colorbar(im)
    cbar.set_label('km', rotation=270, labelpad=+20, size=fontsize)
  else:
    im = axs[1,0].imshow(h, interpolation='nearest', extent=dom.h_ext, aspect='auto', cmap=CMAP)#cmap=plt.get_cmap('jet'))
    axs[1,0].set_xlabel("x", fontsize=fontsize)
    axs[1,0].set_ylabel("y", fontsize=fontsize)
    cbar = plt.colorbar(im)

  #Plot vorticity
  axs[1,1].set_title("Vorticity", fontsize=fontsize) #($\zeta$)
  cmin = np.amin(z)
  cmax = np.amax(z)
  cref=max(abs(cmin),abs(cmax))
  im = axs[1,1].imshow(z, interpolation='nearest', extent=dom.q_ext, aspect='auto', vmin=-cref, vmax=+cref, cmap=CMAP)#plt.get_cmap('seismic'))
  #im = axs[1,1].imshow(z, interpolation='nearest', extent=dom.q_ext, aspect='auto', cmap=plt.get_cmap('seismic'))
  if dom.Lx > 1000: #Earth scale
    axs[1,1].set_xlabel("x (1000 km)", fontsize=fontsize)
  cbar = plt.colorbar(im)
  cbar.set_label('1/s', rotation=270, labelpad=+20, size=fontsize)

  #plt.grid(True)
  #cbar.ax.tick_params(labelsize=fontsize)

  fig.tight_layout()
  if file is None:
      plt.show()
  else:
      plt.savefig(file+name+'.png')
      plt.close()
      
  return fig
    
def plot2D(data, pos="h", dom = Domain2D()):

  # Important remark: matplotlib imshow plots the matrix as we see it printed on the screen
  #      so that is why we reverted the Y axis in the definition of our 2D grid, so that the initial Y is at the bottom
  #      aligned with our indexing (j,i) that starts in the lower corner of the plane

  fontsize=14
  figsize=(8, 6)
  fig, ax = plt.subplots(1,1,figsize=figsize)

  #ax.set_title("Zonal Velocity (u)", fontsize=fontsize)

  if pos == "u":
    #Plot on u edges
    im = ax.imshow(data, interpolation='nearest', extent=dom.u_ext, aspect='auto', cmap=plt.get_cmap('seismic'))
  elif pos == "v":
    #Plot on v edges
    im = ax.imshow(data, interpolation='nearest', extent=dom.v_ext, aspect='auto', cmap=plt.get_cmap('seismic'))
  elif pos == "h":
    #Plot on h points
    im = ax.imshow(data, interpolation='nearest', extent=dom.h_ext, aspect='auto', cmap=plt.get_cmap('seismic'))
  else : #pos == "q":
    #Plot on q points
    im = ax.imshow(data, interpolation='nearest', extent=dom.q_ext, aspect='auto', cmap=plt.get_cmap('seismic'))

  cbar = plt.colorbar(im)
  #cbar.set_label('m/s', rotation=270, labelpad=+20, size=fontsize)
  if dom.Lx > 1000: #Earth scale
    ax.set_xlabel("x (1000 km)", fontsize=fontsize)
    ax.set_ylabel("y (1000 km)", fontsize=fontsize)
  else:
    ax.set_xlabel("x", fontsize=fontsize)
    ax.set_ylabel("y", fontsize=fontsize)

  fig.tight_layout()
  plt.show()
  return fig

# SWE numerical method definition
#-----------------------
#  Numerical grid : C-Grid
#
#
#    Structure near corner (0,0)
#
#  yi
#
#           |          .          |         .        |
#  y1      h01--------u01--------h11-------u11------h21----
#           |          .          |         .        |
#           |          .          |         .        |
#  y1/2    v00........q00........v10.......q10......u20...
#           |          .          |         .        |
#           |          .          |         .        |
#  y0      h00--------u00--------h10-------u10------h20----
#
#  i/j     x0        x1/2         x1     x3/2        x2     x5/2      x3     ....  xj
#
#  Double Periodiciy implies that
#   hmi = h0i,  hjm = hj0
#   umi = u0i,  ujm = uj0
#   vmi = v0i,  vjm = vj0
#
#---------------------------------------------

class SWE_2D_num_method:
  def __init__(self, equation = "swe", method = "en_cons", dom = Domain2D(), sw = SWE_2D() ):
    self.equation = equation # Equation set: adv, swe, lin_swe
    self.method = method # Method to use: en_cons, ...
    self.dom = dom       # Discrete domain
    self.sw = sw         # SW2D problem

    #Rough estimate of gravity wave Courant number
    self.c = np.sqrt(sw.g*sw.hbar)*dom.dt/np.sqrt(dom.dx**2+dom.dy**2)

    print("Gravity wave Courant number")
    print(self.c)

  # Numerical operators
  # ----------------------------
  # pos = origin of data
  #   u , v, h, q points
  #-----------------------

  def components_del_x(self, data, pos = "h", item=False):
    if pos == "h" or pos == "v":
      return [data[:, self.dom.indj_p1],data[:, self.dom.indj]]

    else : #pos == "u" or pos == "q":
      return [data[:, self.dom.indj],data[:, self.dom.indj_m1]]
  def components_del_y(self, data, pos = "h"):
    # Flip y data, since it is stored in reverse way in matrix
    if self.dom.flipped:
      if isinstance(data, np.ndarray):
        data_flip = np.flipud(data)
      else: # torch tensor
        data_flip = torch.flip(data)
    else:
      data_flip = data
        
    if pos == "h" or pos == "u":
      d1 = data_flip[self.dom.indi_p1, :]
      d2 = data_flip[self.dom.indi, :]
    else : #pos == "v" or pos == "q":
      d1 = data_flip[self.dom.indi, :]
      d2 = data_flip[self.dom.indi_m1, :]

    if self.dom.flipped:
      if isinstance(d1, np.ndarray):
         return [np.flipud(d1),np.flipud(d2)] #unflip data
      else:
         return [torch.flip(d1),torch.flip(d2)]
    else:
      return [d1,d2]
        
  def components_bar_x(self, data, pos = "h"):
    if pos == "h" or pos == "v":
      return [data[:, self.dom.indj_p1],data[:, self.dom.indj]]
    else : #pos == "u" or pos == "q":
      return [data[:, self.dom.indj],data[:, self.dom.indj_m1]]

  def components_bar_y(self, data, pos = "h"):
    # Flip y data, since it is stored in reverse way in matrix
    if self.dom.flipped:
      if isinstance(data, np.ndarray):
        data_flip = np.flipud(data)
      else: # torch tensor
        data_flip = torch.flip(data)
    else:
      data_flip = data
        
    if pos == "h" or pos == "u":
      d1 = data_flip[self.dom.indi_p1, :]
      d2 = data_flip[self.dom.indi, :]
    else : #pos == "v" or pos == "q":
      d1 = data_flip[self.dom.indi, :]
      d2 = data_flip[self.dom.indi_m1, :]
        
    if self.dom.flipped:
      if isinstance(d1, np.ndarray):
         return [np.flipud(d1),np.flipud(d2)] #unflip data
      else:
         return [torch.flip(d1),torch.flip(d2)]
    else:
      return [d1,d2]
        
  # finite difference in x
  def del_x(self, data, pos = "h", item=False):
    if pos == "h" or pos == "v":
              #                ([:,j+1] -  [:,j])/dx
      return (data[:, self.dom.indj_p1] - data[:, self.dom.indj])/self.dom.dx

    else : #pos == "u" or pos == "q":
               #            ( [:,j]  -  [:,j-1])/dx
      return (data[:, self.dom.indj] - data[:, self.dom.indj_m1])/self.dom.dx

  # finite difference in y
  def del_y(self, data, pos = "h"):
    # Flip y data, since it is stored in reverse way in matrix
    if self.dom.flipped:
      try: # np.array
        data_flip = np.flipud(data)
      except: # torch tensor
        data_flip = torch.flip(data)
    else:
      data_flip = data
    if pos == "h" or pos == "u":
                #                ([i+1,:] -  [i,:])/dy
      d = (data_flip[self.dom.indi_p1, :] - data_flip[self.dom.indi, :])/self.dom.dy
    else : #pos == "v" or pos == "q":
                #               ([i,:] -  [i-1,:])/dy
      d = (data_flip[self.dom.indi, :] - data_flip[self.dom.indi_m1, :])/self.dom.dy
    if self.dom.flipped:
     return np.flipud(d) #unflip data
    else:
      return d

  # average in x
  def bar_x(self, data, pos = "h"):
    if pos == "h" or pos == "v":
                #              ([:,j+1] +  [:,j])/2
      return (data[:, self.dom.indj_p1] + data[:, self.dom.indj])/2.0

    else : #pos == "u" or pos == "q":
             #              ([:,j] +  [:,j-1])/2
      return (data[:, self.dom.indj] + data[:, self.dom.indj_m1])/2.0

  # average in y
  def bar_y(self, data, pos = "h"):
    # Flip y data, since it is stored in reverse way in matrix
    if self.dom.flipped:
      try: # np.array
        data_flip = np.flipud(data)
      except: # torch tensor
        data_flip = torch.flip(data)
    else:
      data_flip = data
    if pos == "h" or pos == "u":
                   #              ([i+1,:]+ [i,:])/2
      d = (data_flip[self.dom.indi_p1, :] + data_flip[self.dom.indi, :])/2.0
    else : #pos == "v" or pos == "q":
                   #             ([i,:]+ [i-1,:])/2
      d = (data_flip[self.dom.indi, :] + data_flip[self.dom.indi_m1, :])/2.0
    if self.dom.flipped:
     return np.flipud(d) #unflip data
    else:
      return d

  # Relative vorticity
  def rel_vort(self, u, v):
    return self.del_x(v, "v")-self.del_y(u, "u")

  # Potential vorticity
  def pot_vort(self, abs_vort, h):
    return abs_vort/self.bar_y(self.bar_x(h, "h"), "u")

  # Kinetic Energy
  def KE(self, u, v):
    return 0.5*(self.bar_x(u*u, "u")+self.bar_y(v*v, "v"))

  # Total Mass
  def TotMass(self, h):
    return np.sum(np.sum(h))

  # Total Energy
  def TotEnergy(self, h, K):
    return np.sum(np.sum(sw.g*h*h+2.0*h*K))*0.5

  # Total Enstrophy
  def TotEnst(self, pvort, h):
    return np.sum(np.sum(pvort*pvort*self.bar_y(self.bar_x(h, "h"), "u")))*0.5


  # Tendencies Calculation
  # RHS of SWE
  # Defines spatial numerical discretization
  # -----------------------------------------

  def tend(self, u, v, h, new_z=None,AB3=False):
    # Funcionamento:
    ## Ao fazer self.del_x/y(vetor, 'variavel') ou self.bar_x/y(vetor, 'variavel')
    ## 'variavel' se refere a ponto da grade_C esta o vetor.
    
    #Tendency in h
    #--------------------
    if self.equation == "lin_adv" or self.equation == "adv":
        htend = - u*self.del_x(h, "u") - v*self.del_y(h, "v") #Similar to upwind for u>0, v>0

    elif self.equation == "nlin_adv" or self.equation == "swe" or self.equation == "nlin_swe" or self.equation == "AB3AM4":
        uh = self.bar_x(h, "h")*u
        vh = self.bar_y(h, "h")*v
        htend = - self.del_x(uh, "u") - self.del_y(vh, "v")
        if AB3:
            return htend

    elif self.equation == "lin_swe":
        htend = - self.hbar*self.del_x(u, "u") - self.del_y(v, "v")

    else :
        print("Error in numerical method: Unknown equation set")
        return -1

    #Tendency in u
    #--------------------
    if self.equation == "lin_adv" or self.equation == "adv" or self.equation == "nlin_adv":
      utend = np.zeros_like(u)

    elif self.equation == "swe" or self.equation == "nlin_swe":
      K = self.KE(u,v)
      H = self.sw.g*(h + self.sw.eta_b) + K
      abs_vort = self.rel_vort(u, v) + self.sw.f
      q = self.pot_vort(abs_vort, h)
      utend = self.bar_y(q * self.bar_x(vh, "v"), "q") - self.del_x(H, "h")

    elif self.equation == "AB3AM4":
       ### PRESSURE GRADIENT ### 
      cff = self.sw.g
      if isinstance(new_z, np.ndarray): # new_z is a np.array
        grad_ubar = -cff*(self.del_x(new_z*new_z,"h")/2+self.bar_x(self.sw.eta_b,"h")*(self.del_x(new_z,"h")))
      else: # new_z is a tensor (used for carrying gradients in training) #
        grad_ubar = -cff*(self.del_x(new_z*new_z,"h")/2+self.bar_x(torch.from_numpy(self.sw.eta_b),"h")*(self.del_x(new_z,"h")))

       ### ADV ### 
      UFx = (self.bar_x(uh,"u")*self.bar_x(u,"u"))
      UFe = (self.bar_x(vh,"v")*self.bar_y(u,"u"))
      ADV_u = -self.del_x(UFx,"h")-self.del_y(UFe,"q")

       ### COR ### 
      UFx = h*self.sw.f*self.bar_y(v,"v")
      Cor_u = self.bar_x(UFx,"h")

       ### SUM ###
      if isinstance(grad_ubar, np.ndarray):
        utend = grad_ubar  + Cor_u + ADV_u
      else:# new_z is tensor
        utend = grad_ubar  + torch.from_numpy(Cor_u + ADV_u)

    elif self.equation == "lin_swe":
      H = self.sw.g*(h + self.sw.eta_b)
      utend = self.bar_y(self.sw.f * self.bar_x(v, "v"), "q") - self.del_x(H, "h")

    else :
      print("Error in numerical method: Unknown equation set")
      return -1

    #Tendency in v
    #--------------------
    if self.equation == "lin_adv" or self.equation == "adv" or self.equation == "nlin_adv":
      vtend = np.zeros_like(v) #- self.bar_x( q * self.bar_y(uh, "u"), "q") - self.del_y(H, "h")

    elif self.equation == "swe" or self.equation == "nlin_swe":
      vtend = - self.bar_x( q * self.bar_y(uh, "u"), "q") - self.del_y(H, "h")

    elif self.equation == "AB3AM4":
       ### PRESSURE GRADIENT ###
      cff = self.sw.g
      if isinstance(new_z, np.ndarray):
        grad_vbar = -cff*(self.del_y(new_z*new_z,"h")/2+self.bar_y(self.sw.eta_b,"h")*self.del_y(new_z,"h"))
      else:
        grad_vbar = -cff*(self.del_y(new_z*new_z,"h")/2+self.bar_y(torch.from_numpy(self.sw.eta_b),"h")*self.del_y(new_z,"h"))

       ### ADV ### 
      VFe = (self.bar_y(vh,"v")*self.bar_y(v,"v"))
      VFx = (self.bar_y(uh,"u")*self.bar_x(v,"v"))
      ADV_v = -self.del_x(VFx,"q")-self.del_y(VFe,"h")

       ### COR ###
      VFe = h*self.sw.f*self.bar_x(u,"u")
      Cor_v = self.bar_y(VFe,"h")

       ### SUM ###
      if isinstance(grad_vbar, np.ndarray):
        vtend = grad_vbar - Cor_v + ADV_v
      else:# new_z is tensor
        vtend = grad_vbar + torch.from_numpy(- Cor_v + ADV_v)

    elif self.equation == "lin_swe":

      vtend = - self.bar_x(self.sw.f * self.bar_y(u, "u"), "q") - self.del_y(H, "h")

    else :
      print("Error in numerical method: Unknown equation set")
      return -1

    if new_z is not None: # AM4
      return utend, vtend

    return utend, vtend, htend

# Validate operators in numerical methods
def validate_numerical_method(nm = SWE_2D_num_method()):

  error_flag = 0

  # Known periodic function
  def f_ref(x,y):
    return np.cos(2*np.pi*x/(nm.dom.Lx))*np.cos(2*np.pi*y/(nm.dom.Ly))

  def del_x_ref(x,y):
    return (f_ref(x+nm.dom.dx/2, y) - f_ref(x-nm.dom.dx/2, y))/nm.dom.dx

  def del_y_ref(x,y):
    return (f_ref(x, y+nm.dom.dy/2) - f_ref(x, y-nm.dom.dy/2))/nm.dom.dy

  def u_ref(x,y):
    return np.cos(2*np.pi*y/(nm.dom.Ly))

  def v_ref(x,y):
    return np.cos(2*np.pi*x/(nm.dom.Lx))

  def h_ref(x,y):
    return 10000*np.power(np.cos(2*np.pi*x/(nm.dom.Lx)), 80)*np.power(np.cos(2*np.pi*y/(nm.dom.Ly)), 80)

  def vort_ref(x,y):
    return (v_ref(x+nm.dom.dx/2, y) - v_ref(x-nm.dom.dx/2, y))/nm.dom.dx - (u_ref(x, y+nm.dom.dy/2) - u_ref(x, y-nm.dom.dy/2))/nm.dom.dy

  #Validate variable in h position
  # Del_x: h -> u positions (equivalent to centred diferences at u points)
  ref_func = f_ref(nm.dom.Xh, nm.dom.Yh)
  delx_ref_func = del_x_ref(nm.dom.Xu, nm.dom.Yu)
  test_func=nm.del_x(ref_func, pos="h")
  if np.max(np.max(np.abs(test_func-delx_ref_func))) > 10e-12:
    error_flag = error_flag + 1
    print("Warning in validation: Del_x operator not matching analytical")
    plot2D(ref_func, pos = "h", dom=dom)
    plot2D(test_func, pos = "u", dom=dom)
    plot2D(test_func-delx_ref_func, pos = "u", dom=dom)

  # Del_y: h -> v positions (equivalent to centred diferences at v points)
  ref_func = f_ref(nm.dom.Xh, nm.dom.Yh)
  dely_ref_func = del_y_ref(nm.dom.Xv, nm.dom.Yv)
  test_func=nm.del_y(ref_func, pos="h")
  if np.max(np.max(np.abs(test_func-dely_ref_func))) > 10e-12:
    error_flag = error_flag + 1
    print("Warning in validation: Del_y operator not matching analytical")
    plot2D(ref_func, pos = "h", dom=dom)
    plot2D(test_func, pos = "v", dom=dom)
    plot2D(test_func-dely_ref_func, pos = "v", dom=dom)

  #Validate variable in u position
  # Del_x: u -> h positions (equivalent to centred diferences at h points)
  ref_func = f_ref(nm.dom.Xu, nm.dom.Yu)
  delx_ref_func = del_x_ref(nm.dom.Xh, nm.dom.Yh)
  test_func=nm.del_x(ref_func, pos="u")
  if np.max(np.max(np.abs(test_func-delx_ref_func))) > 10e-12:
    error_flag = error_flag + 1
    print("Warning in validation: Del_x operator not matching analytical")
    plot2D(ref_func, pos = "u", dom=dom)
    plot2D(test_func, pos = "h", dom=dom)
    plot2D(test_func-delx_ref_func, pos = "h", dom=dom)


  # Del_y: u -> q positions (equivalent to centred diferences at q points)
  ref_func = f_ref(nm.dom.Xu, nm.dom.Yu)
  dely_ref_func = del_y_ref(nm.dom.Xq, nm.dom.Yq)
  test_func=nm.del_y(ref_func, pos="u")
  if np.max(np.max(np.abs(test_func-dely_ref_func))) > 10e-12:
    error_flag = error_flag + 1
    print("Warning in validation: Del_y operator not matching analytical")
    plot2D(ref_func, pos = "u", dom=dom)
    plot2D(test_func, pos = "q", dom=dom)
    plot2D(test_func-dely_ref_func, pos = "q", dom=dom)

  #Validate variable in v position
  # Del_x: v -> q positions (equivalent to centred diferences at q points)
  ref_func = f_ref(nm.dom.Xv, nm.dom.Yv)
  delx_ref_func = del_x_ref(nm.dom.Xq, nm.dom.Yq)
  test_func=nm.del_x(ref_func, pos="v")
  if np.max(np.max(np.abs(test_func-delx_ref_func))) > 10e-12:
    error_flag = error_flag + 1
    print("Warning in validation: Del_x operator not matching analytical")
    plot2D(ref_func, pos = "v", dom=dom)
    plot2D(test_func, pos = "q", dom=dom)
    plot2D(test_func-delx_ref_func, pos = "q", dom=dom)


  # Del_y: v -> h positions (equivalent to centred diferences at h points)
  ref_func = f_ref(nm.dom.Xv, nm.dom.Yv)
  dely_ref_func = del_y_ref(nm.dom.Xh, nm.dom.Yh)
  test_func=nm.del_y(ref_func, pos="v")
  if np.max(np.max(np.abs(test_func-dely_ref_func))) > 10e-12:
    error_flag = error_flag + 1
    print("Warning in validation: Del_y operator not matching analytical")
    plot2D(ref_func, pos = "v", dom=dom)
    plot2D(test_func, pos = "h", dom=dom)
    plot2D(test_func-dely_ref_func, pos = "h", dom=dom)

  # Vorticity
  ref_u = u_ref(nm.dom.Xu, nm.dom.Yu)
  ref_v = v_ref(nm.dom.Xv, nm.dom.Yv)
  vort_func = vort_ref(nm.dom.Xq, nm.dom.Yq)
  test_func=nm.rel_vort(ref_u, ref_v)
  if np.max(np.max(np.abs(test_func-vort_func)))/np.max(np.max(np.abs(vort_func))) > 10e-12:
    error_flag = error_flag + 1
    print("Warning in validation: vort operator not matching analytical")
    plot2D(ref_u, pos = "u", dom=dom)
    plot2D(ref_v, pos = "v", dom=dom)
    plot2D(test_func, pos = "q", dom=dom)
    plot2D(vort_func, pos = "q", dom=dom)

  #Tendency
  #ref_u = u_ref(nm.dom.Xu, nm.dom.Yu)
  #ref_v = v_ref(nm.dom.Xv, nm.dom.Yv)
  #ref_h = h_ref(nm.dom.Xh, nm.dom.Yh)
  #tu, tv, th = nm.tend( ref_u, ref_v, ref_h)
  #plot2D(ref_u, pos = "u", dom=dom)
  #plot2D(ref_v, pos = "v", dom=dom)
  #plot2D(ref_h, pos = "h", dom=dom)
  #plot2D(th, pos = "h", dom=dom)

  if error_flag == 0:
    print("Numerical validation: Passed! ")
    return
  else:
    print("Numerical validation: Failed! ")
    return

# Time integration step
#-----------------------

## A single step of the RK33 method
def rk33mpas(u0, v0, h0, num_met):
  # Integrates with RK33 of MPAS - See Wicker & Skamarock 2002
  # Wicker, Louis J., and William C. Skamarock. "Time-splitting methods for elastic models using forward time schemes." Monthly weather review 130, no. 8 (2002): 2088-2097.
  # https://www.atmos.albany.edu/facstaff/rfovell/ATM562/wicker-skamarock-2002.pdf

  #We need this copy as each step uses this baseline
  u = np.copy(u0)
  v = np.copy(v0)
  h = np.copy(h0)

  dt = num_met.dom.dt
  dt_step = [dt/3.0, dt/2.0, dt] #RK3
  #dt_step = [dt/2.0, dt] #RK2
  #dt_step = [dt] #euler

  for dts in dt_step: #loop dos estágios RK
    utend, vtend, htend = num_met.tend(u, v, h)
    u = u0 + dts * utend
    v = v0 + dts * vtend
    h = h0 + dts * htend
  #print('rk3, sshInc', np.max(h-h0))
  #print('rk3, uInc', np.max(u-u0))
  return u, v, h #np.copy(u), np.copy(v), np.copy(h)

# A single step of the RK44 method
def rk44(u0, v0, h0, num_met):

  dt = num_met.dom.dt
  utend1, vtend1, htend1 = num_met.tend(u0, v0, h0)
  utend2, vtend2, htend2 = num_met.tend(u0+(dt/2)*utend1, v0+(dt/2)*vtend1, h0+(dt/2)*htend1)
  utend3, vtend3, htend3 = num_met.tend(u0+(dt/2)*utend2, v0+(dt/2)*vtend2, h0+(dt/2)*htend2)
  utend4, vtend4, htend4 = num_met.tend(u0+(dt)*utend3, v0+(dt)*vtend3, h0+(dt)*htend3)

  u = u0 + (dt/6)*(utend1 + 2*utend2 + 2*utend3 + utend4)
  v = v0 + (dt/6)*(vtend1 + 2*vtend2 + 2*vtend3 + vtend4)
  h = h0 + (dt/6)*(htend1 + 2*htend2 + 2*htend3 + htend4)

  return u, v, h

## A single step of the AB3-AM4 method
def AB3AM4(u0,u1,u2,v0,v1,v2,h0,h1,h2,num_met,step='not_ini',model=[None,None],
           Normalização=[[0,1],[0,1],[0,1]],Normalização_out=[[0,1],[0,1],[0,1]],mode = 'forward',printa=False):
    # Here the backward pass is expected to have u0,v0 and h0 the inverted with u2,v2 and h2.
    # In other words: backward mode evolves the 'u0 present' backward.
    #                 forward mode evovles the 'u2 present'forward.
    Topografia = num_met.sw.eta_b
    dt = num_met.dom.dt
        
    if mode == 'backward':
        dt = -dt
        
    if step == 'ini':
      urhs   = u2
      vrhs   = v2
      Drhs   = h2 +Topografia
      cff1 = 0
      cff3 = 0
      cff0 = 1
      cff2 = 0
    elif step == 'second':
      urhs   = u2
      vrhs   = v2
      Drhs   = h2 +Topografia
      cff0= 1.0833333333333    # Logically AB2-AM3 forward-backward
      cff1=-0.1666666666666    # scheme with coefficients chosen for
      cff2= 0.0833333333333    # maximum stability, while maintaining
      cff3= 0.                 # third-accuracy; alpha_max=1.73
      #if mode == 'backward':
      #    cff2=-0.1666666666666    # scheme with coefficients chosen for
      #    cff1= 0.0833333333333    # maximum stability, while maintaining
    else: 
      #if mode == 'forward':
          cff1 = 0.285
          cff3 = 0.013
          cff0 = 0.614
          cff2 = 0.088
          beta=0.281105
          gamma=0.0880
          epsilon=0.013
          urhs   = (3/2+beta)*u2 -(1/2+2*beta)*u1 + (beta)*u0            # lat,lon-1
          vrhs   = (3/2+beta)*v2 -(1/2+2*beta)*v1 + (beta)*v0            # lat-1,lon
          hrhs   = (3/2+beta)*h2 -(1/2+2*beta)*h1 + (beta)*h0
          Drhs   = (3/2+beta)*h2 -(1/2+2*beta)*h1 + (beta)*h0+Topografia # lat,lon
    if model[0] is None:# evovles Zeta with AB3-AM4
        htend = num_met.tend(urhs, vrhs, Drhs, AB3=True)
        if printa:
            print('REAL Vectors AB3-AM4',urhs[0][0], vrhs[0][0], Drhs[0][0])
            print('REAL tendencia AB3',htend[0][0])
        hnew = h2+htend*dt
    else: # evolves Zeta with model
        if mode == 'forward':
            state = [[h2,u2,v2],[h1,u1,v1],[h0,u0,v0]]
        else:
            # Here we need to invert the times 0 and 2 if mode is backward,
            # beacuse NN fucntions use the state vectors in the same position as forward.
            state = [[h0,u0,v0],[h1,u1,v1],[h2,u2,v2]]
        input_tensor = Input_tensor(num_met,state,0,mode,Normalização[0])
        o,_,_,_,_,_,_ = model[0](input_tensor.to(device), n_variables = 1, mode=mode,delta=1) # 1
        update = np.squeeze(o[0].cpu().detach().numpy())
        htend = update  * Normalização_out[0][1] + Normalização_out[0][0]
        hnew = h2+htend.reshape(h2.shape) # *dt

    Dstp   = h2   + Topografia
    Dnew   = hnew + Topografia
    zwrk   = cff0*hnew + cff1*h2  + cff2*h1 + cff3*h0
    if model[1] is None: #evolves v and u with AB3-AM4
        utend,vtend = num_met.tend(urhs, vrhs, Drhs, new_z=zwrk)
        if printa:
            print('REAL tendencia AM4',utend[0][0], vtend[0][0])
        unew = (u2*num_met.bar_x(Dstp,"h")+utend*dt)/num_met.bar_x(Dnew,"h")
        vnew = (v2*num_met.bar_y(Dstp,"h")+vtend*dt)/num_met.bar_y(Dnew,"h")
    else: # evovles u and v with model
        momentum    = [u2,v2]
        numerador   = [num_met.bar_x(Dstp,"h"),num_met.bar_y(Dstp,"h")]
        denominador = [num_met.bar_x(Dnew,"h"),num_met.bar_y(Dnew,"h")]
        for var in [1,2]:
            idx = var-1      
            input_tensor = Input_tensor(num_met,state,var,mode, Normalização[var],zeta_future=hnew) #.detach().cpu().numpy() 
            f,_,_,_,_,_,_  = model[1](input_tensor.to(device),n_variables = 1, mode=mode, delta=0.1) # 0.1
            # Atualiza sistema com tensor de treino com gradientes. Topografia fixo
            update  = np.squeeze(f[0].cpu().detach().numpy()).reshape(u2.shape)
            # Normalize Output #
            update  = update   * Normalização_out[var][1] + Normalização_out[var][0] 
            momentum[idx] = (momentum[idx]*numerador[idx]+update)/denominador[idx] # no *dt

        unew,vnew   = momentum
    return unew,vnew,hnew

# Main time integration loop
#-----------------------------

def time_int(num_met,plot_ktimes = [0],
             model = [None,None],
             Normalização=[None,None,None],
             Normalização_out=[None,None,None],
             mode='forward',save_int=False):
    
  print('código modificado para AB3-AM4')
  
  u2 = num_met.sw.u0() # shape 200x200
  v2 = num_met.sw.v0() # shape 200x200
  h2 = num_met.sw.h0() # shape 200x200  
  start_epoch = 0
  end_epoch   = num_met.dom.nt
  if save_int:
      U2,V2,H2 = [],[],[]
    
  if num_met.equation == 'AB3AM4':
      u2 = num_met.sw.u0() # shape 200x200
      v2 = num_met.sw.v0() # shape 200x200
      h2 = num_met.sw.h0() # shape 200x200
      u,u1 = np.zeros_like(u2),np.zeros_like(u2)
      v,v1 = np.zeros_like(v2),np.zeros_like(v2)
      h,h1 = np.zeros_like(h2),np.zeros_like(h2)
      u1,v1,h1 = AB3AM4(u,u,u2,v,v,v2,h,h,h2,num_met,step='ini',mode=mode)
      if mode == 'forward':
          old_u2,old_v2,old_h2 = u2.copy(),v2.copy(),h2.copy()
          u2, v2, h2 = AB3AM4(u,u2,u1,v,v2,v1,h,h2,h1,num_met,step='second',mode=mode)
          u,v,h = old_u2,old_v2,old_h2
      else:
          u, v, h = AB3AM4(u,u1,u2,v,v1,v2,h,h1,h2,num_met,step='second',mode=mode)
      start_epoch = 2
  else: # Not AB3-AM4, just evolve the whole thing with RK33
    for k in range(start_epoch,end_epoch+2):
        if k in plot_ktimes:
            if save_int:
                U2.append(u2)
                V2.append(v2)
                H2.append(h2)
            else:  
                print(" ")
                print(" Time iteration k=", k-1, " corresponds to time = ", num_met.dom.t[k-1]/oneday, " days")
                plot2D_panel( u2, v2, h2, num_met.rel_vort(u2,v2), num_met.dom)
          # Update RK33 step
        u2, v2, h2 = rk33mpas(u2, v2, h2, num_met)
          # Update RK44
        #u2, v2, h2 = rk44(u2, v2, h2, num_met)

    if len(plot_ktimes) >= 1 :
          print(" ")
          print(" Time final iteration k=", k-1, " corresponds to time = ", num_met.dom.t[k-1]/oneday, " days")
          plot2D_panel( u2, v2, h2, num_met.rel_vort(u2,v2), num_met.dom)

    return u2, v2, h2
      
  # AB3-AM4 Time loop  
  for k in range(start_epoch,end_epoch):
    # PLOT/SAVE System
    if k in plot_ktimes:
      if save_int:
          U2.append(u2)
          V2.append(v2)
          H2.append(h2)
      else:  
          print(" ")
          print(" Time iteration k=", k, " corresponds to time = ", num_met.dom.t[k]/oneday, " days")
          plot2D_panel( u2, v2, h2, num_met.rel_vort(u2,v2), num_met.dom)

    # Update using AB3-AM4
    if model[0] is None:
        if mode == 'forward':
              old_u2,old_v2,old_h2 = u2.copy(),v2.copy(),h2.copy()
              u2,v2,h2 = AB3AM4(u,u1,u2,v,v1,v2,h,h1,h2,num_met,mode=mode)#,model=model,Normalização=Normalização)
              u,v,h    = u1.copy(),v1.copy(),h1.copy()
              u1,v1,h1 = old_u2,old_v2,old_h2
        else:
              old_u,old_v,old_h = u.copy(),v.copy(),h.copy()
              u,v,h    = AB3AM4(u2,u1,u,v2,v1,v,h2,h1,h,num_met,mode=mode)#,model=model,Normalização=Normalização)
              u2,v2,h2 = u1.copy(),v1.copy(),h1.copy()
              u1,v1,h1 = old_u,old_v,old_h
              
    # Update using NN model
    else:
        old_u2,old_v2,old_h2 = u2.copy(),v2.copy(),h2.copy()
        u2,v2,h2 = AB3AM4(u,u1,u2,v,v1,v2,h,h1,h2,num_met,'not_ini',model,
                          Normalização,Normalização_out,mode)
        u,v,h    = u1.copy(),v1.copy(),h1.copy()
        u1,v1,h1 = old_u2,old_v2,old_h2

  # FINISHED SIMULATION #
  if len(plot_ktimes) >= 1 :
    print(" ")
    print(" Time final iteration k=", k+1, " corresponds to time = ", num_met.dom.t[k+1]/oneday, " days")
    plot2D_panel( u2, v2, h2, num_met.rel_vort(u2,v2), num_met.dom)

  if save_int:
      return U2,V2,H2
      
  return u2, v2, h2

def Input_tensor(numet,state,variavel=0,mode='forward',Normalização=None,zeta_future=None,AB3=False):
 
    dt =  numet.dom.dt
    ################################### Generates the Residuals ###############################        
    if variavel==0:     #Zeta
        if AB3:
            b  = pega_incremento2CROCO(numet,state,mode)  # PEGA CELULAS NECESSARIAS PARA O CALCULO DO INCREMENTO DO AB3    
        else:
            b = build_input_vector(numet, state, mode)
            A = add_channels(  torch.from_numpy( b  ).contiguous() )
            return A
    else:
        if AB3:
            b  = pega_momento2CROCO(numet,state,variavel,mode,zeta_future)  # PEGA CELULAS NECESSARIAS PARA O CALCULO DO INCREMENTO DO AM4
        else:
            b = build_input_vectorU(numet, state, variavel, mode, zeta_future)
            A = add_channels(  b.contiguous() )
            return A
    #############################################################################################
    ############## RESHAPES ##############################################
    B  = b                                      #numet.dom.dx ou numet.dom.dy
    DX = np.vstack([np.full((numet.dom.Xh.shape),numet.dom.dx).reshape(1, -1),np.full((numet.dom.Xh.shape),1/numet.dom.dx).reshape(1, -1)])
    DY = np.vstack([np.full((numet.dom.Yh.shape),numet.dom.dy).reshape(1, -1),np.full((numet.dom.Yh.shape),1/numet.dom.dy).reshape(1, -1)])
                                     #dt
    DT = np.full((numet.dom.Yh.shape),dt).reshape(1,-1) # GRADE ZETA
    if isinstance(B, np.ndarray):
        B = add_channels(  torch.from_numpy( B.T         ).contiguous() )
    else:
        B = add_channels(B.T.contiguous() )

    if torch.isnan(B).any():
        print('Nan Values inside the B input vector before Normalization',variavel)
        
    DX = add_channels( torch.from_numpy( DX.T.copy() ).contiguous() )
    DY = add_channels( torch.from_numpy( DY.T.copy() ).contiguous() )
    DT = add_channels( torch.from_numpy( DT.T.copy() ).contiguous() )
    A = torch.concat( (B,DX,DY,DT) ,2) # 

    if torch.isnan(A).any():
        print('Nan Values inside the input vector')
        # Substituir NaNs por zero (ou pela média)
        #A = torch.where(torch.isnan(A), torch.zeros_like(A), A)
        
    return A
    
def pega_momento2CROCO(numet,state,variavel,mode='forward',zeta_new=None): #sis,past,paster,dx,dy,dt,MASCARA):
    #  Lazy Input for Encode-Decode test   #
    #   Grabs the tendency from the AM4    #
    ########################################

    ## OBS: Para acelerar essa parte, eu posso deixar de calcular os rhs e rhs_T, apenas passo os vetores de estado crus
    dx, dy = numet.dom.dx,numet.dom.dy
    dt=numet.dom.dt
    TOPO  = numet.sw.eta_b
    
    beta = 0.281105
    c2 = -(1/2+2*beta)
    cf0 = 0.614
    cf2 = 0.088 
    if mode == 'forward':
        c3 = beta
        c1 = (3/2+beta)
        cf1 = 0.285
        cf3 = 0.013
        times = range(3) 
    else:
        c1 = beta
        c3 = (3/2+beta)
        cf3= 0.285
        cf1 = 0.013
        times = range(2, -1, -1) 
    
    if zeta_new is None:
        print( 'Error, no Zeta_new for computing the AM4 Input Vector Step')
        return
        
    VETOR_var = [[pres*c1 + past*c2 + paster*c3] for pres,past,paster in zip(state[0], state[1], state[2])]
    VETOR_var[0][0] = VETOR_var[0][0]+TOPO
    zwrk = torch.tensor(cf0).to(device)*torch.as_tensor(zeta_new).to(device) + torch.from_numpy(cf1*state[0][0]).to(device)+ torch.from_numpy(cf2*state[1][0]).to(device) + torch.from_numpy(cf3*state[2][0]).to(device)
        
    f, g = numet.sw.f,torch.tensor(numet.sw.g).to(device)
    
    ##### Tendency calculation #####
    if isinstance(zwrk, torch.Tensor):
        zwrk=zwrk.to(device)
    utend,vtend = numet.tend(VETOR_var[1][0], VETOR_var[2][0], VETOR_var[0][0], new_z=zwrk) # .detach().cpu().numpy()

    if variavel == 1:
        VETOR = [utend*dt]
    else:
        VETOR = [vtend*dt]
    #################################
    ######################

    VETOR = [torch.as_tensor(item).to(cpu) for item in VETOR]
    b = torch.stack(VETOR).reshape(len(VETOR), -1)

    ############################################################
    #print('Momentum Input Vector Shape -5 (no dx,1/dx,dy,1/dy,dt) ',b.shape)
    ############################################################

    return b

def build_input_vector(numet, state, mode='forward'):
    """
    state[0] = [h_t,   u_t  , v_t  ]
    state[1] = [h_t-1, u_t-1, v_t-1]
    state[2] = [h_t-2, u_t-2, v_t-2]
    """
    nx, ny = numet.sw.eta_b.shape
    topo_flat = numet.sw.eta_b.flatten() # (N,)
    
    if mode == 'forward':
        h_flat = [s[0].flatten() for s in state]
        u_flat = [s[1].flatten() for s in state]
        v_flat = [s[2].flatten() for s in state]
    else:
        h_flat = [state[i][0].flatten() for i in [2, 1, 0]]
        u_flat = [state[i][1].flatten() for i in [2, 1, 0]]
        v_flat = [state[i][2].flatten() for i in [2, 1, 0]]

    # Função auxiliar para pegar vizinhos de forma flat
    def get_neighbor_flat(flat_array, shift_i, shift_j):
        arr = flat_array.reshape(nx, ny)
        shifted = np.roll(arr, shift=(-shift_i, -shift_j), axis=(0, 1))
        return shifted.flatten()

    # 1. Construir H (20 colunas)
    # Vizinhos: j+1, j-1, j, i+1, i-1
    h_cols = []
    offsets_h = [(0,1), (0,-1), (0,0), (1,0), (-1,0)]
    for si, sj in offsets_h:
        h_cols.append(get_neighbor_flat(h_flat[0], si, sj)) # h2
        h_cols.append(get_neighbor_flat(h_flat[1], si, sj)) # h1
        h_cols.append(get_neighbor_flat(h_flat[2], si, sj)) # h0
        h_cols.append(get_neighbor_flat(topo_flat, si, sj)) # topo

    # 2. Construir U (6 colunas)
    # Vizinhos: (i,j) e (i, j-1)
    u_cols = []
    offsets_u = [(0,0), (0,-1)]
    for si, sj in offsets_u:
        u_cols.append(get_neighbor_flat(u_flat[0], si, sj)) # u2
        u_cols.append(get_neighbor_flat(u_flat[1], si, sj)) # u1
        u_cols.append(get_neighbor_flat(u_flat[2], si, sj)) # u0

    # 3. Construir V (6 colunas)
    # Vizinhos: (i,j) e (i-1, j)
    v_cols = []
    offsets_v = [(0,0), (-1,0)]
    for si, sj in offsets_v:
        v_cols.append(get_neighbor_flat(v_flat[0], si, sj)) # v2
        v_cols.append(get_neighbor_flat(v_flat[1], si, sj)) # v1
        v_cols.append(get_neighbor_flat(v_flat[2], si, sj)) # v0

    # 4. Parâmetros
    n_points = nx * ny
    #dx_1col = np.full(n_points, 1/numet.dom.dx)
    #dy_1col = np.full(n_points, 1/numet.dom.dy)
    dx_col = np.full(n_points, numet.dom.dx)
    dy_col = np.full(n_points, numet.dom.dy)
    dt_col = np.full(n_points, numet.dom.dt)
    padding = np.zeros(n_points)
    # Junta tudo horizontalmente
    # Ordem: 20 colunas H + 6 colunas U + 6 colunas V + 1 padding + 3 constantes
    all_cols = h_cols + u_cols + v_cols + [padding, dx_col, dy_col, dt_col]
    input_vector = np.column_stack(all_cols)

    return input_vector

def build_input_vectorU(numet, state, variavel, mode, new_zeta):
    """
    state[0] = [h_t,   u_t  , v_t  ]
    state[1] = [h_t-1, u_t-1, v_t-1]
    state[2] = [h_t-2, u_t-2, v_t-2]
    """
    try:
        device = new_zeta.device
    except:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nx, ny = numet.sw.eta_b.shape
    topo_flat = numet.sw.eta_b.flatten() # (N,)\
    A, B = (1, 2) if variavel == 1 else (2, 1)
    # Organiza H, U, V e achata logo
    # h_flat[0] é o h2, h_flat[1] é o h1...
    if mode == 'forward':
        h_flat = [s[0].flatten() for s in state]
        u_flat = [s[A].flatten() for s in state]
        v_flat = [s[B].flatten() for s in state]
    else:
        h_flat = [state[i][0].flatten() for i in [2, 1, 0]]
        u_flat = [state[i][A].flatten() for i in [2, 1, 0]]
        v_flat = [state[i][B].flatten() for i in [2, 1, 0]]
        
    h_cols,nzeta_cols,u_cols,v_cols = [],[],[],[]
    if variavel == 1:
        # Construir H (32 colunas) Vizinhos: j+1, j-1, j, i+1, i-1, j+2, i+1ej+1, i-1ej+1
        offsets_h = [(0,1), (0,-1), (0,0), (1,0), (-1,0), (0,2), (1,1),(-1,1)]
        # Construir New Zeta (2 colunas) ij,j+1
        offsets_z = [(0,0), (0,1)]
        # Construir U (15 colunas) Vizinhos: (ij,j-1,j+1,i+1,i-1)  
        offsets_u = [(0,0), (0,-1),(0,1),(1,0),(-1,0)]
        # Construir V (12 colunas) Vizinhos: (ij,i-1,j+1,i-1ej+1)
        offsets_v = [(0,0), (-1,0),(0,1),(-1,1)]
    else:
        offsets_h = [(1,0), (-1,0), (0,0), (0,1), (0,-1), (2,0), (1,1),(1,-1)]
        # Construir New Zeta (2 colunas) ij,j+1
        offsets_z = [(0,0), (1,0)]
        # Construir V (15 colunas) Vizinhos: (ij,j-1,j+1,i+1,i-1)  
        offsets_u = [(0,0), (-1,0),(1,0),(0,1),(0,-1)]
        # Construir U (12 colunas) Vizinhos: (ij,j-1,i+1,i+1ej-1)
        offsets_v = [(0,0), (0,-1),(1,0),(1,-1)]
    
    # Função auxiliar para pegar vizinhos de forma flat
    def get_neighbor_flat(flat_array, shift_i, shift_j):
        if isinstance(flat_array, np.ndarray):
            arr = flat_array.reshape(nx, ny)
            shifted = np.roll(arr, shift=(-shift_i, -shift_j), axis=(0, 1))
            return shifted.flatten()
        
        arr = flat_array.view(nx, ny)
        shifted = torch.roll(arr, shifts=(-shift_i, -shift_j), dims=(0, 1))
        return shifted.flatten()

    # H #
    for si, sj in offsets_h:
        h_cols.append(get_neighbor_flat(h_flat[0], si, sj)) # h2
        h_cols.append(get_neighbor_flat(h_flat[1], si, sj)) # h1
        h_cols.append(get_neighbor_flat(h_flat[2], si, sj)) # h0
        h_cols.append(get_neighbor_flat(topo_flat, si, sj)) # topo

    # Zeta New #
    for si, sj in offsets_z:
        nzeta_cols.append(get_neighbor_flat(new_zeta, si, sj)) # zeta_new
    
    # U #
    for si, sj in offsets_u:
        u_cols.append(get_neighbor_flat(u_flat[0], si, sj)) # u2
        u_cols.append(get_neighbor_flat(u_flat[1], si, sj)) # u1
        u_cols.append(get_neighbor_flat(u_flat[2], si, sj)) # u0

    # V #
    for si, sj in offsets_v:
        v_cols.append(get_neighbor_flat(v_flat[0], si, sj)) # v2
        v_cols.append(get_neighbor_flat(v_flat[1], si, sj)) # v1
        v_cols.append(get_neighbor_flat(v_flat[2], si, sj)) # v0

    # Parâmetros
    n_points = nx * ny
    #dx_1col = np.full(n_points, 1/numet.dom.dx)
    #dy_1col = np.full(n_points, 1/numet.dom.dy)
    f, g = np.full(n_points, numet.sw.f), np.full(n_points, numet.sw.g)
    sinal = (-1)**(variavel-1)
    dx_col = np.full(n_points, numet.dom.dx*sinal)
    dy_col = np.full(n_points, numet.dom.dy)
    dt_col = np.full(n_points, numet.dom.dt)
    padding = np.ones(n_points)*sinal
    # Junta tudo horizontalmente
    # Ordem: 32 colunas H +  2 colunas new_Zeta + 15 colunas U + 12 colunas V + 1 padding + 3 constantes
    all_cols = h_cols + nzeta_cols + u_cols + v_cols + [padding, f, g, dx_col, dy_col, dt_col]
    all_cols_tensor = [
        c if isinstance(c, torch.Tensor) else torch.from_numpy(c).to(device)
        for c in all_cols]

    input_vector = torch.column_stack(all_cols_tensor)
    return input_vector
    
def pega_incremento2CROCO(numet,state,mode='forward'): #sis,past,paster,dx,dy,dt,MASCARA):
    #     Lazy Input for Encode-Decode     #
    # Grabs the tendency from the AB3 step #
    ###################################
    
    dx, dy = numet.dom.dx,numet.dom.dy
    dt=numet.dom.dt
    TOPO = numet.sw.eta_b
    beta = 0.281105
    c2 = -(1/2+2*beta) 
    if mode=='forward':
        c3 = beta
        c1 = (3/2+beta)
        times = range(3)
    else:
        c1 = beta
        c3 = (3/2+beta)
        times = range(2, -1, -1)
    
    VETOR_var = [[pres*c1 + past*c2 + paster*c3] for pres,past,paster in zip(state[0], state[1], state[2])]
    VETOR_var[0][0] = VETOR_var[0][0]+TOPO

    
    ########################################################## CALCULO DO INCREMENTO ######################################
    ### numet #
    htend = numet.tend(VETOR_var[1][0], VETOR_var[2][0], VETOR_var[0][0], AB3=True)   
    VETOR = [htend*dt]
    ##################
    
    ## EMPILHA E VETORIZA DOMINIO APARTIR DA MASCARA DE OCEANO #
    b_vector = [vector.reshape(1,-1) for vector in VETOR]
    b = np.array(b_vector).reshape(len(b_vector),-1)
    ############################################################
    #print('Zeta Input Vector Shape -5 (no dx,1/dx,dy,1/dy,dt) ',b.shape)
    ############################################################

    return b

def Train_LieAE(num_met, modelZ,modelU, argsZ,argsU,
                epochs=200, train_steps=[1,15,30],val_step=30,
                train_time=1000,validation_time=2000,Normalize=True,
                output_file=None,output_figure=None):

   # train_lenght is the number of epochs used in training #
   # train_steps are the number of integration steps before backpropagation #
   # train_time is the initial time of the simulation for training #
   # validation_time+train_time is the initial time of the simulation for validation #

  print('LieAE Training 2.0  ( with pinn )')
  n_backporpags = 1
  train_losses = np.empty((epochs), dtype=object) # tinha epochs,n_backporpags
  test_losses  = np.empty((epochs), dtype=object)
  mlv,mlv_train= np.inf,np.inf # initial value of loss, to track network improvement.
  Normalização = [[0,1],[0,1],[0,1]] #sem Normalização
  Normalização_out = Normalização #sem Normalização
  dt           = num_met.dom.dt
  NN = argsZ.NN
  eta_b = num_met.sw.eta_b # torch.from_numpy(num_met.sw.eta_b).to(cpu)
    
  criterion    = nn.MSELoss().to(device) # nn.HuberLoss().to(device) #
  criterion_L1 = nn.L1Loss().to(device)
    
  optimizer = torch.optim.Adam(list(modelZ.parameters()) +list(modelU.parameters()),# SEM U
                               lr=argsZ.lr, weight_decay=argsZ.wd)

  sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     factor=argsZ.lr_decay,     # shrink LR by learning_rate_chang
                                                     patience=4,     # no CP2 original estava 6...
                                                     min_lr=1e-7,        # Set a minimum learning rate
                                                     eps=1e-7           # Set a  epsilon for "significant" changes
                                                     )
      
  lamb,eta,nu,alpha = argsZ.lamb,argsZ.eta,argsZ.nu,1000

  # IC FOR TRAIN #
  u_train,v_train,h_train, u1_train,v1_train,h1_train, u2_train,v2_train,h2_train = Evolve_system(num_met,train_time,print=False)

  # IC FOR VALIDATION #
  u_val,  v_val, h_val,u1_val,v1_val,h1_val,u2_val,v2_val,h2_val = Evolve_system(num_met,validation_time,False,
                                                                                 u_train.copy(),v_train.copy(),h_train.copy(),
                                                                                 u1_train.copy(),v1_train.copy(),h1_train.copy(),
                                                                                 u2_train.copy(),v2_train.copy(),h2_train.copy(),False)
  
  ### MASK ###
  DOM_shape = h2_train.shape
  mask = Train_mask(DOM_shape,x0=0,x1=1,y0=1/8,y1=7/8,y00=3/8,y11=5/8)
  mask_cpu = mask.to('cpu')
  # SAVE TRAINING POINTS
  ny, nx = h2_train.shape[-2], h2_train.shape[-1]
  plot2D_Zetapanel(h2_train/np.std(h2_train),u2_train/np.std(u2_train),v2_train/np.std(v2_train),
                   mask_cpu.numpy(),mask_cpu.numpy(),mask_cpu.numpy(),
                   np.zeros((ny, nx)),np.zeros((ny, nx)),np.zeros((ny, nx)),
                   num_met.dom,file=output_figure[0],name='Training_Points')  

  # NORMALIZACAO E TRAINING TARGET #
  u_f, v_f, h_f  =  u_train.copy(), v_train.copy(), h_train.copy()
  u1_f,v1_f,h1_f = u1_train.copy(),v1_train.copy(),h1_train.copy()
  u2_f,v2_f,h2_f = u2_train.copy(),v2_train.copy(),h2_train.copy()
  # Target (Forward Encoder e Backward propagation)#
  print("Start Normalization ( with mask )")
  I,O,S = Vector_Creator(num_met,train_steps[-1], u_f,v_f,h_f, u1_f,v1_f,h1_f,
                         u2_f,v2_f,h2_f,mask=mask_cpu,final_state=True,AB3=True)

  Input_vectorH,Input_vectorU,Input_vectorV = I
  Output_vectorH,Output_vectorU,Output_vectorV = O
  h_fwdt,u_fwdt,v_fwdt, h1_fwdt,u1_fwdt,v1_fwdt, h2_fwdt,u2_fwdt,v2_fwdt = S  
    
    # SAVE THE TARGET OF TRAINING #
  plot2D_Zetapanel(h2_train,u2_train,v2_train,
                   h2_fwdt,u2_fwdt,v2_fwdt,
                   h2_train,u2_train,v2_train,
                   num_met.dom,file=output_figure[0],name='Target')

  if Normalize:
      ## NORMALIZATION FOR INPUT VECTOR #
      Input_vectorH = np.squeeze(Input_vectorH)
      Input_vectorU = np.squeeze(Input_vectorU)
      Input_vectorV = np.squeeze(Input_vectorV)
      ## Encoder Type A ##
      if argsZ.NN =='LieAE': # Precisa de Normalização para o encoder
          if modelZ.encoder[0].typeA: # input shape -1,7
              soma          = Input_vectorH[:, 0]+Input_vectorH[:, 1] # para a rede type A, 7.24
              Input_vectorH_lattent = np.concatenate([soma.reshape(-1,1), Input_vectorH[:,2:]],axis=1)
          ####################
              IHm,IHs = Input_vectorH_lattent.mean(axis=0), Input_vectorH_lattent.std(axis=0)
          else: #elif modelZ.encoder[0].typeC: # input shape -1,6
              IHm,IHs = Input_vectorH.mean(axis=0), Input_vectorH.std(axis=0)
      IUm,IUs = Input_vectorU.mean(axis=0), Input_vectorU.std(axis=0)
      IVm,IVs = Input_vectorV.mean(axis=0), Input_vectorV.std(axis=0)
      IMm = (IUm+IVm)/2
      IMs = (IUs+IVs)/2
      # print('Batchnorm needed.')
      Normalização = [np.stack([np.zeros(argsZ.Bdim), np.ones(argsZ.Bdim)]),
                      np.stack([np.zeros(argsU.Bdim), np.ones(argsU.Bdim)]),
                      np.stack([np.zeros(argsU.Bdim), np.ones(argsU.Bdim)])
                     ]
      with torch.no_grad():
        if argsZ.NN == 'LieAE': #and not modelZ.encoder[0].typeB: #not Beta Encoder.
            modelZ.encoder[0].bn0.running_mean = torch.tensor(IHm).to(device)
            modelZ.encoder[0].bn0.running_var = torch.tensor(IHs**2).to(device)
        if argsU.NN == 'LieAE': #and not modelU.encoder[0].typeB:
            modelU.encoder[0].bn0.running_mean = torch.tensor(IMm).to(device)
            modelU.encoder[0].bn0.running_var = torch.tensor(IMs**2).to(device)
       
       # Normaliza Target dos outputs #
      OHm,OHs = np.squeeze(Output_vectorH).mean(axis=0), np.squeeze(Output_vectorH).std(axis=0)
      OUm,OUs = np.squeeze(Output_vectorU).mean(axis=0), np.squeeze(Output_vectorU).std(axis=0)
      OVm,OVs = np.squeeze(Output_vectorV).mean(axis=0), np.squeeze(Output_vectorV).std(axis=0)
      Normalização_out = [np.stack([0,OHs]),
                          np.stack([0,OUs]),
                          np.stack([0,OVs])]
      if argsZ.NN == 'Beta': # BetaNetwork (diferente da rede NN com beta encoder).
          Normalização_out[0] = np.stack([np.zeros_like(OHs), np.ones_like(OHm)])
  
  #print('Input H', IHm,'\n',IHs,'\n',
  #      'Input U', IUm,'\n',IUs,'\n',
  #      'Input V', IVm,'\n',IVs,'\n',
  #      'Output H',OHm,'\n',OHs,'\n',
  #      'Output U',OUm,'\n',OUs,'\n',
  #      'Output V',OVm,'\n',OVs
  #     )

  argsZ.Normalização = Normalização[0]
  argsZ.Normalização_out = Normalização_out[0]
  argsU.Normalização = [Normalização[1],Normalização[2]]
  argsU.Normalização_out = [Normalização_out[1],Normalização_out[2]]

  Ordem = torch.tensor(np.mean(h2_f-u2_f)).to(cpu)
  Ordem = torch.floor(torch.log10(torch.clamp(Ordem,min=1e-10)))
  Ordem = 10.0 ** Ordem
  Ordem = Ordem.float().to(device)
  print('Ordem de diferença das variaveis Zeta e Momento:',Ordem.to('cpu').numpy())
  print('Start Training')
  ### TRAIN LOOP - USAR PROGRESSBAR ###
  epochs_counter = 0
  trainstep_counter = -1 
  total_steps = len(train_steps)
  for epoch in progressbar(range(epochs),'training '):
    
    steps = train_steps
  
    #######################
    ## Forward Time loop ##
    #######################
    #backpropag_counter = -1
      
    numero = 0
    for train_step in steps:# for k in range(n_backporpags):
        loss_fU      = torch.zeros(1, device=device)
        loss_idU     = torch.zeros(1, device=device)
        loss_bU      = torch.zeros(1, device=device)
        loss_fZ      = torch.zeros(1, device=device)
        loss_idZ     = torch.zeros(1, device=device)
        loss_bZ      = torch.zeros(1, device=device)
        loss_consist = torch.zeros(1, device=device)

        modelZ.train()
        modelU.train()
        
        # Fixed BatchNorm Layer #
        if argsZ.NN == 'LieAE':
            modelZ.apply(bns_eval)
            modelZ.encoder[0].apply(freeze_layers)
        #if modelU.encoder[0].typeA or modelU.encoder[0].typeC:
            # Encoders com batchnorm fixo no inicio #
        if argsU.NN == 'LieAE':
            modelU.apply(bns_eval)
        ## BatchNorm mean and var prints ##
        #for name, m in modelZ.named_modules():
        #    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        #        print(f"--- Camada: {name} ---")
        #        print(f"Média: {m.running_mean}")
        #        print(f"Variância: {m.running_var}")
        #for name, m in modelU.named_modules():
        #    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        #        print(f"--- Camada: {name} ---")
        #        print(f"Média: {m.running_mean}")
        #        print(f"Variância: {m.running_var}")
        
         #     Variational AE   #
        # for stochastic system # 
         #   time integration   #
        delta = .1
        
        # Forward
        # u2 is present, u1 present-1 and u0 present-2
        ################
        # FORWARD ===> #
        ################
        
        # SISTEMAS INICIAIS PARA FORWARD PROPAGATION # Variaveis do sistema em Tensor #
        stateZf,stateZaef = torch.from_numpy(h2_train.copy()).to(cpu),torch.from_numpy(h2_train.copy()).to(cpu)
        stateUf,stateUaef = torch.from_numpy(u2_train.copy()).to(cpu),torch.from_numpy(u2_train.copy()).to(cpu)
        stateVf,stateVaef = torch.from_numpy(v2_train.copy()).to(cpu),torch.from_numpy(v2_train.copy()).to(cpu)

        # SISTEMAS EVOLUIDOS (INICIAIS PARA BACKWARD PASS) # Variaveis do sistema em Tensor #
        stateZ,stateZae  = torch.from_numpy(h_fwdt.copy()).to(cpu),torch.from_numpy(h_fwdt.copy()).to(cpu)
        stateU,stateUae  = torch.from_numpy(u_fwdt.copy()).to(cpu),torch.from_numpy(u_fwdt.copy()).to(cpu)
        stateV,stateVae  = torch.from_numpy(v_fwdt.copy()).to(cpu),torch.from_numpy(v_fwdt.copy()).to(cpu)
        
        #######################
        ## Forward Time loop ##
        #######################
        
        #backpropag_counter = -1
        statef = [[h2_train.copy(),u2_train.copy(),v2_train.copy()],
                  [h1_train.copy(),u1_train.copy(),v1_train.copy()],
                  [ h_train.copy(), u_train.copy(), v_train.copy()]]

        # PARA LOSS DIRETAMENTE DO OUTPUT DA REDE       #
        Delta_AE = [torch.zeros((ny, nx), device=cpu), #
                   torch.zeros((ny, nx), device=cpu),  #
                   torch.zeros((ny, nx), device=cpu)]  #
        Delta_AEB =[torch.zeros((ny, nx), device=cpu), #
                   torch.zeros((ny, nx), device=cpu),  #
                   torch.zeros((ny, nx), device=cpu)]  #
        #################################################
        # Variaveis do sistema em Numpy #
        u_f, v_f, h_f  =  u_train.copy(), v_train.copy(), h_train.copy()
        u1_f,v1_f,h1_f = u1_train.copy(),v1_train.copy(),h1_train.copy()
        u2_f,v2_f,h2_f = u2_train.copy(),v2_train.copy(),h2_train.copy()
        
        for ii in range(train_step):
            #TARGET# (Forward propagation e Backward Encoder)
            old_uf,old_vf,old_hf = u2_f.copy(),v2_f.copy(),h2_f.copy()
            u2_f,v2_f,h2_f = AB3AM4(u_f,u1_f,u2_f,v_f,v1_f,v2_f,h_f,h1_f,h2_f,num_met) # ,printa=True
            u_f ,v_f ,h_f  = u1_f.copy(),v1_f.copy(),h1_f.copy()
            u1_f,v1_f,h1_f = old_uf,old_vf,old_hf
            
            # REDE #
            # FWD Zeta #
            input_tensor = Input_tensor(num_met,statef,0,'forward', Normalização[0]).to(device) # O INCREMENTO É CALCULADO COMO NO AB3AM4_fast
            ITS = input_tensor.shape
            input_train  = input_tensor.reshape((nx,ny,ITS[1],ITS[2],ITS[3]))[mask]
            input_border = input_tensor.reshape((nx,ny,ITS[1],ITS[2],ITS[3]))[~mask]
            f,_,_,Encoded,Forwarded,mean,log_var  = modelZ(input_train, n_variables = 1, mode='forward', delta=delta)
            with torch.no_grad(): # CALCULA BORDERS USANDO REDE #
                f0,_,_,_,_,_,_  = modelZ(input_border, n_variables = 1, mode='forward', delta=delta)            
            
            update        = torch.zeros((ny, nx), device=cpu)
            update[mask_cpu]  = torch.squeeze(f[0] ).to(cpu)
            update[~mask_cpu] = torch.squeeze(f0[0]).detach().to(cpu)

            # Normalize Output #
            update  = update   * Normalização_out[0][1] + Normalização_out[0][0] 

            # LOSS COMPUTATION TENSORS #
            stateZf = stateZf  + update #*dt                   # ACUMULA ERRO NO DOMINIO REAL
            
            # AE Zeta #
            if NN == 'LieAE':
                                         # input_train # vetor dinamico - o loss nao bate
                                         # ini_Ztrain
                #modelZ.train()
                _,_,af,_,_,_,_ = modelZ(input_train,   n_variables = 1, mode='encode',  delta=0) 
                with torch.no_grad():
                                              # input_border # vetor dinamico
                                              # ini_Zborder
                    #modelZ.eval()
                    _,_,af0,_,_,_,_ = modelZ(input_border,   n_variables = 1, mode='encode',  delta=0)
                updateae        = torch.zeros((ny, nx), device='cpu')
                updateae[mask_cpu]  = torch.squeeze(af[0] ).to(cpu)
                updateae[~mask_cpu] = torch.squeeze(af0[0]).detach().to(cpu)
                
                updateae = updateae * Normalização_out[0][1] + Normalização_out[0][0] 
                stateZaef = stateZaef  +  updateae #*dt # ACUMULA ERRO NO DOMINIO REAL
                Delta_AE[0] = Delta_AE[0] + updateae #*dt

            # FWD Momentum #
            Dnew   = stateZf.detach().cpu().numpy()+eta_b  # h2_f+eta_b # TENSOR esta na CPU
            Dstp   = statef[0][0]                  +eta_b    # h1_f+eta_b# TENSOR esta na cpu
            denominador   = [torch.from_numpy(num_met.bar_x(Dnew,"h")).to(cpu),
                             torch.from_numpy(num_met.bar_y(Dnew,"h")).to(cpu)]
            numerador     = [torch.from_numpy(num_met.bar_x(Dstp,"h")).to(cpu),
                             torch.from_numpy(num_met.bar_y(Dstp,"h")).to(cpu)]
            state  = [stateUf,stateVf]
            
            for var in [1,2]:
                idx = var-1  
                                                                                               #zeta_future=stateZf ou h2_f
                input_tensor = Input_tensor(num_met,statef,var,'forward', Normalização[var],zeta_future=stateZf).to(device) # 
                ITS = input_tensor.shape
                input_train  = input_tensor.reshape((nx,ny,ITS[1],ITS[2],ITS[3]))[mask]
                input_border = input_tensor.reshape((nx,ny,ITS[1],ITS[2],ITS[3]))[~mask]
                f,_,_,_,_,_,_  = modelU(input_train,   n_variables = 1, mode='forward', delta=delta)
                with torch.no_grad():
                    f0,_,_,_,_,_,_  = modelU(input_border,   n_variables = 1, mode='forward', delta=delta)
                    
                update        = torch.zeros((ny, nx), device=cpu)
                update[mask_cpu]  = torch.squeeze(f[0] ).to(cpu)
                update[~mask_cpu] = torch.squeeze(f0[0]).detach().to(cpu)                     # Borders usando a rede
                
                # Normalize Output #
                update     = update   * Normalização_out[var][1] + Normalização_out[var][0] 
                                    #*dt
                state[idx] = state[idx] * numerador[idx]/denominador[idx]                  # ACUMULA ERRO NO DOMINIO REAL
                updatef    = update/denominador[idx]                                       # ACUMULA ERRO NO DOMINIO REAL
                state[idx] = state[idx] + updatef                                          #*dt # ACUMULA ERRO NO DOMINIO REAL

                # AE Momentum #
                if NN == 'LieAE':
                    _,_,af,_,_,_,_ = modelU(input_train,   n_variables = 1, mode='encode',  delta=0)
                    with torch.no_grad():
                        _,_,af0,_,_,_,_ = modelU(input_border,   n_variables = 1, mode='encode',  delta=0)
                    updateae        = torch.zeros((ny, nx), device=cpu)
                    updateae[mask_cpu]  = torch.squeeze(af[0] ).to(cpu)
                    updateae[~mask_cpu] = torch.squeeze(af0[0]).detach().to(cpu)
                    #
                    updateae      = updateae * Normalização_out[var][1] + Normalização_out[var][0] 
                    updateaf    = updateae /denominador[idx]
                    Delta_AE[var] = Delta_AE[var]+ updateaf #*dt
                    
            stateUf,stateVf       = state
            
            statef[2] = statef[1]
            statef[1] = statef[0]
            statef[0] = [stateZf.detach().cpu().numpy(),
                         stateUf.detach().cpu().numpy(),
                         stateVf.detach().cpu().numpy()]
            ## LOSSES ##
            loss_fZ  =  loss_fZ + criterion(torch.from_numpy(h2_f)[mask_cpu].to(device),stateZf.to(device)[mask])
            loss_fU  =  loss_fU + criterion(torch.from_numpy(u2_f)[mask_cpu].to(device),stateUf.to(device)[mask])
            loss_fU  =  loss_fU + criterion(torch.from_numpy(v2_f)[mask_cpu].to(device),stateVf.to(device)[mask])
            numero+=1
            
            if NN == 'LieAE':
                loss_idZ = loss_idZ + criterion(Delta_AE[0].float().to(device)[mask] ,torch.zeros((ny, nx), device=device)[mask])
                loss_idU = loss_idU + criterion(Delta_AE[1].float().to(device)[mask] ,torch.zeros((ny, nx), device=device)[mask])
                loss_idU = loss_idU + criterion(Delta_AE[2].float().to(device)[mask] ,torch.zeros((ny, nx), device=device)[mask])

        ### SAVE INDIVIDUAL LOSSS PLOT FROM FORWARD PASS ##
        #plot2D_Zetapanel(True_inc[0].detach().numpy(),u2_f-u2_train,v2_f-v2_train,
        #                 Delta_F[0].detach().numpy(),stateUf.detach().numpy()-u2_train,stateVf.detach().numpy()-v2_train,
        #                 True_inc[0].detach().numpy()-Delta_F[0].detach().numpy(),
        #                 u2_f-stateUf.detach().numpy(),
        #                 v2_f-stateVf.detach().numpy(),
        #                 num_met.dom,file=output_figure[0],name='Losses_F')  
        #plot2D_Zetapanel(np.zeros((ny, nx)),np.zeros((ny, nx)),np.zeros((ny, nx)),
        #                 Delta_AE[0].detach().numpy(),Delta_AE[1].detach().numpy(),Delta_AE[2].detach().numpy(),
        #                 np.zeros((ny, nx))-Delta_AE[0].detach().numpy(),
        #                 np.zeros((ny, nx))-Delta_AE[1].detach().numpy(),
        #                 np.zeros((ny, nx))-Delta_AE[2].detach().numpy(),
        #                 num_met.dom,file=output_figure[0],name='Losses_AE')  
        ########################################################
        #################
        # <=== BACKWARD #
        #################
        if NN == 'LieAE':
            
            stateb = [[h2_fwdt.copy(),u2_fwdt.copy(),v2_fwdt.copy()],
                      [h1_fwdt.copy(),u1_fwdt.copy(),v1_fwdt.copy()],
                      [ h_fwdt.copy(), u_fwdt.copy(), v_fwdt.copy()]]
           
            # O inicio de todo backward é igual ao final do train_step[-1] #
            u_f, v_f, h_f  =  u_fwdt.copy(), v_fwdt.copy(), h_fwdt.copy()
            u1_f,v1_f,h1_f = u1_fwdt.copy(),v1_fwdt.copy(),h1_fwdt.copy()
            u2_f,v2_f,h2_f = u2_fwdt.copy(),v2_fwdt.copy(),h2_fwdt.copy()
                
            if True:
                for ii in range(train_step):
                    #TARGET# (Forward propagation e Backward Encoder)
                    old_uf,old_vf,old_hf = u_f.copy(),v_f.copy(),h_f.copy()
                    u_f,v_f,h_f = AB3AM4(u2_f,u1_f,u_f,v2_f,v1_f,v_f,h2_f,h1_f,h_f,num_met,mode='backward') #
                    u2_f ,v2_f ,h2_f  = u1_f.copy(),v1_f.copy(),h1_f.copy()
                    u1_f,v1_f,h1_f = old_uf,old_vf,old_hf
                    input_tensor   = Input_tensor(num_met,stateb,0,'backward', Normalização[0]).to(device)
                    ITS = input_tensor.shape
                    input_train  = input_tensor.reshape((nx,ny,ITS[1],ITS[2],ITS[3]))[mask]
                    input_border = input_tensor.reshape((nx,ny,ITS[1],ITS[2],ITS[3]))[~mask]
                    _,b,_,_,_,_,_  = modelZ(input_train,n_variables = 1, mode='backward',delta=delta)
                                            #ini_Ztrainb
                    _,_,ab,_,_,_,_ = modelZ(input_train,n_variables = 1, mode='encode',  delta=0)
                    with torch.no_grad():
                        _,b0,_,_,_,_,_  = modelZ(input_border,   n_variables = 1, mode='backward', delta=delta)
                                                #ini_Zborderb
                        _,_,ab0,_,_,_,_ = modelZ(input_border,   n_variables = 1, mode='encode'  , delta=0)
        
                    update        = torch.zeros((ny, nx), device=stateZ.device)
                    update[mask_cpu]  = torch.squeeze(b[0] ).to(cpu)
                    update[~mask_cpu] = torch.squeeze(b0[0]).detach().to(cpu)
                    updateae        = torch.zeros((ny, nx), device=stateZae.device)
                    updateae[mask_cpu]  = torch.squeeze(ab[0] ).to(cpu)
                    updateae[~mask_cpu] = torch.squeeze(ab0[0]).detach().to(cpu)
                    
                    # Normalize Output #
                    update   = update   * Normalização_out[0][1] + Normalização_out[0][0] 
                    updateae = updateae * Normalização_out[0][1] + Normalização_out[0][0] 
                    
                    stateZ   = stateZ   + update #*dt era -
                    stateZae = stateZae + updateae #*dt era -
                    Delta_AEB[0] = Delta_AEB[0] + updateae #*dt era +
        
                    # Momentum
                    Dnew   = stateZ.detach().cpu().numpy() +eta_b
                    Dstp   = stateb[2][0]                  +eta_b
                    denominador   = [torch.from_numpy(num_met.bar_x(Dnew,"h")).to(cpu),
                                     torch.from_numpy(num_met.bar_y(Dnew,"h")).to(cpu)]
                    numerador     = [torch.from_numpy(num_met.bar_x(Dstp,"h")).to(cpu),
                                     torch.from_numpy(num_met.bar_y(Dstp,"h")).to(cpu)]
            
                    state   = [stateU,stateV]
                    stateae = [stateUae,stateVae]
                    
                    for var in [1,2]:
                        idx = var-1
                        input_tensor   = Input_tensor(num_met,stateb,var,'backward', Normalização[var],zeta_future=stateZ).to(device) #.detach().cpu().numpy()
                        ITS = input_tensor.shape
                        input_train  = input_tensor.reshape((nx,ny,ITS[1],ITS[2],ITS[3]))[mask]
                        input_border = input_tensor.reshape((nx,ny,ITS[1],ITS[2],ITS[3]))[~mask]
                        _,b,_,_,_,_,_  = modelU(input_train,n_variables = 1, mode='backward',delta=delta)
                        _,_,ab,_,_,_,_ = modelU(input_train,n_variables = 1, mode='encode',  delta=0)
                        with torch.no_grad():
                            _,b0,_,_,_,_,_  = modelU(input_border,   n_variables = 1, mode='backward', delta=delta)
                            _,_,ab0,_,_,_,_ = modelU(input_border,   n_variables = 1, mode='encode'  , delta=0)
    
                        # Atualiza sistema com tensor de treino com gradientes. Topografia fixa
                        update        = torch.zeros((ny, nx), device=state[idx].device)
                        update[mask_cpu]  = torch.squeeze(b[0] ).to(cpu)
                        update[~mask_cpu] = torch.squeeze(b0[0]).detach().to(cpu)
                        updateae        = torch.zeros((ny, nx), device=stateae[idx].device)
                        updateae[mask_cpu]  = torch.squeeze(ab[0] ).to(cpu)
                        updateae[~mask_cpu] = torch.squeeze(ab0[0]).detach().to(cpu)
                        
                        state[idx]    = state[idx] * numerador[idx]/ denominador[idx]
                        
                        # Normalize Output #
                        update       = update   * Normalização_out[var][1] + Normalização_out[var][0] 
                        updateae     = updateae * Normalização_out[var][1] + Normalização_out[var][0] 
                        
                        updateb       = update /denominador[idx]
                        updateab      = updateae/denominador[idx]
                        
                        state[idx]    = state[idx]    + updateb  #*dt era -
                        stateae[idx]  = stateae[idx]  + updateab #*dt era -
                        Delta_AEB[var]= Delta_AEB[var]+ updateab #*dt era -
        
                    stateU,stateV = state
                    stateUae,stateVae = stateae
                    stateb[0] = stateb[1]
                    stateb[1] = stateb[2]
                    stateb[2] = [stateZ.detach().cpu().numpy(),
                                 stateU.detach().cpu().numpy(),
                                 stateV.detach().cpu().numpy()]
                    ## LOSSES ##
                    # AE #
                    loss_idZ = loss_idZ + criterion(Delta_AEB[0].float().to(device)[mask],torch.zeros((ny, nx), device=device)[mask])
                    loss_idU = loss_idU + criterion(Delta_AEB[1].float().to(device)[mask],torch.zeros((ny, nx), device=device)[mask])
                    loss_idU = loss_idU + criterion(Delta_AEB[2].float().to(device)[mask],torch.zeros((ny, nx), device=device)[mask])
                    # Loss Backward:
                    loss_bZ   = loss_bZ + criterion(   stateZ.to(device)[mask], torch.from_numpy(h_f)[mask_cpu].to(device))
                    loss_bU   = loss_bU + criterion(   stateU.to(device)[mask], torch.from_numpy(u_f)[mask_cpu].to(device))
                    loss_bU   = loss_bU + criterion(   stateV.to(device)[mask], torch.from_numpy(v_f)[mask_cpu].to(device))

                ###### # #######       
                # Consist LOSS # 
                ####### ########
                losses_c = []
                for model in [modelZ,modelU]: # SEM U
                    if model.multidynamics:
                        for n in range(n_variables): # numero de variaveis do sistema == numero de dinamicas
                            A = model.dynamics[n].dynamics.weight
                            B = model.backdynamics[n].dynamics.weight
                            K = A.shape[-1]
                            for k in range(1,K+1): # era range(1,K+1)
                                As1 = A[:,:k]
                                Bs1 = B[:k,:]
                                As2 = A[:k,:]
                                Bs2 = B[:,:k]
                                Ik = torch.eye(k).float().to(device)
                                L = (torch.sum((torch.mm(Bs1, As1) - Ik)**2) + \
                                     torch.sum((torch.mm(As2, Bs2) - Ik)**2) ) / (2.0*k)
                                losses_c.append(L)
                    else:
                        A = model.dynamics[0].dynamics.weight
                        B = model.backdynamics[0].dynamics.weight
                        K = A.shape[-1]
                        for k in range(1,K+1): # era range(1,K+1)
                            As1 = A[:,:k]
                            Bs1 = B[:k,:]
                            As2 = A[:k,:]
                            Bs2 = B[:,:k]
                            Ik = torch.eye(k).float().to(device)
                            L = (torch.sum((torch.mm(Bs1, As1) - Ik)**2) + \
                                 torch.sum((torch.mm(As2, Bs2) - Ik)**2) ) / (2.0*k)
                            losses_c.append(L)
                loss_consist = loss_consist + sum(losses_c)
                
        loss_Z =  loss_fZ + lamb/2 * loss_idZ + nu * loss_bZ
        loss_U = (loss_fU + lamb/2 * loss_idU + nu * loss_bU) #*Ordem
        loss =  loss_U + loss_Z + eta * loss_consist  # loss_Z + eta * loss_consist SEM U
        
        # < # < # < # < # < #
        # BACK PROPAGATION  #
        # > # > # > # > # > #
        
        optimizer.zero_grad(set_to_none=True)
        
        modelZ.train()
        modelU.train() # SEM U
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_( list(modelZ.parameters())+list(modelU.parameters()),1.0) # + list(modelU.parameters()) SEM U
        optimizer.step()
        
        ## SAVE Losses ###
        loss_id = lamb * (loss_idZ+loss_idU)/2# SEM U
        loss_f  =         loss_fZ +loss_fU     # SEM U
        loss_b  = nu *   (loss_bZ +loss_bU)    # SEM U
        
        loss          = loss.cpu().detach().numpy()
        loss_identity = loss_id.cpu().detach().numpy()
        loss_forward  = loss_f.cpu().detach().numpy()
        loss_backward = loss_b.cpu().detach().numpy()
        loss_consist  = loss_consist.item()
        # training lists
        train_losses[epoch]=[loss[0],loss_identity[0],
                             loss_forward[0],loss_backward[0],
                             loss_consist,None] # loss_KL.cpu().detach().numpy()
        
        gc.collect()
        torch.cuda.empty_cache()
    
    # SAVE TRAINING LOSS IN FILE ####
    if output_file is None:
        output_file = sys.__stdout__
    if output_figure is None:
        output_figure = sys.__stdout__
    ############################# ####
    with open(output_file, 'a', encoding='utf-8') as file:  
        Idloss = str( train_losses[epoch][1] ) #str( np.nansum(train_losses[epoch, :], axis=(0))[1] )
        Floss  = str( train_losses[epoch][2] ) #str( np.nansum(train_losses[epoch, :], axis=(0))[2] )
        Bloss  = str( train_losses[epoch][3] ) #str( np.nansum(train_losses[epoch, :], axis=(0))[3] )
        Closs  = str( train_losses[epoch][4] ) #str( np.nansum(train_losses[epoch, :], axis=(0))[4] )
        lossT  =      train_losses[epoch][0]        #np.nansum(train_losses[epoch, :], axis=(0))[0]
        print('<p style="color: black;">\n <strong>********** Epoche %s ********** </strong> </p>' %(epoch+1), file=file)
        print('<p style="color: blue;">loss identity: ' +Idloss + '</p>\n', file=file)
        print('<p style="color: blue;">loss forward:  ' +Floss + '</p>\n', file=file)
        print('<p style="color: blue;">loss backward: ' +Bloss + '</p>\n', file=file)
        print('<p style="color: blue;">loss cons:     ' +Closs + '</p>\n', file=file)
        print('<p style="color: blue;">loss sum: '      +str(lossT) + '</p>\n', file=file)
     
    if mlv_train > loss:
        ## PLOT SYSTEM ##
        if NN == 'LieAE':
            plot2D_Zetapanel(stateZaef.cpu().detach().cpu().numpy(),
                             stateUaef.cpu().detach().cpu().numpy(),
                             stateVaef.cpu().detach().cpu().numpy(),
                             statef[0][0],statef[0][1],statef[0][2],
                             stateb[2][0],stateb[2][1],stateb[2][2],
                             num_met.dom,file=output_figure[0],name='Prediction')
        
        ## PLOT SYSTEM ##
        else:
            plot2D_Zetapanel(h2_fwdt,u2_fwdt,v2_fwdt,
                             stateZf.detach().cpu().numpy(),
                             stateUf.detach().cpu().numpy(),
                             stateVf.detach().cpu().numpy(),
                             None,None,None,
                             num_met.dom,file=output_figure[0],name='FeedFwd')
        
        mlv_train = loss.copy()
        
    ##################  
    ### VALIDATION ###
    ##################
    modelZ.eval()
    modelU.eval()
    u_fv , v_fv, h_fv =  u_val.copy(), v_val.copy(), h_val.copy()
    u1_fv,v1_fv,h1_fv = u1_val.copy(),v1_val.copy(),h1_val.copy()
    u2_fv,v2_fv,h2_fv = u2_val.copy(),v2_val.copy(),h2_val.copy()
      
    stateZ,stateZaef = torch.from_numpy(h2_fv.copy()).to(device),torch.from_numpy(h2_fv.copy()).to(device)
    stateU,stateUaef = torch.from_numpy(u2_fv.copy()).to(device),torch.from_numpy(u2_fv.copy()).to(device)
    stateV,stateVaef = torch.from_numpy(v2_fv.copy()).to(device),torch.from_numpy(v2_fv.copy()).to(device)
      
    ################
    # FORWARD ===> #
    # Anda train_step passos no AB3AM4 e na rede:
    statef = [[h2_fv.copy(),u2_fv.copy(),v2_fv.copy()],
              [h1_fv.copy(),u1_fv.copy(),v1_fv.copy()],
              [ h_fv.copy(), u_fv.copy(), v_fv.copy()]]
      
    # AE Fixed Input Tensor for Validation #
    _,_,Zfwd = AB3AM4(u_fv,u1_fv,u2_fv,v_fv,v1_fv,v2_fv,h_fv,h1_fv,h2_fv,num_met)
    Zfwd = torch.from_numpy(Zfwd.copy()) #Zfwd.copy() #
    ini_Ztensor   =  Input_tensor(num_met,statef,0,'forward', Normalização[0])
    ini_Utensor   = [Input_tensor(num_met,statef,1,'forward', Normalização[1],zeta_future=Zfwd),
                     Input_tensor(num_met,statef,2,'forward', Normalização[2],zeta_future=Zfwd)]
    Dstp      = statef[0][0]+eta_b
    Numerador = [torch.from_numpy(num_met.bar_x(Dstp ,"h")).to(device),
                 torch.from_numpy(num_met.bar_y(Dstp ,"h")).to(device)]
      
    with torch.no_grad():  
     for ii in range(val_step):

        #TARGET#
        if ii == val_step-1: # Zbwd for fixed AE computation.
            Zbwd = torch.from_numpy(h_fv.copy()) #  h_fv.copy() #
      
        old_uf,old_vf,old_hf = u2_fv.copy(),v2_fv.copy(),h2_fv.copy()
        u2_fv,v2_fv,h2_fv = AB3AM4(u_fv,u1_fv,u2_fv,v_fv,v1_fv,v2_fv,h_fv,h1_fv,h2_fv,num_met)
        u_fv ,v_fv ,h_fv  = u1_fv.copy(),v1_fv.copy(),h1_fv.copy()
        u1_fv,v1_fv,h1_fv = old_uf,old_vf,old_hf

        # REDE #
        #Zeta
        input_tensor   = Input_tensor(num_met,statef,0,'forward', Normalização[0])
        f,_,_,_,_,_,_  = modelZ(input_tensor.to(device),   n_variables = 1, mode='forward', delta=delta)
        update  = f[0].reshape(DOM_shape)

         # Normalize Output #
        update  = update * Normalização_out[0][1] + Normalização_out[0][0] 
        stateZ  = stateZ + update #*dt

        if NN =='LieAE':
            _,_,af,_,_,_,_ = modelZ(ini_Ztensor.to(device),   n_variables = 1, mode='encode',  delta=0)
            updateae = af[0].reshape(DOM_shape) * Normalização_out[0][1] + Normalização_out[0][0] 
            stateZaef = stateZaef + updateae #*dt

        # Momentum
        Dnew   = stateZ.detach().cpu().numpy()+eta_b # h2_f+eta_b # # TENSOR esta na CPU
        Dstp   = statef[0][0]+ eta_b #h1_f+eta_b #torch.from_numpy(statef[0][0])+eta_b  # TENSOR esta na cpu
        denominador   = [torch.from_numpy(num_met.bar_x(Dnew,"h")).to(device),
                         torch.from_numpy(num_met.bar_y(Dnew,"h")).to(device)]
        numerador     = [torch.from_numpy(num_met.bar_x(Dstp,"h")).to(device),
                         torch.from_numpy(num_met.bar_y(Dstp,"h")).to(device)]
        state = [stateU,stateV]
        stateae = [stateUaef,stateVaef]

        for var in [1,2]:
            idx = var-1                                                                                  #h2_fv
            input_tensor   = Input_tensor(num_met,statef,var,'forward', Normalização[var],zeta_future=stateZ) #.detach().cpu().numpy()
            f,_,_,_,_,_,_  = modelU(input_tensor.to(device),   n_variables = 1, mode='forward', delta=delta)
            # Atualiza sistema com tensor de treino com gradientes. Topografia fixa
            state[idx]   = state[idx] * numerador[idx]/ denominador[idx]
            update       = torch.squeeze(f[0]).reshape(DOM_shape)
            
            # Normalize Output #
            update       = update   * Normalização_out[var][1] + Normalização_out[var][0] 
            updatef      = update /denominador[idx]
            state[idx]   = state[idx]  +updatef #*dt
            
            if NN =='LieAE':
                _,_,af,_,_,_,_ = modelU(ini_Utensor[idx].to(device),   n_variables = 1, mode='encode',  delta=0)
                updatea        = torch.squeeze(af[0]).reshape(DOM_shape) * Normalização_out[var][1] + Normalização_out[var][0]
                #else:
                #    updatea        = torch.squeeze(af[0]).reshape(DOM_shape)
                updateaf      = updatea/Numerador[idx]
                stateae[idx]  = stateae[idx]+updateaf #*dt
            
        stateU,stateV       = state 
        stateUaef,stateVaef = stateae
        statef[2] = statef[1]
        statef[1] = statef[0]
        statef[0] = [stateZ.detach().cpu().numpy(),
        #             u2_fv,v2_fv]
                     stateU.detach().cpu().numpy(),
                     stateV.detach().cpu().numpy()]
     # Loss Forward:
     loss_f       =  criterion(stateZ,torch.from_numpy(h2_fv).to(device))
     loss_f       += criterion(stateU,torch.from_numpy(u2_fv).to(device))
     loss_f       += criterion(stateV,torch.from_numpy(v2_fv).to(device))
     # Loss AE #
     if NN == 'LieAE':
        loss_id      =  criterion(stateZaef,torch.from_numpy(h2_val).to(device))
        loss_id      += criterion(stateUaef,torch.from_numpy(u2_val).to(device))
        loss_id      += criterion(stateVaef,torch.from_numpy(v2_val).to(device))
        
     # <=== BACKWARD #
     #################
        stateZ,stateZae  = torch.from_numpy(h_fv.copy()).to(device),torch.from_numpy(h_fv.copy()).to(device)
        stateU,stateUae  = torch.from_numpy(u_fv.copy()).to(device),torch.from_numpy(u_fv.copy()).to(device)
        stateV,stateVae  = torch.from_numpy(v_fv.copy()).to(device),torch.from_numpy(v_fv.copy()).to(device)
    
        stateb = [[h2_fv.copy(),u2_fv.copy(),v2_fv.copy()],
                  [h1_fv.copy(),u1_fv.copy(),v1_fv.copy()],
                  [ h_fv.copy(), u_fv.copy(), v_fv.copy()]]
         
        # AE Fixed Input Tensor #
        ini_Ztensor =  Input_tensor(num_met,stateb,0,'backward', Normalização[0])
        ini_Utensor = [Input_tensor(num_met,stateb,1,'backward', Normalização[1],zeta_future=Zbwd),
                       Input_tensor(num_met,stateb,2,'backward', Normalização[2],zeta_future=Zbwd)]
        Dstp      = stateb[2][0]+eta_b
        Numerador = [torch.from_numpy(num_met.bar_x(Dstp ,"h")).to(device),
                     torch.from_numpy(num_met.bar_y(Dstp ,"h")).to(device)]
      
        for _ in range(val_step):
             # REDE #
             #Zeta
             input_tensor   = Input_tensor(num_met,stateb,0,'backward', Normalização[0])
             _,b,_,_,_,_,_  = modelZ(input_tensor.to(device),n_variables = 1, mode='backward',delta=delta)
             _,_,ab,_,_,_,_ = modelZ(ini_Ztensor.to(device),n_variables = 1, mode='encode',  delta=0)
             updateb   = b[0].reshape( DOM_shape)
             updateaeb = ab[0].reshape(DOM_shape)
              # Normalize Output #
             updateb   = updateb   * Normalização_out[0][1] + Normalização_out[0][0] 
             updateaeb = updateaeb * Normalização_out[0][1] + Normalização_out[0][0] 
    
             stateZ   = stateZ   + updateb   #*dt era -
             stateZae = stateZae + updateaeb #*dt era -
             # Momentum
             Dnew   = stateZ.detach().cpu().numpy()+eta_b  #
             Dstp   = stateb[2][0]+eta_b #
             denominador   = [torch.from_numpy(num_met.bar_x(Dnew,"h")).to(device),
                              torch.from_numpy(num_met.bar_y(Dnew,"h")).to(device)]
             numerador     = [torch.from_numpy(num_met.bar_x(Dstp,"h")).to(device),
                              torch.from_numpy(num_met.bar_y(Dstp,"h")).to(device)]
             state   = [stateU,stateV]
             stateae = [stateUae,stateVae]
             
             for var in [1,2]:
                 idx = var-1
                 input_tensor   = Input_tensor(num_met,stateb,var,'backward', Normalização[var],zeta_future=stateZ) #
                 _,b,_,_,_,_,_  = modelU(    input_tensor.to(device),n_variables = 1, mode='backward',delta=delta)
                 _,_,ab,_,_,_,_ = modelU(ini_Utensor[idx].to(device),n_variables = 1, mode='encode',  delta=0)
                
                 state[idx]   = state[idx] * numerador[idx]/denominador[idx]
                 update       = torch.squeeze(b[0]).reshape(DOM_shape)  
                 updatea      = torch.squeeze(ab[0]).reshape(DOM_shape) 
                 # Normalize Output #
                 update       = update  * Normalização_out[var][1] + Normalização_out[var][0] 
                 updatea      = updatea * Normalização_out[var][1] + Normalização_out[var][0] 
             
                 updateb  = update /denominador[idx]
                 updateab = updatea/Numerador[idx]   
                 state[idx]    = state[idx]   + updateb  #*dt era -
                 stateae[idx]  = stateae[idx] + updateab #*dt era -
    
             stateU,stateV     = state 
             stateUae,stateVae = stateae
             
             stateb[0] = stateb[1]
             stateb[1] = stateb[2]
             stateb[2] = [stateZ.detach().cpu().numpy(),
                          stateU.detach().cpu().numpy(),
                          stateV.detach().cpu().numpy()]
             
         # Loss Backward:
        loss_id      += criterion(stateZae,torch.from_numpy(h_fv).to(device))
        loss_id      += criterion(stateUae,torch.from_numpy(u_fv).to(device))
        loss_id      += criterion(stateVae,torch.from_numpy(v_fv).to(device))
        loss_id      = loss_id/2
        loss_b       =  criterion(stateZ,  torch.from_numpy(h_val).to(device))
        loss_b       += criterion(stateU,  torch.from_numpy(u_val).to(device))
        loss_b       += criterion(stateV,  torch.from_numpy(v_val).to(device))

    ## LOSS FINAL ##
    loss = loss_f + lamb * loss_id + nu * loss_b + eta * loss_consist
        
    ## SALVA Losses ###
    loss          =    loss.cpu().detach().numpy()
    loss_identity = loss_id.cpu().detach().numpy()
    loss_forward  =  loss_f.cpu().detach().numpy()
    loss_backward =  loss_b.cpu().detach().numpy()
      
    # training lists
    test_losses[epoch]=[loss,loss_identity,loss_forward,loss_backward,loss_consist,None] # loss_KL.cpu().detach().numpy()

    ## END EPOCH ##
    ## print/save Loss_backward and update lr ##
    with open(output_file, 'a', encoding='utf-8') as file:
        print('** Validation **\n', file=file)
        Idloss = str( test_losses[epoch][1] )
        Floss  = str( test_losses[epoch][2] )
        Bloss  = str( test_losses[epoch][3] )
        Loss   = str( test_losses[epoch][0] )
        if argsZ.NN == 'Beta':
            print('<p style="color: red;">Beta Value: '+ str(modelZ.beta.item())+ '</p>\n', file=file)
        print('<p style="color: red;">loss identity: '+Idloss + '</p>\n', file=file)
        print('<p style="color: red;">loss forward:  '+Floss  + '</p>\n', file=file)
        print('<p style="color: red;">loss backward: '+Bloss  + '</p>\n', file=file)
        print('<p style="color: red;">loss sum: '     +Loss   + '</p>\n', file=file)

    ###############################
    # SALVA MELHOR REDE ATÉ AGORA #
    # # # # # # # # # # # # # # # #
    ###### after each epoch, UPDATE Learning Rate #####
    sched.step(loss)
      
    NameZ = 'LieAE_Z_best.pkl'
    NameU = 'LieAE_U_best.pkl'
    modelZ_path = argsZ.folder + NameZ
    modelU_path = argsZ.folder + NameU
      
    if loss < mlv: # salva rede baseado no lossforward
        mlv = copy.deepcopy(loss) # era newloss
        torch.save([argsZ, modelZ, train_losses, test_losses], modelZ_path) #
        torch.save([argsU, modelU, train_losses, test_losses], modelU_path) #
        ## PLOT SYSTEM ## 
        plot2D_Zetapanel(h2_val,u2_val,v2_val,
                         h2_fv ,u2_fv ,v2_fv,
                         h2_val,u2_val,v2_val,
                         num_met.dom,file=output_figure[1],name='Target')
        if NN == 'LieAE':
            plot2D_Zetapanel(stateZaef.detach().cpu().numpy(),stateUaef.detach().cpu().numpy(),stateVaef.detach().cpu().numpy(),
                             statef[0][0],statef[0][1],statef[0][2],
                             stateb[2][0],stateb[2][1],stateb[2][2],
                             num_met.dom,file=output_figure[1],name='Prediction')
        else:
            plot2D_Zetapanel(h2_fv ,u2_fv ,v2_fv,
                             statef[0][0],statef[0][1],statef[0][2],
                             None,None,None,
                             num_met.dom,file=output_figure[1],name='FeedFwd')
    else:
        try:
            # Load the existing file
            argsZ_saved, modelZ_saved, _, _ = torch.load(modelZ_path,weights_only=False)
            argsU_saved, modelU_saved, _, _ = torch.load(modelU_path,weights_only=False)
            # Save it back with updated train and test losess
            torch.save([argsZ_saved, modelZ_saved, train_losses, test_losses], modelZ_path)
            torch.save([argsU_saved, modelU_saved, train_losses, test_losses], modelU_path)
        except:
            # It is possible that the Validation domain (high turbulence) gives nan losses in the first epoch...
            # Therefore, we save the training losses and have no model to load from...
            torch.save([argsZ, modelZ, train_losses, test_losses], modelZ_path)
            torch.save([argsU, modelU, train_losses, test_losses], modelU_path)

    with open(output_file, 'a', encoding='utf-8') as file:
        print('Learning rate: '+str(optimizer.param_groups[0]['lr'])+'\n','MLV:', mlv,'\n MLV_train:',mlv_train ,file=file)

  return modelZ, modelU, optimizer, train_losses, test_losses

def bns_eval(m):
    # Freezes BatchNorm layer.
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.eval()

def freeze_layers(model_part):
    # Nomes exatos que você mencionou
    target_layers = ['linear', 'bilinear', 'beta', 'bilinear2']
    
    for name, module in model_part.named_modules():
        if any(target in name for target in target_layers):
            for param in module.parameters():
                param.requires_grad = False

def Vector_Creator(num_met,time, u_f,v_f,h_f,  u1_f,v1_f,h1_f,
                   u2_f,v2_f,h2_f,mask=None,final_state=True,AB3=True):

  ny, nx = h2_f.shape[-2], h2_f.shape[-1]
  if mask == None:
      mask = torch.zeros((ny, nx), dtype=torch.bool, device='cpu')
      mask[:,:]=True
    
  eta_b = num_met.sw.eta_b
  statef = [[h2_f.copy(),u2_f.copy(),v2_f.copy()],
            [h1_f.copy(),u1_f.copy(),v1_f.copy()],
            [ h_f.copy(), u_f.copy(), v_f.copy()]]
    
  Primeira = True
  for ii in range(time):
      
      #TARGET# (Forward propagation e Backward Encoder)
      old_uf,old_vf,old_hf = u2_f.copy(),v2_f.copy(),h2_f.copy()
      u2_f,v2_f,h2_f = AB3AM4(u_f,u1_f,u2_f,v_f,v1_f,v2_f,h_f,h1_f,h2_f,num_met)  # AB3AM4_fast nao ta bom
      u_f ,v_f ,h_f  = u1_f.copy(),v1_f.copy(),h1_f.copy()
      u1_f,v1_f,h1_f = old_uf,old_vf,old_hf
      
      Input = [Input_tensor(num_met,statef,0,'forward', None, AB3=AB3 ).to(cpu),
               Input_tensor(num_met,statef,1,'forward', None,torch.from_numpy(h2_f).to(cpu)), #torch.from_numpy(h2_f).to(cpu) #h2_f
               Input_tensor(num_met,statef,2,'forward', None,torch.from_numpy(h2_f).to(cpu))] #torch.from_numpy(h2_f).to(cpu) #h2_f

      Output = [torch.from_numpy((h2_f-h1_f)).to(cpu)] #/dt # preve massa
      Dnew = torch.from_numpy(h2_f.copy()).to(cpu) +eta_b
      Dstp = torch.from_numpy(h1_f.copy()).to(cpu) +eta_b
      for pres,past,var in zip([u2_f,v2_f],[u1_f,v1_f],[1,2]):
          denominador   = (Dnew  + torch.roll(Dnew,-1,var%2)) / 2
          numerador     = (Dstp  + torch.roll(Dstp,-1,var%2)) / 2
          Output.append((torch.from_numpy(pres).to(cpu)*denominador-
                         torch.from_numpy(past).to(cpu)*numerador   )) #/dt # preve momento

      if Primeira:
          ITS  = Input[0].shape  
          ITSU = Input[1].shape
          
      Input[0],Output[0] = Input[0].reshape((nx,ny,ITS[1],ITS[2],ITS[3])   )[mask],Output[0][mask]
      Input[1],Output[1] = Input[1].reshape((nx,ny,ITSU[1],ITSU[2],ITSU[3]))[mask],Output[1][mask]
      Input[2],Output[2] = Input[2].reshape((nx,ny,ITSU[1],ITSU[2],ITSU[3]))[mask],Output[2][mask]
      
      if Primeira:
          Input_vectorH  = np.array(Input[0], dtype=np.float32)
          Input_vectorU  = np.array(Input[1], dtype=np.float32)
          Input_vectorV  = np.array(Input[2], dtype=np.float32)
          Output_vectorH  = np.array(Output[0], dtype=np.float32)
          Output_vectorU  = np.array(Output[1], dtype=np.float32)
          Output_vectorV  = np.array(Output[2], dtype=np.float32)
          Primeira = False
      else:
          Input_vectorH  = np.concatenate((Input_vectorH,np.array( Input[0],  dtype=np.float32)),0)
          Input_vectorU  = np.concatenate((Input_vectorU,np.array( Input[1],  dtype=np.float32)),0)
          Input_vectorV  = np.concatenate((Input_vectorV,np.array( Input[2],  dtype=np.float32)),0)
          Output_vectorH = np.concatenate((Output_vectorH,np.array(Output[0], dtype=np.float32)),0)
          Output_vectorU = np.concatenate((Output_vectorU,np.array(Output[1], dtype=np.float32)),0)
          Output_vectorV = np.concatenate((Output_vectorV,np.array(Output[2], dtype=np.float32)),0)
      # Atualiza State #    
      statef = [[h2_f.copy(),u2_f.copy(),v2_f.copy()],
                [h1_f.copy(),u1_f.copy(),v1_f.copy()],
                [ h_f.copy(), u_f.copy(), v_f.copy()]]

  if final_state:
    return [Input_vectorH,Input_vectorU,Input_vectorV], [Output_vectorH,Output_vectorU,Output_vectorV], [h_f.copy(),u_f.copy(),v_f.copy(), h1_f.copy(),u1_f.copy(),v1_f.copy(), h2_f.copy(),u2_f.copy(),v2_f.copy()]
  else:
    return [Input_vectorH,Input_vectorU,Input_vectorV], [Output_vectorH,Output_vectorU,Output_vectorV]

def Evolve_system(num_met,time,start=True, u=None, v=None, h=None,
                  u1=None,v1=None,h1=None,u2=None,v2=None,h2=None,print=True):
  if start:
      u2 = num_met.sw.u0() # shape lat x lon
      v2 = num_met.sw.v0() # shape lat x lon
      h2 = num_met.sw.h0() # shape lat x lon
    
      u,u1 = np.zeros_like(u2),np.zeros_like(u2)
      v,v1 = np.zeros_like(v2),np.zeros_like(v2)
      h,h1 = np.zeros_like(h2),np.zeros_like(h2)
    
      u1,v1,h1 = AB3AM4(u,u,u2,v,v,v2,h,h,h2,num_met,step='ini') # AB3AM4_fast nao ta bom
    
      old_u2,old_v2,old_h2 = u2.copy(),v2.copy(),h2.copy()
      u2, v2, h2 = AB3AM4(u,u2,u1,v,v2,v1,h,h2,h1,num_met,step='second') # AB3AM4_fast nao ta bom
      u,v,h = old_u2,old_v2,old_h2

  ## AVANÇA SISTEMA ATÉ TIME_STEP DO TREINO ##
  for k in range(time):
    old_u2,old_v2,old_h2 = u2.copy(),v2.copy(),h2.copy()
    u2,v2,h2 = AB3AM4(u,u1,u2,v,v1,v2,h,h1,h2,num_met) # AB3AM4_fast nao ta bom
    u,v,h = u1.copy(),v1.copy(),h1.copy()
    u1,v1,h1 = old_u2,old_v2,old_h2
    
  ## PLOT DO SISTEMA INICIAL PARA A VALIDACAO  
  if print:
      plot2D_panel( u2, v2, h2, num_met.rel_vort(u2,v2), num_met.dom)
  
  return  u.copy(),v.copy(),h.copy(), u1.copy(),v1.copy(),h1.copy(), u2.copy(),v2.copy(),h2.copy()

def Train_mask(DOM_shape,x0,x1,y0,y1,y00=None,y11=None,device='cpu',print=False):
  ny, nx = DOM_shape[-2], DOM_shape[-1] 
  mask = torch.zeros((ny, nx), dtype=torch.bool, device=device)
  # MASKING SUBDOMAIN FOR FOCUSED LOSS FUNCTION #    
  x0 = int(nx * 0) # 2/6) # subdomain miolo
  x1 = int(nx * 1) # 4/6) # subdomain miolo
  y0 = int(ny * 1/8) # subdomain
  y1 = int(ny * 7/8) # subdomain

  # 100 RANDOM POINTS IN SUBDOMAN #
  # subdomain size
  sub_ny = y1 - y0
  sub_nx = x1 - x0
  N = sub_ny * sub_nx
  # sample 1000 unique linear indices
  idx = torch.randperm(N, device=device)[:1000]
  # map back to (y, x)
  ys = idx // sub_nx + y0
  xs = idx % sub_nx + x0
  # MASK UNIQUE POINTS
  mask[ys, xs] = True
  # REMOVES NULE CENTRAL DOMAIN FROM UNSTABLE JET #
  if y00 is not None:
      y00= int(ny * 3/8) # removes subdomain part
      y11= int(ny * 5/8) # removes subdomain part
      mask[y00:y11,x0:x1] = False

  if print:
      print('Number of pixel used in training:',mask.to('cpu').sum().numpy())
      
  return mask

def train_DMD(model, numet, variavel_teste=0, ini_train_step=3, train_step=1,
              ini_val_step=100, val_step=100, Normalize=True, mode = 'forward'):

    ## NOT TESTED ###
    # Calcula matriz linear do DMD. treina usando os n-1 primeiros dominios de Simulador_Variables apartir do sim_ini_step até train_step.
    # Testa simulando o operador com o dominio Simulador_variabes[-1] apartir do val_ini_step por val_steps passos no tempo.
    # Although AB3-AM4 is a filtered Forward Backward Scheme that is irreversible in time, I opted for contruscting a 'backward' mode
    # this way one can approximate a backward opperator, although a drift/bias should be expected if training using a simulation of +Δt intead of -Δt.

    eta_b = numet.sw.eta_b
    u,v,h, u1,v1,h1,u2,v2,h2 = Evolve_system(num_met,time=ini_train_step,start=True)
    u_fv,v_fv,h_fv, u1_fv,v1_fv,h1_fv, u2_fv,v2_fv,h2_fv = Evolve_system(num_met,time=ini_val_step,start=True)
    #################
    # TRAIN Vectors #
    #################
    I,O = Vector_Creator(num_met,train_step, u,v,h, u1,v1,h1, u2,v2,h2,
                         mask=None, final_state=False)
    
    Input_vectorH,Input_vectorU,Input_vectorV = I
    Output_vectorH,Output_vectorU,Output_vectorV = O

    # NORMALIZA INPUT E OUTPUT VECTORS:
    if Normalize:
      ## NORMALIZATION FOR INPUT VECTOR #
      IHm,IHs = np.squeeze(Input_vectorH).mean(axis=0), np.squeeze(Input_vectorH).std(axis=0)
      IMm = np.squeeze(np.concatenate((Input_vectorU,Input_vectorV))).mean(axis=0)
      IMs = np.squeeze(np.concatenate((Input_vectorU,Input_vectorV))).std(axis=0)
      Normalização = [np.stack([IHm,IHs]),
                      np.stack([IMm,IMs]),
                      np.stack([IMm,IMs])]
       
       # Normaliza Target dos outputs #
      OHm,OHs = np.squeeze(Output_vectorH).mean(axis=0), np.squeeze(Output_vectorH).std(axis=0)
      OMm = np.squeeze(np.concatenate((Output_vectorU,Output_vectorV))).mean(axis=0)
      OMs = np.squeeze(np.concatenate((Output_vectorU,Output_vectorV))).std(axis=0)
      Normalização_out = [np.stack([OHm,OHs]),
                          np.stack([OMm,OMs]),
                          np.stack([OMm,OMs])]

      InputH_norm = (Input_vectorH-Normalização[0][0])/Normalização[0][1]
      InputU_norm = (Input_vectorU-Normalização[1][0])/Normalização[1][1]
      InputV_norm = (Input_vectorV-Normalização[2][0])/Normalização[2][1]
      OutputH_norm = (Output_vectorH - Normalização_out[0][0] )/Normalização_out[0][1]
      OutputU_norm = (Output_vectorU - Normalização_out[1][0] )/Normalização_out[1][1]
      OutputV_norm = (Output_vectorV - Normalização_out[2][0] )/Normalização_out[2][1]
        
    else:
        Normalização = [None,None,None]
        Normalização_out = [None,None,None]
        InputH_norm, OutputH_norm = Input_vectorH,Output_vectorH
        InputU_norm, OutputU_norm = Input_vectorU,Output_vectorU
        InputV_norm, OutputV_norm = Input_vectorV,Output_vectorV

    ## COM NORMALIZAÇÃO ##
    #Input_norm = Normaliza_Vetor(Input_vector,Inpout_M_S)
    
    ### TREINA MODELO ####
    model.train(Input_norm, Output_norm)
    print('Done training')

     #############
    # VALIDATION #
    #############
    statef = [[h2_fv.copy(),u2_fv.copy(),v2_fv.copy()],
              [h1_fv.copy(),u1_fv.copy(),v1_fv.copy()],
              [ h_fv.copy(), u_fv.copy(), v_fv.copy()]]
    stateZ  = h2_fv.copy()
    stateU  = u2_fv.copy()
    stateV  = v2_fv.copy()
    
    for _ in range(val_step):
        old_uf,old_vf,old_hf = u2_fv.copy(),v2_fv.copy(),h2_fv.copy()
        u2_fv,v2_fv,h2_fv = AB3AM4(u_fv,u1_fv,u2_fv,v_fv,v1_fv,v2_fv,h_fv,h1_fv,h2_fv,num_met)
        u_fv ,v_fv ,h_fv  = u1_fv.copy(),v1_fv.copy(),h1_fv.copy()
        u1_fv,v1_fv,h1_fv = old_uf,old_vf,old_hf

        input_tensor   = Input_tensor(num_met,statef,0,'forward', Normalização[0])
        f,_,_,_,_,_,_  = modelU(input_tensor.to(device),   n_variables = 1, mode='forward', delta=1)
         # Normalize Output #
        update = torch.squeeze(f[0] ).to(cpu)
        update  = update * Normalização_out[0][1] + Normalização_out[0][0] 
        stateZ  = stateZ + update # *dt

        # Momentum
        Dnew   = stateZ.detach().cpu().numpy()+eta_b # h2_f+eta_b # # TENSOR esta na CPU
        Dstp   = statef[0][0]+ eta_b #h1_f+eta_b #torch.from_numpy(statef[0][0])+eta_b  # TENSOR esta na cpu
        denominador   = [torch.from_numpy(num_met.bar_x(Dnew,"h")).to(device),
                         torch.from_numpy(num_met.bar_y(Dnew,"h")).to(device)]
        numerador     = [torch.from_numpy(num_met.bar_x(Dstp,"h")).to(device),
                         torch.from_numpy(num_met.bar_y(Dstp,"h")).to(device)]
        
        state = [stateU,stateV]

        for var in [1,2]:
            idx = var-1                                                                                  #h2_fv
            input_tensor   = Input_tensor(num_met,statef,var,'forward', Normalização[var],zeta_future=stateZ) #.detach().cpu().numpy()
            f,_,_,_,_,_,_  = modelU(input_tensor.to(device),   n_variables = 1, mode='forward', delta=1)
            state[idx]   = state[idx] * numerador[idx]/ denominador[idx]
            update       = torch.squeeze(f[0]).reshape(DOM_shape)
            
            # Normalize Output #
            update   = update   * Normalização_out[var][1] + Normalização_out[var][0] 
            updatef      = update /denominador[idx]
            state[idx]   = state[idx] + updatef #*dt
            
        stateU,stateV       = state 
        
        statef[2] = statef[1]
        statef[1] = statef[0]
        statef[0] = [stateZ.detach().cpu().numpy(),
                     stateU.detach().cpu().numpy(),
                     stateV.detach().cpu().numpy()]

    # PLOT FINAL BIAS #
    plot2D_Zetapanel(h2_fv,u2_fv,v2_fv,
                     statef[0][0],
                     statef[0][1],
                     statef[0][2],
                     h2_fv-statef[0][0],
                     u2_fv-statef[0][1],
                     v2_fv-statef[0][2],
                     num_met.dom)
    return model
