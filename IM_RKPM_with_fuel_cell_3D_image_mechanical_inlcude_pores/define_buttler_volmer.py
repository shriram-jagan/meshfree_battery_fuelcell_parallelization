import time
start_time = time.time()
import numpy as np
from numpy import sign

import matplotlib.pyplot as plt

from tqdm import tqdm

from numba import jit

from scipy.sparse import csc_matrix, csr_matrix, bmat
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import eigs

from numpy.linalg import norm, eig

###################################################################
# define exchange current density, j0, which depends on x, 
# x = real concentration/maximum concentration
###################################################################
@jit
def i_0_complex(x):
    A0 = 0.303490440978371
    A1 = 1.271944700013477
    A2 = 4.420894220185683e+02
    A3 = -5.783762746199664e+03
    A4 = 3.822682327855755e+04
    A5 = -1.416477460103355e+05
    A6 = 3.113802647858406e+05
    A7 = -4.169011915077865e+05
    A8 = 3.347705415406199e+05
    A9 = -1.485221335897379e+05
    A10 = 2.803425068966447e+04
    P = (((((((((A10*x+A9)*x+A8)*x+A7)*x+A6)*x+A5)*x+A4)*x+A3)*x+A2)*x+A1)*x+A0
    dp_dx = 10.0*A10*x**9+9.0*A9*x**8+8.0*A8*x**7+7.0*A7*x**6+6.0*A6*x**5+5.0*A5*x**4+4.0*A4*x**3+3.0*A3*x**2+2.0*A2*x+A1
    return P, dp_dx


###################################################################
# define alpha lattice, which depends on x, 
#x = real concentration/maximum concentration
###################################################################
@jit
def alpha_lattice_complex(x):
    A0 = 2.81990134e-10
    A1 = 4.05602287e-12
    A2 = -1.37296095e-10
    A3 = 1.80800247e-9
    A4 = -1.19248433e-8
    A5 = 4.45750059e-8
    A6 = -9.98827607e-8
    A7 = 1.36152708e-7
    A8 = -1.1015245e-7
    A9 = 4.84951233e-8
    A10 = -8.93152235e-9
    a_lattice = (((((((((A10*x+A9)*x+A8)*x+A7)*x+A6)*x+A5)*x+A4)*x+A3)*x+A2)*x+A1)*x+A0
    dalattice_dx = 10.0*A10*x**9+9.0*A9*x**8+8.0*A8*x**7+7.0*A7*x**6+6.0*A6*x**5+5.0*A5*x**4+4.0*A4*x**3+3.0*A3*x**2+2.0*A2*x+A1
    return a_lattice, dalattice_dx

###################################################################
# define c lattice, which depends on x, 
# x = real concentration/maximum concentration
###################################################################
@jit
def c_lattice_complex(x):
    A0 = 1.39010402e-9
    A1 = 5.3010374e-11
    A2 = 1.64333764e-9
    A3 = -2.25843237e-8
    A4 = 1.55438386e-7
    A5 = -6.06667972e-7
    A6 = 1.42129465e-6
    A7 = -2.03022461e-6
    A8 = 1.72702935e-6
    A9 = -8.02967619e-7
    A10 = 1.57015662e-7
    c_lattice = (((((((((A10*x+A9)*x+A8)*x+A7)*x+A6)*x+A5)*x+A4)*x+A3)*x+A2)*x+A1)*x+A0
    dclattice_dx = 10.0*A10*x**9+9.0*A9*x**8+8.0*A8*x**7+7.0*A7*x**6+6.0*A6*x**5+5.0*A5*x**4+4.0*A4*x**3+3.0*A3*x**2+2.0*A2*x+A1
    return c_lattice, dclattice_dx

######################################################################################################################################
# define D, diffucivity which depends on x = real concentration/maximun concentration, 
# D is a n_g*(n_nodes*n_nodes)*(dimention*dimention) matrix
######################################################################################################################################
@jit
def Dn_complex(x, D_damage):
    for i in range(len(D_damage)):
        if D_damage[i] > 0.9:
            D_damage[i] = 0.9

    ## D_damage[D_damage>0.9] = 0.9
    macro_to_grain = 3.00
    D_x_thresholds = [-0.10000000000000000555,0.00000000000000000000,0.10000000000000000555,0.40000000000000002220,0.59999999999999997780,0.80000000000000004441]

    D_coefs = [0.00000000000000000000,0.00000000000000000000,0.00000000000000000000,0.00000000000001231320,\
               0.00000000000000000000,0.00000000000000000000,0.00000000000000000000,0.00000000000001231320,\
               -0.00000000000000006780,-0.00000000000000041949,0.00000000000000000000,0.00000000000001231320,\
               0.00000000000000880302,-0.00000000000001638013,-0.00000000000000027000,0.00000000000001227362,\
               0.00000000000125685839,-0.00000000000037054532,-0.00000000000000576569,0.00000000000001163483,\
               -0.00000000000000036183,0.00000000000000543084,-0.00000000000000316081,0.00000000000000571475]

    D_coefs = np.array(D_coefs).reshape(6,4)

    expon_D = np.array([3,2,1,0])[:, np.newaxis]

    D_dpolydx_coefs = (D_coefs.T*expon_D).T

    D = x*0
    dD_dx = x*0


    for ii in range(6):
        if ii == 0:
            logic_x = (x <= D_x_thresholds[ii+1])*1
        else:
            if ii == 5:
                logic_x = (x >D_x_thresholds[ii])*1
            else:
                logic_x_1 = (x >D_x_thresholds[ii]) 
                logic_x_2 = (x<=D_x_thresholds[ii+1])
                logic_x = logic_x_1*logic_x_2*1
        D = D+logic_x*(D_coefs[ii,0]*(x-D_x_thresholds[ii])**3+D_coefs[ii,1]*(x-D_x_thresholds[ii])**2+D_coefs[ii,2]*(x-D_x_thresholds[ii])**1+D_coefs[ii,3]*(x-D_x_thresholds[ii])**0) 
        dD_dx = dD_dx+logic_x*(D_dpolydx_coefs[ii,0]*(x-D_x_thresholds[ii])**2+D_dpolydx_coefs[ii,1]*(x-D_x_thresholds[ii])**1+D_dpolydx_coefs[ii,2]*(x-D_x_thresholds[ii])**0)

    return(D*macro_to_grain*(1-D_damage), dD_dx*macro_to_grain*(1-D_damage)) 


################################################################
# define the open circulate potential E_eq
################################################################
@jit
def ocp_complex(x):
    Eeq_x_thresholds = [-0.1000000000,0.0000000000,0.0250000000,0.1000000000,0.2000000000,0.3000000000,0.4000000000,\
                    0.5000000000,0.6000000000,0.7000000000,0.8000000000,0.9000000000,0.9500000000,0.9750000000, \
                        0.9900000000,0.9950000000,0.9990000000,1.0000000000]

    Eeq_coefs = [15.5276001364,-6.5923311959,-2.4960428818,5.6141783138,\
            895.3135766304,-33.6334506526,-3.3486811169,5.3141783138,\
                -81.9793520542,10.4802301423,-3.3516406933,5.2234296539,\
                43.1929468100,-5.9671112557,-3.1630077379,4.9964228573,\
                19.7924511525,0.2951859850,-3.0606415847,4.6636439177,\
                -14.4065560642,4.5837966472,-2.4078308531,4.3803240703,\
                2.0308297558,1.2433285041,-1.9232682056,4.1709723954,\
                23.2107688484,-0.9515644065,-1.6136776121,3.9931096896,\
                12.3318762496,0.9815583839,-1.1076674280,3.8454370532,\
                -21.4709897243,3.6635330598,-0.5413994637,3.7568177705,\
                -1.7180081860,-0.7023601791,-0.4528225435,3.7178421650,\
                -151.6070578413,5.2166938899,-0.6448348249,3.6638183006,\
                -234.9780625093,-49.0200087926,-1.2602183697,3.6256674119,\
                -4304.6501234801,-199.5626481081,-4.1518026765,3.5598529149,\
                -10492.9163916377,-1948.3980027643,-13.0443209531,3.4381460848,\
                154108.9641688380,-6491.3319338437,-33.3152697102,3.3229029154,\
                85857831.5552496761,-111367.9866187039,-77.8486949008,3.0956434993,\
                -301.1173472459,260.2234694492,-43.0111734725,2.9922846494]

    Eeq_coefs = np.array(Eeq_coefs).reshape(18,4)

    expon_Eeq = np.array([3,2,1,0])[:, np.newaxis]

    Eeq_dpolydx_coefs = (Eeq_coefs.T*expon_Eeq).T

    E_eq = x*0
    dEeq_dx = x*0


    for ii in range(18):
        if ii == 0:
            logic_x = (x <= Eeq_x_thresholds[ii+1])*1
        else:
            if ii == 17:
                logic_x = (x >Eeq_x_thresholds[ii])*1
            else:
                logic_x_1 = (x >Eeq_x_thresholds[ii]) 
                logic_x_2 = (x<=Eeq_x_thresholds[ii+1])
                logic_x = logic_x_1*logic_x_2*1
        E_eq = E_eq+logic_x*(Eeq_coefs[ii,0]*(x-Eeq_x_thresholds[ii])**3+Eeq_coefs[ii,1]*(x-Eeq_x_thresholds[ii])**2+Eeq_coefs[ii,2]*(x-Eeq_x_thresholds[ii])**1+Eeq_coefs[ii,3]*(x-Eeq_x_thresholds[ii])**0) 
        dEeq_dx = dEeq_dx+logic_x*(Eeq_dpolydx_coefs[ii,0]*(x-Eeq_x_thresholds[ii])**2+Eeq_dpolydx_coefs[ii,1]*(x-Eeq_x_thresholds[ii])**1+Eeq_dpolydx_coefs[ii,2]*(x-Eeq_x_thresholds[ii])**0)

    return(E_eq, dEeq_dx) 

################################################################
# define current density
################################################################
@jit
def i_se(p_s, j0, E_eq, Fday, R, Tk):
    eta_s = p_s - E_eq
    i_bv = 2*j0*np.sinh(Fday/(2*R*Tk)*eta_s)
    dibv_deta = 2*j0*np.cosh(Fday/(2*R*Tk)*eta_s)*Fday/(2*R*Tk)
    dibv_di0 = 2*np.sinh(Fday/(2*R*Tk)*eta_s)
    return dibv_deta, dibv_di0, i_bv
