# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 18:23:05 2023

@author: Hiroki Kozuki
"""
#%%
'''2.0 - Discretisation via Euler'''
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})

def KdeV_exact(alpha, x, t):
    '''
    Exact, single soliton solution of the KdeV equation. 
    x and t are rescaled into dimensionless units. 
    '''       
    return 12 * (alpha ** 2) * \
            (1 / np.cosh(alpha * (x - 4 * (alpha ** 2) * t))) ** 2

### Instability of Euler Method:

# h = 0.1
alpha = 1
step_Euler = 1

def Euler(u, h, dt):
    '''
    Explicit Euler method.
    Analytical solution is used to initialise the pulse. 
    '''
    return u + u_diff_t(u, h) * dt

def u_diff_t(u, h):
    '''
    Discretisation of the partial derivative of disturbance u 
    w.r.t time t. 
    '''
    return - advection(u,h) - dispersion(u,h)

def advection(u, h):
    '''
    Second term of the KdeV equation. This non-linear term gives a wave speed 
    that depends upon the amplitude of the disturbance u. 
    '''
    u_i_plus_1 = np.append(u[1:], u[:1])
    u_i_minus_1 = np.append(u[-1:], u[:-1])
    return (u_i_plus_1 ** 2 - u_i_minus_1 ** 2) / (4 * h)

def dispersion(u, h):
    '''
    Third term of the KdeV equation. This third order term introduces 
    dispersive broadening of the waveform.
    '''
    u_i_plus_1 = np.append(u[1:], u[:1])
    u_i_minus_1 = np.append(u[-1:], u[:-1])
    u_i_plus_2 = np.append(u[2:], u[:2])
    u_i_minus_2 = np.append(u[-2:], u[:-2])
    return (u_i_plus_2 - 2 * u_i_plus_1 + 2 * u_i_minus_1 - u_i_minus_2) \
            / (2 * h ** 3)


'''
Tried Euler method, got run-time warning and error for all conceivable values 
of h and dt as expected. 
'''
# h = 0.1
# x_min = -5
# x_max = 10
# x = np.linspace(int(x_min), int(x_max), int((x_max - x_min) / h))
# u = KdeV_exact(1, x, 0)
# plt.plot(x, u)

# dt = 0.001
# max = []
# time = []
# for i in range (1, 2001):
#     u = Euler(u, h, dt)
#     max.append(max(u))
#     time.append(i * dt)
#     if i % 600 == 0:
#         plt.plot(x, u, label = 't = {}'.format(i * dt))
# plt.legend()
# plt.xlabel('Displacement [x]') 
# plt.ylabel('Disturbance [u]')   
# plt.title('The Dynamics of a Single Soliton')     
# plt.show()


################################################################################
#%%
'''2.1 - Discretisation via Second and Fourth order Runge-Kutta'''
def RK_2(u, h, dt):
    '''
    Second-order Runge-Kutta method.
    '''
    alpha = 1 / 2
    fa = u_diff_t(u, h)
    fb = u_diff_t(u + alpha * dt * fa, h)
    return u + ((2 * alpha - 1) * fa + fb) * dt / (2 * alpha)

def RK_4(u, h, dt):
    '''
    Fourth-order Runge-Kutta method.
    '''
    fa = u_diff_t(u, h)
    fb = u_diff_t(u + fa * dt / 2, h)
    fc = u_diff_t(u + fb * dt / 2, h)
    fd = u_diff_t(u + fc * dt, h)
    return u + (fa + 2 * fb + 2 * fc + fd) * dt / 6

def Simpson(x, u):
    '''Simpson's Rule for numerical integration.'''
    
    n = len(x)
    if n % 2 == 0:
        raise ValueError("Number of datapoints must be odd.")
    
    h = (x[-1] - x[0]) / (n - 1)
    return h / 3 * (u[0] + 4 * np.sum(u[1:-1:2]) + 2 * np.sum(u[2:-1:2]) 
                    + u[-1])
    

    
################################################################################
#%%
'''3.0.0 - Dynamics with RK_2'''

### Propagating solitons with different soliton parameter a:

## RK_2:
# h = 0.1 and dt = 0.00005 works.
# h = 0.1 and dt = 0.0001 works. 
# when dt = 0.0001, stable for 0.095 < h < 0.2 rouhgly. 
    # Numerical diffusion for h > 0.2.
    # Instability for h < 0.095.  


x_min = -10
x_max = 10
alpha = 1 # Soliton Parameter.
h = 0.1
dt = 0.0001 # change

x = np.linspace(int(x_min), int(x_max), int((x_max - x_min) / h))
u = KdeV_exact(alpha, x, 0)
plt.plot(x, u, label = 't = 0') # Plot u at t = 0. 

u_list = [u]
time = [0]
for i in range (1, 10001): # 2001
    u = RK_2(u, h, dt)
    u_list.append(u)
    time.append(i * dt)
    if i % 1000 == 0:
        plt.plot(x, u, label = 't = {}'.format(i * dt))

plt.title('Propagation of a Single Soliton - RK_2') 
plt.xlabel('Displacement [x]') 
plt.ylabel('Disturbance [u]') 
plt.legend()      
plt.show()

# Space-time diagram
X, T = np.meshgrid(x, time)
plt.contourf(X, T, u_list, 20, cmap = 'rainbow')
plt.colorbar()
plt.show()

################################################################################
#%%
'''3.0.1 - Dynamics with RK_4'''

## RK_4:
# h = 0.1 and dt = 0.001 works well.

x_min = -10
x_max = 10.1 # 59.5 for area and chi-square, 10.1 for others. 
alpha = 1.0 # Soliton Parameter.
h = 0.1
dt = 0.001

plt.figure(figsize = (14, 6))
x = np.linspace(int(x_min), int(x_max), int((x_max - x_min) / h))
u = KdeV_exact(alpha, x, 0)
plt.plot(x, u, label = 't = 0', linewidth = 3) # Plot u at t = 0. 

time = [0]
u_list = [u]
max_list = [np.amax(u)]
chi_squared = [0]
sum_u_list = [np.sum(u)]
integral_list = [Simpson(x, u)]

for i in range (1, 12001): # 2001
    time.append(i * dt)
    u = RK_4(u, h, dt)
    u_list.append(u)
    max_list.append(np.amax(u))
    chi_squared.append(np.sum(((KdeV_exact(alpha, x, i * dt) - u) ** 2) 
                               / KdeV_exact(alpha, x, i * dt)))
    sum_u_list.append(np.sum(u))
    integral_list.append(Simpson(x, u))
    if i % 3000 == 0:
        plt.plot(x, u, linewidth = 3, label = 't = {}'.format(i * dt))
        
# plt.title('Propagation of a Single Soliton - RK_4')  
plt.xlabel('Displacement [x]') 
plt.ylabel('Disturbance [u]')
plt.legend(loc = 'upper right')
# plt.savefig('dynamics_propagation.png', bbox_inches = "tight", dpi = 1000)
plt.show()

### Amplitude of soliton remains constant across several transits.
plt.figure(figsize = (14, 6))
plt.plot(time, max_list, linewidth = 3)
# plt.title('Ampitude of a Soliton Over 10 Time Units')
plt.ylim([11, 13])
plt.xlabel('Time [t]')
plt.ylabel('Amplitude [$u_{max}$]')
# plt.savefig('dynamics_amplitude_conservation.png', bbox_inches = "tight", 
#             dpi = 1000)   
plt.show()

# Conservation of the sum of u(i,n) over all x(i) at each t(n): 
plt.figure(figsize = (14, 6))
plt.plot(time, sum_u_list, linewidth = 3)
# plt.title('Conservation of the sum of u(x(i),t(n)) over all x(i)')
# plt.xlim(0, 30)
# plt.ylim(245.660, 245.670)
plt.xlabel('Time [t]')
plt.ylabel('Sum of $u^{n}_{i}$ over all i')
# plt.savefig('dynamics_sum_conservation.png', bbox_inches = "tight", 
#             dpi = 1000) 
plt.show()

# Conservation of Area under the soliton curve (u(x,t) vs. x):
plt.figure(figsize = (14, 6))
plt.plot(time, integral_list, linewidth = 3)
plt.axhline(y = sum(integral_list) / len(integral_list), color = 'orange', 
            linestyle = '--', linewidth = 3, label = 'Mean area')
# plt.title('Conservation of Area under the soliton curve')
# plt.xlim(0, 30)
plt.ylim(23.99, 24.01)
plt.xlabel('Time [t]')
plt.ylabel('Area Under Soliton Curve')
plt.legend()
# plt.savefig('section_3_area_conservation.png', bbox_inches="tight", 
#             dpi = 1000) 
plt.show()

### Comparison against analytic solutions via the sum of differences squared.
plt.figure(figsize = (14, 6)) 
plt.plot(time, chi_squared, linewidth = 3)
# plt.title('Difference Squared Between Numerical and Analytic Solutions')
plt.xlabel('Time [t]')
plt.ylabel('Chi-squared')
# plt.savefig('dynamics_comparison.png', bbox_inches="tight", dpi = 1000)    
plt.show()

# Space-time diagram
plt.figure(figsize = (14, 6))
X, T = np.meshgrid(x, time)
plt.contourf(X, T, u_list, 20, cmap = 'rainbow')
plt.xlabel('Displacement [x]')
plt.ylabel('Time [t]')   
plt.colorbar()
# plt.savefig('dynamics_spacetime.png', bbox_inches = "tight", dpi = 1000) 
plt.show()

################################################################################
#%%
'''3.0.2 - Dependence of soliton speed on its height/amplitude'''

# Numerical results consistent with theory.

x_min = -10
x_max = 10
h = 0.1
dt = 0.001
x = np.linspace(int(x_min), int(x_max), int((x_max - x_min) / h))

plt.figure(figsize = (14, 6))
max_list = []
speed_list = []
alpha_list = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
color_list = ['purple', 'blue', 'cyan', 'green', 'orange', 'red', 'magenta']

for (alpha, color) in zip(alpha_list, color_list):
    u = KdeV_exact(alpha, x, 0)    
    plt.plot(x, u,  label = 'alpha = {}'.format(alpha), 
                    color = color, linestyle = 'dashed')
    
    for j in range (1, 1001): # propagate for 1 unit of time. 
        u = RK_4(u, h, dt)
    max_list.append(np.amax(u))
    speed = x[np.argmax(u)]
    speed_list.append(speed)
    plt.plot(x, u, color = color)
    
plt.xlabel('Displacement [x]')
plt.ylabel('Amplitude [$u_{max}$]')   
# plt.title('Speeds of Different Size Solitons - Numerical')     
plt.legend()

plt.show()

plt.figure(figsize = (14, 6))
max_list_analytic = []
speed_list_analytic = []
alpha_list_analytic = np.arange(0.1, 1.6, 0.1)
for alpha in alpha_list_analytic:
    u = KdeV_exact(alpha, x, 0)
    plt.plot(x, u, label = 'alpha = {}'.format(round(alpha, 1)), 
             linestyle = 'dashed')
    
    u = KdeV_exact(alpha, x, 1)
    plt.plot(x, u, label = 'alpha = {}'.format(round(alpha, 1)))
    plt.xlabel('Displacement [x]')
    plt.ylabel('Amplitude [$u_{max}$]')
    # plt.title('Speeds of Different Size Solitons - Analytical')
    plt.legend()
    
    max_list_analytic.append(np.amax(u))
    speed_analytic = x[np.argmax(u)]
    speed_list_analytic.append(speed_analytic)

plt.show()

plt.figure(figsize = (14, 6))
plt.plot(max_list_analytic, speed_list_analytic, linestyle = 'dashed', 
         linewidth = 3, label = 'Analytic relation')

# Speed = 4*a**2 and Amplitude = 12*a**2 so theoretical gradient = 1/3. 
linear_fit, linear_fit_cov = np.polyfit(max_list, speed_list, 1, cov = True)
analytic_speed_fit = np.poly1d(linear_fit)
plt.plot(max_list_analytic, analytic_speed_fit(max_list_analytic), 
         linewidth = 3, label = 'Linear fit on numerical values')
plt.scatter(max_list, speed_list, s = 200, label = 'Numerical values', 
            color = 'green')
# plt.title("Dependence of Soliton's Speed on Amplitude") 
plt.xlabel('Amplitude [A]')
plt.ylabel('Speed [$v_{x}$]')   
plt.legend()
# plt.savefig('height_speed.png', bbox_inches = "tight", dpi = 1000)
plt.show()

print('Gradient = {} +/- {} units'.format(linear_fit[0], 
                                          np.sqrt(linear_fit_cov[0,0])))
print('Y-int = {} +/- {} units'.format(linear_fit[1], 
                                       np.sqrt(linear_fit_cov[1,1])))

################################################################################
#%%
'''3.0.3 - Dependence of stability (threshold h and dt) on soliton parameter 
            alpha'''
'''
- Try alpha = 0.2, 0.5, 1.0, 1.5, 2.0.
- For each alpha, try h and dt above and below the calculated threshold.
- Compare numerical solutions against analytic solution after 10 time units 
    using the sum of the squares of their differences (woudl be zero if 
    numerical solution is perfectly consistent with analytic solution). 

- Setting the tolerance for "modulus of relative change in disturbance (u) over 
    a single step in space (x)" to 50 % (i.e., 0.5), we get: 
    
        h <= (ln(3/2) / (2 * alpha)) 
    and 
        dt << (2 * h ** 3)

    as rough requirements for stability for given alpha. 

--> Theory predicts that we require:
###########################################################
# - When alpha = 0.2: h <= 1.01. Corresponds to dt << 2.08     #
# - When alpha = 0.5: h <= 0.405. Corresponds to dt << 0.133   #
# - When alpha = 1.0: h <= 0.203. Corresponds to dt << 0.0167  #
# - When alpha = 1.5: h <= 0.135. Corresponds to dt << 0.00494 #
###########################################################

Results:
=======
- When alpha = 0.2: (NOT consistent with theory)
    Fixing dt = 0.001:
    - h = 4.0 --> Unstable. diff-sqr = 0.00389                (set x_max = 50.1)
    - h = 2.0 --> Unstable. diff-sqr = 0.0118                 (set x_max = 50.1)
    - h = 1.6 --> Unstable. diff-sqr = 0.0171                 (set x_max = 51.1)
    - h = 1.4 --> Unstable. diff-sqr = 0.0195                 (set x_max = 50.1)
    - h = 1.2 --> Unstable. diff-sqr = 0.0232                 (set x_max = 50.1)
    - h = 1.0 --> Unstable. diff-sqr = 0.0289                 (set x_max = 50.1)
    - h = 0.6 --> Unstable. diff-sqr = 0.0584                 (set x_max = 50.1)
    - h = 0.4 --> Unstable. diff-sqr = 0.0920                 (set x_max = 50.1)
    - h = 0.2 --> Unstable. diff-sqr = 0.201                  (set x_max = 50.1)
    - h = 0.1 --> Unstable. diff-sqr = 0.388                  (set x_max = 50.1)
    Fixing h = 1.0:
    - dt = 2 --> RuntimeWarning: overflow and invalid value.     (i from 1 to 6)
    - dt = 1 --> Unstable. diff-sqr = 0.0222.                   (i from 1 to 11)
    - dt = 0.5 --> Unstable. diff-sqr = 0.0269.                 (i from 1 to 21)
    - dt = 0.1 --> Unstable. diff-sqr = 0.0289.                (i from 1 to 101)
    - dt = 0.01 --> Unstable. diff-sqr = 0.0289.              (i from 1 to 1001)
    - dt = 0.001 --> Unstable. diff-sqr = 0.0289.            (i from 1 to 10001)
    - dt = 0.0001 --> Unstable. diff-sqr = 0.0289.          (i from 1 to 100001)

- When alpha = 0.5: (Consistent with theory)
    Fixing dt = 0.001:
    - h = 4.0 --> Unstable. diff-sqr = 46.59                  (set x_max = 50.1)
    - h = 2.0 --> Unstable. diff-sqr = 5.71                   (set x_max = 50.1)
    - h = 1.2 --> Slightly unstable. diff-sqr = 1.15          (set x_max = 50.1)
    - h = 0.7 --> Slightly stable. diff-sqr = 0.205           (set x_max = 50.3)
    - h = 0.6 --> Stable. diff-sqr = 0.108                    (set x_max = 50.1)
    - h = 0.1 --> Very stable. diff-sqr = 0.028               (set x_max = 50.1)
    Fixing h = 0.4:
    - dt = 1 --> RuntimeWarning: overflow and invalid value.    (i from 1 to 11)
    - dt = 0.25 --> RuntimeWarning: overflow and invalid value. (i from 1 to 41)
    - dt = 0.1 --> RuntimeWarning: overflow and invalid value. (i from 1 to 101)
    - dt = 0.01 --> Stable. diff-sqr = 0.0369.                (i from 1 to 1001)
    - dt = 0.001 --> Stable. diff-sqr = 0.0369.              (i from 1 to 10001)
    - dt = 0.0001 --> Stable. diff-sqr = 0.0369.            (i from 1 to 100001)

- When alpha = 1.0: (consistent with theory)
    Fixing dt = 0.001:
    - h = 1.0 --> Unstable. diff-sqr = 404.38                 (set x_max = 50.1)
    - h = 0.5 --> Slightly unstable. diff-sqr = 366.91        (set x_max = 50.5)
    - h = 0.3 --> Stable. diff-sqr = 57.45                    (set x_max = 50.1)
    - h = 0.1 --> Very stable. diff-sqr = 3.27                (set x_max = 50.1)
    Fixing h = 0.2:
    - dt = 1 --> RuntimeWarning: overflow and invalid value.    (i from 1 to 11)
    - dt = 0.1 --> RuntimeWarning: overflow and invalid value. (i from 1 to 101)
    - dt = 0.01 --> RuntimeWarning: overflow and invalid value.(i from 1 to 1001)
    - dt = 0.001 --> Stable. diff-sqr = 15.32.               (i from 1 to 10001)
    - dt = 0.0001 --> Stable. diff-sqr = 15.32.             (i from 1 to 100001)

- When alpha = 1.5: (consistent with theory except when h = 0.5)
    Fixing dt = 0.001:
    - h = 1.0 --> RuntimeWarning: overflow and invalid value.(set x_max = 100.1)
    - h = 0.5 --> Unstable. diff-sqr = 2799.194              (set x_max = 100.5)
    - h = 0.3 --> Slightly stable. diff-sqr = 4203.389       (set x_max = 100.3)
    - h = 0.2 --> Stable. diff-sqr = 3199.827                (set x_max = 100.1)
    - h = 0.1 --> Very stable. diff-sqr = 924.278            (set x_max = 100.3)
    Fixing h = 0.1:
    - dt = 1 --> RuntimeWarning: overflow and invalid value.    (i from 1 to 11)
    - dt = 0.1 --> RuntimeWarning: overflow and invalid value. (i from 1 to 101)
    - dt = 0.01 --> RuntimeWarning: overflow and invalid value. (i from 1 to 1001)
    - dt = 0.001 --> Stable. diff-sqr = 924.278.            (i from 1 to 10001)
    - dt = 0.0001 --> Stable. diff-sqr = 924.277.          (i from 1 to 100001)

################################################################################

Comment:
===========
- The diff-squared value is not always a good indicator of stability. 
- Its consistency with how stable the numerical solutions appear on plots vary 
    depending on alpha. 

'''

x_min = -5
x_max = 50.1
alpha = 0.2 # Soliton Parameter.
h = 1.0 # 1.0
dt = 1 # 0.001

plt.figure(figsize = (14, 6)) 
x = np.linspace(int(x_min), int(x_max), int((x_max - x_min) / h))
u = KdeV_exact(alpha, x, 0)
plt.plot(x, u, label = 't = 0', linewidth = 3) # Plot u at t = 0. 

time = [0]
u_list = [u]
max_list = [np.amax(u)]
diff_squared = [0]
sum_u_list = [np.sum(u)]
integral_list = [Simpson(x, u)]
for i in range (1, 11): # 2001
    time.append(i * dt)
    u = RK_4(u, h, dt)
    u_list.append(u)
    max_list.append(np.amax(u))
    diff_squared.append(np.sum((KdeV_exact(alpha, x, i * dt) - u) ** 2))
    sum_u_list.append(np.sum(u))
    integral_list.append(Simpson(x, u))
    if i % 2 == 0:
        plt.plot(x, u, linewidth = 3, label = 't = {}'.format(i * dt))
        
# plt.title('Propagation of a Single Soliton - RK_4')  
plt.xlabel('Displacement [x]') 
plt.ylabel('Disturbance [u]') 
plt.legend()
# plt.savefig('stability_check_propagation_alpha02.png', bbox_inches = "tight", 
#             dpi = 1000)
plt.show()

### Amplitude of soliton remains constaint across several transits.
plt.figure(figsize = (14, 6)) 
plt.plot(time, max_list, linewidth = 3)
plt.axhline(y = sum(max_list) / len(max_list), color = 'orange', 
            linestyle = '--', linewidth = 3, label = 'Mean amplitude')
# plt.title('Ampitude of a Soliton Over 10 Time Units')
plt.ylim([0.46, 0.495])
plt.xlabel('Time [t]')
plt.ylabel('Amplitude [$u_{max}$]')
plt.legend()
# plt.savefig('stability_check_amplitude_cons_alpha02.png', 
#             bbox_inches = "tight", dpi = 1000)   
plt.show()

# Conservation of the sum of u(i,n) over all x(i) at each t(n):
plt.figure(figsize = (14, 6)) 
plt.plot(time, sum_u_list, linewidth = 3)
# plt.title('Conservation of the sum of u(x(i),t(n)) over all x(i)')
# plt.xlim(0, 30)
# plt.ylim(245.660, 245.670)
plt.xlabel('Time [t]')
plt.ylabel('Sum of $u^{n}_{i}$ over all i')
# plt.savefig('stability_check_sum_cons_alpha02.png', bbox_inches = "tight", 
#             dpi = 1000)
plt.show()

# Conservation of Area under the soliton curve (u(x,t) vs. x):
plt.figure(figsize = (14, 6)) 
plt.plot(time, integral_list, linewidth = 3)
plt.axhline(y = sum(integral_list) / len(integral_list), color = 'orange', 
            linestyle = '--', linewidth = 3, label = 'Mean area')
# plt.title('Conservation of Area under the soliton curve')
# plt.xlim(0, 30)
plt.ylim(4.12, 4.26)
plt.xlabel('Time [t]')
plt.ylabel('Area Under Soliton Curve')
plt.legend()
# plt.savefig('stability_check_area_cons_alpha02.png', bbox_inches = "tight", 
#             dpi = 1000)
plt.show()

### Comparison against analytic solutions via the sum of differences squared.
plt.figure(figsize = (14, 6))  
plt.plot(time, diff_squared, linewidth = 3)
# plt.title('Difference Squared Between Numerical and Analytic Solutions')
plt.xlabel('Time [t]')
plt.ylabel('Difference Squared')
# plt.savefig('stability_check_comparison_alpha02.png', bbox_inches = "tight", 
#             dpi = 1000)   
plt.show()

print('Max diff-sqr = ', max(diff_squared))

# Space-time diagram
plt.figure(figsize = (14, 6)) 
X, T = np.meshgrid(x, time)
plt.contourf(X, T, u_list, 20, cmap = 'rainbow')
plt.colorbar()
plt.xlabel('Displacement [x]')
plt.ylabel('Time [t]')
# plt.savefig('stability_check_spacetime_alpha02.png', bbox_inches = "tight", 
#             dpi = 1000)
plt.show()



################################################################################
#%%
'''
3.1.0 - Collision of Two Solitons

Findings:
---------
- Overall shape of the solitons are conserved before and after collision. 
- Sum of amplitudes of solitons are NOT conserved during a collision. 
    It decreases (non-linearity). 
- Sum of u(x(i),t(n)) over all x(i) is conserved across different t(n). 
- Area under the soliton curve (u(x,t) vs. x) is roughly conserved across 
  different t(n):
- Total velocity of the solitions is conserved.
    - Solitons with similar initial speeds exchange velocities during a 
      collision, while solitons with very different initial speeds retain their 
      original velocities. 
- Phase shifts occur during a collision. The phase shift of one is equal and 
  opposite to the other's. 
    --> Total phase of the solitons are conserved. 
'''

### Collision of two solitons:
x_min_1 = 0
x_max_1 = 81
a1 = 1.2 # 1.0
a2 = 0.8 # 0.8
h = 0.2 # 0.1
dt = 0.001

plt.figure(figsize = (14, 6)) 
x_1 = np.linspace(int(x_min_1), int(x_max_1), int((x_max_1 - x_min_1) / h))
u_1 = KdeV_exact(a1, x_1, 1) + KdeV_exact(a2, x_1, 10)
plt.plot(x_1, u_1, linewidth = 3, label = 't = 0') # Initial plot at t == 0.

# Initialise lists with values at t == 0.
u_list_1 = [u_1]
time_1 = [0]
max_list_1 = [np.amax(u_1)]
sum_u_list_1 = [np.sum(u_1)]
integral_list_1 = [Simpson(x_1, u_1)]

for i in range (1, 12001):
    u_1 = RK_4(u_1, h, dt)
    u_list_1.append(u_1)
    max_list_1.append(np.amax(u_1))
    time_1.append(i * dt)
    sum_u_list_1.append(np.sum(u_1))
    integral_list_1.append(Simpson(x_1, u_1))
    if i % 3000 == 0:
        plt.plot(x_1, u_1, linewidth = 3, label = 't = {}'.format(i * dt))        

# plt.title('Collision of Two Solitons')
plt.xlabel('Displacement [x]')
plt.ylabel('Disturbance [u]')
plt.legend()
# plt.savefig('collision_propagation_1.png', bbox_inches = "tight", dpi = 1000)
plt.show()

# Verify that sum of amplitudes of solitons is NOT conserved during collision:
plt.figure(figsize = (14, 6)) 
plt.plot(time_1, max_list_1, linewidth = 3)
# plt.title('Non-linearity - Amplitude is Not Conserved')
plt.xlim(0, 12)
plt.xlabel('Time [t]')
plt.ylabel('Total amplitude')
# plt.savefig('collision_amplitude_1.png', bbox_inches = "tight", dpi = 1000)
plt.show()

# Conservation of the sum of u(i,n) over all x(i) at each t(n):
plt.figure(figsize = (14, 6)) 
plt.plot(time_1, sum_u_list_1, linewidth = 3)
plt.xlim(0, 12)
plt.ylim(239.405, 239.41)
plt.xlabel('Time [t]')
plt.ylabel('Sum of $u^{n}_{i}$ over all i')
# plt.savefig('collision_sum_1.png', bbox_inches = "tight", dpi = 1000)
plt.show()

# Conservation of Area under the soliton curve (u(x,t) vs. x):
plt.figure(figsize = (14, 6)) 
plt.plot(time_1, integral_list_1, linewidth = 3)
plt.xlim(0, 12)
plt.ylim(47.8, 48.2)
plt.xlabel('Time [t]')
plt.ylabel('Area under soliton curve')
# plt.savefig('collision_area_1.png', bbox_inches = "tight", dpi = 1000)
plt.show()

# Space-time diagram
plt.figure(figsize = (14, 6)) 
X_1, T_1 = np.meshgrid(x_1, time_1)
plt.contourf(X_1, T_1, u_list_1, 20, cmap = 'rainbow')
plt.colorbar()
plt.xlabel('Displacement [x]')
plt.ylabel('Time [t]')
# plt.savefig('collision_spacetime_1.png', bbox_inches = "tight", dpi = 1000)
plt.show()


################################################################################
#%%
'''3.1.1 - Collisions of two solitons with similar initial speeds'''

x_min_2 = 0
x_max_2 = 151
a3 = 1.05 # 0.8
a4 = 1.0 # 0.75
h = 0.2 # 0.1
dt = 0.001

plt.figure(figsize = (14, 6)) 
x_2 = np.linspace(int(x_min_2), int(x_max_2), int((x_max_2 - x_min_2) / h))
u_2 = KdeV_exact(a3, x_2, 1) + KdeV_exact(a4, x_2, 3.5)
plt.plot(x_2, u_2, linewidth = 3, label = 't = 0') # Initial plot at t == 0.

# Initialise lists with values at t == 0.
u_list_2 = [u_2]
time_2 = [0]
max_list_2 = [np.amax(u_2)]
sum_u_list_2 = [np.sum(u_2)]
integral_list_2 = [Simpson(x_2, u_2)]

for i in range (1, 30001): # 100000
    u_2 = RK_4(u_2, h, dt)
    u_list_2.append(u_2)
    max_list_2.append(np.amax(u_2))
    time_2.append(i * dt)
    sum_u_list_2.append(np.sum(u_2))
    integral_list_2.append(Simpson(x_2, u_2))
    if i % 5000 == 0: # 10000
        plt.plot(x_2, u_2, linewidth = 3, label = 't = {}'.format(i * dt))

# plt.title('Collision of Two Solitons with Similar Initial Speeds')
plt.xlabel('Displacement [x]')
plt.ylabel('Disturbance [u]')
plt.legend()
# plt.savefig('collision_propagation_2.png', bbox_inches = "tight", dpi = 1000)
plt.show()

# Verify that sum of amplitudes of solitons is NOT conserved during collision:
plt.figure(figsize = (14, 6)) 
plt.plot(time_2, max_list_2, linewidth = 3)
# plt.title('Non-linearity - Amplitude is Not Conserved')
plt.xlim(0, 30)
plt.xlabel('Time [t]')
plt.ylabel('Sum of amplitudes')
# plt.savefig('collision_amplitude_2.png', bbox_inches = "tight", dpi = 1000)
plt.show()

# Conservation of the sum of u(i,n) over all x(i) at each t(n):
plt.figure(figsize = (14, 6)) 
plt.plot(time_2, sum_u_list_2, linewidth = 3)
# plt.title('Conservation of the sum of u(x(i),t(n)) over all x(i)')
plt.xlim(0, 30)
plt.ylim(245.660, 245.670)
plt.xlabel('Time [t]')
plt.ylabel('Sum of $u^{n}_{i}$ over all i')
# plt.savefig('collision_sum_2.png', bbox_inches = "tight", dpi = 1000)
plt.show()

# Conservation of Area under the soliton curve (u(x,t) vs. x):
plt.figure(figsize = (14, 6)) 
plt.plot(time_2, integral_list_2, linewidth = 3)
plt.title('Conservation of Area under the soliton curve')
plt.xlim(0, 30)
plt.ylim(49.0, 49.4)
plt.xlabel('Time [t]')
plt.ylabel('Area under soliton curve')
# plt.savefig('collision_area_2.png', bbox_inches = "tight", dpi = 1000)
plt.show()

# Space-time diagram
plt.figure(figsize = (14, 6)) 
X_2, T_2 = np.meshgrid(x_2, time_2)
plt.contourf(X_2, T_2, u_list_2, 20, cmap = 'rainbow')
# plt.title('Space-time Diagram for Similar Initial Speeds')
plt.colorbar()
plt.xlim(30, 110)
plt.ylim(4, 27)
plt.xlabel('Displacement [x]')
plt.ylabel('Time [t]')
# plt.savefig('collision_spacetime_2.png', bbox_inches = "tight", dpi = 1000)
plt.show()

################################################################################
#%%
'''3.1.2 - Collision of two solitons with very different initial speeds'''


x_min_3 = 5 # 5
x_max_3 = 40 # 40
a5 = 1.7
a6 = 0.8
h = 0.2 # 0.1
dt = 0.001

plt.figure(figsize = (14, 6)) 
x_3 = np.linspace(int(x_min_3), int(x_max_3), int((x_max_3 - x_min_3) / h))
u_3 = KdeV_exact(a5, x_3, 1) + KdeV_exact(a6, x_3, 8)
plt.plot(x_3, u_3, label = 't = 0') # Initial plot at t == 0.

# Initialise lists with values at t == 0.
u_list_3 = [u_3]
time_3 = [0]
max_list_3 = [np.amax(u_3)]
sum_u_list_3 = [np.sum(u_3)]
integral_list_3 = [Simpson(x_3, u_3)]

for i in range (1, 2001): # 100000
    u_3 = RK_4(u_3, h, dt)
    u_list_3.append(u_3)
    max_list_3.append(np.amax(u_3))
    time_3.append(i * dt)
    sum_u_list_3.append(np.sum(u_3))
    integral_list_3.append(Simpson(x_3, u_3))
    if i % 500 == 0: # 10000
        plt.plot(x_3, u_3, label = 't = {}'.format(i * dt))
        
# plt.title('Collision of Two Solitons with Very Different Initial Speeds')
plt.xlabel('Displacement [x]')
plt.ylabel('Disturbance [u]')
plt.legend()
# plt.savefig('collision_propagation_3.png', bbox_inches = "tight", dpi = 1000)
plt.show()

# Verify that sum of amplitudes of solitons is NOT conserved during collision:
plt.figure(figsize = (14, 6)) 
plt.plot(time_3, max_list_3)
# plt.title('Non-linearity - Amplitude is Not Conserved')
plt.xlim(0, 2)
plt.xlabel('Time [t]')
plt.ylabel('Sum of amplitudes')
# plt.savefig('collision_amplitude_3.png', bbox_inches = "tight", dpi = 1000)
plt.show()

# Conservation of the sum of u(i,n) over all x(i) at each t(n):
plt.figure(figsize = (14, 6)) 
plt.plot(time_3, sum_u_list_3)
# plt.title('Conservation of the sum of u(x(i),t(n)) over all x(i)')
plt.xlim(0, 2)
plt.ylim(298.282, 298.290)
plt.xlabel('Time [t]')
plt.ylabel('Sum of $u^{n}_{i}$ over all i')
# plt.savefig('collision_sum_3.png', bbox_inches = "tight", dpi = 1000)
plt.show()

# Conservation of Area under the soliton curve (u(x,t) vs. x):
plt.figure(figsize = (14, 6)) 
plt.plot(time_3, integral_list_3)
# plt.title('Conservation of Area under the soliton curve')
plt.xlim(0, 2)
plt.ylim(59.85, 60.19)
plt.xlabel('Time [t]')
plt.ylabel('Area under soliton curve')
# plt.savefig('collision_area_3.png', bbox_inches = "tight", dpi = 1000)
plt.show()

# Space-time diagram
plt.figure(figsize = (14, 6)) 
X_3, T_3 = np.meshgrid(x_3, time_3)
plt.contourf(X_3, T_3, u_list_3, 20, alpha = 1, cmap = 'rainbow')
# plt.title('Space-time Diagram for Very Different Initial Speeds')
plt.colorbar()
plt.xlim(10, 35)
plt.ylim(0, 2.0)
plt.xlabel('Displacement [x]')
plt.ylabel('Time [t]')
# plt.savefig('collision_spacetime_3.png', bbox_inches = "tight", dpi = 1000)
plt.show()


################################################################################
#%%
'''3.2.0 - Wave Breaking - Initial sinusoidal waveform'''

def sin(x, a, b):
    '''
    Returns the positive part of a sine curve between x == 0 and π/b. 
    Amplitude is set by parameter 'a'.
    '''
    f = np.array([])
    for i in range(len(x)):       
        if 0 < x[i] < np.pi / b: 
            f = np.append(f, a * np.sin(b * x[i]))
        else:
            f = np.append(f, 0)
    return(f)


################################################################################
#%%
'''3.2.1 - Wave braking when wavelength > initial amplitude'''

x = np.linspace(0, 6, 601)
a_1 = 1 # Initial amplitude
b_1 = 1 # Spatial Frequency k
h = 0.2
dt = 0.001

plt.figure(figsize = (14, 6)) 
f_sin = sin(x, a_1, b_1)
print('period = ', 2 * np.pi / b_1)
print('initial amplitude = ', a_1)
plt.plot(x, f_sin, linewidth = 3, label = 't = 0') # Initial plot at t == 0.
# print('area = ', Simpson(x, f_sin))

for i in range (1, 100001):
    f_sin = RK_4(f_sin, h, dt)
    if i % 50000 == 0:
        plt.plot(x, f_sin, linewidth = 3, label = 't = {}'.format(i * dt))
        # print('area = ',Simpson(x, f_sin))

# plt.title('Wave breaking: Amplitude = {}, Initial period = {}'.format(
#     a_1, round(2 * np.pi / b_1, 2)))
plt.xlabel('Displacement [x]')
plt.ylabel('Disturbance [u]')
plt.xlim(0, 6)
plt.legend()
# plt.savefig('wavebreak_1.png', bbox_inches = "tight", dpi = 1000)
plt.show() 

################################################################################
#%%
'''3.2.2 - Wave braking when wavelength == initial amplitude'''


x = np.linspace(0, 6, 601)
a_2 = 1 # Initial amplitudes
b_2 = 2 * np.pi # Spatial Frequency 
h = 0.2  # originally h == 0.1.
dt = 0.001

plt.figure(figsize = (14, 6)) 
f_sin = sin(x, a_2, b_2)
print('period = ', 2 * np.pi / b_2)
print('initial amplitude = ', a_2)
plt.plot(x, f_sin, linewidth = 3, label = 't = 0') # Initial plot at t == 0.
# print('area = ', Simpson(x, f_sin))

for i in range (1, 100001):
    f_sin = RK_4(f_sin, h, dt)
    if i % 50000 == 0:
        plt.plot(x, f_sin, linewidth = 3, label = 't = {}'.format(i * dt))
        # print('area = ',Simpson(x, f_sin))

# plt.title('Wave breaking: Amplitude = {}, Initial period = {}'.format(
#     round(a_2, 2), round(2 * np.pi / b_2, 2)))
plt.xlabel('Distance')
plt.ylabel('Amplitude')
plt.xlim(0, 6)
plt.legend()
plt.show()


################################################################################
#%%
'''3.2.3 - Wave braking when wavelength < initial amplitude'''

x = np.linspace(0, 6, 601)
a_3 = 1 # Initial amplitude
b_3 = 10 # Spatial Frequency k
h = 0.2  # originally h == 0.1.
dt = 0.001

plt.figure(figsize = (14, 6)) 
f_sin = sin(x, a_3, b_3)
print('period = ', 2 * np.pi / b_3)
print('initial amplitude = ', a_3)
plt.plot(x, f_sin, linewidth = 3, label = 't = 0') # Initial plot at t == 0.
# print('area = ', Simpson(x, f_sin))

for i in range (1, 100001):
    f_sin = RK_4(f_sin, h, dt)
    if i % 50000 == 0:
        plt.plot(x, f_sin, linewidth = 3, label = 't = {}'.format(i * dt))
        # print('area = ',Simpson(x, f_sin))

# plt.title('Wave breaking: Amplitude = {}, Initial period = {}'.format(
#     a_3, round(2 * np.pi / b_3, 2)))
plt.xlabel('Displacement [x]')
plt.ylabel('Disturbance [u]')
plt.xlim(0, 6)
plt.legend()
# plt.savefig('wavebreak_3.png', bbox_inches = "tight", dpi = 1000)
plt.show()


################################################################################
#%%
'''3.3.0 - Unstable shock wave'''

def u_diff_t_shockwave(u, h):
    '''
    Approximation of the partial derivative of disturbance / displacement u 
    w.r.t time t for a shock wave. 
    '''
    return - advection(u, h)

def RK_4_shockwave(u, h, dt):
    fa = u_diff_t_shockwave(u, h)
    fb = u_diff_t_shockwave(u + fa * dt / 2, h)
    fc = u_diff_t_shockwave(u + fb * dt / 2, h)
    fd = u_diff_t_shockwave(u + fc * dt, h)
    return u + (fa + 2 * fb + 2 * fc + fd) * dt / 6


### RK_4 without 3rd order dispersive term --> Propagation of a Shock Wave:

x_min = -3.5
x_max = 5.1
alpha = 1 # Soliton Parameter.
h = 0.1
dt = 0.001

plt.figure(figsize = (14, 6)) 
x = np.linspace(int(x_min), int(x_max), int((x_max - x_min) / h))
u = KdeV_exact(alpha, x, 0)
plt.plot(x, u, linewidth = 3, label = 't = 0') # Plot u at t = 0. 

u_list = [u]
time = [0]
max_list = [np.amax(u)]
sum_u_list = [np.sum(u)]
integral_list = [Simpson(x, u)]

for i in range (1, 501): # 2001
    u = RK_4_shockwave(u, h, dt)
    u_list.append(u)
    time.append(i * dt)
    max_list.append(np.amax(u))
    sum_u_list.append(np.sum(u))
    integral_list.append(Simpson(x, u))
    if i == 50 or i == 100 or i == 200 or i == 500:
        plt.plot(x, u, linewidth = 3, label = 't = {}'.format(round(i * dt, 2)))
        
# plt.title('Propagation of a Shock Wave - RK_4')  
plt.xlabel('Displacement [x]') 
plt.ylabel('Disturbance [u]')   
plt.legend()
# plt.savefig('shock_unstable_propagation.png', bbox_inches = "tight", 
#             dpi = 1000)
plt.show()

### Amplitude of a shock wave does NOT remain constaint over time.
plt.figure(figsize = (14, 6)) 
plt.plot(time, max_list)
# plt.title('Ampitude of a Shock Wave Over 1 Time Unit')
# plt.ylim([0, 15])
plt.xlabel('Time [t]')
plt.ylabel('Amplitude')   
plt.show()

# Conservation of the sum of u(i,n) over all x(i) at each t(n): 
plt.figure(figsize = (14, 6)) 
plt.plot(time, sum_u_list)
# plt.title('Conservation of the sum of u(x(i),t(n)) over all x(i)')
plt.ylim(251.4, 251.45)
plt.xlabel('Time [t]')
plt.ylabel('Sum of $u^{n}_{i}$ over all i')
plt.show()

# Conservation of Area under the soliton curve (u(x,t) vs. x):
plt.figure(figsize = (14, 6)) 
plt.plot(time, integral_list)
# plt.title('Conservation of Area under a Shock Wave')
plt.ylim(23.930, 23.950)
plt.xlabel('Time [t]')
plt.ylabel('Area under shock wave')
plt.show()

# Space-time diagram
plt.figure(figsize = (14, 6)) 
X, T = np.meshgrid(x, time)
plt.contourf(X, T, u_list, 20, cmap = 'rainbow')
plt.colorbar()
plt.xlabel('Displacement [x]')
plt.ylabel('Time [t]')
# plt.savefig('shock_unstable_spacetime.png', bbox_inches = "tight", dpi = 1000)
plt.show()


################################################################################
#%%
'''3.3.1 - Stabilising a shock wave by introducing a 2nd order diffusive term'''

def Diffusion(D, u, h):
    '''
    2nd Order diffusive term. This term damps the wave and stabilises the 
    shock wave solution.
    '''
    u_i_plus_1 = np.append(u[1:], u[:1])
    u_i_minus_1 = np.append(u[-1:], u[:-1])
    return D * (u_i_plus_1 - 2 * u + u_i_minus_1) / (h ** 2)

def u_diff_t_shockwave_diffusion(D, u, h):
    return Diffusion(D, u, h) - advection(u, h)

def RK_4_shockwave_diffusion(D, u, h, dt):
    fa = u_diff_t_shockwave_diffusion(D, u, h)
    fb = u_diff_t_shockwave_diffusion(D, u + fa * dt / 2, h)
    fc = u_diff_t_shockwave_diffusion(D, u + fb * dt / 2, h)
    fd = u_diff_t_shockwave_diffusion(D, u + fc * dt, h)
    return u + (fa + 2 * fb + 2 * fc + fd) * dt / 6


x_min = -6.0
x_max = 10.1
alpha = 1 # Soliton Parameter.
D = 1 # Diffusion coefficient.
h = 0.1
dt = 0.001

plt.figure(figsize = (14, 6)) 
x = np.linspace(int(x_min), int(x_max), int((x_max - x_min) / h))
u = KdeV_exact(alpha, x, 0)
plt.plot(x, u, linewidth = 3, label = 't = 0') # Plot u at t = 0. 

u_list = [u]
time = [0]
max_list = [np.amax(u)]
sum_u_list = [np.sum(u)]
integral_list = [Simpson(x, u)]

for i in range (1, 2001): # 2001
    u = RK_4_shockwave_diffusion(D, u, h, dt)
    u_list.append(u)
    time.append(i * dt)
    max_list.append(np.amax(u))
    sum_u_list.append(np.sum(u))
    integral_list.append(Simpson(x, u))
    if i % 400 == 0:
        plt.plot(x, u, linewidth = 3, label = 't = {}'.format(round(i * dt, 2)))
        
# plt.title('Propagation of a Stable Shock Wave - RK_4')  
plt.xlabel('Displacement [x]') 
plt.ylabel('Disturbance [u]')   
plt.legend()
# plt.savefig('shock_stable_propagation.png', bbox_inches = "tight", dpi = 1000)
plt.show()

### Amplitude of a shock wave does NOT remain constaint over time.
plt.figure(figsize = (14, 6)) 
plt.plot(time, max_list)
# plt.title('Ampitude of a Stable Shock Wave Over 1 Time Unit')
# plt.ylim([0, 15])
plt.xlabel('Time [t]')
plt.ylabel('Amplitude')   
plt.show()

# Conservation of the sum of u(i,n) over all x(i) at each t(n):
plt.figure(figsize = (14, 6)) 
plt.plot(time, sum_u_list)
# plt.title('Conservation of the sum of u(x(i),t(n)) over all x(i)')
plt.ylim(239.98, 240.02)
plt.xlabel('Time [t]')
plt.ylabel('Sum of $u^{n}_{i}$ over all i')
plt.show()

# Conservation of Area under the soliton curve (u(x,t) vs. x):
plt.figure(figsize = (14, 6)) 
plt.plot(time, integral_list)
# plt.title('Conservation of Area under a Stable Shock Wave')
plt.ylim(23.99, 24.01)
plt.xlabel('Time [t]')
plt.ylabel('Area under shock wave')
plt.show()

# Space-time diagram
plt.figure(figsize = (14, 6)) 
X, T = np.meshgrid(x, time)
plt.contourf(X, T, u_list, 20, cmap = 'rainbow')
plt.colorbar()
plt.xlabel('Displacement [x]')
plt.ylabel('Time [t]')
# plt.savefig('shock_stable_spacetime.png', bbox_inches = "tight", dpi = 1000)
plt.show()

