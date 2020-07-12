import os
import sys
import numpy as np
import matplotlib.pyplot as plt

#Find the name of the file you eant post-process: if no argument is present supply with a default one
nargin = len(sys.argv)
if(nargin == 2):
    filename = sys.argv[1]
elif(nargin == 1):
    filename = 'test.cfg'
else:
    print("Too many input arguments")
    sys.exit(1)

#Initialize saving directory and reinitialization type with the default choices of the code
saving_dir = 'Sim'
reinit_type = 'Non_Conservative_Hyperbolic'

#Read the file passed in input
config_file = open(filename, "r")
found_dir = False #Flag to check if we found the name of the directory
found_reinit = False #Flag to check if we found the type of reinitialization
for line in config_file.read().splitlines():
    if('Saving_Directory' in line):
        found_dir = True
        saving_dir = line[19::]
    elif('Reinit_Type' in line):
        found_reinit = True
        reinit_type = line[14::]
    if(found_dir == True and found_reinit == True):
        break

#Close the file
config_file.close()

#Read the file found
datafile = os.getcwd() + '/' + saving_dir + '/benchmark_series.dat'
data = np.loadtxt(datafile)
if(reinit_type == 'Non_Conservative_Hyperbolic'):
    assert np.size(data) % 8 == 0, 'Wrong number of values in the file'
    t = data[0::8]
    Vol = data[1::8]
    chi = data[2::8]
    Xc = data[3::8]
    Yc = data[4::8]
    Uc = data[5::8]
    Vc = data[6::8]
    L2_grad_phi = data[7::8]
elif(reinit_type == 'Conservative'):
    assert np.size(data) % 7 == 0, 'Wrong number of values in the file'
    t = data[0::7]
    Vol = data[1::7]
    chi = data[2::7]
    Xc = data[3::7]
    Yc = data[4::7]
    Uc = data[5::7]
    Vc = data[6::7]

#Plot volume behaviour
plt.figure()
plt.plot(t,Vol)
plt.xlabel('t')
plt.ylabel('Area')
plt.title('Area evolution in time',fontweight="bold")

#Plot circularity behaviour
plt.figure()
plt.plot(t,chi)
plt.xlabel('t')
plt.ylabel('Circularity')
plt.title('Degree of circularity evolution in time',fontweight="bold")

#Plot horizontal component for centroid and rising velocity
plt.figure()
plt.subplot(1,2,1)
plt.plot(t,Xc)
plt.xlabel('t')
plt.ylabel('$x_c$')
plt.title('Time evolution of $x_c$ coordinate of the centroid',fontweight="bold")
plt.subplot(1,2,2)
plt.plot(t,Uc)
plt.xlabel('t')
plt.ylabel('$u_c$')
plt.title('Time evolution of $u_c$ coordinate of the rising velocity',fontweight="bold")

#Plot vertical component for centroid and rising velocity
plt.figure()
plt.subplot(1,2,1)
plt.plot(t,Yc)
plt.xlabel('t')
plt.ylabel('$y_c$')
plt.title('Time evolution of $y_c$ coordinate of the centroid',fontweight="bold")
plt.subplot(1,2,2)
plt.plot(t,Vc)
plt.xlabel('t')
plt.ylabel('$v_c$')
plt.title('Time evolution of $v_c$ coordinate of the rising velocity',fontweight="bold")

#Plot grad_phi if necessary
if(reinit_type == 'Non_Conservative_Hyperbolic'):
    plt.figure()
    plt.plot(t,L2_grad_phi)
    plt.xlabel('t')
    plt.ylabel('grad($\phi$)')
    plt.title('Time evolution of average grad($\phi$)',fontweight="bold")

plt.show()
