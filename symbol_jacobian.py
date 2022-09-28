import sympy as sym
import numpy as np 

sym.init_printing(use_unicode=True)

theta = sym.var('theta[0:17]')
### J0

# TRA0 = sym.Matrix([[sym.cos(theta), -sym.sin(theta), 0, a], 
#                 [sym.sin(theta)*sym.cos(alpha), sym.cos(theta)*sym.cos(alpha), -sym.sin(alpha), -d*sym.sin(alpha)], 
#                 [sym.sin(theta)*sym.sin(alpha), sym.cos(theta)*sym.sin(alpha), sym.cos(alpha), d*sym.cos(alpha)],
#                 [0, 0, 0, 1]])

### Arm 
### Right Arm J0
TRA0 = sym.Matrix([[sym.cos(-sym.pi/2), -sym.sin(-sym.pi/2), 0, 0], 
                [sym.sin(-sym.pi/2)*sym.cos(0), sym.cos(-sym.pi/2)*sym.cos(0), -sym.sin(0), 0*sym.sin(0)], 
                [sym.sin(-sym.pi/2)*sym.sin(0), sym.cos(-sym.pi/2)*sym.sin(0), sym.cos(0), 0*sym.cos(0)],
                [0, 0, 0, 1]])

### Right Arm J1
TRA1 = sym.Matrix([[sym.cos(theta[0] + sym.pi/2), -sym.sin(theta[0] + sym.pi/2), 0, 0.6], 
                [sym.sin(theta[0] + sym.pi/2)*sym.cos(0), sym.cos(theta[0] + sym.pi/2)*sym.cos(0), -sym.sin(0), -0.315*sym.sin(0)], 
                [sym.sin(theta[0] + sym.pi/2)*sym.sin(0), sym.cos(theta[0] + sym.pi/2)*sym.sin(0), sym.cos(0), 0.315*sym.cos(0)],
                [0, 0, 0, 1]])

### Right Arm J2
TRA2 = sym.Matrix([[sym.cos(theta[1]), -sym.sin(theta[1]), 0, 0], 
                [sym.sin(theta[1])*sym.cos(-sym.pi/2), sym.cos(theta[1])*sym.cos(-sym.pi/2), -sym.sin(-sym.pi/2), -0*sym.sin(-sym.pi/2)], 
                [sym.sin(theta[1])*sym.sin(-sym.pi/2), sym.cos(theta[1])*sym.sin(-sym.pi/2), sym.cos(-sym.pi/2), 0*sym.cos(-sym.pi/2)],
                [0, 0, 0, 1]])

### Right Arm J3
TRA3 = sym.Matrix([[sym.cos(theta[2]), -sym.sin(theta[2]), 0, 0], 
                [sym.sin(theta[2])*sym.cos(sym.pi/2), sym.cos(theta[2])*sym.cos(sym.pi/2), -sym.sin(sym.pi/2), -0.33*sym.sin(sym.pi/2)], 
                [sym.sin(theta[2])*sym.sin(sym.pi/2), sym.cos(theta[2])*sym.sin(sym.pi/2), sym.cos(sym.pi/2), 0.33*sym.cos(sym.pi/2)],
                [0, 0, 0, 1]])

### Right Arm J4
TRA4 = sym.Matrix([[sym.cos(theta[3]), -sym.sin(theta[3]), 0, 0.2], 
                [sym.sin(theta[3])*sym.cos(-sym.pi/2), sym.cos(theta[3])*sym.cos(-sym.pi/2), -sym.sin(-sym.pi/2), -0*sym.sin(-sym.pi/2)], 
                [sym.sin(theta[3])*sym.sin(-sym.pi/2), sym.cos(theta[3])*sym.sin(-sym.pi/2), sym.cos(-sym.pi/2), 0*sym.cos(-sym.pi/2)],
                [0, 0, 0, 1]])

### Right Arm J5
TRA5 = sym.Matrix([[sym.cos(theta[4] + sym.pi/2), -sym.sin(theta[4] + sym.pi/2), 0, 0], 
                [sym.sin(theta[4] + sym.pi/2)*sym.cos(sym.pi/2), sym.cos(theta[4] + sym.pi/2)*sym.cos(sym.pi/2), -sym.sin(sym.pi/2), -0.482*sym.sin(sym.pi/2)], 
                [sym.sin(theta[4] + sym.pi/2)*sym.sin(sym.pi/2), sym.cos(theta[4] + sym.pi/2)*sym.sin(sym.pi/2), sym.cos(sym.pi/2), 0.482*sym.cos(sym.pi/2)],
                [0, 0, 0, 1]])

### Right Arm J6
TRA6 = sym.Matrix([[sym.cos(theta[5] + sym.pi/2), -sym.sin(theta[5] + sym.pi/2), 0, 0.0045], 
                [sym.sin(theta[5] + sym.pi/2)*sym.cos(sym.pi/2), sym.cos(theta[5] + sym.pi/2)*sym.cos(sym.pi/2), -sym.sin(sym.pi/2), -0*sym.sin(sym.pi/2)], 
                [sym.sin(theta[5] + sym.pi/2)*sym.sin(sym.pi/2), sym.cos(theta[5] + sym.pi/2)*sym.sin(sym.pi/2), sym.cos(sym.pi/2), 0*sym.cos(sym.pi/2)],
                [0, 0, 0, 1]])

### Right Arm J7
TRA7 = sym.Matrix([[sym.cos(theta[6]), -sym.sin(theta[6]), 0, 0.03], 
                [sym.sin(theta[6])*sym.cos(sym.pi/2), sym.cos(theta[6])*sym.cos(sym.pi/2), -sym.sin(sym.pi/2), -0*sym.sin(sym.pi/2)], 
                [sym.sin(theta[6])*sym.sin(sym.pi/2), sym.cos(theta[6])*sym.sin(sym.pi/2), sym.cos(sym.pi/2), 0*sym.cos(sym.pi/2)],
                [0, 0, 0, 1]])

### Right Arm End
TRAE = sym.Matrix([[sym.cos(sym.pi), -sym.sin(sym.pi), 0, 0], 
                [sym.sin(sym.pi)*sym.cos(-sym.pi/2), sym.cos(sym.pi)*sym.cos(-sym.pi/2), -sym.sin(-sym.pi/2), -0*sym.sin(-sym.pi/2)], 
                [sym.sin(sym.pi)*sym.sin(-sym.pi/2), sym.cos(sym.pi)*sym.sin(-sym.pi/2), sym.cos(-sym.pi/2), 0*sym.cos(-sym.pi/2)],
                [0, 0, 0, 1]])

### Thumb
### Right Thumb J0
TRT0 = sym.Matrix([[sym.cos(sym.pi/2), -sym.sin(sym.pi/2), 0, -78e-3], 
                [sym.sin(sym.pi/2)*sym.cos(0), sym.cos(sym.pi/2)*sym.cos(0), -sym.sin(0), -15e-3*sym.sin(0)], 
                [sym.sin(sym.pi/2)*sym.sin(0), sym.cos(sym.pi/2)*sym.sin(0), sym.cos(0), 15e-3*sym.cos(0)],
                [0, 0, 0, 1]])

### Right Thumb J1
TRT1 = sym.Matrix([[sym.cos(theta[7]), -sym.sin(theta[7]), 0, 6.7e-3], 
                [sym.sin(theta[7])*sym.cos(sym.pi/2), sym.cos(theta[7])*sym.cos(sym.pi/2), -sym.sin(sym.pi/2), -0*sym.sin(sym.pi/2)], 
                [sym.sin(theta[7])*sym.sin(sym.pi/2), sym.cos(theta[7])*sym.sin(sym.pi/2), sym.cos(sym.pi/2), 0*sym.cos(sym.pi/2)],
                [0, 0, 0, 1]])

### Right Thumb J2 
TRT2 = sym.Matrix([[sym.cos(theta[8] + sym.pi/2), -sym.sin(theta[8] + sym.pi/2), 0, 0], 
                [sym.sin(theta[8] + sym.pi/2)*sym.cos(-sym.pi/2), sym.cos(theta[8] + sym.pi/2)*sym.cos(-sym.pi/2), -sym.sin(-sym.pi/2), -36e-3*sym.sin(-sym.pi/2)], 
                [sym.sin(theta[8] + sym.pi/2)*sym.sin(-sym.pi/2), sym.cos(theta[8] + sym.pi/2)*sym.sin(-sym.pi/2), sym.cos(-sym.pi/2), 36e-3*sym.cos(-sym.pi/2)],
                [0, 0, 0, 1]])

### Right Thumb J3 
TRT3 = sym.Matrix([[sym.cos(theta[9]), -sym.sin(theta[9]), 0, 27.5e-3], 
                [sym.sin(theta[9])*sym.cos(-sym.pi/2), sym.cos(theta[9])*sym.cos(-sym.pi/2), -sym.sin(-sym.pi/2), -0*sym.sin(-sym.pi/2)], 
                [sym.sin(theta[9])*sym.sin(-sym.pi/2), sym.cos(theta[9])*sym.sin(-sym.pi/2), sym.cos(-sym.pi/2), 0*sym.cos(-sym.pi/2)],
                [0, 0, 0, 1]])

### Right Thumb J3'
TRT3P = sym.Matrix([[sym.cos(-sym.pi/2), -sym.sin(-sym.pi/2), 0, 4.5e-3], 
                [sym.sin(-sym.pi/2)*sym.cos(0), sym.cos(-sym.pi/2)*sym.cos(0), -sym.sin(0), -0*sym.sin(0)], 
                [sym.sin(-sym.pi/2)*sym.sin(0), sym.cos(-sym.pi/2)*sym.sin(0), sym.cos(0), 0*sym.cos(0)],
                [0, 0, 0, 1]])

### Right Thumb J4
TRT4 = sym.Matrix([[sym.cos(theta[10] + sym.pi/2), -sym.sin(theta[10] + sym.pi/2), 0, 40e-3], 
                [sym.sin(theta[10] + sym.pi/2)*sym.cos(0), sym.cos(theta[10] + sym.pi/2)*sym.cos(0), -sym.sin(0), -0*sym.sin(0)], 
                [sym.sin(theta[10] + sym.pi/2)*sym.sin(0), sym.cos(theta[10] + sym.pi/2)*sym.sin(0), sym.cos(0), 0*sym.cos(0)],
                [0, 0, 0, 1]])

### Right Thumb End
TRTE = sym.Matrix([[sym.cos(0), -sym.sin(0), 0, 5e-3], 
                [sym.sin(0)*sym.cos(sym.pi/2), sym.cos(0)*sym.cos(sym.pi/2), -sym.sin(sym.pi/2), -25e-3*sym.sin(sym.pi/2)], 
                [sym.sin(0)*sym.sin(sym.pi/2), sym.cos(0)*sym.sin(sym.pi/2), sym.cos(sym.pi/2), 25e-3*sym.cos(sym.pi/2)],
                [0, 0, 0, 1]])

### Index 
### Right Index J0
TRI0 = sym.Matrix([[sym.cos(sym.pi/2), -sym.sin(sym.pi/2), 0, -156.5e-3], 
                [sym.sin(sym.pi/2)*sym.cos(0), sym.cos(sym.pi/2)*sym.cos(0), -sym.sin(0), -17.5e-3*sym.sin(0)], 
                [sym.sin(sym.pi/2)*sym.sin(0), sym.cos(sym.pi/2)*sym.sin(0), sym.cos(0), 17.5e-3*sym.cos(0)],
                [0, 0, 0, 1]])

### Right Index J1
TRI1 = sym.Matrix([[sym.cos(theta[11] + sym.pi/2), -sym.sin(theta[11] + sym.pi/2), 0, 6.2e-3], 
                [sym.sin(theta[11] + sym.pi/2)*sym.cos(sym.pi), sym.cos(theta[11] + sym.pi/2)*sym.cos(sym.pi), -sym.sin(sym.pi), -0*sym.sin(sym.pi)], 
                [sym.sin(theta[11] + sym.pi/2)*sym.sin(sym.pi), sym.cos(theta[11] + sym.pi/2)*sym.sin(sym.pi), sym.cos(sym.pi), 0*sym.cos(sym.pi)],
                [0, 0, 0, 1]])

### Right Index J2
TRI2 = sym.Matrix([[sym.cos(theta[12] - sym.pi/2), -sym.sin(theta[12] - sym.pi/2), 0, -20e-3], 
                [sym.sin(theta[12] - sym.pi/2)*sym.cos(sym.pi/2), sym.cos(theta[12] - sym.pi/2)*sym.cos(sym.pi/2), -sym.sin(sym.pi/2), -0*sym.sin(sym.pi/2)], 
                [sym.sin(theta[12] - sym.pi/2)*sym.sin(sym.pi/2), sym.cos(theta[12] - sym.pi/2)*sym.sin(sym.pi/2), sym.cos(sym.pi/2), 0*sym.cos(sym.pi/2)],
                [0, 0, 0, 1]])

### Right Index J2'
TRI2P = sym.Matrix([[sym.cos(-sym.pi/2), -sym.sin(-sym.pi/2), 0, 3e-3], 
                [sym.sin(-sym.pi/2)*sym.cos(0), sym.cos(-sym.pi/2)*sym.cos(0), -sym.sin(0), -0*sym.sin(0)], 
                [sym.sin(-sym.pi/2)*sym.sin(0), sym.cos(-sym.pi/2)*sym.sin(0), sym.cos(0), 0*sym.cos(0)],
                [0, 0, 0, 1]])

### Right Index J3
TRI3 = sym.Matrix([[sym.cos(theta[13] + sym.pi/2), -sym.sin(theta[13] + sym.pi/2), 0, 45.5e-3], 
                [sym.sin(theta[13] + sym.pi/2)*sym.cos(0), sym.cos(theta[13] + sym.pi/2)*sym.cos(0), -sym.sin(0), -0*sym.sin(0)], 
                [sym.sin(theta[13] + sym.pi/2)*sym.sin(0), sym.cos(theta[13] + sym.pi/2)*sym.sin(0), sym.cos(0), 0*sym.cos(0)],
                [0, 0, 0, 1]])

### Right Index End 
TRIE = sym.Matrix([[sym.cos(0), -sym.sin(0), 0, 5e-3], 
                [sym.sin(0)*sym.cos(sym.pi/2), sym.cos(0)*sym.cos(sym.pi/2), -sym.sin(sym.pi/2), -25e-3*sym.sin(sym.pi/2)], 
                [sym.sin(0)*sym.sin(sym.pi/2), sym.cos(0)*sym.sin(sym.pi/2), sym.cos(sym.pi/2), 25e-3*sym.cos(sym.pi/2)],
                [0, 0, 0, 1]])

### Middle 
### Right Middle J0
TRM0 = sym.Matrix([[sym.cos(sym.pi/2), -sym.sin(sym.pi/2), 0, -161.5e-3], 
                [sym.sin(sym.pi/2)*sym.cos(0), sym.cos(sym.pi/2)*sym.cos(0), -sym.sin(0), -17.5e-3*sym.sin(0)], 
                [sym.sin(sym.pi/2)*sym.sin(0), sym.cos(sym.pi/2)*sym.sin(0), sym.cos(0), 17.5e-3*sym.cos(0)],
                [0, 0, 0, 1]])

### Right Middle J1
TRM1 = sym.Matrix([[sym.cos(theta[14] + sym.pi/2), -sym.sin(theta[14] + sym.pi/2), 0, -18.3e-3], 
                [sym.sin(theta[14] + sym.pi/2)*sym.cos(sym.pi), sym.cos(theta[14] + sym.pi/2)*sym.cos(sym.pi), -sym.sin(sym.pi), -0*sym.sin(sym.pi)], 
                [sym.sin(theta[14] + sym.pi/2)*sym.sin(sym.pi), sym.cos(theta[14] + sym.pi/2)*sym.sin(sym.pi), sym.cos(sym.pi), 0*sym.cos(sym.pi)],
                [0, 0, 0, 1]])

### Right Middle J2
TRM2 = sym.Matrix([[sym.cos(theta[15] - sym.pi/2), -sym.sin(theta[15] - sym.pi/2), 0, -20e-3], 
                [sym.sin(theta[15] - sym.pi/2)*sym.cos(sym.pi/2), sym.cos(theta[15] - sym.pi/2)*sym.cos(sym.pi/2), -sym.sin(sym.pi/2), -0*sym.sin(sym.pi/2)], 
                [sym.sin(theta[15] - sym.pi/2)*sym.sin(sym.pi/2), sym.cos(theta[15] - sym.pi/2)*sym.sin(sym.pi/2), sym.cos(sym.pi/2), 0*sym.cos(sym.pi/2)],
                [0, 0, 0, 1]])

### Right Middle J2'
TRM2P = sym.Matrix([[sym.cos(-sym.pi/2), -sym.sin(-sym.pi/2), 0, 3e-3], 
                [sym.sin(-sym.pi/2)*sym.cos(0), sym.cos(-sym.pi/2)*sym.cos(0), -sym.sin(0), -0*sym.sin(0)], 
                [sym.sin(-sym.pi/2)*sym.sin(0), sym.cos(-sym.pi/2)*sym.sin(0), sym.cos(0), 0*sym.cos(0)],
                [0, 0, 0, 1]])

### Right Middle J3
TRM3 = sym.Matrix([[sym.cos(theta[16] + sym.pi/2), -sym.sin(theta[16] + sym.pi/2), 0, 50.5e-3], 
                [sym.sin(theta[16] + sym.pi/2)*sym.cos(0), sym.cos(theta[16] + sym.pi/2)*sym.cos(0), -sym.sin(0), -0*sym.sin(0)], 
                [sym.sin(theta[16] + sym.pi/2)*sym.sin(0), sym.cos(theta[16] + sym.pi/2)*sym.sin(0), sym.cos(0), 0*sym.cos(0)],
                [0, 0, 0, 1]])

### Right Middle End 
TRME = sym.Matrix([[sym.cos(0), -sym.sin(0), 0, 5e-3], 
                [sym.sin(0)*sym.cos(sym.pi/2), sym.cos(0)*sym.cos(sym.pi/2), -sym.sin(sym.pi/2), -25e-3*sym.sin(sym.pi/2)], 
                [sym.sin(0)*sym.sin(sym.pi/2), sym.cos(0)*sym.sin(sym.pi/2), sym.cos(sym.pi/2), 25e-3*sym.cos(sym.pi/2)],
                [0, 0, 0, 1]])

### Total transform matrix
# Fingers
TRT = TRA0*TRA1*TRA2*TRA3*TRA4*TRA5*TRA6*TRA7*TRAE*TRT0*TRT1*TRT2*TRT3*TRT3P*TRT4*TRTE
TRI = TRA0*TRA1*TRA2*TRA3*TRA4*TRA5*TRA6*TRA7*TRAE*TRI0*TRI1*TRI2*TRI2P*TRI3*TRIE
TRM = TRA0*TRA1*TRA2*TRA3*TRA4*TRA5*TRA6*TRA7*TRAE*TRM0*TRM1*TRM2*TRM2P*TRM3*TRME
# Elbow
TREL = TRA0*TRA1*TRA2*TRA3*TRA4
# Wrist
TRWR = TRA0*TRA1*TRA2*TRA3*TRA4*TRA5*TRA6

f = open('jacob_wrist.txt','w')
for i in range(3):
    for j in range(17):
        f.write('J{}{} = '.format(i, j))
        f.write('{}'.format(sym.diff(TRWR[i,3], theta[j])))
        f.write('\n')
f.close()