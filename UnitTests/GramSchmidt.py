import numpy as np
from MGCMTProcessor import MGCMTProcessor

processor = MGCMTProcessor()


# VOORBEELD VAN VERSCHIL CGS EN MGS
########################################################################################
machine_eps = np.finfo(float).eps
v1 = np.array([1, machine_eps, machine_eps])
v2 = np.array([1, machine_eps, 0])
v3 = np.array([1, 0, machine_eps])

A = np.column_stack((v1, v2, v3))
print "START VECTORS"
print A

print "START INPRODUCTS:"
print "<1,2> ", np.inner(v1, v2)
print "<1,3> ", np.inner(v1, v3)
print "<2,3> ", np.inner(v2, v3)



O = processor.gramschmidt(A, modified=0)
#print "START VECTORS AFTER CLASSIC GRAM SCHMIDT"
#print A

print "ORTHONORMAL VECTORS CLASSIC GRAM SCHMIDT"
print O

u1 = O[:, 0]  # [1 2.2e-16 2.2e-16]
u2 = O[:, 1]  # [0 0 -1]
u3 = O[:, 2]  # [0 -0.7 -0.7]

print "INNER PRODUCTS ORTHO VECTORS CLASSIC GRAM SCHMIDT"
print np.inner(u1, u2)  # -2.2e-16
print np.inner(u1, u3)  # -3.1e-16
print np.inner(u2, u3)  # 0.707

print "NORMS ORTHO VECTORS CLASSIC GRAM SCHMIDT"
print np.inner(u1, u1)  # 1.0
print np.inner(u2, u2)  # 1.0
print np.inner(u3, u3)  # 1.0


O1 = processor.gramschmidt(A, modified=1)
#print "START VECTORS AFTER MODIFIED GRAM SCHMIDT"
#print A

print "ORTHONORMAL VECTORS MODIFIED GRAM SCHMIDT"
print O1

u11 = O1[:, 0]  # [1 2.2e-16 2.2e-16]
u22 = O1[:, 1]  # [0 0 -1]
u33 = O1[:, 2]  # [0 -1 0]


print "INNER PRODUCT ORTHO VECTORS MODIFIED GRAM SCHMIDT"
print np.inner(u11, u22)  # -2.2e-16
print np.inner(u11, u33)  # -2.2e-16
print np.inner(u22, u33)  # 0.0

print "NORMS ORTH VECTORS MODIFIED GRAM SCHMIDT"
print np.inner(u11, u11)  # 1.0
print np.inner(u22, u22)  # 1.0
print np.inner(u33, u33)  # 1.0
################################################################################


# VOORBEELD VAN GELIJKHEID CGS EN MGS
########################################################################################
v1 = np.array([1, 2, 5])
v2 = np.array([1, 1, 1])
v3 = np.array([1, 0, 3])

A = np.column_stack((v1, v2, v3))
print "START VECTORS"
print A

print "START INPRODUCTS:"
print "<1,2> ", np.inner(v1, v2)
print "<1,3> ", np.inner(v1, v3)
print "<2,3> ", np.inner(v2, v3)


O = processor.gramschmidt(A, modified=0)
#print "START VECTORS AFTER CLASSIC GRAM SCHMIDT"
#print A

print "ORTHONORMAL VECTORS CLASSIC GRAM SCHMIDT"
print O

u1 = O[:, 0]  # [0.18 0.36 0.91]
u2 = O[:, 1]  # [0.78 0.50 -0.35]
u3 = O[:, 2]  # [0.58 -0.78 0.19]

print "INNER PRODUCTS ORTHO VECTORS CLASSIC GRAM SCHMIDT"
print np.inner(u1, u2)  # 0.0
print np.inner(u1, u3)  # 1.9e-16
print np.inner(u2, u3)  # -2.2e-16

print "NORMS ORTHO VECTORS CLASSIC GRAM SCHMIDT"
print np.inner(u1, u1)  # 1.0
print np.inner(u2, u2)  # 1.0
print np.inner(u3, u3)  # 1.0


O1 = processor.gramschmidt(A, modified=1)
#print "START VECTORS AFTER MODIFIED GRAM SCHMIDT"
#print A

print "ORTHONORMAL VECTORS MODIFIED GRAM SCHMIDT"
print O1

u11 = O1[:, 0]  # [0.18 0.36 0.91]
u22 = O1[:, 1]  # [0.78 0.50 -0.35]
u33 = O1[:, 2]  # [0.58 -0.78 0.19]


print "INNER PRODUCT ORTHO VECTORS MODIFIED GRAM SCHMIDT"
print np.inner(u11, u22)  # 1.1e-16
print np.inner(u11, u33)  # 2.4e-16
print np.inner(u22, u33)  # -9e-17

print "NORMS ORTH VECTORS MODIFIED GRAM SCHMIDT"
print np.inner(u11, u11)  # 1.0
print np.inner(u22, u22)  # 1.0
print np.inner(u33, u33)  # 1.0
################################################################################

