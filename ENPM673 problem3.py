import numpy as np
import math

x1 = 5
x2 = 150
x3 = 150
x4 = 5
y1 = 5
y2 = 5
y3 = 150
y4 = 150
xp1 = 100
xp2 = 200
xp3 = 220
xp4 = 100
yp1 = 100
yp2 = 80
yp3 = 80
yp4 = 200
A = np.matrix([
                [-x1, -y1, -1, 0, 0, 0, x1*xp1, y1*xp1, xp1],
                [0, 0, 0, -x1, -y1, -1, x1*yp1, y1*yp1, yp1],
                [-x2, -y2, -1, 0, 0, 0, x2*xp2, y2*xp2, xp2],
                [0, 0, 0, -x2, -y2, -1, x2*yp2, y2*yp2, yp2],
                [-x3, -y3, -1, 0, 0, 0, x3*xp3, y3*yp3, yp3],
                [0, 0, 0, -x3, -y3, -1, x3*yp3, y3*yp3, yp3],
                [-x4, -y4, -1, 0, 0, 0, x4*xp4, y4*xp4, xp4],
                [0, 0, 0, -x4, -y4, -1, x4*yp4, y4*yp4, yp4]
             ])
# Printing the original matrix A
print("The entered matrix is : ")
print(A)

# AT is the transpose of matrix A, using the function matrix.transpose()
AT = A.transpose()
print("\n \n A transpose is : ")
print(AT)

# AAT is the multiplication of matrix A and AT, used the function numpy.matmul(matrix 1, matrix 2)
AAT = np.matmul(A, AT)
print("\n \n A*AT is : ")
print(AAT)

# ATA  is the multiplication of matrix AT and A, used the function numpy.matmul(matrix 1, matrix 2)
ATA = np.matmul(AT, A)
print("\n \n AT*A is : ")
print(ATA)

# Matrix U gives the eigen values for AAT and matrix S gives the eigen vectors for matrix AAT using function numpy.linalg.eig(matrix)
U, S = np.linalg.eig(AAT)
print("\n \n")
print("Orthogonal eigen values of AAT :")
print(U)
print("\n \n Orthogonal eigen vectors of AAT which is the U matrix of SVD :")
print(S)

# Matrix U_sqr gives the ∑ matrix of SVD which is calculated using numpy.sqrt(matrix)
U_sqr = np.sqrt(U)
print("\n \n The S matrix which is the ∑ matrix of SVD is : ")
print(U_sqr)

# Matrix V gives the eigen vectors for ATA and matrix Y gives the eigen values for matrix ATA using function numpy.linalg.eig(matrix)
V, Y = np.linalg.eig(ATA)
VT = V.transpose()
print("\n \n Orthogonal eigen values of ATA :")
print(VT)
print("\n \n Orthogonal eigen vectors of ATA :")
print(Y)

#Matrix Y_trans gives the transpose of Y matrix which is the desired VT matrix of SVD
print("\n \n Transpose of eigen vectors of ATA is the VT matrix of SVD which is : ")
Y_trans = Y.transpose()
print(Y_trans)

# Calculation using SVD function directly
u, s, vt = np.linalg.svd(A, full_matrices=True)

#print("Orthogonal eigen vectors of AAT :")
print("\n \n The Matrix U is : ")
print(u)

#print("Orthogonal eigen values of AAT :")
print("\n \n The Matrix S is : ")
print(s)

#print("Orthogonal eigen vectors of ATA :")
print("\n \n The Matrix VT is : ")

#vt_trans = vt.transpose()
print(vt)