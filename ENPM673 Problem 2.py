import numpy as np
import matplotlib.pyplot as plt
import random

# Pulling data from Data sets
datafile_1 = np.genfromtxt('data_1.csv', delimiter=',')
# print(datafile_1)
datafile_2 = np.genfromtxt('data_2.csv', delimiter=',')
# print(datafile_2)

datafile_1 = datafile_1[1:]
datafile_2 = datafile_2[1:]

# Least square method for Data Set 1
# Fitting a parabola for y=Ax+B
def fit_parabola(points):
	n_points = len(points)
	# calculating matrix A and B for least squares
	A = []
	B = []
	for point in points:
		x = point[0]
		y = point[1]
		A.append([x**2,x,1])
		B.append([y])

# Array A, B
	A = np.array(A)
	B = np.array(B)
	# print(A)
	# print(B)

# Matrix multiplication for getting solution (sol) = (ATA)^-1.AT.B
	sol = np.matmul(np.transpose(A),A)
	# print(sol)
	sol = np.linalg.inv(sol)
	# print(sol)
	sol = np.matmul(sol,np.transpose(A))
	# print(sol)
	sol = np.matmul(sol,B)

	return sol

plt.scatter(datafile_1[:,0],datafile_1[:,1])
# print(datafile_1[:,0])

datafile_1_list = datafile_1.tolist()

least_square_sol = fit_parabola(datafile_1_list)
# print(least_square_sol)
a = least_square_sol[0]
b = least_square_sol[1]
c = least_square_sol[2]
x = np.linspace(0,500,1000)
# print(x)
y = a*(x**2)+b*x+c
# print(y)

#plotting the parabola
plt.plot(x,y)
plt.title('Least Square Method')
plt.show()


#RANSAC Method for Data Set 2
datafile_2_list = datafile_2.tolist()

dist_thresh = 70        # Distance threshold for line fundamental matrix, we considered this by looking at the data set
thresh_prob = 0.8       # desired probability that we get a good sample
plt.scatter(datafile_2[:,0],datafile_2[:,1])

sol_found = False

for n_interations in range(150):
	#choosing random 3 points
	random_points = []
	random_indices = random.sample(range(len(datafile_2_list)), 3)
	# printing (random_indices)
	for random_index in random_indices:
		random_points.append(datafile_2_list[random_index])

	# print(random_points)

	random_points_sol = fit_parabola(random_points)

	a = random_points_sol[0]
	b = random_points_sol[1]
	c = random_points_sol[2]

	# fit a parabola using these fit_parabola(points)

	inliers = []
	# finding distance of all the points from parbola and append to inliers
	for point in datafile_2_list:
		x_curr_point = point[0]
		y_curr_point = point[1]

		# find distsance of point from parabola
		dist = abs(y_curr_point- (a*(x_curr_point**2)+b*x_curr_point+c))
		# print(dist)
        # if distance is less than the threshold that will be an inlier
		if dist<dist_thresh:
			inliers.append([x_curr_point,y_curr_point])
    # If the probability to find the best sample is more than our considered threshold probability then we have the best solution
	if((float(len(inliers))/len(datafile_2_list))>thresh_prob):
		sol_found = True
		break

# plotting parabola
if(sol_found==True):
	inliers_arr = np.array(inliers)
	plt.scatter(inliers_arr[:,0],inliers_arr[:,1],color="r")
	least_square_sol = fit_parabola(inliers)
	# print(least_square_sol)
	a = least_square_sol[0]
	b = least_square_sol[1]
	c = least_square_sol[2]
	x = np.linspace(0,500,1000)
	y = a*(x**2)+b*x+c

	plt.title('RANSAC Method')
	plt.plot(x,y)
	plt.show()

else:
	print('I counld not find a solution')