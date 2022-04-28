import numpy as np
import seaborn as sea
sea.set_style("whitegrid")

from random import uniform
import scipy
from scipy.spatial import ConvexHull
import alphashape
import shapely.geometry as geom


def get_random_points(num_points, xspan, yspan):
	points = [[uniform(*xspan), uniform(*yspan)]  for i in range(num_points)]
	radius_x = (xspan[1] - xspan[0])/2.0
	radius_y = (yspan[1] - yspan[0])/2.0
	center_x = (xspan[1] + xspan[0])/2.0
	center_y = (yspan[1] + yspan[0])/2.0
	points = [[x,y] for x,y in points if np.sqrt(((x-center_x)/radius_x) ** 2 + ((y-center_y)/radius_y) ** 2) < 1.0]
	points = np.array(points)
	return points


def is_p_inside_points_hull(hull, p):
	new_points = np.append(hull.points, p, axis=0)
	new_hull = ConvexHull(new_points)
	if list(hull.vertices) == list(new_hull.vertices):
		return True
	else:
		return False


def hull2array(hull, array, xspan, yspan):
	x_size = array.shape[0]
	y_size = array.shape[1]
	for i in range(x_size):
		for j in range(y_size):
			x = xspan[0] + i*float(xspan[1]-xspan[0])/float(x_size)
			y = yspan[0] + j*float(yspan[1]-yspan[0])/float(y_size)
			inside = False
			if isinstance(hull, ConvexHull):
				inside = is_p_inside_points_hull(hull, np.array([[x,y]]))
			elif isinstance(hull, geom.Polygon):
				inside = geom.Point(x,y).within(hull)
			if inside:
				array[i, j] = 1.0
			else:
				array[i, j] = 0.0
	return array


class Protein:

	def __init__(self, bulk, boundary=None, hull=None):
		self.size = bulk.shape[0]
		self.bulk = bulk
		self.boundary = boundary
		self.hull = hull
	
	@classmethod
	def generateConvex(cls, size=50, num_points = 15, occupancy=0.8):
		grid_coordinate_span = (0, size)
		points_coordinate_span = (0.5*(1.0-occupancy)*size, size - 0.5*(1.0-occupancy)*size)
		points = get_random_points(num_points, points_coordinate_span, points_coordinate_span)
		hull = scipy.spatial.ConvexHull(points)		
		bulk = hull2array(hull, np.zeros((size, size)), grid_coordinate_span, grid_coordinate_span)
		return cls(bulk)

	@classmethod
	def generateConcave(cls, size=50, alpha=1.0, num_points = 25, occupancy=0.8):
		grid_coordinate_span = (0, size)
		points_coordinate_span = (0.5*(1.0-occupancy)*size, size - 0.5*(1.0-occupancy)*size)
		points = get_random_points(num_points, points_coordinate_span, points_coordinate_span)
		optimal = alpha * alphashape.optimizealpha(points)
		hull = alphashape.alphashape(points, optimal)
		bulk = hull2array(hull, np.zeros((size, size)), grid_coordinate_span, grid_coordinate_span)
		return cls(bulk, hull=hull)
