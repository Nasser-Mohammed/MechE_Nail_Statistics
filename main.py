# import the necessary packages
import copy
import os
import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt
from csv import DictWriter
import glob
import multiprocessing as mp
from filelock import FileLock
from timeit import default_timer as timer
from math import cos, sin, pi


#def is_contour_bad(): function that determines if a contour is roughly a rectangle, still optimizing this though. This
#is used for the final step of finding the benchmarks in the nail
def is_contour_rect(c):
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, .01*peri, True)
	if len(approx) == 4:
		x, y, w, h = cv2.boundingRect(c)
		return True
	else:
		return False

#def write(): helper function so I don't have to remember the syntax to save an image
def write(name, img):
	cv2.imwrite(name+".png", img)

#def plot(): helper function to plot images quickly for testing
def plot(img):
	cv2.imshow("image", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

#def to_csv(): takes our data to write to the csv and write it
def to_csv(data, csv_path):
	columns = ["File_Name", "Distance", "Width_Nail", "Height_of_Benchmark"]
	hashmap = {"File_Name": data[0], "Distance": data[1], "Width_Nail": data[2], "Height_of_Benchmark": data[3]}
	with open(csv_path, "a") as f:
		writer = DictWriter(f, fieldnames=columns)
		writer.writerow(hashmap)
		f.close()


#def cut_half(): cuts the image into two, so we can focus on the top box and bottom box alone
def cut_half(img):
	height, width = img.shape
	img2 = copy.deepcopy(img)
	cropped_up = img2[0: int(height/3.5), :]

	yInit = int(height/3.5)

	for i, x in enumerate(img):
		if i >= yInit:
			break
		for b, y in enumerate(x):
			img[i,b] = 255

	cropped_bottom = img
	return cropped_up, cropped_bottom


#def keep_rectangle(): function that makes every pixel outside of the parameter "rectangle", white. So we can focus on data within the rectangle
def keep_rectangle(image, rectangle):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	xInit = rectangle[0]
	yInit = rectangle[1]
	xFinal = rectangle[2] + xInit
	yFinal = rectangle[3] + yInit
	print(f"KEEPING RECTANGLE....")

	for i in range(len(gray[0])):
		for j in range(len(gray[1])):
			if j > xInit and j < xFinal and i > yInit and i < yFinal:
					continue
			gray[i, j] = 255

	return gray

#def line_score(): function that can help find the straight lines given a contour that has line segments but also curvature
def line_score(c):		#c is a list of points making up a large contour, but it has curves we want to ignore

	lines = []
	sigma = 10			#parameter subject to change, it is just the variance we allow between pixels to determine if they're roughly on line
	line_distance = 75		#parameter, this just defines how many points we need in a row to be considered a line
	values = []
	offset = 0
	index_tmp = 0

	for index, point in enumerate(c):

		line_segment = []
		if point[0] >= 1350 or point[0] <= 150:
			continue
		for pos in range(line_distance):

			if index + pos >= len(c):
				return lines


			if abs(point[1] -c[index+pos, 1]) <= sigma:
				line_segment.append(c[index+pos])
				continue
			else:
				break
		if len(line_segment) >= 30:
			lines.append(line_segment)

	return lines

#def benchmark(): if a 'found' benchmark is way off, it it likely our program picked up something that is not the benchmark
def benchmark_validity(contours, image):
	rectangles = []
	for i, z in enumerate(contours):
		x,y,w,h = cv2.boundingRect(z)
		rectangles.append((x,y,w,h))
		#cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 3)
		#plot(image)
	#contours = sorted(rectangles, key = lambda y: y[0], reverse=True)

	good_rect = []
	for a in rectangles:
		if a[1] >= 175:
			#cv2.rectangle(image, (a[0],a[1]), (a[0]+a[2], a[1]+a[3]), (0,255,0), 3)
			good_rect.append(a)
		
	if len(good_rect) <= 3 and len(good_rect) >=1:
		return good_rect[0]

	elif len(good_rect) < 1:
		return (0,0,0,0)
	differences = []
	offset = 1
	tracker = 1 
	current_min = 1000000
	index = (0,0)
	for i in range(len(rectangles)-1):
		#tracker = 1
		for j in range(len(rectangles)): 
			if tracker + j >= len(rectangles):
				break
			if abs(rectangles[i][0] - rectangles[tracker+j][0]) < current_min:
				current_min = abs(rectangles[i][0] - rectangles[tracker+j][0])
				index = (i, tracker+j)
			#differences.append(((i,j+tracker), abs(rectangles[i][0] - rectangles[j+tracker][0])))
		tracker += 1
		#differences.append(abs(rectangles[i+1][1][0]-rectangles[i][1][0]))
	print(f"FOUND SMALLEST DIFFERENCE FOUND BETWEEN {index[0]} AND {index[1]} difference of {current_min}")

	if rectangles[index[0]][1] + rectangles[index[0]][3] <= 400:
		return [contours[index[1]]]
	elif rectangles[index[1]][1] + rectangles[index[1]][3] <= 400:
		return [contours[index[0]]]
	else:
		good_benchmarks = [contours[index[0]], contours[index[1]]]
		return good_benchmarks

	
#def benchmark(): function that finds the benchmarks within the nail for measurement, and returns it's coordiinates
def benchmark(img, original):

	kernel = np.ones((5,5), np.uint8)
	blur = cv2.GaussianBlur(img, (3,3), 0)

	thresh = cv2.threshold(blur, 75, 255, cv2.THRESH_BINARY)[1]


	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21,21))
	opening = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations = 3)
	cnts= cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	#cv2.drawContours(original, cnts, -1, (0,255,0), 3)
	good_cnts = []
	for i, c in enumerate(cnts):
		if cv2.contourArea(c) >= 35000 or cv2.contourArea(c) <= 2000:
			#print(f"bad area is: {cv2.contourArea(c)}")
			continue
		else:
			#print(f"good area is: {cv2.contourArea(c)}")
			good_cnts.append(c)
	cnts = []

	print(f"DETECTED {len(good_cnts)} POTENTIAL BENCHMARKS")
	if len(good_cnts) == 3:

		x,y,w,h = benchmark_validity(good_cnts, original)
		#cnts = sorted(new_cnts, key = cv2.contourArea)[0:1]
		#x,y,w,h = cv2.boundingRect(cnts[0])
	elif len(good_cnts) >= 1 and len(good_cnts) < 3:
		cnts = sorted(good_cnts, key=cv2.contourArea, reverse = True)[:1]
		x,y,w,h = cv2.boundingRect(cnts[0])
	elif len(good_cnts) > 3 and len(good_cnts) <=6:
		x,y,w,h = benchmark_validity(good_cnts, original)
		#cnts = sorted(good_cnts, key = cv2.contourArea, reverse = True)[0:1]
		#x,y,w,h = cv2.boundingRect(cnts[0])
	else:
		x,y,w,h = (0,0,0,0)

	return x,y,w,h
	good_contours=[]
	for x in cnts:
		if cv2.contourArea(x) < 25000:
			good_contours.append(x)
	final_benchmark = good_contours[0]
	x,y,w,h = cv2.boundingRect(final_benchmark)
	cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), 3) #this just draws the rectangle on the image
	plot(original)
	return x,y,w,h


#def fill_rectangle(): function that fills every pixel within a rectangle with white pixels (255), so that we can focus on contours outside
#of the rectangle
def fill_rectangle(image, rectangle):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	xInit = rectangle[0]
	yInit = rectangle[1]
	xFinal = rectangle[2] + xInit
	yFinal = rectangle[3] + yInit
	print(f"FILLLING RECTANGLE.......")
	for i, x in enumerate(gray):
		if i < yInit or i > yFinal:
			continue
		for b, y in enumerate(x):
			if b < xInit or b > xFinal:
				continue
			gray[i,b] = 255
	return gray



	# for theta in range(361):
	# 	values.append((int(line_distance*cos(theta*pi/180)),int(line_distance*sin(theta*pi/180))))

	# for index, point in enumerate(c):
		#if offset + index_tmp != index:
			#print(f"current index is: {index} target is {index_tmp + offset}")
			#continue
		#cords = []
		#point = point[0] 	#this just gets the actual point because it is in the form [[x,y]] so we make it [x,y]
		#line_segment = [] 	#temporary line segment, we only add if it meets our length requirement
		#index_tmp = 0 		#reset this variable, it it basically how far along the contour list we made it




		#print(f"{point[1]}, {values[5][1]}")
		# cords = [(x[0]+point[0], point[1] - x[1]) for x in values if point[0] + x[0] < 2048 and point[0] + x[0] >= 0 and point[1] - x[1] < 2048 and point[1] -x[1] >= 0]
		# cords = [*set(cords)]
		# print(f"FOUND {len(cords)} POTENTIAL LINES AT POINT ({point[0]}, {point[1]})")
		# offset = 1
		# for i, pos in enumerate(cords):
		# 	m = 0.0
		# 	m = pos[1]/pos[0]
		# 	if offset+index >= len(c):
		# 		return lines
		# 	if offset >= 100:
		# 		index_tmp = index
		# 		break
		# 	if m == 0.0:
		# 		print(f"slope of: {m} causes error")
		# 		offset += 1
		# 		continue
		# 	# print(f"line has slope of: {m}")
		# 	# print(f"checking contour: {c[offset+index, 0]}")
		# 	#print(f"point on radial line related to our countor: x is: {int(c[offset+index, 0, 1]/m)}, y is: {int(m*c[offset+index, 0, 0])}")
		# 	# if c[index+offset, 0, 1]/m < 0.0005 or c[index+offset, 0, 0]*m < 0.0005:
		# 	# 	offset += 1
		# 	# 	continue
		# 	dx = c[offset+index, 0, 1]/m
		# 	dy = c[offset+index, 0, 0]*m
		# 	if dy == 0 or dx == 0:
		# 		offset += 1
		# 		continue
		# 	x_diff = abs(pos[0] -int(dx))
		# 	y_diff = abs(pos[1] - int(dy))
		# 	# if dx > 0.0001:
		# 	# 	x_diff = abs(pos[0]-int(dx))
		# 	# else:
		# 	# 	offset += 1
		# 	# 	continue
		# 	# if dy > 0.0001:
		# 	# 	y_diff = abs(pos[1] - int(dy))
		# 	# else:
		# 	# 	offset += 1
		# 	# 	continue

		# 	if x_diff <= sigma and y_diff <= sigma:
		# 		line_segment.append(c[offset+index, 0])
		# 	offset += 1
		# index_tmp = index
		# if len(line_segment) >= 5:
		# 	lines.append(line_segment)

	return lines
#def find_nail(): first step in our program, finds the contours inside the nail, and then returns the minimum enclosing rectangle of that nail
def height_of_tool(image, sigma=.33):
	imgCopy = copy.deepcopy(image)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (3,3), 0)
	median = np.median(blur)
	lower = int(max(0, (1.0-sigma)*median))
	upper = int(min(255, (1.0+sigma)*median))
	#print(f"threshold parameters for this image edge detection are: upper({upper}), lower({lower})")
	#edged = cv2.Canny(blur, lower, upper)
	#top, bottom = cut_half(blur)
	#edged = cv2.Canny(blur, 25, 255)
	edged = cv2.Canny(blur, 30, 225)

	full_cnts = imutils.grab_contours(cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE))

	top_img, bottom_img = cut_half(blur)

	#thresh_top = cv2.threshold(top_img, 45, 255, cv2.THRESH_BINARY)[1]
	#thresh_bottom = cv2.threshold(bottom_img, 45, 255, cv2.THRESH_BINARY)[1]

	thresh_top = cv2.Canny(top_img, 30, 255)
	thresh_bottom = cv2.Canny(bottom_img, 30, 255)


	top_cnts = cv2.findContours(thresh_top, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	top_cnts = imutils.grab_contours(top_cnts)
	top_cnts = sorted(top_cnts, key = cv2.contourArea, reverse = True)#[:1]
	temp = copy.deepcopy(image)

	array = top_cnts[0]
	array = array[:, 0, :]
	print(f"original size: {array.shape}")

	#top_cnts = [np.delete(array, np.where((array[:, 1] >= 300) | (array[:, 0] >= 1600) | (array[:, 0] <= 435)), axis = 0)]


	bottom_cnts = cv2.findContours(thresh_bottom, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	bottom_cnts = imutils.grab_contours(bottom_cnts)
	bottom_cnts = sorted(bottom_cnts, key = cv2.contourArea, reverse = True)#[:1]
	array = bottom_cnts[0]
	array = array[:, 0, :]
	#bottom_cnts = [np.delete(array, np.where((array[:, 0] >= 1600) | (array[:, 0] <= 150)), axis = 0)]
	#cv2.drawContours(temp, top_cnts, -1, (0,0,255), 3)

	# lines_top = line_score(top_cnts[0])
	# print(f"program detected {len(lines_top)} line segments of top image")

	# lines_bottom = line_score(bottom_cnts[0])
	# print(f"program detected {len(lines_bottom)} line segments of bottom image")


	# if len(lines_top) <= 0:
	# 	return ((0,0,0,0), (0,0,0,0))
	# if len(lines_bottom) <= 0:
	# 	return ((0,0,0,0), (0,0,0,0))
	# lines_top = np.concatenate(lines_top)#.tolist()
	# lines_bottom = np.concatenate(lines_bottom)#.tolist()

	#cv2.drawContours(image, [lines_top], -1, (0,0,255), 3)
	#plot(image)
	# for x in lines_top:
	# 	cords = np.stack(x, axis =0)
	# 	cv2.drawContours(image, [cords], -1,(0,255,0), 3)

	# for y in lines_bottom:
	# 	cords = np.stack(y, axis=0)
	# 	cv2.drawContours(image, [cords], -1, (255, 0, 0), 3)

	#cv2.drawContours(image, tmp, -1, (0,255, 0), 5)
	x1,y1, w1, h1 = cv2.boundingRect(top_cnts[0])
	x2, y2, w2, h2 = cv2.boundingRect(bottom_cnts[0])

	x3, y3, w3, h3 = cv2.boundingRect(full_cnts[0])

	#cv2.rectangle(image, (x1,y1), (x1+w1, y1+h1), (0,255,0), 3)
	#cv2.rectangle(image, (x2,y2), (x2+w2, y2+h2), (255,0,0), 3)
	cv2.rectangle(image, (x3, y3), (x3+w3, y3+h3), (0, 0, 255), 3)
	plot(image)
	exit()
	return ((x1, y1, w1, h1), (x2, y2, w2, h2))



def calculate(path):
	image = cv2.imread(path)
	#change line below for image format
	if not path.endswith(".jpg"):
		return "UNRECOGNIZED FORMAT"
	top_rect, bottom_rect = height_of_tool(image)
	if top_rect == (0,0,0,0) or bottom_rect == (0,0,0,0):
		return image #(path[11:], 0, 0, 0)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	x,y,w,h = benchmark(gray, image)
	cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 3)
	width = w
	benchmark_height = h
	distance = bottom_rect[1] - top_rect[1] + top_rect[3]

	cv2.line(image, (645, top_rect[1]+top_rect[3]), (645, bottom_rect[1]), (0, 0, 255), 5)
	cv2.line(image, (x, y+150), (x+w, y+150), (0, 255, 0), 3)
	cv2.line(image, (x+int((w/2)), y), (x+int(w/2), y+h), (0,0,255), 3)
	image = cv2.putText(image, "Distance in pixels: " + str(distance), (100, top_rect[1]+top_rect[1]+75), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0,0, 255), 3)
	image = cv2.putText(image, "Width of Nail in pixels: " + str(width), (x+w+150, y+100), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (100,255, 100), 3)
	image = cv2.putText(image, "Height of benchmark in pixels: " + str(benchmark_height), (x+w+150, y+350), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (50, 25, 255), 3)
	plot(image)
	return image
	#return (path[11:], distance, width, benchmark_height)


def sorter(data):
	if data[0].isnumeric():
		return data[0].toInt()

def main():

	print(f"STARTING EXECUTION......")
	start = timer()
	path = "./ImagesNewCam"
	files = os.listdir(path)
	arg = [path+"/"+x for x in files if os.path.isfile(path+"/"+x)]
	num_of_files = len(os.listdir(path))
	print(f"number of files to process = {num_of_files}")
	if num_of_files < 20:
		processes = 2
	else: 
		processes = int(num_of_files/10)
	pool = mp.Pool(processes = processes)
	count = 0

	#frame_size = (2048, 2048)
	#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	#out = cv2.VideoWriter("output_video2.mp4", fourcc, 15, frame_size, isColor = True)
	for images in glob.iglob(f"{path}/*"):
		if count >= 5:
			break
		image = calculate(images)
		#out.write(image)
		#to_csv(image, "./outputData.csv")
	# results = pool.map_async(calculate, arg,  chunksize = int(len(arg)/processes))

	dataList = []
	#for index, data in enumerate(results.get()):
		#dataList.append(data)
		#print(f"data sample: {index}, x: {data[0]}, y: {data[1]}")
	#sorted_data = sort(dataList, key = sorter)
	final_time = timer()
	print(f"Program took {final_time-start} seconds to process images")

if __name__ == "__main__":
	main()
