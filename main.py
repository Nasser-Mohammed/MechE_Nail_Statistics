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
def benchmark(img, original, nail_cords):

	blur = cv2.GaussianBlur(img, (3, 3), 0)
	thresh = cv2.threshold(blur, 45, 255, cv2.THRESH_BINARY_INV)[1]
	benchmark_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1)) #(50,1)
	detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, benchmark_kernel, iterations = 2)

	benchmark_kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
	second_transformation = cv2.morphologyEx(detect_horizontal, cv2.MORPH_OPEN, benchmark_kernel2, iterations=3)
	dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
	final_dilation = cv2.morphologyEx(second_transformation, cv2.MORPH_DILATE, dilation_kernel, iterations = 1)

	benchmark_cnts = imutils.grab_contours(cv2.findContours(final_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))
	benchmark_cnts = sorted(benchmark_cnts, key = cv2.contourArea, reverse = True)[1:]
	#cv2.drawContours(img, benchmark_cnts, -1, (0,255,0), 3)
	if len(benchmark_cnts) <= 1:
		retry_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
		retry_benchmark = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, retry_kernel, iterations = 2)
		retry_kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (20,20))
		retry_benchmark2 = cv2.morphologyEx(retry_benchmark, cv2.MORPH_DILATE, retry_kernel2, iterations = 2)
		retry_cnts = imutils.grab_contours(cv2.findContours(retry_benchmark2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))

		if len(retry_cnts) == 0:
			return (0,0,0,0), (0,0,0,0)
		retry_cnts = sorted(retry_cnts, key = cv2.contourArea, reverse = True)
		new_cnts = []
		#cv2.drawContours(original, retry_cnts, -1, (0,255,0), 3)
		#plot(original)
		place = 0
		for c in retry_cnts:
			temp = copy.deepcopy(c)
			temp = np.concatenate(temp, axis = 0)
			temp = temp[(temp[:, 0] >= nail_cords[0] - 15) & (temp[:, 0] <= nail_cords[0] + nail_cords[2] + 15)]
			if cv2.boundingRect(temp) != (0,0,0,0):
				new_cnts.append(c)
			#cv2.drawContours(original, [temp], -1, (0,255,0), 3)
			#plot(original)
		rect_list = [cv2.boundingRect(cnt) for cnt in new_cnts]

		rect_list = sorted(rect_list, key = lambda x:x[1] + x[3])
		
		
		if len(rect_list) > 4:
			return (0,0,0,0), (0,0,0,0)
		elif len(rect_list) == 4:
			benchmark1 = rect_list[1]
			benchmark2 = rect_list[2]
		elif len(rect_list) == 3:
			top = .65
			bottom = .35
			tmp_list = copy.deepcopy(rect_list)
			for x in range(5):
				if len(tmp_list) == 3:
					tmp_list = [rectangle for rectangle in rect_list if rectangle[1] >= int(bottom*img.shape[1]) and rectangle[1] + rectangle[3] <= int(top*img.shape[1])]
					top += 0.05
					bottom -= 0.05
					if len(tmp_list) == 2:
						benchmark1 = tmp_list[0]
						benchmark2 = tmp_list[1]
			return (0,0,0,0), (0,0,0,0)
		elif len(rect_list) == 2:
				benchmark1 = rect_list[0]
				benchmark2 = rect_list[1]
		else:
			return (0,0,0,0), (0,0,0,0)
		# if len(new_cnts) < 2:
		# 	return (0,0,0,0), (0,0,0,0)
		# elif len(new_cnts) == 2:
		# 	benchmark1 = cv2.boundingRect(new_cnts[0])
		# 	benchmark2 = cv2.boundingRect(new_cnts[1])
		# elif len(new_cnts) > 2:
		# 	rect_list = [cv2.boundingRect(cnts) for cnts in new_cnts]
		# 	rect_list = [rect for rect in rect_list if rect != (0,0,0,0)]
		# 	lower_quarter = int(.10*img.shape[1])
		# 	upper_quarter = int(.90*img.shape[1])
		# 	rect_list2 = [cnt for cnt in rect_list if cnt[1] + cnt[3] < upper_quarter and cnt[1] > lower_quarter]
		# 	sizes = [rect[1]*rect[3] for rect in rect_list2]
		# 	if len(rect_list2) > 2:
		# 		pass
		# 	else:
		# 		return (0,0,0,0), (0,0,0,0)
			
	else:
		benchmark1 = cv2.boundingRect(benchmark_cnts[0])
		benchmark2 = cv2.boundingRect(benchmark_cnts[1])

	cv2.rectangle(original, (benchmark1[0], benchmark1[1]), (benchmark1[0] + benchmark1[2], benchmark1[1] + benchmark1[3]), (0, 255, 0), 3)
	cv2.rectangle(original, (benchmark2[0], benchmark2[1]), (benchmark2[0] + benchmark2[2], benchmark2[1] + benchmark2[3]), (0, 255, 0), 3)
	# #benchmark_cnts = benchmark_cnts[0] if len(benchmark_cnts) == 2 else benchmark_cnts[1]
	# horizontal_cnts_grip = sorted(benchmark_cnts, key = cv2.contourArea, reverse = True)[:2]
	# benchmark_cnts = sorted(benchmark_cnts, key = cv2.contourArea, reverse = True)[2:]

	# kernel = np.ones((5,5), np.uint8)
	# blur = cv2.GaussianBlur(img, (3,3), 0)

	# thresh = cv2.threshold(blur, 75, 255, cv2.THRESH_BINARY)[1]


	# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21,21))
	# opening = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations = 3)
	# cnts= cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# cnts = imutils.grab_contours(cnts)
	# #cv2.drawContours(original, cnts, -1, (0,255,0), 3)
	# good_cnts = []
	# for i, c in enumerate(cnts):
	# 	if cv2.contourArea(c) >= 35000 or cv2.contourArea(c) <= 2000:
	# 		#print(f"bad area is: {cv2.contourArea(c)}")
	# 		continue
	# 	else:
	# 		#print(f"good area is: {cv2.contourArea(c)}")
	# 		good_cnts.append(c)
	# cnts = []

	# print(f"DETECTED {len(good_cnts)} POTENTIAL BENCHMARKS")
	# if len(good_cnts) == 3:

	# 	x,y,w,h = benchmark_validity(good_cnts, original)
	# 	#cnts = sorted(new_cnts, key = cv2.contourArea)[0:1]
	# 	#x,y,w,h = cv2.boundingRect(cnts[0])
	# elif len(good_cnts) >= 1 and len(good_cnts) < 3:
	# 	cnts = sorted(good_cnts, key=cv2.contourArea, reverse = True)[:1]
	# 	x,y,w,h = cv2.boundingRect(cnts[0])
	# elif len(good_cnts) > 3 and len(good_cnts) <=6:
	# 	x,y,w,h = benchmark_validity(good_cnts, original)
	# 	#cnts = sorted(good_cnts, key = cv2.contourArea, reverse = True)[0:1]
	# 	#x,y,w,h = cv2.boundingRect(cnts[0])
	# else:
	# 	x,y,w,h = (0,0,0,0)

	# return x,y,w,h
	# good_contours=[]
	# for x in cnts:
	# 	if cv2.contourArea(x) < 25000:
	# 		good_contours.append(x)
	# final_benchmark = good_contours[0]
	# x,y,w,h = cv2.boundingRect(final_benchmark)
	#cv2.rectangle(original, (benchmark1[0], benchmark1[1]), (benchmark1[0] + benchmark1[2], benchmark1[1]+benchmark1[3]), (0, 255, 0), 3) #this just draws the rectangle on the image
	#cv2.rectangle(original, (benchmark2[0], benchmark2[1]), (benchmark2[0] + benchmark2[2], benchmark2[1]+benchmark2[3]), (0, 255, 0), 3)
	# plot(original)
	return benchmark1, benchmark2


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



#def find_nail(): first step in our program, finds the contours inside the nail, and then returns the minimum enclosing rectangle of that nail
def find_grips(image, sigma=.33):
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
	edged = cv2.Canny(blur, 15, 225)
	thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
	grip_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 3))
	detect_grip = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, grip_kernel, iterations = 2)
	

	fill_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (75, 75))
	filled_grip = cv2.morphologyEx(detect_grip, cv2.MORPH_CLOSE, fill_kernel, iterations = 3)

	grip_cnts = imutils.grab_contours(cv2.findContours(filled_grip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))
	grip_cnts = sorted(grip_cnts, key = cv2.contourArea, reverse = True)[:2]

	nail_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
	fill_nail = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, nail_kernel, iterations = 3)
	
	nail_kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 700))
	detect_nail = cv2.morphologyEx(fill_nail, cv2.MORPH_OPEN, nail_kernel2, iterations = 3)
	nail_cnts = imutils.grab_contours(cv2.findContours(detect_nail, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE))

	if len(nail_cnts) >= 1:
		nail_cords = cv2.boundingRect(nail_cnts[0])
	else:
		nail_cords = (0, 0, 0, 0)

	# full_cnts = imutils.grab_contours(cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE))
	# full_cnts = sorted(full_cnts, key = cv2.contourArea, reverse = True)[:int(len(full_cnts)*.8)]

	# contours = np.concatenate(full_cnts, axis=0)
	# contours = contours[:, 0, :]

	if len(grip_cnts) < 2:
		grip1 = (0,0,0,0)
		grip2 = (0,0,0,0)
	else:
		grip1 = cv2.boundingRect(grip_cnts[0])
		grip2 = cv2.boundingRect(grip_cnts[1])

	
	if grip1 != (0,0,0,0):
		cv2.rectangle(image, (grip1[0], grip1[1]), (grip1[0]+grip1[2], grip1[1]+grip1[3]), (0, 0, 255), 3)
	if grip2 != (0,0,0,0):
		cv2.rectangle(image, (grip2[0], grip2[1]), (grip2[0]+grip2[2], grip2[1]+grip2[3]), (255, 0, 0), 3)
	if nail_cords != (0,0,0,0):
		cv2.rectangle(image, (nail_cords[0], nail_cords[1]), (nail_cords[0]+ nail_cords[2], nail_cords[1] + nail_cords[3]), (255,0, 0), 3)


	return ((grip1), (grip2), (nail_cords))


def calculate(path):
	image = cv2.imread(path)
	preserved_image = copy.deepcopy(image)
	#change line below for image format
	if not path.endswith(".jpg"):
		return "UNRECOGNIZED FORMAT"
	#top_rect, bottom_rect = find_nail(image)
	grip1, grip2, nail_width_cords = find_grips(image)
	# if grip1 == (0, 0, 0, 0) or grip2 == (0,0,0,0):
	# 	return ("invalid", 0, 0, 0, 0)
	#if top_rect == (0,0,0,0) or bottom_rect == (0,0,0,0):
		#return image #(path[11:], 0, 0, 0)
	nail_width = nail_width_cords[2]

	if grip1[1] < grip2[1]:
		grip_distance = abs(grip1[1] + grip1[3] - grip2[1])
		top_grip = grip1
		bottom_grip = grip2
	else:
		grip_distance = abs(grip2[1] + grip2[3] - grip1[1])
		top_grip = grip2
		bottom_grip = grip1
	
	gray = cv2.cvtColor(preserved_image, cv2.COLOR_BGR2GRAY)

	benchmark1, benchmark2 = benchmark(gray, image, nail_width_cords)

	if benchmark1[1] < benchmark2[1]:
		benchmark_distance = abs(benchmark1[1] + benchmark1[3] - benchmark2[1])
		top_benchmark = benchmark1
		bottom_benchmark = benchmark2
	else:
		benchmark_distance = abs(benchmark2[1] + benchmark2[3] - benchmark1[1])
		top_benchmark = benchmark2
		bottom_benchmark = benchmark1


	cv2.line(image, (850, top_grip[1]+top_grip[3]), (850, bottom_grip[1]), (0, 255, 0), 5)
	image = cv2.putText(image, "Distance in pixels: " + str(grip_distance), (275, top_grip[1]+top_grip[3] + 75), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 255), 4)
	cv2.line(image, (nail_width_cords[0], int(nail_width_cords[3]/2)), (nail_width_cords[0] + nail_width_cords[2], int(nail_width_cords[3]/2)), (0, 255, 0), 3)
	image = cv2.putText(image, "Width of Nail in pixels: " + str(nail_width), (nail_width_cords[0] + nail_width_cords[2] + 50, int(nail_width_cords[3]/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0,0,255), 4)
	cv2.line(image, (top_benchmark[0] + int(top_benchmark[2]/2), top_benchmark[1] + top_benchmark[3]), (top_benchmark[0] + int(top_benchmark[2]/2), bottom_benchmark[1]), (0, 0, 255), 4)
	image = cv2.putText(image, "Benchmark distance in pixels: " + str(benchmark_distance), (top_benchmark[0] + top_benchmark[2] + 50, top_benchmark[1] + int(benchmark_distance/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 255), 4)

	return image
	#return (path[11:], grip_distance, nail_width, benchmark_distance)
	#return (path[11:], distance, width, benchmark_height)


def sorter(data):
	if data[0].isnumeric():
		return data[0].toInt()

def main():

	print(f"STARTING EXECUTION......")
	start = timer()
	path = "./3"
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

	frame_size = (2448, 2048)
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	out = cv2.VideoWriter("dataset3_vid.mp4", fourcc, 15, frame_size, isColor = True)
	image_list = []
	for images in glob.iglob(f"{path}/*"):
		image = calculate(images)
		image_list.append(image)
		#cv2.resize(image, (2048, 2048))
		#image_list.append(image)
	#to_csv(("ImageName", "GripDistance", "NailWidth", "BenchmarkDistance"), "./outputData.csv")
	#for x in image_list:
	for x in image_list:
		out.write(x)
	out.release()
		#to_csv(x, "./outputData.csv")
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
