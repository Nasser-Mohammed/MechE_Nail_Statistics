# import the necessary packages
import copy
import os
import numpy as np
import imutils
import cv2
import csv 
import glob
import multiprocessing as mp
from timeit import default_timer as timer
import argparse


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

#def compile_video(): helper function to compile videos on the dataset
def compile_video(images, dSet):
	gray = cv2.cvtColor(images[0][4], cv2.COLOR_BGR2GRAY)
	frame_size = gray.shape[1], gray.shape[0]
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	out = cv2.VideoWriter(dSet + "video.mp4", fourcc, 15, frame_size, isColor = True)
	for i, x in enumerate(images):
		out.write(x[4])
	out.release()


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
	columns = ["Image Name", "Grip Distance", "Width of Nail", "Distance Between Benchmarks"]
	#hashmap = {"File_Name": data[0], "Distance": data[1], "Width_Nail": data[2], "Height_of_Benchmark": data[3]}
	if len(data[0]) == 5:
		for d in data:
			d.remove(4)
	with open(csv_path, "w", newline = '') as file:

		writer = csv.writer(file)
		writer.writerow(columns)
		writer.writerows(data)
		file.close()

	
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
		for c in retry_cnts:
			temp = copy.deepcopy(c)
			temp = np.concatenate(temp, axis = 0)
			temp = temp[(temp[:, 0] >= nail_cords[0] - 15) & (temp[:, 0] <= nail_cords[0] + nail_cords[2] + 15)]
			if cv2.boundingRect(temp) != (0,0,0,0):
				new_cnts.append(c)

		if len(new_cnts) == 0:
			return (0,0,0,0), (0,0,0,0)
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
			while top <= .95 and bottom >= .05:
				if len(tmp_list) == 3:
					tmp_list = [rectangle for rectangle in rect_list if rectangle[1] >= int(bottom*img.shape[1]) and rectangle[1] + rectangle[3] <= int(top*img.shape[1])]
					top += 0.05
					bottom -= 0.05
					if len(tmp_list) == 2:
						benchmark1 = tmp_list[0]
						benchmark2 = tmp_list[1]
						break
			if len(tmp_list) > 2 or len(tmp_list) == 0:
				return (0,0,0,0), (0,0,0,0)
		elif len(rect_list) == 2:
				benchmark1 = rect_list[0]
				benchmark2 = rect_list[1]
		else:
			return (0,0,0,0), (0,0,0,0)
			
	else:
		benchmark1 = cv2.boundingRect(benchmark_cnts[0])
		benchmark2 = cv2.boundingRect(benchmark_cnts[1])

	cv2.rectangle(original, (benchmark1[0], benchmark1[1]), (benchmark1[0] + benchmark1[2], benchmark1[1] + benchmark1[3]), (0, 255, 0), 3)
	cv2.rectangle(original, (benchmark2[0], benchmark2[1]), (benchmark2[0] + benchmark2[2], benchmark2[1] + benchmark2[3]), (0, 255, 0), 3)
	return benchmark1, benchmark2

#def find_nail(): first step in our program, finds the contours inside the nail, and then returns the minimum enclosing rectangle of that nail
def find_grips(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (3,3), 0)

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
	# if path.endswith(".jpg") or  path.endswith(".tif"):
	# 	pass
	# else:
	# 	return ("error: unrecognized image format", 0, 0, 0)
	#top_rect, bottom_rect = find_nail(image)
	grip1, grip2, nail_width_cords = find_grips(image)
	if grip1 == (0, 0, 0, 0) or grip2 == (0,0,0,0):
		return ("error:couldn't find grips", 0, 0, 0, image)
	#if top_rect == (0,0,0,0) or bottom_rect == (0,0,0,0):
		#return image #(path[11:], 0, 0, 0)
	nail_width = nail_width_cords[2]


	if grip1 == (0,0,0,0) or grip2 == (0,0,0,0):
		top_grip = grip1
		bottom_grip = grip2
		grip_distance = 0

	elif grip1[1] < grip2[1]:
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
	path = path.split("_")[-1]
	path = path.split(".")[0]
	return (path, grip_distance, nail_width, benchmark_distance, image)


def sorter(data):
	nums = [s for s in data[0] if s.isdigit()]
	return int("".join(nums))


def main():
	parser = argparse.ArgumentParser(description = "Detects metrics on nail being stretched")
	parser.add_argument("path", metavar = "path", type = str, help = "enter path of folder with images you want calculations on")
	parser.add_argument("output", help = "output options", choices = ["csv", "CSV", "Video", "video", "All","all"])
	args = parser.parse_args()
	path = args.path

	operatingSys = os.name
	if operatingSys == "nt":
		dSet = path.split("\\")[-1]
	else:
		dSet = path.split("/")[-1]

	
	print(f"STARTING EXECUTION......")
	start = timer()
	files = os.listdir(path)
	arg = [path+"/"+x for x in files if os.path.isfile(path+"/"+x)]
	num_of_files = len(os.listdir(path))
	print(f"PROCESSING {num_of_files} IMAGES")
	num_cores = mp.cpu_count()
	if num_of_files < 20:
		processes = 2
	else: 
		processes = num_cores - 1
	pool = mp.Pool(processes = processes)


	#frame_size = (2448, 2048)
	#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	#out = cv2.VideoWriter("dataset3_vid.mp4", fourcc, 15, frame_size, isColor = True)
	#for x in image_list:
	# for x in image_list:
	# 	out.write(x)
	# out.release()
	results = pool.map_async(calculate, arg, chunksize = int(len(arg)/processes))

	sorted_data = []
	for index, data in enumerate(results.get()):
		#print(f"data sample: {index}, Image name: {data[0]}, Grip Distance: {data[1]}, Nail Width: {data[2]}, Benchmark Height: {data[3]}")
		sorted_data.append(data)
	sorted_data = sorted(sorted_data, key = sorter)

	if args.output == "video" or args.output == "Video":
		compile_video(sorted_data, dSet)
	elif args.output == "csv" or args.output == "CSV":
		to_csv([(s[0], s[1], s[2], s[3]) for s in sorted_data], "./"+"metrics_for_"+ dSet +".csv")
	else:
		compile_video(sorted_data, dSet)
		to_csv([(s[0], s[1], s[2], s[3]) for s in sorted_data], "./"+"metrics_for_"+ dSet +".csv")
	final_time = timer()
	print(f"Program took {final_time-start} seconds to process images")

if __name__ == "__main__":
	main()
