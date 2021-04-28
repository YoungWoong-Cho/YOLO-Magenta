import cv2
import ffmpeg
import glob
import matplotlib.pyplot as plt
import os
from PIL import Image

from utils import *
from darknet import Darknet

FPS = 1
INPUT_ROOT = 'input'
FRAMES_ROOT = 'frames'

# Extract images from video
def extract_images(input_video, output_dir, fps):
	extracted_path = os.path.join(output_dir, 'extracted')
	if not os.path.exists(extracted_path): os.makedirs(extracted_path)
	os.system(f"ffmpeg -i {input_video} -vf fps={fps} ./{extracted_path}/out%03d.png")

# Create gif from the images
def generate_gif(input_images_dir):
	frames = os.path.join(input_images_dir, '*.png')
	gif_file = os.path.join(input_images_dir, 'out.gif')
	img, *imgs = [Image.open(f) for f in sorted(glob.glob(frames))]
	img.save(fp=gif_file, format='GIF', append_images=imgs,
	         save_all=True, duration=200, loop=0)

def get_image_list(images_dir):
	images_list = [file for file in os.listdir(images_dir) if file.endswith('png')]
	return images_list

# input_video = os.path.join(INPUT_ROOT, os.listdir(INPUT_ROOT)[0])
# extract_images(input_video, FRAMES_ROOT, 1)
# generate_gif(os.path.join(FRAMES_ROOT, 'extracted'))
####################################
#            ---YOLO---            #
####################################
# Load model
cfg_file = './cfg/yolov3.cfg'
weight_file = './weights/yolov3.weights'
namesfile = 'data/coco.names'
m = Darknet(cfg_file)
m.load_weights(weight_file)
class_names = load_class_names(namesfile)
# Set threshold
iou_thresh = 0.4
nms_thresh = 0.6
# Run YOLO
def run_YOLO():
	extracted_path = os.path.join(FRAMES_ROOT, 'extracted')
	detected_path = os.path.join(FRAMES_ROOT, 'detected')
	if not os.path.exists(extracted_path): os.makedirs(extracted_path)
	if not os.path.exists(detected_path): os.makedirs(detected_path)
	images_list = get_image_list(extracted_path)
	# plt.figure(figsize=(10, 6))
	for idx, image in enumerate(images_list):
		# Load the image
		img = cv2.imread(os.path.join(extracted_path, image))
		original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert from BGR to RGB   
		resized_image = cv2.resize(original_image, (m.width, m.height)) # Resize the image
		boxes = detect_objects(m, resized_image, iou_thresh, nms_thresh) # Detect objects in the image
		# print_objects(boxes, class_names) # Print the objects found and the confidence level
		#Plot the image with bounding boxes and corresponding object class labels
		print(f'FRAME {idx}')
		plot_boxes(original_image, boxes, class_names,
					output_dir=detected_path, output_filename=image, plot_labels = True)
		print()

run_YOLO()
generate_gif(os.path.join(FRAMES_ROOT, 'detected'))