# import the necessary packages
from search import config
from imutils import paths
import shutil
import os

# loop over the data splits
for split in (config.TRAIN, config.TEST, config.VAL):
    # grab all image paths in the current split
	print("[INFO] processing '{} split'...".format(split))
	p = os.path.sep.join([config.ORIG_INPUT_DATASET, split])
	imagePaths = list(paths.list_images(p))
	#print(p)
	#print(imagePaths)
    # loop over the image paths
	for imagePath in imagePaths:
		# extract class label from the filename
		filename = imagePath.split(os.path.sep)[-1]
		#print(filename)
		label = config.CLASSES[int(filename.split("_")[0])]
		#print(label)
		# construct the path to the output directory
		dirPath = os.path.sep.join([config.BASE_PATH, split, label])
		#print(dirPath)
		# if the output directory does not exist, create it
		if not os.path.exists(dirPath):
			os.makedirs(dirPath)
		# construct the path to the output image file and copy it
		p = os.path.sep.join([dirPath, filename])
		#print(p)
		#print(imagePath)
		shutil.copy2(imagePath, p)