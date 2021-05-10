# import the necessary packages
import os

# define the base path to the *original* input dataset and then use
# the base path to derive the image and annotations directories
ORIG_BASE_PATH = "ants"
ORIG_IMAGES = os.path.sep.join([ORIG_BASE_PATH, "images"])
ORIG_ANNOTS = os.path.sep.join([ORIG_BASE_PATH, "annotations"])

# define the base path to the *new* dataset after running our dataset
# builder scripts and then use the base path to derive the paths to
# our output class label directories
BASE_PATH = "dataset_ants"
POSITVE_PATH = os.path.sep.join([BASE_PATH, "ant"])
NEGATIVE_PATH = os.path.sep.join([BASE_PATH, "no_ant"])

# define the number of max proposals used when running selective
# search for (1) gathering training data and (2) performing inference
MAX_PROPOSALS = 2000
MAX_PROPOSALS_INFER = 400

# define the maximum number of positive and negative images to be
# generated from each image
MAX_POSITIVE = 900
MAX_NEGATIVE = 250

# initialize the input dimensions to the network
INPUT_DIMS = (224, 224)

# define the path to the output model and label binarizer
MODEL_PATH = "ant_detector.h5"
ENCODER_PATH = "ant_label_encoder.pickle"

# define the minimum probability required for a positive prediction
# (used to filter out false-positive predictions)
MIN_PROBA = 0.3