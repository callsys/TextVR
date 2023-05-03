import copy
import os
import json
import numpy as np
import tqdm
import random
import concurrent
from concurrent.futures import ProcessPoolExecutor
import cv2


ocr_dir = '../vitvr/vitvr_json'
json_path = '../vitvr/vitvr_830.json'

