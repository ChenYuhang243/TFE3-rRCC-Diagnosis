#!/usr/bin/env python
# coding: utf-8
import numpy as np
import os
from pathlib import Path
from PIL import Image
import glob
import random
import sys
import getopt
import tifffile as tif
import gc
from tqdm import tqdm
import staintools
import pandas as pd
import cv2
import filter

argv = sys.argv[1:]
opts,args = getopt.getopt(argv,'l:i:n:')
for opt,arg in opts:
    elif opt in ['-l']:
        LEVEL= arg # [10X/20X/40X]
    elif opt in ["-i"]:
        INDEX=int(arg) # for indexing
    elif opt in ["-n"]:
        NUM = int(arg)
    elif opt in ["-r"]:
        rank = int(arg)
classes_to_index = {'Negative':0,'Positive':1}" # class mapping
df = pd.read_csv("/path/to/csv") # csv includes tile paths
cases = df["tile_path"].to_list()

tile_path = f"/tile/path/parent/dir"
cn_path = f"/normalized/tile/parent/dir"

cases_select = cases[NUM*(INDEX-1):NUM*INDEX]

# Load templates and initialize normalizers for different magnifications
def load_normalizers():
    templates = {level: tif.imread(glob.glob(f"/path/to/normalization/template/{level}/*.tiff")[0]) 
                 for level in ["10X", "20X", "40X"]}
    normalizers = {level: staintools.StainNormalizer(method="vahadane") for level in templates}
    for level, template in templates.items():
        normalizers[level].fit(staintools.LuminosityStandardizer.standardize(template))
    return normalizers

normalizers = load_normalizers()
unnorm_tiles = []
# Normalize tiles
for case in tqdm(cases_select):
    tiles = list(Path(case).rglob("*.tiff"))
    try:
        for tile in tiles:
            level = Path(tile).parent.name
            if level in normalizers:
                try:
                    tile_name = str(tile).replace(tile_path, cn_path)
                    if not Path(tile_name).exists():
                        Path(tile_name).parent.mkdir(parents=True, exist_ok=True)
                        tile_img = tif.imread(str(tile))
                        SIZE = 256
                        if tile_img.shape[:2] != (SIZE, SIZE):
                            tile_img = Image.fromarray(np.uint8(tile_img)).resize((SIZE, SIZE), Image.ANTIALIAS)
                            tile_img = np.asarray(tile_img)
                        tile_img = staintools.LuminosityStandardizer.standardize(tile_img)
                        norm_img = normalizers[level].transform(tile_img)
                        tif.imsave(tile_name, norm_img)
                except Exception as e:
                    print(e, tile)
                    unnorm_tiles.append(tile)
    except Exception as e:
        print(e, case)
        continue