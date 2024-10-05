#!/usr/bin/env python
# coding: utf-8
import openslide
from openslide.deepzoom import DeepZoomGenerator
import numpy as np
#get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
import skimage
from lxml import etree
from pathlib import Path
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import os
import gc
import time
import sys
import getopt
import pandas as pd

def calculate_tissue_percentage(image):
    # Read the image in RGB color space
    # Convert image to HSV color space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # Define range for tissue colors (typically, tissue appears pinkish in H&E stains)
    # These thresholds may need adjustments based on specific stain characteristics
    lower_bound = np.array([130, 20, 70])  # Lower bound for tissue color
    upper_bound = np.array([180, 255, 255])  # Upper bound for tissue color
    # Create a mask to isolate tissue areas
    mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
    # Calculate the percentage of tissue area
    tissue_percentage = (np.sum(mask > 0) / mask.size) 
    return tissue_percentage


def get_slide(slide_path):
    slide = openslide.OpenSlide(slide_path)
    return slide
def normalize_dynamic_range(image, percentile = 95):
    """
    Normalize the dynamic range of an RGB image to 0~255. If the dynamic ranges of patches 
    from a dataset differ, apply this function before feeding images to VahadaneNormalizer,
    e.g. hema slides.
    :param image: A RGB image in np.ndarray with the shape [..., 3].
    :param percentile: Percentile to get the max value.
    """
    max_rgb = []
    for i in range(3):
        value_max = np.percentile(image[..., i], percentile)
        max_rgb.append(value_max)
    max_rgb = np.array(max_rgb)
    new_image = (np.minimum(image.astype(np.float32) * (255.0 / max_rgb), 255.0)).astype(np.uint8)
    return new_image
# filter blank area
def filter_blank(image, threshold = 80):
    image_lab = skimage.color.rgb2lab(np.array(image))
    image_mask = np.zeros(image.shape).astype(np.uint8)
    image_mask[np.where(image_lab[:, :, 0] < threshold)] = 1
    image_filter = np.multiply(image, image_mask)
    percent = ((image_filter != np.array([0,0,0])).astype(float).sum(axis=2) != 0).sum()/(image_filter.shape[0]**2)
    return percent
# parse annotation
def AnnotationParser(path):
    assert Path(path).exists(), "This annotation file does not exist."
    tree = etree.parse(path)
    annotations = tree.xpath("/ASAP_Annotations/Annotations/Annotation")
    annotation_groups = tree.xpath("/ASAP_Annotations/AnnotationGroups/Group")
    classes = [group.attrib["Name"] for group in annotation_groups]
    def read_mask_coord(cls):
        for annotation in annotations:
            if annotation.attrib["PartOfGroup"] == cls:
                contour = []
                for coord in annotation.xpath("Coordinates/Coordinate"):
                    x = np.float(coord.attrib["X"])
                    y = np.float(coord.attrib["Y"])
                    contour.append([round(float(x)),round(float(y))])
                #mask_coords[cls].extend(contour)
                mask_coords[cls].append(contour)
    def read_mask_coords(classes):
        for cls in classes:
            read_mask_coord(cls)
        return mask_coords            
    mask_coords = {}
    for cls in classes:
        mask_coords[cls] = []
    mask_coords = read_mask_coords(classes)
    return mask_coords,classes

# generate mask
def Annotation(slide,path,save_path=None,rule=False,save=False):
    wsi_width,wsi_height = slide.level_dimensions[0]
    masks = {}
    contours = {}
    mask_coords, classes = AnnotationParser(path)
    
    def base_mask(cls,wsi_height,wsi_width):
        masks[cls] = np.zeros((wsi_height,wsi_width),dtype=np.uint8)
    def base_masks(wsi_height,wsi_width):
        for cls in classes:
            base_mask(cls,wsi_height,wsi_width)
        return masks
    
    def main_masks(classes,mask_coords,masks):
        for cls in classes:
            contours = np.array(mask_coords[cls])
            for contour in contours:
                masks[cls] = cv2.drawContours(masks[cls],[np.int32(contour)],0,True,thickness=cv2.FILLED)
        return masks
    def export_mask(save_path,cls):
        assert Path(save_path).is_dir()
        cv2.imwrite(str(Path(save_path)/"{}.tiff".format(cls)),masks[cls],(cv2.IMWRITE_PXM_BINARY,1))
    def export_masks(save_path):
        for cls in masks.keys():
            export_mask(save_path,cls)
    def exclude_masks(masks,rule,classes):
        masks_exclude = masks
        for cls in classes:
            for exclude in rule[cls]["excludes"]:
                if exclude in masks:
                    overlap_area = cv2.bitwise_and(masks[cls],masks[exclude])
                    masks_exclude[cls] = cv2.bitwise_xor(masks[cls],overlap_area)
        return masks_exclude
                    
    masks = base_masks(wsi_height,wsi_width)
    masks = main_masks(classes,mask_coords,masks)
    if rule:
        classes = list(set(classes) & set(rule.keys()))
        masks = exclude_masks(masks,rule,classes)
    if save:
        export_masks(save_path)
    if "artifact" not in classes: 
        masks["artifact"] = np.zeros((wsi_height,wsi_width),dtype=np.uint8)
    if "mark" not in classes:
        masks["mark"] = np.zeros((wsi_height,wsi_width),dtype=np.uint8)
    return masks 

def show_thumb_mask(mask,size=512):
    height, width = mask.shape
    scale = max(size / height, size / width)
    mask_resized = cv2.resize(mask, dsize=None, fx=scale, fy=scale)
    mask_scaled = mask_resized * 255
    plt.imshow(mask_scaled)
    return mask_scaled

def get_mask_slide(masks):
    tumor_slide = openslide.ImageSlide(Image.fromarray(masks["stroma"]))
    return tumor_slide
# tile generation using DeepZoomGenerator
def get_tiles(slide,tumor_slide,tile_size=512,overlap=False,limit_bounds=False):
    slide_tiles = DeepZoomGenerator(slide,tile_size,overlap=overlap,limit_bounds=limit_bounds)
    tumor_tiles = DeepZoomGenerator(tumor_slide,tile_size,overlap=overlap,limit_bounds=limit_bounds)
    return slide_tiles,tumor_tiles
def remove_arti_and_mask(slide_tile,tumor_tile):
    x = slide_tile.shape
    if not x == tumor_tile.shape:
        tumor_tile = tumor_tile[:x[0],:x[1],:] 
    tile_masked= np.multiply(slide_tile,tumor_tile)
    return slide_tile,tile_masked
def get_tile_masked(slide_tile,tumor_tile): 
    x = slide_tile.shape
    y = tumor_tile.shape
    if not x == y:
        h = np.min([x[0],y[0]])
        w = np.min([x[1],y[1]])
        tumor_tile = tumor_tile[:h,:w,:]
        slide_tile = slide_tile[:h,:w,:]
    percent = np.mean(tumor_tile) # percentage of the annoation area in whole slide
    tile_masked= np.where(tumor_tile==(0,0,0),(255,255,255),slide_tile) 
    return tile_masked,percent
def filtered_same(img):### modify to purely count tumor tile
    percent = ((img[:,:,0]==img[:,:,1]).astype(float) *(img[:,:,0]==img[:,:,2]).astype(float)).sum()/(img.shape[0]**2)
    return percent
def filtered(tile):
    tolerance = np.array([230,230,230])
    tile[np.all(tile>tolerance,axis=2)]=0
    percent = ((tile != np.array([0,0,0])).astype(float).sum(axis=2)!=0).sum()/(tile.shape[0]**2)
    return percent
def filtered_cv(img):
    tile = np.copy(img).astype(np.uint8)
    gray = cv2.cvtColor(tile,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    ret,_ = cv2.threshold(blur,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    tile[np.all(tile>ret,axis=2)] = 0
    percent = ((tile != np.array([0,0,0])).astype(float).sum(axis=2)!=0).sum()/(tile.shape[0]**2)
    return percent

def extract_patches(levels,scales):

    for i,level in enumerate(levels):
        if scales[i] =="10X" or scales[i] =="5X":
            slide_tiles,tumor_tiles = get_tiles(slide,tumor_slide,tile_size=256,overlap=OVERLAP,limit_bounds=LIMIT)
        elif scales[i] =="20X" or scales[i] =="40X":
            slide_tiles,tumor_tiles = get_tiles(slide,tumor_slide,tile_size=256,overlap=OVERLAP,limit_bounds=LIMIT)# change tile size
        print(f'processing ---level {scales[i]}')
        print(tile_path)
        tiledir = Path(tile_path)/str(scales[i])
        
        if not Path(tiledir).exists():
            os.makedirs(tiledir)
        assert slide_tiles.level_tiles[level] == tumor_tiles.level_tiles[level]
        cols,rows = slide_tiles.level_tiles[level]
        for row in range(rows):
            for col in range(cols):
                tilename = os.path.join(tiledir,'%d_%d.%s'%(col,row,"tiff"))
                try:
                    if not Path(tilename).exists():
                        slide_tile = np.array(slide_tiles.get_tile(level,(col,row)))
                        tumor_tile = np.array(tumor_tiles.get_tile(level,(col,row)))
                    tile_masked,annotation_percent = get_tile_masked(slide_tile,tumor_tile) # percent of annotated area 
                    tissue_percent = filter_blank(tile_masked) # percent of tissue area
                    if tissue_percent >= 0.50 and annotation_percent >= 0.5:
                        Image.fromarray(np.uint8(tile_masked)).save(tilename)
                    else:
                        pass
                except Exception as e:
                    continue
                    print(e)
                    print("ERROR occurs inner extraction loop")
        print("Done!")
    print("All levels processed!!")
    
INDEX= 0 # default index of the running process
n = 5  # default number of slides per process
argv = sys.argv[1:]
try:
    opts,args = getopt.getopt(argv,"n:i:")
except:
    print("Error")
for opt,arg in opts:
    if opt in ['-n']:
        n = int(arg) # number of slides per process
    elif opt in ['-i']:
        INDEX = int(arg)  # index of the running process 


# params for deepzoom
OVERLAP =0
LIMIT = False
#annotation rule, excludes dict includes annotations to filter out from the stroma annotation
rule = {"stroma":{"excludes":["blood","artifact","mark"]}}

df = pd.read_csv("/path/to/csv/directory") # csv containing slides dirs and slide level labels
classes = ["Negative","Positive"] # class mapping dict
svs_paths = df["slides"].to_list() # retrieve slides list
svs_labels = df["label"].to_list() # retrieve slides corresponding labels
patch_path = "/path/to/save/tiles"
# partition cases based on the process index and number of each process
number = len(svs_paths)
svs_paths = svs_paths[n*(INDEX-1):n*INDEX] 
labels = svs_labels[n*(INDEX-1):n*INDEX]

extracted_case = []
un_extracted_case = []
for i,svs in enumerate(svs_paths):
    start = time.time()
    totol_num = len(svs_paths)
    print(f"processing  {i+1}/{totol_num}:------{svs}")
    label = labels[i]
    xml_path = str(Path(svs).with_suffix(".xml"))
    case_name = Path(svs).stem
    tile_path = Path(patch_path)/classes[label]/case_name
    slide = get_slide(str(svs))
    app = dict(slide.properties)["aperio.AppMag"]
    print(app) # print the slide magnification 
    if app=="40": # if manification is 40X 
        scales =  ['5X','10X','20X',"40X"]
        try:
            masks = Annotation(slide,path=str(xml_path))
            print(f"masks groups includes :{list(masks.keys())}")
            tumor_slide = get_mask_slide(masks) 
            slide_tiles,tumor_tiles = get_tiles(slide,tumor_slide,tile_size=512,overlap=OVERLAP,limit_bounds=LIMIT)
            level_count = slide_tiles.level_count
            levels=[level_count-4,level_count-3,level_count-2,level_count-1] # the max level_count in deepzoom generator is equal to openslide.level 0 which is the manification at scanning
            try:
                extract_patches(levels,scales)
                extracted_case.append(svs)
            except Exception as e:
                un_extracted_case.append(svs)
                print("something is wrong when extracting")
                print("ERROR!",e)
                continue
        except Exception as e:
            print("something is wrong when parsing")
            print("ERROR!",e)
            continue
    elif app=="20":
        scales =  ['5X','10X','20X']
        try:
            masks = Annotation(slide,path=str(xml_path))
            print(f"masks groups includes :{list(masks.keys())}")
            tumor_slide = get_mask_slide(masks) 
            slide_tiles,tumor_tiles = get_tiles(slide,tumor_slide,tile_size=512,overlap=OVERLAP,limit_bounds=LIMIT)
            level_count = slide_tiles.level_count
            levels=[level_count-3,level_count-2,level_count-1]
            try:
                extract_patches(levels,scales)
                extracted_case.append(svs)
            except Exception as e:
                un_extracted_case.append(svs)
                print("Error occurs when extracting")
                print("ERROR!",e)
                continue
        except Exception as e:
            print("something is wrong when parsing")
            print("ERROR!",e)
            continue
    end = time.time()
    print(f"Time consumed : {(end-start)/60} min")
    print(f"******{len(un_extracted_case)}/{len(svs_paths)} unextracted cases******")
