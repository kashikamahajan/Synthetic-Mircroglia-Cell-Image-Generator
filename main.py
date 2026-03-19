import numpy as np
import tifffile
from scipy import ndimage
from skimage import exposure
import imagej
from synthetic_microglia_generator import generate_multicell_dataset
from pathlib import Path

PATH_TO_DIR = Path.cwd()

#CHNAGE THE FOLLOWING DIRECTORIES' PATHS AS PER NEEDED
PATH_TO_MESG =  PATH_TO_DIR / 'SingleBeadCrop_776x834y195z_60_60_56_2.5umZ.tif'
PATH_TO_CONV_PSF =  PATH_TO_DIR / 'SingleBeadCrop_776x834y195z_60_60_56_2.5umZ.tif'
PATH_TO_MICROGLIA_DATASET = PATH_TO_DIR / 'synthetic_image_dataset'
PATH_TO_CONV_IMGS = PATH_TO_DIR / 'convulated_images'
PATH_TO_FINAL_IMGS = PATH_TO_DIR / 'final_augmented_images'

PATH_TO_MICROGLIA_DATASET.mkdir(parents=True, exist_ok=True)
PATH_TO_CONV_IMGS.mkdir(parents=True, exist_ok=True)
PATH_TO_FINAL_IMGS.mkdir(parents=True, exist_ok=True)

def convolution(img):

    image_array = tifffile.imread(img)
    image_array = image_array.astype("float32")
    image_array= exposure.rescale_intensity(image_array, out_range=(0,1))

    psf_image = tifffile.imread(PATH_TO_CONV_PSF)
    psf_image = psf_image.astype("float32")
    psf_image = exposure.rescale_intensity(psf_image, out_range = (0,1))

    convolved_img = ndimage.convolve(image_array, psf_image, mode='nearest', cval=0.0)

    tifffile.imwrite('/Users/kashika/Desktop/Haynes_Lab/final_pipeline/conv_img.tif', convolved_img)

def imagej_filtering(img):
    print("hi")


images_dir_path, labels_dir_path = generate_multicell_dataset(
        input_path="filled_mesh_volume.tif",
        output_dir= PATH_TO_MICROGLIA_DATASET,
        n_volumes=5,                        #number of images to generate
        cells_per_volume=10,                #number of cells in each image
        output_shape=(1024, 256, 256),      #shape of the images
        elastic_alpha=10,                   #deformation parameter
        elastic_sigma=3,                    #deformation paramter
        scale_range=(0.3, 0.7),             #resizing mesh scales
        max_cell_size=150,                  #max cell achievable in each image
        random_seed=42  
    )

#for img in images_dir_path:


image_array = tifffile.imread('/Users/kashika/Desktop/Haynes_Lab/final_pipeline/img.tif')
image_array = image_array.astype("float32")
image_array= exposure.rescale_intensity(image_array, out_range=(0,1))

psf_image = tifffile.imread('/Users/kashika/Desktop/Haynes_Lab/SingleBeadCrop_776x834y195z_60_60_56_2.5umZ.tif')
psf_image = psf_image.astype("float32")
psf_image = exposure.rescale_intensity(psf_image, out_range = (0,1))

convolved_img = ndimage.convolve(image_array, psf_image, mode='nearest', cval=0.0)

tifffile.imwrite('/Users/kashika/Desktop/Haynes_Lab/final_pipeline/conv_img.tif', convolved_img)