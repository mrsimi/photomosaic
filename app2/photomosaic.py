from PIL import Image, ImageOps,ImageEnhance
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image
import glob, os
import math
from scipy import spatial
import random
import string


class Photomosaic:
    def __init__(self, tile_images_path, main_image_path, output_dir, mosaic_size=40, divisions=20, tile_choice=40, opacity=0.5):
        self.tile_images_path = tile_images_path
        self.main_images_path = main_image_path
        self.target_res = (divisions, divisions)
        self.mosaic_size = (mosaic_size, mosaic_size)
        self.tile_choice = tile_choice
        self.output_dir = output_dir
        self.opacity = opacity
    
    def load_images(self, source, transparent=False):
        with Image.open(source) as im:
            im_arr = np.asarray(im)
            
        return im_arr

    def resize_image(self, img : Image, size : tuple) -> np.ndarray:
        resz_img = ImageOps.fit(img, size, Image.LANCZOS, centering=(0.5, 0.5))
        return np.array(resz_img)
    
    @staticmethod
    def generate_random_string(length):
        letters = string.ascii_letters + string.digits
        return ''.join(random.choice(letters) for _ in range(length))
    
    def load_tile_images(self, tiles_path):
        images = []
        path = tiles_path+'/*';
        print('tile_images_path: ', path)
        for file in glob.glob(path):
            im = self.load_images(file)
            images.append(im)
        return images

    def load_tile_imagesv2(self, tiles_path):
        images = []
        for file_path in tiles_path:
            im = self.load_images(file_path)
            images.append(im)
        
        return images

    @staticmethod
    def image_with_opacity(arr, opacity=128):
        im = Image.fromarray(arr)
        im = im.convert('RGBA')
        transparent_im = Image.new('RGBA', im.size, (0, 0, 0, opacity))
        return Image.alpha_composite(im, transparent_im)
    
    @staticmethod
    def load_image_with_op(source):
        with Image.open(source) as im:
            white = np.array([255, 255, 255], np.uint8)
            vector = white-im

            percent = 0.5
            value = im + vector * percent

            img = Image.fromarray(value.astype(np.uint8), 'RGB')
            im_arr = np.asarray(img)

        return im_arr

    def process(self):
        face_im_arr = self.load_images(self.main_images_path)
        main_image_width = face_im_arr.shape[0]
        main_image_height = face_im_arr.shape[1]
        mosaic_template = face_im_arr[::(main_image_width//self.target_res[0]),::(main_image_height//self.target_res[1])]
        print('face image shape: ',face_im_arr.shape)
        print('mosaic template shape: ', mosaic_template.shape)
        
        print(f"Main Image process completed: {main_image_height} by {main_image_width}")
        
        tile_images = self.load_tile_imagesv2(self.tile_images_path)
        print(len(tile_images))
        tile_images = [i for i in tile_images if i.ndim==3]
        tile_images = [self.resize_image(Image.fromarray(i), self.mosaic_size) for i in tile_images]
        tile_images_array = np.asarray(tile_images)
        tile_images_values = np.apply_over_axes(np.mean, tile_images_array, [1,2]).reshape(len(tile_images),3)

        print(f"Tile Image process completed: {len(tile_images)} tile images")
        
        tree = spatial.KDTree(tile_images_values)
        image_idx = np.zeros(self.target_res, dtype=np.uint32)
        print('target-resolution: ', self.target_res)

        for i in range(self.target_res[0]):
            for j in range(self.target_res[1]):
                #print(i,j)
                template = mosaic_template[i, j]
                match = tree.query(template, k=self.tile_choice)
                pick = random.randint(0, self.tile_choice-1)
                image_idx[i, j] = match[1][pick]

        print(f"Matching tiles to sliding windows: {len(tile_images)} tile images")


        canvas = Image.new('RGB', (self.mosaic_size[0]*self.target_res[0], self.mosaic_size[1]*self.target_res[1]))

        for i in range(self.target_res[0]):
            for j in range(self.target_res[1]):
                arr = tile_images[image_idx[j, i]]
                x, y = i*self.mosaic_size[0], j*self.mosaic_size[1]
                im = Image.fromarray(arr)
                #enhancer = ImageEnhance.Brightness(im)
                #lighter_im = enhancer.enhance(1.5) 
               #im = self.image_with_opacity(arr, int(self.opacity*255))
                canvas.paste(im, (x,y))
        
        output_path = f'{self.output_dir}/{self.generate_random_string(5)}_mosaic.jpg'
        canvas.save(output_path)
        return output_path
        
