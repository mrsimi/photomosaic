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
    
    def load_images(self, source):
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

    def process(self):
        face_im_arr = self.load_images(self.main_images_path)
        main_image_width = face_im_arr.shape[0]
        main_image_height = face_im_arr.shape[1]
        mosaic_template = face_im_arr[::(main_image_width//self.target_res[0]),::(main_image_height//self.target_res[1])]
        
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

        for i in range(self.target_res[0]):
            for j in range(self.target_res[1]):
                template = mosaic_template[i, j]
                match = tree.query(template, k=self.tile_choice)
                pick = random.randint(0, self.tile_choice-1)
                image_idx[i, j] = match[1][pick]

        print(f"Matching tiles to sliding windows: {len(tile_images)} tile images")


        canvas = Image.new('RGB', (self.mosaic_size[0]*self.target_res[0], self.mosaic_size[1]*self.target_res[1]))

        for i in range(self.target_res[0]):
            for j in range(self.target_res[1]):
                arr = tile_images[image_idx[j, i]]
                template = mosaic_template[i, j]
                x, y = i*self.mosaic_size[0], j*self.mosaic_size[1]
                
                # Resize the template to match the size of the tile (arr)
                resized_template = np.tile(template, (arr.shape[0], arr.shape[1], 1))

                # Blend resized_template and arr using NumPy
                alpha = 0.2  # Adjust alpha for blending strength
                blended_array = alpha * resized_template + (1 - alpha) * arr

                # Convert the blended array to a Pillow image
                blended_im = Image.fromarray((blended_array * 255).astype(np.uint8)) 
                #im = Image.fromarray(im_blended)
               
                print('array', arr.shape)
                print('template ', template.shape)
                print('resized templated', resized_template.shape)
                canvas.paste(blended_im, (x,y))
        
        output_path = f'{self.output_dir}/{self.generate_random_string(5)}_mosaic.jpg'
        canvas.save(output_path)
        return output_path
        
