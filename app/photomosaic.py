import cv2
import numpy as np
import random
import uuid
import cv2
import numpy as np
import os
import imutils
import tqdm

class Photomosaic:
    def __init__(self, target_path, pallet_images_path, num_tiles_horizontal=10, num_tiles_vertical=10, tile_opacity=40):
        self.large_image = cv2.imread(target_path)
        self.large_image = imutils.resize(self.large_image, width=5000)
        self.tile_size = self.calculate_tile_size(self.large_image, num_tiles_horizontal, num_tiles_vertical)
        self.tile_images = [cv2.imread(img) for img in pallet_images_path]
        self.mosaic = np.zeros_like(self.large_image)
        self.tile_opacity = tile_opacity
    
    def calculate_tile_size(self, large_image, num_horizontal_tiles, num_vertical_tiles):
        rectangle_height = large_image.shape[0]
        rectangle_width = large_image.shape[1]
        tile_width = rectangle_width / num_horizontal_tiles
        tile_height = rectangle_height / num_vertical_tiles
        return int(tile_width), int(tile_height)
    
    def crop_image(self, image, crop_width, crop_height):
        max_resize = max(crop_height, crop_width, image.shape[0],image.shape[1])
        # max_resize = 2 * max_resize

        #print(max_resize)
        image = imutils.resize(image, width=max_resize)
        #print(image.shape)
        # Calculate center coordinates
        image_height, image_width = image.shape[:2]
        center_x = image_width // 2
        center_y = image_height // 2
        
        # Calculate half of the crop dimensions
        half_crop_width = crop_width // 2
        half_crop_height = crop_height // 2
        
        # Calculate crop boundaries
        start_x = center_x - half_crop_width
        end_x = center_x + half_crop_width
        start_y = center_y - half_crop_height
        end_y = center_y + half_crop_height
        
        # Crop the image from the center
        cropped_image = image[start_y:end_y, start_x:end_x]
        
        return cropped_image
    
    
    def color_match_and_blend(self, region, tile):
        # Resize the tile to match the size of the region
        tile_resized = cv2.resize(tile, (region.shape[1], region.shape[0]))
        
        # Compute the average color of the region and resized tile
        region_avg_color = np.mean(region, axis=(0, 1))
        tile_avg_color = np.mean(tile_resized, axis=(0, 1))
        
        # Calculate the color difference between region and resized tile
        color_diff = np.linalg.norm(region_avg_color - tile_avg_color)
        
        # Blend the region and resized tile based on color similarity
        #alpha = 1.0 - (color_diff / 255.0)  # Adjust alpha based on color difference
        #print(alpha)
        alpha = self.tile_opacity/100
        print(alpha)
        blended_region = cv2.addWeighted(region, 1-alpha, tile_resized, alpha, 0)
        
        return blended_region
    
    @staticmethod
    def is_matrix_present(target_matrix, matrix_list):
        for matrix in matrix_list:
            if np.array_equal(target_matrix, matrix):
                return True
        return False

    def get_best_tile(self, used_tiles, best_tiles):
        return best_tiles[0]
        random_index = random.randint(0, len(best_tiles)-1)
        best_tile = best_tiles[random_index]
        return best_tile

    def transform(self):
        used_tiles = []

        total_tiles = (self.large_image.shape[0] // self.tile_size[1]) * (self.large_image.shape[1] // self.tile_size[0])
        print(f'total tiles {total_tiles}')
        #progress_bar = tqdm(total=total_tiles, desc="Transforming", unit=" tiles")
        done_tiles = 0
        
        for y in range(0, self.large_image.shape[0], self.tile_size[1]):
            for x in range(0, self.large_image.shape[1], self.tile_size[0]): 
                region = self.large_image[y:y + self.tile_size[1], x:x + self.tile_size[0]]
                region_avg_color = np.mean(region, axis=(0, 1))  
                min_diffs = []  # List to store minimum differences
                best_tiles = []
                color_diff_dict = {}
                
                for tile in self.tile_images:
                    if self.is_matrix_present(tile, used_tiles) == False:
                        tile_avg_color = np.mean(tile, axis=(0, 1))
                        color_diff = np.linalg.norm(region_avg_color - tile_avg_color)
                        color_diff_dict[color_diff] = tile
                
                smallest_color_diffs = sorted(color_diff_dict.keys())[:3]
                smallest_color_diff_tiles = [color_diff_dict[diff] for diff in smallest_color_diffs]

                #print(f'used tiles {len(used_tiles)}')
                #print(f'smallest_color_diff_tiles  {len(smallest_color_diff_tiles)}')
                        # Randomly select one tile from the top three minimum differences
                best_tile = self.get_best_tile(used_tiles, smallest_color_diff_tiles)

                used_tiles.append(best_tile)

                if len(used_tiles) > int(len(self.tile_images)/2):
                    #print('popping')
                    used_tiles.pop(0)

                blended_region = self.color_match_and_blend(region, self.crop_image(best_tile, crop_height=region.shape[0], crop_width=region.shape[1]))  
                self.mosaic[y:y + self.tile_size[1], x:x + self.tile_size[0]] = blended_region

                done_tiles +=1
                yield f'{done_tiles}/{total_tiles}'
                #print(f'done tiles {done_tiles} / {total_tiles}')

                #progress_bar.update(1) 
        
        #progress_bar.close()

       

    def save_image(self, output_path):
        output_image = str(uuid.uuid4())+'_matched_image.jpg'
        output_path = os.path.join(output_path, output_image)
        cv2.imwrite(output_path,self. mosaic)
        return output_path
