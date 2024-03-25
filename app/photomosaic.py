import cv2
import numpy as np
import random
import uuid
import cv2
import numpy as np
import os
import imutils

class Photomosaic:
    def __init__(self, target_path, pallet_images_path, num_tiles_horizontal=10, num_tiles_vertical=10):
        self.large_image = cv2.imread(target_path)
        self.large_image = imutils.resize(self.large_image, width=4096)
        self.tile_size = self.calculate_tile_size(self.large_image, num_tiles_horizontal, num_tiles_vertical)
        self.tile_images = [cv2.imread(img) for img in pallet_images_path]
    
    def calculate_tile_size(self, large_image, num_horizontal_tiles, num_vertical_tiles):
        rectangle_height = large_image.shape[0]
        rectangle_width = large_image.shape[1]
        tile_width = rectangle_width / num_horizontal_tiles
        tile_height = rectangle_height / num_vertical_tiles
        return int(tile_width), int(tile_height)
    
    def crop_image(self, image, crop_width, crop_height):
        max_resize = max(crop_height, crop_width, image.shape[0],image.shape[1])
        if max_resize < crop_height or max_resize < crop_width:
            max_resize = 2 * max_resize
        image = imutils.resize(image, width=max_resize)

        image_height, image_width = image.shape[:2]
        start_x = (image_width - crop_width) // 2
        end_x = start_x + crop_width
        start_y = (image_height - crop_height) // 2
        end_y = start_y + crop_height
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
        alpha = 1.0 - (color_diff / 255.0)  # Adjust alpha based on color difference
        blended_region = cv2.addWeighted(region, alpha, tile_resized, 1.0 - alpha, 0)
        
        return blended_region

    def transform(self, output_path):
        output_image = str(uuid.uuid4())+'_matched_image.jpg'
        window_size = self.tile_size  # Adjust window size as needed
        mosaic = np.zeros_like(self.large_image)
        used_tiles = []
        used_tiles_length = len(self.tile_size)/2

        for y in range(0, self.large_image.shape[0], self.tile_size[1]):
            for x in range(0, self.large_image.shape[1], self.tile_size[0]): 
                region = self.large_image[y:y + self.tile_size[1], x:x + self.tile_size[0]]
                region_avg_color = np.mean(region, axis=(0, 1))  
                min_diffs = []  # List to store minimum differences
                best_tiles = []
                
                for tile in self.tile_images:
                    tile_avg_color = np.mean(tile, axis=(0, 1))
                    color_diff = np.linalg.norm(region_avg_color - tile_avg_color)
                            
                            # Update min_diffs and best_tiles with top three minimum differences and corresponding tiles
                    if len(min_diffs) < 3:
                        min_diffs.append(color_diff)
                        best_tiles.append(tile)
                    else:
                        max_diff_index = min_diffs.index(max(min_diffs))
                        if color_diff < min_diffs[max_diff_index]:
                            min_diffs[max_diff_index] = color_diff
                            best_tiles[max_diff_index] = tile

                        # Randomly select one tile from the top three minimum differences
                random_index = random.randint(0, 2)  # Random index between 0 and 2
                best_tile = best_tiles[random_index]

                        # Add the selected tile to the used_tiles list
                used_tiles.append(best_tile)

                        # Remove the oldest tile from used_tiles if necessary
                if len(used_tiles) > used_tiles_length:
                    used_tiles.pop(0)
                
                blended_region = self.color_match_and_blend(region, self.crop_image(best_tile, crop_height=self.tile_size[1], crop_width=self.tile_size[0]))  
                mosaic[y:y + self.tile_size[1], x:x + self.tile_size[0]] = imutils.resize(blended_region,width=region.shape[1], height=region.shape[0]) 

        output_path = os.path.join(output_path, output_image)
        cv2.imwrite(output_path, mosaic)
        return output_path
