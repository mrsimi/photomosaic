import cv2
import numpy as np
import random
import uuid
import cv2
import numpy as np
import os
from sklearn.cluster import MiniBatchKMeans

class Photomosaic:
    def __init__(self, target_path, pallet_images_path:list[str], rows=10, columns=10):
        self.target = cv2.imread(target_path) 
        self.target_path = target_path
        self.pallet_images_path = pallet_images_path
        self.rows = rows
        self.columns = columns
        self.image_parts = []
        height, width, _ = self.target.shape
        self.part_height = height // self.rows
        self.part_width = width // self.columns
        self.color_image_parts = []

    def compare_histograms(image_path, reference_histogram):
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hist = cv2.calcHist([image_rgb], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        bhattacharyya_distance = cv2.compareHist(hist, reference_histogram, cv2.HISTCMP_BHATTACHARYYA)
        return bhattacharyya_distance

    
    def color_match(self, target_image):
        given_image_rgb = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
        given_hist = cv2.calcHist([given_image_rgb], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
        cv2.normalize(given_hist, given_hist, 0, 1, cv2.NORM_MINMAX)

        results = {}
        for pallet_path in self.pallet_images_path:
            bhattacharyya_distance = self.compare_histograms(pallet_image, given_hist)
            results[pallet_path] = bhattacharyya_distance

        most_similar_image = min(results, key=results.get)
        # pallet_image = random.choice(self.pallet_images_path)
        
        target_image = cv2.imread(target_image)
        pallet_image = cv2.imread(most_similar_image)
        pallet_image = cv2.resize(pallet_image, (target_image.shape[1], target_image.shape[0]))

        img1_lab = cv2.cvtColor(target_image, cv2.COLOR_BGR2LAB)
        img2_lab = cv2.cvtColor(pallet_image, cv2.COLOR_BGR2LAB)
        
        img1_mean, img1_std = cv2.meanStdDev(img1_lab)
        img2_mean, img2_std = cv2.meanStdDev(img2_lab)

        for i in range(3):  # Iterate over L, A, B channels
            img1_lab[:,:,i] = img1_lab[:,:,i] - img1_mean[i]
            img1_lab[:,:,i] = img1_lab[:,:,i] * (img2_std[i] / img1_std[i])
            img1_lab[:,:,i] = img1_lab[:,:,i] + img2_mean[i]
        
        matched_img = cv2.cvtColor(img1_lab, cv2.COLOR_LAB2BGR)
        return matched_img
    
    def split_images(self):
        image_parts = []
        for row in range(self.rows):
            for col in range(self.columns):
                start_row = row * self.part_height
                end_row = (row + 1) * self.part_height
                start_col = col * self.part_width
                end_col = (col + 1) * self.part_width

                part = self.target[start_row:end_row, start_col:end_col]
                image_parts.append(part)
                
        for i, part in enumerate(image_parts):
            #print('writing to part')
            cv2.imwrite(f'part_{i+1}.jpg', part)
    
    def color_and_combine_images(self):
        for i in range(1, self.rows*self.columns + 1):  
            part = self.color_match(f'part_{i}.jpg')
            self.color_image_parts.append(part)
            
        combined_height = self.rows * self.part_height
        combined_width = self.columns * self.part_width

        combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
        for i, part in enumerate(self.color_image_parts):
            row = i // self.rows
            col = i % self.columns
            start_row = row * self.part_height
            end_row = (row + 1) * self.part_height
            start_col = col * self.part_width
            end_col = (col + 1) * self.part_width
            combined_image[start_row:end_row, start_col:end_col] = part
        
        cv2.imwrite('combined_image.jpg', combined_image)
    
    def transform(self):
        self.split_images()
        self.color_and_combine_images()


class Photomosaicv2:
    def __init__(self, target_path, pallet_images_path, num_tiles_horizontal=5, num_tiles_vertical=5):
        self.large_image = cv2.imread(target_path)
        self.tile_size = self.calculate_tile_size(self.large_image, num_tiles_horizontal, num_tiles_vertical)
        self.resized_tiles = [cv2.resize(cv2.imread(img), self.tile_size) for img in pallet_images_path]
    
    def calculate_tile_size(self, large_image, num_tiles_horizontal, num_tiles_vertical):
        height, width, _ = large_image.shape
        tile_width = width // num_tiles_horizontal
        tile_height = height // num_tiles_vertical
        return tile_width, tile_height
    
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

    def transform(self):
        output_image = str(uuid.uuid4())+'_matched_image.jpg'
        window_size = self.tile_size  # Adjust window size as needed
        mosaic = np.zeros_like(self.large_image)

        for y in range(0, self.large_image.shape[0], window_size[1]):
            for x in range(0, self.large_image.shape[1], window_size[0]):
                region = self.large_image[y:y + window_size[1], x:x + window_size[0]]
                #print(region.shape)
                region_avg_color = np.mean(region, axis=(0, 1))  # Compute average color of the region
                
                # Find the best matching tile image based on average color difference
                min_diff = float('inf')
                best_tile = None
                for tile in self.resized_tiles:
                    tile_avg_color = np.mean(tile, axis=(0, 1))
                    color_diff = np.linalg.norm(region_avg_color - tile_avg_color)
                    if color_diff < min_diff:
                        min_diff = color_diff
                        best_tile = tile
                
                # Replace the region in the mosaic with the best matching tile
                blended_region = self.color_match_and_blend(region, best_tile)    
                mosaic[y:y + window_size[1], x:x + window_size[0]] = cv2.resize(blended_region,(region.shape[1], region.shape[0]))
        
        cv2.imwrite(output_image, mosaic)
        return output_image

if __name__=="__main__":
    target_path = 'img/target.jpeg'
    pallet_images_path = ['img/pallet1.webp','img/pallet3.jpeg', 
                            'img/download.jpeg','img/download2.jpeg', 'img/download3.jpeg', 'img/download4.jpeg']
    photomosaic = Photomosaicv2(target_path, pallet_images_path)
    photomosaic.transform()
