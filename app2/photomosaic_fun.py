import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from scipy import spatial
import random
import requests
from io import BytesIO

def resize_image(img : Image, size : tuple) -> np.ndarray:
        resz_img = ImageOps.fit(img, size, Image.LANCZOS, centering=(0.5, 0.5))
        return np.array(resz_img)

def open_image(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        return image 
    else:
        return None

def blend_image(region, tile, opacity_percent):
    #print('region ', region.shape)
    #print('tile ', tile.shape)
    
#     region_avg_color = np.mean(region, axis=(0, 1))
#     tile_avg_color = np.mean(tile, axis=(0, 1))
#     color_diff = np.linalg.norm(region_avg_color - tile_avg_color)
    alpha = opacity_percent/100 # 1.0 - (color_diff / 255.0)
    blended_region = (alpha * region + (1.0 - alpha) * tile).astype(np.uint8)
    return blended_region

def find_closest_images(given_image, image_list, k=3):
    # mse_values = []
    # for img in image_list:
    #     if img.shape == given_image.shape:
    #         mse_value = np.mean((given_image - img) ** 2)
    #         mse_values.append(mse_value)

    mse_values = [np.mean((given_image - img) ** 2) for img in image_list]
    closest_indices = np.argsort(mse_values)[:k]
    closest_images = [image_list[idx] for idx in closest_indices]
    return random.choice(closest_images)

def process_optimized(target_image_path, tile_images_path, divisions, output_dir, scale=2, opacity_percent=40):
    target_image = Image.open(target_image_path)
    target_image = target_image.convert("RGB")
    original_width, original_height = target_image.size
    print('target image size', target_image.size)
    print('image mode', target_image.mode)
    target_image_resized = resize_image(target_image, (original_width * scale, original_height * scale))
    target_image_array = np.array(target_image_resized)
    grid_size = (target_image_array.shape[0] // divisions, target_image_array.shape[1] // divisions)

    # Preprocess tile images
    tile_images = []
    for tile_path in tile_images_path:
        tile_image = Image.open(tile_path)
        tile_image = tile_image.convert("RGB")
        #print('resize before ', tile_image.size)
        tile_image_resized = resize_image(tile_image, (grid_size[1], grid_size[0]))
        #print('resize after ', tile_image_resized.shape)
        tile_images.append(np.array(tile_image_resized))
        tile_image.close()

    combined_image = np.zeros_like(target_image_array)

    for i in range(divisions):
        for j in range(divisions):
            x1, y1 = j * grid_size[1], i * grid_size[0]
            x2, y2 = (j + 1) * grid_size[1], (i + 1) * grid_size[0]

            grid_image = target_image_array[y1:y2, x1:x2, :]
            #print(f'grid image', grid_image.shape)
            # print(f'tile images', tile_images[0].shape)
            # print(f'grid_size ',grid_size)
            closest_image = find_closest_images(grid_image, tile_images)
            blended_image = blend_image(grid_image, closest_image,opacity_percent)
            combined_image[y1:y2, x1:x2, :] = blended_image

    img = Image.fromarray(combined_image)
    output_path = f'{output_dir}/mosaic.jpg'
    img.save(output_path)
    return output_path