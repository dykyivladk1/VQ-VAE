import os
import shutil


#use collect.py if you want to collect all images in one folder

def collect_images(source_dir, target_dir, image_extensions=['.jpg', '.png', '.jpeg']):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for dirpath, dirnames, filenames in os.walk(source_dir):
        for filename in filenames:
            if any(filename.endswith(ext) for ext in image_extensions):
                shutil.copy(os.path.join(dirpath, filename), target_dir)

collect_images('data/flower_data', 'processed_data')
