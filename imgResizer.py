import os
import cv2
import numpy as np

def resize_with_padding(image, target_size):
    target_width, target_height = target_size
    old_size = image.shape[:2]

    ratio = min(target_width / old_size[1], target_height / old_size[0])
    new_size = (int(old_size[1] * ratio), int(old_size[0] * ratio))

    resized_image = cv2.resize(image, new_size)

    delta_w = target_width - new_size[0]
    delta_h = target_height - new_size[1]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return new_image, ratio, left, top

def update_annotations(label_path, new_path, ratio, left, top, target_size, original_size):
    target_width, target_height = target_size
    original_width, original_height = original_size
    
    with open(label_path, 'r') as file:
        lines = file.readlines()

    new_lines = []
    for line in lines:
        class_index, x_center, y_center, width, height = map(float, line.split())

        x_center = x_center * ratio + left
        y_center = y_center * ratio + top
        width = width * ratio
        height = height * ratio
        x_center = x_center+width/2
        y_center = y_center+height/2

        x_center /= target_width
        y_center /= target_height
        width /= target_width
        height /= target_height
        
        new_lines.append(f"0 {x_center} {y_center} {width} {height}\n")

    with open(new_path, 'w') as file:
        file.writelines(new_lines)

def process_directory(image_dir, label_dir, target_size=(1920, 1080)):
    for root, _, files in os.walk(image_dir):
        print(f"processing {root}...")
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                img_path = os.path.join(root, file)
                image = cv2.imread(img_path)
                if image is not None:
                    resized_image, ratio, left, top = resize_with_padding(image, target_size)
                    
                    # cv2.imshow('Resized Image', resized_image)
                    # print(f"Displaying resized image: {img_path}")
                    
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    
                    cv2.imwrite('DeepFakeDetect\Dataset\\train\imagesr\\'+file, resized_image)
                    # print(f"Resized and saved: DeepFakeDetect\Dataset\\train\imagesr\\{file}")
                    
                    # relative_path = os.path.relpath(root, image_dir)
                    # label_path = os.path.join(label_dir, relative_path, os.path.splitext(file)[0] + '.txt')
                    label_path = root.replace('images','labels')+'\\'+file[:-3]+'txt'
                    new_path = label_dir+'r\\'+file[:-3]+'txt'
                    
                    update_annotations(label_path,new_path, ratio, left, top, target_size, image.shape[:2])
                    # print(f"Updated annotations for: {label_path}")
                else:
                    print(f"Failed to load image: {img_path}")

process_directory('DeepFakeDetect\Dataset\\train\images', 'DeepFakeDetect\Dataset\\train\labels')
print('done')