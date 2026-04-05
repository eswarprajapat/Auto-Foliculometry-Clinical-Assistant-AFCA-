import json
import os
import cv2
import numpy as np

def create_masks_from_json():
    # 1. Setup your paths
    # WARNING: Change this exact filename to match your downloaded MakeSense.ai file!
    json_file = "labels_my-project-name.json" 
    image_dir = "enhanced_images"
    output_dir = "masks"

    # Create the output folder automatically
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load the COCO JSON file
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find '{json_file}'. Check the filename!")
        return

    print("JSON loaded successfully. Generating AFCA training masks...")

    # 3. Create a dictionary to easily find image info by its ID
    images_info = {img['id']: img for img in data['images']}
    success_count = 0

    # 4. Loop through every image that was annotated
    for img_id, img_info in images_info.items():
        filename = img_info['file_name']
        width = img_info['width']
        height = img_info['height']

        # Create a pitch-black canvas of the exact same size as the ultrasound
        mask = np.zeros((height, width), dtype=np.uint8)

        # 5. Find all the follicle annotations drawn on this specific image
        annotations = [ann for ann in data['annotations'] if ann['image_id'] == img_id]

        # 6. Draw each polygon onto the black canvas
        for ann in annotations:
            for segmentation in ann['segmentation']:
                # Convert the flat list of coordinates into pairs [x, y]
                poly = np.array(segmentation, dtype=np.int32).reshape((-1, 2))
                # Fill the polygon with solid white (255)
                cv2.fillPoly(mask, [poly], 255)

        # 7. Save the final mask
        # We save as .png to prevent compression from blurring your crisp edges
        mask_filename = os.path.splitext(filename)[0] + "_mask.png"
        output_path = os.path.join(output_dir, mask_filename)
        cv2.imwrite(output_path, mask)
        success_count += 1

    print(f"Success! {success_count} binary masks have been saved to the '{output_dir}' folder.")

if __name__ == "__main__":
    create_masks_from_json()