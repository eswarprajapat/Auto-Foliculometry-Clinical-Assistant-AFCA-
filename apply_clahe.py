import cv2
import os
import glob


def enhance_ultrasound_images():
    # 1. Define your folders
    input_folder = "raw_images"
    output_folder = "enhanced_images"

    # 2. Create the output folder automatically if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # 3. Configure the CLAHE algorithm
    # clipLimit controls the contrast limit. 3.0 is aggressive and great for ultrasound speckle.
    # tileGridSize divides the image into an 8x8 grid to enhance local contrast without blowing out the whole image.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    # 4. Find all the .jpg images in your raw folder
    image_paths = glob.glob(os.path.join(input_folder, "*.jpg"))

    if len(image_paths) == 0:
        print("No images found in the 'raw_images' folder. Please check your files.")
        return

    print(f"Found {len(image_paths)} images. Starting CLAHE enhancement...")

    # 5. Loop through every single image
    success_count = 0
    for img_path in image_paths:
        # Read the image strictly in Grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Skip if the image is corrupted or can't be read
        if img is None:
            print(f"Error reading {img_path}. Skipping.")
            continue

        # Apply the mathematical CLAHE enhancement
        enhanced_img = clahe.apply(img)

        # Save the new high-contrast image to the output folder
        filename = os.path.basename(img_path)
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, enhanced_img)

        success_count += 1

    print(f"Success! {success_count} images have been enhanced and saved to the '{output_folder}' folder.")


# Run the function
if __name__ == "__main__":
    enhance_ultrasound_images()