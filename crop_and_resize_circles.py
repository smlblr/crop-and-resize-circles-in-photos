import os
import sys
import cv2
import numpy as np

# Show opencv windows
SHOW_WINDOWS = False

# Max radius of circles to be detected by opencv via HoughCircle method
MAX_RADIUS_OF_CIRCLES_TO_BE_DETECTED = 500

# Screen resolution (ex. 1920x1080)
SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080

# Max window resolution to be wanted
MAX_WINDOW_WIDTH, MAX_WINDOW_HEIGHT = SCREEN_WIDTH * 3 // 4, SCREEN_HEIGHT

# Calculate ratio for re-resolution
def calculate_ratio(max_window_width, max_window_height, img_width, img_height):
    if img_width > max_window_width or img_height > max_window_height:
        ratio = min(max_window_width / img_width, max_window_height / img_height)
    else:
        ratio = 1.0
    return ratio

# Resize window in order to overflowing screen
def resize_window_and_show(src, src_variable_name):
    print(f"{src_variable_name}: {src}")
    height, width = src.shape[0], src.shape[1]
    ratio = calculate_ratio(MAX_WINDOW_WIDTH, MAX_WINDOW_HEIGHT, width, height)
    cv2.namedWindow(src_variable_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(src_variable_name, int(width * ratio), int(height * ratio))
    cv2.imshow(src_variable_name, src)
    cv2.waitKey(0)


def main(argv):

    # Check if enough arguments are provided
    if len(argv) < 2:
        print("Usage: python removebg_crop_and_resize_opencv.py src_folder dest_folder [extension] [compression] [resolutions]")
        print("Example: python removebg_crop_and_resize_opencv.py 'src/folder' 'dest/folder' png 9 [1000, 500, 250]")
        return -1
    
    print("Source folder: " + argv[0])
    print("Destination folder: " + argv[1])
    
    # Set source and destination folders
    src_folder = argv[0]
    dest_folder = argv[1]

    # Check if source folder is a valid directory
    if not os.path.isdir(src_folder):
        print("Source folder is not a valid directory.")
        return -1

    # Define valid extensions
    valid_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

    # Set extension to provided value or default to .png
    if len(argv) > 2:
        ext = argv[2]
        if not ext.startswith("."):
            ext = "." + ext
        if ext not in valid_extensions:
            print("Invalid extension: " + ext)
            print("Valid extensions: " + ", ".join(valid_extensions))
            return -1
    else:
        ext = ".png"

    # Set compression to provided value or default to 0
    if len(argv) > 3:
        try:
            compression = int(argv[3])
        except ValueError:
            print("Compression value must be an integer between 0 and 9")
            return -1
    else:
        compression = 0

    # Set resolutions to provided value or default to 1000
    if len(argv) > 4:
        try:
            resolutions = [int(x) for x in argv[4].strip("[]").split(",")]
        except ValueError:
            print("Resolutions must be a list of integers")
            return -1
    else:
        resolutions = [1000]

    # Print settings
    print("Extension: " + ext)
    print("Compression: " + str(compression))
    print("Resolutions: " + str(resolutions))

    # Create destination folder if it doesn't exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    # Initialize counter
    count = 0

    # Loop through all files in source folder and subfolders
    for root, dirs, files in os.walk(src_folder):
        for file_name in files:
            # Get filename without extension
            name_only, ext_org = os.path.splitext(file_name)

            # Load image
            src = cv2.imread(os.path.join(root, file_name))
            if src is None:
                print("Could not open or find the image: ", file_name)
                continue

            if SHOW_WINDOWS:
                resize_window_and_show(src, f'{src=}'.split('=')[0])
            
            # Increment counter
            count += 1

            # Print original file path
            print(f"{count} - Original File: {os.path.join(root, file_name)}")

            # Convert to grayscale
            gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            if SHOW_WINDOWS:
                resize_window_and_show(gray, f'{gray=}'.split('=')[0])
            # Apply median blur to determine the boundaries more accurately
            gray = cv2.medianBlur(gray, 5)
            if SHOW_WINDOWS:
                resize_window_and_show(gray, f'{gray=}'.split('=')[0])
            
            # Detect circles using Hough transform
            rows = gray.shape[0]
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=rows,
                                    param1=400, param2=1,
                                    minRadius=100, maxRadius=MAX_RADIUS_OF_CIRCLES_TO_BE_DETECTED-1)
            
            # Create a copy of the original image to draw circles on
            src_circle = src.copy()

            # If circles are detected, process each one
            if circles is not None:
                circles = np.uint16(np.around(circles))
                circle_count = 0
                for i in circles[0, :]:
                    circle_count += 1
                    center = (i[0], i[1])
                    print(f"Circle {circle_count} - Center: {center}, Radius: {i[2]}")

                    # # Comment to draw all circles (too small, too far left or right, too far up and down) on original image
                    # # Skip circles that are too small or too large
                    # if i[2] < 150:
                    #     print("Circle is too low, skipping...")
                    #     circle_count -= 1
                    #     continue

                    # # Skip circles that are too far left or right
                    # if i[0] < 500 or i[0] > 600:
                    #     print("Circle is too far left or right, skipping...")
                    #     circle_count -= 1
                    #     continue

                    # # Skip circles that are too far up and down
                    # if i[1] < 500 or i[1] > 600:
                    #     print("Circle is too far up and down, skipping...")
                    #     circle_count -= 1
                    #     continue
                    
                    # circle outline mask
                    radius = i[2]
                    mask = np.zeros_like(src)
                    mask = cv2.circle(mask, center, radius, (255,255,255), -1)
                    if SHOW_WINDOWS:
                        resize_window_and_show(mask, f'{mask=}'.split('=')[0])
                    
                    # put the mask into the alpha channel
                    src_circle = cv2.cvtColor(src, cv2.COLOR_BGR2BGRA)
                    src_circle[:, :, 3] = mask[:,:,0]
                    if SHOW_WINDOWS:
                        resize_window_and_show(src_circle, f'{src_circle=}'.split('=')[0])

                    # Crop image to circle
                    src_circle_cropped = src_circle[i[1]-i[2]:i[1]+i[2], i[0]-i[2]:i[0]+i[2]]
                    if SHOW_WINDOWS:
                        resize_window_and_show(src_circle_cropped, f'{src_circle_cropped=}'.split('=')[0])
                    
                    # Grayscale cropped image
                    src_circle_cropped_grayscale = cv2.cvtColor(src_circle_cropped, cv2.COLOR_BGR2GRAY)
                    if SHOW_WINDOWS:
                        resize_window_and_show(src_circle_cropped_grayscale, f'{src_circle_cropped_grayscale=}'.split('=')[0])
                    
                    # Create new file name if more than one circle is detected
                    if circle_count == 1:
                        new_file_name = name_only + "_" + "large" + ext
                    else:
                        new_file_name = name_only + "_" + str(circle_count) + "_" + "large" + ext
                    
                    # Create destination path
                    rel_path = os.path.relpath(root, src_folder)
                    dest_path = os.path.join(dest_folder, rel_path, new_file_name)
                    dest_dir = os.path.dirname(dest_path)

                    # Create destination directory if it doesn't exist
                    if not os.path.exists(dest_dir):
                        os.makedirs(dest_dir, exist_ok=True)

                    try:
                        # Write as full quality png
                            cv2.imwrite(dest_path, src_circle_cropped, [cv2.IMWRITE_PNG_COMPRESSION, compression])
                    except:
                        print("Could not write image: ", dest_path)
                        print("--------------------------------------------------")
                        circle_count -= 1
                        continue

                    # Resize image to specified resolutions and save
                    for res in resolutions:
                        src_circle_cropped_resized = cv2.resize(src_circle_cropped, (res, res))
                        # Create new file name if more than one circle is detected
                        if circle_count == 1:
                            new_file_name = name_only + "_" + str(res) + "x" + str(res) + ext
                        else:
                            new_file_name = name_only + "_" + str(circle_count) + "_" + str(res) + "x" + str(res) + ext
                        
                        dest_path = os.path.join(dest_folder, rel_path, new_file_name)

                        try:
                            # Write as full quality png
                            cv2.imwrite(dest_path, src_circle_cropped_resized, [cv2.IMWRITE_PNG_COMPRESSION, compression])
                        except:
                            print("Could not write image: ", dest_path)
                            print("------------------------------------------------")
                            circle_count -= 1
                            break

                    # Print destination file path
                    print(f"{count} - Removed BG and Cropped File: {dest_path}")

                    # Separator line
                    print("--------------------------------------------------")
    
    return 0
if __name__ == "__main__":
    main(sys.argv[1:])
