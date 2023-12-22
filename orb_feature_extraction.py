import cv2
import numpy as np
import urllib.request
import sys
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors

def download_image(url):
    """Download an image from a URL."""
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def resize_image(image, max_size=1000):
    """Resize the image to a maximum width or height."""
    h, w = image.shape[:2]
    scale = min(max_size / h, max_size / w)

    if scale < 1:
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    return image

def read_binary_visual_words(file_path):
    """Read visual words from a binary file."""
    try:
        with open(file_path, "rb") as file:
            data = file.read()
            if len(data) % 32 != 0:
                raise ValueError("File size is not a multiple of 32 (word size).")
            # Convert to an array of uint8
            visual_words = np.frombuffer(data, dtype=np.uint8).reshape(-1, 32)
    except Exception as e:
        print(f"Error reading visual words file: {e}")
        sys.exit(1)
    return visual_words


def match_features_to_visual_words(descriptors, visual_words):
    """Match ORB descriptors to visual words using k-NN."""
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(visual_words)
    distances, indices = nbrs.kneighbors(descriptors)
    return indices

def extract_and_visualize_features(image):
    """Extract ORB features, print them, and visualize them."""
    # Create an ORB detector with the same parameters as the C++ code
    orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.02, nlevels=100)
    # Detect keypoints and compute the descriptors
    keypoints, descriptors = orb.detectAndCompute(image, None)

    # Print the number of keypoints and descriptors
    print(f"Number of keypoints detected: {len(keypoints)}")
    print(f"Number of descriptors computed: {len(descriptors)}")

    # Print keypoints' information
    print("Keypoint Information:")
    for i, keypoint in enumerate(keypoints):
        print(f"Keypoint {i}: Position (x, y): ({keypoint.pt[0]}, {keypoint.pt[1]}), "
              f"Size: {keypoint.size}, Angle: {keypoint.angle}, Response: {keypoint.response}, "
              f"Octave: {keypoint.octave}, Class ID: {keypoint.class_id}")
        # Flush the output to make sure it's printed to the terminal immediately
        sys.stdout.flush()
    
    print("\nKeypoint Descriptors:")
    for i, descriptor in enumerate(descriptors):
        print(f"Descriptor {i}: {descriptor}")

    # Draw and display the keypoints
    img_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)
    plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.title(f"Number of features: {len(keypoints)}")
    plt.show()

    return keypoints, descriptors



def visualize_all_keypoints(image):
    """Visualize all ORB features without matching to visual words."""
    orb = cv2.ORB_create(nfeatures=2000)
    keypoints = orb.detect(image, None)
    
    img_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)
    plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.title(f"Total number of features: {len(keypoints)}")
    plt.show()

def main(url, visual_words_file):
    image = download_image(url)
    image = resize_image(image)
    
    extract_and_visualize_features(image)
    # Call the function to visualize all keypoints
    # visualize_all_keypoints(image)

    # visual_words = read_binary_visual_words(visual_words_file)
    # extract_and_visualize_features(image, visual_words)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <image_url> <visual_words_file>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
