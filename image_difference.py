import cv2
import numpy as np

def align_images(images):
    # Initialize the ORB feature detector
    orb = cv2.ORB_create()

    # Initialize lists to store keypoints and descriptors for each image
    keypoints_list = []
    descriptors_list = []

    # Detect keypoints and compute descriptors for each image
    for image_path in images:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)

    # Match keypoints and find the homography transformation
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors_list[0], descriptors_list[1])
    src_pts = np.float32([keypoints_list[0][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_list[1][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Align the second image to the first image
    aligned_image = cv2.warpPerspective(cv2.imread(images[1]), M, (images[0].shape[1], images[0].shape[0]))

    return aligned_image

def image_difference(images):
    # Align the images
    aligned_image = align_images(images)

    # Convert the aligned image to grayscale
    aligned_gray = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY)

    # Load the first image in the stack
    reference_image = cv2.imread(images[0])

    # Convert the reference image to grayscale
    reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    # Compute the absolute difference between the reference and aligned image
    difference = cv2.absdiff(reference_gray, aligned_gray)

    # Apply a threshold to obtain a binary image of changes
    _, thresholded = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)

    # Combine the thresholded image with the reference image
    output_image = cv2.bitwise_and(reference_image, reference_image, mask=thresholded)

    return output_image

# Provide a list of image file paths in the stack
image_stack = ["image1.jpg", "image2.jpg", "image3.jpg"]

# Call the image_difference function to obtain the output image
output = image_difference(image_stack)

# Save the output image
cv2.imwrite("output.jpg", output)
