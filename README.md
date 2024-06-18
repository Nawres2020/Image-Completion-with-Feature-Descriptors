# Image-Completion-with-Feature-Descriptors
Objective
The objective of this project is to develop an algorithm that can automatically find and correct missing or corrupted areas in an input image. The algorithm utilizes feature descriptors, specifically SIFT (Scale-Invariant Feature Transform), to identify the correct patches and their locations in the image. By utilizing a set of provided patches that contain the missing information, the algorithm performs image restoration by replacing the missing or corrupted areas with the appropriate patches.




Step 1: Load Images
The first step is to load the images. This project involves two types of images: corrupted images with missing parts and patches containing the missing information of the corrupted images.

Preprocessing images
Before proceeding with other steps, the images undergo preprocessing. This includes applying equalization to the corrupted images to avoid color jumps and enhance some blurred images. Gaussian blur and sharpening operations are also applied to the patches to further enhance them.

Step 2: Extract SIFT Features from the image
SIFT feature extraction is performed on the corrupted image. The steps include creating a SIFT detector and descriptor extractor, detecting keypoints in the equalized corrupted image, computing descriptors for the detected keypoints, and drawing the keypoints on the equalized corrupted image.

Step 3: Extract SIFT Features from the patches
SIFT features are also extracted from the patches. This step involves creating a function called "extractSiftFeaturesFromPatches" that iterates through the patches, detects keypoints, computes descriptors, and displays the images with keypoints and their descriptors. The function is called twice, once for the corrupted images and once for the patches.

Compute the match between Images Using ORB
ORB (Oriented FAST and Rotated BRIEF) feature extraction is performed on both the corrupted image and each patch individually. The ORB keypoints and descriptors are computed, and the matches between the image and patch features are obtained.

Step 4:
a. Compute the match between Images Using SIFT
After extracting SIFT features from the corrupted images and patches, the matches between the image and patch features are computed. A Brute-Force matcher object is created to compute matches based on the L2 distance metric. The matches are sorted based on their distance, and the top matches are displayed.

b. Refine the matches
To improve the matches, a refinement process is performed. Matches with a distance less than a user-defined threshold (ratio * min_distance) are selected. The refined matches are stored and displayed.

Step 5: RANSAC algorithm
The RANSAC (Random Sample Consensus) algorithm is employed to calculate the affine transformation matrix between the images and patches. The findHomography() function with RANSAC is used to obtain the set of inliers. The corresponding points from the refined matches are used to calculate the homography matrix.

Step 6: Overlay the patches over the image
Using the calculated homography matrix, the patches are overlaid onto the image to correct the corrupted regions. By warping the patches using the homography matrix and blending them with the original image, the missing or corrupted areas are seamlessly replaced.

Conclusion
In this project, an algorithm was developed to automatically find and correct missing or corrupted areas in an input image using feature descriptors. By utilizing SIFT features, matches between the image and patches were computed and refined. The RANSAC algorithm was employed to calculate the homography matrix, which allowed for precise alignment and overlaying of the patches onto the image. The algorithm successfully performed image completion by replacing the missing or corrupted areas with the appropriate patches.

[Project_Nawres_Atide_Hamrouni.pdf](https://github.com/user-attachments/files/15890102/Project_Nawres_Atide_Hamrouni.pdf)
