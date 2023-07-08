#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Computer Vision 2023 

// Function for equalization used in previous homework
Mat equalization(Mat src) {
    Mat hist_equalized_image;
    cvtColor(src, hist_equalized_image, COLOR_BGR2YCrCb);

    vector<Mat> vec_channels;
    split(hist_equalized_image, vec_channels);
    equalizeHist(vec_channels[0], vec_channels[0]);
    merge(vec_channels, hist_equalized_image);
    cvtColor(hist_equalized_image, hist_equalized_image, COLOR_YCrCb2BGR);

    return hist_equalized_image;
}

//extractSiftFeaturesFromPatches is a function to extract Sift Features From all the Patches
void extractSiftFeaturesFromPatches(const vector<Mat>& patches, Ptr<Feature2D> detector, Ptr<DescriptorExtractor> extractor,
    vector<vector<KeyPoint>>& keypoints, vector<Mat>& descriptors)
{
    for (int i = 0; i < patches.size(); i++) {
        Mat patch = patches[i];

        // Detect keypoints and compute descriptors for the patch
        vector<KeyPoint> patchKeypoints;
        Mat patchDescriptors;
        detector->detectAndCompute(patch, Mat(), patchKeypoints, patchDescriptors);

        // Store the keypoints and descriptors for the patch
        keypoints.push_back(patchKeypoints);
        descriptors.push_back(patchDescriptors);

        // Display keypoints on the patches
        Mat patchWithKeypoints;
        drawKeypoints(patch, patchKeypoints, patchWithKeypoints);
        String titre = "Patch keypoints number " + to_string(i + 1) ;
        namedWindow(titre, WINDOW_NORMAL);
        imshow(titre, patchWithKeypoints);

    }
}





//_________________________________________ Main__________________________________________________________ //
//________________________________________________________________________________________________________ //

int main()
{

    // Load the corrupted images( just to see all the images )
    /*
    String path_corruptedImage = "C:/Users/21629/Desktop/cours/S2/computer_vision/projet/HW2/image_to_complete*.jpg";  // This path contains all image_to_complete
    // Initialize vectors 
    vector<Mat> lists_corruptedImage;  // I made a list of lists_corruptedImage to see all the image_to_complete with the keypoints, but after I worked with one example
    vector<string> f_corruptedImage;
    glob(path_corruptedImage, f_corruptedImage, true);
    for (size_t i = 0; i < f_corruptedImage.size(); i++) {
        Mat corruptedImage = imread(f_corruptedImage[i]);
        if (corruptedImage.empty()) {
            cout << "Failed to load the corrupted image" << endl;
            return -1;
        }
        lists_corruptedImage.push_back(corruptedImage);
    }
    */

    //_________________________________________ Step 1: Load Images ________________________________________ //


    // Load the corrupted images

    Mat corruptedImage = imread("C:/Users/21629/Desktop/cours/S2/computer_vision/projet/HW2/pratodellavalle/pratodellavalle/image_to_complete.jpg");
    
    //-- Tying my images 
   // Mat corruptedImage = imread("C:/Users/21629/Desktop/cours/S2/computer_vision/projet/HW2/supcom/image_to_complete.jpg");

    // Load the patches
    vector<Mat> patches;
    String path_patches = "C:/Users/21629/Desktop/cours/S2/computer_vision/projet/HW2/pratodellavalle/pratodellavalle/patch_*.jpg"; // This path contains all the patches for the venezia corruptedImage

    //String path_patches = "C:/Users/21629/Desktop/cours/S2/computer_vision/projet/HW2/supcom/patch_*.jpg"; // This path contains all the patches for the venezia corruptedImage
    vector<String> f_patches;
    glob(path_patches, f_patches, true);
    for (size_t i = 0; i < f_patches.size(); ++i) {
        Mat im_patches = imread(f_patches[i]);
        if (im_patches.empty())
            continue;
        // Apply equalization function to the patches
        Mat equalizedPatch = equalization(im_patches);

        // Apply Gaussian blur to the equalized result , to enhance the quality of patches 
        Mat blurredPatch;
        GaussianBlur(equalizedPatch, blurredPatch, Size(0, 0), 2.0);

        // Initialize the sharpened patch for pre-processing
        Mat sharpenedPatch = equalizedPatch.clone();

        // Apply sharpening 
        addWeighted(equalizedPatch, 1.5, blurredPatch, -0.5, 0, sharpenedPatch);
        patches.push_back(sharpenedPatch);

        //displaying patches 
        //imshow("Patch " + to_string(i), patches[i]);
        //waitKey(0); 
    }
    //_________________________________________Preprocessing for images and patches ________________________________________ //


    // For the pre-processing, I do equalization and I notice some patches are blurred (applying the sharpening operation on patches 


    Mat equalizedcorruptedImage = equalization(corruptedImage);



    //_________________________________________ Step 2 : Extract SIFT features from the Corrupted image ________________________________________ //

    // Create SIFT detector and descriptor extractor 
    Ptr<SIFT> detector = SIFT::create();
    Ptr<SIFT> extractor = SIFT::create();

    vector<KeyPoint> keypoints;
    detector->detect(equalizedcorruptedImage, keypoints);
    Mat descriptors;  // will be used to compute the matching patches_image
    extractor->compute(equalizedcorruptedImage, keypoints, descriptors);
    Mat corruptedImage_keypoints;
    drawKeypoints(equalizedcorruptedImage, keypoints, corruptedImage_keypoints);
    namedWindow("Corrupted Image with Keypoints", WINDOW_AUTOSIZE);
    imshow("Corrupted Image with Keypoints", corruptedImage_keypoints);


     //_________________________________________ Step 3 : Extract SIFT features from the patches ________________________________________ //


    // Define the input vectors for the extractSiftFeaturesFromPatches function
    vector<vector<KeyPoint>> patches_keypoints;
    vector<Mat> patches_des;  // this will be used for step 4 

    // call the extractSiftFeaturesFromPatches function 
    extractSiftFeaturesFromPatches(patches, detector, extractor, patches_keypoints, patches_des);



    //--------------------------------- Extra : ORB ---------------------------//

// Start by creating the ORB object
    Ptr<ORB> orb = ORB::create(7000, 1.05f, 8, 10);

    // Perform ORB feature extraction on the Corrupted image
    vector<KeyPoint> orb_keypoints_corruptedImage;
    Mat orb_descriptors_corruptedImage;
    Mat corruptedImage_with_ORB_keypoints;
    orb->detectAndCompute(equalizedcorruptedImage, Mat(), orb_keypoints_corruptedImage, orb_descriptors_corruptedImage);
    drawKeypoints(equalizedcorruptedImage, orb_keypoints_corruptedImage, corruptedImage_with_ORB_keypoints);
    namedWindow("Corrupted Image with ORB feature extraction", WINDOW_AUTOSIZE);
    imshow("Corrupted Image with ORB feature extraction", corruptedImage_with_ORB_keypoints);

    // Perform ORB feature extraction on the patches
    vector<vector<KeyPoint>> orb_patches_keypoints;
    vector<Mat> orb_patches_descriptors;

    for (int j = 0; j < patches.size(); j++) {
        vector<KeyPoint> orb_keypoints;
        Mat patch_res; // the final image of patches 
        Mat orb_descriptors;
        orb->detectAndCompute(patches[j], Mat(), orb_keypoints, orb_descriptors);
        orb_patches_keypoints.push_back(orb_keypoints);
        orb_patches_descriptors.push_back(orb_descriptors);
        drawKeypoints(patches[j], orb_keypoints, patch_res);
        String titre = "Patch With ORB number " + to_string(j);
        imshow(titre, patch_res);
    }




    //_________________________________________ Step 4 - a: Compute match between image and patch features ________________________________________ //
    
    vector<vector<DMatch>> matches_vector; // a DMatch vector to use for refine 



    //--
    BFMatcher matcher(NORM_L2);  // Create a cv::BFMatcher matcher to compute matches using L2 distance

    for (int k = 0; k < patches.size(); k++)
    {
        vector<DMatch> new_matches;   //defining the Dmatch vector for the new refine matches 

        matcher.match(descriptors, patches_des[k], new_matches);  // Match the descriptors of the image and patch

        sort(new_matches.begin(), new_matches.end(), [](const DMatch& x, const DMatch& y)
        {
            return x.distance < y.distance;
      });

        Mat Image_with_Matchers;
        drawMatches(equalizedcorruptedImage, keypoints, patches[k], patches_keypoints[k],
            vector<DMatch>(new_matches.begin(), new_matches.begin() + 20), Image_with_Matchers);

        //Displaying top matches using i choose to display the first 20 matches of each patch and the corr_image
        String titre = " Matches between patches and the corrupted Image number " + to_string(k) ;
        namedWindow(titre, WINDOW_NORMAL);
        //Let's adjust the window size 
        resizeWindow(titre, 800, 600);  
        moveWindow(titre, 150, 150);  
        imshow(titre, Image_with_Matchers);


        


        //_________________________________________ Step 4 - b: Refine the matches ________________________________________ //
        

        // refine the matches based on distance ratio
        vector<DMatch> refineMatches;
        float ratio = 2;  // defining the distance ratio , i start with 0.8 but i m getting an error that the destinationPoints and SourcePoints 
        // is under 4 , so i can't calculate the Homography matrix , thats why i try to get a high ratio
        
        //This part i get it from stackoverflow because i was getting lots of errors 
        for (int i = 0; i < new_matches.size(); i++)
        {
            if (new_matches[i].distance <= (new_matches[0].distance * ratio))
            {
                refineMatches.push_back(new_matches[i]);
            }
        }

        matches_vector.push_back(refineMatches);

        // Display filtered matches using drawMatches
        Mat new_Image_with_Matchers;
        drawMatches(equalizedcorruptedImage, keypoints, patches[k], patches_keypoints[k],refineMatches, new_Image_with_Matchers);

       //Displaying 
        String titre1 = " The new matches " + to_string(k);
        namedWindow(titre1, WINDOW_NORMAL);
        resizeWindow(titre1, 800, 600);
        moveWindow(titre1, 150, 150);
        imshow(titre1, new_Image_with_Matchers);


        //// I m trying here to display the refined matches
            /*
            for (size_t i = 0; i < refineMatches.size(); i++) {
                cout << "match " << i + 1 << ":" << endl;
                cout << "distance: " << refineMatches[i].distance << endl;
                cout << "Query : " << refineMatches[i].queryIdx << endl;
                cout << "Train : " << refineMatches[i].trainIdx << endl;
                cout << "//------------Next Match------------//" << endl;
            }
          */  








        //_________________________________________ Step 5 :use the RANSAC algorithm ________________________________________ //
        
        vector<Point2f> source;
        vector<Point2f> destination;
        for (const DMatch& match : refineMatches)
        {
            source.push_back(keypoints[match.queryIdx].pt);
            destination.push_back(patches_keypoints[k][match.trainIdx].pt);
        }

        
        //cout << "The source points size: " << source.size() << endl;
        //cout << "The destination points size: " << destination.size() << endl;

        /*
        if (source.size() != destination.size()) {
            cout << "Error: srcPoints and dstPoints should have the same size." << endl;
            return -1;
        }
        */

        Mat homography_matrix = findHomography(destination, source, RANSAC);
        cout << "Homography Matrix:\n" << homography_matrix << endl;






        //_________________________________________ Step 6 :overlay the patches over the image  ________________________________________ //

        

        Mat Overlay_Corrupted_Image;
        warpPerspective(patches[k], Overlay_Corrupted_Image, homography_matrix, equalizedcorruptedImage.size());
        Overlay_Corrupted_Image.copyTo(equalizedcorruptedImage, Overlay_Corrupted_Image > 0);



        //Displaying 
        String titre2 = "The overlay image number" + to_string(k);

        namedWindow(titre2, WINDOW_NORMAL);
        resizeWindow(titre2, 800, 600);
        moveWindow(titre2, 150, 150);
        imshow(titre2, equalizedcorruptedImage);

        
        
    }

    
    cv::waitKey(0);
    destroyAllWindows();

    return 0;
}
