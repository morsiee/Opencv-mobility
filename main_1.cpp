/* 
 * File:   main_1.cpp
 * Author: essam
 *
 * Created on July 13, 2015, 2:34 PM
 */

#include <cstdlib>
#include<iostream>
#include <cv.h>
#include <opencv/highgui.h>
#include<opencv/cv.h>
#include <opencv/cv.hpp>

//#include<opencv/cv.hpp>

using namespace cv;
using namespace std;

/*
 * 
 */
int main(int argc, char** argv) {
if (argc != 2) {
        cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
        return -1;
    }

    Mat image, mask;
    image = imread(argv[1], CV_LOAD_IMAGE_COLOR); // Read the file

    if (!image.data) // Check for invalid input
    {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    Size size = image.size.operator ()();
    cout << "image info\t" << " size: " << size.width << ":" << size.height << "\t type:\t" << image.type() << endl;
    
    namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
    imshow("Display window", image); // Show our image inside it.

    waitKey(0); // Wait for a keystroke in the window
    return 0;
}

