/* 
 * File:   MatchTemplate_1.cpp
 * Author: essam
 *
 * Created on August 16, 2015, 11:41 AM
 */

#include <cstdlib>
#include <iostream>
#include <dirent.h>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types_c.h>

using namespace std;

void getFilesInDirectory(vector<string>& out, string directory) {
    DIR* dir;
    dirent* pdir;

    dir = opendir(directory.c_str()); // open current directory

    while (pdir = readdir(dir)) {
        string file(pdir->d_name);
        if (file.compare(".") == 0 || file.compare("..") == 0) {
            continue;
        }

        out.push_back(pdir->d_name);
        //        cout << pdir->d_name << endl;
    }
    closedir(dir);
}

int main(int argc, char** argv) {
    cv::Mat ref = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR | cv::IMREAD_ANYDEPTH);
    if (ref.empty())
        return -1;

    cv::Mat gref;
    cv::cvtColor(ref, gref, CV_BGR2GRAY);

    vector<string> tpl_list;
    string dir = argv[2];
    getFilesInDirectory(tpl_list, dir);
    cout << "Number of files:\t" << tpl_list.size() << endl;

    for (vector<string>::size_type i = 0; i != tpl_list.size(); i++) {

        cout << i << " : " << tpl_list[i] << endl;
        string path = dir + "/" + tpl_list[i];
        cv::Mat tpl = cv::imread(path, CV_LOAD_IMAGE_COLOR | cv::IMREAD_ANYDEPTH);
        if (tpl.empty()) return -1;

        cv::Mat gtpl;
        cv::cvtColor(tpl, gtpl, CV_BGR2GRAY);

        cv::Mat res(ref.rows - tpl.rows + 1, ref.cols - tpl.cols + 1, CV_32FC1);
        cv::matchTemplate(gref, gtpl, res, CV_TM_CCOEFF_NORMED);
        cv::threshold(res, res, 0.7, 1.0, CV_THRESH_TOZERO);

        while (true) {
            double minval, maxval, threshold = 0.8;
            cv::Point minloc, maxloc;
            cv::minMaxLoc(res, &minval, &maxval, &minloc, &maxloc);

            if (maxval >= threshold) {
                cv::rectangle(
                        ref,
                        maxloc,
                        cv::Point(maxloc.x + tpl.cols, maxloc.y + tpl.rows),
                        CV_RGB(0, 255, 0), 2
                        );
                cv::floodFill(res, maxloc, cv::Scalar(0), 0, cv::Scalar(.1), cv::Scalar(1.));
            } else
                break;
        }


    }

    //    cv::Mat ref = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR | cv::IMREAD_ANYDEPTH);
    //    cv::Mat tpl = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR | cv::IMREAD_ANYDEPTH);
    //

    cv::imwrite("/home/essam/work/res.png", ref);

    return 0;
}
