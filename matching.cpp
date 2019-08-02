#include "DBoW3/DBoW3.h"
#include "orbextractor.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/timer.hpp>

using namespace std;
using namespace cv;
const int N = 75;

int main(int argc, char** argv)
{
    if ( argc != 3 )
    {
        cout<<"usage: feature_extraction img1"<<endl;
        return 1;
    }
    Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_UNCHANGED);
    Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_UNCHANGED);
    //初始化
    vector<KeyPoint> keypoints_1, keypoints_2,keypoint1choose,keypoint2choose;
    Mat descriptors_1, descriptors_2,descriptor1choose,descriptor2choose;
    //使用ORB-SLAM2优化过的特征点提取算法
    myslam::ORBextractor* detecter= new myslam::ORBextractor(100,1.2,8,20,7);

    (*detecter)(img_1,cv::Mat(),keypoints_1,descriptors_1);
    (*detecter)(img_2,cv::Mat(),keypoints_2,descriptors_2);
    
    //提取合适范围的关键点
    for(int i = 0; i<keypoints_1.size();i++)
    {
      if(keypoints_1[i].pt.x >= 80&&keypoints_1[i].pt.x <= 1150&&keypoints_1[i].pt.y >= 80&&keypoints_1[i].pt.y <= 290)
      {
	keypoint1choose.push_back(keypoints_1[i]);  
	descriptor1choose.push_back(descriptors_1.row(i));
      }
    }
    for(int i = 0; i<keypoints_2.size();i++)
    {
      if(keypoints_2[i].pt.x >= 80&&keypoints_2[i].pt.x <= 1150&&keypoints_2[i].pt.y >= 80&&keypoints_2[i].pt.y <= 290)
      {
	keypoint2choose.push_back(keypoints_2[i]);
	descriptor2choose.push_back(descriptors_2.row(i));
      }
    }
    
    cout<<"picture1: "<<keypoint1choose.size()<<endl;
    cout<<"picture2: "<<keypoint2choose.size()<<endl;
    //特征匹配
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );
    vector<DMatch> matches;
    matcher->match ( descriptor1choose, descriptor2choose, matches );
    
    cout<<"picture1 match: "<<keypoint1choose.size()<<endl;
    cout<<"picture2 match: "<<keypoint2choose.size()<<endl;
    cout<<"matches: "<<matches.size()<<endl;
    //载入字典及提取局部图像
    cout<<"reading database"<<endl;
    DBoW3::Vocabulary vocab("./vocabulary_kitti.yml.gz");
    if ( vocab.empty() )
    {
        cerr<<"Vocabulary does not exist."<<endl;
        return 1;
    }
    for(int i = 0; i<matches.size();i++)
    {
      float x1,x2,y1,y2;
      x1 = keypoint1choose[i].pt.x;
      y1 = keypoint1choose[i].pt.y;
      x2 = keypoint2choose[matches[i].trainIdx].pt.x;
      y2 = keypoint2choose[matches[i].trainIdx].pt.y;
      Mat imleft = img_1.rowRange(y1-N,y1+N+1).colRange(x1-N,x1+N+1);
      Mat imright = img_2.rowRange(y2-N,y2+N+1).colRange(x2-N,x2+N+1);
      Ptr< Feature2D > detector = ORB::create();
      Mat descriptorleft,descriptorright;
      vector<KeyPoint> keypointsleft,keypointsright;
      detector->detectAndCompute( imleft, Mat(), keypointsleft, descriptorleft );
      detector->detectAndCompute( imright, Mat(), keypointsright, descriptorright );
      DBoW3::BowVector v1;
      vocab.transform( descriptorleft, v1 );
      DBoW3::BowVector v2;
      vocab.transform( descriptorright, v2 );
      double score1 = vocab.score(v1, v2);
      cout<<"score: "<<score1<<endl;
      //imshow("left",imleft);
      //imshow("right",imright);
      //waitKey(0);
    }
    Mat img_match;
    drawMatches ( img_1, keypoint1choose, img_2, keypoint2choose, matches, img_match );
    imshow("matches",img_match);
    
    waitKey(0);
    return 0;
}