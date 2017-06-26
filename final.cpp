#include <iostream> // for standard I/O
#include <string>   // for strings
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>     // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc.hpp>  // Gaussian Blur
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>  // OpenCV window I/O

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>

#include <opencv2/cvconfig.h>

using namespace std;
using namespace cv;


Mat src, src_gray;
Mat dst, dst_norm, dst_norm_scaled;

Mat src2, src2_gray;
Mat dst2, dst2_norm, dst2_norm_scaled;

int blockSize=2;
int apertureSize=3;
double k=0.04;
int thresh = 200;
int max_thresh = 255;

vector< pair<float, float> > v1;
vector< pair<float, float> > v2;

static void help()
{
    cout
        << "------------------------------------------------------------------------------" << endl
        << "This program shows how to read a video file with OpenCV. In addition, it "
        << "tests the similarity of two input videos first with PSNR, and for the frames "
        << "below a PSNR trigger value, also with MSSIM."                                   << endl
        << "Usage:"                                                                         << endl
        << "./video-input-psnr-ssim <referenceVideo> <useCaseTestVideo> <PSNR_Trigger_Value> <Wait_Between_Frames> " << endl
        << "--------------------------------------------------------------------------"     << endl
        << endl;
}

int main(int argc, char *argv[])
{
    help();

    if (argc != 5)
    {
        cout << "Not enough parameters" << endl;
        return -1;
    }
	
    v1.reserve(100);
    v2.reserve(100);
    stringstream conv;

    const string sourceReference = argv[1], sourceCompareWith = argv[2];
    int psnrTriggerValue, delay;
    conv << argv[3] << endl << argv[4];       // put in the strings
    conv >> psnrTriggerValue >> delay;        // take out the numbers

    int frameNum = -1;          // Frame counter
    int frameMatch =0;
    int MatchFlag = 1;
    VideoCapture captRefrnc(sourceReference), captUndTst(sourceCompareWith);

    if (!captRefrnc.isOpened())
    {
        cout  << "Could not open reference " << sourceReference << endl;
        return -1;
    }

    if (!captUndTst.isOpened())
    {
        cout  << "Could not open case test " << sourceCompareWith << endl;
        return -1;
    }

    Size refS = Size((int) captRefrnc.get(CAP_PROP_FRAME_WIDTH),
                     (int) captRefrnc.get(CAP_PROP_FRAME_HEIGHT)),
         uTSi = Size((int) captUndTst.get(CAP_PROP_FRAME_WIDTH),
                     (int) captUndTst.get(CAP_PROP_FRAME_HEIGHT));

    
    const char* WIN_UT = "Under Test";
    const char* WIN_RF = "Reference";

    // Windows
    namedWindow(WIN_RF, WINDOW_AUTOSIZE);
    namedWindow(WIN_UT, WINDOW_AUTOSIZE);
    moveWindow(WIN_RF, 400       , 0);         //750,  2 (bernat =0)
    moveWindow(WIN_UT, refS.width+40, 0);         //1500, 2

    cout << "Reference frame resolution: Width=" << refS.width << "  Height=" << refS.height
         << " of nr#: " << captRefrnc.get(CAP_PROP_FRAME_COUNT) << endl;

    cout << "PSNR trigger value " << setiosflags(ios::fixed) << setprecision(3)
         << psnrTriggerValue << endl;

    Mat frameReference, frameUnderTest, frameSizeChange, temp;
    double psnrV;
    Scalar mssimV;

    for(;;) //Show the image captured in the window and repeat
    {
        captRefrnc >> frameReference;
	captUndTst >> frameUnderTest;

        if (frameReference.empty() || frameUnderTest.empty())
        {
            cout << " < < <  Game over!  > > > ";
            break;
        }

	//+++++++++++++++++++++++++DIFFERENT SIZE+++++++++++++++++++++++++++++=
	if(frameReference.cols != frameUnderTest.cols && frameReference.rows != frameUnderTest.rows){
		
		cv::resize( frameUnderTest, frameSizeChange, cv::Size(frameReference.cols,frameReference.rows), 0,0, CV_INTER_NN);
	}
	else{
		frameSizeChange = frameUnderTest.clone();
	} 
		++frameNum;	

		cvtColor(frameReference, src_gray, COLOR_BGR2GRAY);
		cvtColor(frameSizeChange, src2_gray, COLOR_BGR2GRAY);

		dst = Mat::zeros( frameReference.size(), CV_32FC1);
		dst2 = Mat::zeros( frameReference.size(), CV_32FC1);



 		cv::Ptr<cv::cuda::CornernessCriteria> harris = cv::cuda::createHarrisCorner( src_gray.type(),blockSize, apertureSize, k, BORDER_REFLECT101 );

                cv::Ptr<cv::cuda::CornernessCriteria> harris2 = cv::cuda::createHarrisCorner( src2_gray.type(),blockSize, apertureSize, k, BORDER_REFLECT101 );

                cv::cuda::GpuMat cuda_dst;
                cv::cuda::GpuMat cuda_dst2;
                cv::cuda::GpuMat cuda_src;
                cv::cuda::GpuMat cuda_src2;

                cuda_src.upload(src_gray);
                cuda_src2.upload(src2_gray);

                harris->compute( cuda_src, cuda_dst);
                harris2->compute( cuda_src2, cuda_dst2);



                cuda_dst.download(dst);
                cuda_dst2.download(dst2_norm);

		cornerHarris( src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT );
		cornerHarris( src2_gray, dst2, blockSize, apertureSize, k, BORDER_DEFAULT );
			
		normalize ( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
		normalize ( dst2, dst2_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );


		convertScaleAbs( dst_norm, dst_norm_scaled );
		convertScaleAbs( dst2_norm, dst2_norm_scaled);


		int count=0;
		int count2=0;
 		/// Drawing a circle around corners
  		for( int j = 0; j < dst_norm.rows ; j++ )
     		{ 
			for( int i = 0; i < dst_norm.cols; i++ )
          		{       
                 		if( (int) dst_norm.at<float>(j,i) > thresh )
              			{
              				 circle( frameReference, Point( i, j ), 5,  Scalar(0), 2, 8, 0 );
              				 //circle( temp, Point( i, j ), 5,  Scalar(0), 2, 8, 0 );
					 v1.push_back(make_pair(i,j));
					count++;
              			}
          		}
     		}


 		/// Drawing a circle around corners
  		for( int j = 0; j < dst2_norm.rows ; j++ )
     		{ 
			for( int i = 0; i < dst2_norm.cols; i++ )
          		{       
                 		if( (int) dst2_norm.at<float>(j,i) > thresh )
              			{
              				 circle( frameSizeChange, Point( i, j ), 5,  Scalar(0), 2, 8, 0 );
              				 //circle( frameSizeChange, Point( i, j ), 5,  Scalar(0), 2, 8, 0 );
					 v2.push_back(make_pair(i,j));
					count2++;
              			}
          		}
     		}
		
		if(v1.size() == v2.size() ){

			if(v1==v2)
				frameMatch++;


		}


		v1.clear();
		v2.clear();
		//imshow(WIN_RF, dst_norm_scaled);
	        //imshow(WIN_UT, dst2_norm_scaled);
	
		imshow(WIN_RF, frameReference);
		imshow(WIN_UT, frameSizeChange);
		


	        char c = (char)waitKey(delay);
	        if (c == 27) break;

    }
	cout << endl;
	cout << "match frame : " << frameMatch << endl;
	cout << "Total Frame : " << frameNum  << endl;

	float  percentage = (float)frameMatch/(float)frameNum * 100;
	cout << " Match Percentage " << percentage << "%" << endl;
	

    return 0;
}


