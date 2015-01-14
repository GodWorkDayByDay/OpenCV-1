#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>
#include <opencv2/flann/kdtree_index.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>
#include<opencv2/nonfree/nonfree.hpp>
#include<opencv2/legacy/legacy.hpp>
#include<vector>
#include <math.h>
#include <iostream>
#include <iomanip>
using namespace std;
using namespace cv;
//#define BOUNDARY_VALUE 14	//标准差边界值
//#define MAX 20				//每行分组的数目
//#define IMAGESIZE 400		//图片每行、每列的像素值
//#define NUM 20				//两幅图差异阈值
//
//void resetsize(IplImage* scr,IplImage* dst);//调正图片大小
//void enhance(IplImage* scr,IplImage* dst);//对比度增强
//void imagedevide(IplImage* scr);
//void preprocess(IplImage* dst);
//void imagematch(IplImage* scr1,IplImage* scr2);
//
//int  Threshold(IplImage* scr);
//
//
//IplImage* input = 0;
//IplImage* data = 0;
//IplImage* input_dst = 0;
//IplImage* data_dst = 0;
//
//
//
//int main()
//{
//    //////////////////////////////////////////////////////////////////
//    //读入数据并重新设置大小未标题-1.bmp
//	input = cvLoadImage("/Users/zhaowenichi/test3.jpeg",0);//读入图片，存为灰度图
//	data  = cvLoadImage("/Users/zhaowenichi/test3.jpeg",0);
//    
//    CvSize size_model;//定义图片大小模板
//	size_model.width = IMAGESIZE;
//	size_model.height = IMAGESIZE;
//    input_dst=input;
//	//input_dst = cvCreateImage(size_model,input->depth,input->nChannels);//创建图像头
//	//cvResize(input,input_dst,CV_INTER_LINEAR);//使用双线性差值减小图像
//    
//	data_dst = cvCreateImage(size_model,data->depth,data->nChannels);//创建图像头
//	cvResize(data,data_dst,CV_INTER_LINEAR);//使用双线性差值减小图像。
//    
//    
//    /////////////////////////////////////////////////////////////////
//    //预处理
//	//preprocess(input_dst);			//inpu_dst，data_dst为预处理之后的输出
////preprocess(data_dst);
//    
//    /////////////////////////////////////////////////////////////////
//    //图片匹配
//	imagematch(input_dst,data_dst);
//    
//    /////////////////////////////////////////////////////////////////
//    //图片显示与输出
//	//cvNamedWindow("input", CV_WINDOW_AUTOSIZE );
//    //cvNamedWindow("data", CV_WINDOW_AUTOSIZE );
//	cvNamedWindow("input_dst", CV_WINDOW_AUTOSIZE );
//	//cvNamedWindow("data_dst", CV_WINDOW_AUTOSIZE );
//    
//    //cvShowImage("input", input );
//    //cvShowImage("data", data );
//	cvShowImage("input_dst", input_dst );
//    //cvShowImage("data_dst", data_dst );
//    
//    cvWaitKey(0);
//    
//    //cvReleaseImage( &input );
//    //cvReleaseImage( &data );
//	cvReleaseImage( &input_dst );
//    //cvReleaseImage( &data_dst );
//    //cvDestroyWindow("input");
//    //cvDestroyWindow("data");
//	cvDestroyWindow("input_dst");
//    //cvDestroyWindow("data_dst");
//    
//}
//
//void preprocess(IplImage* dst)
//{
//	int i,j;
//	int SubImgWidth = 320/MAX,SubImgHeight = 320/MAX;
//	int flag[MAX][MAX];				//标记数组，背景部分为1，含有指纹部分为0
//	memset(flag,1,sizeof(flag));	//初始化数组数据
//    
//	CvScalar mean,std_dev;			//mean为平均数，std_dev为标准差
//    
//    /*
//     /////////////////////////////////////////////////////////////////////
//     //标准化图片大小
//     
//     CvSize size_model;//定义图片大小模板
//     size_model.width = IMAGESIZE;
//     size_model.height = IMAGESIZE;//确定新图的矩形框
//     dst = cvCreateImage(size_model,scr->depth,scr->nChannels);//创建图像头
//     cvResize(scr,dst,CV_INTER_LINEAR);//使用双线性差值减小图像。
//     */
//    
//    
//    /////////////////////////////////////////////////////////////////////
//    //对比度增强
//    
//	enhance(dst,dst);//直方图均衡化
//    
//    /////////////////////////////////////////////////////////////////////
//    //图像分割
//    
//	for(i = 0;i<MAX;i++)
//		for(j = 0;j<MAX;j++)
//		{
//			cvSetImageROI(dst,cvRect(i*SubImgWidth,j*SubImgHeight,SubImgWidth,SubImgHeight));
//            
//			cvAvgSdv(dst,&mean,&std_dev,NULL);//求均值和标准差
//            
//			if(std_dev.val[0]<BOUNDARY_VALUE)	//如果标准差比较大，则认为是背景区域，将其变成白色
//			{
//				flag[i][j]=1;
//				cvSet(dst,cvScalar(255));
//			}
//            
//			cvResetImageROI(dst);//取消ROI
//		}
//    
//    /////////////////////////////////////////////////////////////////////
//    //最大差方阈值化
//    
//	int nBestT;
//	nBestT = Threshold(dst);
//    
//	// 利用最佳阈值对原图像作分割处理
//	cvThreshold(dst, dst, nBestT, 255,CV_THRESH_BINARY);
//}
//
////直方图均衡化加强对比度
//void enhance(IplImage* scr,IplImage* dst)
//{
//	cvEqualizeHist(scr,dst);
//}
//
////最大差方阈值化
//int  Threshold(IplImage* scr)
//{
//	int i;
//	int width = scr->width;
//	int height = scr->height;
//	int step = scr->widthStep;
//	uchar* data = (uchar *)scr->imageData;
//    
//	int hist[256] = {0};
//    
//	for(int i=0;i<height;i++)
//	{
//		for(int j=0;j<width;j++)
//		{
//			hist[data[i*step+j]]++;
//		}
//	}
//    
//    
//	// c0组和c1组的均值
//	float u0,u1;
//    
//	// c0组和c1组的概率
//	float w0,w1;
//    
//	// c0组的像素总数
//	int nCount0;
//    
//	// 阈值和最佳阈值（对应方差最大时的阈值）
//	int nT,nBestT;
//    
//	// 方差和最大方差
//	float fVaria,fMaxVaria = 0;
//    
//	// 统计直方图中像素点的总数，并存放到nSum中
//	int nSum=0;
//	for(i = 0; i < 256; i ++)
//		nSum += hist[i];
//    
//    
//	// 令阈值nT从0遍历到255
//	for(nT = 0; nT < 256; nT ++)
//	{
//		// 当阈值为nT时，计算c0组的均值和概率
//		u0 = 0;
//		nCount0 = 0;
//		for(i = 0; i <= nT; i++)
//		{
//			u0 += i * hist[i];
//			nCount0 += hist[i];
//		}
//		u0 /= nCount0;
//		w0 = (float) nCount0 / nSum;
//        
//		// 当阈值为nT时，计算c1组的均值和概率
//		u1 = 0;
//		for(i = nT+1; i < 256; i ++)
//			u1 += i * hist[i];
//		u1 /= (nSum - nCount0);
//		w1 = 1 - w0;
//        
//		// 计算两组间的方差
//		fVaria = w0 * w1 * (u0 - u1) * (u0 - u1);
//        
//		// 记录最大方差和最佳阈值
//		if(fVaria > fMaxVaria)
//		{
//			fMaxVaria = fVaria;
//			nBestT = nT;
//		}
//	}
//    
//	return nBestT;
//}
//
//
//
//void minuteExtract(IplImage* scr1)
//{
//    int sum;
//    for (size_t y=1; y<scr1->height-1; y++) {
//        uchar* ptr1 = (uchar*)scr1->imageData+y*scr1->widthStep;
//        uchar* ptr2 = (uchar *)scr1->imageData + (y-1)*scr1->widthStep;
//        uchar* ptr3 = (uchar *)scr1->imageData + (y+1)*scr1->widthStep;
//        for (size_t x=1 ; x<scr1->width-1; x++) {
//            sum = 0;
//            cout<<ptr1[x]<<" ";
//            int right=ptr1[x+1];
//            int left=ptr1[x-1];
//            int up=ptr2[x];
//            int upright=ptr2[x +1];
//            int upleft=ptr2[x -1];
//            int down=ptr3[x ];
//            int downright=ptr3[x +1];
//            int downleft=ptr3[x -1];
//            sum = abs(upleft - up)+abs(up-upright)+abs(upright-right)+abs(right-downright)+abs(downright-down)+abs(down-downleft)+abs(downleft-left)+abs(left-upleft);
//            if(abs (sum-255*2) <=10){
//                //cvCircle(scr1,cvPoint((int)x,(int)y),1,cvScalar(255,255,0),1);
//            }
//        }
//        cout<<endl;
//    }
//    for (size_t y=1; y<scr1->height-1; y++) {
//        uchar* ptr1 = (uchar*)scr1->imageData+y*scr1->widthStep;
//        uchar* ptr2 = (uchar *)scr1->imageData + (y-1)*scr1->widthStep;
//        uchar* ptr3 = (uchar *)scr1->imageData + (y+1)*scr1->widthStep;
//        for (size_t x=1 ; x<scr1->width-1; x++) {
//            sum = 0;
//            int right=ptr1[x+1];
//            int left=ptr1[x-1];
//            int up=ptr2[x];
//            int upright=ptr2[x +1];
//            int upleft=ptr2[x -1];
//            int down=ptr3[x ];
//            int downright=ptr3[x +1];
//            int downleft=ptr3[x -1];
//            sum = abs(upleft - up)+abs(up-upright)+abs(upright-right)+abs(right-downright)+abs(downright-down)+abs(down-downleft)+abs(downleft-left)+abs(left-upleft);
//            if(abs (sum-255*6) <=5){
//                //cvCircle(scr1,cvPoint((int)x,(int)y),20,cvScalar(0,0,0),-1);
//            }
//        }
//    }
//}
//
//void imagematch(IplImage* scr1,IplImage* scr2)
//{
//    cout<<scr1->height<<endl;
//    cout<<scr1->width<<" "<<scr1->widthStep<<endl;
//    int sum;
//    for (size_t y=1; y<scr1->height-1; y++) {
//        uchar* ptr1 = (uchar*)scr1->imageData+y*scr1->widthStep;
//        uchar* ptr2 = (uchar *)scr1->imageData + (y-1)*scr1->widthStep;
//        uchar* ptr3 = (uchar *)scr1->imageData + (y+1)*scr1->widthStep;
//        for (size_t x=1 ; x<scr1->width-1; x++) {
//            sum = 0;
//            if((int)ptr1[x]<127)
//            {
//                ptr1[x]=0;
//            }
//            else
//                ptr1[x]=255;
//        }
//        for (size_t x=1 ; x<scr1->width-1; x++) {
//                sum = 0;
//                //cout<<(int)ptr1[x]<<" ";
//                int right=(int)ptr1[x+1];
//                int left=(int)ptr1[x-1];
//                int up=(int)ptr2[x];
//                int upright=(int)ptr2[x +1];
//                int upleft=(int)ptr2[x -1];
//                int down=(int)ptr3[x ];
//                int downright=(int)ptr3[x +1];
//                int downleft=(int)ptr3[x -1];
//                sum = abs(upleft - up)+abs(up-upright)+abs(upright-right)+abs(right-downright)+abs(downright-down)+abs(down-downleft)+abs(downleft-left)+abs(left-upleft);
//            if(abs (sum-255*2) ==0){
//                //cout<<x<<","<<y<<endl;
//                //cvCircle(scr1,cvPoint((int)x,(int)y),1,cvScalar(0,0,0),1);
//            }
//        }
//        //cout<<endl;
//    }
//    for (size_t y=1; y<scr1->height-1; y++) {
//        uchar* ptr1 = (uchar *)scr1->imageData +  y   *scr1->widthStep;
//        uchar* ptr2 = (uchar *)scr1->imageData + (y-1)*scr1->widthStep;
//        uchar* ptr3 = (uchar *)scr1->imageData + (y+1)*scr1->widthStep;
//        for (size_t x=1 ; x<scr1->width-1; x++) {
//            sum = 0;
//            int right=(int)ptr1[x+1];
//            int left=(int)ptr1[x-1];
//            int up=(int)ptr2[x];
//            int upright=(int)ptr2[x +1];
//            int upleft=(int)ptr2[x -1];
//            int down=(int)ptr3[x ];
//            int downright=(int)ptr3[x +1];
//            int downleft=(int)ptr3[x -1];
//            sum = abs(upleft-up)+abs(up-upright)+abs(upright-right)+abs(right-downright)+abs(downright-down)+abs(down-downleft)+abs(downleft-left)+abs(left-upleft);
//            if(abs(sum-255*6)==0){
//                cout<<x<<","<<y<<endl;
//                cvCircle(scr1,cvPoint((int)x,(int)y),3,cvScalar(0,255,0),2);
//            }
//        }
//    }
//}
float GetWeightedAngle(Mat &mag,Mat &ang)
{
    float res=0;
    float n=0;
    for (int i=0;i< mag.rows;++i)
    {
        for (int j=0;j< mag.cols;++j)
        {
            res+=ang.at<float>(i,j)*mag.at<float>(i,j);
            n+=mag.at<float>(i,j);
        }
    }
    res/=n;
    return res;
}
int main(int argc, char** argv)
{
    SiftFeatureDetector siftdtc;
    vector<KeyPoint>kp1,kp2;
    
    IplImage* input = cvLoadImage("/Users/zhaowenichi/tmp.bmp",0);
    IplImage* data  = cvLoadImage("/Users/zhaowenichi/0207202791.bmp",0);
   
    
    //sift特征点
   /* siftdtc.detect(input,kp1);
    Mat outimg1;
    drawKeypoints(input,kp1,outimg1);
    imshow("image1 keypoints",outimg1);
    KeyPoint kp;
    vector<KeyPoint>::iterator itvc;
    for(itvc=kp1.begin();itvc!=kp1.end();itvc++)
    {
        cout<<"angle:"<<itvc->angle<<"\t"<<itvc->class_id<<"\t"<<itvc->octave<<"\t"<<itvc->pt<<"\t"<<itvc->response<<endl;
    }
    
    siftdtc.detect(data,kp2);
    Mat outimg2;
    drawKeypoints(data,kp2,outimg2);
    imshow("image2 keypoints",outimg2);
    
    SiftDescriptorExtractor extractor;
    Mat descriptor1,descriptor2;
    BruteForceMatcher<L2<float>> matcher;
    vector<DMatch> matches;
    Mat img_matches;
    extractor.compute(input,kp1,descriptor1);
    extractor.compute(data,kp2,descriptor2);
    imshow("desc",descriptor1);
    cout<<endl<<descriptor1<<endl;
    matcher.match(descriptor1,descriptor2,matches);
    
    drawMatches(input,kp1,data,kp2,matches,img_matches);
    imshow("matches",img_matches);
    */
    
    //方向场
    Mat img=imread("/Users/zhaowenichi/Desktop/ScreenShot.png",0);
    cv::threshold(img,img,128,255,THRESH_BINARY);
    Mat thinned;
    
    thinned=img.clone();
    //Thinning(img,thinned);
    
    //cv::GaussianBlur(thinned,thinned,Size(3,3),1.0);
    Mat gx,gy,ang,mag,aa,bb;
    cv::Sobel(thinned,gx,CV_32FC1,1,0,7);
    cv::Sobel(thinned,gy,CV_32FC1,0,1,7);
    cv::phase(gx,gy,ang,false);
    cv::magnitude(gx,gy,mag);
    
    cv::normalize(mag,mag,0,1,NORM_MINMAX);
    
    
    Mat angRes=Mat::zeros(img.rows,img.cols,CV_8UC1);
    
    int blockSize=img.cols/15-1;
    float r=blockSize;
    
    for (int i=0;i< img.rows-blockSize;i+= blockSize)
    {
        for (int j=0;j< img.cols-blockSize;j+= blockSize)
        {
            aa=mag(Rect(j,i,blockSize,blockSize));
            bb=ang(Rect(j,i,blockSize,blockSize));
            float a=GetWeightedAngle(aa,bb);
            
            float dx=r*cos(a);
            float dy=r*sin(a);
            int x=j;
            int y=i;
            
            cv::line(angRes,cv::Point(x,y),cv::Point(x+dx,y+dy),Scalar::all(255),1,CV_AA);
        }
    }
    imshow("ang",angRes);
    imshow("source",img);
    cv::waitKey(0);
    
    
    
    
    
    
    
    
    
    /*
    int sum ;
    for (size_t y=1; y<input->height; y++)
    {
        uchar* ptr = (uchar *)input->imageData +  y   *input->widthStep;
        for (size_t x=1 ; x<input->width; x++)
        {
            if(ptr[x]<=127)
            ptr[x]=0;
            else
            ptr[x]=255;
        }
    }
    
    for (size_t y=2; y<input->height-1; y++)
    {
        uchar* ptr1 = (uchar *)input->imageData +  y   *input->widthStep;
        uchar* ptr2 = (uchar *)input->imageData + (y-1)*input->widthStep;
        uchar* ptr3 = (uchar *)input->imageData + (y+1)*input->widthStep;
        for (size_t x=2 ; x<input->width-1; x++)
        {
            sum = 0;
            int right=(int)ptr1[x+1];
            int left=(int)ptr1[x-1];
            int up=(int)ptr2[x];
            int upright=(int)ptr2[x +1];
            int upleft=(int)ptr2[x -1];
            int down=(int)ptr3[x ];
            int downright=(int)ptr3[x +1];
            int downleft=(int)ptr3[x -1];
            int center=(int)ptr1[x];
            sum = (abs(upleft-up)+abs(up-upright)+abs(upright-right)+abs(right-downright)+abs(downright-down)+abs(down-downleft)+abs(downleft-left)+abs(left-upleft))/255;
            if(sum==6)
            {
                cout<<"交点"<<":";
                cout<<sum<<","<<x<<","<<y<<endl;
                //cvCircle(input,cvPoint((int)x,(int)y),3,cvScalar(0,0,0),1);
            }
            if(center==0&&sum==2)
            {
                cout<<"端点： "<<sum<<","<<center<<","<<x<<","<<y<<endl;
            }
        }
    }*/
     waitKey();
    return 0;
    
//    cvNamedWindow("window", CV_WINDOW_AUTOSIZE );
//    cvShowImage("window", input );
//    cvReleaseImage( &input );    cvWaitKey(0);
//    cvDestroyWindow("window");
}