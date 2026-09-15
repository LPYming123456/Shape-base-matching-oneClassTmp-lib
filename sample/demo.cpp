#include "line2Dup.h"
#include <memory>
#include <iostream>
#include <assert.h>
#include <chrono>
#include"shapematch.h"
using namespace std;
using namespace cv;

int main(int argc, char** argv){

    std::string tmp,mtc;
    if(argc >= 3)
    {
        tmp = argv[1];
        mtc = argv[2];
    }
//方式一：训练模版
        cv::Mat temp = cv::imread(tmp);
        cv::Mat match = cv::imread(mtc);

        //创建工具
        shape_match::ShapeMatch matchtool("demo");

        //导入模版图像
        matchtool.setTemplateImage(temp);

        //设置模版训练参数(可不进行设置，有默认值)
//        match.setNum_Feature(150);
//        match.setStride({4,8});
//        matchtool.setAngle_Range({1,360});
//        matchtool.setAngle_Step(5);
//        matchtool.setScale_Range({0.9f,1});
//        matchtool.setScale_Step(0.1f);

        //模版训练
        matchtool.Train();

        //获取训练结果-特征点集
//        std::vector<cv::Point> feature_points;
//        feature_points = getTemp_Feature_Point();

        //导入匹配图像
        matchtool.setMatchImage(match);

        //设置匹配参数(可不进行设置，有默认值)
//        matchtool.setMinScore(90);
//        matchtool.setMaxNum(5);
//        matchtool.setNMSThreshold(0.3);

        //匹配
        matchtool.Run();

        //模版信息的存储
//        matchtool.saveToload("savePath");

        //获取结果
        auto result = matchtool.getMatchResult();

        //绘制结果
        //矩形框
        for(auto & rect : result.rect_box)
        {

            cv::Point2f vertices[4];
            rect.points(vertices);   // 取四个角点

            for (int i = 0; i < 4; i++) {
                cv::line(match, vertices[i], vertices[(i + 1) % 4], {255}, 2);
            }
        }

        //特征点
        for(auto& outline:result.outlines)
        {
            for(auto& p:outline)
            {
                cv::circle(match,p,1,{255});
            }
        }
        cv::imshow("",match);
        cv::waitKey(-1);
        return 0;


// 方式二：从本地读取）（已经存储有）
//        cv::Mat match = cv::imread("match.bmp");

//        //创建工具
//        shape_match::ShapeMatch matchtool("demo");

//        //读取模版信息
//        matchtool.readFromload("readPath");

//        //获取训练结果-特征点集
////        std::vector<cv::Point> feature_points;
////        feature_points = getTemp_Feature_Point();

//        //导入匹配图像
//        matchtool.setMatchImage(match);

//        //匹配
//        matchtool.Run();
}
