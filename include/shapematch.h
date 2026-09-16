#ifndef SHAPEMATCH_H
#define SHAPEMATCH_H
#include"line2Dup.h"
#include <memory>
#include <iostream>
#include <assert.h>
#include <chrono>
#include <numeric>

class Timer
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const {
        return std::chrono::duration_cast<second_>
            (clock_::now() - beg_).count(); }
    void out(std::string message = ""){
        double t = elapsed();
        std::cout << message << "\nelasped time:" << t << "s" << std::endl;
        reset();
    }
private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};

namespace shape_match {
class ShapeMatch;
}

class templateParameter
{
public:
    friend class shape_match::ShapeMatch;
private:
    int num_feature = 150;
    std::vector<int> stride = {4,8};
    std::vector<float> angle_range;
    float angle_step = 5;
    std::vector<float> scale_range = {1};
    float scale_step = 0.1;
    int padding = 0;
};

class matchParameter
{
public:
    friend class shape_match::ShapeMatch;

private:
    float minScore = 90;
    int maxNum = 5;
    float NMSThreshold = 0.3;
};

class match_result
{
public:
    int match_num;
    double run_times;
    std::vector<cv::RotatedRect> rect_box;
    std::vector<float> score_box;
    std::vector<std::vector<cv::Point>> outlines;
    std::vector<cv::Point> matchPoint;
};

namespace shape_match {

class ShapeMatch
{
public:
    ShapeMatch();
    ShapeMatch(std::string Class_id):class_id(Class_id){};
    void setTemplateImage(const cv::Mat& image);
    void setTemplateMask(const cv::Mat& mask);
    void setClassId(const std::string classId);
    void setMatchImage(const cv::Mat& image);
    void setMatchMask(const cv::Mat& mask);
    void setMinScore(const float score);
    void setMaxNum(const int num);
    void setNMSThreshold(const float thre);
    void setNum_Feature(const int num);
    void setStride(const std::vector<int> stride);
    void setAngle_Range(const std::vector<float> range);
    void setAngle_Step(const float step);
    void setScale_Range(const std::vector<float> range);
    void setScale_Step(const float step);

    int getNum_Feature(){return tempPar.num_feature;};
    std::vector<int>  getStride(){return tempPar.stride;};
    std::vector<float> getAngle_Range(){return tempPar.angle_range;};
    float getAngel_Step(){return tempPar.angle_step;};
    std::vector<float> getScale_Range(){return tempPar.scale_range;};
    float getScale_Step(){return tempPar.scale_step;};
    std::vector<cv::Point> getTemp_Feature_Point(){return temp_feature_point;};

    float getMinScore(){return matchPar.minScore;};
    int getMaxNum(){return matchPar.maxNum;};
    float getNMSThreshold(){return matchPar.NMSThreshold;};

    cv::Mat getTempImage(){return tempImage;};
    cv::Mat getTempMask(){return tmp_mask;};
    std::string getClass_Id(){return class_id;};
    cv::Mat getMatchImage(){return matchImage;};
    cv::Mat getMatchMask(){return match_mask;};
    match_result getMatchResult(){return result;};

    void restore_default_par();

    void saveToload(std::string savePath);
    void readFromload(std::string loadPath);
    void Train();
    void Run();

private:
    line2Dup::Detector detector;

    //template par
    cv::Mat tempImage;
    cv::Mat tmp_mask;
    std::vector<shape_based_matching::shapeInfo_producer::Info> infos_have_templ;
    std::string class_id = "default";
    templateParameter tempPar;
    bool train_vaild = false;
    std::vector<cv::Point> temp_feature_point;

    //Match par
    cv::Mat matchImage;
    cv::Mat match_mask;
    matchParameter matchPar;

    //result
    match_result result;
};

}

#endif // SHAPEMATCH_H
