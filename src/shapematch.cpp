#include "shapematch.h"
#include <memory>
#include <iostream>
#include <assert.h>
#include <chrono>
#include <fstream>
#include <cstdint>
// NMS, got from cv::dnn so we don't need opencv contrib
// just collapse it
namespace  cv_dnn {
namespace
{

template <typename T>
static inline bool SortScorePairDescend(const std::pair<float, T>& pair1,
                          const std::pair<float, T>& pair2)
{
    return pair1.first > pair2.first;
}

} // namespace
static void GetMaxScoreIndex(const std::vector<float>& scores, const float threshold, const int top_k,
                      std::vector<std::pair<float, int> >& score_index_vec)
{
    for (size_t i = 0; i < scores.size(); ++i)
    {
        if (scores[i] > threshold)
        {
            score_index_vec.push_back(std::make_pair(scores[i], i));
        }
    }
    std::stable_sort(score_index_vec.begin(), score_index_vec.end(),
                     cv_dnn::SortScorePairDescend<int>);
    if (top_k > 0 && top_k < (int)score_index_vec.size())
    {
        score_index_vec.resize(top_k);
    }
}

template <typename BoxType>
static void NMSFast_(const std::vector<BoxType>& bboxes,
      const std::vector<float>& scores, const float score_threshold,
      const float nms_threshold, const float eta, const int top_k,
      std::vector<int>& indices, float (*computeOverlap)(const BoxType&, const BoxType&))
{
    CV_Assert(bboxes.size() == scores.size());
    std::vector<std::pair<float, int> > score_index_vec;
    GetMaxScoreIndex(scores, score_threshold, top_k, score_index_vec);

    // Do nms.
    float adaptive_threshold = nms_threshold;
    indices.clear();
    for (size_t i = 0; i < score_index_vec.size(); ++i) {
        const int idx = score_index_vec[i].second;
        bool keep = true;
        for (int k = 0; k < (int)indices.size() && keep; ++k) {
            const int kept_idx = indices[k];
            float overlap = computeOverlap(bboxes[idx], bboxes[kept_idx]);
            keep = overlap <= adaptive_threshold;
        }
        if (keep)
            indices.push_back(idx);
        if (keep && eta < 1 && adaptive_threshold > 0.5) {
          adaptive_threshold *= eta;
        }
    }
}


// copied from opencv 3.4, not exist in 3.0
template<typename _Tp> static inline
double jaccardDistance__(const cv::Rect_<_Tp>& a, const cv::Rect_<_Tp>& b) {
    _Tp Aa = a.area();
    _Tp Ab = b.area();

    if ((Aa + Ab) <= std::numeric_limits<_Tp>::epsilon()) {
        // jaccard_index = 1 -> distance = 0
        return 0.0;
    }

    double Aab = (a & b).area();
    // distance = 1 - jaccard_index
    return 1.0 - Aab / (Aa + Ab - Aab);
}

template <typename T>
static float rectOverlap(const T& a, const T& b)
{
    return 1.f - static_cast<float>(jaccardDistance__(a, b));
}

static void NMSBoxes(const std::vector<cv::Rect>& bboxes, const std::vector<float>& scores,
                          const float score_threshold, const float nms_threshold,
                          std::vector<int>& indices, const float eta=1, const int top_k=0)
{
    NMSFast_(bboxes, scores, score_threshold, nms_threshold, eta, top_k, indices, rectOverlap);
}

}

static int computeTrainPadding(const cv::Mat& templ, const cv::Mat& mask) {
    int W = templ.cols, H = templ.rows;
    if (!mask.empty()) {
        cv::Rect bbox = cv::boundingRect(mask);
        W = bbox.width;
        H = bbox.height;
    }
    double R = std::sqrt((W/2.0)*(W/2.0) + (H/2.0)*(H/2.0));
    double p_min = R - std::min(W/2.0, H/2.0);
    int p = static_cast<int>(std::ceil(p_min * 1.3));
    return std::max(p, 20);
}

//倾斜四边形的nms抑制

static const float EPS  = 1e-6f;
static bool nearlyEqual(float a, float b, float eps = EPS)
{
    return std::abs(a - b) < eps;
}

static bool nearlyEqual(const cv::Point2f& a, const cv::Point2f& b, float eps = EPS)
{
    return nearlyEqual(a.x, b.x, eps) && nearlyEqual(a.y, b.y, eps);
}

//rotateRect 转顶点 cv::point2f
static std::vector<cv::Point2f> rectToPoints(const cv::RotatedRect& rr)
{
    cv::Point2f pts[4];
    rr.points(pts);

    // 过滤零尺寸
    if (rr.size.width <= EPS || rr.size.height <= EPS)
    {
        return {};
    }

    return {pts[0], pts[1], pts[2], pts[3]};
}

//多边形面积（叉乘）
static float polygonArea(const std::vector<cv::Point2f>& poly)
{
    if (poly.size() < 3) return 0.0f;
    float area = 0.0f;
    for (size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++)
    {
        area += (poly[j].x * poly[i].y - poly[i].x * poly[j].y);
    }
    return std::abs(area) * 0.5f;
}

//点在多边形内（射线法）
static bool pointInPolygon(const cv::Point2f& p,
                           const std::vector<cv::Point2f>& poly)
{
    if (poly.size() < 3) return false;

    bool inside = false;
    for (size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++)
    {
        const auto& pi = poly[i];
        const auto& pj = poly[j];

        // 检查点是否在边上
        float cross = (p.x - pi.x) * (pj.y - pi.y) - (p.y - pi.y) * (pj.x - pi.x);
        if (std::abs(cross) < EPS) {
            float minx = std::min(pi.x, pj.x), maxx = std::max(pi.x, pj.x);
            float miny = std::min(pi.y, pj.y), maxy = std::max(pi.y, pj.y);
            if (p.x >= minx - EPS && p.x <= maxx + EPS &&
                p.y >= miny - EPS && p.y <= maxy + EPS)
            {
                return true;   // 点在边上，算在内部
            }
        }

        // 射线法
        if (((pi.y > p.y) != (pj.y > p.y)) &&
            (p.x < (pj.x - pi.x) * (p.y - pi.y) / (pj.y - pi.y + EPS) + pi.x))
        {
            inside = !inside;
        }
    }
    return inside;
}

// 线段求交
// 返回 true 表示有唯一交点
// 共线重叠时返回 false，交由点在多边形内判断处理
static bool segmentIntersect(const cv::Point2f& p1, const cv::Point2f& p2,
                             const cv::Point2f& p3, const cv::Point2f& p4,
                             cv::Point2f& out)
{
    float d = (p2.x - p1.x) * (p4.y - p3.y) - (p2.y - p1.y) * (p4.x - p3.x);

    // 平行或共线
    if (std::abs(d) < EPS) return false;

    float t = ((p3.x - p1.x) * (p4.y - p3.y) - (p3.y - p1.y) * (p4.x - p3.x)) / d;
    float u = ((p3.x - p1.x) * (p2.y - p1.y) - (p3.y - p1.y) * (p2.x - p1.x)) / d;

    // 交点不在两条线段内（含端点容差）
    if (t < -EPS || t > 1 + EPS || u < -EPS || u > 1 + EPS) return false;

    // 裁剪到 [0,1] 避免浮点越界
    t = std::max(0.0f, std::min(1.0f, t));
    out = p1 + t * (p2 - p1);
    return true;
}

//浮点去重
static void dedupPoints(std::vector<cv::Point2f>& pts, float eps = 1e-3f)
{
    std::vector<cv::Point2f> unique;
    for (const auto& p : pts) {
        bool dup = false;
        for (const auto& q : unique)
        {
            if (nearlyEqual(p, q, eps)) { dup = true; break; }
        }
        if (!dup) unique.push_back(p);
    }
    pts = std::move(unique);
}

//凸包（Andrew 单调链）
static float cross2d(const cv::Point2f& O, const cv::Point2f& A, const cv::Point2f& B)
{
    return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
}

static std::vector<cv::Point2f> convexHull(std::vector<cv::Point2f> pts)
{
    if (pts.size() < 3) return pts;

    std::sort(pts.begin(), pts.end(),
              [](const cv::Point2f& a, const cv::Point2f& b)
    {
                  return a.x < b.x || (a.x == b.x && a.y < b.y);
    });

    std::vector<cv::Point2f> hull;
    // 下凸壳
    for (const auto& p : pts) {
        while (hull.size() >= 2 &&
               cross2d(hull[hull.size()-2], hull[hull.size()-1], p) <= EPS)
        {

            hull.pop_back();
        }
        hull.push_back(p);
    }
    // 上凸壳
    size_t lower = hull.size();
    for (int i = pts.size() - 2; i >= 0; i--) {
        const auto& p = pts[i];
        while (hull.size() > lower &&
               cross2d(hull[hull.size()-2], hull[hull.size()-1], p) <= EPS)
        {
            hull.pop_back();
        }
        hull.push_back(p);
    }
    if (hull.size() > 1) hull.pop_back();   // 去掉重复的起点
    return hull;
}

//旋转矩形 IoU
static float rotatedRectIoU(const cv::RotatedRect& a, const cv::RotatedRect& b)
{
    std::vector<cv::Point2f> poly1 = rectToPoints(a);
    std::vector<cv::Point2f> poly2 = rectToPoints(b);

    // 退化检查
    if (poly1.size() < 4 || poly2.size() < 4) return 0.0f;

    float s_a = polygonArea(poly1);
    float s_b = polygonArea(poly2);
    if (s_a < EPS || s_b < EPS) return 0.0f;

    // ★ 完全包含检查（快速路径）
    // 如果 a 的 4 个顶点全在 b 内，返回 s_a / s_b
    bool a_in_b = true, b_in_a = true;
    for (const auto& p : poly1) if (!pointInPolygon(p, poly2)) { a_in_b = false; break; }
    for (const auto& p : poly2) if (!pointInPolygon(p, poly1)) { b_in_a = false; break; }

    if (a_in_b && b_in_a) return 1.0f;                    // 完全重合
    if (a_in_b) return s_a / s_b;                         // a 在 b 内
    if (b_in_a) return s_b / s_a;                         // b 在 a 内

    // ★ 求交集多边形
    std::vector<cv::Point2f> inter;

    // 1. poly1 在 poly2 内的点
    for (const auto& p : poly1) if (pointInPolygon(p, poly2)) inter.push_back(p);
    // 2. poly2 在 poly1 内的点
    for (const auto& p : poly2) if (pointInPolygon(p, poly1)) inter.push_back(p);
    // 3. 两两边的交点
    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 4; j++) {
            cv::Point2f pt;
            if (segmentIntersect(poly1[i], poly1[(i+1)%4],
                                 poly2[j], poly2[(j+1)%4], pt))
            {
                inter.push_back(pt);
            }
        }
    }

    if (inter.size() < 3) return 0.0f;

    // 去重 + 凸包排序
    dedupPoints(inter);
    if (inter.size() < 3) return 0.0f;
    inter = convexHull(inter);

    if (inter.size() < 3) return 0.0f;
    float s_inter = polygonArea(inter);

    return s_inter / (s_a + s_b - s_inter + EPS);
}

static std::vector<int> rotatedNMS(const std::vector<cv::RotatedRect>& rects,
                            const std::vector<float>& scores,
                            float nms_threshold)
{
    std::vector<int> indices(rects.size());
    std::iota(indices.begin(), indices.end(), 0);

    // 按分数降序
    std::sort(indices.begin(), indices.end(),
              [&](int a, int b) { return scores[a] > scores[b]; });

    std::vector<int> keep;
    std::vector<bool> suppressed(rects.size(), false);

    for (size_t i = 0; i < indices.size(); i++) {
        int idx = indices[i];
        if (suppressed[idx]) continue;
        keep.push_back(idx);

        for (size_t j = i + 1; j < indices.size(); j++) {
            int other = indices[j];
            if (suppressed[other]) continue;

            if (rotatedRectIoU(rects[idx], rects[other]) > nms_threshold) {
                suppressed[other] = true;
            }
        }
    }
    return keep;
}
//end

static int gcd_int(int a, int b) {
    while (b) { int t = a % b; a = b; b = t; }
    return a;
}

static int lcm_int(int a, int b) {
    return a / gcd_int(a, b) * b;   // 先除后乘，避免溢出
}

static int computeStride(const std::vector<int>& pyramids) {
    int s = 1;
    for (int p : pyramids) {
        s = lcm_int(s, p);
    }
    return s * 2;
}

namespace {

void writeMat(std::ofstream& ofs, const cv::Mat& m)
{
    int32_t w = m.empty() ? 0 : m.cols;
    int32_t h = m.empty() ? 0 : m.rows;
    int32_t t = m.empty() ? 0 : m.type();
    ofs.write((char*)&w, 4);
    ofs.write((char*)&h, 4);
    ofs.write((char*)&t, 4);

    std::vector<uchar> buf;
    if (!m.empty()) cv::imencode(".png", m, buf);

    uint64_t len = buf.size();
    ofs.write((char*)&len, 8);
    if (len > 0) ofs.write((char*)buf.data(), len);
}

cv::Mat readMat(std::ifstream& ifs, int flags)
{
    int32_t w, h, t;
    ifs.read((char*)&w, 4);
    ifs.read((char*)&h, 4);
    ifs.read((char*)&t, 4);

    uint64_t len;
    ifs.read((char*)&len, 8);
    if (w == 0 || h == 0 || len == 0) return cv::Mat();

    std::vector<uchar> buf(len);
    ifs.read((char*)buf.data(), len);
    return cv::imdecode(buf, flags);
}

template <typename T>
void writeVec(std::ofstream& ofs, const std::vector<T>& v)
{
    uint32_t n = v.size();
    ofs.write((char*)&n, 4);
    if (n > 0) ofs.write((char*)v.data(), n * sizeof(T));
}

template <typename T>
void readVec(std::ifstream& ifs, std::vector<T>& v)
{
    uint32_t n;
    ifs.read((char*)&n, 4);
    v.resize(n);
    if (n > 0) ifs.read((char*)v.data(), n * sizeof(T));
}

} // namespace

void shape_match::ShapeMatch::setNum_Feature(const int num)
{
  tempPar.num_feature = num;
}

void shape_match::ShapeMatch::setTemplateImage(const cv::Mat& image)
{
    tempImage = image.clone();
}

void shape_match::ShapeMatch::setStride(const std::vector<int> strd)
{
    tempPar.stride = strd;
}

void shape_match::ShapeMatch::setTemplateMask(const cv::Mat& mask)
{
    tmp_mask = mask.clone();
}

void shape_match::ShapeMatch::setClassId(const std::string classId)
{
    class_id = classId;
}

void shape_match::ShapeMatch::setAngle_Range(const std::vector<float> range)
{
    tempPar.angle_range = range;
}

void shape_match::ShapeMatch::setAngle_Step(const float step)
{
    tempPar.angle_step = step;
}

void shape_match::ShapeMatch::setScale_Range(const std::vector<float> range)
{
    tempPar.scale_range = range;
}

void shape_match::ShapeMatch::setScale_Step(const float step)
{
    tempPar.scale_step = step;
}

void shape_match::ShapeMatch::setMatchImage(const cv::Mat &image)
{
    matchImage = image.clone();
}

void shape_match::ShapeMatch::setMatchMask(const cv::Mat &mask)
{
    match_mask = mask.clone();
}

void shape_match::ShapeMatch::setMinScore(const float score)
{
   matchPar.minScore = score;
}

void shape_match::ShapeMatch::setMaxNum(const int num)
{
    matchPar.maxNum = num;
}

void shape_match::ShapeMatch::setNMSThreshold(const float thre)
{
    matchPar.NMSThreshold = thre;
}


void shape_match::ShapeMatch::Train()
{
    infos_have_templ.clear();
    temp_feature_point.clear();
    train_vaild = false;

    if(tempPar.num_feature <= 0 || tempPar.stride.empty())
    {
        std::cout<<"Fail to creator detector Error: num_feature or stride"<<std::endl;
        return ;
    }
    if(tempImage.empty()/* || tmp_mask.empty()*/)
    {
        std::cout<<"Fail to get shapes Error: tempimage or tempmask is empty"<<std::endl;
        return ;
    }
    if(tmp_mask.empty())
    {
        tmp_mask = cv::Mat(tempImage.size(),CV_8UC1,{255});
    }
    if(tmp_mask.size() != tempImage.size() || tmp_mask.type() != CV_8UC1)
    {
        tmp_mask = cv::Mat(tempImage.size(),CV_8UC1,{255});
    }
    if(tempPar.angle_range.size() > 2)
    {
        std::cout<<"Fail to get shapes Error: angle_range unsupport"<<std::endl;
        return ;
    }
    if(tempPar.scale_range.size() > 2)
    {
        std::cout<<"Fail to get shapes Error: scale_range unsupport"<<std::endl;
        return ;
    }
    if (tempPar.scale_step <= 0) {
        std::cout << "Failto get shapes Error:scale_step must > 0" << std::endl;
        return;
    }

    detector = line2Dup::Detector(tempPar.num_feature,tempPar.stride);

    cv::Mat padded_img;
    cv::Mat padded_mask;
    if(tempPar.angle_range.size() == 2)
    {
        // padding to avoid rotating out
        tempPar.padding = computeTrainPadding(tempImage,tmp_mask);
        padded_img = cv::Mat(tempImage.rows + 2*tempPar.padding, tempImage.cols + 2*tempPar.padding, tempImage.type(), cv::Scalar::all(0));
        tempImage.copyTo(padded_img(cv::Rect(tempPar.padding, tempPar.padding, tempImage.cols, tempImage.rows)));

        padded_mask = cv::Mat(tmp_mask.rows + 2*tempPar.padding, tmp_mask.cols + 2*tempPar.padding, tmp_mask.type(), cv::Scalar::all(0));
        tmp_mask.copyTo(padded_mask(cv::Rect(tempPar.padding, tempPar.padding, tmp_mask.cols, tmp_mask.rows)));
    }
    else
    {
        tempPar.padding = 0;
        padded_img = tempImage.clone();
        padded_mask = tmp_mask.clone();
    }

//    float scale_start,scale_end;
//    if(tempPar.scale_range.size() == 2)
//    {
//        scale_start = tempPar.scale_range[0];
//        scale_end = tempPar.scale_range[1];
//    }
//    else
//    {
//        scale_start = tempPar.scale_range[0];
//        scale_end = tempPar.scale_range[0];
//    }

//    int scale_count = (int)std::round((scale_end - scale_start) / tempPar.scale_step) + 1;
//    for(int k = 0; k < scale_count ; k++)
//    {
        shape_based_matching::shapeInfo_producer shapes(padded_img,padded_mask);
        if(tempPar.angle_range.size() == 2)
        {
            shapes.angle_range = tempPar.angle_range;
            shapes.angle_step = tempPar.angle_step;
        }
//        shapes.scale_range = {scale_start + k*tempPar.scale_step};
        shapes.scale_range = {1};
        shapes.produce_infos();

//        bool is_first = true;
//        int first_id;
//        float first_angle = 0;
        for(auto &info : shapes.infos)
        {
            int templ_id;

//            if(is_first){
                templ_id = detector.addTemplate(shapes.src_of(info), class_id, shapes.mask_of(info));
//                first_id = templ_id;
//                first_angle = info.angle;

//                is_first = false;
//            }else{
//                templ_id = detector.addTemplate_rotate(class_id, first_id,
//                                                       info.angle-first_angle,
//                                                       {shapes.src.cols/2.0f, shapes.src.rows/2.0f});
//            }
            if(templ_id != -1)
            {
                infos_have_templ.push_back(info);

                if (std::abs(info.angle) < 1e-3f && std::abs(info.scale - 1.0f) < 1e-3f)
                {
                    auto templ = detector.getTemplates(class_id,templ_id);
                    int dx = templ[0].tl_x;
                    int dy = templ[0].tl_y;
                    for(auto &feature : templ[0].features)
                    {
                        temp_feature_point.push_back(cv::Point(feature.x + dx -tempPar.padding,feature.y + dy -tempPar.padding));
                    }
                }
            }
        }
//    }
//    std::cout<<"infors_have_templ.push num "<<infos_have_templ.size()<<std::endl;
    train_vaild = true;
}

void shape_match::ShapeMatch::Run()
{
    result.rect_box.clear();
    result.score_box.clear();
    result.outlines.clear();
    result.matchPoint.clear();
    result.icp_angles.clear();
    result.match_num = 0;
    result.run_times = 0;

    if(!train_vaild)
    {
        std::cout<<"Fail to run Error: Train vaild false"<<std::endl;
        return;
    }
    if(matchImage.empty())
    {
        std::cout<<"Fail to run Error: matchimage is empty"<<std::endl;
        return ;
    }

    bool use_mask = true;
    if(match_mask.empty())
    {
        match_mask = cv::Mat(matchImage.size(),CV_8UC1,{255});
        use_mask = false;
    }
    else
    {
        if(match_mask.size() != matchImage.size() || match_mask.type() != CV_8UC1)
        {
            match_mask = cv::Mat(matchImage.size(),CV_8UC1,{255});
            use_mask = false;
        }
    }
    if(matchPar.minScore > 100 || matchPar.minScore < 0)
    {
        std::cout<<"Fail to run Error:minscore illegle"<<std::endl;
        return ;
    }
    if(matchPar.maxNum < 0)
    {
        std::cout<<"Fail to run Error:maxnum illegle"<<std::endl;
        return ;
    }

    int padding = 100;
    cv::Mat padded_img = cv::Mat(matchImage.rows + 2*padding,
                                 matchImage.cols + 2*padding, matchImage.type(), cv::Scalar::all(0));
    matchImage.copyTo(padded_img(cv::Rect(padding, padding, matchImage.cols, matchImage.rows)));

    cv::Mat padded_mask = cv::Mat(match_mask.rows + 2*padding,
                                  match_mask.cols + 2*padding,
                                  match_mask.type(), cv::Scalar::all(0));
    match_mask.copyTo(padded_mask(cv::Rect(padding, padding,
                                     match_mask.cols, match_mask.rows)));

    int stride = computeStride(tempPar.stride);
    int n = padded_img.rows/stride;
    int m = padded_img.cols/stride;
    cv::Rect roi(0, 0, stride*m , stride*n);
    cv::Mat img = padded_img(roi).clone();
    cv::Mat mask_cropped = padded_mask(roi).clone();

    Timer timer;
    // match, img, min socre, ids
    std::vector<std::string> ids;
    ids.push_back(class_id);
    std::vector<line2Dup::Match> matches;
    if(!use_mask)//无效掩膜，不使用掩膜，减少计算量
        matches = detector.match(img, matchPar.minScore, ids);
    else
    {
        if(match_mask.size() != matchImage.size())
        {
            std::cout<<"Fail to run Error: matchmask size"<<std::endl;
            return;
        }
        matches = detector.match(img, matchPar.minScore, ids,mask_cropped);
    }
    result.run_times = timer.elapsed();
    timer.out("run end");

//    int max_num = matchPar.maxNum;
//    if(matchPar.maxNum > matches.size())
//        max_num = matches.size();
//    std::cout<<"matches size "<<matches.size()<<std::endl;

    //add
    Scene_kdtree scene;
    KDTree_cpu kdtree;
    scene.init_Scene_kdtree_cpu(detector.dx_, detector.dy_, kdtree);

    std::vector<cv::RotatedRect> rectBox;
    std::vector<float> scoreBox;
    std::vector<std::vector<cv::Point>> outlines;
    std::vector<cv::Point> matchPoint;
    std::vector<double> angles;
    for(int i = 0 ; i < matches.size(); i++)
    {   
        auto match = matches[i];
        auto templ = detector.getTemplates(class_id,match.template_id);

        std::vector<::Vec2f> model_pcd(templ[0].features.size());
        for(int i=0; i<templ[0].features.size(); i++)
        {
            auto& feat = templ[0].features[i];
            model_pcd[i] =
            {
                float(feat.x + match.x),
                float(feat.y + match.y)
            };
        }

        // subpixel, also refine scale
        cuda_icp::RegistrationResult result = cuda_icp::sim3::ICP2D_Point2Plane_cpu(model_pcd, scene);

        float train_img_half_width  = tempImage.cols / 2.0f + tempPar.padding;
        float train_img_half_height = tempImage.rows / 2.0f + tempPar.padding;

        // 映射到 matchImage 坐标系下的模板中心
        float cx = match.x - templ[0].tl_x + train_img_half_width  - padding;
        float cy = match.y - templ[0].tl_y + train_img_half_height - padding;

        float new_cx = result.transformation_[0][0]*cx + result.transformation_[0][1]*cy + result.transformation_[0][2];
        float new_cy = result.transformation_[1][0]*cx + result.transformation_[1][1]*cy + result.transformation_[1][2];

        // 缩放后的宽高
        float w_scaled = tempImage.cols /** infos_have_templ[match.template_id].scale*/;
        float h_scaled = tempImage.rows /** infos_have_templ[match.template_id].scale*/;

        //粗匹配角度
        double init_angle = infos_have_templ[match.template_id].angle;
        if (init_angle >= 180) init_angle -= 360;

        //ICP 修正角度
        double icp_correction_deg = std::atan2(result.transformation_[1][0],
                                               result.transformation_[0][0]) * 180.0 / CV_PI;

        //修正后的模板角度
        double final_angle = init_angle + icp_correction_deg;
        angles.push_back(final_angle);

        // 旋转矩形
        cv::RotatedRect rr({new_cx, new_cy}, {w_scaled, h_scaled}, -final_angle/*-infos_have_templ[match.template_id].angle*/);
        rectBox.push_back(rr);

        int mx = cvRound(match.x - templ[0].tl_x + tempPar.padding - padding);
        int my = cvRound(match.y - templ[0].tl_y + tempPar.padding - padding);
        matchPoint.push_back(cv::Point(mx, my));

        scoreBox.push_back(match.similarity);

        std::vector<cv::Point> outline;
        for(int j=0; j<templ[0].features.size(); j++){
            auto feat = templ[0].features[j];

            float x = feat.x + match.x;
            float y = feat.y + match.y;
            float new_x = result.transformation_[0][0]*x + result.transformation_[0][1]*y + result.transformation_[0][2];
            float new_y = result.transformation_[1][0]*x + result.transformation_[1][1]*y + result.transformation_[1][2];

            outline.push_back(cv::Point(new_x+0.5f-padding,new_y+0.5f-padding));
        }//outline
        outlines.push_back(outline);
    }
    timer.out("icp times");

//    std::cout<<"before indices num "<<rectBox.size()<<std::endl;
//    for (size_t i = 0; i < rectBox.size(); i++) {
//        std::cout << "  [" << i << "] center=("
//                  << rectBox[i].center.x << "," << rectBox[i].center.y << ")"
//                  << " angle=" << rectBox[i].angle
//                  << " score=" << scoreBox[i] << std::endl;
//    }

    std::vector<int> indices;
//    cv_dnn::NMSBoxes(rectBox,scoreBox,matchPar.minScore,matchPar.NMSThreshold,indices,1,5);//rect
    indices = rotatedNMS(rectBox, scoreBox, matchPar.NMSThreshold);
//    std::cout<<"after indices num "<<indices.size()<<std::endl;

    //get result
    int max_num = std::min((int)indices.size(),matchPar.maxNum);
    for(auto &indice : indices)
    {
        if(result.rect_box.size() == max_num) break;
        result.rect_box.push_back(rectBox[indice]);
        result.score_box.push_back(scoreBox[indice]);
        result.outlines.push_back(outlines[indice]);
        result.matchPoint.push_back(matchPoint[indice]);
        result.icp_angles.push_back(angles[indice]);
    }
    result.match_num = result.rect_box.size();
}

void shape_match::ShapeMatch::saveToload(std::string savePath)
{
    if(savePath.empty() || infos_have_templ.empty())
    {
        std::cout<<"Fail to saveToload Error:save path empty"<<std::endl;
    }
    std::string dir = savePath;
    if(!dir.empty() && dir.back() != '/')
        dir += '/';
    detector.writeClasses(dir + class_id + ".yaml");
    shape_based_matching::shapeInfo_producer::save_infos(infos_have_templ, dir + class_id + "_infos.yaml");

    // 打包 //cun 数据 + 配置
    std::string bin_path = dir + class_id + ".sbm";
    std::ofstream ofs(bin_path, std::ios::binary);
    if (!ofs) {
        std::cerr << "cannot open: " << bin_path << std::endl;
        return;
    }

    // magic
    ofs.write("SBM1", 4);

    // class_id
    uint32_t id_len = class_id.size();
    ofs.write((char*)&id_len, 4);
    ofs.write(class_id.data(), id_len);

    // templateParameter
    ofs.write((char*)&tempPar.num_feature, 4);
    writeVec(ofs, tempPar.stride);
    writeVec(ofs, tempPar.angle_range);
    ofs.write((char*)&tempPar.angle_step, 4);
    writeVec(ofs, tempPar.scale_range);
    ofs.write((char*)&tempPar.scale_step, 4);
    ofs.write((char*)&tempPar.padding, 4);

    // matchParameter
    ofs.write((char*)&matchPar.minScore, 4);
    ofs.write((char*)&matchPar.maxNum, 4);
    ofs.write((char*)&matchPar.NMSThreshold, 4);

    // tempImage / tmp_mask
    writeMat(ofs, tempImage);
    writeMat(ofs, tmp_mask);

    // temp_feature_point
    uint64_t pt_count = temp_feature_point.size();
    ofs.write((char*)&pt_count, 8);
    for (auto& p : temp_feature_point) {
        int32_t x = p.x, y = p.y;
        ofs.write((char*)&x, 4);
        ofs.write((char*)&y, 4);
    }

    ofs.close();
    std::cout << "saveToload success: " << bin_path << std::endl;
}

void shape_match::ShapeMatch::readFromload(std::string loadPath)
{
    infos_have_templ.clear();
    temp_feature_point.clear();

    if(loadPath.empty())
    {
        std::cout<<"Fail to readFromload Error:path empty"<<std::endl;
        return;
    }



    std::string dir = loadPath;
    if(!dir.empty() && dir.back() != '/')
        dir += '/';

    std::string bin_path = dir + class_id + ".sbm";
    std::ifstream ifs(bin_path, std::ios::binary);
    if (!ifs) {
        std::cerr << "cannot open: " << bin_path << std::endl;
        return;
    }

    char magic[4];
    ifs.read(magic, 4);
    if (std::string(magic, 4) != "SBM1") {
        std::cerr << "bad magic" << std::endl;
        return;
    }

    // class_id
    uint32_t id_len;
    ifs.read((char*)&id_len, 4);
    std::string saved_id(id_len, '\0');
    ifs.read(&saved_id[0], id_len);
    if (class_id.empty()) class_id = saved_id;

    // templateParameter
    ifs.read((char*)&tempPar.num_feature, 4);
    readVec(ifs, tempPar.stride);
    readVec(ifs, tempPar.angle_range);
    ifs.read((char*)&tempPar.angle_step, 4);
    readVec(ifs, tempPar.scale_range);
    ifs.read((char*)&tempPar.scale_step, 4);
    ifs.read((char*)&tempPar.padding, 4);

    // matchParameter
    ifs.read((char*)&matchPar.minScore, 4);
    ifs.read((char*)&matchPar.maxNum, 4);
    ifs.read((char*)&matchPar.NMSThreshold, 4);

    // tempImage / tmp_mask
    tempImage = readMat(ifs, cv::IMREAD_UNCHANGED);
    tmp_mask  = readMat(ifs, cv::IMREAD_GRAYSCALE);

    if (tmp_mask.empty() && !tempImage.empty()) {
        tmp_mask = cv::Mat(tempImage.size(), CV_8UC1, cv::Scalar(255));
    }

    // temp_feature_point
    uint64_t pt_count;
    ifs.read((char*)&pt_count, 8);
    temp_feature_point.clear();
    temp_feature_point.reserve(pt_count);
    for (uint64_t i = 0; i < pt_count; i++) {
        int32_t x, y;
        ifs.read((char*)&x, 4);
        ifs.read((char*)&y, 4);
        temp_feature_point.push_back(cv::Point(x, y));
    }

    ifs.close();

    std::vector<std::string> ids;
    ids.push_back(class_id);

    detector.readClasses(ids, dir + class_id + ".yaml");
    infos_have_templ = shape_based_matching::shapeInfo_producer::load_infos(dir + class_id + "_infos.yaml");
    if(!infos_have_templ.empty())
    {
        train_vaild = true;
        std::cout<<"readFromload success"<<std::endl;
    }
    else
        std::cout<<"Fail to readFromload Error:data empty"<<std::endl;
}

void shape_match::ShapeMatch::restore_default_par()
{
    tempPar = templateParameter();
    matchPar = matchParameter();
}
