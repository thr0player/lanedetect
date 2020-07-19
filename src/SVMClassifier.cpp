//
//
#include "SVMClassifier.h"
#include "cca.h"
namespace LaneDetect {
#define HOGWIN_SIZE 128
#define STRIDE_SIZE 8

    SVMClassifier::SVMClassifier() : init_model_(false) {
        model_path_ = "/home/lw/Data/hog/";
    }

    int SVMClassifier::LoadData() {
        std::cout << "LoadData:" << model_path_ << std::endl;
        for (int di = 0; di <= 3; ++di) {
            ClassSamples sample;
            sample.label_ = di;
            for (int fi = 1; fi <= 4; ++fi) {
                std::string filename = model_path_ + std::to_string(di) + "/" + std::to_string(fi) + ".png";
                cv::Mat img = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
                if (img.empty()) {
                    std::cout << "fail to load <" << filename << std::endl;
                    exit(0);
                }

                sample.frames_.emplace_back(img.clone());

                for (int ii = 1; ii < 8; ii++) {
                    cv::Mat dstImage;
                    GaussianBlur(img, dstImage, Size(3, 3),
                                 0.2 * ii, 0.2 * ii);
                    sample.frames_.emplace_back(dstImage);
//                    imshow("test" + std::to_string(ii), dstImage);
//                    waitKey();
                }

            }
            dataset_.emplace_back(sample);
        }
        return 0;
    }

    int SVMClassifier::Train() {
        std::cout << "---------------> Train it now" << std::endl;

        //检测窗口(64,128),块尺寸(16,16),块步长(8,8),cell尺寸(8,8),直方图bin个数9
        cv::HOGDescriptor hog(cv::Size(HOGWIN_SIZE, HOGWIN_SIZE), cv::Size(16, 16), cv::Size(STRIDE_SIZE, STRIDE_SIZE), cv::Size(8, 8), 9);
        int DescriptorDim;//HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定

        cv::Mat sampleFeatureMat;//所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数
        cv::Mat sampleLabelMat;//训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，-1表示无人

        int all_samples = 0;
        for (auto &di:dataset_)
            all_samples += di.frames_.size();

        int cur_sample_count = 0;
        bool init_once = false;
        for (int di = 0; di < dataset_.size(); ++di) {

            for (int fi = 0; fi < dataset_.at(di).frames_.size(); ++fi) {
                std::vector<float> descriptors;//HOG描述子向量
                hog.compute(dataset_.at(di).frames_.at(fi), descriptors, cv::Size(8, 8));//计算HOG描述子，检测窗口移动步长(8,8)
                DescriptorDim = descriptors.size();//HOG描述子的维数
                if (!init_once) {
                    //初始化所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数sampleFeatureMat
                    sampleFeatureMat = cv::Mat::zeros(all_samples, DescriptorDim, CV_32FC1);
                    //初始化训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，0表示无人
                    sampleLabelMat = cv::Mat::zeros(all_samples, 1, CV_32SC1);
                    init_once = true;
                }

                //将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
                for (int i = 0; i < DescriptorDim; i++)
                    sampleFeatureMat.at<float>(fi + cur_sample_count, i) = descriptors[i];//第num个样本的特征向量中的第i个元素

                sampleLabelMat.at<int>(fi + cur_sample_count, 0) = dataset_.at(di).label_;//正样本类别为1，有人
            }
            cur_sample_count += dataset_.at(di).frames_.size();
        }

        //// train
        Ptr<SVM> svm = SVM::create();   //声明SVM对象
        svm->setType(SVM::C_SVC);   //SVM模型选择
        svm->setC(0.2);             //惩罚因子设置(原始0.1)
        svm->setKernel(SVM::LINEAR);   //核函数类型：线性
        svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, (int) 1e7, 1e-6));        //迭代要求

        printf("Trained");

        //////////////// train
        std::cout << "data_size" << sampleFeatureMat.rows << "," << sampleFeatureMat.cols << " label_size"
                  << sampleLabelMat.rows << "," << sampleLabelMat.cols << std::endl;

        svm->trainAuto(sampleFeatureMat, ROW_SAMPLE, sampleLabelMat);           //训练
        printf("save");
        svm->save((model_path_ + "hogmarker_svm.xml").c_str());


        /////////////////  evaluate training
        printf("train size %d labl size %d  ", sampleFeatureMat.rows, sampleLabelMat.cols);
        int fail = 0;
        for (size_t row = 0; row < sampleFeatureMat.rows; ++row) {
            Mat sample = sampleFeatureMat.row(row);
            int labelcur = round(svm->predict(sample));
            std::cout << "label:" << sampleLabelMat.at<int>(row, 0) << ", fResponce:" << labelcur << std::endl;
            if (labelcur != sampleLabelMat.at<int>(row)) {
                fail++;
            }
        }
        printf("[SVMClassifier] SVM training DONE!!!! precision: %f, fail count %d",
               (1 - fail / static_cast<float>(sampleLabelMat.rows)), fail);

        return 0;
    }

    int SVMClassifier::ModelInit(std::string filename) {
        if (init_model_) return 0;

        svm_ = ml::SVM::load(filename);
        std::cout<<"load svm "<<svm_->getVarCount() << std::endl;

        if (svm_->empty())
            return -1;

        //格式转换
        Mat sv = svm_->getSupportVectors();
        const int sv_total = sv.rows;
        // get the decision function
        Mat alpha, svidx;
        double rho = svm_->getDecisionFunction( 0, alpha, svidx );

        std::vector< float > hog_detector( sv.cols + 1 );
        memcpy( &hog_detector[0], sv.ptr(), sv.cols*sizeof( hog_detector[0] ) );
        hog_detector[sv.cols] = (float)-rho;

        //设置HOGDescriptor的检测子
        myHOG_ = cv::HOGDescriptor(cv::Size(HOGWIN_SIZE, HOGWIN_SIZE), cv::Size(16, 16),
                                   cv::Size(STRIDE_SIZE, STRIDE_SIZE), cv::Size(8, 8), 9);
        myHOG_.setSVMDetector(hog_detector);
        init_model_ = true;
        return 0;
    }

    int SVMClassifier::Detect(const cv::Mat &img){
        std::string file = model_path_ + "hogmarker_svm.xml";
        if (ModelInit(file) != 0) {
            std::cout << "SVM model incorrect" << std::endl;
            return -1;
        }
        ////////////////////// start to detect
/*        //读入图像进行HOG行人检测
        std::vector<Rect> found, found_filtered;
        myHOG_.detectMultiScale(img, found, 0.1, Size(8, 8), Size(24, 24), 1.15, 2, true);//对图片进行多尺度行人检测
        std::cout << "找到的marker个数：" << found.size() << std::endl;
        //最大的矩形框放入found_filtered
        for(unsigned int i=0; i < found.size(); i++)
        {
            Rect r = found[i];
            unsigned int j=0;
            for(; j < found.size(); j++)
                if(j != i && (r & found[j]) == r)
                    break;
            if( j == found.size())
                found_filtered.push_back(r);
        }

        cv::Mat showimg = img.clone();
        //画矩形框
        for(unsigned int i=0; i<found_filtered.size(); i++)
        {
            Rect r = found_filtered[i];
//            putText(showimg,std::to_string());
//            rectangle(showimg, r.tl(), r.br(), Scalar(0,255,0), 2);
        }
        imwrite("Img.jpg", showimg);
        namedWindow("img", 0);
        imshow("img", showimg);
        waitKey();
        */

        std::vector<DetectClass> found;
        DetectMultiscale(img, found);
        cv::Mat showimg;
        cvtColor(img, showimg,CV_GRAY2BGR);
        //画矩形框
        for(unsigned int i=0; i<found.size(); i++)
        {
            Rect r = found[i].rect_;
            rectangle(showimg, r.tl(), r.br(), Scalar(0,0,0), 1);
            putText(showimg, std::to_string(found[i].label_), Point2f(r.tl().x - 3, r.tl().y - 3),
                    cv::FONT_HERSHEY_COMPLEX, 0.5,
                    Scalar(round(rand() % 255), round(rand() % 255), round(rand() % 255)));
        }
        imwrite(model_path_ + "multiScaleImg.jpg", showimg);
        namedWindow("img", 0);
        imshow("img", showimg);
        waitKey();

        return 0;
    }

    int SVMClassifier::DetectMultiscale(const cv::Mat &test_img, std::vector<DetectClass> &detects) {
        detects.clear();
        auto GetHogFeature = [this](const cv::Mat& img) {
            std::vector<float> des;
            myHOG_.compute(img, des);
            return des;
        };

        auto CheckLabel=[](float label){
            float thres = 0.1;
            if(fabs(label-1)<thres){
                return 1;
            }else if(fabs(label-2)<thres){
                return 2;
            }else if(fabs(label-3)<thres) {
                return 3;
            }else{
                return -1;
            }
        };

        std::cout << "myHOG_.winSize.width:" << myHOG_.winSize.width << ",h" << myHOG_.winSize.height << std::endl;
        std::cout << "myHOG_.blockStride.width:" << myHOG_.blockStride.width << ",h" << myHOG_.blockStride.height << std::endl;
        int neighbor_sizex = myHOG_.winSize.width  + 3;
        int neighbor_sizey = myHOG_.winSize.height + 3;
        std::cout << "myHOG_.neighbor_sizex.:" << neighbor_sizex << ",y" << neighbor_sizey << std::endl;

        for (size_t ix = 0; ix < test_img.cols - neighbor_sizex; ix += myHOG_.blockStride.width) {
            for (size_t iy = 0; iy < test_img.rows - neighbor_sizey; iy += myHOG_.blockStride.height) {
                std::cout << "ix,iy" << ix << "," << iy << std::endl;
                cv::Mat roi = test_img(cv::Rect(ix, iy, HOGWIN_SIZE, HOGWIN_SIZE));
                auto des = GetHogFeature(roi);
                Mat testDescriptor = Mat::zeros(1, des.size(), CV_32FC1);
                for (size_t i = 0; i < des.size(); i++)
                {
                    testDescriptor.at<float>(0, i) = des[i];
                }

                float label = svm_->predict(testDescriptor);
                std::cout << "label:" << label << std::endl;
                int classlabel = CheckLabel(label);
                if (classlabel > 0){
                    DetectClass tmp;
                    tmp.label_ = classlabel;
                    tmp.rect_ = cv::Rect(ix, iy, HOGWIN_SIZE, HOGWIN_SIZE);
                    detects.emplace_back(tmp);
                    std::cout << "ok:" << classlabel << std::endl;

//                    imshow("subwin", roi);
//                    cv::waitKey();

                }else{
                    std::cout << "fail:" << classlabel << std::endl;
                }

            }
        }
        return detects.size();
    }

    int SVMClassifier::DetectCluster(const cv::Mat &test_img) {
        assert(!test_img.empty());
        assert(test_img.channels() == 1);

        std::string file = model_path_ + "hogmarker_svm.xml";
        if (ModelInit(file) != 0) {
            std::cout << "SVM model incorrect" << std::endl;
            return -1;
        }


        auto GetHogFeature = [this](const cv::Mat& img) {
            std::vector<float> des;
            myHOG_.compute(img, des);
            return des;
        };

        auto CheckLabel=[](float label){
            float thres = 0.1;
            if(fabs(label-1)<thres){
                return 1;
            }else if(fabs(label-2)<thres){
                return 2;
            }else if(fabs(label-3)<thres) {
                return 3;
            }else{
                return -1;
            }
        };


        cv::Mat img = test_img.clone();
        cv::Mat label_img;
        std::set<int> labels;
        CcaByTwoPass(img, label_img, labels, CCA_NEIGHBOR, CCA_NEIGHBOR);
        std::cout << "labels_size:" << labels.size() << std::endl;
        std::map<int, DetectClass> label_rect;
        for (int col = 0; col < label_img.cols; ++col) {
            for (int row = 0; row < label_img.rows; ++row) {
                const int &label = label_img.at<int>(row, col);
                if(label !=0 ) {
                    if(label_rect.find(label) == label_rect.end()){
                        label_rect[label] = DetectClass(); //init
                    }
                    DetectClass &dc = label_rect[label];
                    dc.minpt.x = (dc.minpt.x > col) ? col : dc.minpt.x;
                    dc.minpt.y = (dc.minpt.y > row) ? row : dc.minpt.y;
                    dc.maxpt.x = (dc.maxpt.x < col) ? col : dc.maxpt.x;
                    dc.maxpt.y = (dc.maxpt.y < row) ? row : dc.maxpt.y;

                    dc.label_ = label;
                }
            }
        }

        cv::Mat showimg;
        cvtColor(test_img, showimg, CV_GRAY2BGR);
        for (auto &m_dc:label_rect) {
            m_dc.second.GetRect();
//            std::cout<<"rect:"<<m_dc.second.rect_<<std::endl;
            const Rect &r = m_dc.second.rect_;
            rectangle(showimg, r.tl(), r.br(), Scalar(189, 188, 0), 1);
        }


        /////  start to detect
        for (auto &m_dc:label_rect) {
            const Rect &r = m_dc.second.rect_;
            Point2f pointcen = (r.tl() + r.br()) * 0.5;
            int maxlen = (r.width > r.height) ? r.width : r.height;
            const int edge = 6;
            maxlen += edge;
            Rect target_r = Rect(pointcen.x - maxlen * 0.5, pointcen.y - maxlen * 0.5, maxlen, maxlen);
            std::cout << "target_r:" << target_r << std::endl;

            if (target_r.tl().x < 0 || target_r.tl().y < 0 || target_r.br().x >= test_img.cols ||
                target_r.br().y >= test_img.rows) {
                continue;
            }

            ///// detect
            cv::Mat roi = test_img(target_r).clone();
            cv::Mat rec_img;
            cv::resize(roi, rec_img, Size(myHOG_.winSize.width, myHOG_.winSize.height));
            std::cout << "rec_img:" << rec_img.cols << "," << rec_img.rows << std::endl;
            auto des = GetHogFeature(rec_img);
            Mat testDescriptor = Mat::zeros(1, des.size(), CV_32FC1);
            for (size_t i = 0; i < des.size(); i++)
            {
                testDescriptor.at<float>(0, i) = des[i];
            }

            float label = svm_->predict(testDescriptor);
            std::cout << "label:" << label << std::endl;
            int classlabel = CheckLabel(label);
            if (classlabel > 0){
                DetectClass tmp;
                tmp.label_ = classlabel;
                tmp.rect_ = target_r;
//                detects.emplace_back(tmp);
                std::cout << "ok:" << classlabel << std::endl;

//                imshow("subwin", rec_img);
//                cv::waitKey();
                putText(showimg, std::to_string(classlabel), target_r.tl(), CV_FONT_NORMAL, 1, Scalar(189, 0, 188));
                rectangle(showimg, target_r.tl(), target_r.br(), Scalar(189, 188, 0), 1);

            }else{
                std::cout << "fail:" << classlabel << std::endl;
            }

        }

        imshow("cluster_img",showimg);
//        imshow("cluster_label_img",label_img);
//        imwrite(model_path_ + "cluster_img.png", showimg);
        waitKey(10);
        return 0;
    }
}