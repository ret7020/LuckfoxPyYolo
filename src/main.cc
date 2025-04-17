#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "yolov8.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <chrono>


rknn_app_context_t rknn_app_ctx;
int ret = 0;
int model_input_size = 640;
cv::Mat bgr640;
cv::VideoCapture cap;

extern "C"
{

    int init(const char *model_path, int input_size)
    {
        model_input_size = input_size;
        memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
        ret = init_yolov8_model(model_path, &rknn_app_ctx);
        if (ret != 0)
        {
            printf("init_yolov8_model fail! ret=%d model_path=%s\n", ret, model_path);
            return ret;
        }

        bgr640 = cv::Mat(model_input_size, model_input_size, CV_8UC3, rknn_app_ctx.input_mems[0]->virt_addr);
        return ret;
    }

    void camera_init(int camera_ind){
        cap.set(cv::CAP_PROP_FRAME_WIDTH, model_input_size);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, model_input_size);
        cap.open(0);

        // Warmup
        cv::Mat bgr;
        for (int i = 0; i < 5; i++){cap >> bgr;}
    }

    int release()
    {
        cap.release();
        return release_yolov8_model(&rknn_app_ctx);
    }

    void read_image_camera(){
        cv::Mat bgr;
        cap >> bgr;
        cv::resize(bgr, bgr640, cv::Size(model_input_size, model_input_size), 0, 0, cv::INTER_LINEAR);
    }

    void read_image_file(const char *image_path)
    {
        cv::Mat img = cv::imread(image_path);
        cv::resize(img, bgr640, cv::Size(model_input_size, model_input_size), 0, 0, cv::INTER_LINEAR);
    }

    object_detect_result_list inference()
    {
        rknn_run(rknn_app_ctx.rknn_ctx, nullptr);
        // printf("FPS: %lf\n", 1 / std::chrono::duration<double>(end - begin).count());
        object_detect_result_list od_results;
        post_process(&rknn_app_ctx, rknn_app_ctx.output_mems, 0.25, 0.45, &od_results);
        // printf("%d\n\n", od_results.count);

        for (int i = 0; i < od_results.count; i++)
        {
            object_detect_result *det_result = &(od_results.results[i]);
            int x1 = det_result->box.left;
            int y1 = det_result->box.top;
            int x2 = det_result->box.right;
            int y2 = det_result->box.bottom;
            cv::Rect r = cv::Rect(x1, y1, x2 - x1, y2 - y1);
            printf("Object: %d\n", det_result->cls_id);
            cv::rectangle(bgr640, r, cv::Scalar(255, 255, 0), 1, 8, 0);
        }

        cv::imwrite("/root/out.jpg", bgr640);
        return od_results;
    }
}
