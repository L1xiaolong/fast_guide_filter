#include "iostream"
#include "opencv2/opencv.hpp"
#include "opencv2/ximgproc.hpp"

using namespace std;
using namespace cv;

#define CLIP_U8(x)  MIN(MAX(x, 0), 255)
#define R  21
#define EPS 0.001f
#define S  8

void add(float *src1, float *src2, float *dst, int width, int height);
void subtract(float *src1, float *src2, float *dst, int width, int height);
void multiply(float *src1, float *src2, float *dst, int width, int height);
void divide(float *src1, float *src2, float *dst, int width, int height);
void resize_(float *src, float *dst, int h_s, int w_s, int h_d, int w_d);
void box_filter(float* src, float *dst, int height, int width, int size);


int main()
{
    Mat P = imread("../trans.jpg", IMREAD_GRAYSCALE);
    Mat I = imread("../gray.jpg", IMREAD_GRAYSCALE);
    int height = I.rows;
    int width = I.cols;

#if 1
    Mat d;
    ximgproc::guidedFilter(I, P, d, R, EPS);
    imwrite("../1.jpg", d);

    Mat t;
    resize(I, t, Size(), 1.0 / S, 1.0 / S);
    Mat tt;
    resize(t, tt, Size(width, height));
    imwrite("../re.jpg", tt);
#endif

    // fast guide filter
    I.convertTo(I, CV_32F, 1.0 / 255.0);
    P.convertTo(P, CV_32F, 1.0 / 255.0);

    float* _I = (float*) malloc(height * width * sizeof(float));
    float* _P = (float*) malloc(height * width * sizeof(float));

    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            _I[i * width + j] = I.at<float>(i, j);
            _P[i * width + j] = P.at<float>(i, j);
        }
    }

    int h_lr = height / S;
    int w_lr = width / S;

    float *I_ = (float*) malloc(sizeof(float) * h_lr * w_lr);
    float *P_ = (float*) malloc(h_lr * w_lr * sizeof(float));

    resize_(_I, I_, height, width, h_lr, w_lr);
    resize_(_P, P_, height, width, h_lr, w_lr);

    int r_lr = 2 * (R / S) + 1;

    float* M_I = (float*) malloc(h_lr * w_lr * sizeof(float));
    box_filter(I_, M_I, h_lr, w_lr, r_lr);

    float* M_P = (float*) malloc(h_lr * w_lr * sizeof(float));
    box_filter(P_, M_P, h_lr, w_lr, r_lr);

    float* II = (float*) malloc(h_lr * w_lr * sizeof(float));
    multiply(I_, I_, II, w_lr, h_lr);

    float* C_I = (float*) malloc(h_lr * w_lr * sizeof(float));
    box_filter(II, C_I, h_lr, w_lr, r_lr);

    float* IP = (float*) malloc(h_lr * w_lr * sizeof(float));
    multiply(I_, P_, IP, w_lr, h_lr);
    float* C_IP = (float*) malloc(h_lr * w_lr * sizeof(float));
    box_filter(IP, C_IP, h_lr, w_lr, r_lr);

    float* M_II = (float*) malloc(h_lr * w_lr * sizeof(float));
    multiply(M_I, M_I, M_II, w_lr, h_lr);
    float *V_I = (float*) malloc(h_lr * w_lr * sizeof(float));
    subtract(C_I, M_II, V_I, w_lr, h_lr);

    float *M_IP = (float*) malloc(h_lr * w_lr * sizeof(float));
    multiply(M_I, M_P, M_IP, w_lr, h_lr);
    float *COV_IP = (float*) malloc(h_lr * w_lr * sizeof(float));
    subtract(C_IP, M_IP, COV_IP, w_lr, h_lr);

    float* a = (float*) malloc(h_lr * w_lr * sizeof(float));
    divide(COV_IP, V_I, a, w_lr, h_lr);

    float* b = (float*) malloc(h_lr * w_lr * sizeof(float));
    multiply(a, M_I, M_I, w_lr, h_lr);
    subtract(M_P, M_I, b, w_lr, h_lr);

    float* a_mean = (float*) malloc(h_lr * w_lr * sizeof(float));
    box_filter(a, a_mean, h_lr, w_lr, r_lr);

    float* b_mean = (float*) malloc(h_lr * w_lr * sizeof(float));
    box_filter(b, b_mean, h_lr, w_lr, r_lr);

    float *a_up = (float *) malloc(height * width * sizeof(float));
    resize_(a_mean, a_up, h_lr, w_lr, height, width);

    float *b_up = (float *) malloc(height * width * sizeof(float));
    resize_(b_mean, b_up, h_lr, w_lr, height, width);

    float * dst = (float *) malloc(height * width * sizeof(float));
    multiply(a_up, _I, dst, width, height);

    float * dst1 = (float *) malloc(height * width * sizeof(float));
    add(dst, b_up, dst1, width, height);

    Mat res = Mat::zeros(height, width, CV_8UC1);
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            res.at<uchar>(i, j) = (uchar)CLIP_U8(dst1[i * width + j] * 255);
        }
    }

    imwrite("../result.jpg", res);

    return 0;
}

void add(float *src1, float *src2, float *dst, int width, int height)
{
    float *ps1 = src1;
    float *ps2 = src2;
    float *pd = dst;

    for(int i = 0; i < width * height; i++)
    {
        *pd++ = (*ps1) + (*ps2);
        ps1++;
        ps2++;
    }
}

void subtract(float *src1, float *src2, float *dst, int width, int height)
{
    float *ps1 = src1;
    float *ps2 = src2;
    float *pd = dst;

    for(int i = 0; i < width * height; i++)
    {
        *pd++ = (*ps1) - (*ps2);
        ps1++;
        ps2++;
    }

}

void multiply(float *src1, float *src2, float *dst, int width, int height)
{
    float *ps1 = src1;
    float *ps2 = src2;
    float *pd = dst;

    for(int i = 0; i < width * height; i++)
    {
        *pd++ = (*ps1) * (*ps2);
        ps1++;
        ps2++;
    }

}

void divide(float *src1, float *src2, float *dst, int width, int height)
{
    float *ps1 = src1;
    float *ps2 = src2;
    float *pd = dst;

    for(int i = 0; i < width * height; i++)
    {
        *pd = (*ps1) / (*ps2 + EPS);
        pd++;
        ps1++;
        ps2++;
    }
}

void resize_(float *src, float *dst, int h_s, int w_s, int h_d, int w_d)
{
    float scale_h = (float)h_d / (float)h_s;
    float scale_w = (float)w_d / (float)w_s;

    float *ps = src;
    float *pd = dst;

    for(int i = 0; i < h_d; i++)
    {
        for(int j = 0; j < w_d; j++)
        {
            float x = (float)(i + 0.5f) / scale_h - 0.5f;
            float y = (float)(j + 0.5f) / scale_w - 0.5f;

            int x_left = int(x);
            int y_left = int(y);
            int x_right = MIN(x_left + 1, h_s - 1) ;
            int y_right = MIN(y_left + 1, w_s - 1) ;

            if(x_right == x_left){
                x_left--;
            }
            if(y_right == y_left){
                y_left--;
            }

            float y1 = (ps[x_right * w_s + y_right] - ps[x_left * w_s + y_right]) * (x - x_left) / (x_right - x_left) + ps[x_left * w_s + y_right];
            float y2 = (ps[x_right * w_s + y_left] - ps[x_left * w_s + y_left]) * (x - x_left) / (x_right - x_left) + ps[x_left * w_s + y_left];

            float y3 = (y1 - y2) * (y - y_left) / (y_right - y_left) + y2;

            pd[i * w_d + j] = y3;
        }
    }
}

void box_filter(float* src, float *dst, int height, int width, int size)
{
    int kCenter = size / 2;

    float *tmp = (float *) calloc(height * width, sizeof(float));
    float *ps = src;

    // horizon
    for(int i = 0; i < height; i++){
        float * in = (float *)malloc((width + size) * sizeof(float));
        memcpy(in, ps + i * width, kCenter * sizeof(float));
        memcpy(in + kCenter, ps + i * width, width * sizeof(float));
        memcpy(in + (kCenter + width), in + width, kCenter * sizeof(float));

        for (int j = kCenter; j < width + kCenter; j++) {
            float sum = 0;
            for (int k = j - kCenter; k < j + kCenter + 1; k++) {
                sum += in[k];
            }
            tmp[i * width + (j - kCenter)] = sum / size;
        }

        free(in);
    }

    // vertical
    float* in = (float *) malloc(width * (height * size) * sizeof(float));
    memcpy(in, tmp, kCenter * width * sizeof(float));
    memcpy(in + kCenter * width, tmp, height * width * sizeof(float));
    memcpy(in + (kCenter + height) * width, in + height * width, kCenter * width * sizeof(float));

    float *pd = dst;
    for (int i = 0; i < width; i++) {
        for (int j = kCenter; j < height + kCenter; j++) {
            float sum = 0;
            for (int k = j - kCenter; k < j + kCenter + 1; k++) {
                sum += in[k * width + i];
            }
            pd[(j - kCenter) * width + i] = sum / size;

        }
    }

    free(in);
    free(tmp);
}