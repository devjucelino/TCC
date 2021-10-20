#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <unordered_map>
#include <string>
#include <vector>
#include <thread>

using namespace cv;
using namespace std;


///////////////////////////////////////////////////////////////////////////////////////////////////
#define TAM 1 // Parâmetro para leitura do arquivo
#define NUMBYTES 74 // Número de bytes do arquivo de input do sensor contando com o '/0'
#define PATH1 "/sys/bus/w1/devices/28-3c01a81625b4/w1_slave" //caminho do arquivo sensor 1
#define PATH2 "/sys/bus/w1/devices/28-3c01a816346b/w1_slave" //caminho do arquivo sensor 2
#define PATH3 "/sys/bus/w1/devices/28-3c01a8169cc7/w1_slave" //caminho do arquivo sensor 3
#define PATH4 "/sys/bus/w1/devices/28-3c01a816a335/w1_slave" //caminho do arquivo sensor 4
#define PATH5 "/sys/bus/w1/devices/28-d00d991864ff/w1_slave" //caminho do arquivo sensor 5

#ifndef MY_BLOB
#define MY_BLOB

class Blob {
public:
    // variáveis da classe
    std::vector<cv::Point> contour;

    cv::Rect boundingRect;

    cv::Point centerPosition;

    double dblDiagonalSize;

    double dblAspectRatio;

    Blob(std::vector<cv::Point> _contour); //Protótipo da função

};

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

Blob::Blob(std::vector<cv::Point> _contour) {

    contour = _contour;

    boundingRect = cv::boundingRect(contour);

    centerPosition.x = (boundingRect.x + boundingRect.x + boundingRect.width) / 2;
    centerPosition.y = (boundingRect.y + boundingRect.y + boundingRect.height) / 2;

    dblDiagonalSize = sqrt(pow(boundingRect.width, 2) + pow(boundingRect.height, 2));

    dblAspectRatio = (float)boundingRect.width / (float)boundingRect.height;

}

int mostFrequent(long int arr[], int n);
float leitura_sensor(int sensor);


///////////////////////////////////////////////////////////////////////////////////////////////////

// Declaração de variáveis globais
const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);
const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
const cv::Scalar SCALAR_BLUE = cv::Scalar(255.0, 0.0, 0.0);
const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 200.0, 0.0);
const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);
unsigned int contador = 0;
int flag = 1;
long int camera_data;
float sensor_data;



///////////////////////////////////////////////////////////////////////////////////////////////////

void camera_sensores(){

    float sensor_1;
    float sensor_2;
    float sensor_3;
    float sensor_4;
    float sensor_5;
    long int filtro[101];

    cv::VideoCapture capVideo(0);

    cv::Mat imgFrame1;
    cv::Mat imgFrame2;

    while(true){
        capVideo.read(imgFrame1);
        capVideo.read(imgFrame2);

        while (capVideo.isOpened()) {

            std::vector<Blob> blobs;

            cv::Mat imgFrame1Copy = imgFrame1.clone();
            cv::Mat imgFrame2Copy = imgFrame2.clone();

            cv::Mat imgDifference;
            cv::Mat imgThresh;

            cv::cvtColor(imgFrame1Copy, imgFrame1Copy, cv::COLOR_BGR2GRAY);
            cv::cvtColor(imgFrame2Copy, imgFrame2Copy, cv::COLOR_BGR2GRAY);

            cv::GaussianBlur(imgFrame1Copy, imgFrame1Copy, cv::Size(5, 5), 0);
            cv::GaussianBlur(imgFrame2Copy, imgFrame2Copy, cv::Size(5, 5), 0);

            cv::absdiff(imgFrame1Copy, imgFrame2Copy, imgDifference);

            cv::threshold(imgDifference, imgThresh, 30, 255.0, cv::THRESH_BINARY);

            //cv::imshow("imgThresh", imgThresh);

            cv::Mat structuringElement3x3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
            cv::Mat structuringElement5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
            cv::Mat structuringElement7x7 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
            cv::Mat structuringElement9x9 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));
            cv::Mat structuringElement15x15 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));

            cv::dilate(imgThresh, imgThresh, structuringElement15x15);
            cv::dilate(imgThresh, imgThresh, structuringElement15x15);
            cv::erode(imgThresh, imgThresh, structuringElement9x9);

            //cv::GaussianBlur(imgThresh, imgThresh, cv::Size(21, 21), 0);

            cv::Mat imgThreshCopy = imgThresh.clone();

            std::vector<std::vector<cv::Point> > contours;

            cv::findContours(imgThreshCopy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            cv::Mat imgContours(imgThresh.size(), CV_8UC3, SCALAR_BLACK);

            cv::drawContours(imgContours, contours, -1, SCALAR_WHITE, -1);

            //cv::imshow("imgContours", imgContours);

            std::vector<std::vector<cv::Point> > convexHulls(contours.size());

            for (unsigned int i = 0; i < contours.size(); i++) {
                cv::convexHull(contours[i], convexHulls[i]);

            }

            for (auto &convexHull : convexHulls) {
                Blob possibleBlob(convexHull);

                if (possibleBlob.boundingRect.area() > 7000 &&
                    possibleBlob.dblAspectRatio > 0.2 &&
                    possibleBlob.dblAspectRatio < 1.25 &&
                    possibleBlob.boundingRect.width > 50 &&
                    possibleBlob.boundingRect.height > 100 &&
                    possibleBlob.dblDiagonalSize > 30.0 &&
                    (cv::contourArea(possibleBlob.contour) / (double)possibleBlob.boundingRect.area()) > 0.5) {
                    blobs.push_back(possibleBlob);
                }
            }

            cv::Mat imgConvexHulls(imgThresh.size(), CV_8UC3, SCALAR_BLACK);

            convexHulls.clear();

            for (auto &blob : blobs) {
                convexHulls.push_back(blob.contour);
            }

            cv::drawContours(imgConvexHulls, convexHulls, -1, SCALAR_WHITE, -1);

            //cv::imshow("imgConvexHulls", imgConvexHulls);

            imgFrame2Copy = imgFrame2.clone();          // get another copy of frame 2 since we changed the previous frame 2 copy in the processing above


            for (auto &blob : blobs) {                                                  // for each blob
                cv::rectangle(imgFrame2Copy, blob.boundingRect, SCALAR_RED, 2);             // draw a red box around the blob
                cv::circle(imgFrame2Copy, blob.centerPosition, 3, SCALAR_GREEN, -1);        // draw a filled-in green circle at the center

            }

            filtro[contador] = blobs.size();

            if(contador == 100){

                int nonzero = 0;

                for(int i = 0; i < 101; i++){
                    if(filtro[i] != 0){
                        nonzero++;
                    }
                }

                long int nao_nulos[nonzero];

                for(int i = 0; i < 101; i++){
                    for(int j = 0; j < nonzero; j++){
                        if(filtro[i] != 0){
                            nao_nulos[j] = filtro[i];
                            //cout << nao_nulos[j] <<endl;
                        }
                    }
                }


                int n = sizeof(nao_nulos)/sizeof(nao_nulos[0]);
                camera_data = mostFrequent(nao_nulos,n);

                sensor_1 = leitura_sensor(1);
                sensor_2 = leitura_sensor(2);
                sensor_3 = leitura_sensor(3);
                sensor_4 = leitura_sensor(4);
                sensor_5 = leitura_sensor(5);

                sensor_data = (sensor_1 + sensor_2 + sensor_3 + sensor_4 + sensor_5)/5;
                //camera_data = blobs.size();

                cout << "Numero de pessoas = " << camera_data << " e temperatura = " << sensor_data << endl;

                flag = 0; //sinal para  a thread

                contador = 0;

            }

            cv::imshow("imgFrame2Copy", imgFrame2Copy);

                    // now we prepare for the next iteration

            imgFrame1 = imgFrame2.clone();           // move frame 1 up to where frame 2 is

            capVideo.read(imgFrame2);               // read next frame and store it in imgFrame2

            cv::waitKey(1);

            contador++;
        }

    }

}


int main(){

    std::thread cam_sens (camera_sensores);

    system("sudo modprobe w1-gpio");

    system("sudo modprobe w1-therm");

    while (1){

        if (flag == 0){ //considera-se um intervalo de tempo de 10s

            time_t now;

            time(&now);

            ofstream fp;

            fp.open("tcc_data.csv",ios::app);

            fp << format("%d, %f, %s", camera_data, sensor_data, ctime(&now));

            fp.close();

            flag = 1;

        }


    }

    return 0;
}

float leitura_sensor(int sensor){

    ifstream fp;
    char buffer[100]; //buffer para armazenar os dados do arquivo
    char temp_c[5]; //array para armazenar os bytes com a informação da temperatura
    float temp; //variável para armazenar a conversão da temperatura

    memset(buffer, 0, sizeof(buffer)); //zerando todos os bits do buffer

    switch(sensor){
        case 1:
            fp.open(PATH1); //abrindo o arquivo
            break;

        case 2:
            fp.open(PATH2); //abrindo o arquivo
            break;

        case 3:
            fp.open(PATH3); //abrindo o arquivo
            break;

        case 4:
            fp.open(PATH4); //abrindo o arquivo
            break;

        case 5:
            fp.open(PATH5); //abrindo o arquivo
            break;

        default:
            printf("Nenhum sensor selecionado!\n");
            exit(-1);
    }

    fp.read(buffer, TAM*NUMBYTES);

    for(int i = 0; i <= 4; i++){
        temp_c[i] = buffer[69+i];//transferindo para o array temp_c os bytes com a informação da temperatura
    }

    temp = (float)atof(temp_c)/1000; //convertendo o array de string para float

    fp.close();

    return temp;
}

int mostFrequent(long int arr[], int n)
{
    // Sort the array
    sort(arr, arr + n);

    // find the max frequency using linear traversal
    int max_count = 1, res = arr[0], curr_count = 1;
    for (int i = 1; i < n; i++) {
        if (arr[i] == arr[i - 1])
            curr_count++;
        else {
            if (curr_count > max_count) {
                max_count = curr_count;
                res = arr[i - 1];
            }
            curr_count = 1;
        }
    }

    // If last element is most frequent
    if (curr_count > max_count)
    {
        max_count = curr_count;
        res = arr[n - 1];
    }

    return res;
}
