/*
 * MNN_process.cpp
 *
 *  Created on: 2022年2月10日
 *      Author: xiegr19
 *
 *  MNN推理处理模块，
 *
 */
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <typeinfo>
#include <mutex>
#include <memory>
#include <map>
#include <thread>
#include <sstream>
#include <string>
#include <pthread.h>
#include <MNN/Interpreter.hpp>
#include "common_data.h"
#include "packet_process.h"
#include "MNN_process.h"

#define INPUT_NAME "input"
#define OUTPUT_NAME "output"

char * LABEL2ID[CLASSES_SIZE] = {"Analysis","Backdoors", "Benign", "DoS", "Exploits", "Fuzzers", "Generic", "Reconnaissance", "Shellcode", "Worms"};
float minmax_min[23] = {53.0 ,46.0 ,60.0 ,0.0 ,212.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,5.0 ,29.0 ,21.0 ,21.0 ,0.0 ,0.0 ,0.0 ,0.0};
float minmax_max[23] = {1514.0 ,1514.0 ,1518.0 ,530351.1875 ,6056.0 ,4.0 ,4.0 ,1.0 ,1.0 ,2.0 ,0.0 ,65526.0 ,65526.0 ,65533.0 ,1010747656.1875 ,6.0 ,255.0 ,65535.0 ,65535.0 ,1.0 ,1.0 ,4.0 ,4.0};
float minmax_dif[23] = {1461.0000, 1468.0000, 1458.0000, 530351.1875, 5844.0000, 4.0000, 4.0000, 1.0000, 1.0000, 2.0000, 0.00001, 65526.0000, 65526.0000, 65533.0000, 1010747656.1875, 1.0000, 226.0000, 65514.0000, 65514.0000, 1.0000, 1.0000, 4.0000, 4.0000};
pthread_mutex_t waitTimeMutex;
pthread_mutex_t resFileMutex;

extern int startModelNo;
extern int endModelNo;

using namespace std;

string g_modelList[MAX_MODEL_COUNT] = {     // 模型文件列表，需要按照精度从低到高排序, 0-7(h3c);8-13(edgecore);14-24(pi);25-32(CPU)
        "./model/H3C/etnas_unsw_95.09009505478699_0.014346_0.071784_0.1543328.mnn",
        "./model/H3C/etnas_unsw_96.14623969495673_0.02313_0.130536_0.27663296.mnn",
        "./model/H3C/etnas_unsw_96.54315229345794_0.02357_0.136136_0.28937.mnn",
        "./model/H3C/etnas_unsw_96.59946164438857_0.03449_0.214632_0.46004863999999995.mnn",
        "./model/H3C/etnas_unsw_96.85628746538603_0.045434_0.280584_0.56938496.mnn",
        "./model/H3C/etnas_unsw_97.03345601786455_0.063106_0.40116_0.82163192.mnn",
        "./model/H3C/etnas_unsw_97.25045321864334_0.080034_0.491208_0.9529445599999999.mnn",
        "./model/H3C/etnas_unsw_97.27654781675808_0.080418_0.493512_0.9529676.mnn",
        "./model/Edgecore/etnas_unsw_95.09009505478699_0.014346_0.071784_0.1543328.mnn",
        "./model/Edgecore/etnas_unsw_96.14623969495673_0.02313_0.130536_0.27663296.mnn",
        "./model/Edgecore/etnas_unsw_96.54315229345794_0.02357_0.136136_0.28937.mnn",
        "./model/Edgecore/etnas_unsw_96.95654568931253_0.048058_0.29556_0.586928.mnn",
        "./model/Edgecore/etnas_unsw_97.03345601786455_0.063106_0.40116_0.82163192.mnn",
        "./model/Edgecore/etnas_unsw_97.27654781675808_0.080418_0.493512_0.9529676.mnn",
        "./model/PI/etnas_unsw_95.09009505478699_0.014346_0.071784_0.1543328.mnn",
        "./model/PI/etnas_unsw_96.14623969495673_0.02313_0.130536_0.27663296.mnn",
        "./model/PI/etnas_unsw_96.54315229345794_0.02357_0.136136_0.28937.mnn",
        "./model/PI/etnas_unsw_96.59946164438857_0.03449_0.214632_0.46004863999999995.mnn",
        "./model/PI/etnas_unsw_96.85628746538603_0.045434_0.280584_0.56938496.mnn",
        "./model/PI/etnas_unsw_96.88512888988387_0.046058_0.287304_0.58683968.mnn",
        "./model/PI/etnas_unsw_96.92083729127472_0.04777_0.293832_0.58690496.mnn",
        "./model/PI/etnas_unsw_96.95654568931253_0.048058_0.29556_0.586928.mnn",
        "./model/PI/etnas_unsw_97.03345601786455_0.063106_0.40116_0.82163192.mnn",
        "./model/PI/etnas_unsw_97.20375764123072_0.06829_0.42996_0.8567026400000001.mnn",
        "./model/PI/etnas_unsw_97.27654781675808_0.080418_0.493512_0.9529676.mnn",
        "./model/CPU/etnas_unsw_95.09009505478699_0.014346_0.071784_0.1543328.mnn",
        "./model/CPU/etnas_unsw_96.54315229345794_0.02357_0.136136_0.28937.mnn",
        "./model/CPU/etnas_unsw_96.59946164438857_0.03449_0.214632_0.46004863999999995.mnn",
        "./model/CPU/etnas_unsw_96.92083729127472_0.04777_0.293832_0.58690496.mnn",
        "./model/CPU/etnas_unsw_96.95654568931253_0.048058_0.29556_0.586928.mnn",
        "./model/CPU/etnas_unsw_97.03345601786455_0.063106_0.40116_0.82163192.mnn",
        "./model/CPU/etnas_unsw_97.20375764123072_0.06829_0.42996_0.8567026400000001.mnn",
        "./model/CPU/etnas_unsw_97.27654781675808_0.080418_0.493512_0.9529676.mnn"
};
uint32_t g_modelSelect[MAX_MODEL_COUNT] = {0};

#ifdef ETNAS
MNN::Interpreter * g_mnnNets[MAX_MODEL_COUNT];
uint32_t g_mnnNetIndex = MAX_MODEL_COUNT / 2;
#else
MNN::Interpreter * g_mnnNet;
MNN::Session * g_session;
#endif

double total_time = 0;
double total_mnn_time = 0;
FILE *res = NULL;
FILE *prob = NULL;
FILE *finishTimeFile = NULL;
std::mutex mnnMutex;
map<thread::id, MNN::Session *> g_sessionMap;
uint32_t g_MNNStreamsCount = 0;
uint32_t g_MNNCount = 0;
uint32_t g_streamCount = 0;
extern int g_flag;

void initialMNN(){
    res = fopen(WAITTIME_FILE_NAME, "w");
    fclose(res);

    res = fopen("vcls.txt", "w");
    prob = fopen("vprob.txt", "w");
    finishTimeFile = fopen("finishtime.csv", "w");
#ifdef ETNAS
    for(uint32_t i = 0; i < MAX_MODEL_COUNT; i++){
        g_mnnNets[i] = MNN::Interpreter::createFromFile(g_modelList[i].c_str());
    }
#else
    g_mnnNet = MNN::Interpreter::createFromFile("our_model.mnn");
#endif
}
void freeMNN(){
    fflush(res);
    fflush(prob);
    fflush(finishTimeFile);
#ifdef ETNAS
    for(uint32_t i = 0; i < MAX_MODEL_COUNT; i++){
        delete g_mnnNets[i];
        g_mnnNets[i] = NULL;
    }
#else
    delete g_mnnNet;
    g_mnnNet = NULL;
#endif
    fclose(res);
    fclose(prob);
    fclose(finishTimeFile);
}

float calAvg(float * date, uint32_t size){
    if(NULL == date || 0 == size){
        return 0;
    }

    float total = 0;
    for(uint32_t index = 0; index < size; index++){
        total += date[index];
    }
    return total / size;
}

float calVariance(float * date, uint32_t size, float avg){
    if(NULL == date || 0 == size){
        return 0;
    }

    float total = 0;
    for(uint32_t index = 0; index < size; index++){
        total += (date[index] - avg) * (date[index] - avg);
    }
    return sqrt(total / size);
}

#ifdef ETNAS
int EtnasProcessPacket(MNN::Interpreter *mnnNet, MNN::Session * session, uint8_t *packet_data, uint32_t data_len, uint32_t stream_count, char * tupleKey) {
    if(NULL == finishTimeFile){
        printf("Error: finishTimeFile is NULL and return\n");
        return 1;
    }

#if WRITE_MNN_RESULT
    char resultBuffer[PRINT_BUFFER_SIZE] = {0};
    uint32_t resultBufferSize = 0;
#endif

#ifdef DEBUG
    thread::id threadId = this_thread::get_id();
    std::ostringstream oss;
    oss << threadId;
#endif

    struct timespec  mnnStartTime = {0};
    clock_gettime(CLOCK_MONOTONIC_RAW, &mnnStartTime);

    MNN::Tensor *input = NULL; {
        std::unique_lock<std::mutex> lock(mnnMutex);
        input = mnnNet->getSessionInput(session, INPUT_NAME);
    }

float *pBuffer = (float *)packet_data;
#if NORMALIZATION_VARIANCE == 1
    // Calcalated standard deviation
    float mean = calAvg(pBuffer, 23);
    float std = calVariance(pBuffer, 23, mean);
#endif
#if NORMALIZATION_MINMAX == 1
#endif

#if WRITE_MNN_RESULT
    SNPRINTFTOFILE(",{\n");
    SNPRINTFTOFILE("\t\"prex\": [");
    SNPRINTFTOFILE("%f", pBuffer[0]);
    for (uint32_t i = 1; i < data_len / sizeof(float); i++) {
        SNPRINTFTOFILE(", %f", pBuffer[i]);
    }
    SNPRINTFTOFILE("],\n");

    SNPRINTFTOFILE("\t\"x\": [");
#if NORMALIZATION_VARIANCE == 1
        float value = (pBuffer[0] - mean)/std;
#endif
#if NORMALIZATION_MINMAX == 1
        float value = 100 * ((pBuffer[0] - minmax_min[0]) / minmax_dif[0]);
#endif
        input->host<float>()[0] = value;
#if WRITE_MNN_RESULT
        SNPRINTFTOFILE("%f", value);
#endif
#endif
    for (uint32_t i = 1; i < data_len / sizeof(float); i++) {
#if NORMALIZATION_VARIANCE == 1
        float value = (pBuffer[i] - mean)/std;
#endif
#if NORMALIZATION_MINMAX == 1
        float value = 100 * ((pBuffer[i] - minmax_min[i]) / minmax_dif[i]);
#endif
        input->host<float>()[i] = value;
#if WRITE_MNN_RESULT
        SNPRINTFTOFILE(", %f", value);
#endif
    }
#if WRITE_MNN_RESULT
    SNPRINTFTOFILE("],\n");
#endif

    // run session
    struct timespec cstart, ccpend;
    clock_gettime(CLOCK_MONOTONIC_RAW, &cstart);
    mnnNet->runSession(session);
    clock_gettime(CLOCK_MONOTONIC_RAW, &ccpend);
    double timeDiff1 = SubTimeSpecToSec(ccpend, cstart);{
        std::unique_lock<std::mutex> lock(mnnMutex);
        total_time += timeDiff1;
        g_MNNStreamsCount += stream_count;
    }
    // get output data
    MNN::Tensor *output = NULL;{
        std::unique_lock<std::mutex> lock(mnnMutex);
        output = mnnNet->getSessionOutput(session, OUTPUT_NAME);
    }
    auto output_host = make_shared < MNN::Tensor > (output, MNN::Tensor::CAFFE);
    output->copyToHostTensor(output_host.get());

    // Save MNN result
#if WRITE_MNN_RESULT
    auto values = output_host->host<float>();
    SNPRINTFTOFILE("\t\"pred\": [");
    float maxValue = -9999.999;
    int maxIndex = 0;

    SNPRINTFTOFILE("%f", values[0]);
    if(maxValue < values[0]){
        maxIndex = 0;
        maxValue = values[0];
    }
    for(int i = 1; i < CLASSES_SIZE; i++){
        SNPRINTFTOFILE(", %f", values[i]);

        if(maxValue < values[i]){
            maxIndex = i;
            maxValue = values[i];
        }
    }
    SNPRINTFTOFILE("],\n");

    SNPRINTFTOFILE("\t\"attack_cat\": \"%s\",\n", LABEL2ID[maxIndex]);
    SNPRINTFTOFILE("\t\"five_tuple_key\": \"%s\"\n}", tupleKey);
//    fprintf(prob, "%d\n", values[max_index]);

    // Write process time to file
    struct timespec mnnEndTime = {0};
    clock_gettime(CLOCK_MONOTONIC_RAW, &mnnEndTime);
    struct timespec *packetRcvTime = (struct timespec*) (packet_data + MAX_SINGLE_MNN_LEN * MAX_MNN_STREAMS); {
        DEBUGPRINTF("write to file %p with buffer %p and value %ld.%ld - %ld.%ld\n", finishTimeFile, packetRcvTime, mnnEndTime.tv_sec, mnnEndTime.tv_nsec, packetRcvTime[0].tv_sec, packetRcvTime[0].tv_nsec);
        std::unique_lock<std::mutex> lock(mnnMutex);
        for (uint32_t i = 0; i < stream_count; i++) {
            fprintf(finishTimeFile, "%s,%f,%f\n",  tupleKey, timeDiff1/stream_count,  SubTimeSpecToSec(mnnEndTime, packetRcvTime[i]));
        }
        fflush(finishTimeFile);
        total_mnn_time += SubTimeSpecToSec(mnnEndTime, mnnStartTime);
    }
    DEBUGPRINTF("ThreadId %s count %d start %ld.%ld end %ld.%ld core %f total_core %f total_mnn %f\n", oss.str().c_str(), g_MNNCount, mnnStartTime.tv_sec, mnnStartTime.tv_nsec, mnnEndTime.tv_sec, mnnEndTime.tv_nsec, timeDiff1, total_time, total_mnn_time);

    pthread_mutex_lock(&resFileMutex);
    fprintf(res, "%s", resultBuffer);
    fflush(res);
    pthread_mutex_unlock(&resFileMutex);
#endif
    free(packet_data);
    return 0;
}
#else
int mnnProcessPacket(uint8_t *packet_data, uint32_t data_len, uint32_t stream_count) {
    if(NULL == finishTimeFile){
        printf("Error: finishTimeFile is NULL and return\n");
        return 1;
    }
    thread::id threadId = this_thread::get_id();
    std::ostringstream oss;
    oss << threadId;
    auto iter = g_sessionMap.find(threadId);
    if(iter == g_sessionMap.end()){
        MNN::ScheduleConfig netConfig;
        netConfig.numThread = 1;
        std::unique_lock<std::mutex> lock(mnnMutex);
        g_sessionMap[threadId] = g_mnnNet->createSession(netConfig);

        DEBUGPRINTF("ThreadId %s add session:%p\n", oss.str().c_str(), g_sessionMap[threadId]);
    }
    MNN::Session * session = g_sessionMap[threadId];
    stringstream sin;
    sin << threadId;
    DEBUGPRINTF("ThreadId %s use session:%p, with input %p length %d\n", oss.str().c_str(), session, packet_data, data_len);
    struct timespec  mnnStartTime = {0};
    clock_gettime(CLOCK_MONOTONIC_RAW, &mnnStartTime);
    MNN::Tensor *input = NULL; {
        std::unique_lock<std::mutex> lock(mnnMutex);
        input = g_mnnNet->getSessionInput(session, INPUT_NAME);
    }

    for (uint32_t i = 0; i < data_len; i++) {
        input->host<int>()[i] = packet_data[i];
    }

    // run session
    struct timespec cstart, ccpend;
    clock_gettime(CLOCK_MONOTONIC_RAW, &cstart);
    g_mnnNet->runSession(session);
    clock_gettime(CLOCK_MONOTONIC_RAW, &ccpend);
    double timeDiff1 = SubTimeSpecToSec(ccpend, cstart);{
        std::unique_lock<std::mutex> lock(mnnMutex);
        total_time += timeDiff1;
        g_MNNStreamsCount += stream_count;
    }
    // get output data
    MNN::Tensor *output = NULL;{
        std::unique_lock<std::mutex> lock(mnnMutex);
        output = g_mnnNet->getSessionOutput(session, OUTPUT_NAME);
    }
    auto output_host = make_shared < MNN::Tensor > (output, MNN::Tensor::CAFFE);
    output->copyToHostTensor(output_host.get());

    // Write process time to file
    struct timespec mnnEndTime = {0};
    clock_gettime(CLOCK_MONOTONIC_RAW, &mnnEndTime);
    struct timespec *packetRcvTime = (struct timespec*) (packet_data + MAX_SINGLE_MNN_LEN * MAX_MNN_STREAMS); {
        DEBUGPRINTF("write to file %p with buffer %p and value %ld.%ld - %ld.%ld\n", finishTimeFile, packetRcvTime, mnnEndTime.tv_sec, mnnEndTime.tv_nsec, packetRcvTime[0].tv_sec, packetRcvTime[0].tv_nsec);
        std::unique_lock<std::mutex> lock(mnnMutex);
        for (uint32_t i = 0; i < stream_count; i++) {
            fprintf(finishTimeFile, "%f,%f\n",  timeDiff1/stream_count,  SubTimeSpecToSec(mnnEndTime, packetRcvTime[i]));
        }
        fflush(finishTimeFile);
        total_mnn_time += SubTimeSpecToSec(mnnEndTime, mnnStartTime);
    }
    DEBUGPRINTF("ThreadId %s count %d start %ld.%ld end %ld.%ld core %f total_core %f total_mnn %f\n", oss.str().c_str(), g_MNNCount, mnnStartTime.tv_sec, mnnStartTime.tv_nsec, mnnEndTime.tv_sec, mnnEndTime.tv_nsec, timeDiff1, total_time, total_mnn_time);
    free(packet_data);
    return 0;
}
#endif


int getTrend(uint32_t *datas, uint32_t size, uint32_t index){
//    uint32_t avg = 0;

//    for(uint32_t i = 0; i < size; i++){
//        avg += datas[i];
//    }
//    avg = avg / size;
    int lastData = datas[(index + size - 1) % size];
    int firstData = datas[index];

    return lastData - firstData;
}

void writeDoubleListToFile(double * doubleList, uint32_t length, char * fileName){
    pthread_mutex_lock(&waitTimeMutex);
    FILE *fileHandle = fopen(fileName, "a+");
    if(fileHandle != NULL){
        for(uint32_t index = 0; index < length; index++){
            fprintf(fileHandle, "%lf\n", doubleList[index]);
        }
    }
    fflush(fileHandle);
    fclose(fileHandle);
    pthread_mutex_unlock(&waitTimeMutex);
}

#ifdef ETNAS
void*  EtnasThread(void * args){
    extern struct stream_list_indicate g_streamListHeader, g_streamListEnder;
    g_mnnNetIndex = (startModelNo + endModelNo) / 2;
    uint32_t trendList[3] = {g_mnnNetIndex, g_mnnNetIndex, g_mnnNetIndex};
    uint32_t trendIndex = 0;
    uint32_t firstPeriod = 0;
    double waitTimeList[MAX_WAITTIMELIST_SIZE] = {0.0};
    uint32_t waitIndex = 0;

    int threadId = *(int *)args;
    int totalCpuCount = std::thread::hardware_concurrency();
    int hostCpuCore = totalCpuCount - (threadId % totalCpuCount) - 1;
    // 设置CPU亲和属性
    // TODO：采取普适性更好的适配算法
    cpu_set_t mask;
    CPU_ZERO( &mask );
    CPU_SET(hostCpuCore, &mask );
    if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
        printf("WARNING: Could not set CPU Affinity %d for thread %ld, continuing...\n", hostCpuCore, threadId);
    }

    MNN::Session * g_sessions[MAX_MODEL_COUNT];
    for(uint32_t i = 0; i < MAX_MODEL_COUNT; i++){
        MNN::ScheduleConfig netConfig;
        netConfig.numThread = 1;
        std::unique_lock<std::mutex> lock(mnnMutex);
        g_sessions[i] = g_mnnNets[i]->createSession(netConfig);
    }
    int mnnNetIndex = g_mnnNetIndex;
    int tempEndModelNo = endModelNo;

    while(g_flag){
        if(g_streamListHeader.size == 0){
            // write wait time to result file first
            if(waitIndex > 0){
                writeDoubleListToFile(waitTimeList, waitIndex, WAITTIME_FILE_NAME);
                waitIndex = 0;
            }

            struct timeval delay;
            delay.tv_sec = 0;
            delay.tv_usec = 10 * 1000; // delay for 10 ms
            select(0, NULL, NULL, NULL, &delay);
        }else{
            struct stream_list_node * currStream = NULL;

            // Critical region for global stream_list visit
            pthread_mutex_lock(&g_streamListHeader.mutex);
            if(g_streamListHeader.size != 0){
                currStream = g_streamListEnder.next;
                g_streamListEnder.next = currStream->priv;
                if(NULL != g_streamListEnder.next){
                    g_streamListEnder.next->next = NULL;
                }

                trendList[trendIndex] = g_streamListHeader.size;
                trendIndex++;
                if(trendIndex >= MNN_TREND_INTERVAL){
                    trendIndex = 0;
                }
                g_streamListHeader.size -= 1;
            }
            pthread_mutex_unlock(&g_streamListHeader.mutex);

            if(NULL == currStream){
                continue;
            }
            firstPeriod ++;
            if(firstPeriod > 10){
                tempEndModelNo = endModelNo - g_streamListHeader.size / 100;
                if(tempEndModelNo < startModelNo || tempEndModelNo > endModelNo){
                    tempEndModelNo = startModelNo;
                }
                int trend = getTrend(trendList, MNN_TREND_INTERVAL, trendIndex);
                if(trend >= 1 && mnnNetIndex > startModelNo){
                    mnnNetIndex--;
                    DEBUGPRINTF("[%d]Retreat to an imprecise model[%d]\n", firstPeriod, mnnNetIndex);
                }else if(trend < 0 && mnnNetIndex < tempEndModelNo){
                    mnnNetIndex++;
                    DEBUGPRINTF("[%d]Climb to a more accurate model[%d]\n", firstPeriod, mnnNetIndex);
                }
            }

            struct timespec currentClock;
            clock_gettime(CLOCK_MONOTONIC_RAW, &currentClock);
            double waitTime = SubTimeSpecToSec(currentClock, currStream->shelfTime);
            waitTimeList[waitIndex] = waitTime;

            waitIndex ++;
            if(waitIndex == MAX_WAITTIMELIST_SIZE){
                writeDoubleListToFile(waitTimeList, waitIndex, WAITTIME_FILE_NAME);
                waitIndex = 0;
            }

            pthread_mutex_lock(&g_streamListHeader.mutex);
            g_modelSelect[mnnNetIndex] ++;
            g_streamCount++;
            pthread_mutex_unlock(&g_streamListHeader.mutex);

            EtnasProcessPacket(g_mnnNets[mnnNetIndex], g_sessions[mnnNetIndex], currStream->pBuffer, currStream->size, currStream->streamCount, currStream->key);

            free(currStream);
        }
    }
    return NULL;
}
#else
void*  EtnasThread(void * args){
    while(true){
        sleep(1000);
    }
}
#endif
