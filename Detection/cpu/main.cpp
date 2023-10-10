#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <signal.h>
#include <stdlib.h>
#include <string>
#include "common_data.h"
#include "moniter.h"
#include "MNN_process.h"
#include "packet_process.h"
#include "packet_timer.h"
#include "packet_receive.h"

extern std::string g_modelList[MAX_MODEL_COUNT];
extern uint32_t g_modelSelect[MAX_MODEL_COUNT];
extern FILE *prob;

extern int initRingBuffer(void);
extern bool initPacketBloomFilter();
extern bool initDHashBucket();
extern void destroyRingBuffer(void);
extern bool freePacketBuffer();
extern bool freeDHashBucket();
extern int EtnasThread();
extern int g_streamCount;
int g_flag=1;

int g_threadID[MNN_THREAD_COUNT] = {0};
int startModelNo = 0;
int endModelNo = MAX_MODEL_COUNT - 1;

static void sig_usr(int signum){
    g_flag=0;

    printf("Total process stream: %d\n", g_streamCount);
    if(prob != NULL){
        fprintf(prob, "model use time: \n");
        for(int index = 0; index < MAX_MODEL_COUNT; index++){
            fprintf(prob, "%s, %d\n", g_modelList[index].c_str(), g_modelSelect[index]);
        }
    }
    fflush(prob);
}


void init_sigal_handle(struct sigaction *sa_usr){
    sa_usr->sa_flags = 0;
    sa_usr->sa_handler = sig_usr;   //信号处理函数
   
    sigaction(SIGINT, sa_usr, NULL);
}


/***********************
主函数
***********************/
int main(int argc, char **argv) {
    int iRet = 0;
    pthread_t receivePktThread, processPktThread, etnasThread[MNN_THREAD_COUNT], timerThread, moniterThread;
    struct sigaction sa_usr = {0};

    if (argc != 4) {
        printf("please enter eth_interface_name startModelNo endModelNo\n");
        return -1;
    }
    startModelNo = atoi(argv[2]);
    endModelNo = atoi(argv[3]);
    if(startModelNo >= MAX_MODEL_COUNT || endModelNo >= MAX_MODEL_COUNT){
        printf("startModelNo and endModelNo should small than %d\n", MAX_MODEL_COUNT);
        return -1;
    }

    init_sigal_handle(&sa_usr);

    iRet = initRingBuffer();
    if (iRet < 0) {
        printf("create buffer is error\n");
        return -1;
    }

    if (!initPacketBloomFilter()) {
        printf("Run out of memory! turn down bloom filter parameter\n");
        return -1;
    }
    if (!initDHashBucket()) {
        printf("Run out of memory! turn down hash bucket parameter\n");
        return -1;
    }

    initialMNN();

    if (pthread_create(&processPktThread, NULL, packetProcess, NULL) != 0) {
        printf("thread packetProcess create failed\n");
        return -1;
    }

    for(int index = 0; index < MNN_THREAD_COUNT; index++){
        g_threadID[index] = index;
        if (pthread_create(&etnasThread[index], NULL, EtnasThread, (void *)(g_threadID + index)) != 0) {
            pthread_cancel(processPktThread);
            printf("thread monitorProcess create failed\n");
            return -1;
        }
    }

//    sleep(3);

    if (pthread_create(&receivePktThread, NULL, packetRecv, argv[1]) != 0) {
        for(int index = 0; index < MNN_THREAD_COUNT; index++){
            pthread_cancel(etnasThread[index]);
        }
        pthread_cancel(processPktThread);
        printf("thread packetRecv create failed\n");
        return -1;
    }
#if WRITE_MNN_RESULT
    if (pthread_create(&timerThread, NULL, timerProcess, NULL) != 0) {
        pthread_cancel(receivePktThread);
        pthread_cancel(processPktThread);
        for(int index = 0; index < MNN_THREAD_COUNT; index++){
            pthread_cancel(etnasThread[index]);
        }
        printf("thread timerProcess create failed \n");
        return -1;
    }

    if (pthread_create(&moniterThread, NULL, monitorProcess, argv[1]) != 0) {
        pthread_cancel(timerThread);
        for(int index = 0; index < MNN_THREAD_COUNT; index++){
            pthread_cancel(etnasThread[index]);
        }
        pthread_cancel(processPktThread);
        pthread_cancel(receivePktThread);
        printf("thread monitorProcess create failed\n");
        return -1;
    }
#endif

    pthread_join(receivePktThread, NULL);
    pthread_join(processPktThread, NULL);
    for(int index = 0; index < MNN_THREAD_COUNT; index++){
        pthread_join(etnasThread[index], NULL);
    }

#if WRITE_MNN_RESULT
    pthread_join(timerThread, NULL);
    pthread_join(moniterThread, NULL);
#endif

    destroyRingBuffer();
    freeDHashBucket();
    freePacketBuffer();
    freeMNN();
    freeMoniterResource();
    return 1;

}
