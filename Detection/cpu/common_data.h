#ifndef COMMON_DATA_H
#define COMMON_DATA_H

#define STREAM_TIMEOUT            3            // 30s
#define MAX_RINGBUFFER_SIZE     384000        // 1GB
#define MAX_RCV_PKT_BUF_LENGTH  156
#define RCV_PKT_APPEND_HEAD_LEN    sizeof(struct timespec)    // 数据包头添加时间戳
#define MAX_RCV_PKT_LENGTH      (MAX_RCV_PKT_BUF_LENGTH - RCV_PKT_APPEND_HEAD_LEN)
#define CUR_RCV_PKT_LENGTH      138         // DA(6) + SA(6) + VLAN(4) + ETYPE(2) + IP(max 60) + TCP(max 60)

#define VLAN_TYPE           0x8100
#define MAC_LENGTH          12
#define MAX_SINGLE_MNN_LEN     128        // 单条流输入数据大小
#define MAX_MNN_STREAMS        1         // 单次输入MNN的流条数
#define THREASHOLD_LENGTH      252
#define MIN_PACKET_LENGTH      64
#define HASH_BUCKET_MASK       0x7F
#define HASH_BUCKET_SIZE       128     // 首级和次级BUCKET同大小

#define MNN_THREAD_COUNT    6       // MNN处理线程数量
#define CLASSES_SIZE        10      // MNN输出结果长度

#define MNN_STREAM_MAXPACKET 4      // 每条流统计多少个报文
#define MAX_MODEL_COUNT      33     // 模型数量
#define MNN_TREND_INTERVAL   3      // 队列趋势判断区间

#define WRITE_MNN_RESULT     0      // 是否将MNN结果写入文件

#define NORMALIZATION_VARIANCE 0
#define NORMALIZATION_MINMAX   1



// BloomFilter参数
#define BF_MAX_STREAMS      320000
#define BF_CONFLICT_RATE    0.000001

#ifndef IPPROTO_TCP 
#define IPPROTO_TCP     6
#endif

#ifndef IPPROTO_UDP 
#define IPPROTO_UDP     17
#endif

#ifdef DEBUG
    #define DEBUGPRINTF(format, ...) printf (format, ##__VA_ARGS__)
#else
    #define DEBUGPRINTF(format, ...)
#endif

inline double SubTimeSpecToSec(struct timespec endTime, struct timespec startTime){
    long timeDiff = 0.0;
    if(endTime.tv_sec < startTime.tv_sec){
        return timeDiff;
    }
    timeDiff = (endTime.tv_sec - startTime.tv_sec) * 1000000;
    timeDiff += endTime.tv_nsec/1000;
    timeDiff -= startTime.tv_nsec/1000;

    return (double)timeDiff/1000000;
}
#endif
