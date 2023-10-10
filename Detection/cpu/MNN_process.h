/*
 * MNN_process.h
 *
 *  Created on: 2022年2月10日
 *      Author: xiegr19
 */

#ifndef MNN_PROCESS_H_
#define MNN_PROCESS_H_

#include <stdint.h>

#define WAITTIME_FILE_NAME      "waittime.list"
#define MAX_WAITTIMELIST_SIZE   1000        // 最大连续处理1000条流之后，将等待时间写入结果文件
#define PRINT_BUFFER_SIZE       65500       // 缓存字符串buffer的大小
#define SNPRINTFTOFILE(format, ...) resultBufferSize += snprintf(resultBuffer + resultBufferSize, PRINT_BUFFER_SIZE - resultBufferSize, format, ##__VA_ARGS__)


// 初始化MNN处理模块
void initialMNN();

// 释放MNN处理模块的资源
void freeMNN();

// 对报文进行MNN推理
int mnnProcessPacket(uint8_t *packet_data, uint32_t data_len, uint32_t stream_count);

// Etnas模式，不通过线程池，而是固定执行的方式
void*  EtnasThread(void * args);

#endif /* MNN_PROCESS_H_ */
