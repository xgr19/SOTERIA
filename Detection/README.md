
## Code Architecture

```
-- CPU
	-- ringbuffer.cpp/ringbuffer.h：Encapsulation of ringbuffer data structure. The main purpose is to cache the packets sent from P4 switching chip.
	
	-- bloomfilter.cpp/bloomfilter.h: Encapsulation of bloomfilter data structure. The main purpose is to filter processed packets quickly. Only the first three packets from the same flow/stream will only be inferenced by the CPU.
	
	-- packet_receive.cpp/packet_receive.h：Receive packets sent from the P4 switching chip and store them in the ringbuffer. 
	
	-- packet_process.cpp/packet_process.h：Fetch a packet from the ringbuffer and then filter it using BloomFilter. Determing the flow characteristics of the packet. If the conditions are met, the packet will sent to MNN_process engine for further processing.
	
	-- packet_timer.cpp/packet_timer.h: The timing module consists of two parts: 1) part 1 determines whether the current stored data has timed out. If so, they will be sent to the MNN engine for processing，even if less than three packets are collected for the current flow; 2) part 2 monitors and collects statistics of the resources occupied by the system.
	
	-- MNN_process.cpp/MNN_process.h: The MNN inference engine uses a thread pool to improve the system performance. It mainly predicts the content of the incoming data flow, determines whether the packet is malicious, and outputs the inference result.

```

## compile CPU code:
- Download MNN source code and compile it on P4 switch  
- Compile CPU code, CPU code and MNN in the same directory by default. If MNN needs to be placed in another path, modify the MNN_ROOT macro in the Makefile under CPU code to point it to the new path.
```
	cd <your_cpu_code_path>
	make
```

## run cpu program:
- Set up the Running Environment：
```
	export LD_LIBRARY_PATH=<path_of_MNN>/build:$LD_LIBRARY_PATH // For CPU programs to find MNN's lib library
	cd $SDE/build/bf-drivers/kdrv/bf_kpkt		
	insmod bf_kpkt.ko kpkt_rx_count=4096 kpkt_mode=1            // H3C's TOFINO switch specificically requires replacing the kernel module of the switch in order to receive the packets sent by the switch chip
	sysctl -w net.core.rmem_max=268435456                       // Increase the cache size of the kernel protocol stack to avoid packet loss in the case of outburst of packets
```
- Copy MNN model file to your cpu code path
- Run cpu program
```
	cd <your_cpu_code_path>
	./IntensiveModule eth1		// Should change eth1 according to the switch environment.
```
