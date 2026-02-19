# Hardware-Accelerated-CNN-ARM-Zynq

![Xilinx](https://img.shields.io/badge/Xilinx-Zynq--7000-blue) ![Python](https://img.shields.io/badge/Python-3.8+-yellow) ![OpenCV](https://img.shields.io/badge/OpenCV-DNN-green) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

## Overview
This repository contains the source code and hardware bitstreams for a real-time, hardware-accelerated object detection system built on the Xilinx PYNQ-Z2 (Zynq-7000 SoC). Developed for the **Bharat AI-SoC Student Challenge**, the system bypasses traditional CPU bottlenecks by utilizing a **Heterogeneous Asynchronous Pipeline (HAP)**.



## Architecture
The workload is strictly partitioned between the Processing System (PS) and Programmable Logic (PL) to ensure non-blocking execution:
* **PS (ARM Cortex-A9):** Manages USB video acquisition, Python-based asynchronous threading, and MobileNet-SSD inference via OpenCV DNN (NEON optimized).
* **PL (Artix-7 FPGA):** Hosts a custom Vitis HLS IP core acting as a spatial feature extractor, processing pixels at hardware speed.
* **Interconnect:** Zero-copy AXI DMA transfers frame buffers directly between DDR3 memory and the FPGA, bypassing the CPU cache.

## Performance
| Metric | Sequential CPU Baseline | Heterogeneous Asynchronous (Proposed) |
| :--- | :--- | :--- |
| **Throughput** | ~2 FPS | **10+ FPS** |
| **System Latency** | >300ms | **<100ms** |
| **CPU Load** | 100% (Blocked) | **Parallel / Balanced** |

## Requirements
### Hardware
* TUL PYNQ-Z2 Development Board (Xilinx XC7Z020)
* USB Web Camera (e.g., Logitech C270)
* MicroSD Card (16GB+)

### Software
* PYNQ Linux Image (v2.7+)
* Python 3.8+
* `pynq`, `cv2`, `numpy`, `threading`

## Repository Structure
```text
├── hw/
│   ├── cnn_core.cpp and cnn_tb.cpp    # Vitis HLS source for custom IP
│   ├── design_1.bit             # Synthesized FPGA bitstream
│   └── design_1.hwh             # Hardware handoff file
├── sw/
│   ├── Final ARM juoy Codes.py            # Primary multi-threaded execution script
│   ├── MobileNetSSD_deploy.prototxt
│   └── MobileNetSSD_deploy.caffemodel
├── docs/
│   └── Vivado Block design.png # System block diagrams
└── README.md
