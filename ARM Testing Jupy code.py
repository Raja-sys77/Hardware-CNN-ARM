import cv2
import numpy as np
import time
import threading
from pynq import Overlay, allocate
import IPython.display

# ==========================================
# 1. HARDWARE (PL)
# ==========================================
print("⚡ Loading Bitstream...")
ol = Overlay("design_1.bit")
dma = ol.axi_dma_0
accel = ol.cnn_accelerator_0

# Configure IP
H_IP, W_IP = 192, 192
accel.register_map.rows = H_IP
accel.register_map.cols = W_IP
accel.register_map.CTRL.AP_START = 1
accel.register_map.CTRL.AUTO_RESTART = 1
weights_buf = allocate(shape=(50,), dtype=np.int32)
accel.register_map.Memory_weights = weights_buf.device_address

# DMA Buffers
input_buffer = allocate(shape=(H_IP, W_IP, 4), dtype=np.uint8)
output_buffer = allocate(shape=(H_IP, W_IP, 4), dtype=np.uint8)

print("✅ FPGA Ready.")

# ==========================================
# 2. AI MODEL (CPU)
# ==========================================
print("🧠 Loading MobileNet-SSD...")
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# ==========================================
# 3. SHARED DATA
# ==========================================
frame_shared = None
detections_shared = None
pl_metric_shared = 0
ai_fps = 0.0
pl_fps = 0.0
lock = threading.Lock()
running = True

# ==========================================
# 4. WORKER THREADS
# ==========================================

# --- THREAD A: AI (CPU) ---
def ai_worker():
    global frame_shared, detections_shared, ai_fps, running
    while running:
        with lock:
            if frame_shared is None: 
                time.sleep(0.01); continue
            local_frame = frame_shared.copy()
        
        t_start = time.time()
        # Resize to 300x300 for Model
        blob = cv2.dnn.blobFromImage(cv2.resize(local_frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        out = net.forward()
        t_end = time.time()
        
        with lock:
            detections_shared = out
            ai_fps = 1.0 / (t_end - t_start)
        
        # Sleep to let Video thread run
        time.sleep(0.1)

# --- THREAD B: FPGA (PL) ---
def pl_worker():
    global frame_shared, pl_metric_shared, pl_fps, running
    while running:
        with lock:
            if frame_shared is None: 
                time.sleep(0.01); continue
            local_frame = frame_shared.copy()
            
        t_start = time.time()
        
        # Hardware Process
        small = cv2.resize(local_frame, (W_IP, H_IP), interpolation=cv2.INTER_NEAREST)
        frame_rgba = cv2.cvtColor(small, cv2.COLOR_BGR2BGRA)
        
        input_buffer[:] = frame_rgba
        dma.sendchannel.transfer(input_buffer)
        dma.recvchannel.transfer(output_buffer)
        dma.sendchannel.wait()
        dma.recvchannel.wait()
        
        # Read Result
        result = output_buffer.copy()
        t_end = time.time()
        
        with lock:
            pl_fps = 1.0 / (t_end - t_start)
        
        # CRITICAL FIX: Sleep to cap PL at ~30 FPS
        # This prevents PL from eating all CPU bandwidth
        time.sleep(0.03)

# Start Threads
t1 = threading.Thread(target=ai_worker)
t2 = threading.Thread(target=pl_worker)
t1.start()
t2.start()
print("✅ Threads Started (Balanced Mode).")

# ==========================================
# 5. MAIN VIDEO LOOP (Optimized Display)
# ==========================================
# Use 320x240 for Capture if 640x360 is too slow on your network
gst_str = "v4l2src device=/dev/video0 ! image/jpeg, width=640, height=360, framerate=30/1 ! jpegdec ! videoconvert ! appsink drop=1"
cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("❌ Camera Fail.")
    running = False
    t1.join(); t2.join()
else:
    print("🚀 STARTING OPTIMIZED DEMO...")
    try:
        frame_count = 0
        display_counter = 0
        start_time = time.time()
        video_fps = 0
        
        while True:
            # 1. Capture (As fast as possible)
            ret, frame = cap.read()
            if not ret: break
            
            # Update Shared Frame
            with lock:
                frame_shared = frame.copy()
                dets = detections_shared
                val_ai_fps = ai_fps
                val_pl_fps = pl_fps
            
            # Calculate Capture FPS (The True Speed)
            frame_count += 1
            if time.time() - start_time > 1:
                video_fps = frame_count / (time.time() - start_time)
                frame_count = 0
                start_time = time.time()

            # 2. Display Strategy: Only show every 2nd or 3rd frame
            # This allows the loop to run faster than the browser can render
            display_counter += 1
            if display_counter % 2 == 0: 
                display = frame.copy()
                
                # Draw Boxes
                if dets is not None:
                    (h, w) = display.shape[:2]
                    for i in range(dets.shape[2]):
                        confidence = dets[0, 0, i, 2]
                        if confidence > 0.4:
                            idx = int(dets[0, 0, i, 1])
                            box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
                            (startX, startY, endX, endY) = box.astype("int")
                            cv2.rectangle(display, (startX, startY), (endX, endY), (0, 255, 0), 2)
                            label = f"{CLASSES[idx]}"
                            cv2.putText(display, label, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Stats
                cv2.putText(display, f"Video: {int(video_fps)} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display, f"AI: {val_ai_fps:.1f} FPS", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(display, f"PL: {val_pl_fps:.1f} FPS", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
                
                # Send to Browser
                _, fmt = cv2.imencode('.jpg', display)
                IPython.display.display(IPython.display.Image(data=fmt))
                IPython.display.clear_output(wait=True)
            
    except KeyboardInterrupt:
        print("🛑 Stopped.")
    finally:
        running = False
        t1.join(); t2.join()
        cap.release()
        input_buffer.freebuffer(); output_buffer.freebuffer(); weights_buf.freebuffer()
