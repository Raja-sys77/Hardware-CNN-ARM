#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <ap_int.h>

// ==========================================
// INTERNAL DEFINITIONS (NO HEADER FILE)
// ==========================================
#define MAX_WIDTH 416   
#define K 3             // Kernel Size (3x3)

typedef ap_axis<32,1,1,1> axis_t;
typedef ap_int<8> data_t;      // Input Pixel
typedef ap_int<8> weight_t;    // Weights
typedef ap_int<16> res_t;      // Result

// ==========================================
// ACCELERATOR LOGIC
// ==========================================
void cnn_accelerator(hls::stream<axis_t>& in_stream, 
                     hls::stream<axis_t>& out_stream, 
                     int rows, 
                     int cols, 
                     weight_t weights[K][K]) {
    
    // INTERFACES
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE s_axilite port=rows bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=cols bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=weights bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    // INTERNAL BUFFERS
    // Line Buffer stores previous rows (sized for full 416 width)
    data_t line_buf[K-1][MAX_WIDTH];
    #pragma HLS ARRAY_PARTITION variable=line_buf dim=1 complete
    #pragma HLS RESOURCE variable=line_buf core=RAM_2P_BRAM
    
    // Window stores the 3x3 grid
    data_t window[K][K];
    #pragma HLS ARRAY_PARTITION variable=window complete

    // Local Weight Cache
    weight_t local_weights[K][K];
    #pragma HLS ARRAY_PARTITION variable=local_weights complete
    
    // Load weights
    for(int i=0; i<K; i++) {
        for(int j=0; j<K; j++) {
            local_weights[i][j] = weights[i][j];
        }
    }

    // MAIN LOOP
    for(int y = 0; y < rows; y++) {
        for(int x = 0; x < cols; x++) {
            #pragma HLS PIPELINE II=1

            // 1. READ INPUT
            axis_t temp_in = in_stream.read();
            data_t pixel_in = temp_in.data;

            // 2. SHIFT BUFFERS
            for(int i=0; i<K; i++) {
                for(int j=0; j<K-1; j++) {
                    window[i][j] = window[i][j+1];
                }
            }
            window[0][K-1] = line_buf[0][x];
            window[1][K-1] = line_buf[1][x];
            window[2][K-1] = pixel_in;

            line_buf[0][x] = line_buf[1][x];
            line_buf[1][x] = pixel_in;

            // 3. MATH (Convolution)
            res_t sum = 0;
            // Only compute valid windows
            if (y >= K-1 && x >= K-1) {
                for(int i=0; i<K; i++) {
                    for(int j=0; j<K; j++) {
                        sum += window[i][j] * local_weights[i][j];
                    }
                }
            }

            // 4. WRITE OUTPUT
            axis_t temp_out;
            temp_out.data = sum; 
            temp_out.keep = temp_in.keep;
            temp_out.strb = temp_in.strb;
            temp_out.dest = temp_in.dest;
            temp_out.id   = temp_in.id;
            temp_out.user = temp_in.user;
            
            // Correct TLAST Logic
            temp_out.last = (y == rows-1 && x == cols-1) ? 1 : 0;

            out_stream.write(temp_out);
        }
    }
}