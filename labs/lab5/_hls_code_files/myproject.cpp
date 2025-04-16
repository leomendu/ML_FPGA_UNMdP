#include <iostream>

#include "myproject.h"
#include "parameters.h"

void inference(
	hls::stream<axis_int_t>& input,
	int *result
) {

    // #pragma HLS INTERFACE mode=s_axilite port=return
    #pragma HLS INTERFACE mode=ap_ctrl_hs port=return
    #pragma HLS INTERFACE axis register both port=input
    #pragma HLS INTERFACE ap_vld port=result register
    // #pragma HLS PIPELINE

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<weight2_t, 3136>(w2, "w2.txt");
        nnet::load_weights_from_txt<bias2_t, 4>(b2, "b2.txt");
        nnet::load_weights_from_txt<weight5_t, 12>(w5, "w5.txt");
        nnet::load_weights_from_txt<bias5_t, 3>(b5, "b5.txt");
        nnet::load_weights_from_txt<weight8_t, 6>(w8, "w8.txt");
        nnet::load_weights_from_txt<bias8_t, 2>(b8, "b8.txt");
        nnet::load_weights_from_txt<weight11_t, 20>(w11, "w11.txt");
        nnet::load_weights_from_txt<bias11_t, 10>(b11, "b11.txt");
        loaded_weights = true;
    }
#endif

    input_t fc1_input_input[N_INPUT_1_1];
    result_t layer13_out[N_LAYER_11];

    axis_int_t val;

    for(int h=0; h<N_INPUT_1_1; h++){

    #pragma HLS PIPELINE

    			// Read and cache value
    			val = input.read();
    			fc1_input_input[h] = val.data/255;

    		}

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    layer2_t layer2_out[N_LAYER_2];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    nnet::dense<input_t, layer2_t, config2>(fc1_input_input, layer2_out, w2, b2); // fc1

    layer4_t layer4_out[N_LAYER_2];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
    nnet::linear<layer2_t, layer4_t, linear_config4>(layer2_out, layer4_out); // relu1

    layer5_t layer5_out[N_LAYER_5];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    nnet::dense<layer4_t, layer5_t, config5>(layer4_out, layer5_out, w5, b5); // fc2

    layer7_t layer7_out[N_LAYER_5];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    nnet::linear<layer5_t, layer7_t, linear_config7>(layer5_out, layer7_out); // relu2

    layer8_t layer8_out[N_LAYER_8];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0
    nnet::dense<layer7_t, layer8_t, config8>(layer7_out, layer8_out, w8, b8); // fc3

    layer10_t layer10_out[N_LAYER_8];
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0
    nnet::linear<layer8_t, layer10_t, linear_config10>(layer8_out, layer10_out); // relu3

    layer11_t layer11_out[N_LAYER_11];
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0
    nnet::dense<layer10_t, layer11_t, config11>(layer10_out, layer11_out, w11, b11); // output

    nnet::softmax<layer11_t, result_t, softmax_config13>(layer11_out, layer13_out); // softmax

    std::cout << "0: " << layer11_out[0] << std::endl;
    std::cout << "1: " << layer11_out[1] << std::endl;
    std::cout << "2: " << layer11_out[2] << std::endl;
    std::cout << "3: " << layer11_out[3] << std::endl;
    std::cout << "4: " << layer11_out[4] << std::endl;
    std::cout << "5: " << layer11_out[5] << std::endl;
    std::cout << "6: " << layer11_out[6] << std::endl;
    std::cout << "7: " << layer11_out[7] << std::endl;
    std::cout << "8: " << layer11_out[8] << std::endl;
    std::cout << "9: " << layer11_out[9] << std::endl;

    // Decision
	if(layer13_out[0] >= 0.5){
		*result = 0;
	} else if(layer13_out[1] >= 0.5){
		*result = 1;
	} else if(layer13_out[2] >= 0.5){
		*result = 2;
	} else if(layer13_out[3] >= 0.5){
		*result = 3;
	} else if(layer13_out[4] >= 0.5){
		*result = 4;
	} else if(layer13_out[5] >= 0.5){
		*result = 5;
	} else if(layer13_out[6] >= 0.5){
		*result = 6;
	} else if(layer13_out[7] >= 0.5){
		*result = 7;
	} else if(layer13_out[8] >= 0.5){
		*result = 8;
	} else {
		*result = 9;
	}

}
