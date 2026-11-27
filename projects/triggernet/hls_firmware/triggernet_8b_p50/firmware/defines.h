#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <array>
#include <cstddef>
#include <cstdio>
#include <tuple>
#include <tuple>


// hls-fpga-machine-learning insert numbers

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<16,6> _0_accum_t;
typedef ap_fixed<16,6> layer2_t;
typedef ap_fixed<8,2> _0_weight_t;
typedef ap_fixed<8,2> _0_bias_t;
typedef ap_uint<1> layer2_index;
typedef ap_ufixed<8,3> layer3_t;
typedef ap_fixed<18,8> _1_table_t;
typedef ap_fixed<16,6> _2_accum_t;
typedef ap_fixed<16,6> layer4_t;
typedef ap_fixed<8,2> _2_weight_t;
typedef ap_fixed<8,2> _2_bias_t;
typedef ap_uint<1> layer4_index;
typedef ap_ufixed<8,3> layer5_t;
typedef ap_fixed<18,8> _3_table_t;
typedef ap_fixed<16,6> _4_accum_t;
typedef ap_fixed<16,6> layer6_t;
typedef ap_fixed<8,2> _4_weight_t;
typedef ap_fixed<8,2> _4_bias_t;
typedef ap_uint<1> layer6_index;
typedef ap_ufixed<8,3> layer7_t;
typedef ap_fixed<18,8> _5_table_t;
typedef ap_fixed<16,6> _6_accum_t;
typedef ap_fixed<16,6> result_t;
typedef ap_fixed<8,2> _6_weight_t;
typedef ap_fixed<8,2> _6_bias_t;
typedef ap_uint<1> layer8_index;

// hls-fpga-machine-learning insert emulator-defines


#endif
