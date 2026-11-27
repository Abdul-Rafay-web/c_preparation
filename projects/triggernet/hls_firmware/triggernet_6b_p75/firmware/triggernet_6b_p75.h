#ifndef TRIGGERNET_6B_P75_H_
#define TRIGGERNET_6B_P75_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"


// Prototype of top level function for C-synthesis
void triggernet_6b_p75(
    input_t input_1[16],
    result_t layer8_out[5]
);

// hls-fpga-machine-learning insert emulator-defines


#endif
