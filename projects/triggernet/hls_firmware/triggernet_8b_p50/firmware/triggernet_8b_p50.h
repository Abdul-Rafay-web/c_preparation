#ifndef TRIGGERNET_8B_P50_H_
#define TRIGGERNET_8B_P50_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"


// Prototype of top level function for C-synthesis
void triggernet_8b_p50(
    input_t input_1[16],
    result_t layer8_out[5]
);

// hls-fpga-machine-learning insert emulator-defines


#endif
