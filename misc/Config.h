#pragma once

#include "ap_int.h"
#include "hls_stream.h"
#include "ap_axi_sdata.h"

typedef float TYPE_WEIGHT;
typedef float TYPE_STATE;
const int DATA_WIDTH_WEIGHT = sizeof(TYPE_WEIGHT)*8;
const int DATA_WIDTH_STATE = sizeof(TYPE_STATE)*8;

const float h = 0.1; //ms

struct weight_axis{
  ap_uint<DATA_WIDTH_WEIGHT> w_exc;
  ap_uint<DATA_WIDTH_WEIGHT> w_inh;
};

struct out_axis{
  ap_uint<1> spike;
};

struct state_axis{
  TYPE_STATE I_exc;
  TYPE_STATE I_inh;
  TYPE_STATE V_m;
  ap_uint<32> t_r; // TODO: could be 16b, but Vitis HLS always interprets this as 32b >:(
  TYPE_STATE I_ext;
};
