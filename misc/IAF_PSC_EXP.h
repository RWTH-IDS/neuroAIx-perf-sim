#pragma once

#include "Config.h"

void IAF_PSC_EXP(
  // params
  ap_uint<5> weight_frac_exc,
  ap_uint<5> weight_frac_inh,
  ap_uint<16> t_r,
  TYPE_STATE V_thresh,
  TYPE_STATE V_reset,
  TYPE_STATE P11exc,
  TYPE_STATE P11inh,
  TYPE_STATE P20,
  TYPE_STATE P21exc,
  TYPE_STATE P21inh,
  TYPE_STATE P22,
  // weights
  hls::stream<weight_axis> *input_weight_axis,
  // neuron state
  hls::stream<state_axis> *input_state_axis,
  hls::stream<state_axis> *output_state_axis,
  // spiking neuron
  hls::stream<out_axis> *m_axis);
