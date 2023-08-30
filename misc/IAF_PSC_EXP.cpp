#include "IAF_PSC_EXP.h"

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
  hls::stream<out_axis> *m_axis) {
  // constrain pipeline depth into (20,30) at the expense of frequency
#pragma HLS LATENCY min=20 max=30
  // force pipeline initiation interval (II) to be 1 cycles -> every cycle ready to read new data
#pragma HLS PIPELINE II=1
  // define axi stream ports
#pragma HLS INTERFACE axis register port=input_weight_axis
#pragma HLS INTERFACE axis register port=input_state_axis
#pragma HLS INTERFACE axis register port=output_state_axis
#pragma HLS INTERFACE axis register port=m_axis

  weight_axis weight_in;
  TYPE_WEIGHT w_exc, w_inh;
  state_axis state_in, state_out;
  out_axis spike_out;

  // read
  input_weight_axis->read(weight_in);
  input_state_axis->read(state_in);

  w_exc = ((TYPE_WEIGHT)weight_in.w_exc) / ((TYPE_WEIGHT)(1<<weight_frac_exc));
  w_inh = ((TYPE_WEIGHT)weight_in.w_inh) / ((TYPE_WEIGHT)(1<<weight_frac_inh));

  // compute
  state_out.I_exc = P11exc * state_in.I_exc + w_exc;
  state_out.I_inh = P11inh * state_in.I_inh + w_inh;
  state_out.t_r = state_in.t_r;
  state_out.I_ext = state_in.I_ext;

  if(state_in.t_r == 0) {  // refractory check
    state_out.V_m = P20*state_in.I_ext + P21exc*state_in.I_exc + P21inh*state_in.I_inh + P22*state_in.V_m;
    if(state_out.V_m >= V_thresh) {  // if spike: output spiking neuron ID
      state_out.t_r = t_r;
      state_out.V_m = V_reset;

      spike_out.spike = 1;
      m_axis->write(spike_out);
    }
  } else {
    state_out.V_m = state_in.V_m;
    state_out.t_r = state_in.t_r - 1;  // decrease refractory time
  }

  // write
  output_state_axis->write(state_out);
}
