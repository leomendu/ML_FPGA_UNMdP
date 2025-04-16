-makelib xcelium_lib/xilinx_vip -sv \
  "/tools/Xilinx/Vivado/2022.2/data/xilinx_vip/hdl/axi4stream_vip_axi4streampc.sv" \
  "/tools/Xilinx/Vivado/2022.2/data/xilinx_vip/hdl/axi_vip_axi4pc.sv" \
  "/tools/Xilinx/Vivado/2022.2/data/xilinx_vip/hdl/xil_common_vip_pkg.sv" \
  "/tools/Xilinx/Vivado/2022.2/data/xilinx_vip/hdl/axi4stream_vip_pkg.sv" \
  "/tools/Xilinx/Vivado/2022.2/data/xilinx_vip/hdl/axi_vip_pkg.sv" \
  "/tools/Xilinx/Vivado/2022.2/data/xilinx_vip/hdl/axi4stream_vip_if.sv" \
  "/tools/Xilinx/Vivado/2022.2/data/xilinx_vip/hdl/axi_vip_if.sv" \
  "/tools/Xilinx/Vivado/2022.2/data/xilinx_vip/hdl/clk_vip_if.sv" \
  "/tools/Xilinx/Vivado/2022.2/data/xilinx_vip/hdl/rst_vip_if.sv" \
-endlib
-makelib xcelium_lib/xpm -sv \
  "/tools/Xilinx/Vivado/2022.2/data/ip/xpm/xpm_cdc/hdl/xpm_cdc.sv" \
  "/tools/Xilinx/Vivado/2022.2/data/ip/xpm/xpm_memory/hdl/xpm_memory.sv" \
-endlib
-makelib xcelium_lib/xpm \
  "/tools/Xilinx/Vivado/2022.2/data/ip/xpm/xpm_VCOMP.vhd" \
-endlib
-makelib xcelium_lib/xil_defaultlib \
  "../../../bd/MLZedboard/ipshared/7b64/hdl/axil.vhdl" \
  "../../../bd/MLZedboard/ipshared/7b64/hdl/axif.vhdl" \
  "../../../bd/MLZedboard/ipshared/7b64/hdl/tdpram.vhdl" \
  "../../../bd/MLZedboard/ipshared/7b64/hdl/graysync.vhdl" \
  "../../../bd/MLZedboard/ipshared/7b64/hdl/fifo.vhdl" \
  "../../../bd/MLZedboard/ipshared/7b64/hdl/comblock.vhdl" \
  "../../../bd/MLZedboard/ipshared/7b64/hdl/axi_comblock.vhdl" \
  "../../../bd/MLZedboard/ip/MLZedboard_comblock_0_0/sim/MLZedboard_comblock_0_0.vhd" \
-endlib
-makelib xcelium_lib/xbip_utils_v3_0_10 \
  "../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/364f/hdl/xbip_utils_v3_0_vh_rfs.vhd" \
-endlib
-makelib xcelium_lib/axi_utils_v2_0_6 \
  "../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/1971/hdl/axi_utils_v2_0_vh_rfs.vhd" \
-endlib
-makelib xcelium_lib/xbip_pipe_v3_0_6 \
  "../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/7468/hdl/xbip_pipe_v3_0_vh_rfs.vhd" \
-endlib
-makelib xcelium_lib/xbip_dsp48_wrapper_v3_0_4 \
  "../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/cdbf/hdl/xbip_dsp48_wrapper_v3_0_vh_rfs.vhd" \
-endlib
-makelib xcelium_lib/xbip_dsp48_addsub_v3_0_6 \
  "../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/910d/hdl/xbip_dsp48_addsub_v3_0_vh_rfs.vhd" \
-endlib
-makelib xcelium_lib/xbip_dsp48_multadd_v3_0_6 \
  "../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/b0ac/hdl/xbip_dsp48_multadd_v3_0_vh_rfs.vhd" \
-endlib
-makelib xcelium_lib/xbip_bram18k_v3_0_6 \
  "../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/d367/hdl/xbip_bram18k_v3_0_vh_rfs.vhd" \
-endlib
-makelib xcelium_lib/mult_gen_v12_0_18 \
  "../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/ab19/hdl/mult_gen_v12_0_vh_rfs.vhd" \
-endlib
-makelib xcelium_lib/floating_point_v7_1_15 \
  "../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/22f8/hdl/floating_point_v7_1_rfs.vhd" \
-endlib
-makelib xcelium_lib/floating_point_v7_1_15 \
  "../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/22f8/hdl/floating_point_v7_1_rfs.v" \
-endlib
-makelib xcelium_lib/xil_defaultlib \
  "../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/6fba/hdl/verilog/inference_dcmp_64ns_64ns_1_2_no_dsp_1.v" \
  "../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/6fba/hdl/verilog/inference_dense_latency_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_config2_s.v" \
  "../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/6fba/hdl/verilog/inference_dense_latency_ap_fixed_16_7_4_0_0_ap_fixed_16_6_5_3_0_config5_s.v" \
  "../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/6fba/hdl/verilog/inference_dense_latency_ap_fixed_16_7_4_0_0_ap_fixed_16_6_5_3_0_config8_s.v" \
  "../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/6fba/hdl/verilog/inference_dense_latency_ap_fixed_16_7_4_0_0_ap_fixed_16_6_5_3_0_config11_s.v" \
  "../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/6fba/hdl/verilog/inference_flow_control_loop_pipe_sequential_init.v" \
  "../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/6fba/hdl/verilog/inference_hls_deadlock_idx0_monitor.v" \
  "../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/6fba/hdl/verilog/inference_inference_Pipeline_VITIS_LOOP_41_1.v" \
  "../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/6fba/hdl/verilog/inference_linear_ap_fixed_16_6_5_3_0_ap_fixed_16_7_4_0_0_linear_config4_s.v" \
  "../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/6fba/hdl/verilog/inference_linear_ap_fixed_16_6_5_3_0_ap_fixed_16_7_4_0_0_linear_config7_s.v" \
  "../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/6fba/hdl/verilog/inference_linear_ap_fixed_16_6_5_3_0_ap_fixed_16_7_4_0_0_linear_config10_s.v" \
  "../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/6fba/hdl/verilog/inference_mul_16s_5ns_19_1_1.v" \
  "../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/6fba/hdl/verilog/inference_mul_16s_5s_19_1_1.v" \
  "../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/6fba/hdl/verilog/inference_mul_32s_34ns_65_1_1.v" \
  "../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/6fba/hdl/verilog/inference_regslice_both.v" \
  "../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/6fba/hdl/verilog/inference_sigmoid_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_sigmoid_config13_s.v" \
  "../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/6fba/hdl/verilog/inference_sigmoid_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_sigmoid_config13_s_sigmoid_tabkb.v" \
  "../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/6fba/hdl/verilog/inference.v" \
  "../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/6fba/hdl/ip/inference_dcmp_64ns_64ns_1_2_no_dsp_1_ip.v" \
  "../../../bd/MLZedboard/ip/MLZedboard_inference_0_0/sim/MLZedboard_inference_0_0.v" \
-endlib
-makelib xcelium_lib/axi_infrastructure_v1_1_0 \
  "../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/ec67/hdl/axi_infrastructure_v1_1_vl_rfs.v" \
-endlib
-makelib xcelium_lib/axi_vip_v1_1_13 -sv \
  "../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/ffc2/hdl/axi_vip_v1_1_vl_rfs.sv" \
-endlib
-makelib xcelium_lib/processing_system7_vip_v1_0_15 -sv \
  "../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/ee60/hdl/processing_system7_vip_v1_0_vl_rfs.sv" \
-endlib
-makelib xcelium_lib/xil_defaultlib \
  "../../../bd/MLZedboard/ip/MLZedboard_processing_system7_0_0/sim/MLZedboard_processing_system7_0_0.v" \
-endlib
-makelib xcelium_lib/generic_baseblocks_v2_1_0 \
  "../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/b752/hdl/generic_baseblocks_v2_1_vl_rfs.v" \
-endlib
-makelib xcelium_lib/fifo_generator_v13_2_7 \
  "../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/83df/simulation/fifo_generator_vlog_beh.v" \
-endlib
-makelib xcelium_lib/fifo_generator_v13_2_7 \
  "../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/83df/hdl/fifo_generator_v13_2_rfs.vhd" \
-endlib
-makelib xcelium_lib/fifo_generator_v13_2_7 \
  "../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/83df/hdl/fifo_generator_v13_2_rfs.v" \
-endlib
-makelib xcelium_lib/axi_data_fifo_v2_1_26 \
  "../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/3111/hdl/axi_data_fifo_v2_1_vl_rfs.v" \
-endlib
-makelib xcelium_lib/axi_register_slice_v2_1_27 \
  "../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/f0b4/hdl/axi_register_slice_v2_1_vl_rfs.v" \
-endlib
-makelib xcelium_lib/axi_protocol_converter_v2_1_27 \
  "../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/aeb3/hdl/axi_protocol_converter_v2_1_vl_rfs.v" \
-endlib
-makelib xcelium_lib/xil_defaultlib \
  "../../../bd/MLZedboard/ip/MLZedboard_auto_pc_0/sim/MLZedboard_auto_pc_0.v" \
-endlib
-makelib xcelium_lib/lib_cdc_v1_0_2 \
  "../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/ef1e/hdl/lib_cdc_v1_0_rfs.vhd" \
-endlib
-makelib xcelium_lib/proc_sys_reset_v5_0_13 \
  "../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/8842/hdl/proc_sys_reset_v5_0_vh_rfs.vhd" \
-endlib
-makelib xcelium_lib/xil_defaultlib \
  "../../../bd/MLZedboard/ip/MLZedboard_rst_ps7_0_100M_0/sim/MLZedboard_rst_ps7_0_100M_0.vhd" \
-endlib
-makelib xcelium_lib/xlconstant_v1_1_7 \
  "../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/badb/hdl/xlconstant_v1_1_vl_rfs.v" \
-endlib
-makelib xcelium_lib/xil_defaultlib \
  "../../../bd/MLZedboard/ip/MLZedboard_xlconstant_0_0/sim/MLZedboard_xlconstant_0_0.v" \
  "../../../bd/MLZedboard/ip/MLZedboard_xlconstant_1_0/sim/MLZedboard_xlconstant_1_0.v" \
  "../../../bd/MLZedboard/sim/MLZedboard.v" \
-endlib
-makelib xcelium_lib/xil_defaultlib \
  glbl.v
-endlib

