vlib work
vlib riviera

vlib riviera/xilinx_vip
vlib riviera/xpm
vlib riviera/xil_defaultlib
vlib riviera/xbip_utils_v3_0_10
vlib riviera/axi_utils_v2_0_6
vlib riviera/xbip_pipe_v3_0_6
vlib riviera/xbip_dsp48_wrapper_v3_0_4
vlib riviera/xbip_dsp48_addsub_v3_0_6
vlib riviera/xbip_dsp48_multadd_v3_0_6
vlib riviera/xbip_bram18k_v3_0_6
vlib riviera/mult_gen_v12_0_18
vlib riviera/floating_point_v7_1_15
vlib riviera/axi_infrastructure_v1_1_0
vlib riviera/axi_vip_v1_1_13
vlib riviera/processing_system7_vip_v1_0_15
vlib riviera/generic_baseblocks_v2_1_0
vlib riviera/fifo_generator_v13_2_7
vlib riviera/axi_data_fifo_v2_1_26
vlib riviera/axi_register_slice_v2_1_27
vlib riviera/axi_protocol_converter_v2_1_27
vlib riviera/lib_cdc_v1_0_2
vlib riviera/proc_sys_reset_v5_0_13
vlib riviera/xlconstant_v1_1_7

vmap xilinx_vip riviera/xilinx_vip
vmap xpm riviera/xpm
vmap xil_defaultlib riviera/xil_defaultlib
vmap xbip_utils_v3_0_10 riviera/xbip_utils_v3_0_10
vmap axi_utils_v2_0_6 riviera/axi_utils_v2_0_6
vmap xbip_pipe_v3_0_6 riviera/xbip_pipe_v3_0_6
vmap xbip_dsp48_wrapper_v3_0_4 riviera/xbip_dsp48_wrapper_v3_0_4
vmap xbip_dsp48_addsub_v3_0_6 riviera/xbip_dsp48_addsub_v3_0_6
vmap xbip_dsp48_multadd_v3_0_6 riviera/xbip_dsp48_multadd_v3_0_6
vmap xbip_bram18k_v3_0_6 riviera/xbip_bram18k_v3_0_6
vmap mult_gen_v12_0_18 riviera/mult_gen_v12_0_18
vmap floating_point_v7_1_15 riviera/floating_point_v7_1_15
vmap axi_infrastructure_v1_1_0 riviera/axi_infrastructure_v1_1_0
vmap axi_vip_v1_1_13 riviera/axi_vip_v1_1_13
vmap processing_system7_vip_v1_0_15 riviera/processing_system7_vip_v1_0_15
vmap generic_baseblocks_v2_1_0 riviera/generic_baseblocks_v2_1_0
vmap fifo_generator_v13_2_7 riviera/fifo_generator_v13_2_7
vmap axi_data_fifo_v2_1_26 riviera/axi_data_fifo_v2_1_26
vmap axi_register_slice_v2_1_27 riviera/axi_register_slice_v2_1_27
vmap axi_protocol_converter_v2_1_27 riviera/axi_protocol_converter_v2_1_27
vmap lib_cdc_v1_0_2 riviera/lib_cdc_v1_0_2
vmap proc_sys_reset_v5_0_13 riviera/proc_sys_reset_v5_0_13
vmap xlconstant_v1_1_7 riviera/xlconstant_v1_1_7

vlog -work xilinx_vip  -sv2k12 "+incdir+/tools/Xilinx/Vivado/2022.2/data/xilinx_vip/include" \
"/tools/Xilinx/Vivado/2022.2/data/xilinx_vip/hdl/axi4stream_vip_axi4streampc.sv" \
"/tools/Xilinx/Vivado/2022.2/data/xilinx_vip/hdl/axi_vip_axi4pc.sv" \
"/tools/Xilinx/Vivado/2022.2/data/xilinx_vip/hdl/xil_common_vip_pkg.sv" \
"/tools/Xilinx/Vivado/2022.2/data/xilinx_vip/hdl/axi4stream_vip_pkg.sv" \
"/tools/Xilinx/Vivado/2022.2/data/xilinx_vip/hdl/axi_vip_pkg.sv" \
"/tools/Xilinx/Vivado/2022.2/data/xilinx_vip/hdl/axi4stream_vip_if.sv" \
"/tools/Xilinx/Vivado/2022.2/data/xilinx_vip/hdl/axi_vip_if.sv" \
"/tools/Xilinx/Vivado/2022.2/data/xilinx_vip/hdl/clk_vip_if.sv" \
"/tools/Xilinx/Vivado/2022.2/data/xilinx_vip/hdl/rst_vip_if.sv" \

vlog -work xpm  -sv2k12 "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/6fba/hdl/verilog" "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/ec67/hdl" "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/ee60/hdl" "+incdir+/tools/Xilinx/Vivado/2022.2/data/xilinx_vip/include" \
"/tools/Xilinx/Vivado/2022.2/data/ip/xpm/xpm_cdc/hdl/xpm_cdc.sv" \
"/tools/Xilinx/Vivado/2022.2/data/ip/xpm/xpm_memory/hdl/xpm_memory.sv" \

vcom -work xpm -93  \
"/tools/Xilinx/Vivado/2022.2/data/ip/xpm/xpm_VCOMP.vhd" \

vcom -work xil_defaultlib -93  \
"../../../bd/MLZedboard/ipshared/7b64/hdl/axil.vhdl" \
"../../../bd/MLZedboard/ipshared/7b64/hdl/axif.vhdl" \
"../../../bd/MLZedboard/ipshared/7b64/hdl/tdpram.vhdl" \
"../../../bd/MLZedboard/ipshared/7b64/hdl/graysync.vhdl" \
"../../../bd/MLZedboard/ipshared/7b64/hdl/fifo.vhdl" \
"../../../bd/MLZedboard/ipshared/7b64/hdl/comblock.vhdl" \
"../../../bd/MLZedboard/ipshared/7b64/hdl/axi_comblock.vhdl" \
"../../../bd/MLZedboard/ip/MLZedboard_comblock_0_0/sim/MLZedboard_comblock_0_0.vhd" \

vcom -work xbip_utils_v3_0_10 -93  \
"../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/364f/hdl/xbip_utils_v3_0_vh_rfs.vhd" \

vcom -work axi_utils_v2_0_6 -93  \
"../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/1971/hdl/axi_utils_v2_0_vh_rfs.vhd" \

vcom -work xbip_pipe_v3_0_6 -93  \
"../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/7468/hdl/xbip_pipe_v3_0_vh_rfs.vhd" \

vcom -work xbip_dsp48_wrapper_v3_0_4 -93  \
"../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/cdbf/hdl/xbip_dsp48_wrapper_v3_0_vh_rfs.vhd" \

vcom -work xbip_dsp48_addsub_v3_0_6 -93  \
"../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/910d/hdl/xbip_dsp48_addsub_v3_0_vh_rfs.vhd" \

vcom -work xbip_dsp48_multadd_v3_0_6 -93  \
"../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/b0ac/hdl/xbip_dsp48_multadd_v3_0_vh_rfs.vhd" \

vcom -work xbip_bram18k_v3_0_6 -93  \
"../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/d367/hdl/xbip_bram18k_v3_0_vh_rfs.vhd" \

vcom -work mult_gen_v12_0_18 -93  \
"../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/ab19/hdl/mult_gen_v12_0_vh_rfs.vhd" \

vcom -work floating_point_v7_1_15 -93  \
"../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/22f8/hdl/floating_point_v7_1_rfs.vhd" \

vlog -work floating_point_v7_1_15  -v2k5 "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/6fba/hdl/verilog" "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/ec67/hdl" "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/ee60/hdl" "+incdir+/tools/Xilinx/Vivado/2022.2/data/xilinx_vip/include" \
"../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/22f8/hdl/floating_point_v7_1_rfs.v" \

vlog -work xil_defaultlib  -v2k5 "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/6fba/hdl/verilog" "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/ec67/hdl" "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/ee60/hdl" "+incdir+/tools/Xilinx/Vivado/2022.2/data/xilinx_vip/include" \
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

vlog -work axi_infrastructure_v1_1_0  -v2k5 "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/6fba/hdl/verilog" "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/ec67/hdl" "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/ee60/hdl" "+incdir+/tools/Xilinx/Vivado/2022.2/data/xilinx_vip/include" \
"../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/ec67/hdl/axi_infrastructure_v1_1_vl_rfs.v" \

vlog -work axi_vip_v1_1_13  -sv2k12 "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/6fba/hdl/verilog" "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/ec67/hdl" "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/ee60/hdl" "+incdir+/tools/Xilinx/Vivado/2022.2/data/xilinx_vip/include" \
"../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/ffc2/hdl/axi_vip_v1_1_vl_rfs.sv" \

vlog -work processing_system7_vip_v1_0_15  -sv2k12 "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/6fba/hdl/verilog" "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/ec67/hdl" "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/ee60/hdl" "+incdir+/tools/Xilinx/Vivado/2022.2/data/xilinx_vip/include" \
"../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/ee60/hdl/processing_system7_vip_v1_0_vl_rfs.sv" \

vlog -work xil_defaultlib  -v2k5 "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/6fba/hdl/verilog" "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/ec67/hdl" "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/ee60/hdl" "+incdir+/tools/Xilinx/Vivado/2022.2/data/xilinx_vip/include" \
"../../../bd/MLZedboard/ip/MLZedboard_processing_system7_0_0/sim/MLZedboard_processing_system7_0_0.v" \

vlog -work generic_baseblocks_v2_1_0  -v2k5 "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/6fba/hdl/verilog" "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/ec67/hdl" "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/ee60/hdl" "+incdir+/tools/Xilinx/Vivado/2022.2/data/xilinx_vip/include" \
"../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/b752/hdl/generic_baseblocks_v2_1_vl_rfs.v" \

vlog -work fifo_generator_v13_2_7  -v2k5 "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/6fba/hdl/verilog" "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/ec67/hdl" "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/ee60/hdl" "+incdir+/tools/Xilinx/Vivado/2022.2/data/xilinx_vip/include" \
"../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/83df/simulation/fifo_generator_vlog_beh.v" \

vcom -work fifo_generator_v13_2_7 -93  \
"../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/83df/hdl/fifo_generator_v13_2_rfs.vhd" \

vlog -work fifo_generator_v13_2_7  -v2k5 "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/6fba/hdl/verilog" "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/ec67/hdl" "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/ee60/hdl" "+incdir+/tools/Xilinx/Vivado/2022.2/data/xilinx_vip/include" \
"../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/83df/hdl/fifo_generator_v13_2_rfs.v" \

vlog -work axi_data_fifo_v2_1_26  -v2k5 "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/6fba/hdl/verilog" "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/ec67/hdl" "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/ee60/hdl" "+incdir+/tools/Xilinx/Vivado/2022.2/data/xilinx_vip/include" \
"../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/3111/hdl/axi_data_fifo_v2_1_vl_rfs.v" \

vlog -work axi_register_slice_v2_1_27  -v2k5 "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/6fba/hdl/verilog" "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/ec67/hdl" "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/ee60/hdl" "+incdir+/tools/Xilinx/Vivado/2022.2/data/xilinx_vip/include" \
"../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/f0b4/hdl/axi_register_slice_v2_1_vl_rfs.v" \

vlog -work axi_protocol_converter_v2_1_27  -v2k5 "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/6fba/hdl/verilog" "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/ec67/hdl" "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/ee60/hdl" "+incdir+/tools/Xilinx/Vivado/2022.2/data/xilinx_vip/include" \
"../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/aeb3/hdl/axi_protocol_converter_v2_1_vl_rfs.v" \

vlog -work xil_defaultlib  -v2k5 "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/6fba/hdl/verilog" "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/ec67/hdl" "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/ee60/hdl" "+incdir+/tools/Xilinx/Vivado/2022.2/data/xilinx_vip/include" \
"../../../bd/MLZedboard/ip/MLZedboard_auto_pc_0/sim/MLZedboard_auto_pc_0.v" \

vcom -work lib_cdc_v1_0_2 -93  \
"../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/ef1e/hdl/lib_cdc_v1_0_rfs.vhd" \

vcom -work proc_sys_reset_v5_0_13 -93  \
"../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/8842/hdl/proc_sys_reset_v5_0_vh_rfs.vhd" \

vcom -work xil_defaultlib -93  \
"../../../bd/MLZedboard/ip/MLZedboard_rst_ps7_0_100M_0/sim/MLZedboard_rst_ps7_0_100M_0.vhd" \

vlog -work xlconstant_v1_1_7  -v2k5 "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/6fba/hdl/verilog" "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/ec67/hdl" "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/ee60/hdl" "+incdir+/tools/Xilinx/Vivado/2022.2/data/xilinx_vip/include" \
"../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/badb/hdl/xlconstant_v1_1_vl_rfs.v" \

vlog -work xil_defaultlib  -v2k5 "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/6fba/hdl/verilog" "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/ec67/hdl" "+incdir+../../../../Lab8-01.gen/sources_1/bd/MLZedboard/ipshared/ee60/hdl" "+incdir+/tools/Xilinx/Vivado/2022.2/data/xilinx_vip/include" \
"../../../bd/MLZedboard/ip/MLZedboard_xlconstant_0_0/sim/MLZedboard_xlconstant_0_0.v" \
"../../../bd/MLZedboard/ip/MLZedboard_xlconstant_1_0/sim/MLZedboard_xlconstant_1_0.v" \
"../../../bd/MLZedboard/sim/MLZedboard.v" \

vlog -work xil_defaultlib \
"glbl.v"

