%%writefile obj_det_yolo_job.sh
ME=`basename $0`

#The default path for the job is your home directory, so we change directory to where the files are.
cd $PBS_O_WORKDIR

DEVICE=$2
FP_MODEL=$3
INPUT_FILE=$4
RESULTS_BASE=$1


MODELPATH="$HOME/Benchmarking_DL_Models/Yolo_V3_Model/${FP_MODEL}/frozen_darknet_yolov3_model.xml"
RESULTS_PATH="${RESULTS_BASE}"
mkdir -p $RESULTS_PATH
echo "$ME is using results path $RESULTS_PATH"

if [ "$DEVICE" = "HETERO:FPGA,CPU" ]; then
    # Environment variables and compilation for edge compute nodes with FPGAs
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/altera/aocl-pro-rte/aclrte-linux64/
    # Environment variables and compilation for edge compute nodes with FPGAs
    source /opt/fpga_support_files/setup_env.sh
    aocl program acl0 /opt/intel/openvino/bitstreams/a10_vision_design_bitstreams/2019R1_PL1_FP11_MobileNet_Clamp.aocx
fi

# Running the object detection code
! python3 object_detection_demo_yolov3_async.py  -m ${MODELPATH} \
                                                 -i $INPUT_FILE \
                                                 -o $RESULTS_PATH \
                                                 -d $DEVICE \
                                                 --labels tensorflow-yolo-v3/coco.names \
                                                 -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_avx2.so
