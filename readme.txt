1. pth文件转onnx
将官网提供的pth/自训练生成的pth文件转为onnx文件
python ICNet_pth2onnx.py rankid0_icnet_resnet50_192_0.687_best_model.pth icnet.onnx

"""
# 此步骤需要在910环境执行
# 入参1：需要转换的pth文件
# 入参2：输出的onnx文件名

# 代码中指定了输入节点的名称和shape
input_names = ["actual_input_1"]
dummy_input = torch.randn(1, 3, 1024, 2048)
"""


2. onnx文件转om
bash onnx2om.sh icnet.onnx

"""
# 入参1： onnx文件
# 返回：与onnx文件同名的om文件

# 指令内容：
atc --framework=5 --model=$ONNX_MODEL --output=$OUTPUT --out_nodes="Resize_513:0" --input_format=NCHW --input_shape="actual_input_1: 1,3,1024,2048" --log=debug --soc_version=Ascend310

# input_shape参数可通过Netron工具查看输入节点的名称和shape, 与pth转onnx步骤中的参数一致
# out_nodes为指定输出节点, 通过Netron可以看到onnx文件有四个输出, 分别为889,860,830,804
INPUTS
     actual_input_1  name: actual_input_1
                     type: float32[1,3,1024,2048]
OUTPUTS                    
     889   name: 889
           type: float32[1,19,1024,2048]
     860   name: 860
           type: float32[1,19,256,512]
     830   name: 830
           type: float32[1,19,128,256]
     804   name: 804
           type: float32[1,19,64,128]

# 其分别对应evaluate.py脚本outputs = model(image)中的outputs[0], outputs[1], outputs[2], outputs[3]
# 脚本中仅需要outputs[0]的数据做推理使用self.metric.update(outputs[0], mask)
# 因此在转om的时候, 仅存储outputs[0]节点的数据即可, 即889的节点输出
# 通过Netron工具可以看到889节点对应的输出名为Resize_513
NODE PROPERTIES
      type   Resize
      name   Resize_513

OUTPUTS
      Y      name: 889
"""

3. 数据预处理，生成bin文件和info文件
（1）将数据集转换为bin文件
python pre_dataset.py /home/sasa/dataset/cityscapes/ ./pre_dataset_bin

"""
# 入参1：数据集路径
# 入参2：生成的bin文件的存储路径

# 数据集预处理的方法同evaluate.py脚本
image = Image.open(img_paths[i]).convert('RGB')  # image shape: (W,H,3)
image = _img_transform(image)  # image shape: (3,H,W) [0,1]
image = torch.unsqueeze(image, 0)  # image shape: (1,3,H,W) [0,1]   
"""

（2）将bin文件提取为info
python get_info.py bin ./pre_dataset_bin ./icnet_pre_bin_1024_2048.info 1024 2048

"""
1024 2048对应输入图片的宽和高，此参数应与pth转onnx脚本中保持一致
"""


4. 使用benchmark工具将预处理的数据和om文件进行处理，生成bin文件
执行脚本：bash benchmark.sh

"""
# benchmark.sh
dos2unix *
source set_npu_env.sh

# pretrained
# ./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=./ICNet_pretrained_bs1_1024_2048.om -input_text_path=./icnet_prep_bin_1024_2048.info -input_width=1024 -input_height=2048 -output_binary=True -useDvpp=False


# pth
./benchmark.x86_64 -model_type=vision -device_id=1 -batch_size=1 -om_path=./ICNet_pth_bs1_1024_2048.om -input_text_path=./icnet_pre_bin_1024_2048.info -input_width=1024 -input_height=2048 -output_binary=True -useDvpp=False
"""

5. 执行离线推理脚本
执行脚本：bash eval.sh
推理日志存储在同级目录test.log下
"""
# eval.sh

source ./set_npu_env.sh

DATASET_PATH='/home/sasa/dataset/cityscapes/'
RESULT_PATH='./result_pth_bin/dumpOutput_device1/'
OUTDIR='./prefix'

python -u evaluate.py \
             $DATASET_PATH \
             $RESULT_PATH \
             $OUTDIR \
             >test.log &
             
# evaluate.py脚本中需要将benchmark生成的bin文件转成所需的float32数据，bin文件读取和转换方法如下：
def bin2tensor(self, annotation_file):
        filepath = annotation_file + '_1.bin'
        binfile = open(filepath, 'rb')  
        size = os.path.getsize(filepath)  
        # print("==========size=========:", size)
        res = []
        L = int(size / 4)
        # print(L,type(L))
        for i in range(L):
            data = binfile.read(4)  
            num = struct.unpack('f', data)
            res.append(num[0])
            # print("=======num========:", L, i, num, num[0])
        binfile.close()

        dim_res = np.array(res).reshape(1, 19, 1024, 2048)
        tensor_res = torch.tensor(dim_res, dtype=torch.float32)
        print(filepath, tensor_res.dtype, tensor_res.shape)

        return tensor_res

"""