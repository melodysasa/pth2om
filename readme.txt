1. pth�ļ�תonnx
�������ṩ��pth/��ѵ�����ɵ�pth�ļ�תΪonnx�ļ�
python ICNet_pth2onnx.py rankid0_icnet_resnet50_192_0.687_best_model.pth icnet.onnx

"""
# �˲�����Ҫ��910����ִ��
# ���1����Ҫת����pth�ļ�
# ���2�������onnx�ļ���

# ������ָ��������ڵ�����ƺ�shape
input_names = ["actual_input_1"]
dummy_input = torch.randn(1, 3, 1024, 2048)
"""


2. onnx�ļ�תom
bash onnx2om.sh icnet.onnx

"""
# ���1�� onnx�ļ�
# ���أ���onnx�ļ�ͬ����om�ļ�

# ָ�����ݣ�
atc --framework=5 --model=$ONNX_MODEL --output=$OUTPUT --out_nodes="Resize_513:0" --input_format=NCHW --input_shape="actual_input_1: 1,3,1024,2048" --log=debug --soc_version=Ascend310

# input_shape������ͨ��Netron���߲鿴����ڵ�����ƺ�shape, ��pthתonnx�����еĲ���һ��
# out_nodesΪָ������ڵ�, ͨ��Netron���Կ���onnx�ļ����ĸ����, �ֱ�Ϊ889,860,830,804
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

# ��ֱ��Ӧevaluate.py�ű�outputs = model(image)�е�outputs[0], outputs[1], outputs[2], outputs[3]
# �ű��н���Ҫoutputs[0]������������ʹ��self.metric.update(outputs[0], mask)
# �����תom��ʱ��, ���洢outputs[0]�ڵ�����ݼ���, ��889�Ľڵ����
# ͨ��Netron���߿��Կ���889�ڵ��Ӧ�������ΪResize_513
NODE PROPERTIES
      type   Resize
      name   Resize_513

OUTPUTS
      Y      name: 889
"""

3. ����Ԥ��������bin�ļ���info�ļ�
��1�������ݼ�ת��Ϊbin�ļ�
python pre_dataset.py /home/sasa/dataset/cityscapes/ ./pre_dataset_bin

"""
# ���1�����ݼ�·��
# ���2�����ɵ�bin�ļ��Ĵ洢·��

# ���ݼ�Ԥ����ķ���ͬevaluate.py�ű�
image = Image.open(img_paths[i]).convert('RGB')  # image shape: (W,H,3)
image = _img_transform(image)  # image shape: (3,H,W) [0,1]
image = torch.unsqueeze(image, 0)  # image shape: (1,3,H,W) [0,1]   
"""

��2����bin�ļ���ȡΪinfo
python get_info.py bin ./pre_dataset_bin ./icnet_pre_bin_1024_2048.info 1024 2048

"""
1024 2048��Ӧ����ͼƬ�Ŀ�͸ߣ��˲���Ӧ��pthתonnx�ű��б���һ��
"""


4. ʹ��benchmark���߽�Ԥ��������ݺ�om�ļ����д�������bin�ļ�
ִ�нű���bash benchmark.sh

"""
# benchmark.sh
dos2unix *
source set_npu_env.sh

# pretrained
# ./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=./ICNet_pretrained_bs1_1024_2048.om -input_text_path=./icnet_prep_bin_1024_2048.info -input_width=1024 -input_height=2048 -output_binary=True -useDvpp=False


# pth
./benchmark.x86_64 -model_type=vision -device_id=1 -batch_size=1 -om_path=./ICNet_pth_bs1_1024_2048.om -input_text_path=./icnet_pre_bin_1024_2048.info -input_width=1024 -input_height=2048 -output_binary=True -useDvpp=False
"""

5. ִ����������ű�
ִ�нű���bash eval.sh
������־�洢��ͬ��Ŀ¼test.log��
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
             
# evaluate.py�ű�����Ҫ��benchmark���ɵ�bin�ļ�ת�������float32���ݣ�bin�ļ���ȡ��ת���������£�
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