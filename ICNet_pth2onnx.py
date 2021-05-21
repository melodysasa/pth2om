import torch
import torch.onnx
import sys
import ssl

from models import ICNet

def convert(pth_file, onnx_file):
    
    model = ICNet(nclass=19, backbone='resnet50')
    print(model)
    pretrained_net = torch.load(pth_file, map_location='cpu')
    model.load_state_dict(pretrained_net)
    model.eval()
    input_names = ["actual_input_1"]
    dummy_input = torch.randn(1, 3, 1024, 2048)
    
    torch.onnx.export(model, dummy_input, onnx_file, input_names=input_names, opset_version=11, verbose=True)

if __name__ == "__main__":

    pth_file = sys.argv[1]
    onnx_file = sys.argv[2]

    ssl._create_default_https_context = ssl._create_unverified_context
    convert(pth_file, onnx_file)
