import torch

path ='/home/xuwei/projects/synchronous/chenfeng_code/regression_method/model/UCF_QNRF/model_test_all.pt'

torch_model = torch.load(path) # pytorch模型加载
batch_size = 1  #批处理大小
input_shape = (3,244,244)   #输入数据

x = torch.randn(batch_size, *input_shape).cuda()	# 生成张量
gt_shape = (244,244)
gt = torch.randn(3, *gt_shape)
refine_flag = torch.ones(1)
print(x.shape, gt.shape)

export_onnx_file = '/home/xuwei/projects/synchronous/chenfeng_code/regression_method/model_test_all.onnx'						# 目的ONNX文件名

torch.onnx.export(torch_model,
                  x,
                    export_onnx_file,
                    opset_version=10,
                    do_constant_folding=True,	# 是否执行常量折叠优化
                    input_names=["input"],		# 输入名
                    output_names=["output"],	# 输出名}
                                            )
print('end')

import torch
from fpn_onnx import AutoScale
from config import args
import os
torch_model = AutoScale()  					# 由研究员提供python.py文件
# args.pre = '/home/xuwei/projects/synchronous/chenfeng_code/regression_method/model/UCF_QNRF/model_test.pt'
# if args.pre:
#     if os.path.isfile(args.pre):
#         print("=> loading checkpoint '{}'".format(args.pre))
#         # checkpoint = torch.load(args.pre, map_location=lambda storage, loc: storage, pickle_module=pickle)
#         checkpoint = torch.load(args.pre)
#         torch_model.load_state_dict(checkpoint['state_dict'])
#         # rate_model.load_state_dict(checkpoint['rate_state_dict'])
#     else:
#         print("=> no checkpoint found at '{}'".format(args.pre))


# batch_size = 1 								# 批处理大小
# input_shape = (3, 640, 480) 				# 输入数据
# x = torch.randn(batch_size,*input_shape) 	# 生成张量
# export_onnx_file = '/home/xuwei/projects/synchronous/chenfeng_code/regression_method/model_test.onnx'
# print("test")
# torch.onnx.export(torch_model,
#                     x,
#                     export_onnx_file,
#                     opset_version=10,
#                     do_constant_folding=True,	# 是否执行常量折叠优化
#                     input_names=["input"],		# 输入名
#                     output_names=["output"],	# 输出名
#                     )
#
# print("end")