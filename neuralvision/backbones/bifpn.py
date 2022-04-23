# import math

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.ops.boxes import nms as nms_torch


# class Conv2dStaticSamePadding(nn.Module):
#     """
#     created by Zylo117
#     The real keras/tensorflow conv2d with same padding
#     """

#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         kernel_size,
#         stride=1,
#         bias=True,
#         groups=1,
#         dilation=1,
#         **kwargs,
#     ):
#         super().__init__()
#         self.conv = nn.Conv2d(
#             in_channels,
#             out_channels,
#             kernel_size,
#             stride=stride,
#             bias=bias,
#             groups=groups,
#         )
#         self.stride = self.conv.stride
#         self.kernel_size = self.conv.kernel_size
#         self.dilation = self.conv.dilation

#         if isinstance(self.stride, int):
#             self.stride = [self.stride] * 2
#         elif len(self.stride) == 1:
#             self.stride = [self.stride[0]] * 2

#         if isinstance(self.kernel_size, int):
#             self.kernel_size = [self.kernel_size] * 2
#         elif len(self.kernel_size) == 1:
#             self.kernel_size = [self.kernel_size[0]] * 2

#     def forward(self, x):
#         h, w = x.shape[-2:]

#         extra_h = (
#             (math.ceil(w / self.stride[1]) - 1) * self.stride[1]
#             - w
#             + self.kernel_size[1]
#         )
#         extra_v = (
#             (math.ceil(h / self.stride[0]) - 1) * self.stride[0]
#             - h
#             + self.kernel_size[0]
#         )

#         left = extra_h // 2
#         right = extra_h - left
#         top = extra_v // 2
#         bottom = extra_v - top

#         x = F.pad(x, [left, right, top, bottom])

#         x = self.conv(x)
#         return x


# class MaxPool2dStaticSamePadding(nn.Module):
#     """
#     created by Zylo117
#     The real keras/tensorflow MaxPool2d with same padding
#     """

#     def __init__(self, *args, **kwargs):
#         super().__init__()
#         self.pool = nn.MaxPool2d(*args, **kwargs)
#         self.stride = self.pool.stride
#         self.kernel_size = self.pool.kernel_size

#         if isinstance(self.stride, int):
#             self.stride = [self.stride] * 2
#         elif len(self.stride) == 1:
#             self.stride = [self.stride[0]] * 2

#         if isinstance(self.kernel_size, int):
#             self.kernel_size = [self.kernel_size] * 2
#         elif len(self.kernel_size) == 1:
#             self.kernel_size = [self.kernel_size[0]] * 2

#     def forward(self, x):
#         h, w = x.shape[-2:]

#         extra_h = (
#             (math.ceil(w / self.stride[1]) - 1) * self.stride[1]
#             - w
#             + self.kernel_size[1]
#         )
#         extra_v = (
#             (math.ceil(h / self.stride[0]) - 1) * self.stride[0]
#             - h
#             + self.kernel_size[0]
#         )

#         left = extra_h // 2
#         right = extra_h - left
#         top = extra_v // 2
#         bottom = extra_v - top

#         x = F.pad(x, [left, right, top, bottom])

#         x = self.pool(x)
#         return x


# class SwishImplementation(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, i):
#         result = i * torch.sigmoid(i)
#         ctx.save_for_backward(i)
#         return result

#     @staticmethod
#     def backward(ctx, grad_output):
#         i = ctx.saved_variables[0]
#         sigmoid_i = torch.sigmoid(i)
#         return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


# class MemoryEfficientSwish(nn.Module):
#     def forward(self, x):
#         return SwishImplementation.apply(x)


# class Swish(nn.Module):
#     def forward(self, x):
#         return x * torch.sigmoid(x)


# def nms(dets, thresh):
#     return nms_torch(dets[:, :4], dets[:, 4], thresh)


# class SeparableConvBlock(nn.Module):
#     """
#     created by Zylo117
#     """

#     def __init__(
#         self,
#         in_channels,
#         out_channels=None,
#         norm=True,
#         activation=False,
#         onnx_export=False,
#     ):
#         super(SeparableConvBlock, self).__init__()
#         if out_channels is None:
#             out_channels = in_channels

#         # Q: whether separate conv
#         #  share bias between depthwise_conv and pointwise_conv
#         #  or just pointwise_conv apply bias.
#         # A: Confirmed, just pointwise_conv applies bias, depthwise_conv has no bias.

#         self.depthwise_conv = Conv2dStaticSamePadding(
#             in_channels,
#             in_channels,
#             kernel_size=3,
#             stride=1,
#             groups=in_channels,
#             bias=False,
#         )
#         self.pointwise_conv = Conv2dStaticSamePadding(
#             in_channels, out_channels, kernel_size=1, stride=1
#         )

#         self.norm = norm
#         if self.norm:
#             # Warning: pytorch momentum is different from tensorflow's, momentum_pytorch = 1 - momentum_tensorflow
#             self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

#         self.activation = activation
#         if self.activation:
#             self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

#     def forward(self, x):
#         x = self.depthwise_conv(x)
#         x = self.pointwise_conv(x)

#         if self.norm:
#             x = self.bn(x)

#         if self.activation:
#             x = self.swish(x)

#         return x


# class BiFPN(nn.Module):
#     """
#     modified by Zylo117
#     """

#     def __init__(
#         self,
#         num_channels,
#         conv_channels,
#         first_time=False,
#         attention=True,
#         use_p8=False,
#         epsilon=1e-4,
#         onnx_export=False,
#     ):
#         """

#         Args:
#             num_channels:
#             conv_channels:
#             first_time: whether the input comes directly from the efficientnet,
#                         if True, downchannel it first, and downsample P5 to generate P6 then P7
#             epsilon: epsilon of fast weighted attention sum of BiFPN, not the BN's epsilon
#             onnx_export: if True, use Swish instead of MemoryEfficientSwish
#         """
#         super(BiFPN, self).__init__()
#         self.epsilon = epsilon
#         self.use_p8 = use_p8

#         # Conv layers
#         self.conv6_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
#         self.conv5_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
#         self.conv4_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
#         self.conv3_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
#         self.conv4_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
#         self.conv5_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
#         self.conv6_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
#         self.conv7_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
#         if use_p8:
#             self.conv7_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
#             self.conv8_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)

#         # Feature scaling layers
#         self.p6_upsample = nn.Upsample(scale_factor=2, mode="nearest")
#         self.p5_upsample = nn.Upsample(scale_factor=2, mode="nearest")
#         self.p4_upsample = nn.Upsample(scale_factor=2, mode="nearest")
#         self.p3_upsample = nn.Upsample(scale_factor=2, mode="nearest")

#         self.p4_downsample = MaxPool2dStaticSamePadding(3, 2)
#         self.p5_downsample = MaxPool2dStaticSamePadding(3, 2)
#         self.p6_downsample = MaxPool2dStaticSamePadding(3, 2)
#         self.p7_downsample = MaxPool2dStaticSamePadding(3, 2)
#         if use_p8:
#             self.p7_upsample = nn.Upsample(scale_factor=2, mode="nearest")
#             self.p8_downsample = MaxPool2dStaticSamePadding(3, 2)

#         self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

#         self.first_time = first_time
#         if self.first_time:
#             self.p5_down_channel = nn.Sequential(
#                 Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
#                 nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
#             )
#             self.p4_down_channel = nn.Sequential(
#                 Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
#                 nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
#             )
#             self.p3_down_channel = nn.Sequential(
#                 Conv2dStaticSamePadding(conv_channels[0], num_channels, 1),
#                 nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
#             )

#             self.p5_to_p6 = nn.Sequential(
#                 Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
#                 nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
#                 MaxPool2dStaticSamePadding(3, 2),
#             )
#             self.p6_to_p7 = nn.Sequential(MaxPool2dStaticSamePadding(3, 2))
#             if use_p8:
#                 self.p7_to_p8 = nn.Sequential(MaxPool2dStaticSamePadding(3, 2))

#             self.p4_down_channel_2 = nn.Sequential(
#                 Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
#                 nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
#             )
#             self.p5_down_channel_2 = nn.Sequential(
#                 Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
#                 nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
#             )

#         # Weight
#         self.p6_w1 = nn.Parameter(
#             torch.ones(2, dtype=torch.float32), requires_grad=True
#         )
#         self.p6_w1_relu = nn.ReLU()
#         self.p5_w1 = nn.Parameter(
#             torch.ones(2, dtype=torch.float32), requires_grad=True
#         )
#         self.p5_w1_relu = nn.ReLU()
#         self.p4_w1 = nn.Parameter(
#             torch.ones(2, dtype=torch.float32), requires_grad=True
#         )
#         self.p4_w1_relu = nn.ReLU()
#         self.p3_w1 = nn.Parameter(
#             torch.ones(2, dtype=torch.float32), requires_grad=True
#         )
#         self.p3_w1_relu = nn.ReLU()

#         self.p4_w2 = nn.Parameter(
#             torch.ones(3, dtype=torch.float32), requires_grad=True
#         )
#         self.p4_w2_relu = nn.ReLU()
#         self.p5_w2 = nn.Parameter(
#             torch.ones(3, dtype=torch.float32), requires_grad=True
#         )
#         self.p5_w2_relu = nn.ReLU()
#         self.p6_w2 = nn.Parameter(
#             torch.ones(3, dtype=torch.float32), requires_grad=True
#         )
#         self.p6_w2_relu = nn.ReLU()
#         self.p7_w2 = nn.Parameter(
#             torch.ones(2, dtype=torch.float32), requires_grad=True
#         )
#         self.p7_w2_relu = nn.ReLU()

#         self.attention = attention

#     def forward(self, inputs):
#         """
#         illustration of a minimal bifpn unit
#             P7_0 -------------------------> P7_2 -------->
#                |-------------|                ↑
#                              ↓                |
#             P6_0 ---------> P6_1 ---------> P6_2 -------->
#                |-------------|--------------↑ ↑
#                              ↓                |
#             P5_0 ---------> P5_1 ---------> P5_2 -------->
#                |-------------|--------------↑ ↑
#                              ↓                |
#             P4_0 ---------> P4_1 ---------> P4_2 -------->
#                |-------------|--------------↑ ↑
#                              |--------------↓ |
#             P3_0 -------------------------> P3_2 -------->
#         """

#         # downsample channels using same-padding conv2d to target phase's if not the same
#         # judge: same phase as target,
#         # if same, pass;
#         # elif earlier phase, downsample to target phase's by pooling
#         # elif later phase, upsample to target phase's by nearest interpolation

#         if self.attention:
#             outs = self._forward_fast_attention(inputs)
#         else:
#             outs = self._forward(inputs)

#         return outs

#     def _forward_fast_attention(self, inputs):
#         if self.first_time:
#             p3, p4, p5 = inputs

#             p6_in = self.p5_to_p6(p5)
#             p7_in = self.p6_to_p7(p6_in)

#             p3_in = self.p3_down_channel(p3)
#             p4_in = self.p4_down_channel(p4)
#             p5_in = self.p5_down_channel(p5)

#         else:
#             # P3_0, P4_0, P5_0, P6_0 and P7_0
#             p3_in, p4_in, p5_in, p6_in, p7_in = inputs

#         # P7_0 to P7_2

#         # Weights for P6_0 and P7_0 to P6_1
#         p6_w1 = self.p6_w1_relu(self.p6_w1)
#         weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
#         # Connections for P6_0 and P7_0 to P6_1 respectively
#         p6_up = self.conv6_up(
#             self.swish(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in))
#         )

#         # Weights for P5_0 and P6_1 to P5_1
#         p5_w1 = self.p5_w1_relu(self.p5_w1)
#         weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
#         # Connections for P5_0 and P6_1 to P5_1 respectively
#         p5_up = self.conv5_up(
#             self.swish(weight[0] * p5_in + weight[1] * self.p5_upsample(p6_up))
#         )

#         # Weights for P4_0 and P5_1 to P4_1
#         p4_w1 = self.p4_w1_relu(self.p4_w1)
#         weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
#         # Connections for P4_0 and P5_1 to P4_1 respectively
#         p4_up = self.conv4_up(
#             self.swish(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_up))
#         )

#         # Weights for P3_0 and P4_1 to P3_2
#         p3_w1 = self.p3_w1_relu(self.p3_w1)
#         weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
#         # Connections for P3_0 and P4_1 to P3_2 respectively
#         p3_out = self.conv3_up(
#             self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_up))
#         )

#         if self.first_time:
#             p4_in = self.p4_down_channel_2(p4)
#             p5_in = self.p5_down_channel_2(p5)

#         # Weights for P4_0, P4_1 and P3_2 to P4_2
#         p4_w2 = self.p4_w2_relu(self.p4_w2)
#         weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
#         # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
#         p4_out = self.conv4_down(
#             self.swish(
#                 weight[0] * p4_in
#                 + weight[1] * p4_up
#                 + weight[2] * self.p4_downsample(p3_out)
#             )
#         )

#         # Weights for P5_0, P5_1 and P4_2 to P5_2
#         p5_w2 = self.p5_w2_relu(self.p5_w2)
#         weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
#         # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
#         p5_out = self.conv5_down(
#             self.swish(
#                 weight[0] * p5_in
#                 + weight[1] * p5_up
#                 + weight[2] * self.p5_downsample(p4_out)
#             )
#         )

#         # Weights for P6_0, P6_1 and P5_2 to P6_2
#         p6_w2 = self.p6_w2_relu(self.p6_w2)
#         weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
#         # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
#         p6_out = self.conv6_down(
#             self.swish(
#                 weight[0] * p6_in
#                 + weight[1] * p6_up
#                 + weight[2] * self.p6_downsample(p5_out)
#             )
#         )

#         # Weights for P7_0 and P6_2 to P7_2
#         p7_w2 = self.p7_w2_relu(self.p7_w2)
#         weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
#         # Connections for P7_0 and P6_2 to P7_2
#         p7_out = self.conv7_down(
#             self.swish(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out))
#         )

#         return p3_out, p4_out, p5_out, p6_out, p7_out

#     def _forward(self, inputs):
#         if self.first_time:
#             p3, p4, p5 = inputs

#             p6_in = self.p5_to_p6(p5)
#             p7_in = self.p6_to_p7(p6_in)
#             if self.use_p8:
#                 p8_in = self.p7_to_p8(p7_in)

#             p3_in = self.p3_down_channel(p3)
#             p4_in = self.p4_down_channel(p4)
#             p5_in = self.p5_down_channel(p5)

#         else:
#             if self.use_p8:
#                 # P3_0, P4_0, P5_0, P6_0, P7_0 and P8_0
#                 p3_in, p4_in, p5_in, p6_in, p7_in, p8_in = inputs
#             else:
#                 # P3_0, P4_0, P5_0, P6_0 and P7_0
#                 p3_in, p4_in, p5_in, p6_in, p7_in = inputs

#         if self.use_p8:
#             # P8_0 to P8_2

#             # Connections for P7_0 and P8_0 to P7_1 respectively
#             p7_up = self.conv7_up(self.swish(p7_in + self.p7_upsample(p8_in)))

#             # Connections for P6_0 and P7_0 to P6_1 respectively
#             p6_up = self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_up)))
#         else:
#             # P7_0 to P7_2

#             # Connections for P6_0 and P7_0 to P6_1 respectively
#             p6_up = self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_in)))

#         # Connections for P5_0 and P6_1 to P5_1 respectively
#         p5_up = self.conv5_up(self.swish(p5_in + self.p5_upsample(p6_up)))

#         # Connections for P4_0 and P5_1 to P4_1 respectively
#         p4_up = self.conv4_up(self.swish(p4_in + self.p4_upsample(p5_up)))

#         # Connections for P3_0 and P4_1 to P3_2 respectively
#         p3_out = self.conv3_up(self.swish(p3_in + self.p3_upsample(p4_up)))

#         if self.first_time:
#             p4_in = self.p4_down_channel_2(p4)
#             p5_in = self.p5_down_channel_2(p5)

#         # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
#         p4_out = self.conv4_down(self.swish(p4_in + p4_up + self.p4_downsample(p3_out)))

#         # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
#         p5_out = self.conv5_down(self.swish(p5_in + p5_up + self.p5_downsample(p4_out)))

#         # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
#         p6_out = self.conv6_down(self.swish(p6_in + p6_up + self.p6_downsample(p5_out)))

#         if self.use_p8:
#             # Connections for P7_0, P7_1 and P6_2 to P7_2 respectively
#             p7_out = self.conv7_down(
#                 self.swish(p7_in + p7_up + self.p7_downsample(p6_out))
#             )

#             # Connections for P8_0 and P7_2 to P8_2
#             p8_out = self.conv8_down(self.swish(p8_in + self.p8_downsample(p7_out)))

#             return p3_out, p4_out, p5_out, p6_out, p7_out, p8_out
#         else:
#             # Connections for P7_0 and P6_2 to P7_2
#             p7_out = self.conv7_down(self.swish(p7_in + self.p7_downsample(p6_out)))

#             return p3_out, p4_out, p5_out, p6_out, p7_out


# class EfficientNet(nn.Module):
#     """
#     An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods

#     Args:
#         blocks_args (list): A list of BlockArgs to construct blocks
#         global_params (namedtuple): A set of GlobalParams shared between blocks

#     Example:
#         model = EfficientNet.from_pretrained('efficientnet-b0')

#     """

#     def __init__(self, blocks_args=None, global_params=None):
#         super().__init__()
#         assert isinstance(blocks_args, list), "blocks_args should be a list"
#         assert len(blocks_args) > 0, "block args must be greater than 0"
#         self._global_params = global_params
#         self._blocks_args = blocks_args

#         # Get static or dynamic convolution depending on image size
#         Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

#         # Batch norm parameters
#         bn_mom = 1 - self._global_params.batch_norm_momentum
#         bn_eps = self._global_params.batch_norm_epsilon

#         # Stem
#         in_channels = 3  # rgb
#         out_channels = round_filters(
#             32, self._global_params
#         )  # number of output channels
#         self._conv_stem = Conv2d(
#             in_channels, out_channels, kernel_size=3, stride=2, bias=False
#         )
#         self._bn0 = nn.BatchNorm2d(
#             num_features=out_channels, momentum=bn_mom, eps=bn_eps
#         )

#         # Build blocks
#         self._blocks = nn.ModuleList([])
#         for block_args in self._blocks_args:

#             # Update block input and output filters based on depth multiplier.
#             block_args = block_args._replace(
#                 input_filters=round_filters(
#                     block_args.input_filters, self._global_params
#                 ),
#                 output_filters=round_filters(
#                     block_args.output_filters, self._global_params
#                 ),
#                 num_repeat=round_repeats(block_args.num_repeat, self._global_params),
#             )

#             # The first block needs to take care of stride and filter size increase.
#             self._blocks.append(MBConvBlock(block_args, self._global_params))
#             if block_args.num_repeat > 1:
#                 block_args = block_args._replace(
#                     input_filters=block_args.output_filters, stride=1
#                 )
#             for _ in range(block_args.num_repeat - 1):
#                 self._blocks.append(MBConvBlock(block_args, self._global_params))

#         # Head
#         in_channels = block_args.output_filters  # output of final block
#         out_channels = round_filters(1280, self._global_params)
#         self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
#         self._bn1 = nn.BatchNorm2d(
#             num_features=out_channels, momentum=bn_mom, eps=bn_eps
#         )

#         # Final linear layer
#         self._avg_pooling = nn.AdaptiveAvgPool2d(1)
#         self._dropout = nn.Dropout(self._global_params.dropout_rate)
#         self._fc = nn.Linear(out_channels, self._global_params.num_classes)
#         self._swish = MemoryEfficientSwish()

#     def set_swish(self, memory_efficient=True):
#         """Sets swish function as memory efficient (for training) or standard (for export)"""
#         self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
#         for block in self._blocks:
#             block.set_swish(memory_efficient)

#     def extract_features(self, inputs):
#         """Returns output of the final convolution layer"""

#         # Stem
#         x = self._swish(self._bn0(self._conv_stem(inputs)))

#         # Blocks
#         for idx, block in enumerate(self._blocks):
#             drop_connect_rate = self._global_params.drop_connect_rate
#             if drop_connect_rate:
#                 drop_connect_rate *= float(idx) / len(self._blocks)
#             x = block(x, drop_connect_rate=drop_connect_rate)
#         # Head
#         x = self._swish(self._bn1(self._conv_head(x)))

#         return x

#     def forward(self, inputs):
#         """Calls extract_features to extract features, applies final linear layer, and returns logits."""
#         bs = inputs.size(0)
#         # Convolution layers
#         x = self.extract_features(inputs)

#         # Pooling and final linear layer
#         x = self._avg_pooling(x)
#         x = x.view(bs, -1)
#         x = self._dropout(x)
#         x = self._fc(x)
#         return x

#     @classmethod
#     def from_name(cls, model_name, override_params=None):
#         cls._check_model_name_is_valid(model_name)
#         blocks_args, global_params = get_model_params(model_name, override_params)
#         return cls(blocks_args, global_params)

#     @classmethod
#     def from_pretrained(
#         cls,
#         model_name,
#         load_weights=True,
#         advprop=False,
#         num_classes=1000,
#         in_channels=3,
#     ):
#         model = cls.from_name(model_name, override_params={"num_classes": num_classes})
#         if load_weights:
#             load_pretrained_weights(
#                 model, model_name, load_fc=(num_classes == 1000), advprop=advprop
#             )
#         if in_channels != 3:
#             Conv2d = get_same_padding_conv2d(image_size=model._global_params.image_size)
#             out_channels = round_filters(32, model._global_params)
#             model._conv_stem = Conv2d(
#                 in_channels, out_channels, kernel_size=3, stride=2, bias=False
#             )
#         return model

#     @classmethod
#     def get_image_size(cls, model_name):
#         cls._check_model_name_is_valid(model_name)
#         _, _, res, _ = efficientnet_params(model_name)
#         return res

#     @classmethod
#     def _check_model_name_is_valid(cls, model_name):
#         """Validates model name."""
#         valid_models = ["efficientnet-b" + str(i) for i in range(9)]
#         if model_name not in valid_models:
#             raise ValueError("model_name should be one of: " + ", ".join(valid_models))


# if __name__ == "__main__":

#     from neuralvision.backbones.resnet import ResNet

#     output_channels = [128, 256, 128, 128, 64, 64]
#     image_channels = 3
#     output_feature_sizes = [[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]]

#     backbone = ResNet(output_channels, image_channels, output_feature_sizes)

#     backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6, 7]
#     fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384, 384]
#     fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8, 8]

#     input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
#     box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5, 5]
#     pyramid_levels = [5, 5, 5, 5, 5, 5, 5, 5, 6]
#     anchor_scale = [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 5.0, 4.0]
#     conv_channel_coef = {
#         # the channels of P3/P4/P5.
#         0: [40, 112, 320],
#         1: [40, 112, 320],
#         2: [48, 120, 352],
#         3: [48, 136, 384],
#         4: [56, 160, 448],
#         5: [64, 176, 512],
#         6: [72, 200, 576],
#         7: [72, 200, 576],
#         8: [80, 224, 640],
#     }

#     compound_coef = 8

#     bifpn = nn.Sequential(
#         *[
#             BiFPN(
#                 num_channels=fpn_num_filters[compound_coef],
#                 conv_channels=conv_channel_coef[compound_coef],
#                 first_time=True if _ == 0 else False,
#                 attention=True if compound_coef < 6 else False,
#                 use_p8=compound_coef > 7,
#             )
#             for _ in range(fpn_cell_repeats[compound_coef])
#         ]
#     )

#     test_img = torch.randn(2, image_channels, 128, 1024)

#     # print number of parameters
#     print("bifpn   : ", sum([p.numel() for p in bifpn.parameters()]))
#     print("backbone: ", sum([p.numel() for p in backbone.parameters()]))

#     features = backbone(test_img)

#     for f in features:
#         print(f.shape)

#     # features = bifpn((p3, p4, p5))

#     # FEATURE_SIZE = 160
#     # NUM_LAYERS = 5


### ---

from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from timm.models.layers import create_conv2d, create_pool2d, get_act_layer
import logging
from torch.functional import F
from functools import partial

_DEBUG = False
_USE_SCALE = False
_ACT_LAYER = get_act_layer("silu")

from .bifpnstuff import get_fpn_config


class SequentialList(nn.Sequential):
    """This module exists to work around torchscript typing issues list -> list"""

    def __init__(self, *args):
        super(SequentialList, self).__init__(*args)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        for module in self:
            x = module(x)
        return x


class ConvBnAct2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        padding="",
        bias=False,
        norm_layer=nn.BatchNorm2d,
        act_layer=_ACT_LAYER,
    ):
        super(ConvBnAct2d, self).__init__()
        self.conv = create_conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            bias=bias,
        )
        self.bn = None if norm_layer is None else norm_layer(out_channels)
        self.act = None if act_layer is None else act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class SeparableConv2d(nn.Module):
    """Separable Conv"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        padding="",
        bias=False,
        channel_multiplier=1.0,
        pw_kernel_size=1,
        norm_layer=nn.BatchNorm2d,
        act_layer=_ACT_LAYER,
    ):
        super(SeparableConv2d, self).__init__()
        self.conv_dw = create_conv2d(
            in_channels,
            int(in_channels * channel_multiplier),
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            depthwise=True,
        )

        self.conv_pw = create_conv2d(
            int(in_channels * channel_multiplier),
            out_channels,
            pw_kernel_size,
            padding=padding,
            bias=bias,
        )

        self.bn = None if norm_layer is None else norm_layer(out_channels)
        self.act = None if act_layer is None else act_layer(inplace=True)

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class Interpolate2d(nn.Module):
    r"""Resamples a 2d Image

    The input data is assumed to be of the form
    `minibatch x channels x [optional depth] x [optional height] x width`.
    Hence, for spatial inputs, we expect a 4D Tensor and for volumetric inputs, we expect a 5D Tensor.

    The algorithms available for upsampling are nearest neighbor and linear,
    bilinear, bicubic and trilinear for 3D, 4D and 5D input Tensor,
    respectively.

    One can either give a :attr:`scale_factor` or the target output :attr:`size` to
    calculate the output size. (You cannot give both, as it is ambiguous)

    Args:
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int], optional):
            output spatial sizes
        scale_factor (float or Tuple[float] or Tuple[float, float] or Tuple[float, float, float], optional):
            multiplier for spatial size. Has to match input size if it is a tuple.
        mode (str, optional): the upsampling algorithm: one of ``'nearest'``,
            ``'linear'``, ``'bilinear'``, ``'bicubic'`` and ``'trilinear'``.
            Default: ``'nearest'``
        align_corners (bool, optional): if ``True``, the corner pixels of the input
            and output tensors are aligned, and thus preserving the values at
            those pixels. This only has effect when :attr:`mode` is
            ``'linear'``, ``'bilinear'``, or ``'trilinear'``. Default: ``False``
    """
    __constants__ = ["size", "scale_factor", "mode", "align_corners", "name"]
    name: str
    size: Optional[Union[int, Tuple[int, int]]]
    scale_factor: Optional[Union[float, Tuple[float, float]]]
    mode: str
    align_corners: Optional[bool]

    def __init__(
        self,
        size: Optional[Union[int, Tuple[int, int]]] = None,
        scale_factor: Optional[Union[float, Tuple[float, float]]] = None,
        mode: str = "nearest",
        align_corners: bool = False,
    ) -> None:
        super(Interpolate2d, self).__init__()
        self.name = type(self).__name__
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = None if mode == "nearest" else align_corners

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            input,
            self.size,
            self.scale_factor,
            self.mode,
            self.align_corners,
            recompute_scale_factor=False,
        )


class ResampleFeatureMap(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        input_size,
        output_size,
        pad_type="",
        downsample=None,
        upsample=None,
        norm_layer=nn.BatchNorm2d,
        apply_bn=False,
        redundant_bias=False,
    ):
        super(ResampleFeatureMap, self).__init__()
        downsample = downsample or "max"
        upsample = upsample or "nearest"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.output_size = output_size

        if in_channels != out_channels:
            self.add_module(
                "conv",
                ConvBnAct2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    padding=pad_type,
                    norm_layer=norm_layer if apply_bn else None,
                    bias=not apply_bn or redundant_bias,
                    act_layer=None,
                ),
            )

        if input_size[0] > output_size[0] and input_size[1] > output_size[1]:
            if downsample in ("max", "avg"):
                stride_size_h = int((input_size[0] - 1) // output_size[0] + 1)
                stride_size_w = int((input_size[1] - 1) // output_size[1] + 1)
                if stride_size_h == stride_size_w:
                    kernel_size = stride_size_h + 1
                    stride = stride_size_h
                else:
                    # FIXME need to support tuple kernel / stride input to padding fns
                    kernel_size = (stride_size_h + 1, stride_size_w + 1)
                    stride = (stride_size_h, stride_size_w)
                down_inst = create_pool2d(
                    downsample, kernel_size=kernel_size, stride=stride, padding=pad_type
                )
            else:
                if (
                    _USE_SCALE
                ):  # FIXME not sure if scale vs size is better, leaving both in to test for now
                    scale = (
                        output_size[0] / input_size[0],
                        output_size[1] / input_size[1],
                    )
                    down_inst = Interpolate2d(scale_factor=scale, mode=downsample)
                else:
                    down_inst = Interpolate2d(size=output_size, mode=downsample)
            self.add_module("downsample", down_inst)
        else:
            if input_size[0] < output_size[0] or input_size[1] < output_size[1]:
                if _USE_SCALE:
                    scale = (
                        output_size[0] / input_size[0],
                        output_size[1] / input_size[1],
                    )
                    self.add_module(
                        "upsample", Interpolate2d(scale_factor=scale, mode=upsample)
                    )
                else:
                    self.add_module(
                        "upsample", Interpolate2d(size=output_size, mode=upsample)
                    )


class FpnCombine(nn.Module):
    def __init__(
        self,
        feature_info,
        fpn_channels,
        inputs_offsets,
        output_size,
        pad_type="",
        downsample=None,
        upsample=None,
        norm_layer=nn.BatchNorm2d,
        apply_resample_bn=False,
        redundant_bias=False,
        weight_method="attn",
    ):
        super(FpnCombine, self).__init__()
        self.inputs_offsets = inputs_offsets
        self.weight_method = weight_method

        self.resample = nn.ModuleDict()
        for idx, offset in enumerate(inputs_offsets):
            self.resample[str(offset)] = ResampleFeatureMap(
                feature_info[offset]["num_chs"],
                fpn_channels,
                input_size=feature_info[offset]["size"],
                output_size=output_size,
                pad_type=pad_type,
                downsample=downsample,
                upsample=upsample,
                norm_layer=norm_layer,
                apply_bn=apply_resample_bn,
                redundant_bias=redundant_bias,
            )

        if weight_method == "attn" or weight_method == "fastattn":
            self.edge_weights = nn.Parameter(
                torch.ones(len(inputs_offsets)), requires_grad=True
            )  # WSM
        else:
            self.edge_weights = None

    def forward(self, x: List[torch.Tensor]):
        dtype = x[0].dtype
        nodes = []
        for offset, resample in zip(self.inputs_offsets, self.resample.values()):
            input_node = x[offset]
            input_node = resample(input_node)
            nodes.append(input_node)

        if self.weight_method == "attn":
            normalized_weights = torch.softmax(self.edge_weights.to(dtype=dtype), dim=0)
            out = torch.stack(nodes, dim=-1) * normalized_weights
        elif self.weight_method == "fastattn":
            edge_weights = nn.functional.relu(self.edge_weights.to(dtype=dtype))
            weights_sum = torch.sum(edge_weights)
            out = torch.stack(
                [
                    (nodes[i] * edge_weights[i]) / (weights_sum + 0.0001)
                    for i in range(len(nodes))
                ],
                dim=-1,
            )
        elif self.weight_method == "sum":
            out = torch.stack(nodes, dim=-1)
        else:
            raise ValueError("unknown weight_method {}".format(self.weight_method))
        out = torch.sum(out, dim=-1)
        return out


class BiFpnLayer(nn.Module):
    def __init__(
        self,
        feature_info,
        feat_sizes,
        fpn_config,
        fpn_channels,
        num_levels=5,
        pad_type="",
        downsample=None,
        upsample=None,
        norm_layer=nn.BatchNorm2d,
        act_layer=_ACT_LAYER,
        apply_resample_bn=False,
        pre_act=True,
        separable_conv=True,
        redundant_bias=False,
    ):
        super(BiFpnLayer, self).__init__()
        self.num_levels = num_levels
        # fill feature info for all FPN nodes (chs and feat size) before creating FPN nodes
        fpn_feature_info = feature_info + [
            dict(num_chs=fpn_channels, size=feat_sizes[fc["feat_level"]])
            for fc in fpn_config.nodes
        ]

        self.fnode = nn.ModuleList()
        for i, fnode_cfg in enumerate(fpn_config.nodes):
            logging.debug("fnode {} : {}".format(i, fnode_cfg))
            combine = FpnCombine(
                fpn_feature_info,
                fpn_channels,
                tuple(fnode_cfg["inputs_offsets"]),
                output_size=feat_sizes[fnode_cfg["feat_level"]],
                pad_type=pad_type,
                downsample=downsample,
                upsample=upsample,
                norm_layer=norm_layer,
                apply_resample_bn=apply_resample_bn,
                redundant_bias=redundant_bias,
                weight_method=fnode_cfg["weight_method"],
            )

            after_combine = nn.Sequential()
            conv_kwargs = dict(
                in_channels=fpn_channels,
                out_channels=fpn_channels,
                kernel_size=3,
                padding=pad_type,
                bias=False,
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
            if pre_act:
                conv_kwargs["bias"] = redundant_bias
                conv_kwargs["act_layer"] = None
                after_combine.add_module("act", act_layer(inplace=True))
            after_combine.add_module(
                "conv",
                SeparableConv2d(**conv_kwargs)
                if separable_conv
                else ConvBnAct2d(**conv_kwargs),
            )

            self.fnode.append(Fnode(combine=combine, after_combine=after_combine))

        self.feature_info = fpn_feature_info[-num_levels::]

    def forward(self, x: List[torch.Tensor]):
        for fn in self.fnode:
            x.append(fn(x))
        return x[-self.num_levels : :]


class Fnode(nn.Module):
    """A simple wrapper used in place of nn.Sequential for torchscript typing
    Handles input type List[Tensor] -> output type Tensor
    """

    def __init__(self, combine: nn.Module, after_combine: nn.Module):
        super(Fnode, self).__init__()
        self.combine = combine
        self.after_combine = after_combine

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return self.after_combine(self.combine(x))


def get_feat_sizes(image_size: Tuple[int, int], max_level: int):
    """Get feat widths and heights for all levels.
    Args:
      image_size: a tuple (H, W)
      max_level: maximum feature level.
    Returns:
      feat_sizes: a list of tuples (height, width) for each level.
    """
    feat_size = image_size
    feat_sizes = [feat_size]
    for _ in range(1, max_level + 1):
        feat_size = ((feat_size[0] - 1) // 2 + 1, (feat_size[1] - 1) // 2 + 1)
        feat_sizes.append(feat_size)
    return feat_sizes


class BiFpn(nn.Module):
    def __init__(self, config, feature_info):
        super(BiFpn, self).__init__()
        self.num_levels = config.num_levels
        norm_layer = config.norm_layer or nn.BatchNorm2d
        if config.norm_kwargs:
            norm_layer = partial(norm_layer, **config.norm_kwargs)
        act_layer = get_act_layer(config.act_type) or _ACT_LAYER
        fpn_config = config.fpn_config or get_fpn_config(
            config.fpn_name, min_level=config.min_level, max_level=config.max_level
        )

        feat_sizes = get_feat_sizes(config.image_size, max_level=config.max_level)
        prev_feat_size = feat_sizes[config.min_level]
        self.resample = nn.ModuleDict()
        for level in range(config.num_levels):
            feat_size = feat_sizes[level + config.min_level]
            if level < len(feature_info):
                in_chs = feature_info[level]["num_chs"]
                feature_info[level]["size"] = feat_size
            else:
                # Adds a coarser level by downsampling the last feature map
                self.resample[str(level)] = ResampleFeatureMap(
                    in_channels=in_chs,
                    out_channels=config.fpn_channels,
                    input_size=prev_feat_size,
                    output_size=feat_size,
                    pad_type=config.pad_type,
                    downsample=config.downsample_type,
                    upsample=config.upsample_type,
                    norm_layer=norm_layer,
                    apply_bn=config.apply_resample_bn,
                    redundant_bias=config.redundant_bias,
                )
                in_chs = config.fpn_channels
                feature_info.append(dict(num_chs=in_chs, size=feat_size))
            prev_feat_size = feat_size

        self.cell = SequentialList()
        for rep in range(config.fpn_cell_repeats):
            logging.debug("building cell {}".format(rep))
            fpn_layer = BiFpnLayer(
                feature_info=feature_info,
                feat_sizes=feat_sizes,
                fpn_config=fpn_config,
                fpn_channels=config.fpn_channels,
                num_levels=config.num_levels,
                pad_type=config.pad_type,
                downsample=config.downsample_type,
                upsample=config.upsample_type,
                norm_layer=norm_layer,
                act_layer=act_layer,
                separable_conv=config.separable_conv,
                apply_resample_bn=config.apply_resample_bn,
                pre_act=not config.conv_bn_relu_pattern,
                redundant_bias=config.redundant_bias,
            )
            self.cell.add_module(str(rep), fpn_layer)
            feature_info = fpn_layer.feature_info

    def forward(self, x: List[torch.Tensor]):
        for resample in self.resample.values():
            x.append(resample(x[-1]))
        x = self.cell(x)
        return x


def get_feature_info(backbone):
    if isinstance(backbone.feature_info, Callable):
        # old accessor for timm versions <= 0.1.30, efficientnet and mobilenetv3 and related nets only
        feature_info = [
            dict(num_chs=f["num_chs"], reduction=f["reduction"])
            for i, f in enumerate(backbone.feature_info())
        ]
    else:
        # new feature info accessor, timm >= 0.2, all models supported
        feature_info = backbone.feature_info.get_dicts(keys=["num_chs", "reduction"])
    return feature_info


if __name__ == "__main__":

    cfg = None

    bifpn = BiFpn(cfg, feature_info)
