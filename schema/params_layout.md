# Node Params Layout

Flat `int` array per `Node`. CPU kernels ignore grid/block indices.

## Conv2d / Conv2dRelu
`[0:stride_h, 1:stride_w, 2:padding_h, 3:padding_w, 4:dilation_h, 5:dilation_w, 6:groups, 7:grid_x, 8:grid_y, 9:grid_z, 10:block_x, 11:block_y, 12:block_z]`

## MaxPool2d
`[0:kernel_h, 1:kernel_w, 2:stride_h, 3:stride_w, 4:padding_h, 5:padding_w, 6:dilation_h, 7:dilation_w, 8:ceil_mode, 9:grid_x, 10:grid_y, 11:grid_z, 12:block_x, 13:block_y, 14:block_z]`

## AvgPool2d
`[0:kernel_h, 1:kernel_w, 2:stride_h, 3:stride_w, 4:padding_h, 5:padding_w, 6:ceil_mode, 7:grid_x, 8:grid_y, 9:grid_z, 10:block_x, 11:block_y, 12:block_z]`

## AdaptiveAvgPool2d
`[0:output_h, 1:output_w, 2:grid_x, 3:grid_y, 4:grid_z, 5:block_x, 6:block_y, 7:block_z]`

## Linear / LinearRelu / Add / AddRelu / Relu / View / Placeholder
`[]`