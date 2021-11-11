## BSRN: Lightweight and Efficient Image Super-Resolution with Block State-based Recursive Network

原始模型参考 [github链接](https://github.com/idearibosome/tf-bsrn-sr/), 迁移训练代码到NPU

### Requirments
- Tensorflow 1.15.0
- Ascend 910
- GPU运行平台 Tesla V100
- NPU运行平台 ModelArts
- 其他依赖参考requirment.txt
- Dataset: 
  - 训练数据集：DIV2K 
  - 验证数据集：BSD100

- Model: **BSRN** for Super Resolution(超分辨率)

### 代码路径解释
```shell
├── temp										----存放训练结果及数据集文件
│   ├── result									----存放训练结果（自动生成）
│   │   ├── model.ckpt							----存放固化的模型pbtxt文件
│   │   ├── result-pictures						----存放验证数据（运行validate_gpu.sh）自动生成的超分辨率图片
│   │   │   ├── ensemble
│   │   │   │   ├── x2
│   │   │   │   ├── x3
│   │   │   │   └── x4
│   └── dataset									----数据集文件
│       ├── BSD100								----验证数据集
│       │   ├── LR
│       │   │   ├── x2
│       │   │   ├── x3
│       │   │   └── x4
│       │   └── SR
│       └── DIV2K								----训练数据集
│           ├── DIV2K_train_HR
│           └── DIV2K_train_LR_bicubic
│               ├── X2
│               ├── X3
│               └── X4
├── tf-bsrn-sr
│   ├── checkpoints								----原始代码提供的训练好的模型文件，用作精度和性能比较
│   ├── dataloaders								----数据预处理和加载脚本，可以得到batch-size大小的数据
│   ├── models									----模型网络定义，保存，恢复及优化相关脚本
│   ├── scripts									----存放模型训练和验证脚本
│    	├── run_gpu.sh							----使用gpu
│    	├── run_npu.sh							----使用npu
│    	└── validate.sh							----验证模型精度
│   ├── boot_modelarts.py
│   ├── help_modelarts.py
│   ├── test_bsrn.py							----测试模型
│   ├── train.py								----训练模型
│   ├── output.txt								----训练输出
│   └── validate_bsrn.py						----验证模型
├── statics										----存放图片静态数据（用于md文件）
├── LICENSE
├── README.md
└── requirments.txt  							---- 依赖配置文件

```

### 数据集准备 
- Dataset: (请参考(https://github.com/idearibosome/tf-bsrn-sr/自行下载)
  - 训练数据集：DIV2K 
  - 验证数据集：BSD100
```shell
数据集组织
├── dataset									----数据集文件
    ├── BSD100								----验证数据集
    │   ├── LR
    │   │   ├── x2
    │   │   ├── x3
    │   │   └── x4
    │   └── SR
    └── DIV2K								----训练数据集
        ├── DIV2K_train_HR
        └── DIV2K_train_LR_bicubic
            ├── X2
            ├── X3
            └── X4
```
### GPU训练
命令行切换路径到`tf-bsrn-sr/`

- 训练bsrn, 详细的参数设置请参考脚本中的注释
```shell
nohup bash scripts/run_gpu.sh > output.txt 2>&1 &
```
- 训练之前需修改`boot_modelarts.py`中第77行代码为bash_header = os.path.join(code_dir, 'scripts/run_gpu.sh')

### GPU离线推理<font color='red'> 【在线推理待完善】 </font>

命令行切换路径到`tf-bsrn-sr/`，执行以下命令，详细的参数设置请参考脚本中的注释
```shell
bash scripts/test.sh
```
### GPU评估

命令行切换路径到`tf-bsrn-sr/`，执行以下命令，详细的参数设置请参考脚本中的注释

```shell
bash scripts/validate.sh
```

### NPU训练、推理、评估

使用pycharm ModelArts进行训练

ModelArts的使用请参考[模型开发向导_昇腾CANN社区版(5.0.2.alpha005)(训练)_TensorFlow模型迁移和训练_华为云 (huaweicloud.com)](https://support.huaweicloud.com/tfmigr-cann502alpha5training/atlasmprtg_13_9002.html)

配置方式请参考：

<img src="statics\modelarts配置.PNG" alt="modelarts配置" style="zoom: 67%;" />

（修改`boot_modelarts.py`中第77行代码bash_header = os.path.join(code_dir, 'scripts/run_npu.sh')，可以设置在NPU上跑还是在GPU上跑）

### 指标对比
均使用相同的训练集以及测试集，训练参数都相同。

NPU Checkpoints: ([百度云链接，提取码：xxxx]()) <font color='red'> 【链接待完善】 </font>

GPU Checkpoints: ([百度云链接，提取码：xxxx]()) <font color='red'> 【链接待完善】 </font>

作者论文中提供的各项指标值为：训练以USR_8x数据集为例。

|           | PSNR              | SSIM           | UQIM           |
| --------- | ----------------- | -------------- | -------------- |
| SRDRM     | 28.36/24.64/21.20 | 0.80/0.68/0.60 | 2.78/2.46/2.18 |
| SRDRM-GAN | 28.55/24.62/20.25 | 0.81/0.69/0.61 | 2.77/2.48/2.17 |

**(Average PSNR, SSIM, and UIQM scores for 2×/4×/8× SISR on USR-248 test set.)**


##### USR_8X  <font color='red'> 【srdrm-gan npu指标 待完善】 </font>
<table>
    <tr>
       <td>metrics</td>
       <td colspan="2" align="center">PSNR</td>
       <td colspan="2" align="center">SSIM</td>
       <td colspan="2" align="center">UQIM</td>
    </tr>
    <tr>
      <td>chip</td>
      <td>gpu</td>
      <td>npu</td>
      <td>gpu</td>
      <td>npu</td>
      <td>gpu</td>
      <td>npu</td>
    </tr>
    <tr>
      <td>srdrm</td>
      <td>22.74</td>
      <td>23.70</td>
      <td>0.63</td>
      <td>0.63</td>
      <td>2.28</td>
      <td>2.26</td>
    </tr>
    <tr>
      <td>srdrm-gan</td>
      <td>21.93</td>
      <td></td>
      <td>0.58</td>
      <td></td>
      <td>3.00</td>
      <td></td>
    </tr>
</table>



### 性能对比

此处只展示srdrm-gan模型在USR_2X数据集上的训练与测试结果。其他的性能数据可参考文件[百度网盘_task_log]()

#### 训练性能

###### 1.SRDRM for USR_8x

 NPU性能log截图

![image-20211013102523123](https://gitee.com/windclub/image_bed/raw/master/img/20211013102523.png)

 GPU性能log截图

![image-20211013102301982](https://gitee.com/windclub/image_bed/raw/master/img/20211013102309.png)

|   平台   | BatchSize | 训练性能(fps) |
| :------: | :-------: | :-----------: |
|   NPU    |     2     |      12       |
| GPU V100 |     2     |       8       |

NPU训练性能约为GPU训练性能的1.5倍。

###### 2.SRDRM-GAN for USR_8x

NPU性能log截图 <font color='red'> 【待完善】 </font>

GPU性能log截图

![image-20211013103121415](https://gitee.com/windclub/image_bed/raw/master/img/20211013103326.png)

|   平台   | BatchSize | 训练性能(fps) |
| :------: | :-------: | :-----------: |
|   NPU    |           |               |
| GPU V100 |     2     |       3       |

#### 推理性能 <font color='red'> 【待完善】 </font>

NPU性能log截图

GPU性能log截图



|   平台   | BatchSize | 训练性能(fps) |
| :------: | :-------: | :-----------: |
|   NPU    |           |               |
| GPU V100 |           |               |

#### 性能调优 <font color='red'> 【待完善】 </font>

##### NPU AutoTune性能

训练时开启AutoTune:

npu训练性能（命令行截图）

| 平台 | BatchSize | 训练性能(fps) |
| :--: | :-------: | :-----------: |
| NPU  |           |               |

- 2020-06-20 19:06:09.349677: I tf_adapter/kernels/geop_npu.cc:338] [GEOP] GeOp Finalize start, tf session: direct24135e275a110a29, graph_id_: 1

  2020-06-20 19:06:09.349684: I tf_adapter/kernels/geop_npu.cc:342] tf session: direct24135e275a110a29, graph id: 1

  2020-06-20 19:06:09.397087: I tf_adapter/kernels/geop_npu.cc:347] [GEOP] GE Remove Graph success. tf session: direct24135e275a110a29 , graph id: 1

  2020-06-20 19:06:09.397105: I tf_adapter/kernels/geop_npu.cc:368] [GEOP] GeOp Finalize success, tf session: direct24135e275a110a29, graph_id_: 1

  2020-06-20 19:06:09.398108: I tf_adapter/kernels/geop_npu.cc:338] [GEOP] GeOp Finalize start, tf session: direct24135e275a110a29, graph_id_: 31

  2020-06-20 19:06:09.398122: I tf_adapter/kernels/geop_npu.cc:368] [GEOP] GeOp Finalize success, tf session: direct24135e275a110a29, graph_id_: 31

  2020-06-20 19:06:09.398247: I tf_adapter/kernels/host_queue_dataset_op.cc:71] Start destroy tdt.

  2020-06-20 19:06:09.412269: I tf_adapter/kernels/host_queue_dataset_op.cc:77] Tdt client close success.

  2020-06-20 19:06:09.412288: I tf_adapter/kernels/host_queue_dataset_op.cc:83] dlclose handle finish.

  2020-06-20 19:06:09.412316: I tf_adapter/kernels/geop_npu.cc:338] [GEOP] GeOp Finalize start, tf session: direct24135e275a110a29, graph_id_: 51

  2020-06-20 19:06:09.412323: I tf_adapter/kernels/geop_npu.cc:342] tf session: direct24135e275a110a29, graph id: 51

  2020-06-20 19:06:09.553281: I tf_adapter/kernels/geop_npu.cc:347] [GEOP] GE Remove Graph success. tf session: direct24135e275a110a29 , graph id: 51

  2020-06-20 19:06:09.553299: I tf_adapter/kernels/geop_npu.cc:368] [GEOP] GeOp Finalize success, tf session: direct24135e275a110a29, graph_id_: 51

  2020-06-20 19:06:10.619514: I tf_adapter/kernels/host_queue_dataset_op.cc:172] HostQueueDatasetOp's iterator is released.

  2020-06-20 19:06:10.620037: I tf_adapter/kernels/geop_npu.cc:338] [GEOP] GeOp Finalize start, tf session: direct24135e275a110a29, graph_id_: 41

  2020-06-20 19:06:10.620054: I tf_adapter/kernels/geop_npu.cc:342] tf session: direct24135e275a110a29, graph id: 41

  2020-06-20 19:06:10.621564: I tf_adapter/kernels/geop_npu.cc:347] [GEOP] GE Remove Graph success. tf session: direct24135e275a110a29 , graph id: 41

  2020-06-20 19:06:10.622904: I tf_adapter/util/session_manager.cc:50] find ge session connect with tf session direct24135e275a110a29

  2020-06-20 19:06:10.975070: I tf_adapter/util/session_manager.cc:55] destory ge session connect with tf session direct24135e275a110a29 success.

  2020-06-20 19:06:11.380491: I tf_adapter/kernels/geop_npu.cc:388] [GEOP] Close TsdClient.

  2020-06-20 19:06:11.664666: I tf_adapter/kernels/geop_npu.cc:393] [GEOP] Close TsdClient success.

  2020-06-20 19:06:11.665011: I tf_adapter/kernels/geop_npu.cc:368] [GEOP] GeOp Finalize success, tf session: direct24135e275a110a29, graph_id_: 41 step  epoch  top1    top5     loss   checkpoint_time(UTC)85068    3.0  50.988   76.99    3.09  

  2020-06-20 18:06:0690072    3.0  51.569   77.51    3.03  

  2020-06-20 18:11:1495076    3.0  51.689   77.33    3.00  

  2020-06-20 18:16:22100080    3.0  51.426   77.04    3.08  

  2020-06-20 18:25:11105084    3.0  51.581   77.50    3.03  

  2020-06-20 18:34:23Finished evaluation