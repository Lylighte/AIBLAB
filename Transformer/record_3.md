(transformer) lighte@LIGHTE-5900:/mnt/c/Users/Lighte/GitHub/AIBLAB/Transformer$ ./run_wsl.sh
============================================
WSL 实验开始: Tue Jun  2 18:07:48 CST 2026
运行环境: OK
GPU: True
============================================

========== Group 1: lr × batch_size (epochs=10) ==========
>>> lr=5e-5  bs=32  ep=10  [dim=100 & 200]
num_classes: 2
vocab_size: 30522
input_ids shape: torch.Size([32, 512])
attention_mask shape: torch.Size([32, 512])
labels shape: torch.Size([32])
labels[:10]: tensor([1, 0, 0, 0, 1, 0, 1, 0, 1, 1])
test model before training
sentence_vector shape: torch.Size([32, 100])
logits shape: torch.Size([32, 2])
sentence_vector[0]: tensor([-0.1636,  0.6823, -0.8960,  0.4199,  0.4550,  0.6948, -0.7176, -0.4712,
         0.8122,  0.6005, -0.8442, -0.2405, -0.5408, -0.1770, -0.8909, -0.0661,
         0.8785,  0.7287, -0.6298, -0.8703, -0.8713,  0.2248, -0.8814, -0.6040,
        -0.5299,  0.6594,  0.5825,  0.9569, -0.9860,  0.0701,  0.5430,  0.8895,
        -0.8670,  0.0185,  0.3337,  0.3122, -0.3275,  0.7517, -0.2496,  0.5628,
         0.6918,  0.0393,  0.7528, -0.6562,  0.0311, -0.0504,  0.2817,  0.3132,
        -0.9306,  0.0386, -0.2573, -0.9140, -0.1712, -0.0712,  0.8910, -0.2114,
        -0.8171, -0.0269,  0.3192, -0.5593,  0.8623, -0.3802, -0.4907,  0.9789,
         0.3527,  0.3319, -0.5187,  0.3159,  0.5580,  0.5293, -0.4308,  0.3140,
        -0.2296,  0.9486, -0.1325,  0.8152,  0.9649,  0.9422,  0.8006,  0.0828,
         0.5313,  0.5958, -0.7757, -0.5251, -0.9755, -0.9894,  0.9376, -0.4876,
         0.6006, -0.9874,  0.0649, -0.9175, -0.6795,  0.5127,  0.1349,  0.6792,
        -0.6103,  0.1400, -0.1970,  0.4093], device='cuda:0',
       grad_fn=<SelectBackward0>)
logits[0]: tensor([ 0.2705, -0.3977], device='cuda:0', grad_fn=<SelectBackward0>)
============================================================
实验：对比 100维 vs 200维 词嵌入向量
============================================================
Training embed_dim=100...
Epoch 1 Loss: 0.6699
Epoch 2 Loss: 0.5568
Epoch 3 Loss: 0.4784
Epoch 4 Loss: 0.4335
Epoch 5 Loss: 0.4039
Epoch 6 Loss: 0.3897
Epoch 7 Loss: 0.3734
Epoch 8 Loss: 0.3632
Epoch 9 Loss: 0.3557
Epoch 10 Loss: 0.3469
Eval embed_dim=100...
/home/lighte/miniconda3/envs/transformer/lib/python3.12/site-packages/torch/nn/modules/transformer.py:529: UserWarning: The PyTorch API of nested tensors is in prototype stage and will change in the near future. We recommend specifying layout=torch.jagged when constructing a nested tensor, as this layout receives active development, has better operator coverage, and works with torch.compile. (Triggered internally at /pytorch/aten/src/ATen/NestedTensorImpl.cpp:178.)
  output = torch._nested_tensor_from_mask(

100-dim Embedding Accuracy: 0.8332
Training embed_dim=200...
Epoch 1 Loss: 0.6307
Epoch 2 Loss: 0.4844
Epoch 3 Loss: 0.4182
Epoch 4 Loss: 0.3846
Epoch 5 Loss: 0.3659
Epoch 6 Loss: 0.3457
Epoch 7 Loss: 0.3343
Epoch 8 Loss: 0.3180
Epoch 9 Loss: 0.3065
Epoch 10 Loss: 0.2999
Eval embed_dim=200...
200-dim Embedding Accuracy: 0.8427
cost time:1816.9338 seconds

>>> lr=5e-5  bs=64  ep=10  [dim=100 & 200]
num_classes: 2
vocab_size: 30522
input_ids shape: torch.Size([64, 512])
attention_mask shape: torch.Size([64, 512])
labels shape: torch.Size([64])
labels[:10]: tensor([0, 0, 0, 0, 1, 1, 1, 0, 0, 1])
test model before training
sentence_vector shape: torch.Size([64, 100])
logits shape: torch.Size([64, 2])
sentence_vector[0]: tensor([-0.5671, -0.9235,  0.5984,  0.0023, -0.2884, -0.9725,  0.6812, -0.2888,
        -0.9728, -0.7004,  0.3221, -0.4562, -0.6910,  0.7946,  0.8251, -0.6153,
        -0.5875, -0.2862, -0.4099,  0.3088, -0.4345, -0.5519,  0.0915,  0.5161,
         0.7056,  0.9540,  0.2513,  0.2458, -0.9201, -0.9744,  0.8658, -0.4892,
        -0.5125,  0.9190,  0.9294, -0.2665,  0.4087,  0.7386, -0.0563,  0.8085,
        -0.3315,  0.9175, -0.2491,  0.8255, -0.4800, -0.4442, -0.2239, -0.3320,
         0.9587, -0.1559, -0.1905, -0.8779, -0.1425, -0.7304, -0.9670,  0.5620,
        -0.8855,  0.2378,  0.6895,  0.3974, -0.9109,  0.4389, -0.8536,  0.1451,
        -0.8931, -0.5652,  0.2687, -0.7957, -0.2808, -0.7017, -0.3651,  0.8868,
        -0.1992,  0.9562, -0.7824,  0.2829,  0.3637, -0.8083,  0.6833,  0.8680,
        -0.9654,  0.7669, -0.1235,  0.6172, -0.1621,  0.2004,  0.7271,  0.9636,
         0.2353,  0.8998,  0.7185, -0.7024, -0.0984, -0.1520,  0.6569,  0.4042,
         0.1270, -0.0258, -0.5685,  0.9676], device='cuda:0',
       grad_fn=<SelectBackward0>)
logits[0]: tensor([-0.0412, -0.4807], device='cuda:0', grad_fn=<SelectBackward0>)
============================================================
实验：对比 100维 vs 200维 词嵌入向量
============================================================
Training embed_dim=100...
Epoch 1 Loss: 0.6824
Epoch 2 Loss: 0.5878
Epoch 3 Loss: 0.5273
Epoch 4 Loss: 0.5013
Epoch 5 Loss: 0.4684
Epoch 6 Loss: 0.4489
Epoch 7 Loss: 0.4347
Epoch 8 Loss: 0.4177
Epoch 9 Loss: 0.4106
Epoch 10 Loss: 0.4052
Eval embed_dim=100...
/home/lighte/miniconda3/envs/transformer/lib/python3.12/site-packages/torch/nn/modules/transformer.py:529: UserWarning: The PyTorch API of nested tensors is in prototype stage and will change in the near future. We recommend specifying layout=torch.jagged when constructing a nested tensor, as this layout receives active development, has better operator coverage, and works with torch.compile. (Triggered internally at /pytorch/aten/src/ATen/NestedTensorImpl.cpp:178.)
  output = torch._nested_tensor_from_mask(

100-dim Embedding Accuracy: 0.8118
Training embed_dim=200...
Epoch 1 Loss: 0.6555
Epoch 2 Loss: 0.5258
Epoch 3 Loss: 0.4527
Epoch 4 Loss: 0.4130
Epoch 5 Loss: 0.3859
Epoch 6 Loss: 0.3626
Epoch 7 Loss: 0.3531
Epoch 8 Loss: 0.3412
Epoch 9 Loss: 0.3314
Epoch 10 Loss: 0.3238
Eval embed_dim=200...
200-dim Embedding Accuracy: 0.8358
cost time:1756.5637 seconds

>>> lr=1e-4  bs=32  ep=10  [dim=100 & 200]
num_classes: 2
vocab_size: 30522
input_ids shape: torch.Size([32, 512])
attention_mask shape: torch.Size([32, 512])
labels shape: torch.Size([32])
labels[:10]: tensor([0, 0, 1, 0, 0, 0, 1, 1, 1, 1])
test model before training
sentence_vector shape: torch.Size([32, 100])
logits shape: torch.Size([32, 2])
sentence_vector[0]: tensor([-0.9030, -0.7075, -0.8809, -0.3543,  0.9628,  0.3522,  0.2143,  0.6917,
        -0.8876,  0.9157, -0.7794,  0.9330,  0.5446, -0.7625, -0.1326,  0.7884,
        -0.0830,  0.2348, -0.5870,  0.3215,  0.0129, -0.1646,  0.8217, -0.7503,
        -0.9667,  0.1667, -0.9412, -0.3574, -0.4996, -0.5261,  0.9712,  0.7798,
         0.5840, -0.8047,  0.4511, -0.1796, -0.5108, -0.3730,  0.7857,  0.0122,
         0.4212,  0.0294, -0.3508, -0.3384, -0.8238, -0.5173, -0.8641,  0.3087,
        -0.9948, -0.6816, -0.2261,  0.5854,  0.9197,  0.8576, -0.7241,  0.9204,
        -0.6251,  0.7883,  0.3072,  0.4583, -0.9769, -0.0501, -0.7682,  0.5563,
        -0.5623,  0.7610,  0.3015,  0.9027, -0.6976, -0.5509,  0.2897, -0.7627,
         0.6699,  0.9234,  0.7832, -0.5387, -0.9545,  0.0085,  0.6862,  0.8141,
        -0.2668, -0.4715, -0.1787,  0.6930, -0.4578,  0.5684, -0.3659, -0.7237,
        -0.8923,  0.7786,  0.7252,  0.9011,  0.7189, -0.3849,  0.4526, -0.0678,
         0.9172, -0.5303,  0.2083,  0.0055], device='cuda:0',
       grad_fn=<SelectBackward0>)
logits[0]: tensor([0.2940, 0.0930], device='cuda:0', grad_fn=<SelectBackward0>)
============================================================
实验：对比 100维 vs 200维 词嵌入向量
============================================================
Training embed_dim=100...
Epoch 1 Loss: 0.6474
Epoch 2 Loss: 0.5158
Epoch 3 Loss: 0.4480
Epoch 4 Loss: 0.4103
Epoch 5 Loss: 0.3842
Epoch 6 Loss: 0.3645
Epoch 7 Loss: 0.3514
Epoch 8 Loss: 0.3392
Epoch 9 Loss: 0.3292
Epoch 10 Loss: 0.3219
Eval embed_dim=100...
/home/lighte/miniconda3/envs/transformer/lib/python3.12/site-packages/torch/nn/modules/transformer.py:529: UserWarning: The PyTorch API of nested tensors is in prototype stage and will change in the near future. We recommend specifying layout=torch.jagged when constructing a nested tensor, as this layout receives active development, has better operator coverage, and works with torch.compile. (Triggered internally at /pytorch/aten/src/ATen/NestedTensorImpl.cpp:178.)
  output = torch._nested_tensor_from_mask(

100-dim Embedding Accuracy: 0.8413
Training embed_dim=200...
Epoch 1 Loss: 0.6456
Epoch 2 Loss: 0.5004
Epoch 3 Loss: 0.4166
Epoch 4 Loss: 0.3712
Epoch 5 Loss: 0.3410
Epoch 6 Loss: 0.3187
Epoch 7 Loss: 0.2987
Epoch 8 Loss: 0.2826
Epoch 9 Loss: 0.2695
Epoch 10 Loss: 0.2607
Eval embed_dim=200...
200-dim Embedding Accuracy: 0.8494
cost time:1768.5675 seconds

>>> lr=1e-4  bs=64  ep=10  [dim=100 & 200]
num_classes: 2
vocab_size: 30522
input_ids shape: torch.Size([64, 512])
attention_mask shape: torch.Size([64, 512])
labels shape: torch.Size([64])
labels[:10]: tensor([1, 0, 0, 1, 1, 0, 0, 0, 1, 1])
test model before training
sentence_vector shape: torch.Size([64, 100])
logits shape: torch.Size([64, 2])
sentence_vector[0]: tensor([ 1.7574e-04, -8.0472e-01,  4.7262e-01,  6.9958e-01,  6.2527e-01,
        -2.0763e-01,  4.1823e-01, -2.3303e-01,  8.0703e-01, -9.0787e-01,
        -9.3095e-01,  4.5934e-01, -5.7389e-01, -4.8358e-01,  6.6225e-02,
        -5.1736e-01,  5.4591e-01, -9.4850e-01, -2.4902e-01, -6.8468e-01,
         7.1622e-02, -1.5720e-01, -9.4663e-01, -5.8707e-01, -1.6561e-01,
         9.5602e-01, -4.7521e-01,  9.3191e-01,  5.8201e-01, -5.5353e-01,
        -3.5444e-01,  8.9220e-01, -6.6929e-01, -1.9123e-01, -9.2472e-01,
        -9.6821e-01,  6.5808e-01, -9.8019e-01, -4.5750e-01, -8.4152e-01,
        -7.4432e-01, -6.7827e-01,  1.7025e-01,  6.4903e-01, -9.0452e-01,
         1.4955e-01,  7.9322e-01,  9.0906e-01, -7.6064e-01, -6.6472e-01,
         3.6889e-01,  9.3538e-01,  8.9966e-01,  5.3325e-01, -3.4104e-01,
        -7.3177e-01, -2.7271e-01, -2.4444e-01, -8.0629e-01,  6.6290e-01,
        -8.3654e-01, -9.0666e-01,  8.3115e-01,  7.6304e-01,  4.0345e-01,
        -8.5377e-01,  6.0802e-01,  4.2580e-01, -3.6905e-01,  9.0975e-01,
        -5.6717e-02,  4.7377e-01, -2.2836e-01,  6.0299e-01, -7.0705e-01,
         4.9448e-01,  7.1859e-01, -9.5898e-02,  8.9863e-01,  6.3937e-01,
         4.5199e-01, -3.8951e-01,  8.3777e-01,  9.4767e-01, -6.8625e-01,
         8.9560e-01,  3.9117e-01, -2.4405e-01, -3.3225e-01,  3.7648e-01,
         3.0847e-03, -1.2619e-01,  1.4197e-01,  6.3392e-01, -6.7750e-01,
         3.8211e-01,  8.6055e-01,  9.8672e-01, -9.5223e-01, -3.4143e-01],
       device='cuda:0', grad_fn=<SelectBackward0>)
logits[0]: tensor([ 0.4483, -0.3740], device='cuda:0', grad_fn=<SelectBackward0>)
============================================================
实验：对比 100维 vs 200维 词嵌入向量
============================================================
Training embed_dim=100...
Epoch 1 Loss: 0.6889
Epoch 2 Loss: 0.5549
Epoch 3 Loss: 0.4714
Epoch 4 Loss: 0.4312
Epoch 5 Loss: 0.4096
Epoch 6 Loss: 0.3831
Epoch 7 Loss: 0.3651
Epoch 8 Loss: 0.3491
Epoch 9 Loss: 0.3403
Epoch 10 Loss: 0.3345
Eval embed_dim=100...
/home/lighte/miniconda3/envs/transformer/lib/python3.12/site-packages/torch/nn/modules/transformer.py:529: UserWarning: The PyTorch API of nested tensors is in prototype stage and will change in the near future. We recommend specifying layout=torch.jagged when constructing a nested tensor, as this layout receives active development, has better operator coverage, and works with torch.compile. (Triggered internally at /pytorch/aten/src/ATen/NestedTensorImpl.cpp:178.)
  output = torch._nested_tensor_from_mask(

100-dim Embedding Accuracy: 0.8393
Training embed_dim=200...
Epoch 1 Loss: 0.6680
Epoch 2 Loss: 0.4840
Epoch 3 Loss: 0.4171
Epoch 4 Loss: 0.3698
Epoch 5 Loss: 0.3379
Epoch 6 Loss: 0.3181
Epoch 7 Loss: 0.3021
Epoch 8 Loss: 0.2903
Epoch 9 Loss: 0.2787
Epoch 10 Loss: 0.2743
Eval embed_dim=200...
200-dim Embedding Accuracy: 0.8520
cost time:1813.3345 seconds

>>> lr=2e-4  bs=32  ep=10  [dim=100 & 200]
num_classes: 2
vocab_size: 30522
input_ids shape: torch.Size([32, 512])
attention_mask shape: torch.Size([32, 512])
labels shape: torch.Size([32])
labels[:10]: tensor([1, 1, 1, 0, 0, 0, 1, 1, 1, 0])
test model before training
sentence_vector shape: torch.Size([32, 100])
logits shape: torch.Size([32, 2])
sentence_vector[0]: tensor([-0.8356, -0.8736, -0.8188, -0.5791,  0.5753,  0.2960,  0.9301,  0.6869,
         0.9842, -0.5698,  0.6125,  0.5864, -0.7833,  0.3893,  0.8153, -0.6470,
         0.9328, -0.7488,  0.7934, -0.5133, -0.1895,  0.2289,  0.9282,  0.2674,
        -0.9113, -0.4999,  0.6378,  0.7137,  0.8049,  0.7278, -0.7936,  0.7006,
         0.8157, -0.3301,  0.1562, -0.7951, -0.6053, -0.4956,  0.8326,  0.1215,
         0.8900,  0.6447, -0.5639,  0.9926, -0.9000,  0.2496, -0.6632, -0.8489,
        -0.0784, -0.6782, -0.3408, -0.3650, -0.9410, -0.3471, -0.8031, -0.2102,
         0.2031,  0.9682,  0.5400, -0.2551,  0.4072,  0.8713, -0.8574,  0.8866,
        -0.8146, -0.3723,  0.0842, -0.2802,  0.8046,  0.3476, -0.1692, -0.6174,
         0.7935, -0.7742, -0.2384, -0.8711,  0.4232, -0.2326, -0.9686, -0.9745,
         0.4844,  0.8235,  0.1984, -0.3584, -0.9132,  0.7862,  0.0423,  0.6949,
        -0.0134, -0.5476,  0.6354, -0.8108, -0.8988, -0.5128, -0.3608,  0.4670,
         0.2739,  0.5605, -0.1855, -0.0528], device='cuda:0',
       grad_fn=<SelectBackward0>)
logits[0]: tensor([-0.1231,  0.3746], device='cuda:0', grad_fn=<SelectBackward0>)
============================================================
实验：对比 100维 vs 200维 词嵌入向量
============================================================
Training embed_dim=100...
Epoch 1 Loss: 0.6957
Epoch 2 Loss: 0.6946
Epoch 3 Loss: 0.6939
Epoch 4 Loss: 0.6946
Epoch 5 Loss: 0.6939
Epoch 6 Loss: 0.6936
Epoch 7 Loss: 0.6936
Epoch 8 Loss: 0.6936
Epoch 9 Loss: 0.6934
Epoch 10 Loss: 0.6932
Eval embed_dim=100...
/home/lighte/miniconda3/envs/transformer/lib/python3.12/site-packages/torch/nn/modules/transformer.py:529: UserWarning: The PyTorch API of nested tensors is in prototype stage and will change in the near future. We recommend specifying layout=torch.jagged when constructing a nested tensor, as this layout receives active development, has better operator coverage, and works with torch.compile. (Triggered internally at /pytorch/aten/src/ATen/NestedTensorImpl.cpp:178.)
  output = torch._nested_tensor_from_mask(

100-dim Embedding Accuracy: 0.7057
Training embed_dim=200...
Epoch 1 Loss: 0.6976
Epoch 2 Loss: 0.6942
Epoch 3 Loss: 0.6621
Epoch 4 Loss: 0.6561
Epoch 5 Loss: 0.6556
Epoch 6 Loss: 0.6557
Epoch 7 Loss: 0.6555
Epoch 8 Loss: 0.6551
Epoch 9 Loss: 0.6550
Epoch 10 Loss: 0.6549
Eval embed_dim=200...
200-dim Embedding Accuracy: 0.6586
cost time:1832.9596 seconds

>>> lr=2e-4  bs=64  ep=10  [dim=100 & 200]
num_classes: 2
vocab_size: 30522
input_ids shape: torch.Size([64, 512])
attention_mask shape: torch.Size([64, 512])
labels shape: torch.Size([64])
labels[:10]: tensor([0, 1, 1, 0, 1, 1, 1, 1, 1, 1])
test model before training
sentence_vector shape: torch.Size([64, 100])
logits shape: torch.Size([64, 2])
sentence_vector[0]: tensor([ 0.0863,  0.6972, -0.8279,  0.9960,  0.6189,  0.6677, -0.0183,  0.3355,
        -0.8902, -0.0201, -0.1286,  0.9757,  0.1833, -0.9136, -0.1385,  0.5261,
         0.7998, -0.1416, -0.8907, -0.9078,  0.9466, -0.8606, -0.0379, -0.7183,
        -0.8377,  0.7910, -0.8244, -0.8081,  0.5448,  0.3064,  0.6206,  0.4448,
        -0.6111,  0.9796, -0.6497,  0.3680,  0.1822,  0.8977, -0.9318,  0.6913,
        -0.8107, -0.9101, -0.5149,  0.5462, -0.4395, -0.9206, -0.6851,  0.6350,
         0.0398, -0.6466,  0.4915, -0.9486, -0.5406,  0.3111,  0.6745, -0.5107,
         0.0034,  0.6341, -0.8367, -0.7972, -0.2074, -0.2181, -0.8486,  0.4426,
        -0.0687, -0.8109,  0.8650, -0.3880,  0.0505,  0.8789,  0.0389,  0.3209,
        -0.9385,  0.4678,  0.2198,  0.6426, -0.7435, -0.4750,  0.5276,  0.7364,
         0.9031,  0.8167,  0.7734,  0.0889,  0.5785, -0.7579,  0.6817, -0.4893,
         0.1473,  0.8240, -0.8675,  0.5740, -0.2261, -0.7591,  0.7339, -0.8405,
        -0.5678,  0.5345,  0.3153,  0.8787], device='cuda:0',
       grad_fn=<SelectBackward0>)
logits[0]: tensor([-0.4093,  0.5277], device='cuda:0', grad_fn=<SelectBackward0>)
============================================================
实验：对比 100维 vs 200维 词嵌入向量
============================================================
Training embed_dim=100...
Epoch 1 Loss: 0.6905
Epoch 2 Loss: 0.5201
Epoch 3 Loss: 0.4278
Epoch 4 Loss: 0.3903
Epoch 5 Loss: 0.3625
Epoch 6 Loss: 0.3408
Epoch 7 Loss: 0.3222
Epoch 8 Loss: 0.3112
Epoch 9 Loss: 0.3033
Epoch 10 Loss: 0.2970
Eval embed_dim=100...
/home/lighte/miniconda3/envs/transformer/lib/python3.12/site-packages/torch/nn/modules/transformer.py:529: UserWarning: The PyTorch API of nested tensors is in prototype stage and will change in the near future. We recommend specifying layout=torch.jagged when constructing a nested tensor, as this layout receives active development, has better operator coverage, and works with torch.compile. (Triggered internally at /pytorch/aten/src/ATen/NestedTensorImpl.cpp:178.)
  output = torch._nested_tensor_from_mask(

100-dim Embedding Accuracy: 0.8484
Training embed_dim=200...
Epoch 1 Loss: 0.6965
Epoch 2 Loss: 0.6942
Epoch 3 Loss: 0.6936
Epoch 4 Loss: 0.6940
Epoch 5 Loss: 0.6937
Epoch 6 Loss: 0.6935
Epoch 7 Loss: 0.6935
Epoch 8 Loss: 0.6935
Epoch 9 Loss: 0.6933
Epoch 10 Loss: 0.6933
Eval embed_dim=200...
200-dim Embedding Accuracy: 0.6602
cost time:1825.7110 seconds

========== Group 2: epochs comparison (lr=1e-4, bs=32) ==========
>>> lr=1e-4  bs=32  ep=5  [dim=100 & 200]
num_classes: 2
vocab_size: 30522
input_ids shape: torch.Size([32, 512])
attention_mask shape: torch.Size([32, 512])
labels shape: torch.Size([32])
labels[:10]: tensor([1, 1, 1, 0, 0, 1, 1, 1, 1, 0])
test model before training
sentence_vector shape: torch.Size([32, 100])
logits shape: torch.Size([32, 2])
sentence_vector[0]: tensor([-0.5860, -0.2756,  0.2822, -0.6384,  0.8009,  0.0553, -0.5086, -0.6502,
        -0.7258,  0.1820,  0.2435,  0.3684,  0.9706, -0.7299,  0.3127, -0.0573,
        -0.9709,  0.8003,  0.6624,  0.1791,  0.5771, -0.7516, -0.2641,  0.6813,
         0.7341,  0.4875, -0.3117, -0.4656,  0.2809,  0.0366, -0.2009, -0.9250,
        -0.8364, -0.3786, -0.6496,  0.0261, -0.5740,  0.6701, -0.9171,  0.8327,
        -0.0201, -0.7850, -0.9834, -0.9978,  0.5412, -0.4766,  0.9329, -0.5561,
         0.7811, -0.0097,  0.5835, -0.7727,  0.7263,  0.6419, -0.1203, -0.1918,
         0.9275, -0.4410, -0.7485,  0.7237, -0.1224, -0.7516, -0.0999, -0.6045,
        -0.4349,  0.1553,  0.4514,  0.8239, -0.2442,  0.5074, -0.9084,  0.9320,
         0.8646,  0.9120, -0.2791, -0.9776, -0.0736, -0.6926, -0.3663,  0.6779,
         0.8793, -0.3995,  0.5072, -0.4501,  0.8429,  0.7038,  0.6084,  0.9323,
         0.1973, -0.4919, -0.0581, -0.5969,  0.2380,  0.4077, -0.8379,  0.8931,
         0.0353,  0.9835, -0.6836, -0.5401], device='cuda:0',
       grad_fn=<SelectBackward0>)
logits[0]: tensor([-0.2803, -0.1466], device='cuda:0', grad_fn=<SelectBackward0>)
============================================================
实验：对比 100维 vs 200维 词嵌入向量
============================================================
Training embed_dim=100...
Epoch 1 Loss: 0.6882
Epoch 2 Loss: 0.6051
Epoch 3 Loss: 0.5012
Epoch 4 Loss: 0.4434
Epoch 5 Loss: 0.4156
Eval embed_dim=100...
/home/lighte/miniconda3/envs/transformer/lib/python3.12/site-packages/torch/nn/modules/transformer.py:529: UserWarning: The PyTorch API of nested tensors is in prototype stage and will change in the near future. We recommend specifying layout=torch.jagged when constructing a nested tensor, as this layout receives active development, has better operator coverage, and works with torch.compile. (Triggered internally at /pytorch/aten/src/ATen/NestedTensorImpl.cpp:178.)
  output = torch._nested_tensor_from_mask(

100-dim Embedding Accuracy: 0.8054
Training embed_dim=200...
Epoch 1 Loss: 0.6479
Epoch 2 Loss: 0.4668
Epoch 3 Loss: 0.3985
Epoch 4 Loss: 0.3622
Epoch 5 Loss: 0.3411
Eval embed_dim=200...
200-dim Embedding Accuracy: 0.8373
cost time:956.5131 seconds

>>> lr=1e-4  bs=32  ep=10  [dim=100 & 200]
num_classes: 2
vocab_size: 30522
input_ids shape: torch.Size([32, 512])
attention_mask shape: torch.Size([32, 512])
labels shape: torch.Size([32])
labels[:10]: tensor([0, 1, 0, 1, 0, 1, 1, 1, 1, 0])
test model before training
sentence_vector shape: torch.Size([32, 100])
logits shape: torch.Size([32, 2])
sentence_vector[0]: tensor([-0.7921,  0.8453,  0.0507, -0.6991,  0.6301, -0.9372, -0.5916,  0.3861,
         0.7779,  0.3374,  0.4340,  0.9864, -0.7170,  0.7057, -0.2835, -0.8674,
        -0.7669,  0.0118,  0.3132,  0.9387, -0.2973, -0.8122, -0.0727, -0.5281,
         0.4253,  0.1157,  0.5623,  0.7007,  0.9056,  0.2704, -0.7183,  0.7788,
        -0.2049,  0.3448,  0.8338,  0.6091,  0.3380,  0.0836, -0.6386,  0.0909,
         0.2575, -0.8621,  0.6821, -0.9087, -0.1461, -0.6182,  0.0302,  0.6420,
        -0.0517,  0.7493,  0.6628, -0.7989,  0.8995,  0.2586,  0.9576, -0.9095,
        -0.1726, -0.5661, -0.3413, -0.8488, -0.8928, -0.7015, -0.9112, -0.4586,
        -0.0485, -0.2300, -0.4997,  0.6055, -0.9199, -0.8322,  0.6823,  0.9408,
         0.0438,  0.8662, -0.4586, -0.9701, -0.8504,  0.5766,  0.2509, -0.7947,
         0.0735,  0.9758, -0.8508, -0.8902,  0.8798,  0.4214,  0.1213, -0.1889,
        -0.4989,  0.3340,  0.5405,  0.8187,  0.9775,  0.9389, -0.8055,  0.1871,
        -0.7590, -0.2312, -0.3449, -0.8142], device='cuda:0',
       grad_fn=<SelectBackward0>)
logits[0]: tensor([ 0.2971, -0.6895], device='cuda:0', grad_fn=<SelectBackward0>)
============================================================
实验：对比 100维 vs 200维 词嵌入向量
============================================================
Training embed_dim=100...
Epoch 1 Loss: 0.6745
Epoch 2 Loss: 0.5322
Epoch 3 Loss: 0.4641
Epoch 4 Loss: 0.4164
Epoch 5 Loss: 0.3830
Epoch 6 Loss: 0.3640
Epoch 7 Loss: 0.3483
Epoch 8 Loss: 0.3369
Epoch 9 Loss: 0.3288
Epoch 10 Loss: 0.3223
Eval embed_dim=100...
/home/lighte/miniconda3/envs/transformer/lib/python3.12/site-packages/torch/nn/modules/transformer.py:529: UserWarning: The PyTorch API of nested tensors is in prototype stage and will change in the near future. We recommend specifying layout=torch.jagged when constructing a nested tensor, as this layout receives active development, has better operator coverage, and works with torch.compile. (Triggered internally at /pytorch/aten/src/ATen/NestedTensorImpl.cpp:178.)
  output = torch._nested_tensor_from_mask(

100-dim Embedding Accuracy: 0.8397
Training embed_dim=200...
Epoch 1 Loss: 0.6336
Epoch 2 Loss: 0.4829
Epoch 3 Loss: 0.4070
Epoch 4 Loss: 0.3607
Epoch 5 Loss: 0.3311
Epoch 6 Loss: 0.3104
Epoch 7 Loss: 0.2940
Epoch 8 Loss: 0.2780
Epoch 9 Loss: 0.2645
Epoch 10 Loss: 0.2553
Eval embed_dim=200...
200-dim Embedding Accuracy: 0.8556
cost time:1828.6029 seconds

============================================
实验完成: Tue Jun  2 21:57:33 CST 2026

结果汇总 (outputs/score.txt):
--------------------------------------------
Result Comparison:
100-dim Model Accuracy: 0.7853
200-dim Model Accuracy: 0.8214
cost time:630.9646 seconds
100 Model Accuracy: 0.7908
cost time:323.1905 seconds
============================================