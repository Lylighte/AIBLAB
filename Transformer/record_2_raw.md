(transformer) lighte@LIGHTE-5900:/mnt/c/Users/Lighte/GitHub/AIBLAB/Transformer$ ./run_experiments.sh 
==========================================
Experiment Start: Wed May 27 14:05:54 CST 2026
==========================================

========== 学习率调优（第二轮） ==========
>>> lr=5e-5 batch=64 epochs=10 emb_dim=100
2
[tensor([[ 101, 2145, 1996,  ..., 2162, 2003,  102],
        [ 101, 2295, 3581,  ...,    0,    0,    0],
        [ 101, 2023, 2003,  ...,    0,    0,    0],
        ...,
        [ 101, 2065, 2017,  ...,    0,    0,    0],
        [ 101, 2070, 2111,  ...,    0,    0,    0],
        [ 101, 2307, 3185,  ...,    0,    0,    0]]), tensor([[1, 1, 1,  ..., 1, 1, 1],
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        ...,
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0]]), tensor([1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1,
        0, 1, 1, 0, 0, 1, 1, 1])]
/mnt/c/Users/Lighte/GitHub/AIBLAB/Transformer/nlp_code.py:87: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
test model before training
tensor([[ 0.6567,  0.2228, -0.6713,  ...,  0.6636, -0.1735, -0.7034],
        [ 0.8691,  0.3317, -0.8716,  ...,  0.3756, -0.1762, -0.5967],
        [-0.0178,  0.0259, -0.8064,  ...,  0.7627,  0.1562, -0.2778],
        ...,
        [ 0.6511, -0.2642, -0.5227,  ...,  0.5919,  0.2172, -0.1739],
        [ 0.4369, -0.2657, -0.5685,  ...,  0.3626, -0.0855, -0.6416],
        [-0.4024, -0.1450, -0.6398,  ...,  0.7277,  0.3229, -0.7380]],
       device='cuda:0', grad_fn=<TanhBackward0>)
tensor([[-0.0833,  0.4175],
        [-0.1237,  0.1256],
        [ 0.0739, -0.0219],
        [-0.0965,  0.1397],
        [-0.1325,  0.5222],
        [ 0.1895,  0.1964],
        [-0.1552,  0.3360],
        [-0.0884,  0.4545],
        [-0.4108,  0.1382],
        [-0.0258,  0.5310],
        [-0.1209,  0.3674],
        [-0.1002,  0.2797],
        [-0.2592,  0.4144],
        [-0.4403,  0.3630],
        [-0.3161,  0.4541],
        [ 0.0611,  0.3627],
        [-0.2016,  0.3408],
        [-0.2843,  0.2931],
        [ 0.0067,  0.2801],
        [-0.1037,  0.2254],
        [ 0.0070,  0.3778],
        [-0.2379,  0.4963],
        [-0.0695,  0.3620],
        [-0.3226,  0.4253],
        [-0.1934,  0.6628],
        [-0.0605,  0.4396],
        [-0.1430,  0.5154],
        [-0.1390,  0.4673],
        [-0.0990,  0.5761],
        [-0.1764,  0.2772],
        [-0.0681,  0.1582],
        [ 0.1890,  0.4336]], device='cuda:0', grad_fn=<AddmmBackward0>)
/mnt/c/Users/Lighte/GitHub/AIBLAB/Transformer/nlp_code.py:87: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
Training vector_dim=100...
Epoch 1 Loss: 0.6412
Epoch 2 Loss: 0.5143
Epoch 3 Loss: 0.4402
Epoch 4 Loss: 0.4018
Epoch 5 Loss: 0.3712
Epoch 6 Loss: 0.3451
Epoch 7 Loss: 0.3259
Epoch 8 Loss: 0.3110
Epoch 9 Loss: 0.2978
Epoch 10 Loss: 0.2896
Eval vector_dim=100...

============================================================
  RESULT | lr=5e-05  batch=64  epochs=10  emb_dim=100
  ACCURACY: 0.8165  |  TIME: 801.95s
============================================================

>>> lr=1e-4 batch=64 epochs=10 emb_dim=100
2
[tensor([[ 101, 2023, 3185,  ...,    0,    0,    0],
        [ 101, 1996, 3772,  ...,    0,    0,    0],
        [ 101, 2643, 1010,  ...,    0,    0,    0],
        ...,
        [ 101, 2023, 3185,  ...,    0,    0,    0],
        [ 101, 1006, 9875,  ...,    0,    0,    0],
        [ 101, 1996, 3185,  ...,    0,    0,    0]]), tensor([[1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        ...,
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0]]), tensor([1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0,
        0, 1, 0, 1, 0, 1, 0, 0])]
/mnt/c/Users/Lighte/GitHub/AIBLAB/Transformer/nlp_code.py:87: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
test model before training
tensor([[-0.9069, -0.3064,  0.5617,  ..., -0.5872,  0.4822, -0.9464],
        [-0.9640,  0.5194, -0.5298,  ..., -0.5317,  0.5406, -0.9320],
        [-0.9819,  0.1040, -0.4266,  ..., -0.7783, -0.0376, -0.8760],
        ...,
        [-0.9500, -0.2068, -0.6505,  ..., -0.7052,  0.8065, -0.9859],
        [-0.9785,  0.6434, -0.6333,  ..., -0.8268,  0.7344, -0.9471],
        [-0.9077,  0.6821,  0.1059,  ..., -0.8262,  0.4641, -0.9804]],
       device='cuda:0', grad_fn=<TanhBackward0>)
tensor([[ 0.0404,  0.4722],
        [ 0.2903,  0.7752],
        [ 0.0812,  0.6919],
        [ 0.1531,  0.5963],
        [-0.0120,  0.6574],
        [-0.2619,  0.8644],
        [-0.2288,  0.8172],
        [-0.0840,  0.4816],
        [-0.0898,  0.7254],
        [ 0.1035,  0.5852],
        [-0.1820,  0.7791],
        [-0.0886,  0.5081],
        [-0.1566,  0.4600],
        [-0.1151,  0.3923],
        [-0.2169,  0.6217],
        [ 0.2263,  0.5144],
        [ 0.1130,  0.6298],
        [ 0.1028,  0.6554],
        [-0.0437,  0.6818],
        [-0.1468,  0.4506],
        [-0.0590,  0.5183],
        [ 0.1472,  0.5453],
        [ 0.2572,  0.5192],
        [ 0.0299,  0.5556],
        [ 0.2213,  0.4969],
        [ 0.0309,  0.6698],
        [-0.0344,  0.4864],
        [ 0.1163,  0.4245],
        [ 0.5310,  0.6045],
        [-0.1375,  0.6043],
        [-0.0408,  0.6053],
        [ 0.0461,  0.5946]], device='cuda:0', grad_fn=<AddmmBackward0>)
/mnt/c/Users/Lighte/GitHub/AIBLAB/Transformer/nlp_code.py:87: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
Training vector_dim=100...
Epoch 1 Loss: 0.6318
Epoch 2 Loss: 0.4865
Epoch 3 Loss: 0.4134
Epoch 4 Loss: 0.3726
Epoch 5 Loss: 0.3357
Epoch 6 Loss: 0.2962
Epoch 7 Loss: 0.2644
Epoch 8 Loss: 0.2402
Epoch 9 Loss: 0.2127
Epoch 10 Loss: 0.1973
Eval vector_dim=100...

============================================================
  RESULT | lr=0.0001  batch=64  epochs=10  emb_dim=100
  ACCURACY: 0.8224  |  TIME: 813.48s
============================================================

>>> lr=2e-4 batch=64 epochs=10 emb_dim=100
2
[tensor([[ 101, 1045, 2387,  ...,    0,    0,    0],
        [ 101, 1012, 1012,  ...,    0,    0,    0],
        [ 101, 1996, 6517,  ...,    0,    0,    0],
        ...,
        [ 101, 1037, 2186,  ...,    0,    0,    0],
        [ 101, 2045, 2020,  ...,    0,    0,    0],
        [ 101, 2200, 2919,  ..., 1010, 1045,  102]]), tensor([[1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        ...,
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 1, 1, 1]]), tensor([1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
        0, 1, 1, 0, 1, 1, 0, 0])]
/mnt/c/Users/Lighte/GitHub/AIBLAB/Transformer/nlp_code.py:87: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
test model before training
tensor([[ 0.0197,  0.3347,  0.7764,  ..., -0.3173,  0.2156, -0.7108],
        [ 0.0734, -0.2693,  0.7654,  ...,  0.2551,  0.2323, -0.7260],
        [-0.0897, -0.4094,  0.0450,  ..., -0.1745,  0.5349, -0.7730],
        ...,
        [-0.1908, -0.0888, -0.0530,  ..., -0.5008,  0.2596, -0.8527],
        [-0.1037, -0.2700,  0.6089,  ..., -0.1552,  0.5516, -0.8663],
        [-0.0030, -0.1475,  0.0992,  ..., -0.4651,  0.5122, -0.6253]],
       device='cuda:0', grad_fn=<TanhBackward0>)
tensor([[-0.2383, -0.5371],
        [ 0.2698, -0.2612],
        [-0.2437, -0.7461],
        [-0.4726, -0.7805],
        [-0.1178, -0.4262],
        [-0.2707, -0.3356],
        [-0.2830,  0.0319],
        [-0.2547, -0.6320],
        [-0.3146, -0.4299],
        [-0.0638, -0.5087],
        [-0.1470, -0.5934],
        [-0.0234, -0.4612],
        [-0.4045, -0.2679],
        [-0.2852, -0.8495],
        [-0.3825, -0.7389],
        [ 0.1242, -0.1092],
        [-0.1836, -0.5240],
        [-0.1743, -0.5497],
        [-0.0701, -0.3443],
        [-0.0719, -0.0957],
        [-0.2365, -0.3229],
        [-0.0211, -0.2717],
        [ 0.0323, -0.4655],
        [-0.2881, -0.5329],
        [-0.0091, -0.2646],
        [-0.1767, -0.5053],
        [-0.2024, -0.4831],
        [-0.1671, -0.6250],
        [-0.1970, -0.7765],
        [-0.4290, -0.6894],
        [-0.0655, -0.3074],
        [-0.0755, -0.6299]], device='cuda:0', grad_fn=<AddmmBackward0>)
/mnt/c/Users/Lighte/GitHub/AIBLAB/Transformer/nlp_code.py:87: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
Training vector_dim=100...
Epoch 1 Loss: 0.6906
Epoch 2 Loss: 0.5389
Epoch 3 Loss: 0.4279
Epoch 4 Loss: 0.3738
Epoch 5 Loss: 0.3205
Epoch 6 Loss: 0.2769
Epoch 7 Loss: 0.2343
Epoch 8 Loss: 0.2010
Epoch 9 Loss: 0.1670
Epoch 10 Loss: 0.1433
Eval vector_dim=100...

============================================================
  RESULT | lr=0.0002  batch=64  epochs=10  emb_dim=100
  ACCURACY: 0.8079  |  TIME: 809.31s
============================================================


==========================================
Experiment End: Wed May 27 14:48:10 CST 2026
==========================================