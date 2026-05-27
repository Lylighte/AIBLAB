(transformer) lighte@LIGHTE-5900:/mnt/c/Users/Lighte/GitHub/AIBLAB/Transformer$ ./run_experiments.sh 
==========================================
Experiment Start: Wed May 27 11:19:50 CST 2026
==========================================

========== 1. 学习率对比 ==========
>>> lr=1e-4 batch=64 epochs=10 emb_dim=100
2
[tensor([[  101,  1037,  2843,  ...,     0,     0,     0],
        [  101, 27594,  2545,  ...,     0,     0,     0],
        [  101,  2009,  1005,  ...,     0,     0,     0],
        ...,
        [  101,  1996,  3185,  ...,     0,     0,     0],
        [  101, 21425,  1010,  ...,     0,     0,     0],
        [  101,  1045,  4669,  ...,     0,     0,     0]]), tensor([[1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        ...,
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0]]), tensor([0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0,
        0, 1, 0, 0, 1, 0, 0, 1])]
/mnt/c/Users/Lighte/GitHub/AIBLAB/Transformer/nlp_code.py:87: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
test model before training
tensor([[ 0.7428,  0.3828, -0.3580,  ...,  0.5183, -0.7223,  0.0754],
        [ 0.3706,  0.5624,  0.4280,  ..., -0.1251, -0.2174,  0.2856],
        [-0.3106,  0.7190, -0.1401,  ..., -0.1252, -0.6064, -0.2290],
        ...,
        [-0.4666,  0.9542, -0.1633,  ..., -0.1913, -0.6179, -0.3543],
        [ 0.1401,  0.5140,  0.0545,  ..., -0.2770, -0.6676, -0.0820],
        [ 0.4558,  0.8499, -0.5220,  ...,  0.1020, -0.5230,  0.0396]],
       device='cuda:0', grad_fn=<TanhBackward0>)
tensor([[ 0.3779, -0.1644],
        [ 0.7728, -0.0923],
        [ 0.2679,  0.2864],
        [ 0.4785, -0.2213],
        [ 0.5199,  0.1308],
        [ 0.7692, -0.1373],
        [ 0.3792,  0.0873],
        [ 0.3567,  0.1001],
        [ 0.3260,  0.0695],
        [ 0.6319, -0.0580],
        [ 0.2803,  0.3493],
        [ 0.4821, -0.0979],
        [ 0.6027,  0.1216],
        [ 0.3912, -0.0157],
        [ 0.8250, -0.1816],
        [ 0.4795,  0.2789],
        [ 0.5406, -0.1808],
        [ 0.7074, -0.0054],
        [ 0.6600,  0.0490],
        [ 0.2963, -0.0358],
        [ 0.5400,  0.4013],
        [ 0.6532,  0.0427],
        [ 0.5476,  0.1289],
        [ 0.5106, -0.1374],
        [ 0.5492, -0.0039],
        [ 0.2034,  0.1754],
        [ 0.6916, -0.0310],
        [ 0.3290, -0.0073],
        [ 0.7439, -0.0634],
        [ 0.3456,  0.0928],
        [ 0.5995, -0.1126],
        [ 0.5305, -0.1288]], device='cuda:0', grad_fn=<AddmmBackward0>)
/mnt/c/Users/Lighte/GitHub/AIBLAB/Transformer/nlp_code.py:87: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
Training vector_dim=100...
Epoch 1 Loss: 0.6417
Epoch 2 Loss: 0.4786
Epoch 3 Loss: 0.3963
Epoch 4 Loss: 0.3495
Epoch 5 Loss: 0.3037
Epoch 6 Loss: 0.2761
Epoch 7 Loss: 0.2470
Epoch 8 Loss: 0.2215
Epoch 9 Loss: 0.1983
Epoch 10 Loss: 0.1820
Eval vector_dim=100...

============================================================
  RESULT | lr=0.0001  batch=64  epochs=10  emb_dim=100
  ACCURACY: 0.8240  |  TIME: 0.00s
============================================================

>>> lr=5e-4 batch=64 epochs=10 emb_dim=100
2
[tensor([[ 101, 1996, 3114,  ...,    0,    0,    0],
        [ 101, 1996, 2273,  ..., 7432, 5307,  102],
        [ 101, 2023, 3185,  ..., 2002, 2097,  102],
        ...,
        [ 101, 2045, 2003,  ...,    0,    0,    0],
        [ 101, 7110, 1005,  ...,    0,    0,    0],
        [ 101, 1026, 7987,  ...,    0,    0,    0]]), tensor([[1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 1, 1, 1],
        [1, 1, 1,  ..., 1, 1, 1],
        ...,
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0]]), tensor([0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0,
        0, 1, 0, 1, 1, 0, 0, 1])]
/mnt/c/Users/Lighte/GitHub/AIBLAB/Transformer/nlp_code.py:87: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
test model before training
tensor([[ 0.6138, -0.6936, -0.4446,  ..., -0.6338, -0.0506, -0.8958],
        [ 0.6013, -0.5783, -0.0552,  ..., -0.7885, -0.1690, -0.4787],
        [ 0.1517, -0.0176, -0.6327,  ..., -0.6443,  0.1002, -0.9200],
        ...,
        [ 0.5302, -0.0927, -0.0124,  ..., -0.5273, -0.4496, -0.5880],
        [ 0.7487, -0.5705, -0.2982,  ..., -0.7415,  0.4351, -0.8721],
        [ 0.5136,  0.1605, -0.7237,  ..., -0.7928, -0.1616, -0.2730]],
       device='cuda:0', grad_fn=<TanhBackward0>)
tensor([[-0.0482, -0.1516],
        [-0.0491, -0.1071],
        [-0.3028, -0.2332],
        [-0.0975, -0.3223],
        [-0.0580, -0.3138],
        [-0.1778, -0.3693],
        [-0.0816, -0.3071],
        [-0.0284, -0.3593],
        [ 0.1112, -0.1162],
        [-0.1259, -0.3130],
        [-0.0576, -0.4419],
        [-0.5540, -0.3245],
        [ 0.2407, -0.1092],
        [-0.2691, -0.4701],
        [ 0.3473, -0.1609],
        [ 0.2731, -0.2787],
        [-0.0776, -0.2955],
        [-0.0805, -0.2037],
        [ 0.0157, -0.3667],
        [-0.2935, -0.2026],
        [-0.1398, -0.0438],
        [-0.0765, -0.4465],
        [-0.1629, -0.2731],
        [ 0.1703, -0.0876],
        [-0.2182, -0.2088],
        [-0.0300, -0.2020],
        [ 0.0273, -0.2909],
        [-0.0278, -0.4626],
        [ 0.0104, -0.1036],
        [ 0.1092, -0.0346],
        [-0.0785, -0.3080],
        [ 0.0440, -0.0082]], device='cuda:0', grad_fn=<AddmmBackward0>)
/mnt/c/Users/Lighte/GitHub/AIBLAB/Transformer/nlp_code.py:87: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
Training vector_dim=100...
Epoch 1 Loss: 0.6968
Epoch 2 Loss: 0.6944
Epoch 3 Loss: 0.6939
Epoch 4 Loss: 0.6939
Epoch 5 Loss: 0.6939
Epoch 6 Loss: 0.6938
Epoch 7 Loss: 0.6938
Epoch 8 Loss: 0.6935
Epoch 9 Loss: 0.6934
Epoch 10 Loss: 0.6932
Eval vector_dim=100...

============================================================
  RESULT | lr=0.0005  batch=64  epochs=10  emb_dim=100
  ACCURACY: 0.6406  |  TIME: 0.00s
============================================================

>>> lr=1e-3 batch=64 epochs=10 emb_dim=100
^CTraceback (most recent call last):
  File "/mnt/c/Users/Lighte/GitHub/AIBLAB/Transformer/nlp_code.py", line 118, in <module>
    train_loader, test_loader, num_classes, vocab_size = prepare_data(BATCH_SIZE)
                                                         ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/c/Users/Lighte/GitHub/AIBLAB/Transformer/nlp_code.py", line 50, in prepare_data
    train_encodings = tokenizer(train_list, padding=True, max_length=512,
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lighte/miniconda3/envs/transformer/lib/python3.12/site-packages/transformers/tokenization_utils_base.py", line 2513, in __call__
    encodings = self._encode_plus(
                ^^^^^^^^^^^^^^^^^^
  File "/home/lighte/miniconda3/envs/transformer/lib/python3.12/site-packages/transformers/tokenization_utils_tokenizers.py", line 967, in _encode_plus
    self._convert_encoding(
KeyboardInterrupt

(transformer) lighte@LIGHTE-5900:/mnt/c/Users/Lighte/GitHub/AIBLAB/Transformer$ ./run_experiments.sh 
==========================================
Experiment Start: Wed May 27 12:04:49 CST 2026
==========================================

========== 1. 学习率对比 ==========
>>> lr=1e-3 batch=64 epochs=10 emb_dim=100
2
[tensor([[  101,  1045,  2001,  ...,     0,     0,     0],
        [  101,  1996,  2332,  ...,     0,     0,     0],
        [  101,  1057,  5603,  ...,     0,     0,     0],
        ...,
        [  101,  2053,  3185,  ...,     0,     0,     0],
        [  101,  2044,  2195,  ...,     0,     0,     0],
        [  101,  3492, 15640,  ...,     0,     0,     0]]), tensor([[1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        ...,
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0]]), tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1,
        0, 0, 0, 0, 0, 0, 1, 0])]
/mnt/c/Users/Lighte/GitHub/AIBLAB/Transformer/nlp_code.py:87: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
test model before training
tensor([[ 0.8440, -0.8088,  0.0971,  ..., -0.8588,  0.9590, -0.3280],
        [ 0.4735, -0.6492,  0.3453,  ..., -0.8630,  0.9049, -0.0922],
        [ 0.8950, -0.5588, -0.1842,  ..., -0.7987,  0.9380, -0.4977],
        ...,
        [ 0.8615, -0.9031,  0.3802,  ..., -0.6926,  0.8045, -0.6856],
        [ 0.8392, -0.9447, -0.2074,  ..., -0.5707,  0.5331, -0.5292],
        [ 0.8946, -0.6924,  0.6045,  ..., -0.6331,  0.5881, -0.7691]],
       device='cuda:0', grad_fn=<TanhBackward0>)
tensor([[-0.4609, -0.1406],
        [-0.3124,  0.1128],
        [-0.4342,  0.1604],
        [-0.3209,  0.2015],
        [-0.4066,  0.2429],
        [-0.3936,  0.1280],
        [-0.5255,  0.0760],
        [-0.4573,  0.2160],
        [-0.4212,  0.0718],
        [-0.3282,  0.2083],
        [-0.3106, -0.0051],
        [-0.3840,  0.1592],
        [-0.3697,  0.0278],
        [-0.3151,  0.2479],
        [-0.3569,  0.1012],
        [-0.3624,  0.1577],
        [-0.2511,  0.1494],
        [-0.4480, -0.0273],
        [-0.4169,  0.0668],
        [-0.1747,  0.2318],
        [-0.2797,  0.2590],
        [-0.1854,  0.1100],
        [-0.5567, -0.0469],
        [-0.3654,  0.2322],
        [-0.3693,  0.0416],
        [-0.3576,  0.1389],
        [-0.1634,  0.1897],
        [-0.4148, -0.1303],
        [-0.5586, -0.1971],
        [-0.4557,  0.1301],
        [-0.3700,  0.1088],
        [-0.3731,  0.2025]], device='cuda:0', grad_fn=<AddmmBackward0>)
/mnt/c/Users/Lighte/GitHub/AIBLAB/Transformer/nlp_code.py:87: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
Training vector_dim=100...
Epoch 1 Loss: 0.6986
Epoch 2 Loss: 0.6949
Epoch 3 Loss: 0.6952
Epoch 4 Loss: 0.6943
Epoch 5 Loss: 0.6941
Epoch 6 Loss: 0.6938
Epoch 7 Loss: 0.6936
Epoch 8 Loss: 0.6937
Epoch 9 Loss: 0.6935
Epoch 10 Loss: 0.6933
Eval vector_dim=100...

============================================================
  RESULT | lr=0.001  batch=64  epochs=10  emb_dim=100
  ACCURACY: 0.6261  |  TIME: 991.14s
============================================================


========== 2. 批次大小对比 ==========
>>> lr=5e-4 batch=32 epochs=10 emb_dim=100
2
[tensor([[  101,  7619, 13044,  ...,     0,     0,     0],
        [  101,  2023,  2058,  ...,  2041,  1997,   102],
        [  101,  2009,  1005,  ...,     0,     0,     0],
        ...,
        [  101,  2054,  4627,  ...,     0,     0,     0],
        [  101,  1045,  1998,  ...,     0,     0,     0],
        [  101,  1037,  2442,  ...,     0,     0,     0]]), tensor([[1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 1, 1, 1],
        [1, 1, 1,  ..., 0, 0, 0],
        ...,
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0]]), tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1,
        0, 0, 1, 1, 0, 1, 0, 1])]
/mnt/c/Users/Lighte/GitHub/AIBLAB/Transformer/nlp_code.py:87: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
test model before training
tensor([[-0.5065,  0.8421,  0.1466,  ..., -0.8937, -0.3896, -0.8853],
        [-0.4928,  0.9265,  0.2264,  ..., -0.1259, -0.6462, -0.3532],
        [-0.6304,  0.4964, -0.7372,  ..., -0.8229, -0.3438, -0.3841],
        ...,
        [-0.4803, -0.6947, -0.4274,  ..., -0.6756, -0.7189, -0.4626],
        [-0.2548,  0.7128,  0.3816,  ..., -0.6637,  0.0286, -0.9147],
        [-0.5612,  0.1830,  0.0314,  ...,  0.0211, -0.3757, -0.7448]],
       device='cuda:0', grad_fn=<TanhBackward0>)
tensor([[ 0.5402,  0.3441],
        [ 0.5088,  0.1554],
        [ 0.6476,  0.0252],
        [ 0.3972, -0.0292],
        [ 0.1364,  0.2789],
        [ 0.9026,  0.0014],
        [ 0.3586,  0.3154],
        [ 0.5689,  0.1949],
        [ 0.6898,  0.1178],
        [ 0.3661,  0.0446],
        [ 0.3249,  0.3505],
        [ 0.0528,  0.4882],
        [ 0.5430,  0.2591],
        [ 0.6507, -0.0270],
        [ 0.7458,  0.0241],
        [ 0.6632,  0.2081],
        [ 0.4739,  0.0424],
        [ 0.4758,  0.0384],
        [ 0.4516,  0.0519],
        [ 0.6484,  0.0807],
        [ 0.4589,  0.2668],
        [ 0.4266,  0.3655],
        [ 0.3151,  0.4149],
        [ 0.5504,  0.1353],
        [ 0.4740,  0.0639],
        [ 0.7432,  0.1935],
        [ 0.5185,  0.1306],
        [ 0.4287,  0.2665],
        [ 0.7960, -0.0410],
        [ 0.5720,  0.2062],
        [ 0.4778,  0.3805],
        [ 0.5187, -0.0286]], device='cuda:0', grad_fn=<AddmmBackward0>)
Training vector_dim=100...
Epoch 1 Loss: 0.6973
Epoch 2 Loss: 0.6952
Epoch 3 Loss: 0.6945
Epoch 4 Loss: 0.6942
Epoch 5 Loss: 0.6941
Epoch 6 Loss: 0.6941
Epoch 7 Loss: 0.6936
Epoch 8 Loss: 0.6936
Epoch 9 Loss: 0.6936
Epoch 10 Loss: 0.6933
Eval vector_dim=100...

============================================================
  RESULT | lr=0.0005  batch=32  epochs=10  emb_dim=100
  ACCURACY: 0.6306  |  TIME: 964.09s
============================================================

>>> lr=5e-4 batch=64 epochs=10 emb_dim=100
2
[tensor([[ 101, 2043, 1045,  ...,    0,    0,    0],
        [ 101, 2023, 3185,  ...,    0,    0,    0],
        [ 101, 2096, 3666,  ...,    0,    0,    0],
        ...,
        [ 101, 2402, 1998,  ...,    0,    0,    0],
        [ 101, 2023, 3185,  ...,    0,    0,    0],
        [ 101, 2054, 2064,  ...,    0,    0,    0]]), tensor([[1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        ...,
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0]]), tensor([0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
        0, 0, 0, 1, 0, 0, 1, 0])]
/mnt/c/Users/Lighte/GitHub/AIBLAB/Transformer/nlp_code.py:87: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
test model before training
tensor([[ 0.7595, -0.4825, -0.6612,  ..., -0.0582,  0.6834, -0.0630],
        [ 0.9177, -0.3638, -0.9540,  ..., -0.4334,  0.9194,  0.7326],
        [ 0.9683, -0.8090, -0.7258,  ...,  0.2502,  0.4670,  0.5426],
        ...,
        [ 0.9259, -0.2628, -0.6902,  ...,  0.0985,  0.5573,  0.6751],
        [ 0.9049, -0.5702, -0.8623,  ...,  0.1824,  0.7042,  0.4760],
        [ 0.5411, -0.0832, -0.9402,  ..., -0.1705,  0.9686,  0.1136]],
       device='cuda:0', grad_fn=<TanhBackward0>)
tensor([[-0.2287, -0.0799],
        [-0.1496, -0.0487],
        [-0.0883,  0.3996],
        [-0.0592, -0.1317],
        [-0.0326,  0.1386],
        [-0.0835,  0.0820],
        [-0.0727, -0.2597],
        [-0.0598,  0.3613],
        [-0.2491,  0.0849],
        [-0.0090, -0.2823],
        [ 0.1501, -0.2095],
        [-0.1333, -0.0576],
        [-0.3183,  0.1189],
        [-0.0142,  0.0112],
        [-0.0580, -0.0341],
        [ 0.1937,  0.0390],
        [ 0.0599, -0.3075],
        [ 0.0284,  0.2762],
        [ 0.0387, -0.0696],
        [ 0.1407,  0.1688],
        [ 0.2692,  0.2535],
        [-0.2339, -0.0912],
        [-0.0380,  0.0836],
        [-0.2531, -0.0913],
        [-0.2151, -0.4395],
        [ 0.2548,  0.1962],
        [-0.1169,  0.3011],
        [-0.1231,  0.3910],
        [ 0.1181,  0.0497],
        [-0.0464,  0.1916],
        [ 0.0661,  0.0625],
        [-0.1749,  0.0543]], device='cuda:0', grad_fn=<AddmmBackward0>)
/mnt/c/Users/Lighte/GitHub/AIBLAB/Transformer/nlp_code.py:87: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
Training vector_dim=100...
Epoch 1 Loss: 0.6972
Epoch 2 Loss: 0.6947
Epoch 3 Loss: 0.6945
Epoch 4 Loss: 0.6944
Epoch 5 Loss: 0.6940
Epoch 6 Loss: 0.6914
Epoch 7 Loss: 0.6938
Epoch 8 Loss: 0.6935
Epoch 9 Loss: 0.6935
Epoch 10 Loss: 0.6933
Eval vector_dim=100...

============================================================
  RESULT | lr=0.0005  batch=64  epochs=10  emb_dim=100
  ACCURACY: 0.6576  |  TIME: 965.08s
============================================================

>>> lr=5e-4 batch=128 epochs=10 emb_dim=100
2
[tensor([[  101,  1037,  2613,  ...,     0,     0,     0],
        [  101,  3100,  1012,  ...,  1998,  4012,   102],
        [  101,  7418, 12170,  ...,     0,     0,     0],
        ...,
        [  101,  2096,  3149,  ...,     0,     0,     0],
        [  101,  2023, 12661,  ...,     0,     0,     0],
        [  101,  1045,  2196,  ...,  2016,  1005,   102]]), tensor([[1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 1, 1, 1],
        [1, 1, 1,  ..., 0, 0, 0],
        ...,
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 1, 1, 1]]), tensor([0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0,
        1, 0, 1, 0, 1, 0, 0, 0])]
/mnt/c/Users/Lighte/GitHub/AIBLAB/Transformer/nlp_code.py:87: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
test model before training
tensor([[ 0.1243,  0.5897,  0.9268,  ..., -0.5618, -0.6871, -0.3647],
        [-0.0799,  0.7347,  0.9738,  ..., -0.5189, -0.3582, -0.0638],
        [ 0.4801,  0.1760,  0.8594,  ..., -0.3448,  0.3385,  0.0494],
        ...,
        [ 0.5190,  0.4582,  0.8024,  ..., -0.8063, -0.1508, -0.5738],
        [ 0.0360,  0.1994,  0.9149,  ..., -0.5253, -0.2863, -0.1502],
        [ 0.3761,  0.7330,  0.9187,  ..., -0.4390,  0.1167, -0.5028]],
       device='cuda:0', grad_fn=<TanhBackward0>)
tensor([[-0.7712,  0.0937],
        [-0.7806,  0.2868],
        [-0.3896,  0.2638],
        [-0.7107,  0.3265],
        [-0.5214,  0.1009],
        [-0.5893,  0.3114],
        [-0.6885,  0.6189],
        [-0.7159,  0.3464],
        [-0.8500,  0.1258],
        [-0.7104,  0.2907],
        [-0.4200,  0.1709],
        [-0.7276,  0.4930],
        [-0.7205,  0.2150],
        [-0.6380,  0.2447],
        [-0.6042,  0.0970],
        [-0.6546,  0.0528],
        [-0.4477,  0.5690],
        [-0.6751,  0.3143],
        [-0.6886,  0.3318],
        [-0.5594,  0.0813],
        [-0.7122,  0.3731],
        [-0.7118,  0.2035],
        [-0.5272,  0.2473],
        [-0.7922,  0.2597],
        [-0.6969,  0.5431],
        [-0.6052,  0.3087],
        [-0.8608,  0.1510],
        [-0.7995,  0.1897],
        [-0.6160,  0.2526],
        [-0.7266,  0.0723],
        [-0.5316,  0.4426],
        [-0.8023,  0.4325]], device='cuda:0', grad_fn=<AddmmBackward0>)
/mnt/c/Users/Lighte/GitHub/AIBLAB/Transformer/nlp_code.py:87: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
Training vector_dim=100...
^CTraceback (most recent call last):
  File "/mnt/c/Users/Lighte/GitHub/AIBLAB/Transformer/nlp_code.py", line 245, in <module>
    acc = eval_model(EMB_DIM)
          ^^^^^^^^^^^^^^^^^^^
  File "/mnt/c/Users/Lighte/GitHub/AIBLAB/Transformer/nlp_code.py", line 182, in eval_model
    model = train_model(vector_dim)
            ^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/c/Users/Lighte/GitHub/AIBLAB/Transformer/nlp_code.py", line 170, in train_model
    total_loss += loss.item()
                  ^^^^^^^^^^^
KeyboardInterrupt

(transformer) lighte@LIGHTE-5900:/mnt/c/Users/Lighte/GitHub/AIBLAB/Transformer$ ./run_experiments.sh 
==========================================
Experiment Start: Wed May 27 13:21:33 CST 2026
==========================================

========== 1. 学习率对比 ==========

========== 2. 批次大小对比 ==========
>>> lr=5e-4 batch=64 epochs=10 emb_dim=100
2
[tensor([[ 101, 2023, 3185,  ...,    0,    0,    0],
        [ 101, 1045, 2106,  ...,    0,    0,    0],
        [ 101, 1045, 2031,  ...,    0,    0,    0],
        ...,
        [ 101, 7918, 2038,  ...,    0,    0,    0],
        [ 101, 1037, 3232,  ...,    0,    0,    0],
        [ 101, 7244, 1998,  ...,    0,    0,    0]]), tensor([[1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        ...,
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0]]), tensor([1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1,
        1, 0, 0, 1, 0, 0, 0, 1])]
/mnt/c/Users/Lighte/GitHub/AIBLAB/Transformer/nlp_code.py:87: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
test model before training
tensor([[ 0.4323,  0.4178,  0.8127,  ..., -0.6774, -0.1324, -0.0567],
        [ 0.0109,  0.2032,  0.6898,  ..., -0.7175, -0.5217, -0.7326],
        [ 0.2367,  0.8077,  0.5622,  ..., -0.6810, -0.4000, -0.4133],
        ...,
        [ 0.4344,  0.8039,  0.6792,  ..., -0.8403, -0.1497, -0.5354],
        [ 0.1170,  0.3623,  0.5435,  ..., -0.6980, -0.5218, -0.4820],
        [ 0.2623,  0.7684,  0.6082,  ..., -0.7299, -0.5998, -0.4408]],
       device='cuda:0', grad_fn=<TanhBackward0>)
tensor([[ 0.5423, -0.3867],
        [ 0.2891, -0.0485],
        [ 0.1028, -0.2536],
        [ 0.3729, -0.2654],
        [ 0.3760, -0.3981],
        [ 0.1236,  0.0098],
        [ 0.2921, -0.3290],
        [ 0.0791,  0.0966],
        [ 0.3033, -0.2158],
        [ 0.2330, -0.2709],
        [ 0.3707, -0.0542],
        [-0.0377, -0.2290],
        [ 0.1227, -0.0615],
        [ 0.4301, -0.0366],
        [ 0.4903, -0.1718],
        [ 0.1509, -0.1617],
        [ 0.3027, -0.0729],
        [ 0.4436, -0.2702],
        [ 0.2767, -0.3081],
        [ 0.1950, -0.0413],
        [ 0.2937, -0.2527],
        [-0.0083, -0.0972],
        [ 0.1897, -0.1674],
        [ 0.2754, -0.1430],
        [-0.0583, -0.1101],
        [ 0.2604, -0.2223],
        [ 0.4272, -0.0276],
        [ 0.5484, -0.1477],
        [ 0.2115, -0.0596],
        [ 0.0900, -0.3255],
        [ 0.1040, -0.4004],
        [ 0.2434, -0.2008]], device='cuda:0', grad_fn=<AddmmBackward0>)
/mnt/c/Users/Lighte/GitHub/AIBLAB/Transformer/nlp_code.py:87: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
Training vector_dim=100...
Epoch 1 Loss: 0.6979
Epoch 2 Loss: 0.6945
Epoch 3 Loss: 0.6947
Epoch 4 Loss: 0.6945
Epoch 5 Loss: 0.6940
Epoch 6 Loss: 0.6938
Epoch 7 Loss: 0.6935
Epoch 8 Loss: 0.6935
Epoch 9 Loss: 0.6934
Epoch 10 Loss: 0.6933
Eval vector_dim=100...

============================================================
  RESULT | lr=0.0005  batch=64  epochs=10  emb_dim=100
  ACCURACY: 0.6257  |  TIME: 795.51s
============================================================


========== 3. 训练轮数对比 ==========
>>> lr=5e-4 batch=64 epochs=5 emb_dim=100
2
[tensor([[  101,  2893,  2000,  ...,  2047,  2088,   102],
        [  101,  2482, 15460,  ...,     0,     0,     0],
        [  101,  1037, 12383,  ...,     0,     0,     0],
        ...,
        [  101,  2009,  2001,  ...,     0,     0,     0],
        [  101,  2045,  2031,  ...,  2191,  4515,   102],
        [  101,  2023,  6579,  ...,     0,     0,     0]]), tensor([[1, 1, 1,  ..., 1, 1, 1],
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        ...,
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 1, 1, 1],
        [1, 1, 1,  ..., 0, 0, 0]]), tensor([1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1,
        1, 1, 0, 0, 1, 0, 0, 1])]
/mnt/c/Users/Lighte/GitHub/AIBLAB/Transformer/nlp_code.py:87: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
test model before training
tensor([[-0.5237,  0.9825, -0.5566,  ..., -0.9766,  0.7766,  0.7682],
        [-0.2900,  0.9700, -0.0490,  ..., -0.9680,  0.6531,  0.1094],
        [-0.7737,  0.9874, -0.4828,  ..., -0.9449,  0.7388,  0.5857],
        ...,
        [-0.7464,  0.9729, -0.4449,  ..., -0.9937,  0.4420,  0.6152],
        [-0.3177,  0.9585, -0.2195,  ..., -0.9808,  0.6246,  0.3033],
        [-0.8307,  0.9826,  0.3584,  ..., -0.9816,  0.3380,  0.8793]],
       device='cuda:0', grad_fn=<TanhBackward0>)
tensor([[-0.1427, -0.1226],
        [-0.1470, -0.1102],
        [-0.1372, -0.1998],
        [ 0.0068, -0.3358],
        [-0.1986, -0.3069],
        [-0.0611, -0.1047],
        [ 0.0338, -0.3852],
        [-0.2095, -0.3172],
        [-0.3297, -0.2721],
        [ 0.2058, -0.2662],
        [-0.2259, -0.3021],
        [-0.2668, -0.3480],
        [-0.2933, -0.3143],
        [-0.3302, -0.4712],
        [-0.3968, -0.2057],
        [-0.5082, -0.4560],
        [-0.3641, -0.2560],
        [-0.4543, -0.3023],
        [-0.2400, -0.1237],
        [ 0.0577, -0.2323],
        [-0.2279, -0.0620],
        [-0.4933, -0.2856],
        [-0.3092, -0.4041],
        [-0.3464,  0.0656],
        [-0.1618, -0.1037],
        [-0.2381, -0.3814],
        [-0.1560, -0.1199],
        [-0.0650, -0.3282],
        [-0.4989, -0.3605],
        [-0.2821, -0.1360],
        [-0.5358, -0.4052],
        [-0.0886, -0.2056]], device='cuda:0', grad_fn=<AddmmBackward0>)
/mnt/c/Users/Lighte/GitHub/AIBLAB/Transformer/nlp_code.py:87: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
Training vector_dim=100...
Epoch 1 Loss: 0.6980
Epoch 2 Loss: 0.6944
Epoch 3 Loss: 0.6942
Epoch 4 Loss: 0.6937
Epoch 5 Loss: 0.6934
Eval vector_dim=100...

============================================================
  RESULT | lr=0.0005  batch=64  epochs=5  emb_dim=100
  ACCURACY: 0.6198  |  TIME: 418.47s
============================================================

>>> lr=5e-4 batch=64 epochs=10 emb_dim=100
2
[tensor([[ 101, 1996, 2034,  ...,    0,    0,    0],
        [ 101, 2941, 1010,  ...,    0,    0,    0],
        [ 101, 2092, 1010,  ...,    0,    0,    0],
        ...,
        [ 101, 2387, 2023,  ...,    0,    0,    0],
        [ 101, 2023, 2089,  ...,    0,    0,    0],
        [ 101, 1045, 3342,  ...,    0,    0,    0]]), tensor([[1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        ...,
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0]]), tensor([0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1,
        1, 1, 0, 0, 1, 0, 0, 1])]
/mnt/c/Users/Lighte/GitHub/AIBLAB/Transformer/nlp_code.py:87: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
test model before training
tensor([[-0.5161, -0.7268, -0.6536,  ...,  0.2909,  0.8218,  0.8762],
        [ 0.1085, -0.7481, -0.0199,  ..., -0.0718,  0.5185,  0.5018],
        [ 0.5656, -0.4139, -0.2055,  ...,  0.5469,  0.6386,  0.9284],
        ...,
        [-0.4702, -0.7953, -0.6996,  ..., -0.2239,  0.8323,  0.7103],
        [ 0.0982, -0.7603,  0.2451,  ...,  0.1371,  0.7514,  0.8943],
        [ 0.0993, -0.5517,  0.1810,  ...,  0.4579,  0.2764,  0.8841]],
       device='cuda:0', grad_fn=<TanhBackward0>)
tensor([[-0.4854,  0.1453],
        [-0.2134, -0.0108],
        [-0.1768,  0.4440],
        [-0.1128, -0.0196],
        [-0.4399,  0.5267],
        [-0.2534,  0.5574],
        [-0.2557,  0.4780],
        [-0.0384, -0.0632],
        [-0.0433,  0.4967],
        [-0.4819,  0.2750],
        [-0.2996,  0.3220],
        [ 0.0223,  0.0227],
        [-0.3353,  0.5138],
        [-0.1969,  0.0458],
        [-0.1494, -0.0485],
        [-0.0459,  0.0998],
        [-0.4203,  0.2733],
        [-0.2224,  0.1395],
        [-0.4756,  0.0968],
        [-0.1879,  0.3565],
        [-0.0530,  0.2009],
        [-0.2121,  0.1240],
        [-0.0734,  0.5213],
        [-0.2800,  0.5170],
        [-0.0087,  0.1902],
        [-0.0169,  0.2748],
        [-0.3402, -0.0329],
        [-0.4463,  0.2410],
        [-0.3552,  0.1130],
        [-0.3316,  0.3848],
        [-0.1278,  0.1970],
        [-0.2568,  0.0485]], device='cuda:0', grad_fn=<AddmmBackward0>)
/mnt/c/Users/Lighte/GitHub/AIBLAB/Transformer/nlp_code.py:87: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
Training vector_dim=100...
Epoch 1 Loss: 0.6981
Epoch 2 Loss: 0.6943
Epoch 3 Loss: 0.6944
Epoch 4 Loss: 0.6942
Epoch 5 Loss: 0.6938
Epoch 6 Loss: 0.6937
Epoch 7 Loss: 0.6936
Epoch 8 Loss: 0.6934
Epoch 9 Loss: 0.6934
Epoch 10 Loss: 0.6933
Eval vector_dim=100...

============================================================
  RESULT | lr=0.0005  batch=64  epochs=10  emb_dim=100
  ACCURACY: 0.6338  |  TIME: 782.16s
============================================================

>>> lr=5e-4 batch=64 epochs=20 emb_dim=100
2
[tensor([[ 101, 1045, 2572,  ...,    0,    0,    0],
        [ 101, 5432, 1024,  ..., 2052, 2272,  102],
        [ 101, 2205, 2919,  ...,    0,    0,    0],
        ...,
        [ 101, 2070, 3152,  ...,    0,    0,    0],
        [ 101, 5934, 1998,  ...,    0,    0,    0],
        [ 101, 2043, 1045,  ...,    0,    0,    0]]), tensor([[1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 1, 1, 1],
        [1, 1, 1,  ..., 0, 0, 0],
        ...,
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0]]), tensor([0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0,
        1, 0, 1, 0, 0, 1, 1, 1])]
/mnt/c/Users/Lighte/GitHub/AIBLAB/Transformer/nlp_code.py:87: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
test model before training
tensor([[ 0.4770, -0.4231, -0.8554,  ...,  0.8560,  0.8070,  0.0617],
        [ 0.8157, -0.9075, -0.8936,  ...,  0.7758,  0.9657,  0.1937],
        [ 0.7812, -0.2119, -0.7797,  ...,  0.8782,  0.9026,  0.3213],
        ...,
        [ 0.4667,  0.3236, -0.8420,  ...,  0.9041,  0.9053,  0.2752],
        [ 0.4565, -0.5749, -0.6622,  ...,  0.5961,  0.8671,  0.5374],
        [ 0.3803,  0.0460, -0.6773,  ...,  0.9420,  0.9600,  0.3900]],
       device='cuda:0', grad_fn=<TanhBackward0>)
tensor([[ 0.2946,  0.2606],
        [ 0.4964,  0.3657],
        [ 0.2868,  0.3674],
        [ 0.2491,  0.3180],
        [ 0.4976,  0.3575],
        [ 0.2604,  0.1677],
        [ 0.3394,  0.2299],
        [ 0.4158, -0.0082],
        [ 0.3773,  0.3270],
        [ 0.4716, -0.0122],
        [ 0.2837,  0.0858],
        [ 0.3483,  0.1109],
        [ 0.2731,  0.1874],
        [ 0.3057,  0.4305],
        [ 0.4587,  0.3430],
        [ 0.1132,  0.0794],
        [ 0.4433,  0.2371],
        [ 0.3657,  0.2770],
        [ 0.3113,  0.2842],
        [ 0.4835,  0.0668],
        [ 0.2602,  0.0685],
        [ 0.2788,  0.0778],
        [ 0.2019,  0.3769],
        [ 0.6250,  0.2142],
        [ 0.2439, -0.0774],
        [ 0.7318,  0.0947],
        [ 0.3589,  0.1869],
        [ 0.3165,  0.1627],
        [ 0.3928,  0.2324],
        [ 0.4777,  0.3832],
        [ 0.2626,  0.3071],
        [ 0.2090,  0.2319]], device='cuda:0', grad_fn=<AddmmBackward0>)
/mnt/c/Users/Lighte/GitHub/AIBLAB/Transformer/nlp_code.py:87: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
Training vector_dim=100...