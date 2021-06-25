[root@gpu1 vse_infty-master-my-graph-gru-vse++nowarmup]# sh train_region.sh 
2021-06-15 09:48:16,701 Namespace(backbone_lr_factor=0.01, backbone_path='', backbone_source='detector', backbone_warmup_epochs=5, batch_size=128, data_name='coco', data_path='../data/coco', embed_size=1024, embedding_warmup_epochs=2, grad_clip=2.0, img_dim=2048, input_scale_factor=1, learning_rate=0.0005, log_step=200, logger_name='runs/coco_butd_region_bert/log', lr_update=15, margin=0.2, max_violation=False, model_name='runs/coco_butd_region_bert', no_imgnorm=False, no_txtnorm=False, num_epochs=25, optim='adam', precomp_enc_type='basic', reset_start_epoch=False, resume='', val_step=500, vocab_path='./vocab/', vse_mean_warmup_epochs=1, word_dim=300, workers=10)
2021-06-15 09:48:16,701 Model name '/public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased' not found in model shortcut name list (bert-base-uncased, bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, bert-base-multilingual-cased, bert-base-chinese, bert-base-german-cased, bert-large-uncased-whole-word-masking, bert-large-cased-whole-word-masking, bert-large-uncased-whole-word-masking-finetuned-squad, bert-large-cased-whole-word-masking-finetuned-squad, bert-base-cased-finetuned-mrpc). Assuming '/public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased' is a path or url to a directory containing tokenizer files.
2021-06-15 09:48:16,701 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/added_tokens.json. We won't load it.
2021-06-15 09:48:16,701 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/special_tokens_map.json. We won't load it.
2021-06-15 09:48:16,702 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/tokenizer_config.json. We won't load it.
2021-06-15 09:48:16,702 loading file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/vocab.txt
2021-06-15 09:48:16,702 loading file None
2021-06-15 09:48:16,702 loading file None
2021-06-15 09:48:16,702 loading file None
/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-vse++nowarmup/lib/encoders.py:48: UserWarning: nn.init.constant is now deprecated in favor of nn.init.constant_.
  nn.init.constant(self.W[1].weight, 0)
/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-vse++nowarmup/lib/encoders.py:49: UserWarning: nn.init.constant is now deprecated in favor of nn.init.constant_.
  nn.init.constant(self.W[1].bias, 0)
2021-06-15 09:48:47,901 loading configuration file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/config.json
2021-06-15 09:48:47,902 Model config {
  "attention_probs_dropout_prob": 0.1,
  "finetuning_task": null,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "num_labels": 2,
  "output_attentions": false,
  "output_hidden_states": false,
  "pruned_heads": {},
  "torchscript": false,
  "type_vocab_size": 2,
  "use_bfloat16": false,
  "vocab_size": 30522
}

2021-06-15 09:48:47,902 loading weights file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/pytorch_model.bin
2021-06-15 09:48:55,484 Use adam as the optimizer, with init lr 0.0005
2021-06-15 09:48:55,485 Image encoder is data paralleled now.
2021-06-15 09:48:55,485 runs/coco_butd_region_bert/log
2021-06-15 09:48:55,485 runs/coco_butd_region_bert
2021-06-15 09:48:55,486 image encoder trainable parameters: 20490144
2021-06-15 09:48:55,491 txt encoder trainable parameters: 137319072
/root/anaconda3/lib/python3.6/site-packages/torch/nn/modules/rnn.py:735: UserWarning: RNN module weights are not part of single contiguous chunk of memory. This means they need to be compacted at every call, possibly greatly increasing memory usage. To compact weights again call flatten_parameters(). (Triggered internally at  /pytorch/aten/src/ATen/native/cudnn/RNN.cpp:1264.)
  self.dropout, self.training, self.bidirectional, self.batch_first)
^C^CException ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7f78d1f715f8>>
Traceback (most recent call last):
  File "/root/anaconda3/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 1101, in __del__
    self._shutdown_workers()
  File "/root/anaconda3/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 1075, in _shutdown_workers
    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)
  File "/root/anaconda3/lib/python3.6/multiprocessing/process.py", line 124, in join
    res = self._popen.wait(timeout)
  File "/root/anaconda3/lib/python3.6/multiprocessing/popen_fork.py", line 47, in wait
    if not wait([self.sentinel], timeout):
  File "/root/anaconda3/lib/python3.6/multiprocessing/connection.py", line 911, in wait
    ready = selector.select(timeout)
  File "/root/anaconda3/lib/python3.6/selectors.py", line 376, in select
    fd_event_list = self._poll.poll(timeout)
KeyboardInterrupt: 
Traceback (most recent call last):
  File "train.py", line 267, in <module>
    main()
  File "train.py", line 99, in main
    train(opt, train_loader, model, epoch, val_loader)
  File "train.py", line 146, in train
    model.train_emb(images, captions, lengths, image_lengths=img_lengths)
  File "/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-vse++nowarmup/lib/vse.py", line 200, in train_emb
    clip_grad_norm_(self.params, self.grad_clip)
  File "/root/anaconda3/lib/python3.6/site-packages/torch/nn/utils/clip_grad.py", line 35, in clip_grad_norm_
    if clip_coef < 1:
KeyboardInterrupt
[root@gpu1 vse_infty-master-my-graph-gru-vse++nowarmup]# sh train_region.sh 
Traceback (most recent call last):
  File "train.py", line 9, in <module>
    from lib.vse import VSEModel
  File "/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-vse++nowarmup/lib/vse.py", line 11, in <module>
    from lib.loss import ContrastiveLoss
  File "/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-vse++nowarmup/lib/loss.py", line 154
    cost_s = cost_s.max(1)[0]
                            ^
TabError: inconsistent use of tabs and spaces in indentation
[root@gpu1 vse_infty-master-my-graph-gru-vse++nowarmup]# sh train_region.sh 
2021-06-15 09:50:51,595 Namespace(backbone_lr_factor=0.01, backbone_path='', backbone_source='detector', backbone_warmup_epochs=5, batch_size=128, data_name='coco', data_path='../data/coco', embed_size=1024, embedding_warmup_epochs=2, grad_clip=2.0, img_dim=2048, input_scale_factor=1, learning_rate=0.0005, log_step=200, logger_name='runs/coco_butd_region_bert/log', lr_update=15, margin=0.2, max_violation=False, model_name='runs/coco_butd_region_bert', no_imgnorm=False, no_txtnorm=False, num_epochs=25, optim='adam', precomp_enc_type='basic', reset_start_epoch=False, resume='', val_step=500, vocab_path='./vocab/', vse_mean_warmup_epochs=1, word_dim=300, workers=10)
2021-06-15 09:50:51,595 Model name '/public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased' not found in model shortcut name list (bert-base-uncased, bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, bert-base-multilingual-cased, bert-base-chinese, bert-base-german-cased, bert-large-uncased-whole-word-masking, bert-large-cased-whole-word-masking, bert-large-uncased-whole-word-masking-finetuned-squad, bert-large-cased-whole-word-masking-finetuned-squad, bert-base-cased-finetuned-mrpc). Assuming '/public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased' is a path or url to a directory containing tokenizer files.
2021-06-15 09:50:51,595 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/added_tokens.json. We won't load it.
2021-06-15 09:50:51,595 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/special_tokens_map.json. We won't load it.
2021-06-15 09:50:51,595 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/tokenizer_config.json. We won't load it.
2021-06-15 09:50:51,596 loading file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/vocab.txt
2021-06-15 09:50:51,596 loading file None
2021-06-15 09:50:51,596 loading file None
2021-06-15 09:50:51,596 loading file None
/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-vse++nowarmup/lib/encoders.py:48: UserWarning: nn.init.constant is now deprecated in favor of nn.init.constant_.
  nn.init.constant(self.W[1].weight, 0)
/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-vse++nowarmup/lib/encoders.py:49: UserWarning: nn.init.constant is now deprecated in favor of nn.init.constant_.
  nn.init.constant(self.W[1].bias, 0)
2021-06-15 09:51:20,546 loading configuration file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/config.json
2021-06-15 09:51:20,548 Model config {
  "attention_probs_dropout_prob": 0.1,
  "finetuning_task": null,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "num_labels": 2,
  "output_attentions": false,
  "output_hidden_states": false,
  "pruned_heads": {},
  "torchscript": false,
  "type_vocab_size": 2,
  "use_bfloat16": false,
  "vocab_size": 30522
}

2021-06-15 09:51:20,548 loading weights file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/pytorch_model.bin
2021-06-15 09:51:27,729 Use adam as the optimizer, with init lr 0.0005
2021-06-15 09:51:27,730 Image encoder is data paralleled now.
2021-06-15 09:51:27,730 runs/coco_butd_region_bert/log
2021-06-15 09:51:27,730 runs/coco_butd_region_bert
2021-06-15 09:51:27,732 image encoder trainable parameters: 20490144
2021-06-15 09:51:27,737 txt encoder trainable parameters: 137319072
/root/anaconda3/lib/python3.6/site-packages/torch/nn/modules/rnn.py:735: UserWarning: RNN module weights are not part of single contiguous chunk of memory. This means they need to be compacted at every call, possibly greatly increasing memory usage. To compact weights again call flatten_parameters(). (Triggered internally at  /pytorch/aten/src/ATen/native/cudnn/RNN.cpp:1264.)
  self.dropout, self.training, self.bidirectional, self.batch_first)
2021-06-15 09:55:08,653 Epoch: [0][199/4426]	Eit 200  lr 0.0005  Le 51.2221 (52.9275)	Time 0.889 (0.000)	Data 0.002 (0.000)	
2021-06-15 09:58:25,107 Epoch: [0][399/4426]	Eit 400  lr 0.0005  Le 50.1378 (51.9476)	Time 1.077 (0.000)	Data 0.002 (0.000)	
2021-06-15 10:01:43,518 Epoch: [0][599/4426]	Eit 600  lr 0.0005  Le 38.5714 (49.5946)	Time 0.814 (0.000)	Data 0.002 (0.000)	
2021-06-15 10:04:56,132 Epoch: [0][799/4426]	Eit 800  lr 0.0005  Le 39.4861 (47.1433)	Time 1.178 (0.000)	Data 0.002 (0.000)	
2021-06-15 10:08:11,612 Epoch: [0][999/4426]	Eit 1000  lr 0.0005  Le 39.2705 (45.3186)	Time 0.855 (0.000)	Data 0.002 (0.000)	
2021-06-15 10:11:24,570 Epoch: [0][1199/4426]	Eit 1200  lr 0.0005  Le 36.6892 (43.9211)	Time 1.015 (0.000)	Data 0.002 (0.000)	
2021-06-15 10:14:42,680 Epoch: [0][1399/4426]	Eit 1400  lr 0.0005  Le 40.1800 (42.7304)	Time 0.871 (0.000)	Data 0.002 (0.000)	
2021-06-15 10:17:57,450 Epoch: [0][1599/4426]	Eit 1600  lr 0.0005  Le 33.5283 (41.7647)	Time 1.245 (0.000)	Data 0.002 (0.000)	
2021-06-15 10:21:11,642 Epoch: [0][1799/4426]	Eit 1800  lr 0.0005  Le 34.4352 (40.9186)	Time 0.799 (0.000)	Data 0.002 (0.000)	
2021-06-15 10:24:27,847 Epoch: [0][1999/4426]	Eit 2000  lr 0.0005  Le 34.0323 (40.1730)	Time 0.728 (0.000)	Data 0.002 (0.000)	
2021-06-15 10:27:45,362 Epoch: [0][2199/4426]	Eit 2200  lr 0.0005  Le 32.9125 (39.5320)	Time 0.812 (0.000)	Data 0.002 (0.000)	
2021-06-15 10:31:04,134 Epoch: [0][2399/4426]	Eit 2400  lr 0.0005  Le 34.4282 (38.9449)	Time 0.869 (0.000)	Data 0.002 (0.000)	
2021-06-15 10:34:19,318 Epoch: [0][2599/4426]	Eit 2600  lr 0.0005  Le 29.8328 (38.3853)	Time 1.002 (0.000)	Data 0.002 (0.000)	
2021-06-15 10:37:34,708 Epoch: [0][2799/4426]	Eit 2800  lr 0.0005  Le 29.3820 (37.9071)	Time 0.825 (0.000)	Data 0.002 (0.000)	
2021-06-15 10:40:49,130 Epoch: [0][2999/4426]	Eit 3000  lr 0.0005  Le 30.5655 (37.4773)	Time 1.132 (0.000)	Data 0.002 (0.000)	
2021-06-15 10:44:02,874 Epoch: [0][3199/4426]	Eit 3200  lr 0.0005  Le 29.8914 (37.0569)	Time 0.977 (0.000)	Data 0.004 (0.000)	
2021-06-15 10:47:18,786 Epoch: [0][3399/4426]	Eit 3400  lr 0.0005  Le 30.9751 (36.6788)	Time 1.311 (0.000)	Data 0.002 (0.000)	
2021-06-15 10:50:32,338 Epoch: [0][3599/4426]	Eit 3600  lr 0.0005  Le 32.7706 (36.3289)	Time 1.104 (0.000)	Data 0.002 (0.000)	
2021-06-15 10:53:45,810 Epoch: [0][3799/4426]	Eit 3800  lr 0.0005  Le 32.2437 (36.0169)	Time 1.097 (0.000)	Data 0.002 (0.000)	
2021-06-15 10:56:56,912 Epoch: [0][3999/4426]	Eit 4000  lr 0.0005  Le 33.7062 (35.7113)	Time 0.949 (0.000)	Data 0.002 (0.000)	
2021-06-15 11:00:11,281 Epoch: [0][4199/4426]	Eit 4200  lr 0.0005  Le 32.1753 (35.4152)	Time 0.973 (0.000)	Data 0.002 (0.000)	
2021-06-15 11:03:28,892 Epoch: [0][4399/4426]	Eit 4400  lr 0.0005  Le 32.0207 (35.1387)	Time 1.071 (0.000)	Data 0.002 (0.000)	
2021-06-15 11:04:03,139 Test: [0/40]	Le 59.4314 (59.4313)	Time 8.076 (0.000)	
2021-06-15 11:04:19,425 calculate similarity time: 0.06977558135986328
2021-06-15 11:04:19,982 Image to text: 72.9, 94.9, 98.0, 1.0, 2.4
2021-06-15 11:04:20,354 Text to image: 57.3, 87.9, 95.2, 1.0, 4.1
2021-06-15 11:04:20,354 Current rsum is 506.28000000000003
2021-06-15 11:04:23,286 runs/coco_butd_region_bert/log
2021-06-15 11:04:23,286 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-15 11:04:23,287 image encoder trainable parameters: 20490144
2021-06-15 11:04:23,293 txt encoder trainable parameters: 137319072
2021-06-15 11:07:19,688 Epoch: [1][174/4426]	Eit 4600  lr 0.0005  Le 33.7458 (28.6870)	Time 1.012 (0.000)	Data 0.002 (0.000)	
2021-06-15 11:10:36,458 Epoch: [1][374/4426]	Eit 4800  lr 0.0005  Le 29.4175 (28.4490)	Time 1.026 (0.000)	Data 0.003 (0.000)	
2021-06-15 11:13:51,034 Epoch: [1][574/4426]	Eit 5000  lr 0.0005  Le 28.4301 (28.3462)	Time 0.938 (0.000)	Data 0.002 (0.000)	
2021-06-15 11:17:07,154 Epoch: [1][774/4426]	Eit 5200  lr 0.0005  Le 25.7449 (28.2517)	Time 1.042 (0.000)	Data 0.002 (0.000)	
2021-06-15 11:20:23,093 Epoch: [1][974/4426]	Eit 5400  lr 0.0005  Le 29.4325 (28.1808)	Time 1.001 (0.000)	Data 0.002 (0.000)	
2021-06-15 11:23:40,952 Epoch: [1][1174/4426]	Eit 5600  lr 0.0005  Le 24.0411 (28.1069)	Time 0.979 (0.000)	Data 0.002 (0.000)	
2021-06-15 11:26:56,770 Epoch: [1][1374/4426]	Eit 5800  lr 0.0005  Le 26.5767 (28.0471)	Time 0.918 (0.000)	Data 0.002 (0.000)	
2021-06-15 11:30:10,965 Epoch: [1][1574/4426]	Eit 6000  lr 0.0005  Le 27.8280 (28.0231)	Time 0.862 (0.000)	Data 0.006 (0.000)	
2021-06-15 11:33:25,427 Epoch: [1][1774/4426]	Eit 6200  lr 0.0005  Le 25.0941 (27.9838)	Time 0.971 (0.000)	Data 0.004 (0.000)	
2021-06-15 11:36:36,608 Epoch: [1][1974/4426]	Eit 6400  lr 0.0005  Le 28.9193 (27.9129)	Time 1.243 (0.000)	Data 0.002 (0.000)	
2021-06-15 11:39:46,623 Epoch: [1][2174/4426]	Eit 6600  lr 0.0005  Le 28.1706 (27.8740)	Time 0.928 (0.000)	Data 0.002 (0.000)	
2021-06-15 11:43:02,334 Epoch: [1][2374/4426]	Eit 6800  lr 0.0005  Le 28.8527 (27.8218)	Time 1.140 (0.000)	Data 0.003 (0.000)	
2021-06-15 11:46:17,489 Epoch: [1][2574/4426]	Eit 7000  lr 0.0005  Le 25.4597 (27.7609)	Time 1.174 (0.000)	Data 0.002 (0.000)	
2021-06-15 11:49:31,598 Epoch: [1][2774/4426]	Eit 7200  lr 0.0005  Le 25.4197 (27.7166)	Time 1.118 (0.000)	Data 0.002 (0.000)	
2021-06-15 11:52:45,609 Epoch: [1][2974/4426]	Eit 7400  lr 0.0005  Le 26.2150 (27.6509)	Time 0.756 (0.000)	Data 0.002 (0.000)	
2021-06-15 11:56:04,169 Epoch: [1][3174/4426]	Eit 7600  lr 0.0005  Le 24.2003 (27.6013)	Time 0.936 (0.000)	Data 0.002 (0.000)	
2021-06-15 11:59:16,820 Epoch: [1][3374/4426]	Eit 7800  lr 0.0005  Le 25.6881 (27.5484)	Time 0.896 (0.000)	Data 0.002 (0.000)	
2021-06-15 12:02:33,525 Epoch: [1][3574/4426]	Eit 8000  lr 0.0005  Le 22.7053 (27.4832)	Time 0.772 (0.000)	Data 0.002 (0.000)	
2021-06-15 12:05:45,464 Epoch: [1][3774/4426]	Eit 8200  lr 0.0005  Le 25.4759 (27.4403)	Time 0.928 (0.000)	Data 0.002 (0.000)	
2021-06-15 12:08:59,958 Epoch: [1][3974/4426]	Eit 8400  lr 0.0005  Le 25.8991 (27.3968)	Time 1.041 (0.000)	Data 0.002 (0.000)	
2021-06-15 12:12:14,716 Epoch: [1][4174/4426]	Eit 8600  lr 0.0005  Le 23.7507 (27.3369)	Time 1.377 (0.000)	Data 0.002 (0.000)	
2021-06-15 12:15:28,953 Epoch: [1][4374/4426]	Eit 8800  lr 0.0005  Le 22.9915 (27.2898)	Time 0.744 (0.000)	Data 0.002 (0.000)	
2021-06-15 12:16:27,025 Test: [0/40]	Le 59.8660 (59.8660)	Time 7.849 (0.000)	
2021-06-15 12:16:42,552 calculate similarity time: 0.07026529312133789
2021-06-15 12:16:43,079 Image to text: 75.1, 95.2, 98.2, 1.0, 2.4
2021-06-15 12:16:43,514 Text to image: 60.8, 90.3, 95.8, 1.0, 3.8
2021-06-15 12:16:43,514 Current rsum is 515.44
2021-06-15 12:16:47,143 runs/coco_butd_region_bert/log
2021-06-15 12:16:47,144 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-15 12:16:47,147 image encoder trainable parameters: 20490144
2021-06-15 12:16:47,156 txt encoder trainable parameters: 137319072
2021-06-15 12:19:22,817 Epoch: [2][149/4426]	Eit 9000  lr 0.0005  Le 30.5270 (25.1226)	Time 0.992 (0.000)	Data 0.002 (0.000)	
2021-06-15 12:22:41,350 Epoch: [2][349/4426]	Eit 9200  lr 0.0005  Le 27.5500 (25.2456)	Time 1.214 (0.000)	Data 0.002 (0.000)	
2021-06-15 12:25:57,666 Epoch: [2][549/4426]	Eit 9400  lr 0.0005  Le 27.8436 (25.2815)	Time 0.916 (0.000)	Data 0.002 (0.000)	
2021-06-15 12:29:12,372 Epoch: [2][749/4426]	Eit 9600  lr 0.0005  Le 26.6105 (25.2132)	Time 0.967 (0.000)	Data 0.002 (0.000)	
2021-06-15 12:32:28,604 Epoch: [2][949/4426]	Eit 9800  lr 0.0005  Le 21.1381 (25.1743)	Time 1.034 (0.000)	Data 0.002 (0.000)	
2021-06-15 12:35:45,235 Epoch: [2][1149/4426]	Eit 10000  lr 0.0005  Le 26.7913 (25.2064)	Time 0.999 (0.000)	Data 0.002 (0.000)	
2021-06-15 12:39:01,312 Epoch: [2][1349/4426]	Eit 10200  lr 0.0005  Le 20.1678 (25.1762)	Time 1.287 (0.000)	Data 0.002 (0.000)	
2021-06-15 12:42:15,489 Epoch: [2][1549/4426]	Eit 10400  lr 0.0005  Le 26.2212 (25.1560)	Time 1.002 (0.000)	Data 0.002 (0.000)	
2021-06-15 12:45:29,705 Epoch: [2][1749/4426]	Eit 10600  lr 0.0005  Le 27.4777 (25.1040)	Time 0.911 (0.000)	Data 0.002 (0.000)	
2021-06-15 12:48:46,528 Epoch: [2][1949/4426]	Eit 10800  lr 0.0005  Le 21.8271 (25.1180)	Time 0.868 (0.000)	Data 0.002 (0.000)	
2021-06-15 12:52:02,236 Epoch: [2][2149/4426]	Eit 11000  lr 0.0005  Le 24.8818 (25.0930)	Time 0.781 (0.000)	Data 0.002 (0.000)	
2021-06-15 12:55:17,065 Epoch: [2][2349/4426]	Eit 11200  lr 0.0005  Le 23.6229 (25.0926)	Time 0.938 (0.000)	Data 0.002 (0.000)	
2021-06-15 12:58:29,250 Epoch: [2][2549/4426]	Eit 11400  lr 0.0005  Le 24.3821 (25.0751)	Time 0.728 (0.000)	Data 0.002 (0.000)	
2021-06-15 13:01:45,522 Epoch: [2][2749/4426]	Eit 11600  lr 0.0005  Le 27.6506 (25.0464)	Time 0.780 (0.000)	Data 0.002 (0.000)	
2021-06-15 13:05:00,431 Epoch: [2][2949/4426]	Eit 11800  lr 0.0005  Le 28.0189 (25.0334)	Time 1.119 (0.000)	Data 0.002 (0.000)	
2021-06-15 13:08:18,056 Epoch: [2][3149/4426]	Eit 12000  lr 0.0005  Le 21.0090 (25.0020)	Time 0.993 (0.000)	Data 0.002 (0.000)	
2021-06-15 13:11:33,908 Epoch: [2][3349/4426]	Eit 12200  lr 0.0005  Le 22.8554 (24.9658)	Time 0.730 (0.000)	Data 0.002 (0.000)	
2021-06-15 13:14:50,272 Epoch: [2][3549/4426]	Eit 12400  lr 0.0005  Le 26.0779 (24.9469)	Time 1.045 (0.000)	Data 0.002 (0.000)	
2021-06-15 13:18:04,224 Epoch: [2][3749/4426]	Eit 12600  lr 0.0005  Le 30.4421 (24.9231)	Time 0.992 (0.000)	Data 0.002 (0.000)	
2021-06-15 13:21:22,185 Epoch: [2][3949/4426]	Eit 12800  lr 0.0005  Le 26.3165 (24.9137)	Time 1.042 (0.000)	Data 0.002 (0.000)	
2021-06-15 13:24:38,990 Epoch: [2][4149/4426]	Eit 13000  lr 0.0005  Le 23.9858 (24.8967)	Time 1.041 (0.000)	Data 0.002 (0.000)	
2021-06-15 13:27:52,901 Epoch: [2][4349/4426]	Eit 13200  lr 0.0005  Le 23.5036 (24.8679)	Time 0.964 (0.000)	Data 0.002 (0.000)	
2021-06-15 13:29:15,038 Test: [0/40]	Le 59.7348 (59.7347)	Time 8.043 (0.000)	
2021-06-15 13:29:30,819 calculate similarity time: 0.06957578659057617
2021-06-15 13:29:31,314 Image to text: 77.8, 96.6, 99.2, 1.0, 2.1
2021-06-15 13:29:31,626 Text to image: 62.5, 90.7, 96.5, 1.0, 3.6
2021-06-15 13:29:31,627 Current rsum is 523.3399999999999
2021-06-15 13:29:35,277 runs/coco_butd_region_bert/log
2021-06-15 13:29:35,277 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-15 13:29:35,280 image encoder trainable parameters: 20490144
2021-06-15 13:29:35,293 txt encoder trainable parameters: 137319072
2021-06-15 13:31:46,871 Epoch: [3][124/4426]	Eit 13400  lr 0.0005  Le 22.1829 (23.0738)	Time 1.030 (0.000)	Data 0.002 (0.000)	
2021-06-15 13:35:02,637 Epoch: [3][324/4426]	Eit 13600  lr 0.0005  Le 26.0764 (23.4611)	Time 1.130 (0.000)	Data 0.002 (0.000)	
2021-06-15 13:38:17,255 Epoch: [3][524/4426]	Eit 13800  lr 0.0005  Le 22.1576 (23.6057)	Time 0.809 (0.000)	Data 0.002 (0.000)	
2021-06-15 13:41:33,079 Epoch: [3][724/4426]	Eit 14000  lr 0.0005  Le 22.4166 (23.5554)	Time 0.821 (0.000)	Data 0.002 (0.000)	
2021-06-15 13:44:47,312 Epoch: [3][924/4426]	Eit 14200  lr 0.0005  Le 31.4536 (23.5027)	Time 0.984 (0.000)	Data 0.002 (0.000)	
2021-06-15 13:47:51,671 Epoch: [3][1124/4426]	Eit 14400  lr 0.0005  Le 21.2794 (23.5234)	Time 0.973 (0.000)	Data 0.002 (0.000)	
2021-06-15 13:51:07,419 Epoch: [3][1324/4426]	Eit 14600  lr 0.0005  Le 25.9946 (23.5404)	Time 1.228 (0.000)	Data 0.002 (0.000)	
2021-06-15 13:54:24,092 Epoch: [3][1524/4426]	Eit 14800  lr 0.0005  Le 26.7739 (23.5399)	Time 0.902 (0.000)	Data 0.002 (0.000)	
2021-06-15 13:57:39,250 Epoch: [3][1724/4426]	Eit 15000  lr 0.0005  Le 22.5623 (23.4953)	Time 1.023 (0.000)	Data 0.002 (0.000)	
2021-06-15 14:00:56,296 Epoch: [3][1924/4426]	Eit 15200  lr 0.0005  Le 22.4513 (23.5198)	Time 1.081 (0.000)	Data 0.002 (0.000)	
2021-06-15 14:04:12,398 Epoch: [3][2124/4426]	Eit 15400  lr 0.0005  Le 19.3295 (23.4956)	Time 0.995 (0.000)	Data 0.004 (0.000)	
2021-06-15 14:07:29,859 Epoch: [3][2324/4426]	Eit 15600  lr 0.0005  Le 23.5116 (23.4678)	Time 1.067 (0.000)	Data 0.002 (0.000)	
2021-06-15 14:10:45,570 Epoch: [3][2524/4426]	Eit 15800  lr 0.0005  Le 22.7500 (23.4540)	Time 1.011 (0.000)	Data 0.002 (0.000)	
2021-06-15 14:14:01,295 Epoch: [3][2724/4426]	Eit 16000  lr 0.0005  Le 22.6901 (23.4360)	Time 0.835 (0.000)	Data 0.002 (0.000)	
2021-06-15 14:17:17,291 Epoch: [3][2924/4426]	Eit 16200  lr 0.0005  Le 25.8858 (23.4355)	Time 0.982 (0.000)	Data 0.002 (0.000)	
2021-06-15 14:20:35,169 Epoch: [3][3124/4426]	Eit 16400  lr 0.0005  Le 20.0392 (23.4180)	Time 0.985 (0.000)	Data 0.002 (0.000)	
2021-06-15 14:23:51,824 Epoch: [3][3324/4426]	Eit 16600  lr 0.0005  Le 26.6495 (23.4232)	Time 1.160 (0.000)	Data 0.002 (0.000)	
2021-06-15 14:27:08,398 Epoch: [3][3524/4426]	Eit 16800  lr 0.0005  Le 23.0446 (23.4051)	Time 0.868 (0.000)	Data 0.002 (0.000)	
2021-06-15 14:30:27,747 Epoch: [3][3724/4426]	Eit 17000  lr 0.0005  Le 21.7716 (23.4146)	Time 1.004 (0.000)	Data 0.002 (0.000)	
2021-06-15 14:33:45,689 Epoch: [3][3924/4426]	Eit 17200  lr 0.0005  Le 24.4452 (23.4012)	Time 0.815 (0.000)	Data 0.002 (0.000)	
2021-06-15 14:37:03,115 Epoch: [3][4124/4426]	Eit 17400  lr 0.0005  Le 24.1603 (23.3744)	Time 0.903 (0.000)	Data 0.002 (0.000)	
2021-06-15 14:40:19,057 Epoch: [3][4324/4426]	Eit 17600  lr 0.0005  Le 22.2066 (23.3592)	Time 0.831 (0.000)	Data 0.003 (0.000)	
2021-06-15 14:42:07,274 Test: [0/40]	Le 60.2283 (60.2283)	Time 8.925 (0.000)	
2021-06-15 14:42:23,182 calculate similarity time: 0.06577086448669434
2021-06-15 14:42:23,703 Image to text: 79.3, 97.3, 99.1, 1.0, 2.2
2021-06-15 14:42:24,142 Text to image: 63.5, 90.9, 96.5, 1.0, 3.5
2021-06-15 14:42:24,143 Current rsum is 526.5600000000001
2021-06-15 14:42:27,990 runs/coco_butd_region_bert/log
2021-06-15 14:42:27,991 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-15 14:42:27,996 image encoder trainable parameters: 20490144
2021-06-15 14:42:28,006 txt encoder trainable parameters: 137319072
2021-06-15 14:44:16,644 Epoch: [4][99/4426]	Eit 17800  lr 0.0005  Le 19.2532 (21.9839)	Time 1.153 (0.000)	Data 0.002 (0.000)	
2021-06-15 14:47:35,536 Epoch: [4][299/4426]	Eit 18000  lr 0.0005  Le 22.0811 (22.1045)	Time 1.502 (0.000)	Data 0.003 (0.000)	
2021-06-15 14:50:49,320 Epoch: [4][499/4426]	Eit 18200  lr 0.0005  Le 20.6043 (22.0863)	Time 0.886 (0.000)	Data 0.003 (0.000)	
2021-06-15 14:54:05,243 Epoch: [4][699/4426]	Eit 18400  lr 0.0005  Le 21.4526 (22.0930)	Time 0.932 (0.000)	Data 0.002 (0.000)	
2021-06-15 14:57:24,537 Epoch: [4][899/4426]	Eit 18600  lr 0.0005  Le 21.7037 (22.1545)	Time 0.917 (0.000)	Data 0.002 (0.000)	
2021-06-15 15:00:43,985 Epoch: [4][1099/4426]	Eit 18800  lr 0.0005  Le 18.9048 (22.1509)	Time 0.736 (0.000)	Data 0.002 (0.000)	
2021-06-15 15:04:00,809 Epoch: [4][1299/4426]	Eit 19000  lr 0.0005  Le 23.2916 (22.1714)	Time 1.037 (0.000)	Data 0.002 (0.000)	
2021-06-15 15:07:18,324 Epoch: [4][1499/4426]	Eit 19200  lr 0.0005  Le 20.8448 (22.1569)	Time 0.760 (0.000)	Data 0.002 (0.000)	
2021-06-15 15:10:35,694 Epoch: [4][1699/4426]	Eit 19400  lr 0.0005  Le 20.5638 (22.1254)	Time 0.831 (0.000)	Data 0.002 (0.000)	
2021-06-15 15:13:48,708 Epoch: [4][1899/4426]	Eit 19600  lr 0.0005  Le 18.1513 (22.1256)	Time 0.999 (0.000)	Data 0.002 (0.000)	
2021-06-15 15:17:03,634 Epoch: [4][2099/4426]	Eit 19800  lr 0.0005  Le 24.5909 (22.1314)	Time 1.018 (0.000)	Data 0.002 (0.000)	
2021-06-15 15:20:17,994 Epoch: [4][2299/4426]	Eit 20000  lr 0.0005  Le 21.6143 (22.1143)	Time 1.130 (0.000)	Data 0.002 (0.000)	
2021-06-15 15:23:32,513 Epoch: [4][2499/4426]	Eit 20200  lr 0.0005  Le 18.8164 (22.1132)	Time 1.005 (0.000)	Data 0.002 (0.000)	
2021-06-15 15:26:48,902 Epoch: [4][2699/4426]	Eit 20400  lr 0.0005  Le 21.9237 (22.1230)	Time 1.007 (0.000)	Data 0.002 (0.000)	
2021-06-15 15:30:04,083 Epoch: [4][2899/4426]	Eit 20600  lr 0.0005  Le 25.2351 (22.1396)	Time 1.048 (0.000)	Data 0.002 (0.000)	
2021-06-15 15:33:19,886 Epoch: [4][3099/4426]	Eit 20800  lr 0.0005  Le 22.7684 (22.1375)	Time 0.984 (0.000)	Data 0.002 (0.000)	
2021-06-15 15:36:35,610 Epoch: [4][3299/4426]	Eit 21000  lr 0.0005  Le 18.3855 (22.1357)	Time 1.108 (0.000)	Data 0.002 (0.000)	
2021-06-15 15:39:50,840 Epoch: [4][3499/4426]	Eit 21200  lr 0.0005  Le 23.5806 (22.1402)	Time 0.884 (0.000)	Data 0.002 (0.000)	
2021-06-15 15:43:10,475 Epoch: [4][3699/4426]	Eit 21400  lr 0.0005  Le 22.4964 (22.1542)	Time 0.941 (0.000)	Data 0.002 (0.000)	
2021-06-15 15:46:29,758 Epoch: [4][3899/4426]	Eit 21600  lr 0.0005  Le 23.0515 (22.1606)	Time 0.889 (0.000)	Data 0.002 (0.000)	
2021-06-15 15:49:46,993 Epoch: [4][4099/4426]	Eit 21800  lr 0.0005  Le 24.9737 (22.1550)	Time 1.001 (0.000)	Data 0.002 (0.000)	
2021-06-15 15:53:03,161 Epoch: [4][4299/4426]	Eit 22000  lr 0.0005  Le 20.9062 (22.1559)	Time 0.786 (0.000)	Data 0.002 (0.000)	
2021-06-15 15:55:10,425 Test: [0/40]	Le 60.2767 (60.2767)	Time 8.532 (0.000)	
2021-06-15 15:55:20,637 calculate similarity time: 0.05229687690734863
2021-06-15 15:55:21,227 Image to text: 77.6, 96.8, 99.3, 1.0, 2.7
2021-06-15 15:55:21,637 Text to image: 64.4, 91.5, 96.5, 1.0, 3.5
2021-06-15 15:55:21,638 Current rsum is 526.06
2021-06-15 15:55:23,101 runs/coco_butd_region_bert/log
2021-06-15 15:55:23,102 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-15 15:55:23,104 image encoder trainable parameters: 20490144
2021-06-15 15:55:23,109 txt encoder trainable parameters: 137319072
2021-06-15 15:56:43,928 Epoch: [5][74/4426]	Eit 22200  lr 0.0005  Le 22.1895 (21.2953)	Time 1.027 (0.000)	Data 0.002 (0.000)	
2021-06-15 15:59:57,948 Epoch: [5][274/4426]	Eit 22400  lr 0.0005  Le 21.4845 (21.0758)	Time 0.991 (0.000)	Data 0.002 (0.000)	
2021-06-15 16:03:14,643 Epoch: [5][474/4426]	Eit 22600  lr 0.0005  Le 26.6726 (21.1500)	Time 1.140 (0.000)	Data 0.002 (0.000)	
2021-06-15 16:06:29,299 Epoch: [5][674/4426]	Eit 22800  lr 0.0005  Le 18.9342 (21.1756)	Time 0.747 (0.000)	Data 0.002 (0.000)	
2021-06-15 16:09:45,550 Epoch: [5][874/4426]	Eit 23000  lr 0.0005  Le 22.5203 (21.1082)	Time 1.016 (0.000)	Data 0.002 (0.000)	
2021-06-15 16:13:03,648 Epoch: [5][1074/4426]	Eit 23200  lr 0.0005  Le 22.4639 (21.0775)	Time 0.791 (0.000)	Data 0.002 (0.000)	
2021-06-15 16:16:21,490 Epoch: [5][1274/4426]	Eit 23400  lr 0.0005  Le 19.6307 (21.1217)	Time 1.090 (0.000)	Data 0.009 (0.000)	
2021-06-15 16:19:36,610 Epoch: [5][1474/4426]	Eit 23600  lr 0.0005  Le 22.6308 (21.1718)	Time 1.040 (0.000)	Data 0.002 (0.000)	
2021-06-15 16:22:56,522 Epoch: [5][1674/4426]	Eit 23800  lr 0.0005  Le 22.4282 (21.1580)	Time 0.998 (0.000)	Data 0.002 (0.000)	
2021-06-15 16:26:14,594 Epoch: [5][1874/4426]	Eit 24000  lr 0.0005  Le 20.2368 (21.1714)	Time 0.816 (0.000)	Data 0.002 (0.000)	
2021-06-15 16:29:28,835 Epoch: [5][2074/4426]	Eit 24200  lr 0.0005  Le 19.0110 (21.1691)	Time 0.994 (0.000)	Data 0.002 (0.000)	
2021-06-15 16:32:42,778 Epoch: [5][2274/4426]	Eit 24400  lr 0.0005  Le 24.6953 (21.1687)	Time 1.163 (0.000)	Data 0.002 (0.000)	
2021-06-15 16:35:56,425 Epoch: [5][2474/4426]	Eit 24600  lr 0.0005  Le 20.8302 (21.1714)	Time 0.906 (0.000)	Data 0.002 (0.000)	
2021-06-15 16:39:13,533 Epoch: [5][2674/4426]	Eit 24800  lr 0.0005  Le 19.4380 (21.1845)	Time 1.011 (0.000)	Data 0.003 (0.000)	
2021-06-15 16:42:27,208 Epoch: [5][2874/4426]	Eit 25000  lr 0.0005  Le 21.1685 (21.1761)	Time 0.960 (0.000)	Data 0.002 (0.000)	
2021-06-15 16:45:47,594 Epoch: [5][3074/4426]	Eit 25200  lr 0.0005  Le 18.5156 (21.1571)	Time 0.835 (0.000)	Data 0.002 (0.000)	
2021-06-15 16:49:01,289 Epoch: [5][3274/4426]	Eit 25400  lr 0.0005  Le 17.1252 (21.1583)	Time 0.777 (0.000)	Data 0.002 (0.000)	
2021-06-15 16:52:17,120 Epoch: [5][3474/4426]	Eit 25600  lr 0.0005  Le 21.6858 (21.1596)	Time 1.025 (0.000)	Data 0.003 (0.000)	
2021-06-15 16:55:35,921 Epoch: [5][3674/4426]	Eit 25800  lr 0.0005  Le 19.4675 (21.1761)	Time 0.993 (0.000)	Data 0.002 (0.000)	
2021-06-15 16:58:51,746 Epoch: [5][3874/4426]	Eit 26000  lr 0.0005  Le 21.6105 (21.1782)	Time 0.777 (0.000)	Data 0.002 (0.000)	
2021-06-15 17:02:06,783 Epoch: [5][4074/4426]	Eit 26200  lr 0.0005  Le 19.0263 (21.1778)	Time 0.745 (0.000)	Data 0.002 (0.000)	
2021-06-15 17:05:22,931 Epoch: [5][4274/4426]	Eit 26400  lr 0.0005  Le 26.6696 (21.1962)	Time 1.198 (0.000)	Data 0.002 (0.000)	
2021-06-15 17:07:58,272 Test: [0/40]	Le 60.0963 (60.0962)	Time 8.467 (0.000)	
2021-06-15 17:08:14,248 calculate similarity time: 0.0864863395690918
2021-06-15 17:08:14,670 Image to text: 79.5, 97.4, 99.2, 1.0, 2.6
2021-06-15 17:08:14,983 Text to image: 64.7, 91.8, 96.8, 1.0, 3.7
2021-06-15 17:08:14,983 Current rsum is 529.34
2021-06-15 17:08:18,505 runs/coco_butd_region_bert/log
2021-06-15 17:08:18,506 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-15 17:08:18,509 image encoder trainable parameters: 20490144
2021-06-15 17:08:18,521 txt encoder trainable parameters: 137319072
2021-06-15 17:09:14,165 Epoch: [6][49/4426]	Eit 26600  lr 0.0005  Le 21.0156 (20.4494)	Time 0.831 (0.000)	Data 0.002 (0.000)	
2021-06-15 17:12:29,447 Epoch: [6][249/4426]	Eit 26800  lr 0.0005  Le 22.0107 (20.1269)	Time 1.235 (0.000)	Data 0.002 (0.000)	
2021-06-15 17:15:44,479 Epoch: [6][449/4426]	Eit 27000  lr 0.0005  Le 24.1278 (20.3053)	Time 0.771 (0.000)	Data 0.002 (0.000)	
2021-06-15 17:18:58,604 Epoch: [6][649/4426]	Eit 27200  lr 0.0005  Le 21.3183 (20.3258)	Time 1.019 (0.000)	Data 0.002 (0.000)	
2021-06-15 17:22:12,357 Epoch: [6][849/4426]	Eit 27400  lr 0.0005  Le 22.7071 (20.3561)	Time 0.812 (0.000)	Data 0.002 (0.000)	
2021-06-15 17:25:31,594 Epoch: [6][1049/4426]	Eit 27600  lr 0.0005  Le 18.6501 (20.3596)	Time 1.060 (0.000)	Data 0.002 (0.000)	
2021-06-15 17:28:50,336 Epoch: [6][1249/4426]	Eit 27800  lr 0.0005  Le 20.0741 (20.3761)	Time 0.969 (0.000)	Data 0.002 (0.000)	
2021-06-15 17:32:07,300 Epoch: [6][1449/4426]	Eit 28000  lr 0.0005  Le 19.3982 (20.3612)	Time 0.866 (0.000)	Data 0.002 (0.000)	
2021-06-15 17:35:24,027 Epoch: [6][1649/4426]	Eit 28200  lr 0.0005  Le 23.0871 (20.3739)	Time 1.024 (0.000)	Data 0.002 (0.000)	
2021-06-15 17:38:38,860 Epoch: [6][1849/4426]	Eit 28400  lr 0.0005  Le 17.9946 (20.3754)	Time 0.726 (0.000)	Data 0.002 (0.000)	
2021-06-15 17:41:56,389 Epoch: [6][2049/4426]	Eit 28600  lr 0.0005  Le 21.3718 (20.3935)	Time 0.763 (0.000)	Data 0.002 (0.000)	
2021-06-15 17:45:13,266 Epoch: [6][2249/4426]	Eit 28800  lr 0.0005  Le 21.6808 (20.3869)	Time 0.723 (0.000)	Data 0.002 (0.000)	
2021-06-15 17:48:30,331 Epoch: [6][2449/4426]	Eit 29000  lr 0.0005  Le 17.8867 (20.4042)	Time 0.996 (0.000)	Data 0.010 (0.000)	
2021-06-15 17:51:47,884 Epoch: [6][2649/4426]	Eit 29200  lr 0.0005  Le 22.6924 (20.4000)	Time 1.132 (0.000)	Data 0.002 (0.000)	
2021-06-15 17:55:03,525 Epoch: [6][2849/4426]	Eit 29400  lr 0.0005  Le 20.1503 (20.4097)	Time 0.997 (0.000)	Data 0.002 (0.000)	
2021-06-15 17:58:20,035 Epoch: [6][3049/4426]	Eit 29600  lr 0.0005  Le 19.0587 (20.4238)	Time 1.106 (0.000)	Data 0.002 (0.000)	
2021-06-15 18:01:35,144 Epoch: [6][3249/4426]	Eit 29800  lr 0.0005  Le 19.8595 (20.4066)	Time 1.088 (0.000)	Data 0.002 (0.000)	
2021-06-15 18:04:40,608 Epoch: [6][3449/4426]	Eit 30000  lr 0.0005  Le 20.2546 (20.4002)	Time 0.585 (0.000)	Data 0.002 (0.000)	
2021-06-15 18:07:51,571 Epoch: [6][3649/4426]	Eit 30200  lr 0.0005  Le 21.5638 (20.4216)	Time 1.162 (0.000)	Data 0.004 (0.000)	
2021-06-15 18:11:07,727 Epoch: [6][3849/4426]	Eit 30400  lr 0.0005  Le 23.6149 (20.4156)	Time 1.085 (0.000)	Data 0.002 (0.000)	
2021-06-15 18:14:23,098 Epoch: [6][4049/4426]	Eit 30600  lr 0.0005  Le 18.6280 (20.4221)	Time 0.986 (0.000)	Data 0.003 (0.000)	
2021-06-15 18:17:39,776 Epoch: [6][4249/4426]	Eit 30800  lr 0.0005  Le 22.7514 (20.4379)	Time 0.976 (0.000)	Data 0.002 (0.000)	
2021-06-15 18:20:41,651 Test: [0/40]	Le 59.4860 (59.4859)	Time 8.043 (0.000)	
2021-06-15 18:20:57,661 calculate similarity time: 0.07973074913024902
2021-06-15 18:20:58,180 Image to text: 79.4, 97.2, 99.6, 1.0, 2.6
2021-06-15 18:20:58,625 Text to image: 64.2, 91.8, 96.6, 1.0, 3.6
2021-06-15 18:20:58,625 Current rsum is 528.8600000000001
2021-06-15 18:21:00,124 runs/coco_butd_region_bert/log
2021-06-15 18:21:00,125 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-15 18:21:00,127 image encoder trainable parameters: 20490144
2021-06-15 18:21:00,133 txt encoder trainable parameters: 137319072
2021-06-15 18:21:32,157 Epoch: [7][24/4426]	Eit 31000  lr 0.0005  Le 21.5684 (19.6741)	Time 1.125 (0.000)	Data 0.002 (0.000)	
2021-06-15 18:24:49,935 Epoch: [7][224/4426]	Eit 31200  lr 0.0005  Le 21.7853 (19.6585)	Time 1.093 (0.000)	Data 0.006 (0.000)	
2021-06-15 18:28:06,633 Epoch: [7][424/4426]	Eit 31400  lr 0.0005  Le 18.9642 (19.5329)	Time 0.759 (0.000)	Data 0.002 (0.000)	
2021-06-15 18:31:21,728 Epoch: [7][624/4426]	Eit 31600  lr 0.0005  Le 21.6734 (19.6167)	Time 1.244 (0.000)	Data 0.003 (0.000)	
2021-06-15 18:34:36,185 Epoch: [7][824/4426]	Eit 31800  lr 0.0005  Le 18.0391 (19.5912)	Time 0.993 (0.000)	Data 0.002 (0.000)	
2021-06-15 18:37:52,822 Epoch: [7][1024/4426]	Eit 32000  lr 0.0005  Le 17.6502 (19.5106)	Time 0.839 (0.000)	Data 0.002 (0.000)	
2021-06-15 18:41:10,014 Epoch: [7][1224/4426]	Eit 32200  lr 0.0005  Le 23.3065 (19.5334)	Time 0.888 (0.000)	Data 0.002 (0.000)	
2021-06-15 18:44:26,728 Epoch: [7][1424/4426]	Eit 32400  lr 0.0005  Le 20.8163 (19.5719)	Time 0.884 (0.000)	Data 0.003 (0.000)	
2021-06-15 18:47:43,779 Epoch: [7][1624/4426]	Eit 32600  lr 0.0005  Le 21.9740 (19.5908)	Time 1.156 (0.000)	Data 0.002 (0.000)	
2021-06-15 18:51:00,388 Epoch: [7][1824/4426]	Eit 32800  lr 0.0005  Le 23.3837 (19.5935)	Time 1.037 (0.000)	Data 0.002 (0.000)	
2021-06-15 18:54:13,726 Epoch: [7][2024/4426]	Eit 33000  lr 0.0005  Le 23.8136 (19.6142)	Time 0.786 (0.000)	Data 0.002 (0.000)	
2021-06-15 18:57:28,660 Epoch: [7][2224/4426]	Eit 33200  lr 0.0005  Le 18.8928 (19.6256)	Time 0.896 (0.000)	Data 0.002 (0.000)	
2021-06-15 19:00:43,064 Epoch: [7][2424/4426]	Eit 33400  lr 0.0005  Le 22.7591 (19.6407)	Time 0.788 (0.000)	Data 0.002 (0.000)	
2021-06-15 19:03:59,714 Epoch: [7][2624/4426]	Eit 33600  lr 0.0005  Le 16.1981 (19.6347)	Time 0.816 (0.000)	Data 0.002 (0.000)	
2021-06-15 19:07:18,428 Epoch: [7][2824/4426]	Eit 33800  lr 0.0005  Le 21.7540 (19.6416)	Time 1.329 (0.000)	Data 0.002 (0.000)	
2021-06-15 19:10:32,205 Epoch: [7][3024/4426]	Eit 34000  lr 0.0005  Le 23.2799 (19.6426)	Time 0.823 (0.000)	Data 0.002 (0.000)	
2021-06-15 19:13:47,036 Epoch: [7][3224/4426]	Eit 34200  lr 0.0005  Le 22.9219 (19.6568)	Time 0.983 (0.000)	Data 0.002 (0.000)	
2021-06-15 19:17:06,295 Epoch: [7][3424/4426]	Eit 34400  lr 0.0005  Le 15.9093 (19.6686)	Time 0.818 (0.000)	Data 0.002 (0.000)	
2021-06-15 19:20:21,864 Epoch: [7][3624/4426]	Eit 34600  lr 0.0005  Le 20.1270 (19.6692)	Time 0.796 (0.000)	Data 0.002 (0.000)	
2021-06-15 19:23:39,727 Epoch: [7][3824/4426]	Eit 34800  lr 0.0005  Le 21.6394 (19.6774)	Time 0.990 (0.000)	Data 0.002 (0.000)	
2021-06-15 19:26:55,421 Epoch: [7][4024/4426]	Eit 35000  lr 0.0005  Le 16.5840 (19.6972)	Time 0.788 (0.000)	Data 0.002 (0.000)	
2021-06-15 19:30:11,170 Epoch: [7][4224/4426]	Eit 35200  lr 0.0005  Le 22.8472 (19.7079)	Time 0.807 (0.000)	Data 0.002 (0.000)	
2021-06-15 19:33:27,305 Epoch: [7][4424/4426]	Eit 35400  lr 0.0005  Le 21.7606 (19.7095)	Time 0.858 (0.000)	Data 0.002 (0.000)	
2021-06-15 19:33:36,713 Test: [0/40]	Le 59.9257 (59.9256)	Time 8.065 (0.000)	
2021-06-15 19:33:52,710 calculate similarity time: 0.06160712242126465
2021-06-15 19:33:53,197 Image to text: 79.2, 97.6, 99.4, 1.0, 2.0
2021-06-15 19:33:53,647 Text to image: 65.8, 92.1, 97.1, 1.0, 3.6
2021-06-15 19:33:53,647 Current rsum is 531.1800000000001
2021-06-15 19:33:57,216 runs/coco_butd_region_bert/log
2021-06-15 19:33:57,216 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-15 19:33:57,218 image encoder trainable parameters: 20490144
2021-06-15 19:33:57,226 txt encoder trainable parameters: 137319072
2021-06-15 19:37:23,782 Epoch: [8][199/4426]	Eit 35600  lr 0.0005  Le 17.8692 (19.0430)	Time 0.863 (0.000)	Data 0.002 (0.000)	
2021-06-15 19:40:37,273 Epoch: [8][399/4426]	Eit 35800  lr 0.0005  Le 20.5490 (19.0435)	Time 0.775 (0.000)	Data 0.003 (0.000)	
2021-06-15 19:43:51,312 Epoch: [8][599/4426]	Eit 36000  lr 0.0005  Le 20.3977 (19.0381)	Time 0.963 (0.000)	Data 0.002 (0.000)	
2021-06-15 19:47:07,539 Epoch: [8][799/4426]	Eit 36200  lr 0.0005  Le 19.1881 (19.0141)	Time 0.795 (0.000)	Data 0.002 (0.000)	
2021-06-15 19:50:24,130 Epoch: [8][999/4426]	Eit 36400  lr 0.0005  Le 15.3971 (19.1170)	Time 1.235 (0.000)	Data 0.002 (0.000)	
2021-06-15 19:53:39,790 Epoch: [8][1199/4426]	Eit 36600  lr 0.0005  Le 19.9916 (19.1589)	Time 0.851 (0.000)	Data 0.003 (0.000)	
2021-06-15 19:56:56,291 Epoch: [8][1399/4426]	Eit 36800  lr 0.0005  Le 20.8128 (19.1389)	Time 1.216 (0.000)	Data 0.002 (0.000)	
2021-06-15 20:00:15,490 Epoch: [8][1599/4426]	Eit 37000  lr 0.0005  Le 20.1788 (19.1414)	Time 0.748 (0.000)	Data 0.002 (0.000)	
2021-06-15 20:03:33,478 Epoch: [8][1799/4426]	Eit 37200  lr 0.0005  Le 23.4796 (19.1402)	Time 1.097 (0.000)	Data 0.002 (0.000)	
2021-06-15 20:06:52,557 Epoch: [8][1999/4426]	Eit 37400  lr 0.0005  Le 18.2187 (19.1546)	Time 0.885 (0.000)	Data 0.002 (0.000)	
2021-06-15 20:10:10,881 Epoch: [8][2199/4426]	Eit 37600  lr 0.0005  Le 20.0547 (19.1295)	Time 1.027 (0.000)	Data 0.002 (0.000)	
2021-06-15 20:13:22,909 Epoch: [8][2399/4426]	Eit 37800  lr 0.0005  Le 17.7106 (19.1199)	Time 0.590 (0.000)	Data 0.002 (0.000)	
2021-06-15 20:16:31,319 Epoch: [8][2599/4426]	Eit 38000  lr 0.0005  Le 15.7721 (19.1206)	Time 0.923 (0.000)	Data 0.002 (0.000)	
2021-06-15 20:19:45,635 Epoch: [8][2799/4426]	Eit 38200  lr 0.0005  Le 19.6650 (19.1176)	Time 1.044 (0.000)	Data 0.003 (0.000)	
2021-06-15 20:22:59,585 Epoch: [8][2999/4426]	Eit 38400  lr 0.0005  Le 21.0327 (19.1446)	Time 0.809 (0.000)	Data 0.002 (0.000)	
2021-06-15 20:26:16,391 Epoch: [8][3199/4426]	Eit 38600  lr 0.0005  Le 18.1975 (19.1439)	Time 1.061 (0.000)	Data 0.002 (0.000)	
2021-06-15 20:29:31,072 Epoch: [8][3399/4426]	Eit 38800  lr 0.0005  Le 15.2980 (19.1428)	Time 0.942 (0.000)	Data 0.002 (0.000)	
2021-06-15 20:32:46,470 Epoch: [8][3599/4426]	Eit 39000  lr 0.0005  Le 19.8740 (19.1538)	Time 0.934 (0.000)	Data 0.002 (0.000)	
2021-06-15 20:36:02,326 Epoch: [8][3799/4426]	Eit 39200  lr 0.0005  Le 22.0195 (19.1605)	Time 1.085 (0.000)	Data 0.002 (0.000)	
2021-06-15 20:39:19,108 Epoch: [8][3999/4426]	Eit 39400  lr 0.0005  Le 21.1596 (19.1818)	Time 0.873 (0.000)	Data 0.002 (0.000)	
2021-06-15 20:42:38,846 Epoch: [8][4199/4426]	Eit 39600  lr 0.0005  Le 26.2794 (19.1814)	Time 0.970 (0.000)	Data 0.002 (0.000)	
2021-06-15 20:45:55,829 Epoch: [8][4399/4426]	Eit 39800  lr 0.0005  Le 20.5432 (19.1880)	Time 1.112 (0.000)	Data 0.003 (0.000)	
2021-06-15 20:46:29,450 Test: [0/40]	Le 59.9742 (59.9741)	Time 8.080 (0.000)	
2021-06-15 20:46:45,197 calculate similarity time: 0.06206989288330078
2021-06-15 20:46:45,641 Image to text: 78.5, 97.3, 99.6, 1.0, 2.1
2021-06-15 20:46:45,970 Text to image: 65.5, 92.2, 96.9, 1.0, 3.5
2021-06-15 20:46:45,970 Current rsum is 529.98
2021-06-15 20:46:47,437 runs/coco_butd_region_bert/log
2021-06-15 20:46:47,437 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-15 20:46:47,439 image encoder trainable parameters: 20490144
2021-06-15 20:46:47,444 txt encoder trainable parameters: 137319072
2021-06-15 20:49:45,371 Epoch: [9][174/4426]	Eit 40000  lr 0.0005  Le 16.5918 (18.0776)	Time 1.062 (0.000)	Data 0.002 (0.000)	
2021-06-15 20:53:03,987 Epoch: [9][374/4426]	Eit 40200  lr 0.0005  Le 17.6391 (18.2959)	Time 1.040 (0.000)	Data 0.002 (0.000)	
2021-06-15 20:56:18,889 Epoch: [9][574/4426]	Eit 40400  lr 0.0005  Le 22.1594 (18.3202)	Time 0.765 (0.000)	Data 0.002 (0.000)	
2021-06-15 20:59:33,527 Epoch: [9][774/4426]	Eit 40600  lr 0.0005  Le 19.6042 (18.3928)	Time 0.938 (0.000)	Data 0.002 (0.000)	
2021-06-15 21:02:51,698 Epoch: [9][974/4426]	Eit 40800  lr 0.0005  Le 20.0748 (18.4321)	Time 0.976 (0.000)	Data 0.002 (0.000)	
2021-06-15 21:06:06,793 Epoch: [9][1174/4426]	Eit 41000  lr 0.0005  Le 20.8600 (18.4867)	Time 1.099 (0.000)	Data 0.002 (0.000)	
2021-06-15 21:09:20,910 Epoch: [9][1374/4426]	Eit 41200  lr 0.0005  Le 17.4107 (18.5261)	Time 0.845 (0.000)	Data 0.002 (0.000)	
2021-06-15 21:12:39,092 Epoch: [9][1574/4426]	Eit 41400  lr 0.0005  Le 21.0206 (18.5346)	Time 0.981 (0.000)	Data 0.002 (0.000)	
2021-06-15 21:15:55,449 Epoch: [9][1774/4426]	Eit 41600  lr 0.0005  Le 16.1203 (18.5393)	Time 1.059 (0.000)	Data 0.002 (0.000)	
2021-06-15 21:19:12,278 Epoch: [9][1974/4426]	Eit 41800  lr 0.0005  Le 14.6612 (18.5443)	Time 1.020 (0.000)	Data 0.002 (0.000)	
2021-06-15 21:22:29,328 Epoch: [9][2174/4426]	Eit 42000  lr 0.0005  Le 19.7138 (18.5451)	Time 1.065 (0.000)	Data 0.002 (0.000)	
2021-06-15 21:25:48,518 Epoch: [9][2374/4426]	Eit 42200  lr 0.0005  Le 21.6273 (18.5893)	Time 1.396 (0.000)	Data 0.002 (0.000)	
2021-06-15 21:29:02,902 Epoch: [9][2574/4426]	Eit 42400  lr 0.0005  Le 14.3713 (18.6017)	Time 0.815 (0.000)	Data 0.002 (0.000)	
2021-06-15 21:32:20,045 Epoch: [9][2774/4426]	Eit 42600  lr 0.0005  Le 23.6479 (18.6082)	Time 0.871 (0.000)	Data 0.002 (0.000)	
2021-06-15 21:35:36,924 Epoch: [9][2974/4426]	Eit 42800  lr 0.0005  Le 22.9612 (18.6113)	Time 1.057 (0.000)	Data 0.002 (0.000)	
2021-06-15 21:38:52,811 Epoch: [9][3174/4426]	Eit 43000  lr 0.0005  Le 19.2968 (18.6234)	Time 1.156 (0.000)	Data 0.002 (0.000)	
2021-06-15 21:42:08,023 Epoch: [9][3374/4426]	Eit 43200  lr 0.0005  Le 18.4528 (18.6303)	Time 1.031 (0.000)	Data 0.002 (0.000)	
2021-06-15 21:45:24,463 Epoch: [9][3574/4426]	Eit 43400  lr 0.0005  Le 17.5000 (18.6216)	Time 1.125 (0.000)	Data 0.009 (0.000)	
2021-06-15 21:48:43,189 Epoch: [9][3774/4426]	Eit 43600  lr 0.0005  Le 20.6292 (18.6264)	Time 0.880 (0.000)	Data 0.010 (0.000)	
2021-06-15 21:51:58,351 Epoch: [9][3974/4426]	Eit 43800  lr 0.0005  Le 16.4122 (18.6294)	Time 0.918 (0.000)	Data 0.002 (0.000)	
2021-06-15 21:55:15,226 Epoch: [9][4174/4426]	Eit 44000  lr 0.0005  Le 17.7923 (18.6352)	Time 1.391 (0.000)	Data 0.002 (0.000)	
2021-06-15 21:58:30,606 Epoch: [9][4374/4426]	Eit 44200  lr 0.0005  Le 17.0792 (18.6335)	Time 0.990 (0.000)	Data 0.002 (0.000)	
2021-06-15 21:59:29,111 Test: [0/40]	Le 60.3254 (60.3254)	Time 8.067 (0.000)	
2021-06-15 21:59:45,112 calculate similarity time: 0.06989049911499023
2021-06-15 21:59:45,601 Image to text: 80.6, 97.0, 99.2, 1.0, 2.1
2021-06-15 21:59:45,917 Text to image: 66.0, 92.3, 96.8, 1.0, 3.4
2021-06-15 21:59:45,917 Current rsum is 531.86
2021-06-15 21:59:49,553 runs/coco_butd_region_bert/log
2021-06-15 21:59:49,553 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-15 21:59:49,557 image encoder trainable parameters: 20490144
2021-06-15 21:59:49,570 txt encoder trainable parameters: 137319072
2021-06-15 22:02:24,840 Epoch: [10][149/4426]	Eit 44400  lr 0.0005  Le 15.1821 (18.1856)	Time 0.805 (0.000)	Data 0.002 (0.000)	
2021-06-15 22:05:41,337 Epoch: [10][349/4426]	Eit 44600  lr 0.0005  Le 22.4027 (17.9444)	Time 0.990 (0.000)	Data 0.002 (0.000)	
2021-06-15 22:08:58,519 Epoch: [10][549/4426]	Eit 44800  lr 0.0005  Le 17.4680 (17.9896)	Time 0.914 (0.000)	Data 0.002 (0.000)	
2021-06-15 22:12:14,957 Epoch: [10][749/4426]	Eit 45000  lr 0.0005  Le 20.9435 (17.9483)	Time 1.103 (0.000)	Data 0.002 (0.000)	
2021-06-15 22:15:31,030 Epoch: [10][949/4426]	Eit 45200  lr 0.0005  Le 20.3350 (17.9360)	Time 1.178 (0.000)	Data 0.002 (0.000)	
2021-06-15 22:18:47,250 Epoch: [10][1149/4426]	Eit 45400  lr 0.0005  Le 16.3029 (17.9216)	Time 0.791 (0.000)	Data 0.002 (0.000)	
2021-06-15 22:21:58,072 Epoch: [10][1349/4426]	Eit 45600  lr 0.0005  Le 18.1790 (17.9022)	Time 0.976 (0.000)	Data 0.002 (0.000)	
2021-06-15 22:25:08,443 Epoch: [10][1549/4426]	Eit 45800  lr 0.0005  Le 14.7757 (17.9250)	Time 0.805 (0.000)	Data 0.002 (0.000)	
2021-06-15 22:28:23,398 Epoch: [10][1749/4426]	Eit 46000  lr 0.0005  Le 15.6340 (17.9474)	Time 0.979 (0.000)	Data 0.002 (0.000)	
2021-06-15 22:31:38,919 Epoch: [10][1949/4426]	Eit 46200  lr 0.0005  Le 16.2323 (17.9802)	Time 1.254 (0.000)	Data 0.002 (0.000)	
2021-06-15 22:34:58,174 Epoch: [10][2149/4426]	Eit 46400  lr 0.0005  Le 17.3119 (18.0051)	Time 1.038 (0.000)	Data 0.003 (0.000)	
2021-06-15 22:38:17,630 Epoch: [10][2349/4426]	Eit 46600  lr 0.0005  Le 16.7978 (18.0393)	Time 1.113 (0.000)	Data 0.002 (0.000)	
2021-06-15 22:41:32,777 Epoch: [10][2549/4426]	Eit 46800  lr 0.0005  Le 17.8979 (18.0594)	Time 0.913 (0.000)	Data 0.002 (0.000)	
2021-06-15 22:44:49,942 Epoch: [10][2749/4426]	Eit 47000  lr 0.0005  Le 17.4615 (18.0629)	Time 0.858 (0.000)	Data 0.002 (0.000)	
2021-06-15 22:48:05,521 Epoch: [10][2949/4426]	Eit 47200  lr 0.0005  Le 18.9107 (18.0687)	Time 0.758 (0.000)	Data 0.003 (0.000)	
2021-06-15 22:51:22,851 Epoch: [10][3149/4426]	Eit 47400  lr 0.0005  Le 16.0846 (18.0807)	Time 1.153 (0.000)	Data 0.002 (0.000)	
2021-06-15 22:54:40,261 Epoch: [10][3349/4426]	Eit 47600  lr 0.0005  Le 14.8058 (18.0881)	Time 0.769 (0.000)	Data 0.002 (0.000)	
2021-06-15 22:57:59,367 Epoch: [10][3549/4426]	Eit 47800  lr 0.0005  Le 17.9907 (18.0870)	Time 1.007 (0.000)	Data 0.003 (0.000)	
2021-06-15 23:01:15,761 Epoch: [10][3749/4426]	Eit 48000  lr 0.0005  Le 16.8821 (18.1142)	Time 0.996 (0.000)	Data 0.002 (0.000)	
2021-06-15 23:04:30,480 Epoch: [10][3949/4426]	Eit 48200  lr 0.0005  Le 18.2437 (18.1258)	Time 1.186 (0.000)	Data 0.002 (0.000)	
2021-06-15 23:07:44,224 Epoch: [10][4149/4426]	Eit 48400  lr 0.0005  Le 19.6102 (18.1305)	Time 0.890 (0.000)	Data 0.002 (0.000)	
2021-06-15 23:11:02,988 Epoch: [10][4349/4426]	Eit 48600  lr 0.0005  Le 19.0272 (18.1383)	Time 0.934 (0.000)	Data 0.002 (0.000)	
2021-06-15 23:12:26,001 Test: [0/40]	Le 60.3316 (60.3316)	Time 8.155 (0.000)	
2021-06-15 23:12:42,134 calculate similarity time: 0.07404255867004395
2021-06-15 23:12:42,676 Image to text: 81.1, 97.8, 99.6, 1.0, 1.7
2021-06-15 23:12:43,001 Text to image: 65.5, 92.8, 97.0, 1.0, 3.2
2021-06-15 23:12:43,001 Current rsum is 533.74
2021-06-15 23:12:46,597 runs/coco_butd_region_bert/log
2021-06-15 23:12:46,597 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-15 23:12:46,600 image encoder trainable parameters: 20490144
2021-06-15 23:12:46,612 txt encoder trainable parameters: 137319072
2021-06-15 23:14:58,139 Epoch: [11][124/4426]	Eit 48800  lr 0.0005  Le 19.3503 (17.1803)	Time 0.893 (0.000)	Data 0.003 (0.000)	
2021-06-15 23:18:16,189 Epoch: [11][324/4426]	Eit 49000  lr 0.0005  Le 18.5612 (17.2857)	Time 1.157 (0.000)	Data 0.002 (0.000)	
2021-06-15 23:21:31,887 Epoch: [11][524/4426]	Eit 49200  lr 0.0005  Le 16.6286 (17.4753)	Time 1.056 (0.000)	Data 0.002 (0.000)	
2021-06-15 23:24:49,214 Epoch: [11][724/4426]	Eit 49400  lr 0.0005  Le 14.8243 (17.5027)	Time 0.997 (0.000)	Data 0.002 (0.000)	
2021-06-15 23:28:06,477 Epoch: [11][924/4426]	Eit 49600  lr 0.0005  Le 15.2216 (17.4992)	Time 0.904 (0.000)	Data 0.002 (0.000)	
2021-06-15 23:31:23,280 Epoch: [11][1124/4426]	Eit 49800  lr 0.0005  Le 14.0311 (17.5275)	Time 1.042 (0.000)	Data 0.002 (0.000)	
2021-06-15 23:34:39,686 Epoch: [11][1324/4426]	Eit 50000  lr 0.0005  Le 19.1043 (17.5510)	Time 0.947 (0.000)	Data 0.002 (0.000)	
2021-06-15 23:37:55,604 Epoch: [11][1524/4426]	Eit 50200  lr 0.0005  Le 19.2978 (17.5869)	Time 0.996 (0.000)	Data 0.002 (0.000)	
2021-06-15 23:41:11,882 Epoch: [11][1724/4426]	Eit 50400  lr 0.0005  Le 15.1100 (17.6247)	Time 0.809 (0.000)	Data 0.002 (0.000)	
2021-06-15 23:44:27,091 Epoch: [11][1924/4426]	Eit 50600  lr 0.0005  Le 15.4167 (17.6359)	Time 0.954 (0.000)	Data 0.002 (0.000)	
2021-06-15 23:47:41,211 Epoch: [11][2124/4426]	Eit 50800  lr 0.0005  Le 18.4753 (17.6551)	Time 0.765 (0.000)	Data 0.002 (0.000)	
2021-06-15 23:51:02,242 Epoch: [11][2324/4426]	Eit 51000  lr 0.0005  Le 16.9331 (17.6795)	Time 0.935 (0.000)	Data 0.002 (0.000)	
2021-06-15 23:54:20,963 Epoch: [11][2524/4426]	Eit 51200  lr 0.0005  Le 21.3086 (17.6641)	Time 1.152 (0.000)	Data 0.002 (0.000)	
2021-06-15 23:57:38,367 Epoch: [11][2724/4426]	Eit 51400  lr 0.0005  Le 19.2679 (17.6512)	Time 1.084 (0.000)	Data 0.002 (0.000)	
2021-06-16 00:00:54,633 Epoch: [11][2924/4426]	Eit 51600  lr 0.0005  Le 16.1373 (17.6544)	Time 1.111 (0.000)	Data 0.002 (0.000)	
2021-06-16 00:04:10,666 Epoch: [11][3124/4426]	Eit 51800  lr 0.0005  Le 18.0431 (17.6573)	Time 1.016 (0.000)	Data 0.002 (0.000)	
2021-06-16 00:07:27,288 Epoch: [11][3324/4426]	Eit 52000  lr 0.0005  Le 15.8360 (17.6599)	Time 1.021 (0.000)	Data 0.002 (0.000)	
2021-06-16 00:10:47,258 Epoch: [11][3524/4426]	Eit 52200  lr 0.0005  Le 18.9084 (17.6558)	Time 1.059 (0.000)	Data 0.002 (0.000)	
2021-06-16 00:14:04,867 Epoch: [11][3724/4426]	Eit 52400  lr 0.0005  Le 17.8544 (17.6633)	Time 1.041 (0.000)	Data 0.002 (0.000)	
2021-06-16 00:17:21,722 Epoch: [11][3924/4426]	Eit 52600  lr 0.0005  Le 21.9309 (17.6728)	Time 1.313 (0.000)	Data 0.002 (0.000)	
2021-06-16 00:20:38,086 Epoch: [11][4124/4426]	Eit 52800  lr 0.0005  Le 16.9445 (17.6776)	Time 1.073 (0.000)	Data 0.002 (0.000)	
2021-06-16 00:23:55,118 Epoch: [11][4324/4426]	Eit 53000  lr 0.0005  Le 19.2007 (17.6857)	Time 0.883 (0.000)	Data 0.002 (0.000)	
2021-06-16 00:25:41,815 Test: [0/40]	Le 59.5112 (59.5112)	Time 8.062 (0.000)	
2021-06-16 00:25:58,290 calculate similarity time: 0.06731867790222168
2021-06-16 00:25:58,746 Image to text: 79.7, 97.8, 99.2, 1.0, 1.8
2021-06-16 00:25:59,058 Text to image: 66.3, 92.1, 96.9, 1.0, 3.3
2021-06-16 00:25:59,058 Current rsum is 531.98
2021-06-16 00:26:00,536 runs/coco_butd_region_bert/log
2021-06-16 00:26:00,536 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-16 00:26:00,537 image encoder trainable parameters: 20490144
2021-06-16 00:26:00,543 txt encoder trainable parameters: 137319072
2021-06-16 00:27:48,053 Epoch: [12][99/4426]	Eit 53200  lr 0.0005  Le 17.1394 (16.9689)	Time 1.129 (0.000)	Data 0.002 (0.000)	
2021-06-16 00:30:58,975 Epoch: [12][299/4426]	Eit 53400  lr 0.0005  Le 20.8441 (16.8487)	Time 0.906 (0.000)	Data 0.002 (0.000)	
2021-06-16 00:34:09,057 Epoch: [12][499/4426]	Eit 53600  lr 0.0005  Le 19.4962 (16.8071)	Time 0.976 (0.000)	Data 0.002 (0.000)	
2021-06-16 00:37:25,882 Epoch: [12][699/4426]	Eit 53800  lr 0.0005  Le 15.6773 (16.9550)	Time 0.783 (0.000)	Data 0.002 (0.000)	
2021-06-16 00:40:39,495 Epoch: [12][899/4426]	Eit 54000  lr 0.0005  Le 23.0581 (17.0176)	Time 1.022 (0.000)	Data 0.002 (0.000)	
2021-06-16 00:43:55,680 Epoch: [12][1099/4426]	Eit 54200  lr 0.0005  Le 17.3701 (17.0557)	Time 1.025 (0.000)	Data 0.002 (0.000)	
2021-06-16 00:47:10,781 Epoch: [12][1299/4426]	Eit 54400  lr 0.0005  Le 19.0060 (17.0991)	Time 0.997 (0.000)	Data 0.003 (0.000)	
2021-06-16 00:50:27,930 Epoch: [12][1499/4426]	Eit 54600  lr 0.0005  Le 16.9750 (17.1030)	Time 0.805 (0.000)	Data 0.002 (0.000)	
2021-06-16 00:53:44,796 Epoch: [12][1699/4426]	Eit 54800  lr 0.0005  Le 17.0179 (17.1140)	Time 0.982 (0.000)	Data 0.002 (0.000)	
2021-06-16 00:57:01,560 Epoch: [12][1899/4426]	Eit 55000  lr 0.0005  Le 16.2318 (17.1562)	Time 0.894 (0.000)	Data 0.002 (0.000)	
2021-06-16 01:00:20,016 Epoch: [12][2099/4426]	Eit 55200  lr 0.0005  Le 11.6424 (17.1766)	Time 0.839 (0.000)	Data 0.002 (0.000)	
2021-06-16 01:03:37,156 Epoch: [12][2299/4426]	Eit 55400  lr 0.0005  Le 15.1078 (17.1725)	Time 0.890 (0.000)	Data 0.002 (0.000)	
2021-06-16 01:06:52,150 Epoch: [12][2499/4426]	Eit 55600  lr 0.0005  Le 19.7261 (17.1859)	Time 0.817 (0.000)	Data 0.003 (0.000)	
2021-06-16 01:10:05,459 Epoch: [12][2699/4426]	Eit 55800  lr 0.0005  Le 15.4858 (17.2203)	Time 0.980 (0.000)	Data 0.002 (0.000)	
2021-06-16 01:13:23,785 Epoch: [12][2899/4426]	Eit 56000  lr 0.0005  Le 18.7738 (17.2309)	Time 0.788 (0.000)	Data 0.002 (0.000)	
2021-06-16 01:16:39,165 Epoch: [12][3099/4426]	Eit 56200  lr 0.0005  Le 15.8707 (17.2438)	Time 0.861 (0.000)	Data 0.002 (0.000)	
2021-06-16 01:19:56,025 Epoch: [12][3299/4426]	Eit 56400  lr 0.0005  Le 14.3142 (17.2509)	Time 1.025 (0.000)	Data 0.002 (0.000)	
2021-06-16 01:23:13,296 Epoch: [12][3499/4426]	Eit 56600  lr 0.0005  Le 16.4669 (17.2580)	Time 0.927 (0.000)	Data 0.002 (0.000)	
2021-06-16 01:26:29,862 Epoch: [12][3699/4426]	Eit 56800  lr 0.0005  Le 17.1426 (17.2655)	Time 1.045 (0.000)	Data 0.002 (0.000)	
2021-06-16 01:29:48,300 Epoch: [12][3899/4426]	Eit 57000  lr 0.0005  Le 17.4796 (17.2620)	Time 1.041 (0.000)	Data 0.002 (0.000)	
2021-06-16 01:33:03,764 Epoch: [12][4099/4426]	Eit 57200  lr 0.0005  Le 16.0264 (17.2573)	Time 0.993 (0.000)	Data 0.002 (0.000)	
2021-06-16 01:36:20,115 Epoch: [12][4299/4426]	Eit 57400  lr 0.0005  Le 16.2906 (17.2550)	Time 1.225 (0.000)	Data 0.002 (0.000)	
2021-06-16 01:38:30,730 Test: [0/40]	Le 59.4512 (59.4511)	Time 8.227 (0.000)	
2021-06-16 01:38:46,391 calculate similarity time: 0.06939935684204102
2021-06-16 01:38:46,929 Image to text: 82.4, 98.0, 99.4, 1.0, 2.3
2021-06-16 01:38:47,361 Text to image: 66.2, 92.2, 96.9, 1.0, 3.6
2021-06-16 01:38:47,361 Current rsum is 535.1800000000001
2021-06-16 01:38:51,005 runs/coco_butd_region_bert/log
2021-06-16 01:38:51,006 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-16 01:38:51,008 image encoder trainable parameters: 20490144
2021-06-16 01:38:51,019 txt encoder trainable parameters: 137319072
2021-06-16 01:40:12,074 Epoch: [13][74/4426]	Eit 57600  lr 0.0005  Le 16.7471 (16.9306)	Time 1.151 (0.000)	Data 0.002 (0.000)	
2021-06-16 01:43:30,848 Epoch: [13][274/4426]	Eit 57800  lr 0.0005  Le 18.0773 (16.5027)	Time 1.044 (0.000)	Data 0.002 (0.000)	
2021-06-16 01:46:48,938 Epoch: [13][474/4426]	Eit 58000  lr 0.0005  Le 15.3501 (16.5837)	Time 0.910 (0.000)	Data 0.002 (0.000)	
2021-06-16 01:50:02,924 Epoch: [13][674/4426]	Eit 58200  lr 0.0005  Le 12.9612 (16.6130)	Time 1.001 (0.000)	Data 0.002 (0.000)	
2021-06-16 01:53:21,090 Epoch: [13][874/4426]	Eit 58400  lr 0.0005  Le 16.2188 (16.6698)	Time 1.029 (0.000)	Data 0.002 (0.000)	
2021-06-16 01:56:39,291 Epoch: [13][1074/4426]	Eit 58600  lr 0.0005  Le 17.5076 (16.6622)	Time 1.130 (0.000)	Data 0.002 (0.000)	
2021-06-16 01:59:57,479 Epoch: [13][1274/4426]	Eit 58800  lr 0.0005  Le 21.3751 (16.6932)	Time 0.959 (0.000)	Data 0.002 (0.000)	
2021-06-16 02:03:13,535 Epoch: [13][1474/4426]	Eit 59000  lr 0.0005  Le 17.2210 (16.7571)	Time 0.999 (0.000)	Data 0.002 (0.000)	
2021-06-16 02:06:29,273 Epoch: [13][1674/4426]	Eit 59200  lr 0.0005  Le 14.8651 (16.7407)	Time 1.064 (0.000)	Data 0.002 (0.000)	
2021-06-16 02:09:46,043 Epoch: [13][1874/4426]	Eit 59400  lr 0.0005  Le 19.4912 (16.7907)	Time 1.229 (0.000)	Data 0.002 (0.000)	
2021-06-16 02:13:03,705 Epoch: [13][2074/4426]	Eit 59600  lr 0.0005  Le 13.2169 (16.7992)	Time 0.981 (0.000)	Data 0.002 (0.000)	
2021-06-16 02:16:19,981 Epoch: [13][2274/4426]	Eit 59800  lr 0.0005  Le 21.3247 (16.7846)	Time 0.940 (0.000)	Data 0.002 (0.000)	
2021-06-16 02:19:36,090 Epoch: [13][2474/4426]	Eit 60000  lr 0.0005  Le 15.8187 (16.7773)	Time 0.867 (0.000)	Data 0.002 (0.000)	
2021-06-16 02:22:55,410 Epoch: [13][2674/4426]	Eit 60200  lr 0.0005  Le 16.1394 (16.7797)	Time 1.068 (0.000)	Data 0.002 (0.000)	
2021-06-16 02:26:07,963 Epoch: [13][2874/4426]	Eit 60400  lr 0.0005  Le 19.2418 (16.7727)	Time 1.134 (0.000)	Data 0.002 (0.000)	
2021-06-16 02:29:26,457 Epoch: [13][3074/4426]	Eit 60600  lr 0.0005  Le 14.0026 (16.7919)	Time 0.866 (0.000)	Data 0.002 (0.000)	
2021-06-16 02:32:44,102 Epoch: [13][3274/4426]	Eit 60800  lr 0.0005  Le 16.6467 (16.7877)	Time 0.963 (0.000)	Data 0.002 (0.000)	
2021-06-16 02:35:56,696 Epoch: [13][3474/4426]	Eit 61000  lr 0.0005  Le 16.8898 (16.7927)	Time 1.018 (0.000)	Data 0.002 (0.000)	
2021-06-16 02:39:16,204 Epoch: [13][3674/4426]	Eit 61200  lr 0.0005  Le 17.7560 (16.7908)	Time 0.973 (0.000)	Data 0.004 (0.000)	
2021-06-16 02:42:20,485 Epoch: [13][3874/4426]	Eit 61400  lr 0.0005  Le 14.6902 (16.7993)	Time 0.922 (0.000)	Data 0.002 (0.000)	
2021-06-16 02:45:35,962 Epoch: [13][4074/4426]	Eit 61600  lr 0.0005  Le 13.8630 (16.7996)	Time 1.101 (0.000)	Data 0.003 (0.000)	
2021-06-16 02:48:53,254 Epoch: [13][4274/4426]	Eit 61800  lr 0.0005  Le 14.8145 (16.7974)	Time 1.034 (0.000)	Data 0.003 (0.000)	
2021-06-16 02:51:31,598 Test: [0/40]	Le 59.9556 (59.9555)	Time 8.077 (0.000)	
2021-06-16 02:51:47,528 calculate similarity time: 0.06969904899597168
2021-06-16 02:51:47,958 Image to text: 80.7, 97.4, 99.3, 1.0, 2.0
2021-06-16 02:51:48,284 Text to image: 65.5, 92.0, 96.7, 1.0, 3.6
2021-06-16 02:51:48,285 Current rsum is 531.58
2021-06-16 02:51:49,754 runs/coco_butd_region_bert/log
2021-06-16 02:51:49,754 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-16 02:51:49,756 image encoder trainable parameters: 20490144
2021-06-16 02:51:49,762 txt encoder trainable parameters: 137319072
2021-06-16 02:52:47,449 Epoch: [14][49/4426]	Eit 62000  lr 0.0005  Le 15.5441 (16.2319)	Time 1.054 (0.000)	Data 0.002 (0.000)	
2021-06-16 02:56:04,977 Epoch: [14][249/4426]	Eit 62200  lr 0.0005  Le 12.2362 (16.0580)	Time 0.857 (0.000)	Data 0.002 (0.000)	
2021-06-16 02:59:22,717 Epoch: [14][449/4426]	Eit 62400  lr 0.0005  Le 16.9161 (15.8056)	Time 0.869 (0.000)	Data 0.002 (0.000)	
2021-06-16 03:02:39,439 Epoch: [14][649/4426]	Eit 62600  lr 0.0005  Le 18.3115 (15.8030)	Time 0.960 (0.000)	Data 0.002 (0.000)	
2021-06-16 03:05:55,583 Epoch: [14][849/4426]	Eit 62800  lr 0.0005  Le 10.6916 (15.7754)	Time 0.800 (0.000)	Data 0.002 (0.000)	
2021-06-16 03:09:11,669 Epoch: [14][1049/4426]	Eit 63000  lr 0.0005  Le 14.6980 (15.7952)	Time 0.820 (0.000)	Data 0.002 (0.000)	
2021-06-16 03:12:24,840 Epoch: [14][1249/4426]	Eit 63200  lr 0.0005  Le 13.1644 (15.8087)	Time 1.104 (0.000)	Data 0.002 (0.000)	
2021-06-16 03:15:42,007 Epoch: [14][1449/4426]	Eit 63400  lr 0.0005  Le 19.0304 (15.8527)	Time 0.859 (0.000)	Data 0.002 (0.000)	
2021-06-16 03:19:00,200 Epoch: [14][1649/4426]	Eit 63600  lr 0.0005  Le 12.9712 (15.8709)	Time 0.788 (0.000)	Data 0.002 (0.000)	
2021-06-16 03:22:18,011 Epoch: [14][1849/4426]	Eit 63800  lr 0.0005  Le 18.7240 (15.9055)	Time 1.011 (0.000)	Data 0.003 (0.000)	
2021-06-16 03:25:34,774 Epoch: [14][2049/4426]	Eit 64000  lr 0.0005  Le 17.8684 (15.9461)	Time 0.857 (0.000)	Data 0.008 (0.000)	
2021-06-16 03:28:53,521 Epoch: [14][2249/4426]	Eit 64200  lr 0.0005  Le 14.1470 (15.9595)	Time 1.067 (0.000)	Data 0.002 (0.000)	
2021-06-16 03:32:08,834 Epoch: [14][2449/4426]	Eit 64400  lr 0.0005  Le 16.5738 (15.9805)	Time 0.908 (0.000)	Data 0.002 (0.000)	
2021-06-16 03:35:27,337 Epoch: [14][2649/4426]	Eit 64600  lr 0.0005  Le 13.0979 (15.9842)	Time 1.093 (0.000)	Data 0.002 (0.000)	
2021-06-16 03:38:44,108 Epoch: [14][2849/4426]	Eit 64800  lr 0.0005  Le 14.1087 (15.9832)	Time 0.824 (0.000)	Data 0.002 (0.000)	
2021-06-16 03:42:02,772 Epoch: [14][3049/4426]	Eit 65000  lr 0.0005  Le 16.7274 (16.0149)	Time 0.898 (0.000)	Data 0.002 (0.000)	
2021-06-16 03:45:16,505 Epoch: [14][3249/4426]	Eit 65200  lr 0.0005  Le 11.5800 (16.0101)	Time 1.014 (0.000)	Data 0.002 (0.000)	
2021-06-16 03:48:33,444 Epoch: [14][3449/4426]	Eit 65400  lr 0.0005  Le 17.5630 (16.0254)	Time 0.931 (0.000)	Data 0.002 (0.000)	
2021-06-16 03:51:49,358 Epoch: [14][3649/4426]	Eit 65600  lr 0.0005  Le 16.5589 (16.0516)	Time 0.809 (0.000)	Data 0.002 (0.000)	
2021-06-16 03:55:07,171 Epoch: [14][3849/4426]	Eit 65800  lr 0.0005  Le 15.7704 (16.0438)	Time 0.805 (0.000)	Data 0.002 (0.000)	
2021-06-16 03:58:22,947 Epoch: [14][4049/4426]	Eit 66000  lr 0.0005  Le 15.9499 (16.0375)	Time 1.090 (0.000)	Data 0.002 (0.000)	
2021-06-16 04:01:39,751 Epoch: [14][4249/4426]	Eit 66200  lr 0.0005  Le 13.5377 (16.0325)	Time 0.782 (0.000)	Data 0.002 (0.000)	
2021-06-16 04:04:40,767 Test: [0/40]	Le 61.0171 (61.0171)	Time 8.158 (0.000)	
2021-06-16 04:04:56,772 calculate similarity time: 0.06688261032104492
2021-06-16 04:04:57,305 Image to text: 79.2, 97.9, 99.4, 1.0, 2.8
2021-06-16 04:04:57,616 Text to image: 66.1, 92.1, 96.6, 1.0, 3.8
2021-06-16 04:04:57,616 Current rsum is 531.2
2021-06-16 04:04:59,051 runs/coco_butd_region_bert/log
2021-06-16 04:04:59,051 runs/coco_butd_region_bert
2021-06-16 04:04:59,051 Current epoch num is 15, decrease all lr by 10
2021-06-16 04:04:59,051 new lr 5e-05
2021-06-16 04:04:59,051 new lr 5e-06
2021-06-16 04:04:59,052 new lr 5e-05
Use VSE++ objective.
2021-06-16 04:04:59,053 image encoder trainable parameters: 20490144
2021-06-16 04:04:59,058 txt encoder trainable parameters: 137319072
2021-06-16 04:05:31,299 Epoch: [15][24/4426]	Eit 66400  lr 5e-05  Le 10.6935 (14.5046)	Time 0.798 (0.000)	Data 0.002 (0.000)	
2021-06-16 04:08:45,549 Epoch: [15][224/4426]	Eit 66600  lr 5e-05  Le 12.7105 (14.9938)	Time 0.992 (0.000)	Data 0.002 (0.000)	
2021-06-16 04:11:57,969 Epoch: [15][424/4426]	Eit 66800  lr 5e-05  Le 12.6591 (14.6014)	Time 1.216 (0.000)	Data 0.002 (0.000)	
2021-06-16 04:15:15,297 Epoch: [15][624/4426]	Eit 67000  lr 5e-05  Le 12.2759 (14.6246)	Time 1.013 (0.000)	Data 0.002 (0.000)	
2021-06-16 04:18:28,973 Epoch: [15][824/4426]	Eit 67200  lr 5e-05  Le 17.2083 (14.5524)	Time 0.877 (0.000)	Data 0.002 (0.000)	
2021-06-16 04:21:48,451 Epoch: [15][1024/4426]	Eit 67400  lr 5e-05  Le 17.5076 (14.4833)	Time 0.800 (0.000)	Data 0.002 (0.000)	
2021-06-16 04:25:02,635 Epoch: [15][1224/4426]	Eit 67600  lr 5e-05  Le 9.5559 (14.4253)	Time 0.825 (0.000)	Data 0.002 (0.000)	
2021-06-16 04:28:19,533 Epoch: [15][1424/4426]	Eit 67800  lr 5e-05  Le 11.5305 (14.3746)	Time 0.978 (0.000)	Data 0.002 (0.000)	
2021-06-16 04:31:34,240 Epoch: [15][1624/4426]	Eit 68000  lr 5e-05  Le 15.3235 (14.3580)	Time 1.000 (0.000)	Data 0.003 (0.000)	
2021-06-16 04:34:51,481 Epoch: [15][1824/4426]	Eit 68200  lr 5e-05  Le 17.8769 (14.3245)	Time 0.871 (0.000)	Data 0.002 (0.000)	
2021-06-16 04:38:08,623 Epoch: [15][2024/4426]	Eit 68400  lr 5e-05  Le 12.0895 (14.2789)	Time 1.011 (0.000)	Data 0.002 (0.000)	
2021-06-16 04:41:27,708 Epoch: [15][2224/4426]	Eit 68600  lr 5e-05  Le 15.7245 (14.2537)	Time 0.891 (0.000)	Data 0.002 (0.000)	
2021-06-16 04:44:46,738 Epoch: [15][2424/4426]	Eit 68800  lr 5e-05  Le 12.5835 (14.2268)	Time 0.986 (0.000)	Data 0.002 (0.000)	
2021-06-16 04:48:01,947 Epoch: [15][2624/4426]	Eit 69000  lr 5e-05  Le 17.1694 (14.1938)	Time 0.723 (0.000)	Data 0.002 (0.000)	
2021-06-16 04:51:06,815 Epoch: [15][2824/4426]	Eit 69200  lr 5e-05  Le 9.2956 (14.1652)	Time 1.190 (0.000)	Data 0.003 (0.000)	
2021-06-16 04:54:20,213 Epoch: [15][3024/4426]	Eit 69400  lr 5e-05  Le 14.1281 (14.1545)	Time 1.134 (0.000)	Data 0.002 (0.000)	
2021-06-16 04:57:35,568 Epoch: [15][3224/4426]	Eit 69600  lr 5e-05  Le 16.0958 (14.1328)	Time 0.867 (0.000)	Data 0.002 (0.000)	
2021-06-16 05:00:50,002 Epoch: [15][3424/4426]	Eit 69800  lr 5e-05  Le 18.6633 (14.1257)	Time 0.743 (0.000)	Data 0.002 (0.000)	
2021-06-16 05:04:05,952 Epoch: [15][3624/4426]	Eit 70000  lr 5e-05  Le 16.5058 (14.1255)	Time 1.102 (0.000)	Data 0.002 (0.000)	
2021-06-16 05:07:21,108 Epoch: [15][3824/4426]	Eit 70200  lr 5e-05  Le 10.8861 (14.1113)	Time 1.040 (0.000)	Data 0.002 (0.000)	
2021-06-16 05:10:38,829 Epoch: [15][4024/4426]	Eit 70400  lr 5e-05  Le 14.4258 (14.0927)	Time 0.973 (0.000)	Data 0.003 (0.000)	
2021-06-16 05:13:58,093 Epoch: [15][4224/4426]	Eit 70600  lr 5e-05  Le 17.9600 (14.0827)	Time 0.860 (0.000)	Data 0.002 (0.000)	
2021-06-16 05:17:13,953 Epoch: [15][4424/4426]	Eit 70800  lr 5e-05  Le 11.0862 (14.0711)	Time 1.260 (0.000)	Data 0.002 (0.000)	
2021-06-16 05:17:23,424 Test: [0/40]	Le 60.4284 (60.4283)	Time 8.189 (0.000)	
2021-06-16 05:17:39,189 calculate similarity time: 0.05425071716308594
2021-06-16 05:17:39,675 Image to text: 82.4, 97.9, 99.5, 1.0, 2.0
2021-06-16 05:17:39,996 Text to image: 67.0, 92.9, 96.9, 1.0, 3.7
2021-06-16 05:17:39,996 Current rsum is 536.56
2021-06-16 05:17:43,549 runs/coco_butd_region_bert/log
2021-06-16 05:17:43,550 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-16 05:17:43,553 image encoder trainable parameters: 20490144
2021-06-16 05:17:43,565 txt encoder trainable parameters: 137319072
2021-06-16 05:21:06,072 Epoch: [16][199/4426]	Eit 71000  lr 5e-05  Le 18.1392 (13.6564)	Time 0.981 (0.000)	Data 0.002 (0.000)	
2021-06-16 05:24:20,668 Epoch: [16][399/4426]	Eit 71200  lr 5e-05  Le 10.0956 (13.5598)	Time 0.997 (0.000)	Data 0.002 (0.000)	
2021-06-16 05:27:36,327 Epoch: [16][599/4426]	Eit 71400  lr 5e-05  Le 16.7481 (13.4381)	Time 0.741 (0.000)	Data 0.002 (0.000)	
2021-06-16 05:30:54,351 Epoch: [16][799/4426]	Eit 71600  lr 5e-05  Le 12.6329 (13.5509)	Time 0.810 (0.000)	Data 0.002 (0.000)	
2021-06-16 05:34:10,956 Epoch: [16][999/4426]	Eit 71800  lr 5e-05  Le 14.7911 (13.4941)	Time 1.057 (0.000)	Data 0.002 (0.000)	
2021-06-16 05:37:24,439 Epoch: [16][1199/4426]	Eit 72000  lr 5e-05  Le 15.8955 (13.4324)	Time 0.757 (0.000)	Data 0.002 (0.000)	
2021-06-16 05:40:40,066 Epoch: [16][1399/4426]	Eit 72200  lr 5e-05  Le 14.0380 (13.4382)	Time 1.152 (0.000)	Data 0.002 (0.000)	
2021-06-16 05:43:57,001 Epoch: [16][1599/4426]	Eit 72400  lr 5e-05  Le 12.7458 (13.4726)	Time 1.023 (0.000)	Data 0.002 (0.000)	
2021-06-16 05:47:12,486 Epoch: [16][1799/4426]	Eit 72600  lr 5e-05  Le 14.4707 (13.4694)	Time 0.871 (0.000)	Data 0.002 (0.000)	
2021-06-16 05:50:29,498 Epoch: [16][1999/4426]	Eit 72800  lr 5e-05  Le 14.4748 (13.4849)	Time 1.122 (0.000)	Data 0.002 (0.000)	
2021-06-16 05:53:46,159 Epoch: [16][2199/4426]	Eit 73000  lr 5e-05  Le 13.3148 (13.4906)	Time 0.826 (0.000)	Data 0.002 (0.000)	
2021-06-16 05:57:00,934 Epoch: [16][2399/4426]	Eit 73200  lr 5e-05  Le 9.1126 (13.4772)	Time 0.805 (0.000)	Data 0.002 (0.000)	
2021-06-16 06:00:19,319 Epoch: [16][2599/4426]	Eit 73400  lr 5e-05  Le 11.6769 (13.4509)	Time 0.971 (0.000)	Data 0.002 (0.000)	
2021-06-16 06:03:37,881 Epoch: [16][2799/4426]	Eit 73600  lr 5e-05  Le 8.6506 (13.4358)	Time 0.790 (0.000)	Data 0.002 (0.000)	
2021-06-16 06:06:52,709 Epoch: [16][2999/4426]	Eit 73800  lr 5e-05  Le 12.2246 (13.4331)	Time 0.770 (0.000)	Data 0.002 (0.000)	
2021-06-16 06:10:09,106 Epoch: [16][3199/4426]	Eit 74000  lr 5e-05  Le 15.5082 (13.4327)	Time 0.852 (0.000)	Data 0.003 (0.000)	
2021-06-16 06:13:23,727 Epoch: [16][3399/4426]	Eit 74200  lr 5e-05  Le 11.5609 (13.4411)	Time 1.018 (0.000)	Data 0.002 (0.000)	
2021-06-16 06:16:40,768 Epoch: [16][3599/4426]	Eit 74400  lr 5e-05  Le 11.6530 (13.4468)	Time 1.076 (0.000)	Data 0.002 (0.000)	
2021-06-16 06:19:56,743 Epoch: [16][3799/4426]	Eit 74600  lr 5e-05  Le 12.1302 (13.4426)	Time 0.799 (0.000)	Data 0.002 (0.000)	
2021-06-16 06:23:15,524 Epoch: [16][3999/4426]	Eit 74800  lr 5e-05  Le 18.4246 (13.4244)	Time 0.984 (0.000)	Data 0.002 (0.000)	
2021-06-16 06:26:36,024 Epoch: [16][4199/4426]	Eit 75000  lr 5e-05  Le 18.3774 (13.4457)	Time 0.836 (0.000)	Data 0.002 (0.000)	
2021-06-16 06:29:50,959 Epoch: [16][4399/4426]	Eit 75200  lr 5e-05  Le 14.7664 (13.4330)	Time 0.988 (0.000)	Data 0.002 (0.000)	
2021-06-16 06:30:23,853 Test: [0/40]	Le 60.1726 (60.1726)	Time 7.619 (0.000)	
2021-06-16 06:30:39,598 calculate similarity time: 0.07753300666809082
2021-06-16 06:30:39,981 Image to text: 82.9, 98.0, 99.5, 1.0, 2.1
2021-06-16 06:30:40,292 Text to image: 67.3, 92.8, 96.7, 1.0, 3.7
2021-06-16 06:30:40,292 Current rsum is 537.26
2021-06-16 06:30:43,946 runs/coco_butd_region_bert/log
2021-06-16 06:30:43,946 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-16 06:30:43,950 image encoder trainable parameters: 20490144
2021-06-16 06:30:43,962 txt encoder trainable parameters: 137319072
2021-06-16 06:33:41,277 Epoch: [17][174/4426]	Eit 75400  lr 5e-05  Le 10.7778 (12.8530)	Time 0.946 (0.000)	Data 0.002 (0.000)	
2021-06-16 06:36:57,500 Epoch: [17][374/4426]	Eit 75600  lr 5e-05  Le 12.8510 (12.8959)	Time 0.933 (0.000)	Data 0.002 (0.000)	
2021-06-16 06:40:13,285 Epoch: [17][574/4426]	Eit 75800  lr 5e-05  Le 9.1883 (12.9869)	Time 1.027 (0.000)	Data 0.002 (0.000)	
2021-06-16 06:43:29,124 Epoch: [17][774/4426]	Eit 76000  lr 5e-05  Le 16.5349 (12.9965)	Time 0.818 (0.000)	Data 0.002 (0.000)	
2021-06-16 06:46:46,162 Epoch: [17][974/4426]	Eit 76200  lr 5e-05  Le 15.0587 (12.9741)	Time 1.153 (0.000)	Data 0.002 (0.000)	
2021-06-16 06:50:03,739 Epoch: [17][1174/4426]	Eit 76400  lr 5e-05  Le 9.7396 (13.0059)	Time 1.101 (0.000)	Data 0.002 (0.000)	
2021-06-16 06:53:21,301 Epoch: [17][1374/4426]	Eit 76600  lr 5e-05  Le 20.0457 (13.0858)	Time 1.056 (0.000)	Data 0.002 (0.000)	
2021-06-16 06:56:37,039 Epoch: [17][1574/4426]	Eit 76800  lr 5e-05  Le 16.0123 (13.0791)	Time 1.021 (0.000)	Data 0.002 (0.000)	
2021-06-16 06:59:40,387 Epoch: [17][1774/4426]	Eit 77000  lr 5e-05  Le 12.2900 (13.1042)	Time 1.019 (0.000)	Data 0.002 (0.000)	
2021-06-16 07:02:52,605 Epoch: [17][1974/4426]	Eit 77200  lr 5e-05  Le 19.7382 (13.1203)	Time 0.970 (0.000)	Data 0.002 (0.000)	
2021-06-16 07:06:07,221 Epoch: [17][2174/4426]	Eit 77400  lr 5e-05  Le 10.1750 (13.1147)	Time 0.858 (0.000)	Data 0.002 (0.000)	
2021-06-16 07:09:26,519 Epoch: [17][2374/4426]	Eit 77600  lr 5e-05  Le 13.6709 (13.1141)	Time 1.216 (0.000)	Data 0.002 (0.000)	
2021-06-16 07:12:41,340 Epoch: [17][2574/4426]	Eit 77800  lr 5e-05  Le 17.9372 (13.1580)	Time 1.040 (0.000)	Data 0.002 (0.000)	
2021-06-16 07:15:58,453 Epoch: [17][2774/4426]	Eit 78000  lr 5e-05  Le 12.9302 (13.1400)	Time 1.215 (0.000)	Data 0.003 (0.000)	
2021-06-16 07:19:13,014 Epoch: [17][2974/4426]	Eit 78200  lr 5e-05  Le 13.5311 (13.1035)	Time 0.812 (0.000)	Data 0.002 (0.000)	
2021-06-16 07:22:30,968 Epoch: [17][3174/4426]	Eit 78400  lr 5e-05  Le 12.5427 (13.1124)	Time 1.001 (0.000)	Data 0.002 (0.000)	
2021-06-16 07:25:47,074 Epoch: [17][3374/4426]	Eit 78600  lr 5e-05  Le 10.6937 (13.1238)	Time 1.180 (0.000)	Data 0.002 (0.000)	
2021-06-16 07:29:05,186 Epoch: [17][3574/4426]	Eit 78800  lr 5e-05  Le 8.6441 (13.0998)	Time 0.772 (0.000)	Data 0.002 (0.000)	
2021-06-16 07:32:19,081 Epoch: [17][3774/4426]	Eit 79000  lr 5e-05  Le 9.8750 (13.0951)	Time 1.141 (0.000)	Data 0.002 (0.000)	
2021-06-16 07:35:36,218 Epoch: [17][3974/4426]	Eit 79200  lr 5e-05  Le 11.7654 (13.0987)	Time 0.923 (0.000)	Data 0.002 (0.000)	
2021-06-16 07:38:54,842 Epoch: [17][4174/4426]	Eit 79400  lr 5e-05  Le 9.9073 (13.0930)	Time 1.309 (0.000)	Data 0.002 (0.000)	
2021-06-16 07:42:10,952 Epoch: [17][4374/4426]	Eit 79600  lr 5e-05  Le 15.0709 (13.0920)	Time 0.965 (0.000)	Data 0.002 (0.000)	
2021-06-16 07:43:09,266 Test: [0/40]	Le 60.4090 (60.4089)	Time 7.943 (0.000)	
2021-06-16 07:43:25,071 calculate similarity time: 0.08060097694396973
2021-06-16 07:43:25,588 Image to text: 82.7, 97.7, 99.5, 1.0, 2.0
2021-06-16 07:43:25,901 Text to image: 67.2, 92.8, 96.8, 1.0, 3.7
2021-06-16 07:43:25,901 Current rsum is 536.6999999999999
2021-06-16 07:43:27,365 runs/coco_butd_region_bert/log
2021-06-16 07:43:27,365 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-16 07:43:27,367 image encoder trainable parameters: 20490144
2021-06-16 07:43:27,372 txt encoder trainable parameters: 137319072
2021-06-16 07:46:03,059 Epoch: [18][149/4426]	Eit 79800  lr 5e-05  Le 13.9147 (12.8445)	Time 1.174 (0.000)	Data 0.002 (0.000)	
2021-06-16 07:49:18,556 Epoch: [18][349/4426]	Eit 80000  lr 5e-05  Le 11.1915 (12.7457)	Time 0.832 (0.000)	Data 0.002 (0.000)	
2021-06-16 07:52:32,474 Epoch: [18][549/4426]	Eit 80200  lr 5e-05  Le 15.0964 (13.0125)	Time 0.940 (0.000)	Data 0.002 (0.000)	
2021-06-16 07:55:46,141 Epoch: [18][749/4426]	Eit 80400  lr 5e-05  Le 12.6635 (12.9578)	Time 0.931 (0.000)	Data 0.002 (0.000)	
2021-06-16 07:59:04,124 Epoch: [18][949/4426]	Eit 80600  lr 5e-05  Le 14.3282 (12.9621)	Time 0.985 (0.000)	Data 0.002 (0.000)	
2021-06-16 08:02:22,139 Epoch: [18][1149/4426]	Eit 80800  lr 5e-05  Le 18.7250 (12.9599)	Time 0.899 (0.000)	Data 0.003 (0.000)	
2021-06-16 08:05:36,582 Epoch: [18][1349/4426]	Eit 81000  lr 5e-05  Le 8.9811 (12.9601)	Time 1.067 (0.000)	Data 0.002 (0.000)	
2021-06-16 08:08:54,015 Epoch: [18][1549/4426]	Eit 81200  lr 5e-05  Le 12.5593 (12.9594)	Time 0.993 (0.000)	Data 0.002 (0.000)	
2021-06-16 08:12:08,740 Epoch: [18][1749/4426]	Eit 81400  lr 5e-05  Le 10.6723 (12.9450)	Time 1.081 (0.000)	Data 0.002 (0.000)	
2021-06-16 08:15:22,717 Epoch: [18][1949/4426]	Eit 81600  lr 5e-05  Le 9.1139 (12.9292)	Time 0.985 (0.000)	Data 0.002 (0.000)	
2021-06-16 08:18:42,223 Epoch: [18][2149/4426]	Eit 81800  lr 5e-05  Le 10.7199 (12.9124)	Time 0.909 (0.000)	Data 0.002 (0.000)	
2021-06-16 08:22:00,427 Epoch: [18][2349/4426]	Eit 82000  lr 5e-05  Le 14.0483 (12.8837)	Time 0.896 (0.000)	Data 0.002 (0.000)	
2021-06-16 08:25:17,385 Epoch: [18][2549/4426]	Eit 82200  lr 5e-05  Le 12.8975 (12.8663)	Time 0.791 (0.000)	Data 0.002 (0.000)	
2021-06-16 08:28:33,716 Epoch: [18][2749/4426]	Eit 82400  lr 5e-05  Le 13.7211 (12.8428)	Time 0.829 (0.000)	Data 0.003 (0.000)	
2021-06-16 08:31:50,073 Epoch: [18][2949/4426]	Eit 82600  lr 5e-05  Le 9.8397 (12.8392)	Time 1.265 (0.000)	Data 0.002 (0.000)	
2021-06-16 08:35:05,088 Epoch: [18][3149/4426]	Eit 82800  lr 5e-05  Le 13.0797 (12.8362)	Time 1.035 (0.000)	Data 0.002 (0.000)	
2021-06-16 08:38:21,699 Epoch: [18][3349/4426]	Eit 83000  lr 5e-05  Le 14.0655 (12.8550)	Time 0.973 (0.000)	Data 0.002 (0.000)	
2021-06-16 08:41:41,473 Epoch: [18][3549/4426]	Eit 83200  lr 5e-05  Le 15.1848 (12.8538)	Time 0.816 (0.000)	Data 0.002 (0.000)	
2021-06-16 08:44:57,954 Epoch: [18][3749/4426]	Eit 83400  lr 5e-05  Le 12.3333 (12.8491)	Time 0.898 (0.000)	Data 0.003 (0.000)	
2021-06-16 08:48:14,452 Epoch: [18][3949/4426]	Eit 83600  lr 5e-05  Le 15.4859 (12.8603)	Time 0.955 (0.000)	Data 0.002 (0.000)	
2021-06-16 08:51:28,856 Epoch: [18][4149/4426]	Eit 83800  lr 5e-05  Le 12.5887 (12.8590)	Time 0.778 (0.000)	Data 0.002 (0.000)	
2021-06-16 08:54:43,033 Epoch: [18][4349/4426]	Eit 84000  lr 5e-05  Le 8.3218 (12.8642)	Time 0.854 (0.000)	Data 0.002 (0.000)	
2021-06-16 08:56:04,668 Test: [0/40]	Le 60.1192 (60.1192)	Time 7.802 (0.000)	
2021-06-16 08:56:20,486 calculate similarity time: 0.07872223854064941
2021-06-16 08:56:21,019 Image to text: 82.6, 98.0, 99.5, 1.0, 2.0
2021-06-16 08:56:21,331 Text to image: 67.3, 92.9, 96.6, 1.0, 3.7
2021-06-16 08:56:21,331 Current rsum is 536.9200000000001
2021-06-16 08:56:22,804 runs/coco_butd_region_bert/log
2021-06-16 08:56:22,804 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-16 08:56:22,806 image encoder trainable parameters: 20490144
2021-06-16 08:56:22,811 txt encoder trainable parameters: 137319072
2021-06-16 08:58:30,760 Epoch: [19][124/4426]	Eit 84200  lr 5e-05  Le 9.9279 (12.6970)	Time 0.992 (0.000)	Data 0.016 (0.000)	
2021-06-16 09:01:46,873 Epoch: [19][324/4426]	Eit 84400  lr 5e-05  Le 12.7860 (12.6741)	Time 1.323 (0.000)	Data 0.002 (0.000)	
2021-06-16 09:05:04,402 Epoch: [19][524/4426]	Eit 84600  lr 5e-05  Le 13.8089 (12.8221)	Time 1.001 (0.000)	Data 0.002 (0.000)	
2021-06-16 09:08:09,346 Epoch: [19][724/4426]	Eit 84800  lr 5e-05  Le 13.3101 (12.8758)	Time 0.913 (0.000)	Data 0.002 (0.000)	
2021-06-16 09:11:24,561 Epoch: [19][924/4426]	Eit 85000  lr 5e-05  Le 10.5051 (12.8159)	Time 1.309 (0.000)	Data 0.003 (0.000)	
2021-06-16 09:14:41,958 Epoch: [19][1124/4426]	Eit 85200  lr 5e-05  Le 13.6930 (12.7436)	Time 1.008 (0.000)	Data 0.002 (0.000)	
2021-06-16 09:17:54,590 Epoch: [19][1324/4426]	Eit 85400  lr 5e-05  Le 11.2351 (12.7630)	Time 0.821 (0.000)	Data 0.002 (0.000)	
2021-06-16 09:21:10,576 Epoch: [19][1524/4426]	Eit 85600  lr 5e-05  Le 12.5296 (12.7795)	Time 0.934 (0.000)	Data 0.002 (0.000)	
2021-06-16 09:24:27,646 Epoch: [19][1724/4426]	Eit 85800  lr 5e-05  Le 12.1804 (12.7716)	Time 0.924 (0.000)	Data 0.002 (0.000)	
2021-06-16 09:27:42,600 Epoch: [19][1924/4426]	Eit 86000  lr 5e-05  Le 17.4072 (12.7599)	Time 1.046 (0.000)	Data 0.002 (0.000)	
2021-06-16 09:30:59,833 Epoch: [19][2124/4426]	Eit 86200  lr 5e-05  Le 17.3524 (12.7735)	Time 1.041 (0.000)	Data 0.002 (0.000)	
2021-06-16 09:34:16,681 Epoch: [19][2324/4426]	Eit 86400  lr 5e-05  Le 8.1957 (12.7598)	Time 1.203 (0.000)	Data 0.002 (0.000)	
2021-06-16 09:37:35,052 Epoch: [19][2524/4426]	Eit 86600  lr 5e-05  Le 9.8188 (12.7414)	Time 0.774 (0.000)	Data 0.002 (0.000)	
2021-06-16 09:40:49,876 Epoch: [19][2724/4426]	Eit 86800  lr 5e-05  Le 15.5201 (12.7357)	Time 1.005 (0.000)	Data 0.002 (0.000)	
2021-06-16 09:44:05,448 Epoch: [19][2924/4426]	Eit 87000  lr 5e-05  Le 10.1778 (12.7308)	Time 0.828 (0.000)	Data 0.002 (0.000)	
2021-06-16 09:47:22,729 Epoch: [19][3124/4426]	Eit 87200  lr 5e-05  Le 11.9644 (12.7330)	Time 0.938 (0.000)	Data 0.002 (0.000)	
2021-06-16 09:50:42,767 Epoch: [19][3324/4426]	Eit 87400  lr 5e-05  Le 12.5204 (12.7548)	Time 1.024 (0.000)	Data 0.002 (0.000)	
2021-06-16 09:53:59,203 Epoch: [19][3524/4426]	Eit 87600  lr 5e-05  Le 15.7714 (12.7510)	Time 0.814 (0.000)	Data 0.002 (0.000)	
2021-06-16 09:57:16,113 Epoch: [19][3724/4426]	Eit 87800  lr 5e-05  Le 13.7139 (12.7471)	Time 1.367 (0.000)	Data 0.002 (0.000)	
2021-06-16 10:00:32,362 Epoch: [19][3924/4426]	Eit 88000  lr 5e-05  Le 16.2370 (12.7275)	Time 0.886 (0.000)	Data 0.002 (0.000)	
2021-06-16 10:03:52,689 Epoch: [19][4124/4426]	Eit 88200  lr 5e-05  Le 7.6249 (12.7322)	Time 1.122 (0.000)	Data 0.003 (0.000)	
2021-06-16 10:07:05,641 Epoch: [19][4324/4426]	Eit 88400  lr 5e-05  Le 12.4704 (12.7236)	Time 0.915 (0.000)	Data 0.002 (0.000)	
2021-06-16 10:08:54,552 Test: [0/40]	Le 60.0692 (60.0692)	Time 8.063 (0.000)	
2021-06-16 10:09:10,586 calculate similarity time: 0.07378196716308594
2021-06-16 10:09:11,084 Image to text: 83.3, 98.0, 99.5, 1.0, 2.0
2021-06-16 10:09:11,403 Text to image: 67.6, 92.8, 96.8, 1.0, 3.6
2021-06-16 10:09:11,403 Current rsum is 538.0
2021-06-16 10:09:15,017 runs/coco_butd_region_bert/log
2021-06-16 10:09:15,017 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-16 10:09:15,020 image encoder trainable parameters: 20490144
2021-06-16 10:09:15,031 txt encoder trainable parameters: 137319072
2021-06-16 10:11:01,312 Epoch: [20][99/4426]	Eit 88600  lr 5e-05  Le 10.0577 (12.8138)	Time 0.790 (0.000)	Data 0.002 (0.000)	
2021-06-16 10:14:15,519 Epoch: [20][299/4426]	Eit 88800  lr 5e-05  Le 12.0486 (12.8656)	Time 0.860 (0.000)	Data 0.003 (0.000)	
2021-06-16 10:17:30,897 Epoch: [20][499/4426]	Eit 89000  lr 5e-05  Le 12.7938 (12.7108)	Time 1.172 (0.000)	Data 0.002 (0.000)	
2021-06-16 10:20:46,518 Epoch: [20][699/4426]	Eit 89200  lr 5e-05  Le 13.6688 (12.6770)	Time 0.897 (0.000)	Data 0.002 (0.000)	
2021-06-16 10:24:02,637 Epoch: [20][899/4426]	Eit 89400  lr 5e-05  Le 15.3020 (12.6674)	Time 0.947 (0.000)	Data 0.002 (0.000)	
2021-06-16 10:27:22,234 Epoch: [20][1099/4426]	Eit 89600  lr 5e-05  Le 14.8102 (12.6270)	Time 0.963 (0.000)	Data 0.002 (0.000)	
2021-06-16 10:30:38,876 Epoch: [20][1299/4426]	Eit 89800  lr 5e-05  Le 14.0191 (12.6008)	Time 0.965 (0.000)	Data 0.002 (0.000)	
2021-06-16 10:33:58,257 Epoch: [20][1499/4426]	Eit 90000  lr 5e-05  Le 11.2564 (12.5834)	Time 1.027 (0.000)	Data 0.002 (0.000)	
2021-06-16 10:37:12,324 Epoch: [20][1699/4426]	Eit 90200  lr 5e-05  Le 14.3222 (12.6166)	Time 0.813 (0.000)	Data 0.002 (0.000)	
2021-06-16 10:40:28,112 Epoch: [20][1899/4426]	Eit 90400  lr 5e-05  Le 12.7783 (12.6280)	Time 1.119 (0.000)	Data 0.002 (0.000)	
2021-06-16 10:43:44,769 Epoch: [20][2099/4426]	Eit 90600  lr 5e-05  Le 12.2165 (12.6219)	Time 0.829 (0.000)	Data 0.002 (0.000)	
2021-06-16 10:47:00,547 Epoch: [20][2299/4426]	Eit 90800  lr 5e-05  Le 13.4490 (12.6127)	Time 0.781 (0.000)	Data 0.002 (0.000)	
2021-06-16 10:50:16,959 Epoch: [20][2499/4426]	Eit 91000  lr 5e-05  Le 15.6245 (12.5941)	Time 0.826 (0.000)	Data 0.002 (0.000)	
2021-06-16 10:53:36,187 Epoch: [20][2699/4426]	Eit 91200  lr 5e-05  Le 11.2912 (12.5988)	Time 0.948 (0.000)	Data 0.002 (0.000)	
2021-06-16 10:56:54,079 Epoch: [20][2899/4426]	Eit 91400  lr 5e-05  Le 15.6578 (12.6059)	Time 1.101 (0.000)	Data 0.002 (0.000)	
2021-06-16 11:00:08,907 Epoch: [20][3099/4426]	Eit 91600  lr 5e-05  Le 12.0204 (12.5857)	Time 1.014 (0.000)	Data 0.002 (0.000)	
2021-06-16 11:03:26,390 Epoch: [20][3299/4426]	Eit 91800  lr 5e-05  Le 15.3486 (12.5786)	Time 0.925 (0.000)	Data 0.004 (0.000)	
2021-06-16 11:06:44,196 Epoch: [20][3499/4426]	Eit 92000  lr 5e-05  Le 12.4043 (12.5731)	Time 0.993 (0.000)	Data 0.002 (0.000)	
2021-06-16 11:10:00,973 Epoch: [20][3699/4426]	Eit 92200  lr 5e-05  Le 14.9611 (12.5798)	Time 0.901 (0.000)	Data 0.002 (0.000)	
2021-06-16 11:13:18,765 Epoch: [20][3899/4426]	Eit 92400  lr 5e-05  Le 12.4039 (12.5831)	Time 0.904 (0.000)	Data 0.002 (0.000)	
2021-06-16 11:16:24,027 Epoch: [20][4099/4426]	Eit 92600  lr 5e-05  Le 14.2348 (12.5739)	Time 0.588 (0.000)	Data 0.002 (0.000)	
2021-06-16 11:19:38,522 Epoch: [20][4299/4426]	Eit 92800  lr 5e-05  Le 13.7424 (12.5895)	Time 0.939 (0.000)	Data 0.002 (0.000)	
2021-06-16 11:21:51,208 Test: [0/40]	Le 60.0877 (60.0876)	Time 8.216 (0.000)	
2021-06-16 11:22:06,910 calculate similarity time: 0.06686282157897949
2021-06-16 11:22:07,354 Image to text: 83.6, 98.0, 99.4, 1.0, 1.8
2021-06-16 11:22:07,808 Text to image: 67.4, 92.7, 96.8, 1.0, 3.6
2021-06-16 11:22:07,808 Current rsum is 537.86
2021-06-16 11:22:09,359 runs/coco_butd_region_bert/log
2021-06-16 11:22:09,359 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-16 11:22:09,361 image encoder trainable parameters: 20490144
2021-06-16 11:22:09,366 txt encoder trainable parameters: 137319072
2021-06-16 11:23:30,498 Epoch: [21][74/4426]	Eit 93000  lr 5e-05  Le 13.1074 (12.3081)	Time 1.181 (0.000)	Data 0.002 (0.000)	
2021-06-16 11:26:44,770 Epoch: [21][274/4426]	Eit 93200  lr 5e-05  Le 15.1511 (12.3163)	Time 0.793 (0.000)	Data 0.002 (0.000)	
2021-06-16 11:29:58,515 Epoch: [21][474/4426]	Eit 93400  lr 5e-05  Le 13.5357 (12.4060)	Time 0.760 (0.000)	Data 0.002 (0.000)	
2021-06-16 11:33:12,152 Epoch: [21][674/4426]	Eit 93600  lr 5e-05  Le 8.5475 (12.2725)	Time 0.947 (0.000)	Data 0.002 (0.000)	
2021-06-16 11:36:25,616 Epoch: [21][874/4426]	Eit 93800  lr 5e-05  Le 13.7236 (12.2709)	Time 1.083 (0.000)	Data 0.003 (0.000)	
2021-06-16 11:39:46,895 Epoch: [21][1074/4426]	Eit 94000  lr 5e-05  Le 10.3345 (12.3492)	Time 0.877 (0.000)	Data 0.002 (0.000)	
2021-06-16 11:43:04,766 Epoch: [21][1274/4426]	Eit 94200  lr 5e-05  Le 17.7570 (12.3707)	Time 1.065 (0.000)	Data 0.002 (0.000)	
2021-06-16 11:46:18,162 Epoch: [21][1474/4426]	Eit 94400  lr 5e-05  Le 14.3119 (12.4085)	Time 1.256 (0.000)	Data 0.002 (0.000)	
2021-06-16 11:49:33,620 Epoch: [21][1674/4426]	Eit 94600  lr 5e-05  Le 14.3218 (12.3958)	Time 0.876 (0.000)	Data 0.002 (0.000)	
2021-06-16 11:52:49,597 Epoch: [21][1874/4426]	Eit 94800  lr 5e-05  Le 11.9468 (12.4363)	Time 0.998 (0.000)	Data 0.002 (0.000)	
2021-06-16 11:56:06,219 Epoch: [21][2074/4426]	Eit 95000  lr 5e-05  Le 14.1617 (12.4437)	Time 1.010 (0.000)	Data 0.002 (0.000)	
2021-06-16 11:59:23,891 Epoch: [21][2274/4426]	Eit 95200  lr 5e-05  Le 13.4817 (12.4571)	Time 1.126 (0.000)	Data 0.003 (0.000)	
2021-06-16 12:02:41,467 Epoch: [21][2474/4426]	Eit 95400  lr 5e-05  Le 10.3389 (12.4742)	Time 0.896 (0.000)	Data 0.002 (0.000)	
2021-06-16 12:05:58,354 Epoch: [21][2674/4426]	Eit 95600  lr 5e-05  Le 16.9834 (12.4868)	Time 0.964 (0.000)	Data 0.002 (0.000)	
2021-06-16 12:09:11,744 Epoch: [21][2874/4426]	Eit 95800  lr 5e-05  Le 13.4651 (12.4852)	Time 0.850 (0.000)	Data 0.002 (0.000)	
2021-06-16 12:12:28,911 Epoch: [21][3074/4426]	Eit 96000  lr 5e-05  Le 11.8697 (12.4906)	Time 1.239 (0.000)	Data 0.002 (0.000)	
2021-06-16 12:15:42,828 Epoch: [21][3274/4426]	Eit 96200  lr 5e-05  Le 9.9657 (12.4839)	Time 0.907 (0.000)	Data 0.002 (0.000)	
2021-06-16 12:19:00,472 Epoch: [21][3474/4426]	Eit 96400  lr 5e-05  Le 11.4338 (12.4736)	Time 0.759 (0.000)	Data 0.002 (0.000)	
2021-06-16 12:22:15,838 Epoch: [21][3674/4426]	Eit 96600  lr 5e-05  Le 10.3738 (12.4812)	Time 1.145 (0.000)	Data 0.002 (0.000)	
2021-06-16 12:25:31,717 Epoch: [21][3874/4426]	Eit 96800  lr 5e-05  Le 10.2208 (12.4884)	Time 1.149 (0.000)	Data 0.002 (0.000)	
2021-06-16 12:28:47,664 Epoch: [21][4074/4426]	Eit 97000  lr 5e-05  Le 17.1842 (12.4864)	Time 1.025 (0.000)	Data 0.002 (0.000)	
2021-06-16 12:32:03,537 Epoch: [21][4274/4426]	Eit 97200  lr 5e-05  Le 9.6825 (12.4959)	Time 1.043 (0.000)	Data 0.002 (0.000)	
2021-06-16 12:34:41,974 Test: [0/40]	Le 59.9896 (59.9895)	Time 8.037 (0.000)	
2021-06-16 12:34:58,239 calculate similarity time: 0.06361603736877441
2021-06-16 12:34:58,670 Image to text: 83.2, 97.7, 99.6, 1.0, 1.7
2021-06-16 12:34:58,984 Text to image: 67.8, 92.8, 96.7, 1.0, 3.7
2021-06-16 12:34:58,984 Current rsum is 537.76
2021-06-16 12:35:00,409 runs/coco_butd_region_bert/log
2021-06-16 12:35:00,410 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-16 12:35:00,411 image encoder trainable parameters: 20490144
2021-06-16 12:35:00,416 txt encoder trainable parameters: 137319072
2021-06-16 12:35:56,764 Epoch: [22][49/4426]	Eit 97400  lr 5e-05  Le 12.2645 (12.8882)	Time 1.259 (0.000)	Data 0.002 (0.000)	
2021-06-16 12:39:12,549 Epoch: [22][249/4426]	Eit 97600  lr 5e-05  Le 12.0572 (12.6769)	Time 0.898 (0.000)	Data 0.002 (0.000)	
2021-06-16 12:42:28,657 Epoch: [22][449/4426]	Eit 97800  lr 5e-05  Le 13.5427 (12.5282)	Time 0.969 (0.000)	Data 0.002 (0.000)	
2021-06-16 12:45:47,074 Epoch: [22][649/4426]	Eit 98000  lr 5e-05  Le 7.8480 (12.5793)	Time 0.975 (0.000)	Data 0.002 (0.000)	
2021-06-16 12:49:03,610 Epoch: [22][849/4426]	Eit 98200  lr 5e-05  Le 11.4161 (12.4976)	Time 0.776 (0.000)	Data 0.002 (0.000)	
2021-06-16 12:52:20,470 Epoch: [22][1049/4426]	Eit 98400  lr 5e-05  Le 13.2371 (12.4184)	Time 0.857 (0.000)	Data 0.002 (0.000)	
2021-06-16 12:55:39,476 Epoch: [22][1249/4426]	Eit 98600  lr 5e-05  Le 15.2391 (12.4236)	Time 1.305 (0.000)	Data 0.002 (0.000)	
2021-06-16 12:58:56,032 Epoch: [22][1449/4426]	Eit 98800  lr 5e-05  Le 14.3865 (12.3933)	Time 0.800 (0.000)	Data 0.002 (0.000)	
2021-06-16 13:02:15,573 Epoch: [22][1649/4426]	Eit 99000  lr 5e-05  Le 13.0116 (12.3938)	Time 1.045 (0.000)	Data 0.002 (0.000)	
2021-06-16 13:05:26,824 Epoch: [22][1849/4426]	Eit 99200  lr 5e-05  Le 14.5179 (12.3815)	Time 0.902 (0.000)	Data 0.002 (0.000)	
2021-06-16 13:08:44,865 Epoch: [22][2049/4426]	Eit 99400  lr 5e-05  Le 14.4754 (12.3690)	Time 0.801 (0.000)	Data 0.002 (0.000)	
2021-06-16 13:12:00,264 Epoch: [22][2249/4426]	Eit 99600  lr 5e-05  Le 13.4500 (12.3689)	Time 1.008 (0.000)	Data 0.002 (0.000)	
2021-06-16 13:15:16,039 Epoch: [22][2449/4426]	Eit 99800  lr 5e-05  Le 12.2714 (12.3686)	Time 0.807 (0.000)	Data 0.002 (0.000)	
2021-06-16 13:18:35,506 Epoch: [22][2649/4426]	Eit 100000  lr 5e-05  Le 12.3292 (12.3497)	Time 1.015 (0.000)	Data 0.002 (0.000)	
2021-06-16 13:21:50,849 Epoch: [22][2849/4426]	Eit 100200  lr 5e-05  Le 13.5459 (12.3332)	Time 0.809 (0.000)	Data 0.003 (0.000)	
2021-06-16 13:25:01,328 Epoch: [22][3049/4426]	Eit 100400  lr 5e-05  Le 13.9053 (12.3357)	Time 0.880 (0.000)	Data 0.002 (0.000)	
2021-06-16 13:28:11,104 Epoch: [22][3249/4426]	Eit 100600  lr 5e-05  Le 12.6471 (12.3401)	Time 1.178 (0.000)	Data 0.002 (0.000)	
2021-06-16 13:31:29,889 Epoch: [22][3449/4426]	Eit 100800  lr 5e-05  Le 13.5270 (12.3490)	Time 0.942 (0.000)	Data 0.002 (0.000)	
2021-06-16 13:34:46,953 Epoch: [22][3649/4426]	Eit 101000  lr 5e-05  Le 11.2271 (12.3294)	Time 0.809 (0.000)	Data 0.002 (0.000)	
2021-06-16 13:38:06,075 Epoch: [22][3849/4426]	Eit 101200  lr 5e-05  Le 11.5928 (12.3317)	Time 1.134 (0.000)	Data 0.002 (0.000)	
2021-06-16 13:41:22,319 Epoch: [22][4049/4426]	Eit 101400  lr 5e-05  Le 10.7364 (12.3189)	Time 0.884 (0.000)	Data 0.002 (0.000)	
2021-06-16 13:44:38,246 Epoch: [22][4249/4426]	Eit 101600  lr 5e-05  Le 9.5180 (12.3179)	Time 1.145 (0.000)	Data 0.002 (0.000)	
2021-06-16 13:47:38,968 Test: [0/40]	Le 60.1370 (60.1369)	Time 7.861 (0.000)	
2021-06-16 13:47:54,865 calculate similarity time: 0.05679035186767578
2021-06-16 13:47:55,391 Image to text: 82.9, 98.0, 99.4, 1.0, 1.9
2021-06-16 13:47:55,839 Text to image: 67.5, 92.5, 96.7, 1.0, 3.6
2021-06-16 13:47:55,839 Current rsum is 537.0
2021-06-16 13:47:57,329 runs/coco_butd_region_bert/log
2021-06-16 13:47:57,329 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-16 13:47:57,330 image encoder trainable parameters: 20490144
2021-06-16 13:47:57,336 txt encoder trainable parameters: 137319072
2021-06-16 13:48:29,575 Epoch: [23][24/4426]	Eit 101800  lr 5e-05  Le 14.8503 (12.8980)	Time 0.982 (0.000)	Data 0.002 (0.000)	
2021-06-16 13:51:46,780 Epoch: [23][224/4426]	Eit 102000  lr 5e-05  Le 13.8798 (11.9870)	Time 1.135 (0.000)	Data 0.002 (0.000)	
2021-06-16 13:55:04,767 Epoch: [23][424/4426]	Eit 102200  lr 5e-05  Le 11.7028 (11.9865)	Time 1.118 (0.000)	Data 0.002 (0.000)	
2021-06-16 13:58:23,922 Epoch: [23][624/4426]	Eit 102400  lr 5e-05  Le 16.0244 (11.9962)	Time 0.763 (0.000)	Data 0.002 (0.000)	
2021-06-16 14:01:40,967 Epoch: [23][824/4426]	Eit 102600  lr 5e-05  Le 14.4843 (12.0890)	Time 1.101 (0.000)	Data 0.002 (0.000)	
2021-06-16 14:04:56,021 Epoch: [23][1024/4426]	Eit 102800  lr 5e-05  Le 13.1439 (12.1496)	Time 0.949 (0.000)	Data 0.002 (0.000)	
2021-06-16 14:08:15,048 Epoch: [23][1224/4426]	Eit 103000  lr 5e-05  Le 10.6963 (12.1109)	Time 0.837 (0.000)	Data 0.002 (0.000)	
2021-06-16 14:11:29,779 Epoch: [23][1424/4426]	Eit 103200  lr 5e-05  Le 12.7426 (12.1740)	Time 0.898 (0.000)	Data 0.002 (0.000)	
2021-06-16 14:14:45,350 Epoch: [23][1624/4426]	Eit 103400  lr 5e-05  Le 9.7382 (12.1604)	Time 1.023 (0.000)	Data 0.002 (0.000)	
2021-06-16 14:18:03,521 Epoch: [23][1824/4426]	Eit 103600  lr 5e-05  Le 11.4168 (12.1505)	Time 0.822 (0.000)	Data 0.002 (0.000)	
2021-06-16 14:21:22,367 Epoch: [23][2024/4426]	Eit 103800  lr 5e-05  Le 8.4833 (12.1834)	Time 1.383 (0.000)	Data 0.003 (0.000)	
2021-06-16 14:24:36,366 Epoch: [23][2224/4426]	Eit 104000  lr 5e-05  Le 12.0609 (12.1712)	Time 1.034 (0.000)	Data 0.002 (0.000)	
2021-06-16 14:27:51,509 Epoch: [23][2424/4426]	Eit 104200  lr 5e-05  Le 12.6494 (12.1842)	Time 0.972 (0.000)	Data 0.002 (0.000)	
2021-06-16 14:31:08,694 Epoch: [23][2624/4426]	Eit 104400  lr 5e-05  Le 14.2614 (12.1967)	Time 0.897 (0.000)	Data 0.003 (0.000)	
2021-06-16 14:34:26,964 Epoch: [23][2824/4426]	Eit 104600  lr 5e-05  Le 11.5336 (12.1966)	Time 0.827 (0.000)	Data 0.003 (0.000)	
2021-06-16 14:37:46,611 Epoch: [23][3024/4426]	Eit 104800  lr 5e-05  Le 13.3537 (12.1931)	Time 0.966 (0.000)	Data 0.002 (0.000)	
2021-06-16 14:41:02,074 Epoch: [23][3224/4426]	Eit 105000  lr 5e-05  Le 16.5484 (12.1883)	Time 0.807 (0.000)	Data 0.002 (0.000)	
2021-06-16 14:44:19,649 Epoch: [23][3424/4426]	Eit 105200  lr 5e-05  Le 9.7862 (12.1647)	Time 1.029 (0.000)	Data 0.002 (0.000)	
2021-06-16 14:47:35,351 Epoch: [23][3624/4426]	Eit 105400  lr 5e-05  Le 14.6939 (12.1798)	Time 0.945 (0.000)	Data 0.003 (0.000)	
2021-06-16 14:50:52,861 Epoch: [23][3824/4426]	Eit 105600  lr 5e-05  Le 10.4962 (12.1755)	Time 1.188 (0.000)	Data 0.002 (0.000)	
2021-06-16 14:54:10,868 Epoch: [23][4024/4426]	Eit 105800  lr 5e-05  Le 14.4247 (12.1926)	Time 0.846 (0.000)	Data 0.002 (0.000)	
2021-06-16 14:57:29,505 Epoch: [23][4224/4426]	Eit 106000  lr 5e-05  Le 13.1742 (12.1829)	Time 0.798 (0.000)	Data 0.002 (0.000)	
2021-06-16 15:00:44,216 Epoch: [23][4424/4426]	Eit 106200  lr 5e-05  Le 8.7844 (12.1807)	Time 1.019 (0.000)	Data 0.002 (0.000)	
2021-06-16 15:00:53,937 Test: [0/40]	Le 60.1801 (60.1800)	Time 8.330 (0.000)	
2021-06-16 15:01:10,104 calculate similarity time: 0.06995105743408203
2021-06-16 15:01:10,613 Image to text: 82.5, 97.7, 99.4, 1.0, 1.9
2021-06-16 15:01:10,928 Text to image: 67.7, 92.5, 96.8, 1.0, 3.7
2021-06-16 15:01:10,928 Current rsum is 536.6600000000001
2021-06-16 15:01:12,382 runs/coco_butd_region_bert/log
2021-06-16 15:01:12,382 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-16 15:01:12,384 image encoder trainable parameters: 20490144
2021-06-16 15:01:12,403 txt encoder trainable parameters: 137319072
2021-06-16 15:04:34,632 Epoch: [24][199/4426]	Eit 106400  lr 5e-05  Le 13.9156 (11.8844)	Time 0.748 (0.000)	Data 0.002 (0.000)	
2021-06-16 15:07:54,541 Epoch: [24][399/4426]	Eit 106600  lr 5e-05  Le 11.4180 (11.8975)	Time 1.053 (0.000)	Data 0.002 (0.000)	
2021-06-16 15:11:09,404 Epoch: [24][599/4426]	Eit 106800  lr 5e-05  Le 11.1866 (11.9914)	Time 0.966 (0.000)	Data 0.002 (0.000)	
2021-06-16 15:14:25,655 Epoch: [24][799/4426]	Eit 107000  lr 5e-05  Le 16.5970 (11.9963)	Time 0.836 (0.000)	Data 0.002 (0.000)	
2021-06-16 15:17:44,752 Epoch: [24][999/4426]	Eit 107200  lr 5e-05  Le 14.9490 (12.0291)	Time 1.035 (0.000)	Data 0.002 (0.000)	
2021-06-16 15:21:02,650 Epoch: [24][1199/4426]	Eit 107400  lr 5e-05  Le 9.4977 (12.1073)	Time 1.085 (0.000)	Data 0.002 (0.000)	
2021-06-16 15:24:21,051 Epoch: [24][1399/4426]	Eit 107600  lr 5e-05  Le 17.6340 (12.1297)	Time 0.915 (0.000)	Data 0.002 (0.000)	
2021-06-16 15:27:35,601 Epoch: [24][1599/4426]	Eit 107800  lr 5e-05  Le 16.2289 (12.1185)	Time 1.206 (0.000)	Data 0.002 (0.000)	
2021-06-16 15:30:53,144 Epoch: [24][1799/4426]	Eit 108000  lr 5e-05  Le 11.5881 (12.0920)	Time 1.001 (0.000)	Data 0.002 (0.000)	
2021-06-16 15:33:56,473 Epoch: [24][1999/4426]	Eit 108200  lr 5e-05  Le 10.8733 (12.1087)	Time 0.683 (0.000)	Data 0.002 (0.000)	
2021-06-16 15:37:11,618 Epoch: [24][2199/4426]	Eit 108400  lr 5e-05  Le 12.0319 (12.1142)	Time 1.000 (0.000)	Data 0.002 (0.000)	
2021-06-16 15:40:28,906 Epoch: [24][2399/4426]	Eit 108600  lr 5e-05  Le 13.4527 (12.0999)	Time 0.767 (0.000)	Data 0.002 (0.000)	
2021-06-16 15:43:43,225 Epoch: [24][2599/4426]	Eit 108800  lr 5e-05  Le 14.5758 (12.1020)	Time 0.782 (0.000)	Data 0.002 (0.000)	
2021-06-16 15:47:01,371 Epoch: [24][2799/4426]	Eit 109000  lr 5e-05  Le 7.8047 (12.0883)	Time 0.839 (0.000)	Data 0.003 (0.000)	
2021-06-16 15:50:19,796 Epoch: [24][2999/4426]	Eit 109200  lr 5e-05  Le 18.7273 (12.0651)	Time 1.027 (0.000)	Data 0.002 (0.000)	
2021-06-16 15:53:38,828 Epoch: [24][3199/4426]	Eit 109400  lr 5e-05  Le 12.4229 (12.0738)	Time 0.908 (0.000)	Data 0.002 (0.000)	
2021-06-16 15:56:55,059 Epoch: [24][3399/4426]	Eit 109600  lr 5e-05  Le 15.5300 (12.0574)	Time 1.104 (0.000)	Data 0.002 (0.000)	
2021-06-16 16:00:12,301 Epoch: [24][3599/4426]	Eit 109800  lr 5e-05  Le 12.3844 (12.0732)	Time 1.093 (0.000)	Data 0.002 (0.000)	
2021-06-16 16:03:27,116 Epoch: [24][3799/4426]	Eit 110000  lr 5e-05  Le 12.0545 (12.0765)	Time 0.955 (0.000)	Data 0.002 (0.000)	
2021-06-16 16:06:44,258 Epoch: [24][3999/4426]	Eit 110200  lr 5e-05  Le 15.1967 (12.0630)	Time 0.974 (0.000)	Data 0.002 (0.000)	
2021-06-16 16:09:58,260 Epoch: [24][4199/4426]	Eit 110400  lr 5e-05  Le 9.4029 (12.0659)	Time 0.849 (0.000)	Data 0.002 (0.000)	
2021-06-16 16:13:14,204 Epoch: [24][4399/4426]	Eit 110600  lr 5e-05  Le 11.9265 (12.0812)	Time 1.061 (0.000)	Data 0.002 (0.000)	
2021-06-16 16:13:48,464 Test: [0/40]	Le 60.1210 (60.1210)	Time 7.551 (0.000)	
2021-06-16 16:14:04,749 calculate similarity time: 0.07332944869995117
2021-06-16 16:14:05,215 Image to text: 82.8, 97.9, 99.4, 1.0, 1.9
2021-06-16 16:14:05,538 Text to image: 67.8, 92.7, 96.7, 1.0, 3.7
2021-06-16 16:14:05,538 Current rsum is 537.32
You have new mail in /var/spool/mail/root
[root@gpu1 vse_infty-master-my-graph-gru-vse++nowarmup]# CUDA_VISIBLE_DEVICES=0,1 python3 eval.py --dataset coco --data_path ../data/coco
INFO:root:Evaluating runs/coco_butd_region_bert...
INFO:lib.evaluation:Namespace(backbone_lr_factor=0.01, backbone_path='', backbone_source='detector', backbone_warmup_epochs=5, batch_size=128, data_name='coco', data_path='../data/coco', embed_size=1024, embedding_warmup_epochs=2, grad_clip=2.0, img_dim=2048, input_scale_factor=1, learning_rate=0.0005, log_step=200, logger_name='runs/coco_butd_region_bert/log', lr_update=15, margin=0.2, max_violation=True, model_name='runs/coco_butd_region_bert', no_imgnorm=False, no_txtnorm=False, num_epochs=25, optim='adam', precomp_enc_type='basic', reset_start_epoch=False, resume='', val_step=500, vocab_path='./vocab/', vocab_size=30522, vse_mean_warmup_epochs=1, word_dim=300, workers=5)
INFO:transformers.tokenization_utils:loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-vse++nowarmup/lib/encoders.py:48: UserWarning: nn.init.constant is now deprecated in favor of nn.init.constant_.
  nn.init.constant(self.W[1].weight, 0)
/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-vse++nowarmup/lib/encoders.py:49: UserWarning: nn.init.constant is now deprecated in favor of nn.init.constant_.
  nn.init.constant(self.W[1].bias, 0)
INFO:transformers.configuration_utils:loading configuration file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/config.json
INFO:transformers.configuration_utils:Model config {
  "attention_probs_dropout_prob": 0.1,
  "finetuning_task": null,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "num_labels": 2,
  "output_attentions": false,
  "output_hidden_states": false,
  "pruned_heads": {},
  "torchscript": false,
  "type_vocab_size": 2,
  "use_bfloat16": false,
  "vocab_size": 30522
}

INFO:transformers.modeling_utils:loading weights file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/pytorch_model.bin
INFO:lib.vse:Use adam as the optimizer, with init lr 0.0005
INFO:lib.vse:Image encoder is data paralleled now.
INFO:lib.evaluation:Loading dataset
INFO:lib.evaluation:Computing results...
/root/anaconda3/lib/python3.6/site-packages/torch/nn/modules/rnn.py:735: UserWarning: RNN module weights are not part of single contiguous chunk of memory. This means they need to be compacted at every call, possibly greatly increasing memory usage. To compact weights again call flatten_parameters(). (Triggered internally at  /pytorch/aten/src/ATen/native/cudnn/RNN.cpp:1264.)
  self.dropout, self.training, self.bidirectional, self.batch_first)
INFO:lib.evaluation:Test: [0/196]	Le 62.7427 (62.7427)	Time 5.419 (0.000)	
INFO:lib.evaluation:Test: [10/196]	Le 62.7872 (62.1258)	Time 0.364 (0.000)	
INFO:lib.evaluation:Test: [20/196]	Le 61.7205 (61.7425)	Time 0.222 (0.000)	
INFO:lib.evaluation:Test: [30/196]	Le 62.4590 (61.8600)	Time 0.263 (0.000)	
INFO:lib.evaluation:Test: [40/196]	Le 62.8390 (61.6729)	Time 0.206 (0.000)	
INFO:lib.evaluation:Test: [50/196]	Le 62.4343 (61.6613)	Time 0.241 (0.000)	
INFO:lib.evaluation:Test: [60/196]	Le 61.2830 (61.7219)	Time 0.217 (0.000)	
INFO:lib.evaluation:Test: [70/196]	Le 62.9243 (61.6790)	Time 0.260 (0.000)	
INFO:lib.evaluation:Test: [80/196]	Le 62.1015 (61.7222)	Time 0.309 (0.000)	
INFO:lib.evaluation:Test: [90/196]	Le 61.9429 (61.7649)	Time 0.228 (0.000)	
INFO:lib.evaluation:Test: [100/196]	Le 63.1182 (61.8093)	Time 0.226 (0.000)	
INFO:lib.evaluation:Test: [110/196]	Le 60.2261 (61.7916)	Time 0.205 (0.000)	
INFO:lib.evaluation:Test: [120/196]	Le 66.0503 (61.8445)	Time 0.278 (0.000)	
INFO:lib.evaluation:Test: [130/196]	Le 60.8764 (61.8730)	Time 0.218 (0.000)	
INFO:lib.evaluation:Test: [140/196]	Le 63.7038 (61.9122)	Time 0.223 (0.000)	
INFO:lib.evaluation:Test: [150/196]	Le 60.1666 (61.9064)	Time 0.216 (0.000)	
INFO:lib.evaluation:Test: [160/196]	Le 61.6007 (61.9056)	Time 0.207 (0.000)	
INFO:lib.evaluation:Test: [170/196]	Le 61.6006 (61.9070)	Time 0.236 (0.000)	
INFO:lib.evaluation:Test: [180/196]	Le 60.6564 (61.8688)	Time 0.207 (0.000)	
INFO:lib.evaluation:Test: [190/196]	Le 60.7082 (61.9176)	Time 0.212 (0.000)	
INFO:lib.evaluation:Images: 5000, Captions: 25000
INFO:lib.evaluation:calculate similarity time: 0.08779740333557129
INFO:lib.evaluation:Image to text: 81.5, 97.1, 98.9, 1.0, 1.5
INFO:lib.evaluation:Text to image: 66.9, 91.7, 96.2, 1.0, 4.4
INFO:lib.evaluation:rsum: 532.3 ar: 92.5 ari: 84.9
INFO:lib.evaluation:calculate similarity time: 0.0764913558959961
INFO:lib.evaluation:Image to text: 78.9, 95.2, 97.5, 1.0, 1.9
INFO:lib.evaluation:Text to image: 65.2, 90.6, 95.7, 1.0, 4.0
INFO:lib.evaluation:rsum: 523.1 ar: 90.5 ari: 83.8
INFO:lib.evaluation:calculate similarity time: 0.10118770599365234
INFO:lib.evaluation:Image to text: 80.3, 96.6, 98.6, 1.0, 1.7
INFO:lib.evaluation:Text to image: 65.5, 91.5, 96.4, 1.0, 3.9
INFO:lib.evaluation:rsum: 528.9 ar: 91.8 ari: 84.5
INFO:lib.evaluation:calculate similarity time: 0.0683140754699707
INFO:lib.evaluation:Image to text: 79.3, 95.8, 98.9, 1.0, 1.6
INFO:lib.evaluation:Text to image: 63.7, 91.9, 96.8, 1.0, 3.6
INFO:lib.evaluation:rsum: 526.3 ar: 91.3 ari: 84.1
INFO:lib.evaluation:calculate similarity time: 0.05503678321838379
INFO:lib.evaluation:Image to text: 80.9, 96.6, 99.1, 1.0, 1.6
INFO:lib.evaluation:Text to image: 65.7, 92.1, 96.7, 1.0, 3.6
INFO:lib.evaluation:rsum: 531.1 ar: 92.2 ari: 84.8
INFO:lib.evaluation:-----------------------------------
INFO:lib.evaluation:Mean metrics: 
INFO:lib.evaluation:rsum: 528.4
INFO:lib.evaluation:Average i2t Recall: 91.7
INFO:lib.evaluation:Image to text: 80.2 96.3 98.6 1.0 1.7
INFO:lib.evaluation:Average t2i Recall: 84.4
INFO:lib.evaluation:Text to image: 65.4 91.6 96.4 1.0 3.9
INFO:lib.evaluation:Namespace(backbone_lr_factor=0.01, backbone_path='', backbone_source='detector', backbone_warmup_epochs=5, batch_size=128, data_name='coco', data_path='../data/coco', embed_size=1024, embedding_warmup_epochs=2, grad_clip=2.0, img_dim=2048, input_scale_factor=1, learning_rate=0.0005, log_step=200, logger_name='runs/coco_butd_region_bert/log', lr_update=15, margin=0.2, max_violation=True, model_name='runs/coco_butd_region_bert', no_imgnorm=False, no_txtnorm=False, num_epochs=25, optim='adam', precomp_enc_type='basic', reset_start_epoch=False, resume='', val_step=500, vocab_path='./vocab/', vocab_size=30522, vse_mean_warmup_epochs=1, word_dim=300, workers=5)
INFO:transformers.tokenization_utils:loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
INFO:transformers.configuration_utils:loading configuration file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/config.json
INFO:transformers.configuration_utils:Model config {
  "attention_probs_dropout_prob": 0.1,
  "finetuning_task": null,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "num_labels": 2,
  "output_attentions": false,
  "output_hidden_states": false,
  "pruned_heads": {},
  "torchscript": false,
  "type_vocab_size": 2,
  "use_bfloat16": false,
  "vocab_size": 30522
}

INFO:transformers.modeling_utils:loading weights file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/pytorch_model.bin
INFO:lib.vse:Use adam as the optimizer, with init lr 0.0005
INFO:lib.vse:Image encoder is data paralleled now.
INFO:lib.evaluation:Loading dataset
INFO:lib.evaluation:Computing results...
INFO:lib.evaluation:Test: [0/196]	Le 62.7427 (62.7427)	Time 2.132 (0.000)	
INFO:lib.evaluation:Test: [10/196]	Le 62.7872 (62.1258)	Time 0.293 (0.000)	
INFO:lib.evaluation:Test: [20/196]	Le 61.7205 (61.7425)	Time 0.207 (0.000)	
INFO:lib.evaluation:Test: [30/196]	Le 62.4590 (61.8600)	Time 0.227 (0.000)	
INFO:lib.evaluation:Test: [40/196]	Le 62.8390 (61.6729)	Time 0.236 (0.000)	
INFO:lib.evaluation:Test: [50/196]	Le 62.4343 (61.6613)	Time 0.234 (0.000)	
INFO:lib.evaluation:Test: [60/196]	Le 61.2830 (61.7219)	Time 0.227 (0.000)	
INFO:lib.evaluation:Test: [70/196]	Le 62.9243 (61.6790)	Time 0.216 (0.000)	
INFO:lib.evaluation:Test: [80/196]	Le 62.1015 (61.7222)	Time 0.338 (0.000)	
INFO:lib.evaluation:Test: [90/196]	Le 61.9429 (61.7649)	Time 0.252 (0.000)	
INFO:lib.evaluation:Test: [100/196]	Le 63.1182 (61.8093)	Time 0.214 (0.000)	
INFO:lib.evaluation:Test: [110/196]	Le 60.2261 (61.7916)	Time 0.204 (0.000)	
INFO:lib.evaluation:Test: [120/196]	Le 66.0503 (61.8445)	Time 0.234 (0.000)	
INFO:lib.evaluation:Test: [130/196]	Le 60.8764 (61.8730)	Time 0.221 (0.000)	
INFO:lib.evaluation:Test: [140/196]	Le 63.7038 (61.9122)	Time 0.208 (0.000)	
INFO:lib.evaluation:Test: [150/196]	Le 60.1666 (61.9064)	Time 0.257 (0.000)	
INFO:lib.evaluation:Test: [160/196]	Le 61.6007 (61.9056)	Time 0.218 (0.000)	
INFO:lib.evaluation:Test: [170/196]	Le 61.6006 (61.9070)	Time 0.236 (0.000)	
INFO:lib.evaluation:Test: [180/196]	Le 60.6564 (61.8688)	Time 0.211 (0.000)	
INFO:lib.evaluation:Test: [190/196]	Le 60.7082 (61.9176)	Time 0.213 (0.000)	
INFO:lib.evaluation:Images: 5000, Captions: 25000
INFO:lib.evaluation:calculate similarity time: 0.6290230751037598
INFO:lib.evaluation:rsum: 437.0
INFO:lib.evaluation:Average i2t Recall: 79.0
INFO:lib.evaluation:Image to text: 59.0 85.7 92.2 1.0 4.0
INFO:lib.evaluation:Average t2i Recall: 66.7
INFO:lib.evaluation:Text to image: 43.1 73.4 83.6 2.0 15.2
INFO:root:Evaluating runs/coco_butd_grid_bert...
Traceback (most recent call last):
  File "eval.py", line 58, in <module>
    main()
  File "eval.py", line 46, in main
    evaluation.evalrank(model_path, data_path=opt.data_path, split='testall', fold5=True)
  File "/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-vse++nowarmup/lib/evaluation.py", line 196, in evalrank
    checkpoint = torch.load(model_path)
  File "/root/anaconda3/lib/python3.6/site-packages/torch/serialization.py", line 571, in load
    with _open_file_like(f, 'rb') as opened_file:
  File "/root/anaconda3/lib/python3.6/site-packages/torch/serialization.py", line 229, in _open_file_like
    return _open_file(name_or_buffer, mode)
  File "/root/anaconda3/lib/python3.6/site-packages/torch/serialization.py", line 210, in __init__
    super(_open_file, self).__init__(open(name, mode))
FileNotFoundError: [Errno 2] No such file or directory: 'runs/coco_butd_grid_bert/model_best.pth'
You have new mail in /var/spool/mail/root
[root@gpu1 vse_infty-master-my-graph-gru-vse++nowarmup]# 
