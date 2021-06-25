[root@gpu1 vse_infty-master-my-graph-gru-vse++]# sh train_region.sh 
2021-06-22 19:49:48,184 Namespace(backbone_lr_factor=0.01, backbone_path='', backbone_source='detector', backbone_warmup_epochs=5, batch_size=128, data_name='coco', data_path='../data/coco', embed_size=1024, embedding_warmup_epochs=2, grad_clip=2.0, img_dim=2048, input_scale_factor=1, learning_rate=0.0005, log_step=200, logger_name='runs/coco_butd_region_bert/log', lr_update=15, margin=0.2, max_violation=False, model_name='runs/coco_butd_region_bert', no_imgnorm=False, no_txtnorm=False, num_epochs=25, optim='adam', precomp_enc_type='basic', reset_start_epoch=False, resume='', val_step=500, vocab_path='./vocab/', vse_mean_warmup_epochs=1, word_dim=300, workers=10)
2021-06-22 19:49:48,185 Model name '/public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased' not found in model shortcut name list (bert-base-uncased, bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, bert-base-multilingual-cased, bert-base-chinese, bert-base-german-cased, bert-large-uncased-whole-word-masking, bert-large-cased-whole-word-masking, bert-large-uncased-whole-word-masking-finetuned-squad, bert-large-cased-whole-word-masking-finetuned-squad, bert-base-cased-finetuned-mrpc). Assuming '/public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased' is a path or url to a directory containing tokenizer files.
2021-06-22 19:49:48,185 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/added_tokens.json. We won't load it.
2021-06-22 19:49:48,185 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/special_tokens_map.json. We won't load it.
2021-06-22 19:49:48,185 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/tokenizer_config.json. We won't load it.
2021-06-22 19:49:48,186 loading file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/vocab.txt
2021-06-22 19:49:48,186 loading file None
2021-06-22 19:49:48,186 loading file None
2021-06-22 19:49:48,186 loading file None
/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-vse++/lib/encoders.py:48: UserWarning: nn.init.constant is now deprecated in favor of nn.init.constant_.
  nn.init.constant(self.W[1].weight, 0)
/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-vse++/lib/encoders.py:49: UserWarning: nn.init.constant is now deprecated in favor of nn.init.constant_.
  nn.init.constant(self.W[1].bias, 0)
2021-06-22 19:50:20,674 loading configuration file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/config.json
2021-06-22 19:50:20,675 Model config {
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

2021-06-22 19:50:20,676 loading weights file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/pytorch_model.bin
2021-06-22 19:50:29,427 Use adam as the optimizer, with init lr 0.0005
2021-06-22 19:50:29,428 Image encoder is data paralleled now.
2021-06-22 19:50:29,428 runs/coco_butd_region_bert/log
2021-06-22 19:50:29,428 runs/coco_butd_region_bert
2021-06-22 19:50:29,430 image encoder trainable parameters: 20490144
2021-06-22 19:50:29,436 txt encoder trainable parameters: 137319072
/root/anaconda3/lib/python3.6/site-packages/torch/nn/modules/rnn.py:735: UserWarning: RNN module weights are not part of single contiguous chunk of memory. This means they need to be compacted at every call, possibly greatly increasing memory usage. To compact weights again call flatten_parameters(). (Triggered internally at  /pytorch/aten/src/ATen/native/cudnn/RNN.cpp:1264.)
  self.dropout, self.training, self.bidirectional, self.batch_first)
2021-06-22 19:53:56,876 Epoch: [0][199/4426]	Eit 200  lr 0.0005  Le 398.8092 (544.1175)	Time 0.950 (0.000)	Data 0.002 (0.000)	
^C^CException ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7f744d60e630>>
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
  File "/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-vse++/lib/vse.py", line 200, in train_emb
    clip_grad_norm_(self.params, self.grad_clip)
  File "/root/anaconda3/lib/python3.6/site-packages/torch/nn/utils/clip_grad.py", line 35, in clip_grad_norm_
    if clip_coef < 1:
KeyboardInterrupt
^CError in atexit._run_exitfuncs:
Traceback (most recent call last):
  File "/root/anaconda3/lib/python3.6/multiprocessing/popen_fork.py", line 28, in poll
    pid, sts = os.waitpid(self.pid, flag)
KeyboardInterrupt
[root@gpu1 vse_infty-master-my-graph-gru-vse++]# sh train_region.sh 
2021-06-22 19:55:34,674 Namespace(backbone_lr_factor=0.01, backbone_path='', backbone_source='detector', backbone_warmup_epochs=5, batch_size=128, data_name='f30k', data_path='../data/f30k', embed_size=1024, embedding_warmup_epochs=2, grad_clip=2.0, img_dim=2048, input_scale_factor=1, learning_rate=0.0005, log_step=200, logger_name='runs/f30k_butd_region_bert/log', lr_update=15, margin=0.2, max_violation=False, model_name='runs/f30k_butd_region_bert', no_imgnorm=False, no_txtnorm=False, num_epochs=25, optim='adam', precomp_enc_type='basic', reset_start_epoch=False, resume='', val_step=500, vocab_path='./vocab/', vse_mean_warmup_epochs=1, word_dim=300, workers=10)
2021-06-22 19:55:34,674 Model name '/public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased' not found in model shortcut name list (bert-base-uncased, bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, bert-base-multilingual-cased, bert-base-chinese, bert-base-german-cased, bert-large-uncased-whole-word-masking, bert-large-cased-whole-word-masking, bert-large-uncased-whole-word-masking-finetuned-squad, bert-large-cased-whole-word-masking-finetuned-squad, bert-base-cased-finetuned-mrpc). Assuming '/public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased' is a path or url to a directory containing tokenizer files.
2021-06-22 19:55:34,674 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/added_tokens.json. We won't load it.
2021-06-22 19:55:34,674 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/special_tokens_map.json. We won't load it.
2021-06-22 19:55:34,674 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/tokenizer_config.json. We won't load it.
2021-06-22 19:55:34,675 loading file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/vocab.txt
2021-06-22 19:55:34,675 loading file None
2021-06-22 19:55:34,675 loading file None
2021-06-22 19:55:34,675 loading file None
/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-vse++/lib/encoders.py:48: UserWarning: nn.init.constant is now deprecated in favor of nn.init.constant_.
  nn.init.constant(self.W[1].weight, 0)
/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-vse++/lib/encoders.py:49: UserWarning: nn.init.constant is now deprecated in favor of nn.init.constant_.
  nn.init.constant(self.W[1].bias, 0)
2021-06-22 19:55:46,509 loading configuration file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/config.json
2021-06-22 19:55:46,510 Model config {
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

2021-06-22 19:55:46,511 loading weights file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/pytorch_model.bin
2021-06-22 19:55:54,981 Use adam as the optimizer, with init lr 0.0005
2021-06-22 19:55:54,982 Image encoder is data paralleled now.
2021-06-22 19:55:54,982 runs/f30k_butd_region_bert/log
2021-06-22 19:55:54,982 runs/f30k_butd_region_bert
2021-06-22 19:55:54,984 image encoder trainable parameters: 20490144
2021-06-22 19:55:54,991 txt encoder trainable parameters: 137319072
/root/anaconda3/lib/python3.6/site-packages/torch/nn/modules/rnn.py:735: UserWarning: RNN module weights are not part of single contiguous chunk of memory. This means they need to be compacted at every call, possibly greatly increasing memory usage. To compact weights again call flatten_parameters(). (Triggered internally at  /pytorch/aten/src/ATen/native/cudnn/RNN.cpp:1264.)
  self.dropout, self.training, self.bidirectional, self.batch_first)
2021-06-22 19:59:47,215 Epoch: [0][199/1133]	Eit 200  lr 0.0005  Le 834.3099 (985.3883)	Time 1.064 (0.000)	Data 0.002 (0.000)	
2021-06-22 20:03:22,933 Epoch: [0][399/1133]	Eit 400  lr 0.0005  Le 723.5608 (765.4219)	Time 1.219 (0.000)	Data 0.002 (0.000)	
2021-06-22 20:06:57,954 Epoch: [0][599/1133]	Eit 600  lr 0.0005  Le 429.5767 (660.3787)	Time 1.049 (0.000)	Data 0.003 (0.000)	
2021-06-22 20:10:34,085 Epoch: [0][799/1133]	Eit 800  lr 0.0005  Le 351.5663 (598.5516)	Time 1.113 (0.000)	Data 0.002 (0.000)	
2021-06-22 20:14:10,084 Epoch: [0][999/1133]	Eit 1000  lr 0.0005  Le 362.9807 (554.2041)	Time 0.985 (0.000)	Data 0.002 (0.000)	
2021-06-22 20:16:35,138 Test: [0/40]	Le 380.3646 (380.3643)	Time 3.724 (0.000)	
2021-06-22 20:16:51,721 calculate similarity time: 0.08586287498474121
2021-06-22 20:16:52,219 Image to text: 44.3, 74.9, 85.6, 2.0, 7.5
2021-06-22 20:16:52,535 Text to image: 35.7, 66.5, 78.3, 3.0, 11.6
2021-06-22 20:16:52,535 Current rsum is 385.38
2021-06-22 20:16:57,075 runs/f30k_butd_region_bert/log
2021-06-22 20:16:57,075 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-22 20:16:57,079 image encoder trainable parameters: 20490144
2021-06-22 20:16:57,090 txt encoder trainable parameters: 137319072
2021-06-22 20:18:13,908 Epoch: [1][67/1133]	Eit 1200  lr 0.0005  Le 43.0426 (43.8803)	Time 1.188 (0.000)	Data 0.002 (0.000)	
2021-06-22 20:20:59,143 Epoch: [1][267/1133]	Eit 1400  lr 0.0005  Le 38.6963 (41.8015)	Time 0.679 (0.000)	Data 0.002 (0.000)	
2021-06-22 20:23:19,005 Epoch: [1][467/1133]	Eit 1600  lr 0.0005  Le 38.3807 (40.6136)	Time 0.650 (0.000)	Data 0.003 (0.000)	
2021-06-22 20:25:37,569 Epoch: [1][667/1133]	Eit 1800  lr 0.0005  Le 38.2965 (39.5772)	Time 0.634 (0.000)	Data 0.002 (0.000)	
2021-06-22 20:27:57,530 Epoch: [1][867/1133]	Eit 2000  lr 0.0005  Le 31.5539 (38.6375)	Time 0.648 (0.000)	Data 0.002 (0.000)	
2021-06-22 20:30:16,388 Epoch: [1][1067/1133]	Eit 2200  lr 0.0005  Le 36.8570 (37.9100)	Time 0.622 (0.000)	Data 0.002 (0.000)	
2021-06-22 20:31:05,202 Test: [0/40]	Le 59.6360 (59.6359)	Time 3.340 (0.000)	
2021-06-22 20:31:15,822 calculate similarity time: 0.053266286849975586
2021-06-22 20:31:16,306 Image to text: 69.6, 90.5, 95.4, 1.0, 3.8
2021-06-22 20:31:16,626 Text to image: 53.0, 80.5, 87.4, 1.0, 8.1
2021-06-22 20:31:16,627 Current rsum is 476.36
2021-06-22 20:31:20,848 runs/f30k_butd_region_bert/log
2021-06-22 20:31:20,848 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-22 20:31:20,851 image encoder trainable parameters: 20490144
2021-06-22 20:31:20,862 txt encoder trainable parameters: 137319072
2021-06-22 20:32:57,975 Epoch: [2][135/1133]	Eit 2400  lr 0.0005  Le 30.5260 (31.1414)	Time 0.598 (0.000)	Data 0.010 (0.000)	
2021-06-22 20:35:16,718 Epoch: [2][335/1133]	Eit 2600  lr 0.0005  Le 30.2500 (31.1089)	Time 0.642 (0.000)	Data 0.002 (0.000)	
2021-06-22 20:37:37,165 Epoch: [2][535/1133]	Eit 2800  lr 0.0005  Le 29.8994 (30.9278)	Time 0.671 (0.000)	Data 0.002 (0.000)	
2021-06-22 20:39:56,480 Epoch: [2][735/1133]	Eit 3000  lr 0.0005  Le 32.8105 (30.7787)	Time 0.652 (0.000)	Data 0.002 (0.000)	
2021-06-22 20:42:16,869 Epoch: [2][935/1133]	Eit 3200  lr 0.0005  Le 33.7345 (30.5823)	Time 0.633 (0.000)	Data 0.002 (0.000)	
2021-06-22 20:44:37,690 Test: [0/40]	Le 58.3636 (58.3635)	Time 3.608 (0.000)	
2021-06-22 20:44:48,154 calculate similarity time: 0.04856061935424805
2021-06-22 20:44:48,656 Image to text: 73.9, 93.6, 96.4, 1.0, 2.6
2021-06-22 20:44:49,009 Text to image: 56.2, 82.5, 89.4, 1.0, 7.1
2021-06-22 20:44:49,009 Current rsum is 491.9
2021-06-22 20:44:53,129 runs/f30k_butd_region_bert/log
2021-06-22 20:44:53,129 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-22 20:44:53,133 image encoder trainable parameters: 20490144
2021-06-22 20:44:53,143 txt encoder trainable parameters: 137319072
2021-06-22 20:44:59,340 Epoch: [3][3/1133]	Eit 3400  lr 0.0005  Le 26.2073 (25.2730)	Time 0.710 (0.000)	Data 0.002 (0.000)	
2021-06-22 20:47:18,792 Epoch: [3][203/1133]	Eit 3600  lr 0.0005  Le 28.0133 (27.1161)	Time 0.608 (0.000)	Data 0.002 (0.000)	
2021-06-22 20:49:38,196 Epoch: [3][403/1133]	Eit 3800  lr 0.0005  Le 24.2597 (26.9053)	Time 0.643 (0.000)	Data 0.002 (0.000)	
2021-06-22 20:51:57,973 Epoch: [3][603/1133]	Eit 4000  lr 0.0005  Le 26.4909 (26.8301)	Time 0.630 (0.000)	Data 0.002 (0.000)	
2021-06-22 20:54:16,790 Epoch: [3][803/1133]	Eit 4200  lr 0.0005  Le 30.2434 (26.8043)	Time 0.675 (0.000)	Data 0.002 (0.000)	
2021-06-22 20:56:37,280 Epoch: [3][1003/1133]	Eit 4400  lr 0.0005  Le 25.7705 (26.7847)	Time 0.721 (0.000)	Data 0.002 (0.000)	
2021-06-22 20:58:12,054 Test: [0/40]	Le 58.3453 (58.3453)	Time 3.796 (0.000)	
2021-06-22 20:58:22,486 calculate similarity time: 0.04832601547241211
2021-06-22 20:58:22,954 Image to text: 77.6, 94.6, 97.7, 1.0, 2.3
2021-06-22 20:58:23,268 Text to image: 58.8, 84.3, 90.3, 1.0, 6.5
2021-06-22 20:58:23,268 Current rsum is 503.35999999999996
2021-06-22 20:58:27,665 runs/f30k_butd_region_bert/log
2021-06-22 20:58:27,665 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-22 20:58:27,668 image encoder trainable parameters: 20490144
2021-06-22 20:58:27,679 txt encoder trainable parameters: 137319072
2021-06-22 20:59:22,363 Epoch: [4][71/1133]	Eit 4600  lr 0.0005  Le 26.9772 (24.3010)	Time 0.632 (0.000)	Data 0.002 (0.000)	
2021-06-22 21:01:44,376 Epoch: [4][271/1133]	Eit 4800  lr 0.0005  Le 21.6426 (24.3670)	Time 0.670 (0.000)	Data 0.002 (0.000)	
2021-06-22 21:04:03,997 Epoch: [4][471/1133]	Eit 5000  lr 0.0005  Le 21.1315 (24.3003)	Time 0.961 (0.000)	Data 0.002 (0.000)	
2021-06-22 21:06:23,804 Epoch: [4][671/1133]	Eit 5200  lr 0.0005  Le 23.0708 (24.2647)	Time 0.720 (0.000)	Data 0.002 (0.000)	
2021-06-22 21:08:41,822 Epoch: [4][871/1133]	Eit 5400  lr 0.0005  Le 25.4640 (24.2344)	Time 0.762 (0.000)	Data 0.002 (0.000)	
2021-06-22 21:11:01,359 Epoch: [4][1071/1133]	Eit 5600  lr 0.0005  Le 22.5124 (24.2383)	Time 0.616 (0.000)	Data 0.002 (0.000)	
2021-06-22 21:11:46,931 Test: [0/40]	Le 58.4612 (58.4612)	Time 3.456 (0.000)	
2021-06-22 21:11:57,495 calculate similarity time: 0.05946969985961914
2021-06-22 21:11:58,014 Image to text: 77.2, 94.5, 97.8, 1.0, 2.2
2021-06-22 21:11:58,484 Text to image: 59.9, 84.5, 90.6, 1.0, 6.5
2021-06-22 21:11:58,484 Current rsum is 504.5
2021-06-22 21:12:02,311 runs/f30k_butd_region_bert/log
2021-06-22 21:12:02,311 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-22 21:12:02,316 image encoder trainable parameters: 20490144
2021-06-22 21:12:02,330 txt encoder trainable parameters: 137319072
2021-06-22 21:13:42,283 Epoch: [5][139/1133]	Eit 5800  lr 0.0005  Le 23.7123 (21.7943)	Time 0.793 (0.000)	Data 0.002 (0.000)	
2021-06-22 21:16:02,687 Epoch: [5][339/1133]	Eit 6000  lr 0.0005  Le 22.7334 (22.0034)	Time 0.773 (0.000)	Data 0.002 (0.000)	
2021-06-22 21:18:22,755 Epoch: [5][539/1133]	Eit 6200  lr 0.0005  Le 23.5470 (22.1246)	Time 0.647 (0.000)	Data 0.002 (0.000)	
2021-06-22 21:20:43,485 Epoch: [5][739/1133]	Eit 6400  lr 0.0005  Le 24.9367 (22.1817)	Time 0.670 (0.000)	Data 0.002 (0.000)	
2021-06-22 21:23:03,478 Epoch: [5][939/1133]	Eit 6600  lr 0.0005  Le 20.9931 (22.2156)	Time 0.802 (0.000)	Data 0.002 (0.000)	
2021-06-22 21:25:19,672 Test: [0/40]	Le 59.3073 (59.3073)	Time 3.346 (0.000)	
2021-06-22 21:25:30,183 calculate similarity time: 0.04862618446350098
2021-06-22 21:25:30,702 Image to text: 78.1, 94.3, 97.3, 1.0, 2.4
2021-06-22 21:25:31,014 Text to image: 59.8, 85.2, 90.6, 1.0, 6.7
2021-06-22 21:25:31,015 Current rsum is 505.34
2021-06-22 21:25:34,804 runs/f30k_butd_region_bert/log
2021-06-22 21:25:34,804 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-22 21:25:34,808 image encoder trainable parameters: 20490144
2021-06-22 21:25:34,821 txt encoder trainable parameters: 137319072
2021-06-22 21:25:43,759 Epoch: [6][7/1133]	Eit 6800  lr 0.0005  Le 20.8099 (19.6643)	Time 0.690 (0.000)	Data 0.002 (0.000)	
2021-06-22 21:28:03,393 Epoch: [6][207/1133]	Eit 7000  lr 0.0005  Le 21.1774 (20.4746)	Time 0.639 (0.000)	Data 0.002 (0.000)	
2021-06-22 21:30:22,741 Epoch: [6][407/1133]	Eit 7200  lr 0.0005  Le 18.2607 (20.5059)	Time 0.759 (0.000)	Data 0.002 (0.000)	
2021-06-22 21:32:42,365 Epoch: [6][607/1133]	Eit 7400  lr 0.0005  Le 15.3504 (20.4598)	Time 0.696 (0.000)	Data 0.002 (0.000)	
2021-06-22 21:35:02,550 Epoch: [6][807/1133]	Eit 7600  lr 0.0005  Le 19.2162 (20.5384)	Time 0.668 (0.000)	Data 0.002 (0.000)	
2021-06-22 21:37:23,610 Epoch: [6][1007/1133]	Eit 7800  lr 0.0005  Le 19.3106 (20.5570)	Time 0.615 (0.000)	Data 0.002 (0.000)	
2021-06-22 21:38:53,516 Test: [0/40]	Le 58.1684 (58.1684)	Time 3.591 (0.000)	
2021-06-22 21:39:03,981 calculate similarity time: 0.05596017837524414
2021-06-22 21:39:04,487 Image to text: 78.9, 94.5, 97.1, 1.0, 2.6
2021-06-22 21:39:04,853 Text to image: 60.8, 85.5, 91.4, 1.0, 6.2
2021-06-22 21:39:04,853 Current rsum is 508.09999999999997
2021-06-22 21:39:08,515 runs/f30k_butd_region_bert/log
2021-06-22 21:39:08,515 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-22 21:39:08,518 image encoder trainable parameters: 20490144
2021-06-22 21:39:08,531 txt encoder trainable parameters: 137319072
2021-06-22 21:40:04,644 Epoch: [7][75/1133]	Eit 8000  lr 0.0005  Le 16.8124 (19.1721)	Time 0.809 (0.000)	Data 0.002 (0.000)	
2021-06-22 21:42:23,928 Epoch: [7][275/1133]	Eit 8200  lr 0.0005  Le 22.2859 (19.3180)	Time 0.708 (0.000)	Data 0.002 (0.000)	
2021-06-22 21:44:43,929 Epoch: [7][475/1133]	Eit 8400  lr 0.0005  Le 18.4963 (19.2856)	Time 0.691 (0.000)	Data 0.002 (0.000)	
2021-06-22 21:47:02,536 Epoch: [7][675/1133]	Eit 8600  lr 0.0005  Le 17.8308 (19.3017)	Time 0.724 (0.000)	Data 0.002 (0.000)	
2021-06-22 21:49:22,528 Epoch: [7][875/1133]	Eit 8800  lr 0.0005  Le 20.0146 (19.3315)	Time 0.660 (0.000)	Data 0.002 (0.000)	
2021-06-22 21:51:42,776 Epoch: [7][1075/1133]	Eit 9000  lr 0.0005  Le 19.6973 (19.3722)	Time 0.639 (0.000)	Data 0.010 (0.000)	
2021-06-22 21:52:26,116 Test: [0/40]	Le 58.4978 (58.4978)	Time 3.314 (0.000)	
2021-06-22 21:52:36,708 calculate similarity time: 0.050208330154418945
2021-06-22 21:52:37,220 Image to text: 79.4, 95.1, 97.5, 1.0, 2.2
2021-06-22 21:52:37,562 Text to image: 61.3, 85.5, 91.4, 1.0, 6.2
2021-06-22 21:52:37,562 Current rsum is 510.20000000000005
2021-06-22 21:52:41,631 runs/f30k_butd_region_bert/log
2021-06-22 21:52:41,632 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-22 21:52:41,635 image encoder trainable parameters: 20490144
2021-06-22 21:52:41,645 txt encoder trainable parameters: 137319072
2021-06-22 21:54:25,237 Epoch: [8][143/1133]	Eit 9200  lr 0.0005  Le 16.6730 (17.8457)	Time 0.673 (0.000)	Data 0.002 (0.000)	
2021-06-22 21:56:45,465 Epoch: [8][343/1133]	Eit 9400  lr 0.0005  Le 20.3843 (17.8559)	Time 0.658 (0.000)	Data 0.002 (0.000)	
2021-06-22 21:59:05,275 Epoch: [8][543/1133]	Eit 9600  lr 0.0005  Le 15.4034 (17.8598)	Time 0.962 (0.000)	Data 0.002 (0.000)	
2021-06-22 22:01:25,431 Epoch: [8][743/1133]	Eit 9800  lr 0.0005  Le 16.7697 (18.0227)	Time 0.790 (0.000)	Data 0.002 (0.000)	
2021-06-22 22:03:45,869 Epoch: [8][943/1133]	Eit 10000  lr 0.0005  Le 17.2053 (18.0409)	Time 0.728 (0.000)	Data 0.002 (0.000)	
2021-06-22 22:06:01,051 Test: [0/40]	Le 58.6385 (58.6385)	Time 3.763 (0.000)	
2021-06-22 22:06:11,562 calculate similarity time: 0.05839848518371582
2021-06-22 22:06:12,065 Image to text: 79.8, 95.5, 97.8, 1.0, 2.4
2021-06-22 22:06:12,422 Text to image: 60.8, 85.3, 90.9, 1.0, 6.9
2021-06-22 22:06:12,422 Current rsum is 510.04
2021-06-22 22:06:14,520 runs/f30k_butd_region_bert/log
2021-06-22 22:06:14,520 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-22 22:06:14,524 image encoder trainable parameters: 20490144
2021-06-22 22:06:14,537 txt encoder trainable parameters: 137319072
2021-06-22 22:06:26,489 Epoch: [9][11/1133]	Eit 10200  lr 0.0005  Le 15.7997 (17.0482)	Time 0.771 (0.000)	Data 0.002 (0.000)	
2021-06-22 22:08:47,167 Epoch: [9][211/1133]	Eit 10400  lr 0.0005  Le 19.9400 (16.8837)	Time 0.664 (0.000)	Data 0.002 (0.000)	
2021-06-22 22:11:05,153 Epoch: [9][411/1133]	Eit 10600  lr 0.0005  Le 18.6165 (16.9119)	Time 0.700 (0.000)	Data 0.002 (0.000)	
2021-06-22 22:13:25,370 Epoch: [9][611/1133]	Eit 10800  lr 0.0005  Le 22.1589 (17.0211)	Time 0.747 (0.000)	Data 0.002 (0.000)	
2021-06-22 22:15:45,364 Epoch: [9][811/1133]	Eit 11000  lr 0.0005  Le 15.1414 (17.0405)	Time 0.678 (0.000)	Data 0.002 (0.000)	
2021-06-22 22:18:05,948 Epoch: [9][1011/1133]	Eit 11200  lr 0.0005  Le 14.6349 (17.0720)	Time 0.852 (0.000)	Data 0.002 (0.000)	
2021-06-22 22:19:34,935 Test: [0/40]	Le 58.3539 (58.3538)	Time 3.568 (0.000)	
2021-06-22 22:19:45,442 calculate similarity time: 0.05915355682373047
2021-06-22 22:19:45,944 Image to text: 78.6, 94.8, 97.1, 1.0, 2.3
2021-06-22 22:19:46,381 Text to image: 60.8, 85.6, 91.2, 1.0, 6.3
2021-06-22 22:19:46,381 Current rsum is 508.1
2021-06-22 22:19:48,920 runs/f30k_butd_region_bert/log
2021-06-22 22:19:48,921 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-22 22:19:48,925 image encoder trainable parameters: 20490144
2021-06-22 22:19:48,937 txt encoder trainable parameters: 137319072
2021-06-22 22:20:48,775 Epoch: [10][79/1133]	Eit 11400  lr 0.0005  Le 15.7993 (15.7616)	Time 0.794 (0.000)	Data 0.002 (0.000)	
2021-06-22 22:23:07,834 Epoch: [10][279/1133]	Eit 11600  lr 0.0005  Le 15.2853 (15.8398)	Time 0.659 (0.000)	Data 0.002 (0.000)	
2021-06-22 22:25:28,211 Epoch: [10][479/1133]	Eit 11800  lr 0.0005  Le 15.8205 (15.9288)	Time 0.660 (0.000)	Data 0.002 (0.000)	
2021-06-22 22:27:48,488 Epoch: [10][679/1133]	Eit 12000  lr 0.0005  Le 15.6410 (15.9504)	Time 0.669 (0.000)	Data 0.003 (0.000)	
2021-06-22 22:30:09,085 Epoch: [10][879/1133]	Eit 12200  lr 0.0005  Le 15.5746 (16.0276)	Time 0.700 (0.000)	Data 0.004 (0.000)	
2021-06-22 22:32:28,793 Epoch: [10][1079/1133]	Eit 12400  lr 0.0005  Le 14.3901 (16.1532)	Time 0.754 (0.000)	Data 0.002 (0.000)	
2021-06-22 22:33:09,227 Test: [0/40]	Le 58.5039 (58.5039)	Time 4.004 (0.000)	
2021-06-22 22:33:19,813 calculate similarity time: 0.0648496150970459
2021-06-22 22:33:20,330 Image to text: 78.5, 95.5, 98.1, 1.0, 2.1
2021-06-22 22:33:20,693 Text to image: 62.2, 86.5, 91.5, 1.0, 6.5
2021-06-22 22:33:20,694 Current rsum is 512.36
2021-06-22 22:33:25,230 runs/f30k_butd_region_bert/log
2021-06-22 22:33:25,231 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-22 22:33:25,235 image encoder trainable parameters: 20490144
2021-06-22 22:33:25,248 txt encoder trainable parameters: 137319072
2021-06-22 22:35:12,479 Epoch: [11][147/1133]	Eit 12600  lr 0.0005  Le 14.2832 (14.9889)	Time 1.213 (0.000)	Data 0.002 (0.000)	
2021-06-22 22:38:39,563 Epoch: [11][347/1133]	Eit 12800  lr 0.0005  Le 18.5723 (15.1418)	Time 1.072 (0.000)	Data 0.002 (0.000)	
2021-06-22 22:41:24,008 Epoch: [11][547/1133]	Eit 13000  lr 0.0005  Le 16.7161 (15.1971)	Time 1.379 (0.000)	Data 0.002 (0.000)	
2021-06-22 22:45:20,629 Epoch: [11][747/1133]	Eit 13200  lr 0.0005  Le 12.6609 (15.3293)	Time 1.447 (0.000)	Data 0.002 (0.000)	
2021-06-22 22:49:16,404 Epoch: [11][947/1133]	Eit 13400  lr 0.0005  Le 15.9866 (15.4069)	Time 1.009 (0.000)	Data 0.002 (0.000)	
2021-06-22 22:52:55,863 Test: [0/40]	Le 59.4065 (59.4064)	Time 3.793 (0.000)	
2021-06-22 22:53:13,678 calculate similarity time: 0.07875609397888184
2021-06-22 22:53:14,210 Image to text: 79.1, 95.0, 97.0, 1.0, 2.2
2021-06-22 22:53:14,540 Text to image: 61.9, 86.3, 91.6, 1.0, 6.4
2021-06-22 22:53:14,540 Current rsum is 510.94000000000005
2021-06-22 22:53:16,715 runs/f30k_butd_region_bert/log
2021-06-22 22:53:16,715 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-22 22:53:16,718 image encoder trainable parameters: 20490144
2021-06-22 22:53:16,729 txt encoder trainable parameters: 137319072
2021-06-22 22:53:38,608 Epoch: [12][15/1133]	Eit 13600  lr 0.0005  Le 15.2695 (14.3113)	Time 1.288 (0.000)	Data 0.002 (0.000)	
2021-06-22 22:57:31,568 Epoch: [12][215/1133]	Eit 13800  lr 0.0005  Le 14.9235 (14.3668)	Time 1.575 (0.000)	Data 0.002 (0.000)	
2021-06-22 23:01:24,281 Epoch: [12][415/1133]	Eit 14000  lr 0.0005  Le 19.8809 (14.4305)	Time 1.136 (0.000)	Data 0.002 (0.000)	
2021-06-22 23:05:17,137 Epoch: [12][615/1133]	Eit 14200  lr 0.0005  Le 16.2006 (14.5170)	Time 1.340 (0.000)	Data 0.002 (0.000)	
2021-06-22 23:09:13,110 Epoch: [12][815/1133]	Eit 14400  lr 0.0005  Le 13.2427 (14.5341)	Time 1.177 (0.000)	Data 0.002 (0.000)	
2021-06-22 23:13:02,970 Epoch: [12][1015/1133]	Eit 14600  lr 0.0005  Le 16.8320 (14.5919)	Time 1.345 (0.000)	Data 0.002 (0.000)	
2021-06-22 23:15:22,946 Test: [0/40]	Le 58.5375 (58.5375)	Time 3.587 (0.000)	
2021-06-22 23:15:40,769 calculate similarity time: 0.07072281837463379
2021-06-22 23:15:41,332 Image to text: 80.5, 95.2, 97.9, 1.0, 2.2
2021-06-22 23:15:41,648 Text to image: 61.7, 86.0, 91.3, 1.0, 6.3
2021-06-22 23:15:41,648 Current rsum is 512.64
2021-06-22 23:15:46,054 runs/f30k_butd_region_bert/log
2021-06-22 23:15:46,054 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-22 23:15:46,058 image encoder trainable parameters: 20490144
2021-06-22 23:15:46,069 txt encoder trainable parameters: 137319072
2021-06-22 23:17:23,504 Epoch: [13][83/1133]	Eit 14800  lr 0.0005  Le 14.1209 (13.3663)	Time 1.358 (0.000)	Data 0.008 (0.000)	
2021-06-22 23:21:13,261 Epoch: [13][283/1133]	Eit 15000  lr 0.0005  Le 10.6627 (13.6680)	Time 1.061 (0.000)	Data 0.002 (0.000)	
2021-06-22 23:23:46,690 Epoch: [13][483/1133]	Eit 15200  lr 0.0005  Le 14.6728 (13.7136)	Time 0.661 (0.000)	Data 0.002 (0.000)	
2021-06-22 23:26:07,353 Epoch: [13][683/1133]	Eit 15400  lr 0.0005  Le 11.7663 (13.7870)	Time 0.658 (0.000)	Data 0.002 (0.000)	
2021-06-22 23:28:26,684 Epoch: [13][883/1133]	Eit 15600  lr 0.0005  Le 16.2853 (13.8798)	Time 0.740 (0.000)	Data 0.002 (0.000)	
2021-06-22 23:30:45,860 Epoch: [13][1083/1133]	Eit 15800  lr 0.0005  Le 12.6628 (13.9614)	Time 0.812 (0.000)	Data 0.009 (0.000)	
2021-06-22 23:31:24,400 Test: [0/40]	Le 59.8199 (59.8198)	Time 3.536 (0.000)	
2021-06-22 23:31:34,963 calculate similarity time: 0.06241154670715332
2021-06-22 23:31:35,486 Image to text: 79.9, 95.5, 97.9, 1.0, 2.0
2021-06-22 23:31:35,828 Text to image: 61.9, 86.1, 91.4, 1.0, 6.6
2021-06-22 23:31:35,828 Current rsum is 512.64
2021-06-22 23:31:38,085 runs/f30k_butd_region_bert/log
2021-06-22 23:31:38,086 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-22 23:31:38,091 image encoder trainable parameters: 20490144
2021-06-22 23:31:38,106 txt encoder trainable parameters: 137319072
2021-06-22 23:33:27,482 Epoch: [14][151/1133]	Eit 16000  lr 0.0005  Le 12.2371 (13.1839)	Time 0.671 (0.000)	Data 0.002 (0.000)	
2021-06-22 23:35:46,381 Epoch: [14][351/1133]	Eit 16200  lr 0.0005  Le 14.7809 (13.2333)	Time 0.656 (0.000)	Data 0.002 (0.000)	
2021-06-22 23:38:05,085 Epoch: [14][551/1133]	Eit 16400  lr 0.0005  Le 15.5575 (13.3099)	Time 0.687 (0.000)	Data 0.002 (0.000)	
2021-06-22 23:40:23,008 Epoch: [14][751/1133]	Eit 16600  lr 0.0005  Le 14.1496 (13.3805)	Time 0.678 (0.000)	Data 0.002 (0.000)	
2021-06-22 23:42:42,227 Epoch: [14][951/1133]	Eit 16800  lr 0.0005  Le 12.6150 (13.4327)	Time 0.677 (0.000)	Data 0.002 (0.000)	
2021-06-22 23:44:53,106 Test: [0/40]	Le 58.8797 (58.8797)	Time 3.852 (0.000)	
2021-06-22 23:45:03,531 calculate similarity time: 0.048479557037353516
2021-06-22 23:45:04,033 Image to text: 79.3, 94.7, 97.5, 1.0, 2.2
2021-06-22 23:45:04,358 Text to image: 61.9, 85.5, 91.4, 1.0, 6.9
2021-06-22 23:45:04,358 Current rsum is 510.24
2021-06-22 23:45:06,456 runs/f30k_butd_region_bert/log
2021-06-22 23:45:06,456 runs/f30k_butd_region_bert
2021-06-22 23:45:06,456 Current epoch num is 15, decrease all lr by 10
2021-06-22 23:45:06,456 new lr 5e-05
2021-06-22 23:45:06,456 new lr 5e-06
2021-06-22 23:45:06,456 new lr 5e-05
Use VSE++ objective.
2021-06-22 23:45:06,460 image encoder trainable parameters: 20490144
2021-06-22 23:45:06,473 txt encoder trainable parameters: 137319072
2021-06-22 23:45:23,911 Epoch: [15][19/1133]	Eit 17000  lr 5e-05  Le 9.8656 (12.6252)	Time 0.854 (0.000)	Data 0.002 (0.000)	
2021-06-22 23:47:42,899 Epoch: [15][219/1133]	Eit 17200  lr 5e-05  Le 14.0913 (11.7900)	Time 0.721 (0.000)	Data 0.002 (0.000)	
2021-06-22 23:50:01,437 Epoch: [15][419/1133]	Eit 17400  lr 5e-05  Le 10.2432 (11.5805)	Time 0.695 (0.000)	Data 0.002 (0.000)	
2021-06-22 23:52:21,557 Epoch: [15][619/1133]	Eit 17600  lr 5e-05  Le 10.2487 (11.4760)	Time 0.643 (0.000)	Data 0.002 (0.000)	
2021-06-22 23:54:40,816 Epoch: [15][819/1133]	Eit 17800  lr 5e-05  Le 11.0013 (11.4058)	Time 0.768 (0.000)	Data 0.002 (0.000)	
2021-06-22 23:57:01,251 Epoch: [15][1019/1133]	Eit 18000  lr 5e-05  Le 11.6318 (11.3187)	Time 0.632 (0.000)	Data 0.002 (0.000)	
2021-06-22 23:58:24,238 Test: [0/40]	Le 58.9815 (58.9815)	Time 3.742 (0.000)	
2021-06-22 23:58:34,727 calculate similarity time: 0.050974130630493164
2021-06-22 23:58:35,234 Image to text: 80.9, 96.1, 98.1, 1.0, 1.9
2021-06-22 23:58:35,585 Text to image: 63.1, 86.5, 91.7, 1.0, 6.8
2021-06-22 23:58:35,585 Current rsum is 516.36
2021-06-22 23:58:40,184 runs/f30k_butd_region_bert/log
2021-06-22 23:58:40,184 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-22 23:58:40,188 image encoder trainable parameters: 20490144
2021-06-22 23:58:40,201 txt encoder trainable parameters: 137319072
2021-06-22 23:59:44,514 Epoch: [16][87/1133]	Eit 18200  lr 5e-05  Le 11.9740 (10.5761)	Time 0.630 (0.000)	Data 0.002 (0.000)	
2021-06-23 00:02:04,584 Epoch: [16][287/1133]	Eit 18400  lr 5e-05  Le 10.6330 (10.5272)	Time 0.701 (0.000)	Data 0.002 (0.000)	
2021-06-23 00:04:22,467 Epoch: [16][487/1133]	Eit 18600  lr 5e-05  Le 7.8973 (10.4143)	Time 0.683 (0.000)	Data 0.002 (0.000)	
2021-06-23 00:06:41,771 Epoch: [16][687/1133]	Eit 18800  lr 5e-05  Le 10.6128 (10.4166)	Time 0.631 (0.000)	Data 0.002 (0.000)	
2021-06-23 00:09:04,052 Epoch: [16][887/1133]	Eit 19000  lr 5e-05  Le 8.8755 (10.4053)	Time 0.700 (0.000)	Data 0.002 (0.000)	
2021-06-23 00:11:26,003 Epoch: [16][1087/1133]	Eit 19200  lr 5e-05  Le 10.2429 (10.3905)	Time 0.688 (0.000)	Data 0.002 (0.000)	
2021-06-23 00:12:00,789 Test: [0/40]	Le 59.1221 (59.1220)	Time 3.445 (0.000)	
2021-06-23 00:12:11,371 calculate similarity time: 0.05277895927429199
2021-06-23 00:12:11,775 Image to text: 81.4, 95.9, 98.4, 1.0, 2.0
2021-06-23 00:12:12,101 Text to image: 63.4, 86.5, 91.6, 1.0, 6.9
2021-06-23 00:12:12,101 Current rsum is 517.2400000000001
2021-06-23 00:12:16,146 runs/f30k_butd_region_bert/log
2021-06-23 00:12:16,147 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-23 00:12:16,151 image encoder trainable parameters: 20490144
2021-06-23 00:12:16,162 txt encoder trainable parameters: 137319072
2021-06-23 00:14:08,858 Epoch: [17][155/1133]	Eit 19400  lr 5e-05  Le 9.9680 (10.1031)	Time 0.912 (0.000)	Data 0.002 (0.000)	
2021-06-23 00:16:28,278 Epoch: [17][355/1133]	Eit 19600  lr 5e-05  Le 11.8767 (10.0348)	Time 0.700 (0.000)	Data 0.002 (0.000)	
2021-06-23 00:18:47,930 Epoch: [17][555/1133]	Eit 19800  lr 5e-05  Le 12.2664 (10.0391)	Time 0.652 (0.000)	Data 0.002 (0.000)	
2021-06-23 00:21:08,533 Epoch: [17][755/1133]	Eit 20000  lr 5e-05  Le 11.3184 (10.0663)	Time 0.688 (0.000)	Data 0.002 (0.000)	
2021-06-23 00:23:28,616 Epoch: [17][955/1133]	Eit 20200  lr 5e-05  Le 8.1566 (10.0848)	Time 0.736 (0.000)	Data 0.002 (0.000)	
2021-06-23 00:25:37,469 Test: [0/40]	Le 58.8777 (58.8777)	Time 3.825 (0.000)	
2021-06-23 00:25:47,996 calculate similarity time: 0.06431937217712402
2021-06-23 00:25:48,415 Image to text: 82.4, 96.2, 98.4, 1.0, 1.9
2021-06-23 00:25:48,726 Text to image: 63.3, 86.5, 91.7, 1.0, 6.9
2021-06-23 00:25:48,727 Current rsum is 518.5600000000001
2021-06-23 00:25:52,591 runs/f30k_butd_region_bert/log
2021-06-23 00:25:52,591 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-23 00:25:52,595 image encoder trainable parameters: 20490144
2021-06-23 00:25:52,608 txt encoder trainable parameters: 137319072
2021-06-23 00:26:13,061 Epoch: [18][23/1133]	Eit 20400  lr 5e-05  Le 9.9279 (9.1165)	Time 0.690 (0.000)	Data 0.003 (0.000)	
2021-06-23 00:28:33,525 Epoch: [18][223/1133]	Eit 20600  lr 5e-05  Le 9.5229 (9.7007)	Time 0.696 (0.000)	Data 0.002 (0.000)	
2021-06-23 00:30:53,333 Epoch: [18][423/1133]	Eit 20800  lr 5e-05  Le 9.1063 (9.7583)	Time 0.733 (0.000)	Data 0.002 (0.000)	
2021-06-23 00:33:15,104 Epoch: [18][623/1133]	Eit 21000  lr 5e-05  Le 8.6776 (9.8364)	Time 0.707 (0.000)	Data 0.002 (0.000)	
2021-06-23 00:35:33,979 Epoch: [18][823/1133]	Eit 21200  lr 5e-05  Le 9.3993 (9.8162)	Time 0.609 (0.000)	Data 0.002 (0.000)	
2021-06-23 00:37:53,045 Epoch: [18][1023/1133]	Eit 21400  lr 5e-05  Le 9.3228 (9.8014)	Time 0.708 (0.000)	Data 0.002 (0.000)	
2021-06-23 00:39:12,485 Test: [0/40]	Le 58.9857 (58.9856)	Time 3.686 (0.000)	
2021-06-23 00:39:22,983 calculate similarity time: 0.0485227108001709
2021-06-23 00:39:23,488 Image to text: 82.2, 96.1, 98.2, 1.0, 1.9
2021-06-23 00:39:23,842 Text to image: 63.3, 86.7, 91.9, 1.0, 6.9
2021-06-23 00:39:23,842 Current rsum is 518.36
2021-06-23 00:39:25,890 runs/f30k_butd_region_bert/log
2021-06-23 00:39:25,891 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-23 00:39:25,894 image encoder trainable parameters: 20490144
2021-06-23 00:39:25,908 txt encoder trainable parameters: 137319072
2021-06-23 00:40:33,749 Epoch: [19][91/1133]	Eit 21600  lr 5e-05  Le 10.2179 (9.6873)	Time 0.626 (0.000)	Data 0.002 (0.000)	
2021-06-23 00:42:53,684 Epoch: [19][291/1133]	Eit 21800  lr 5e-05  Le 10.3661 (9.7485)	Time 0.736 (0.000)	Data 0.003 (0.000)	
2021-06-23 00:45:12,951 Epoch: [19][491/1133]	Eit 22000  lr 5e-05  Le 12.3618 (9.6525)	Time 0.721 (0.000)	Data 0.002 (0.000)	
2021-06-23 00:47:33,758 Epoch: [19][691/1133]	Eit 22200  lr 5e-05  Le 8.2337 (9.6138)	Time 0.636 (0.000)	Data 0.002 (0.000)	
2021-06-23 00:49:55,246 Epoch: [19][891/1133]	Eit 22400  lr 5e-05  Le 10.2477 (9.6007)	Time 0.730 (0.000)	Data 0.002 (0.000)	
2021-06-23 00:52:15,182 Epoch: [19][1091/1133]	Eit 22600  lr 5e-05  Le 8.9235 (9.6116)	Time 0.609 (0.000)	Data 0.002 (0.000)	
2021-06-23 00:52:47,302 Test: [0/40]	Le 58.8428 (58.8428)	Time 3.507 (0.000)	
2021-06-23 00:52:57,730 calculate similarity time: 0.04881405830383301
2021-06-23 00:52:58,235 Image to text: 82.5, 96.1, 98.2, 1.0, 1.9
2021-06-23 00:52:58,566 Text to image: 63.8, 86.8, 91.7, 1.0, 6.8
2021-06-23 00:52:58,566 Current rsum is 519.0400000000001
2021-06-23 00:53:02,449 runs/f30k_butd_region_bert/log
2021-06-23 00:53:02,450 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-23 00:53:02,453 image encoder trainable parameters: 20490144
2021-06-23 00:53:02,466 txt encoder trainable parameters: 137319072
2021-06-23 00:54:59,220 Epoch: [20][159/1133]	Eit 22800  lr 5e-05  Le 9.7633 (9.6768)	Time 0.727 (0.000)	Data 0.002 (0.000)	
2021-06-23 00:57:18,809 Epoch: [20][359/1133]	Eit 23000  lr 5e-05  Le 8.8689 (9.6361)	Time 0.680 (0.000)	Data 0.002 (0.000)	
2021-06-23 00:59:38,853 Epoch: [20][559/1133]	Eit 23200  lr 5e-05  Le 8.1673 (9.5461)	Time 0.693 (0.000)	Data 0.002 (0.000)	
2021-06-23 01:01:58,768 Epoch: [20][759/1133]	Eit 23400  lr 5e-05  Le 8.1501 (9.5661)	Time 0.615 (0.000)	Data 0.002 (0.000)	
2021-06-23 01:04:18,298 Epoch: [20][959/1133]	Eit 23600  lr 5e-05  Le 10.3116 (9.5609)	Time 0.646 (0.000)	Data 0.002 (0.000)	
2021-06-23 01:06:21,408 Test: [0/40]	Le 58.9513 (58.9513)	Time 3.487 (0.000)	
2021-06-23 01:06:31,919 calculate similarity time: 0.04976201057434082
2021-06-23 01:06:32,424 Image to text: 81.9, 96.6, 98.4, 1.0, 1.9
2021-06-23 01:06:32,791 Text to image: 63.7, 86.5, 91.7, 1.0, 7.0
2021-06-23 01:06:32,791 Current rsum is 518.78
2021-06-23 01:06:34,579 runs/f30k_butd_region_bert/log
2021-06-23 01:06:34,579 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-23 01:06:34,583 image encoder trainable parameters: 20490144
2021-06-23 01:06:34,595 txt encoder trainable parameters: 137319072
2021-06-23 01:06:57,157 Epoch: [21][27/1133]	Eit 23800  lr 5e-05  Le 8.5164 (9.2334)	Time 0.797 (0.000)	Data 0.002 (0.000)	
2021-06-23 01:09:16,940 Epoch: [21][227/1133]	Eit 24000  lr 5e-05  Le 7.6250 (9.2716)	Time 0.810 (0.000)	Data 0.002 (0.000)	
2021-06-23 01:11:35,984 Epoch: [21][427/1133]	Eit 24200  lr 5e-05  Le 10.1161 (9.3361)	Time 0.736 (0.000)	Data 0.002 (0.000)	
2021-06-23 01:13:57,449 Epoch: [21][627/1133]	Eit 24400  lr 5e-05  Le 8.8999 (9.2367)	Time 0.694 (0.000)	Data 0.002 (0.000)	
2021-06-23 01:16:16,885 Epoch: [21][827/1133]	Eit 24600  lr 5e-05  Le 11.2494 (9.2281)	Time 0.795 (0.000)	Data 0.002 (0.000)	
2021-06-23 01:18:37,238 Epoch: [21][1027/1133]	Eit 24800  lr 5e-05  Le 8.0143 (9.2270)	Time 0.675 (0.000)	Data 0.002 (0.000)	
2021-06-23 01:19:53,357 Test: [0/40]	Le 58.8448 (58.8448)	Time 3.200 (0.000)	
2021-06-23 01:20:03,861 calculate similarity time: 0.05041360855102539
2021-06-23 01:20:04,371 Image to text: 81.2, 96.2, 98.4, 1.0, 1.9
2021-06-23 01:20:04,712 Text to image: 63.9, 86.7, 91.5, 1.0, 7.0
2021-06-23 01:20:04,712 Current rsum is 517.9
2021-06-23 01:20:06,639 runs/f30k_butd_region_bert/log
2021-06-23 01:20:06,639 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-23 01:20:06,642 image encoder trainable parameters: 20490144
2021-06-23 01:20:06,655 txt encoder trainable parameters: 137319072
2021-06-23 01:21:17,042 Epoch: [22][95/1133]	Eit 25000  lr 5e-05  Le 8.0856 (9.2189)	Time 0.744 (0.000)	Data 0.002 (0.000)	
2021-06-23 01:23:36,712 Epoch: [22][295/1133]	Eit 25200  lr 5e-05  Le 9.2424 (9.2195)	Time 0.750 (0.000)	Data 0.002 (0.000)	
2021-06-23 01:25:56,682 Epoch: [22][495/1133]	Eit 25400  lr 5e-05  Le 9.3388 (9.1427)	Time 0.704 (0.000)	Data 0.002 (0.000)	
2021-06-23 01:28:17,543 Epoch: [22][695/1133]	Eit 25600  lr 5e-05  Le 12.5876 (9.1207)	Time 0.736 (0.000)	Data 0.003 (0.000)	
2021-06-23 01:30:36,451 Epoch: [22][895/1133]	Eit 25800  lr 5e-05  Le 9.2315 (9.1235)	Time 0.662 (0.000)	Data 0.002 (0.000)	
2021-06-23 01:32:54,670 Epoch: [22][1095/1133]	Eit 26000  lr 5e-05  Le 10.0517 (9.1311)	Time 0.634 (0.000)	Data 0.002 (0.000)	
2021-06-23 01:33:23,961 Test: [0/40]	Le 58.7917 (58.7917)	Time 3.571 (0.000)	
2021-06-23 01:33:34,448 calculate similarity time: 0.0495913028717041
2021-06-23 01:33:34,952 Image to text: 81.9, 96.1, 98.1, 1.0, 1.9
2021-06-23 01:33:35,300 Text to image: 63.2, 86.3, 91.6, 1.0, 7.0
2021-06-23 01:33:35,300 Current rsum is 517.1800000000001
2021-06-23 01:33:37,021 runs/f30k_butd_region_bert/log
2021-06-23 01:33:37,022 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-23 01:33:37,025 image encoder trainable parameters: 20490144
2021-06-23 01:33:37,037 txt encoder trainable parameters: 137319072
2021-06-23 01:35:34,424 Epoch: [23][163/1133]	Eit 26200  lr 5e-05  Le 8.0990 (9.0707)	Time 0.650 (0.000)	Data 0.002 (0.000)	
2021-06-23 01:37:53,035 Epoch: [23][363/1133]	Eit 26400  lr 5e-05  Le 9.5222 (8.9545)	Time 0.823 (0.000)	Data 0.002 (0.000)	
2021-06-23 01:40:13,474 Epoch: [23][563/1133]	Eit 26600  lr 5e-05  Le 11.2946 (8.9889)	Time 0.815 (0.000)	Data 0.002 (0.000)	
2021-06-23 01:42:32,131 Epoch: [23][763/1133]	Eit 26800  lr 5e-05  Le 9.1368 (9.0341)	Time 0.680 (0.000)	Data 0.003 (0.000)	
2021-06-23 01:44:53,824 Epoch: [23][963/1133]	Eit 27000  lr 5e-05  Le 10.2011 (9.0118)	Time 0.739 (0.000)	Data 0.002 (0.000)	
2021-06-23 01:46:54,404 Test: [0/40]	Le 58.9857 (58.9857)	Time 3.442 (0.000)	
2021-06-23 01:47:04,937 calculate similarity time: 0.04987478256225586
2021-06-23 01:47:05,356 Image to text: 81.7, 95.8, 98.4, 1.0, 1.9
2021-06-23 01:47:05,677 Text to image: 63.5, 86.8, 91.6, 1.0, 7.1
2021-06-23 01:47:05,678 Current rsum is 517.8399999999999
2021-06-23 01:47:07,442 runs/f30k_butd_region_bert/log
2021-06-23 01:47:07,442 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-23 01:47:07,446 image encoder trainable parameters: 20490144
2021-06-23 01:47:07,458 txt encoder trainable parameters: 137319072
2021-06-23 01:47:33,398 Epoch: [24][31/1133]	Eit 27200  lr 5e-05  Le 7.6942 (8.5711)	Time 0.612 (0.000)	Data 0.002 (0.000)	
2021-06-23 01:49:53,491 Epoch: [24][231/1133]	Eit 27400  lr 5e-05  Le 6.2787 (8.6643)	Time 0.666 (0.000)	Data 0.002 (0.000)	
2021-06-23 01:52:14,649 Epoch: [24][431/1133]	Eit 27600  lr 5e-05  Le 7.0846 (8.7778)	Time 0.684 (0.000)	Data 0.003 (0.000)	
2021-06-23 01:54:34,866 Epoch: [24][631/1133]	Eit 27800  lr 5e-05  Le 11.3489 (8.8516)	Time 0.614 (0.000)	Data 0.002 (0.000)	
2021-06-23 01:56:54,184 Epoch: [24][831/1133]	Eit 28000  lr 5e-05  Le 8.4848 (8.8003)	Time 0.688 (0.000)	Data 0.002 (0.000)	
2021-06-23 01:59:12,612 Epoch: [24][1031/1133]	Eit 28200  lr 5e-05  Le 8.1322 (8.8146)	Time 0.751 (0.000)	Data 0.002 (0.000)	
2021-06-23 02:00:26,222 Test: [0/40]	Le 58.9258 (58.9258)	Time 3.444 (0.000)	
2021-06-23 02:00:36,669 calculate similarity time: 0.05287289619445801
2021-06-23 02:00:37,172 Image to text: 81.3, 96.3, 98.3, 1.0, 2.0
2021-06-23 02:00:37,536 Text to image: 63.7, 86.5, 91.5, 1.0, 7.1
2021-06-23 02:00:37,536 Current rsum is 517.66
[root@gpu1 vse_infty-master-my-graph-gru-vse++]# CUDA_VISIBLE_DEVICES=0,1 python3 eval.py --dataset f30k --data_path ../data/f30k
INFO:root:Evaluating runs/f30k_butd_region_bert...
INFO:lib.evaluation:Namespace(backbone_lr_factor=0.01, backbone_path='', backbone_source='detector', backbone_warmup_epochs=5, batch_size=128, data_name='f30k', data_path='../data/f30k', embed_size=1024, embedding_warmup_epochs=2, grad_clip=2.0, img_dim=2048, input_scale_factor=1, learning_rate=0.0005, log_step=200, logger_name='runs/f30k_butd_region_bert/log', lr_update=15, margin=0.2, max_violation=True, model_name='runs/f30k_butd_region_bert', no_imgnorm=False, no_txtnorm=False, num_epochs=25, optim='adam', precomp_enc_type='basic', reset_start_epoch=False, resume='', val_step=500, vocab_path='./vocab/', vocab_size=30522, vse_mean_warmup_epochs=1, word_dim=300, workers=5)
INFO:transformers.tokenization_utils:loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-vse++/lib/encoders.py:48: UserWarning: nn.init.constant is now deprecated in favor of nn.init.constant_.
  nn.init.constant(self.W[1].weight, 0)
/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-vse++/lib/encoders.py:49: UserWarning: nn.init.constant is now deprecated in favor of nn.init.constant_.
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
INFO:lib.evaluation:Test: [0/40]	Le 61.4020 (61.4019)	Time 4.697 (0.000)	
INFO:lib.evaluation:Test: [10/40]	Le 59.1205 (60.0232)	Time 0.363 (0.000)	
INFO:lib.evaluation:Test: [20/40]	Le 59.4691 (59.9436)	Time 0.243 (0.000)	
INFO:lib.evaluation:Test: [30/40]	Le 61.4245 (60.1824)	Time 0.442 (0.000)	
INFO:lib.evaluation:Images: 1000, Captions: 5000
INFO:lib.evaluation:calculate similarity time: 0.05926966667175293
INFO:lib.evaluation:rsum: 520.5
INFO:lib.evaluation:Average i2t Recall: 92.4
INFO:lib.evaluation:Image to text: 83.6 95.9 97.6 1.0 2.1
INFO:lib.evaluation:Average t2i Recall: 81.1
INFO:lib.evaluation:Text to image: 63.8 87.3 92.4 1.0 5.6
INFO:root:Evaluating runs/release_weights/f30k_butd_grid_bert...
Traceback (most recent call last):
  File "eval.py", line 58, in <module>
    main()
  File "eval.py", line 54, in main
    evaluation.evalrank(model_path, data_path=opt.data_path, split='test', fold5=False, save_path=save_path)
  File "/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-vse++/lib/evaluation.py", line 196, in evalrank
    checkpoint = torch.load(model_path)
  File "/root/anaconda3/lib/python3.6/site-packages/torch/serialization.py", line 571, in load
    with _open_file_like(f, 'rb') as opened_file:
  File "/root/anaconda3/lib/python3.6/site-packages/torch/serialization.py", line 229, in _open_file_like
    return _open_file(name_or_buffer, mode)
  File "/root/anaconda3/lib/python3.6/site-packages/torch/serialization.py", line 210, in __init__
    super(_open_file, self).__init__(open(name, mode))
FileNotFoundError: [Errno 2] No such file or directory: 'runs/release_weights/f30k_butd_grid_bert/model_best.pth'
You have new mail in /var/spool/mail/root
[root@gpu1 vse_infty-master-my-graph-gru-vse++]# 

