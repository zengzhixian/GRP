[root@gpu1 vse_infty-master-my-graph-gru-vse]# sh train_region.sh 
2021-06-24 19:56:17,090 Namespace(backbone_lr_factor=0.01, backbone_path='', backbone_source='detector', backbone_warmup_epochs=5, batch_size=128, data_name='f30k', data_path='../data/f30k', embed_size=1024, embedding_warmup_epochs=2, grad_clip=2.0, img_dim=2048, input_scale_factor=1, learning_rate=0.0005, log_step=200, logger_name='runs/f30k_butd_region_bert/log', lr_update=15, margin=0.2, max_violation=False, model_name='runs/f30k_butd_region_bert', no_imgnorm=False, no_txtnorm=False, num_epochs=25, optim='adam', precomp_enc_type='basic', reset_start_epoch=False, resume='', val_step=500, vocab_path='./vocab/', vse_mean_warmup_epochs=1, word_dim=300, workers=10)
2021-06-24 19:56:17,090 Model name '/public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased' not found in model shortcut name list (bert-base-uncased, bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, bert-base-multilingual-cased, bert-base-chinese, bert-base-german-cased, bert-large-uncased-whole-word-masking, bert-large-cased-whole-word-masking, bert-large-uncased-whole-word-masking-finetuned-squad, bert-large-cased-whole-word-masking-finetuned-squad, bert-base-cased-finetuned-mrpc). Assuming '/public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased' is a path or url to a directory containing tokenizer files.
2021-06-24 19:56:17,090 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/added_tokens.json. We won't load it.
2021-06-24 19:56:17,090 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/special_tokens_map.json. We won't load it.
2021-06-24 19:56:17,090 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/tokenizer_config.json. We won't load it.
2021-06-24 19:56:17,091 loading file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/vocab.txt
2021-06-24 19:56:17,091 loading file None
2021-06-24 19:56:17,091 loading file None
2021-06-24 19:56:17,091 loading file None
/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-vse/lib/encoders.py:48: UserWarning: nn.init.constant is now deprecated in favor of nn.init.constant_.
  nn.init.constant(self.W[1].weight, 0)
/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-vse/lib/encoders.py:49: UserWarning: nn.init.constant is now deprecated in favor of nn.init.constant_.
  nn.init.constant(self.W[1].bias, 0)
2021-06-24 19:56:26,365 loading configuration file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/config.json
2021-06-24 19:56:26,366 Model config {
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

2021-06-24 19:56:26,366 loading weights file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/pytorch_model.bin
2021-06-24 19:56:34,807 Use adam as the optimizer, with init lr 0.0005
2021-06-24 19:56:34,808 Image encoder is data paralleled now.
2021-06-24 19:56:34,808 runs/f30k_butd_region_bert/log
2021-06-24 19:56:34,808 runs/f30k_butd_region_bert
2021-06-24 19:56:34,809 image encoder trainable parameters: 20490144
2021-06-24 19:56:34,814 txt encoder trainable parameters: 137319072
/root/anaconda3/lib/python3.6/site-packages/torch/nn/modules/rnn.py:735: UserWarning: RNN module weights are not part of single contiguous chunk of memory. This means they need to be compacted at every call, possibly greatly increasing memory usage. To compact weights again call flatten_parameters(). (Triggered internally at  /pytorch/aten/src/ATen/native/cudnn/RNN.cpp:1264.)
  self.dropout, self.training, self.bidirectional, self.batch_first)
^[[A
2021-06-24 20:00:46,838 Epoch: [0][199/1133]	Eit 200  lr 0.0005  Le 566.7277 (979.1514)	Time 0.907 (0.000)	Data 0.002 (0.000)	
2021-06-24 20:04:45,085 Epoch: [0][399/1133]	Eit 400  lr 0.0005  Le 610.7213 (761.6793)	Time 1.413 (0.000)	Data 0.002 (0.000)	
2021-06-24 20:08:41,092 Epoch: [0][599/1133]	Eit 600  lr 0.0005  Le 388.8767 (657.4810)	Time 1.405 (0.000)	Data 0.002 (0.000)	
Traceback (most recent call last):
  File "train.py", line 267, in <module>
    main()
  File "train.py", line 99, in main
    train(opt, train_loader, model, epoch, val_loader)
  File "train.py", line 146, in train
    model.train_emb(images, captions, lengths, image_lengths=img_lengths)
  File "/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-vse/lib/vse.py", line 186, in train_emb
    img_emb, cap_emb = self.forward_emb(images, captions, lengths, image_lengths=image_lengths)
  File "/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-vse/lib/vse.py", line 168, in forward_emb
    cap_emb = self.txt_enc(captions, lengths)
  File "/root/anaconda3/lib/python3.6/site-packages/torch/nn/modules/module.py", line 722, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/root/anaconda3/lib/python3.6/site-packages/torch/nn/parallel/data_parallel.py", line 155, in forward
    outputs = self.parallel_apply(replicas, inputs, kwargs)
  File "/root/anaconda3/lib/python3.6/site-packages/torch/nn/parallel/data_parallel.py", line 165, in parallel_apply
    return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])
  File "/root/anaconda3/lib/python3.6/site-packages/torch/nn/parallel/parallel_apply.py", line 85, in parallel_apply
    output.reraise()
  File "/root/anaconda3/lib/python3.6/site-packages/torch/_utils.py", line 395, in reraise
    raise self.exc_type(msg)
RuntimeError: Caught RuntimeError in replica 0 on device 0.
Original Traceback (most recent call last):
  File "/root/anaconda3/lib/python3.6/site-packages/torch/nn/parallel/parallel_apply.py", line 60, in _worker
    output = module(*input, **kwargs)
  File "/root/anaconda3/lib/python3.6/site-packages/torch/nn/modules/module.py", line 722, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-vse/lib/encoders.py", line 291, in forward
    cap_emb, hidden_state = self.rnn(bert_emb)
  File "/root/anaconda3/lib/python3.6/site-packages/torch/nn/modules/module.py", line 722, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/root/anaconda3/lib/python3.6/site-packages/torch/nn/modules/rnn.py", line 735, in forward
    self.dropout, self.training, self.bidirectional, self.batch_first)
RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED

[root@gpu1 vse_infty-master-my-graph-gru-vse]# sh train_region.sh 
2021-06-24 20:09:50,126 Namespace(backbone_lr_factor=0.01, backbone_path='', backbone_source='detector', backbone_warmup_epochs=5, batch_size=128, data_name='f30k', data_path='../data/f30k', embed_size=1024, embedding_warmup_epochs=2, grad_clip=2.0, img_dim=2048, input_scale_factor=1, learning_rate=0.0005, log_step=200, logger_name='runs/f30k_butd_region_bert/log', lr_update=15, margin=0.2, max_violation=False, model_name='runs/f30k_butd_region_bert', no_imgnorm=False, no_txtnorm=False, num_epochs=25, optim='adam', precomp_enc_type='basic', reset_start_epoch=False, resume='', val_step=500, vocab_path='./vocab/', vse_mean_warmup_epochs=1, word_dim=300, workers=10)
2021-06-24 20:09:50,126 Model name '/public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased' not found in model shortcut name list (bert-base-uncased, bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, bert-base-multilingual-cased, bert-base-chinese, bert-base-german-cased, bert-large-uncased-whole-word-masking, bert-large-cased-whole-word-masking, bert-large-uncased-whole-word-masking-finetuned-squad, bert-large-cased-whole-word-masking-finetuned-squad, bert-base-cased-finetuned-mrpc). Assuming '/public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased' is a path or url to a directory containing tokenizer files.
2021-06-24 20:09:50,126 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/added_tokens.json. We won't load it.
2021-06-24 20:09:50,127 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/special_tokens_map.json. We won't load it.
2021-06-24 20:09:50,127 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/tokenizer_config.json. We won't load it.
2021-06-24 20:09:50,127 loading file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/vocab.txt
2021-06-24 20:09:50,127 loading file None
2021-06-24 20:09:50,127 loading file None
2021-06-24 20:09:50,127 loading file None
/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-vse/lib/encoders.py:48: UserWarning: nn.init.constant is now deprecated in favor of nn.init.constant_.
  nn.init.constant(self.W[1].weight, 0)
/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-vse/lib/encoders.py:49: UserWarning: nn.init.constant is now deprecated in favor of nn.init.constant_.
  nn.init.constant(self.W[1].bias, 0)
2021-06-24 20:09:59,012 loading configuration file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/config.json
2021-06-24 20:09:59,012 Model config {
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

2021-06-24 20:09:59,013 loading weights file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/pytorch_model.bin
2021-06-24 20:10:07,213 Use adam as the optimizer, with init lr 0.0005
2021-06-24 20:10:07,214 Image encoder is data paralleled now.
2021-06-24 20:10:07,214 runs/f30k_butd_region_bert/log
2021-06-24 20:10:07,214 runs/f30k_butd_region_bert
2021-06-24 20:10:07,216 image encoder trainable parameters: 20490144
2021-06-24 20:10:07,221 txt encoder trainable parameters: 137319072
/root/anaconda3/lib/python3.6/site-packages/torch/nn/modules/rnn.py:735: UserWarning: RNN module weights are not part of single contiguous chunk of memory. This means they need to be compacted at every call, possibly greatly increasing memory usage. To compact weights again call flatten_parameters(). (Triggered internally at  /pytorch/aten/src/ATen/native/cudnn/RNN.cpp:1264.)
  self.dropout, self.training, self.bidirectional, self.batch_first)
2021-06-24 20:14:16,410 Epoch: [0][199/1133]	Eit 200  lr 0.0005  Le 635.7645 (991.5995)	Time 1.171 (0.000)	Data 0.002 (0.000)	
2021-06-24 20:18:10,740 Epoch: [0][399/1133]	Eit 400  lr 0.0005  Le 382.8786 (767.9838)	Time 1.328 (0.000)	Data 0.002 (0.000)	
2021-06-24 20:22:09,800 Epoch: [0][599/1133]	Eit 600  lr 0.0005  Le 614.5844 (663.2085)	Time 1.296 (0.000)	Data 0.004 (0.000)	
2021-06-24 20:25:58,491 Epoch: [0][799/1133]	Eit 800  lr 0.0005  Le 329.4166 (599.8517)	Time 1.053 (0.000)	Data 0.002 (0.000)	
2021-06-24 20:29:54,251 Epoch: [0][999/1133]	Eit 1000  lr 0.0005  Le 416.0729 (556.2702)	Time 0.920 (0.000)	Data 0.002 (0.000)	
2021-06-24 20:32:30,477 Test: [0/40]	Le 336.7339 (336.7336)	Time 3.982 (0.000)	
2021-06-24 20:32:47,929 calculate similarity time: 0.10480356216430664
2021-06-24 20:32:48,422 Image to text: 45.4, 77.3, 87.5, 2.0, 6.3
2021-06-24 20:32:48,761 Text to image: 36.4, 64.9, 77.3, 3.0, 11.8
2021-06-24 20:32:48,761 Current rsum is 388.8
2021-06-24 20:32:51,864 runs/f30k_butd_region_bert/log
2021-06-24 20:32:51,864 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 20:32:51,865 image encoder trainable parameters: 20490144
2021-06-24 20:32:51,871 txt encoder trainable parameters: 137319072
2021-06-24 20:34:15,481 Epoch: [1][67/1133]	Eit 1200  lr 0.0005  Le 321.2946 (284.0490)	Time 1.125 (0.000)	Data 0.002 (0.000)	
2021-06-24 20:38:07,534 Epoch: [1][267/1133]	Eit 1400  lr 0.0005  Le 238.8996 (295.2180)	Time 1.389 (0.000)	Data 0.002 (0.000)	
2021-06-24 20:42:00,919 Epoch: [1][467/1133]	Eit 1600  lr 0.0005  Le 288.8522 (286.4422)	Time 1.421 (0.000)	Data 0.002 (0.000)	
2021-06-24 20:45:53,906 Epoch: [1][667/1133]	Eit 1800  lr 0.0005  Le 288.4382 (281.0352)	Time 1.157 (0.000)	Data 0.002 (0.000)	
2021-06-24 20:49:50,430 Epoch: [1][867/1133]	Eit 2000  lr 0.0005  Le 365.8186 (276.6234)	Time 1.445 (0.000)	Data 0.002 (0.000)	
2021-06-24 20:53:40,707 Epoch: [1][1067/1133]	Eit 2200  lr 0.0005  Le 261.9089 (272.0618)	Time 1.277 (0.000)	Data 0.002 (0.000)	
2021-06-24 20:54:59,110 Test: [0/40]	Le 321.5262 (321.5259)	Time 3.600 (0.000)	
2021-06-24 20:55:16,018 calculate similarity time: 0.07567739486694336
2021-06-24 20:55:16,479 Image to text: 55.4, 84.8, 91.3, 1.0, 4.9
2021-06-24 20:55:16,803 Text to image: 42.4, 72.9, 82.9, 2.0, 8.8
2021-06-24 20:55:16,803 Current rsum is 429.68
2021-06-24 20:55:20,643 runs/f30k_butd_region_bert/log
2021-06-24 20:55:20,644 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 20:55:20,647 image encoder trainable parameters: 20490144
2021-06-24 20:55:20,657 txt encoder trainable parameters: 137319072
2021-06-24 20:58:00,831 Epoch: [2][135/1133]	Eit 2400  lr 0.0005  Le 165.2511 (208.3713)	Time 0.946 (0.000)	Data 0.002 (0.000)	
2021-06-24 21:01:56,248 Epoch: [2][335/1133]	Eit 2600  lr 0.0005  Le 167.9312 (209.3495)	Time 1.266 (0.000)	Data 0.002 (0.000)	
2021-06-24 21:05:49,969 Epoch: [2][535/1133]	Eit 2800  lr 0.0005  Le 186.3402 (209.6149)	Time 1.231 (0.000)	Data 0.002 (0.000)	
2021-06-24 21:09:47,415 Epoch: [2][735/1133]	Eit 3000  lr 0.0005  Le 205.8503 (208.1472)	Time 0.995 (0.000)	Data 0.002 (0.000)	
2021-06-24 21:13:44,007 Epoch: [2][935/1133]	Eit 3200  lr 0.0005  Le 186.8056 (209.2654)	Time 0.899 (0.000)	Data 0.002 (0.000)	
2021-06-24 21:17:39,054 Test: [0/40]	Le 343.4601 (343.4599)	Time 3.601 (0.000)	
2021-06-24 21:17:56,146 calculate similarity time: 0.07420563697814941
2021-06-24 21:17:56,632 Image to text: 57.0, 83.8, 91.1, 1.0, 4.7
2021-06-24 21:17:56,944 Text to image: 45.4, 75.5, 84.9, 2.0, 8.5
2021-06-24 21:17:56,944 Current rsum is 437.64000000000004
2021-06-24 21:18:00,793 runs/f30k_butd_region_bert/log
2021-06-24 21:18:00,793 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 21:18:00,796 image encoder trainable parameters: 20490144
2021-06-24 21:18:00,808 txt encoder trainable parameters: 137319072
2021-06-24 21:18:08,677 Epoch: [3][3/1133]	Eit 3400  lr 0.0005  Le 260.8788 (233.3799)	Time 0.948 (0.000)	Data 0.002 (0.000)	
2021-06-24 21:21:56,810 Epoch: [3][203/1133]	Eit 3600  lr 0.0005  Le 196.5075 (172.0920)	Time 1.048 (0.000)	Data 0.002 (0.000)	
2021-06-24 21:25:50,961 Epoch: [3][403/1133]	Eit 3800  lr 0.0005  Le 220.0902 (171.9348)	Time 0.899 (0.000)	Data 0.002 (0.000)	
2021-06-24 21:29:46,229 Epoch: [3][603/1133]	Eit 4000  lr 0.0005  Le 219.3359 (172.7556)	Time 1.470 (0.000)	Data 0.002 (0.000)	
2021-06-24 21:33:40,293 Epoch: [3][803/1133]	Eit 4200  lr 0.0005  Le 154.3810 (172.8536)	Time 0.976 (0.000)	Data 0.002 (0.000)	
2021-06-24 21:37:33,122 Epoch: [3][1003/1133]	Eit 4400  lr 0.0005  Le 180.9217 (173.3874)	Time 1.269 (0.000)	Data 0.002 (0.000)	
2021-06-24 21:40:06,796 Test: [0/40]	Le 382.5530 (382.5527)	Time 3.321 (0.000)	
2021-06-24 21:40:23,912 calculate similarity time: 0.0822746753692627
2021-06-24 21:40:24,344 Image to text: 60.6, 87.2, 94.0, 1.0, 4.1
2021-06-24 21:40:24,658 Text to image: 48.0, 77.2, 86.1, 2.0, 7.8
2021-06-24 21:40:24,658 Current rsum is 453.1
2021-06-24 21:40:28,456 runs/f30k_butd_region_bert/log
2021-06-24 21:40:28,456 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 21:40:28,458 image encoder trainable parameters: 20490144
2021-06-24 21:40:28,465 txt encoder trainable parameters: 137319072
2021-06-24 21:41:56,555 Epoch: [4][71/1133]	Eit 4600  lr 0.0005  Le 162.7488 (154.0725)	Time 1.168 (0.000)	Data 0.002 (0.000)	
2021-06-24 21:45:50,746 Epoch: [4][271/1133]	Eit 4800  lr 0.0005  Le 121.3224 (152.0827)	Time 1.046 (0.000)	Data 0.002 (0.000)	
2021-06-24 21:49:38,637 Epoch: [4][471/1133]	Eit 5000  lr 0.0005  Le 213.4968 (154.0672)	Time 0.643 (0.000)	Data 0.002 (0.000)	
2021-06-24 21:53:32,197 Epoch: [4][671/1133]	Eit 5200  lr 0.0005  Le 112.8979 (152.8273)	Time 1.279 (0.000)	Data 0.002 (0.000)	
2021-06-24 21:57:24,308 Epoch: [4][871/1133]	Eit 5400  lr 0.0005  Le 128.1303 (151.8718)	Time 1.190 (0.000)	Data 0.002 (0.000)	
2021-06-24 22:01:17,906 Epoch: [4][1071/1133]	Eit 5600  lr 0.0005  Le 166.5514 (151.3797)	Time 1.209 (0.000)	Data 0.002 (0.000)	
2021-06-24 22:02:31,865 Test: [0/40]	Le 346.7586 (346.7583)	Time 3.591 (0.000)	
2021-06-24 22:02:49,289 calculate similarity time: 0.07788515090942383
2021-06-24 22:02:49,844 Image to text: 61.7, 87.2, 93.3, 1.0, 3.6
2021-06-24 22:02:50,169 Text to image: 49.1, 77.5, 86.3, 2.0, 7.7
2021-06-24 22:02:50,169 Current rsum is 455.15999999999997
2021-06-24 22:02:54,109 runs/f30k_butd_region_bert/log
2021-06-24 22:02:54,110 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 22:02:54,113 image encoder trainable parameters: 20490144
2021-06-24 22:02:54,124 txt encoder trainable parameters: 137319072
2021-06-24 22:05:37,938 Epoch: [5][139/1133]	Eit 5800  lr 0.0005  Le 148.8882 (132.5558)	Time 1.472 (0.000)	Data 0.003 (0.000)	
2021-06-24 22:09:33,218 Epoch: [5][339/1133]	Eit 6000  lr 0.0005  Le 156.8277 (133.4509)	Time 1.249 (0.000)	Data 0.002 (0.000)	
2021-06-24 22:13:28,141 Epoch: [5][539/1133]	Eit 6200  lr 0.0005  Le 102.1067 (134.0253)	Time 1.020 (0.000)	Data 0.002 (0.000)	
2021-06-24 22:17:23,809 Epoch: [5][739/1133]	Eit 6400  lr 0.0005  Le 164.1939 (135.4961)	Time 1.352 (0.000)	Data 0.002 (0.000)	
2021-06-24 22:21:11,547 Epoch: [5][939/1133]	Eit 6600  lr 0.0005  Le 151.1608 (135.8762)	Time 1.315 (0.000)	Data 0.003 (0.000)	
2021-06-24 22:25:01,097 Test: [0/40]	Le 343.1658 (343.1656)	Time 3.422 (0.000)	
2021-06-24 22:25:18,228 calculate similarity time: 0.08558988571166992
2021-06-24 22:25:18,750 Image to text: 63.1, 87.8, 94.3, 1.0, 4.1
2021-06-24 22:25:19,212 Text to image: 50.2, 78.4, 87.1, 1.0, 7.6
2021-06-24 22:25:19,212 Current rsum is 460.88
2021-06-24 22:25:23,149 runs/f30k_butd_region_bert/log
2021-06-24 22:25:23,149 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 22:25:23,152 image encoder trainable parameters: 20490144
2021-06-24 22:25:23,163 txt encoder trainable parameters: 137319072
2021-06-24 22:25:35,898 Epoch: [6][7/1133]	Eit 6800  lr 0.0005  Le 178.5976 (119.1257)	Time 0.901 (0.000)	Data 0.002 (0.000)	
2021-06-24 22:29:32,103 Epoch: [6][207/1133]	Eit 7000  lr 0.0005  Le 155.7453 (120.7044)	Time 1.190 (0.000)	Data 0.002 (0.000)	
2021-06-24 22:33:26,456 Epoch: [6][407/1133]	Eit 7200  lr 0.0005  Le 147.9300 (122.2431)	Time 1.185 (0.000)	Data 0.002 (0.000)	
2021-06-24 22:37:15,019 Epoch: [6][607/1133]	Eit 7400  lr 0.0005  Le 93.9304 (121.1235)	Time 1.168 (0.000)	Data 0.002 (0.000)	
2021-06-24 22:41:11,946 Epoch: [6][807/1133]	Eit 7600  lr 0.0005  Le 114.5027 (122.2525)	Time 1.385 (0.000)	Data 0.002 (0.000)	
2021-06-24 22:45:06,044 Epoch: [6][1007/1133]	Eit 7800  lr 0.0005  Le 149.0164 (123.2472)	Time 0.876 (0.000)	Data 0.002 (0.000)	
2021-06-24 22:47:27,294 Test: [0/40]	Le 340.2956 (340.2953)	Time 3.411 (0.000)	
2021-06-24 22:47:44,208 calculate similarity time: 0.05913853645324707
2021-06-24 22:47:44,649 Image to text: 63.7, 90.2, 94.8, 1.0, 3.6
2021-06-24 22:47:44,973 Text to image: 51.2, 79.8, 87.7, 1.0, 7.4
2021-06-24 22:47:44,973 Current rsum is 467.34000000000003
2021-06-24 22:47:48,669 runs/f30k_butd_region_bert/log
2021-06-24 22:47:48,670 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 22:47:48,673 image encoder trainable parameters: 20490144
2021-06-24 22:47:48,684 txt encoder trainable parameters: 137319072
2021-06-24 22:49:21,860 Epoch: [7][75/1133]	Eit 8000  lr 0.0005  Le 80.7189 (107.0668)	Time 1.285 (0.000)	Data 0.003 (0.000)	
2021-06-24 22:53:15,903 Epoch: [7][275/1133]	Eit 8200  lr 0.0005  Le 207.6584 (110.3707)	Time 0.973 (0.000)	Data 0.002 (0.000)	
2021-06-24 22:57:11,681 Epoch: [7][475/1133]	Eit 8400  lr 0.0005  Le 72.7365 (112.3252)	Time 1.098 (0.000)	Data 0.002 (0.000)	
2021-06-24 23:01:06,864 Epoch: [7][675/1133]	Eit 8600  lr 0.0005  Le 81.0890 (113.4056)	Time 1.032 (0.000)	Data 0.003 (0.000)	
2021-06-24 23:04:59,218 Epoch: [7][875/1133]	Eit 8800  lr 0.0005  Le 140.3404 (113.5485)	Time 0.864 (0.000)	Data 0.002 (0.000)	
2021-06-24 23:08:53,701 Epoch: [7][1075/1133]	Eit 9000  lr 0.0005  Le 120.2498 (113.7303)	Time 1.418 (0.000)	Data 0.003 (0.000)	
2021-06-24 23:10:02,192 Test: [0/40]	Le 329.0123 (329.0121)	Time 3.680 (0.000)	
2021-06-24 23:10:19,666 calculate similarity time: 0.08499431610107422
2021-06-24 23:10:20,076 Image to text: 64.6, 89.3, 94.1, 1.0, 3.4
2021-06-24 23:10:20,384 Text to image: 52.9, 79.7, 87.9, 1.0, 7.2
2021-06-24 23:10:20,384 Current rsum is 468.53999999999996
2021-06-24 23:10:24,109 runs/f30k_butd_region_bert/log
2021-06-24 23:10:24,109 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 23:10:24,112 image encoder trainable parameters: 20490144
2021-06-24 23:10:24,124 txt encoder trainable parameters: 137319072
2021-06-24 23:13:13,408 Epoch: [8][143/1133]	Eit 9200  lr 0.0005  Le 164.6563 (107.6220)	Time 0.923 (0.000)	Data 0.002 (0.000)	
2021-06-24 23:16:59,518 Epoch: [8][343/1133]	Eit 9400  lr 0.0005  Le 75.0879 (104.9507)	Time 1.336 (0.000)	Data 0.002 (0.000)	
2021-06-24 23:20:53,346 Epoch: [8][543/1133]	Eit 9600  lr 0.0005  Le 106.1325 (105.3912)	Time 0.991 (0.000)	Data 0.003 (0.000)	
2021-06-24 23:24:50,992 Epoch: [8][743/1133]	Eit 9800  lr 0.0005  Le 134.2666 (106.7266)	Time 1.227 (0.000)	Data 0.003 (0.000)	
2021-06-24 23:28:44,281 Epoch: [8][943/1133]	Eit 10000  lr 0.0005  Le 92.0020 (107.6500)	Time 1.260 (0.000)	Data 0.003 (0.000)	
2021-06-24 23:32:28,609 Test: [0/40]	Le 303.3698 (303.3695)	Time 3.674 (0.000)	
2021-06-24 23:32:45,317 calculate similarity time: 0.08572196960449219
2021-06-24 23:32:45,856 Image to text: 65.7, 89.7, 94.5, 1.0, 3.3
2021-06-24 23:32:46,308 Text to image: 53.1, 79.8, 87.8, 1.0, 6.8
2021-06-24 23:32:46,308 Current rsum is 470.70000000000005
2021-06-24 23:32:50,072 runs/f30k_butd_region_bert/log
2021-06-24 23:32:50,072 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 23:32:50,076 image encoder trainable parameters: 20490144
2021-06-24 23:32:50,085 txt encoder trainable parameters: 137319072
2021-06-24 23:33:07,301 Epoch: [9][11/1133]	Eit 10200  lr 0.0005  Le 80.8649 (84.8802)	Time 1.158 (0.000)	Data 0.002 (0.000)	
2021-06-24 23:37:01,036 Epoch: [9][211/1133]	Eit 10400  lr 0.0005  Le 100.0202 (95.9005)	Time 0.939 (0.000)	Data 0.002 (0.000)	
2021-06-24 23:40:52,999 Epoch: [9][411/1133]	Eit 10600  lr 0.0005  Le 77.4862 (96.1943)	Time 1.458 (0.000)	Data 0.002 (0.000)	
2021-06-24 23:44:41,127 Epoch: [9][611/1133]	Eit 10800  lr 0.0005  Le 97.4490 (95.6925)	Time 0.911 (0.000)	Data 0.002 (0.000)	
2021-06-24 23:48:35,310 Epoch: [9][811/1133]	Eit 11000  lr 0.0005  Le 92.3278 (97.7187)	Time 0.967 (0.000)	Data 0.002 (0.000)	
2021-06-24 23:52:30,474 Epoch: [9][1011/1133]	Eit 11200  lr 0.0005  Le 118.3886 (98.4575)	Time 1.311 (0.000)	Data 0.002 (0.000)	
2021-06-24 23:54:53,234 Test: [0/40]	Le 298.9448 (298.9445)	Time 3.692 (0.000)	
2021-06-24 23:55:10,115 calculate similarity time: 0.07237911224365234
2021-06-24 23:55:10,667 Image to text: 67.9, 91.2, 95.2, 1.0, 3.2
2021-06-24 23:55:11,012 Text to image: 53.5, 80.9, 88.5, 1.0, 7.1
2021-06-24 23:55:11,012 Current rsum is 477.15999999999997
2021-06-24 23:55:14,651 runs/f30k_butd_region_bert/log
2021-06-24 23:55:14,652 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 23:55:14,655 image encoder trainable parameters: 20490144
2021-06-24 23:55:14,666 txt encoder trainable parameters: 137319072
2021-06-24 23:56:52,190 Epoch: [10][79/1133]	Eit 11400  lr 0.0005  Le 72.2127 (88.7612)	Time 1.398 (0.000)	Data 0.002 (0.000)	
2021-06-25 00:00:48,980 Epoch: [10][279/1133]	Eit 11600  lr 0.0005  Le 105.8688 (91.6238)	Time 1.224 (0.000)	Data 0.002 (0.000)	
2021-06-25 00:04:43,911 Epoch: [10][479/1133]	Eit 11800  lr 0.0005  Le 91.6255 (93.6526)	Time 1.010 (0.000)	Data 0.002 (0.000)	
2021-06-25 00:08:39,771 Epoch: [10][679/1133]	Eit 12000  lr 0.0005  Le 55.6579 (93.6505)	Time 1.214 (0.000)	Data 0.002 (0.000)	
2021-06-25 00:12:35,899 Epoch: [10][879/1133]	Eit 12200  lr 0.0005  Le 110.5211 (93.2712)	Time 1.199 (0.000)	Data 0.002 (0.000)	
2021-06-25 00:16:23,351 Epoch: [10][1079/1133]	Eit 12400  lr 0.0005  Le 80.8463 (93.9022)	Time 0.902 (0.000)	Data 0.002 (0.000)	
2021-06-25 00:17:26,581 Test: [0/40]	Le 318.8179 (318.8176)	Time 3.457 (0.000)	
2021-06-25 00:17:43,809 calculate similarity time: 0.05657243728637695
2021-06-25 00:17:44,332 Image to text: 71.2, 91.2, 95.4, 1.0, 3.0
2021-06-25 00:17:44,670 Text to image: 53.8, 81.6, 88.7, 1.0, 6.8
2021-06-25 00:17:44,670 Current rsum is 481.98
2021-06-25 00:17:48,308 runs/f30k_butd_region_bert/log
2021-06-25 00:17:48,308 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-25 00:17:48,312 image encoder trainable parameters: 20490144
2021-06-25 00:17:48,323 txt encoder trainable parameters: 137319072
2021-06-25 00:20:46,253 Epoch: [11][147/1133]	Eit 12600  lr 0.0005  Le 56.6821 (83.6318)	Time 0.956 (0.000)	Data 0.002 (0.000)	
2021-06-25 00:24:41,894 Epoch: [11][347/1133]	Eit 12800  lr 0.0005  Le 86.1831 (85.3391)	Time 1.262 (0.000)	Data 0.002 (0.000)	
2021-06-25 00:28:35,702 Epoch: [11][547/1133]	Eit 13000  lr 0.0005  Le 124.6728 (86.2676)	Time 1.264 (0.000)	Data 0.002 (0.000)	
2021-06-25 00:32:30,678 Epoch: [11][747/1133]	Eit 13200  lr 0.0005  Le 68.1046 (86.4663)	Time 1.352 (0.000)	Data 0.002 (0.000)	
2021-06-25 00:36:23,077 Epoch: [11][947/1133]	Eit 13400  lr 0.0005  Le 110.2845 (87.0631)	Time 0.959 (0.000)	Data 0.002 (0.000)	
2021-06-25 00:40:01,378 Test: [0/40]	Le 315.9052 (315.9050)	Time 3.638 (0.000)	
2021-06-25 00:40:18,508 calculate similarity time: 0.0715034008026123
2021-06-25 00:40:19,048 Image to text: 68.2, 91.4, 95.4, 1.0, 2.9
2021-06-25 00:40:19,468 Text to image: 53.8, 81.5, 88.9, 1.0, 6.8
2021-06-25 00:40:19,468 Current rsum is 479.24
2021-06-25 00:40:21,007 runs/f30k_butd_region_bert/log
2021-06-25 00:40:21,007 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-25 00:40:21,008 image encoder trainable parameters: 20490144
2021-06-25 00:40:21,014 txt encoder trainable parameters: 137319072
2021-06-25 00:40:42,900 Epoch: [12][15/1133]	Eit 13600  lr 0.0005  Le 111.8943 (85.6860)	Time 0.847 (0.000)	Data 0.002 (0.000)	
2021-06-25 00:44:30,876 Epoch: [12][215/1133]	Eit 13800  lr 0.0005  Le 57.6685 (84.4482)	Time 1.290 (0.000)	Data 0.003 (0.000)	
2021-06-25 00:48:25,251 Epoch: [12][415/1133]	Eit 14000  lr 0.0005  Le 118.3898 (83.7379)	Time 1.150 (0.000)	Data 0.002 (0.000)	
2021-06-25 00:52:19,029 Epoch: [12][615/1133]	Eit 14200  lr 0.0005  Le 82.6468 (85.2816)	Time 1.499 (0.000)	Data 0.002 (0.000)	
2021-06-25 00:56:12,970 Epoch: [12][815/1133]	Eit 14400  lr 0.0005  Le 88.2475 (85.0838)	Time 1.163 (0.000)	Data 0.002 (0.000)	
2021-06-25 01:00:05,577 Epoch: [12][1015/1133]	Eit 14600  lr 0.0005  Le 108.6489 (86.0170)	Time 1.381 (0.000)	Data 0.002 (0.000)	
2021-06-25 01:02:25,944 Test: [0/40]	Le 346.4683 (346.4681)	Time 3.599 (0.000)	
2021-06-25 01:02:43,056 calculate similarity time: 0.08213067054748535
2021-06-25 01:02:43,492 Image to text: 70.5, 92.1, 95.6, 1.0, 3.0
2021-06-25 01:02:43,804 Text to image: 53.9, 81.9, 88.9, 1.0, 7.0
2021-06-25 01:02:43,804 Current rsum is 482.9
2021-06-25 01:02:47,526 runs/f30k_butd_region_bert/log
2021-06-25 01:02:47,526 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-25 01:02:47,529 image encoder trainable parameters: 20490144
2021-06-25 01:02:47,541 txt encoder trainable parameters: 137319072
2021-06-25 01:04:26,847 Epoch: [13][83/1133]	Eit 14800  lr 0.0005  Le 50.4796 (79.3694)	Time 1.251 (0.000)	Data 0.002 (0.000)	
2021-06-25 01:08:22,054 Epoch: [13][283/1133]	Eit 15000  lr 0.0005  Le 58.9444 (78.3865)	Time 1.295 (0.000)	Data 0.002 (0.000)	
2021-06-25 01:12:08,150 Epoch: [13][483/1133]	Eit 15200  lr 0.0005  Le 69.4686 (78.3551)	Time 1.390 (0.000)	Data 0.002 (0.000)	
2021-06-25 01:16:03,494 Epoch: [13][683/1133]	Eit 15400  lr 0.0005  Le 67.5072 (79.0843)	Time 1.506 (0.000)	Data 0.002 (0.000)	
2021-06-25 01:19:57,914 Epoch: [13][883/1133]	Eit 15600  lr 0.0005  Le 80.9013 (79.4317)	Time 1.075 (0.000)	Data 0.002 (0.000)	
2021-06-25 01:23:51,885 Epoch: [13][1083/1133]	Eit 15800  lr 0.0005  Le 60.7885 (79.5823)	Time 1.105 (0.000)	Data 0.002 (0.000)	
2021-06-25 01:24:51,807 Test: [0/40]	Le 305.7986 (305.7983)	Time 3.368 (0.000)	
2021-06-25 01:25:08,817 calculate similarity time: 0.06272435188293457
2021-06-25 01:25:09,346 Image to text: 70.1, 92.5, 96.7, 1.0, 2.8
2021-06-25 01:25:09,658 Text to image: 54.2, 82.0, 89.5, 1.0, 7.0
2021-06-25 01:25:09,659 Current rsum is 484.96000000000004
2021-06-25 01:25:13,230 runs/f30k_butd_region_bert/log
2021-06-25 01:25:13,231 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-25 01:25:13,234 image encoder trainable parameters: 20490144
2021-06-25 01:25:13,246 txt encoder trainable parameters: 137319072
2021-06-25 01:28:14,779 Epoch: [14][151/1133]	Eit 16000  lr 0.0005  Le 46.7384 (78.3511)	Time 1.163 (0.000)	Data 0.002 (0.000)	
2021-06-25 01:32:06,392 Epoch: [14][351/1133]	Eit 16200  lr 0.0005  Le 80.7836 (77.6085)	Time 1.245 (0.000)	Data 0.003 (0.000)	
2021-06-25 01:36:00,254 Epoch: [14][551/1133]	Eit 16400  lr 0.0005  Le 54.2945 (78.3272)	Time 0.980 (0.000)	Data 0.002 (0.000)	
2021-06-25 01:39:48,773 Epoch: [14][751/1133]	Eit 16600  lr 0.0005  Le 72.9137 (78.6193)	Time 1.265 (0.000)	Data 0.002 (0.000)	
2021-06-25 01:43:41,665 Epoch: [14][951/1133]	Eit 16800  lr 0.0005  Le 75.5717 (78.8546)	Time 0.930 (0.000)	Data 0.002 (0.000)	
2021-06-25 01:47:16,707 Test: [0/40]	Le 299.4531 (299.4528)	Time 3.475 (0.000)	
2021-06-25 01:47:34,081 calculate similarity time: 0.07265877723693848
2021-06-25 01:47:34,655 Image to text: 68.7, 90.7, 95.2, 1.0, 2.8
2021-06-25 01:47:34,997 Text to image: 54.7, 82.3, 89.6, 1.0, 6.5
2021-06-25 01:47:34,998 Current rsum is 481.20000000000005
2021-06-25 01:47:36,460 runs/f30k_butd_region_bert/log
2021-06-25 01:47:36,460 runs/f30k_butd_region_bert
2021-06-25 01:47:36,460 Current epoch num is 15, decrease all lr by 10
2021-06-25 01:47:36,460 new lr 5e-05
2021-06-25 01:47:36,460 new lr 5e-06
2021-06-25 01:47:36,460 new lr 5e-05
Use VSE++ objective.
2021-06-25 01:47:36,462 image encoder trainable parameters: 20490144
2021-06-25 01:47:36,468 txt encoder trainable parameters: 137319072
2021-06-25 01:48:02,882 Epoch: [15][19/1133]	Eit 17000  lr 5e-05  Le 72.6938 (72.0219)	Time 1.006 (0.000)	Data 0.002 (0.000)	
2021-06-25 01:51:56,449 Epoch: [15][219/1133]	Eit 17200  lr 5e-05  Le 61.8910 (66.1129)	Time 1.415 (0.000)	Data 0.002 (0.000)	
2021-06-25 01:55:52,197 Epoch: [15][419/1133]	Eit 17400  lr 5e-05  Le 42.3501 (65.2655)	Time 0.940 (0.000)	Data 0.002 (0.000)	
2021-06-25 01:59:46,412 Epoch: [15][619/1133]	Eit 17600  lr 5e-05  Le 52.0535 (64.3077)	Time 0.950 (0.000)	Data 0.002 (0.000)	
2021-06-25 02:03:40,938 Epoch: [15][819/1133]	Eit 17800  lr 5e-05  Le 53.1278 (62.9864)	Time 1.290 (0.000)	Data 0.002 (0.000)	
2021-06-25 02:07:35,113 Epoch: [15][1019/1133]	Eit 18000  lr 5e-05  Le 61.6388 (62.6544)	Time 0.983 (0.000)	Data 0.002 (0.000)	
2021-06-25 02:09:40,878 Test: [0/40]	Le 295.6000 (295.5998)	Time 3.367 (0.000)	
2021-06-25 02:09:58,317 calculate similarity time: 0.0655202865600586
2021-06-25 02:09:58,859 Image to text: 72.1, 92.4, 95.8, 1.0, 2.6
2021-06-25 02:09:59,291 Text to image: 56.1, 82.8, 90.1, 1.0, 6.4
2021-06-25 02:09:59,292 Current rsum is 489.32000000000005
2021-06-25 02:10:02,806 runs/f30k_butd_region_bert/log
2021-06-25 02:10:02,806 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-25 02:10:02,813 image encoder trainable parameters: 20490144
2021-06-25 02:10:02,826 txt encoder trainable parameters: 137319072
2021-06-25 02:11:46,978 Epoch: [16][87/1133]	Eit 18200  lr 5e-05  Le 44.9040 (58.5933)	Time 0.861 (0.000)	Data 0.002 (0.000)	
2021-06-25 02:15:41,450 Epoch: [16][287/1133]	Eit 18400  lr 5e-05  Le 80.7300 (57.9844)	Time 1.140 (0.000)	Data 0.002 (0.000)	
2021-06-25 02:19:35,501 Epoch: [16][487/1133]	Eit 18600  lr 5e-05  Le 90.2708 (58.5272)	Time 1.225 (0.000)	Data 0.002 (0.000)	
2021-06-25 02:23:29,585 Epoch: [16][687/1133]	Eit 18800  lr 5e-05  Le 50.1139 (57.9301)	Time 0.984 (0.000)	Data 0.002 (0.000)	
2021-06-25 02:27:25,052 Epoch: [16][887/1133]	Eit 19000  lr 5e-05  Le 70.7412 (57.7791)	Time 1.408 (0.000)	Data 0.003 (0.000)	
2021-06-25 02:31:20,221 Epoch: [16][1087/1133]	Eit 19200  lr 5e-05  Le 57.0089 (57.6469)	Time 1.125 (0.000)	Data 0.003 (0.000)	
2021-06-25 02:32:16,341 Test: [0/40]	Le 294.0128 (294.0126)	Time 3.873 (0.000)	
2021-06-25 02:32:33,654 calculate similarity time: 0.07276439666748047
2021-06-25 02:32:34,178 Image to text: 71.9, 93.1, 96.2, 1.0, 2.5
2021-06-25 02:32:34,618 Text to image: 56.8, 83.3, 90.3, 1.0, 6.4
2021-06-25 02:32:34,618 Current rsum is 491.55999999999995
2021-06-25 02:32:38,267 runs/f30k_butd_region_bert/log
2021-06-25 02:32:38,267 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-25 02:32:38,271 image encoder trainable parameters: 20490144
2021-06-25 02:32:38,281 txt encoder trainable parameters: 137319072
2021-06-25 02:35:42,775 Epoch: [17][155/1133]	Eit 19400  lr 5e-05  Le 64.6748 (52.5627)	Time 1.063 (0.000)	Data 0.002 (0.000)	
2021-06-25 02:39:29,551 Epoch: [17][355/1133]	Eit 19600  lr 5e-05  Le 58.6421 (53.9342)	Time 1.342 (0.000)	Data 0.002 (0.000)	
2021-06-25 02:43:22,867 Epoch: [17][555/1133]	Eit 19800  lr 5e-05  Le 36.0451 (54.5491)	Time 1.111 (0.000)	Data 0.002 (0.000)	
2021-06-25 02:47:16,977 Epoch: [17][755/1133]	Eit 20000  lr 5e-05  Le 57.9976 (53.7805)	Time 1.135 (0.000)	Data 0.002 (0.000)	
2021-06-25 02:51:11,708 Epoch: [17][955/1133]	Eit 20200  lr 5e-05  Le 44.9779 (53.9158)	Time 1.339 (0.000)	Data 0.002 (0.000)	
2021-06-25 02:54:44,293 Test: [0/40]	Le 288.1492 (288.1489)	Time 3.167 (0.000)	
2021-06-25 02:55:01,635 calculate similarity time: 0.06722831726074219
2021-06-25 02:55:02,103 Image to text: 72.2, 93.3, 96.5, 1.0, 2.5
2021-06-25 02:55:02,417 Text to image: 57.1, 83.3, 90.2, 1.0, 6.3
2021-06-25 02:55:02,417 Current rsum is 492.53999999999996
2021-06-25 02:55:06,159 runs/f30k_butd_region_bert/log
2021-06-25 02:55:06,160 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-25 02:55:06,163 image encoder trainable parameters: 20490144
2021-06-25 02:55:06,171 txt encoder trainable parameters: 137319072
2021-06-25 02:55:37,036 Epoch: [18][23/1133]	Eit 20400  lr 5e-05  Le 37.6707 (46.5574)	Time 1.407 (0.000)	Data 0.002 (0.000)	
2021-06-25 02:59:29,159 Epoch: [18][223/1133]	Eit 20600  lr 5e-05  Le 41.7481 (50.1880)	Time 1.382 (0.000)	Data 0.009 (0.000)	
2021-06-25 03:03:23,669 Epoch: [18][423/1133]	Eit 20800  lr 5e-05  Le 64.2565 (50.9279)	Time 0.979 (0.000)	Data 0.002 (0.000)	
2021-06-25 03:07:09,114 Epoch: [18][623/1133]	Eit 21000  lr 5e-05  Le 37.5057 (50.8916)	Time 1.260 (0.000)	Data 0.002 (0.000)	
2021-06-25 03:11:01,088 Epoch: [18][823/1133]	Eit 21200  lr 5e-05  Le 35.7308 (51.6408)	Time 0.992 (0.000)	Data 0.002 (0.000)	
2021-06-25 03:14:57,329 Epoch: [18][1023/1133]	Eit 21400  lr 5e-05  Le 55.6694 (51.8903)	Time 1.160 (0.000)	Data 0.002 (0.000)	
2021-06-25 03:17:09,570 Test: [0/40]	Le 289.6690 (289.6688)	Time 3.728 (0.000)	
2021-06-25 03:17:26,382 calculate similarity time: 0.08417820930480957
2021-06-25 03:17:26,909 Image to text: 72.0, 93.0, 96.6, 1.0, 2.5
2021-06-25 03:17:27,354 Text to image: 57.0, 83.3, 89.9, 1.0, 6.5
2021-06-25 03:17:27,354 Current rsum is 491.86000000000007
2021-06-25 03:17:28,875 runs/f30k_butd_region_bert/log
2021-06-25 03:17:28,875 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-25 03:17:28,876 image encoder trainable parameters: 20490144
2021-06-25 03:17:28,882 txt encoder trainable parameters: 137319072
2021-06-25 03:19:17,962 Epoch: [19][91/1133]	Eit 21600  lr 5e-05  Le 44.4880 (53.1731)	Time 0.989 (0.000)	Data 0.002 (0.000)	
2021-06-25 03:23:11,726 Epoch: [19][291/1133]	Eit 21800  lr 5e-05  Le 29.2836 (50.0204)	Time 1.518 (0.000)	Data 0.002 (0.000)	
2021-06-25 03:27:08,699 Epoch: [19][491/1133]	Eit 22000  lr 5e-05  Le 43.3618 (50.2323)	Time 1.034 (0.000)	Data 0.002 (0.000)	
2021-06-25 03:30:59,589 Epoch: [19][691/1133]	Eit 22200  lr 5e-05  Le 38.9665 (50.2342)	Time 1.093 (0.000)	Data 0.002 (0.000)	
2021-06-25 03:34:48,697 Epoch: [19][891/1133]	Eit 22400  lr 5e-05  Le 37.6765 (50.3884)	Time 0.653 (0.000)	Data 0.002 (0.000)	
2021-06-25 03:38:38,496 Epoch: [19][1091/1133]	Eit 22600  lr 5e-05  Le 65.2158 (50.3652)	Time 1.271 (0.000)	Data 0.002 (0.000)	
2021-06-25 03:39:29,236 Test: [0/40]	Le 292.8117 (292.8114)	Time 3.450 (0.000)	
2021-06-25 03:39:46,761 calculate similarity time: 0.06373858451843262
2021-06-25 03:39:47,285 Image to text: 72.7, 93.0, 96.4, 1.0, 2.5
2021-06-25 03:39:47,620 Text to image: 56.9, 83.5, 90.3, 1.0, 6.3
2021-06-25 03:39:47,620 Current rsum is 492.82000000000005
2021-06-25 03:39:51,338 runs/f30k_butd_region_bert/log
2021-06-25 03:39:51,338 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-25 03:39:51,342 image encoder trainable parameters: 20490144
2021-06-25 03:39:51,354 txt encoder trainable parameters: 137319072
2021-06-25 03:43:01,596 Epoch: [20][159/1133]	Eit 22800  lr 5e-05  Le 76.8765 (53.1579)	Time 0.988 (0.000)	Data 0.002 (0.000)	
2021-06-25 03:46:56,527 Epoch: [20][359/1133]	Eit 23000  lr 5e-05  Le 72.8545 (52.4162)	Time 1.412 (0.000)	Data 0.002 (0.000)	
2021-06-25 03:50:49,378 Epoch: [20][559/1133]	Eit 23200  lr 5e-05  Le 70.4906 (50.8931)	Time 1.249 (0.000)	Data 0.002 (0.000)	
2021-06-25 03:54:40,611 Epoch: [20][759/1133]	Eit 23400  lr 5e-05  Le 48.7262 (50.5987)	Time 1.276 (0.000)	Data 0.002 (0.000)	
2021-06-25 03:58:36,248 Epoch: [20][959/1133]	Eit 23600  lr 5e-05  Le 37.1696 (50.6887)	Time 0.979 (0.000)	Data 0.002 (0.000)	
2021-06-25 04:02:01,877 Test: [0/40]	Le 286.9624 (286.9622)	Time 3.642 (0.000)	
2021-06-25 04:02:19,153 calculate similarity time: 0.08819770812988281
2021-06-25 04:02:19,760 Image to text: 73.1, 93.6, 96.6, 1.0, 2.5
2021-06-25 04:02:20,181 Text to image: 57.2, 83.8, 90.2, 1.0, 6.4
2021-06-25 04:02:20,182 Current rsum is 494.4599999999999
2021-06-25 04:02:23,829 runs/f30k_butd_region_bert/log
2021-06-25 04:02:23,829 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-25 04:02:23,833 image encoder trainable parameters: 20490144
2021-06-25 04:02:23,845 txt encoder trainable parameters: 137319072
2021-06-25 04:02:57,576 Epoch: [21][27/1133]	Eit 23800  lr 5e-05  Le 28.9254 (42.6579)	Time 0.982 (0.000)	Data 0.003 (0.000)	
2021-06-25 04:06:46,934 Epoch: [21][227/1133]	Eit 24000  lr 5e-05  Le 37.6528 (47.8557)	Time 1.288 (0.000)	Data 0.002 (0.000)	
2021-06-25 04:10:43,591 Epoch: [21][427/1133]	Eit 24200  lr 5e-05  Le 45.2115 (49.3176)	Time 1.373 (0.000)	Data 0.002 (0.000)	
2021-06-25 04:14:36,389 Epoch: [21][627/1133]	Eit 24400  lr 5e-05  Le 32.8796 (48.9819)	Time 0.990 (0.000)	Data 0.005 (0.000)	
2021-06-25 04:18:28,575 Epoch: [21][827/1133]	Eit 24600  lr 5e-05  Le 63.5175 (49.0524)	Time 1.228 (0.000)	Data 0.002 (0.000)	
2021-06-25 04:22:25,435 Epoch: [21][1027/1133]	Eit 24800  lr 5e-05  Le 22.1234 (48.7475)	Time 1.047 (0.000)	Data 0.002 (0.000)	
2021-06-25 04:24:28,346 Test: [0/40]	Le 300.5771 (300.5769)	Time 3.405 (0.000)	
2021-06-25 04:24:45,558 calculate similarity time: 0.07784867286682129
2021-06-25 04:24:46,056 Image to text: 72.0, 93.5, 96.9, 1.0, 2.5
2021-06-25 04:24:46,382 Text to image: 57.6, 83.7, 90.0, 1.0, 6.4
2021-06-25 04:24:46,382 Current rsum is 493.76
2021-06-25 04:24:47,882 runs/f30k_butd_region_bert/log
2021-06-25 04:24:47,882 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-25 04:24:47,884 image encoder trainable parameters: 20490144
2021-06-25 04:24:47,889 txt encoder trainable parameters: 137319072
2021-06-25 04:26:44,378 Epoch: [22][95/1133]	Eit 25000  lr 5e-05  Le 48.1488 (49.8325)	Time 1.383 (0.000)	Data 0.002 (0.000)	
2021-06-25 04:30:38,678 Epoch: [22][295/1133]	Eit 25200  lr 5e-05  Le 36.9955 (48.1352)	Time 0.985 (0.000)	Data 0.002 (0.000)	
2021-06-25 04:34:24,436 Epoch: [22][495/1133]	Eit 25400  lr 5e-05  Le 64.1003 (48.1072)	Time 1.407 (0.000)	Data 0.002 (0.000)	
2021-06-25 04:38:20,048 Epoch: [22][695/1133]	Eit 25600  lr 5e-05  Le 72.4469 (48.3538)	Time 1.241 (0.000)	Data 0.002 (0.000)	
2021-06-25 04:42:14,249 Epoch: [22][895/1133]	Eit 25800  lr 5e-05  Le 65.6395 (48.4091)	Time 1.285 (0.000)	Data 0.002 (0.000)	
2021-06-25 04:46:08,620 Epoch: [22][1095/1133]	Eit 26000  lr 5e-05  Le 33.4548 (48.1697)	Time 1.510 (0.000)	Data 0.002 (0.000)	
2021-06-25 04:46:54,065 Test: [0/40]	Le 293.1266 (293.1264)	Time 3.432 (0.000)	
2021-06-25 04:47:11,536 calculate similarity time: 0.07410836219787598
2021-06-25 04:47:12,088 Image to text: 73.6, 93.1, 96.7, 1.0, 2.4
2021-06-25 04:47:12,398 Text to image: 57.4, 83.9, 90.1, 1.0, 6.4
2021-06-25 04:47:12,398 Current rsum is 494.79999999999995
2021-06-25 04:47:15,910 runs/f30k_butd_region_bert/log
2021-06-25 04:47:15,910 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-25 04:47:15,912 image encoder trainable parameters: 20490144
2021-06-25 04:47:15,918 txt encoder trainable parameters: 137319072
2021-06-25 04:50:31,762 Epoch: [23][163/1133]	Eit 26200  lr 5e-05  Le 73.8456 (46.9551)	Time 1.298 (0.000)	Data 0.002 (0.000)	
2021-06-25 04:54:23,523 Epoch: [23][363/1133]	Eit 26400  lr 5e-05  Le 49.6288 (46.7792)	Time 1.230 (0.000)	Data 0.002 (0.000)	
2021-06-25 04:58:18,266 Epoch: [23][563/1133]	Eit 26600  lr 5e-05  Le 28.4227 (46.9722)	Time 1.360 (0.000)	Data 0.002 (0.000)	
2021-06-25 05:02:04,238 Epoch: [23][763/1133]	Eit 26800  lr 5e-05  Le 43.0975 (47.1428)	Time 1.258 (0.000)	Data 0.014 (0.000)	
2021-06-25 05:05:55,794 Epoch: [23][963/1133]	Eit 27000  lr 5e-05  Le 40.2992 (47.6552)	Time 1.002 (0.000)	Data 0.002 (0.000)	
2021-06-25 05:09:20,365 Test: [0/40]	Le 290.4921 (290.4918)	Time 3.637 (0.000)	
2021-06-25 05:09:37,426 calculate similarity time: 0.06838512420654297
2021-06-25 05:09:37,966 Image to text: 73.3, 92.7, 96.9, 1.0, 2.4
2021-06-25 05:09:38,421 Text to image: 57.6, 83.9, 90.4, 1.0, 6.4
2021-06-25 05:09:38,421 Current rsum is 494.76
2021-06-25 05:09:39,932 runs/f30k_butd_region_bert/log
2021-06-25 05:09:39,933 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-25 05:09:39,934 image encoder trainable parameters: 20490144
2021-06-25 05:09:39,940 txt encoder trainable parameters: 137319072
2021-06-25 05:10:19,439 Epoch: [24][31/1133]	Eit 27200  lr 5e-05  Le 50.5971 (49.4982)	Time 1.404 (0.000)	Data 0.002 (0.000)	
2021-06-25 05:14:11,836 Epoch: [24][231/1133]	Eit 27400  lr 5e-05  Le 54.9324 (48.8086)	Time 1.249 (0.000)	Data 0.002 (0.000)	
2021-06-25 05:18:05,483 Epoch: [24][431/1133]	Eit 27600  lr 5e-05  Le 27.6975 (47.4442)	Time 1.099 (0.000)	Data 0.002 (0.000)	
2021-06-25 05:21:58,955 Epoch: [24][631/1133]	Eit 27800  lr 5e-05  Le 29.7301 (47.1362)	Time 1.430 (0.000)	Data 0.002 (0.000)	
2021-06-25 05:25:54,841 Epoch: [24][831/1133]	Eit 28000  lr 5e-05  Le 22.7564 (46.8309)	Time 1.263 (0.000)	Data 0.003 (0.000)	
2021-06-25 05:29:47,065 Epoch: [24][1031/1133]	Eit 28200  lr 5e-05  Le 60.0976 (46.8733)	Time 0.946 (0.000)	Data 0.003 (0.000)	
2021-06-25 05:31:45,578 Test: [0/40]	Le 292.8667 (292.8664)	Time 3.568 (0.000)	
2021-06-25 05:32:02,898 calculate similarity time: 0.07123208045959473
2021-06-25 05:32:03,365 Image to text: 74.4, 93.6, 96.7, 1.0, 2.4
2021-06-25 05:32:03,675 Text to image: 57.8, 84.2, 90.3, 1.0, 6.4
2021-06-25 05:32:03,676 Current rsum is 497.02
You have new mail in /var/spool/mail/root
[root@gpu1 vse_infty-master-my-graph-gru-vse]# CUDA_VISIBLE_DEVICES=0,1 python3 eval.py --dataset f30k --data_path ../data/f30k
INFO:root:Evaluating runs/f30k_butd_region_bert...
INFO:lib.evaluation:Namespace(backbone_lr_factor=0.01, backbone_path='', backbone_source='detector', backbone_warmup_epochs=5, batch_size=128, data_name='f30k', data_path='../data/f30k', embed_size=1024, embedding_warmup_epochs=2, grad_clip=2.0, img_dim=2048, input_scale_factor=1, learning_rate=0.0005, log_step=200, logger_name='runs/f30k_butd_region_bert/log', lr_update=15, margin=0.2, max_violation=True, model_name='runs/f30k_butd_region_bert', no_imgnorm=False, no_txtnorm=False, num_epochs=25, optim='adam', precomp_enc_type='basic', reset_start_epoch=False, resume='', val_step=500, vocab_path='./vocab/', vocab_size=30522, vse_mean_warmup_epochs=1, word_dim=300, workers=5)
INFO:transformers.tokenization_utils:loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-vse/lib/encoders.py:48: UserWarning: nn.init.constant is now deprecated in favor of nn.init.constant_.
  nn.init.constant(self.W[1].weight, 0)
/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-vse/lib/encoders.py:49: UserWarning: nn.init.constant is now deprecated in favor of nn.init.constant_.
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
INFO:lib.evaluation:Test: [0/40]	Le 330.9362 (330.9360)	Time 4.489 (0.000)	
INFO:lib.evaluation:Test: [10/40]	Le 288.1095 (314.4824)	Time 0.350 (0.000)	
INFO:lib.evaluation:Test: [20/40]	Le 311.9587 (319.6136)	Time 0.255 (0.000)	
INFO:lib.evaluation:Test: [30/40]	Le 486.0343 (347.4700)	Time 0.439 (0.000)	
INFO:lib.evaluation:Images: 1000, Captions: 5000
INFO:lib.evaluation:calculate similarity time: 0.0751643180847168
INFO:lib.evaluation:rsum: 497.1
INFO:lib.evaluation:Average i2t Recall: 87.6
INFO:lib.evaluation:Image to text: 74.3 92.4 96.0 1.0 2.9
INFO:lib.evaluation:Average t2i Recall: 78.1
INFO:lib.evaluation:Text to image: 58.5 84.8 91.1 1.0 5.8
INFO:root:Evaluating runs/release_weights/f30k_butd_grid_bert...
Traceback (most recent call last):
  File "eval.py", line 58, in <module>
    main()
  File "eval.py", line 54, in main
    evaluation.evalrank(model_path, data_path=opt.data_path, split='test', fold5=False, save_path=save_path)
  File "/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-vse/lib/evaluation.py", line 196, in evalrank
    checkpoint = torch.load(model_path)
  File "/root/anaconda3/lib/python3.6/site-packages/torch/serialization.py", line 571, in load
    with _open_file_like(f, 'rb') as opened_file:
  File "/root/anaconda3/lib/python3.6/site-packages/torch/serialization.py", line 229, in _open_file_like
    return _open_file(name_or_buffer, mode)
  File "/root/anaconda3/lib/python3.6/site-packages/torch/serialization.py", line 210, in __init__
    super(_open_file, self).__init__(open(name, mode))
FileNotFoundError: [Errno 2] No such file or directory: 'runs/release_weights/f30k_butd_grid_bert/model_best.pth'
[root@gpu1 vse_infty-master-my-graph-gru-vse]# 

