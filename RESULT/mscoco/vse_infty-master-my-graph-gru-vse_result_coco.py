[root@gpu1 vse_infty-master-my-graph-gru-vse]# sh train_region.sh 
2021-06-21 08:47:05,117 Namespace(backbone_lr_factor=0.01, backbone_path='', backbone_source='detector', backbone_warmup_epochs=5, batch_size=128, data_name='coco', data_path='../data/coco', embed_size=1024, embedding_warmup_epochs=2, grad_clip=2.0, img_dim=2048, input_scale_factor=1, learning_rate=0.0005, log_step=200, logger_name='runs/coco_butd_region_bert/log', lr_update=15, margin=0.2, max_violation=False, model_name='runs/coco_butd_region_bert', no_imgnorm=False, no_txtnorm=False, num_epochs=25, optim='adam', precomp_enc_type='basic', reset_start_epoch=False, resume='', val_step=500, vocab_path='./vocab/', vse_mean_warmup_epochs=1, word_dim=300, workers=10)
2021-06-21 08:47:05,118 Model name '/public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased' not found in model shortcut name list (bert-base-uncased, bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, bert-base-multilingual-cased, bert-base-chinese, bert-base-german-cased, bert-large-uncased-whole-word-masking, bert-large-cased-whole-word-masking, bert-large-uncased-whole-word-masking-finetuned-squad, bert-large-cased-whole-word-masking-finetuned-squad, bert-base-cased-finetuned-mrpc). Assuming '/public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased' is a path or url to a directory containing tokenizer files.
2021-06-21 08:47:05,118 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/added_tokens.json. We won't load it.
2021-06-21 08:47:05,118 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/special_tokens_map.json. We won't load it.
2021-06-21 08:47:05,118 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/tokenizer_config.json. We won't load it.
2021-06-21 08:47:05,118 loading file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/vocab.txt
2021-06-21 08:47:05,118 loading file None
2021-06-21 08:47:05,118 loading file None
2021-06-21 08:47:05,118 loading file None
/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-vse/lib/encoders.py:48: UserWarning: nn.init.constant is now deprecated in favor of nn.init.constant_.
  nn.init.constant(self.W[1].weight, 0)
/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-vse/lib/encoders.py:49: UserWarning: nn.init.constant is now deprecated in favor of nn.init.constant_.
  nn.init.constant(self.W[1].bias, 0)
2021-06-21 08:47:34,337 loading configuration file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/config.json
2021-06-21 08:47:34,338 Model config {
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

2021-06-21 08:47:34,339 loading weights file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/pytorch_model.bin
2021-06-21 08:47:42,377 Use adam as the optimizer, with init lr 0.0005
2021-06-21 08:47:42,379 Image encoder is data paralleled now.
2021-06-21 08:47:42,379 runs/coco_butd_region_bert/log
2021-06-21 08:47:42,379 runs/coco_butd_region_bert
2021-06-21 08:47:42,383 image encoder trainable parameters: 20490144
2021-06-21 08:47:42,395 txt encoder trainable parameters: 137319072
/root/anaconda3/lib/python3.6/site-packages/torch/nn/modules/rnn.py:735: UserWarning: RNN module weights are not part of single contiguous chunk of memory. This means they need to be compacted at every call, possibly greatly increasing memory usage. To compact weights again call flatten_parameters(). (Triggered internally at  /pytorch/aten/src/ATen/native/cudnn/RNN.cpp:1264.)
  self.dropout, self.training, self.bidirectional, self.batch_first)
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
  File "/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-vse/lib/encoders.py", line 298, in forward
    GCN_img_emd = self.Rs_GCN_1(GCN_img_emd)
  File "/root/anaconda3/lib/python3.6/site-packages/torch/nn/modules/module.py", line 722, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-vse/lib/encoders.py", line 75, in forward
    g_v = self.g(v).view(batch_size, self.inter_channels, -1)
  File "/root/anaconda3/lib/python3.6/site-packages/torch/nn/modules/module.py", line 722, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/root/anaconda3/lib/python3.6/site-packages/torch/nn/modules/conv.py", line 257, in forward
    self.padding, self.dilation, self.groups)
RuntimeError: Given groups=1, weight of size [1024, 1024, 1], expected input[64, 2048, 26] to have 1024 channels, but got 2048 channels instead

[root@gpu1 vse_infty-master-my-graph-gru-vse]# sh train_region.sh 
2021-06-21 08:52:23,730 Namespace(backbone_lr_factor=0.01, backbone_path='', backbone_source='detector', backbone_warmup_epochs=5, batch_size=128, data_name='coco', data_path='../data/coco', embed_size=1024, embedding_warmup_epochs=2, grad_clip=2.0, img_dim=2048, input_scale_factor=1, learning_rate=0.0005, log_step=200, logger_name='runs/coco_butd_region_bert/log', lr_update=15, margin=0.2, max_violation=False, model_name='runs/coco_butd_region_bert', no_imgnorm=False, no_txtnorm=False, num_epochs=25, optim='adam', precomp_enc_type='basic', reset_start_epoch=False, resume='', val_step=500, vocab_path='./vocab/', vse_mean_warmup_epochs=1, word_dim=300, workers=10)
2021-06-21 08:52:23,730 Model name '/public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased' not found in model shortcut name list (bert-base-uncased, bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, bert-base-multilingual-cased, bert-base-chinese, bert-base-german-cased, bert-large-uncased-whole-word-masking, bert-large-cased-whole-word-masking, bert-large-uncased-whole-word-masking-finetuned-squad, bert-large-cased-whole-word-masking-finetuned-squad, bert-base-cased-finetuned-mrpc). Assuming '/public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased' is a path or url to a directory containing tokenizer files.
2021-06-21 08:52:23,730 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/added_tokens.json. We won't load it.
2021-06-21 08:52:23,730 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/special_tokens_map.json. We won't load it.
2021-06-21 08:52:23,730 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/tokenizer_config.json. We won't load it.
2021-06-21 08:52:23,730 loading file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/vocab.txt
2021-06-21 08:52:23,730 loading file None
2021-06-21 08:52:23,730 loading file None
2021-06-21 08:52:23,731 loading file None
/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-vse/lib/encoders.py:48: UserWarning: nn.init.constant is now deprecated in favor of nn.init.constant_.
  nn.init.constant(self.W[1].weight, 0)
/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-vse/lib/encoders.py:49: UserWarning: nn.init.constant is now deprecated in favor of nn.init.constant_.
  nn.init.constant(self.W[1].bias, 0)
2021-06-21 08:52:52,914 loading configuration file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/config.json
2021-06-21 08:52:52,915 Model config {
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

2021-06-21 08:52:52,915 loading weights file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/pytorch_model.bin
2021-06-21 08:53:00,600 Use adam as the optimizer, with init lr 0.0005
2021-06-21 08:53:00,601 Image encoder is data paralleled now.
2021-06-21 08:53:00,601 runs/coco_butd_region_bert/log
2021-06-21 08:53:00,601 runs/coco_butd_region_bert
2021-06-21 08:53:00,604 image encoder trainable parameters: 20490144
2021-06-21 08:53:00,612 txt encoder trainable parameters: 137319072
/root/anaconda3/lib/python3.6/site-packages/torch/nn/modules/rnn.py:735: UserWarning: RNN module weights are not part of single contiguous chunk of memory. This means they need to be compacted at every call, possibly greatly increasing memory usage. To compact weights again call flatten_parameters(). (Triggered internally at  /pytorch/aten/src/ATen/native/cudnn/RNN.cpp:1264.)
  self.dropout, self.training, self.bidirectional, self.batch_first)
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
  File "/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-vse/lib/encoders.py", line 298, in forward
    GCN_img_emd = self.Rs_GCN_1(GCN_img_emd)
  File "/root/anaconda3/lib/python3.6/site-packages/torch/nn/modules/module.py", line 722, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-vse/lib/encoders.py", line 75, in forward
    g_v = self.g(v).view(batch_size, self.inter_channels, -1)
  File "/root/anaconda3/lib/python3.6/site-packages/torch/nn/modules/module.py", line 722, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/root/anaconda3/lib/python3.6/site-packages/torch/nn/modules/conv.py", line 257, in forward
    self.padding, self.dilation, self.groups)
RuntimeError: Given groups=1, weight of size [1024, 1024, 1], expected input[64, 2048, 21] to have 1024 channels, but got 2048 channels instead

[root@gpu1 vse_infty-master-my-graph-gru-vse]# sh train_region.sh 
2021-06-21 11:49:14,347 Namespace(backbone_lr_factor=0.01, backbone_path='', backbone_source='detector', backbone_warmup_epochs=5, batch_size=128, data_name='coco', data_path='../data/coco', embed_size=1024, embedding_warmup_epochs=2, grad_clip=2.0, img_dim=2048, input_scale_factor=1, learning_rate=0.0005, log_step=200, logger_name='runs/coco_butd_region_bert/log', lr_update=15, margin=0.2, max_violation=False, model_name='runs/coco_butd_region_bert', no_imgnorm=False, no_txtnorm=False, num_epochs=25, optim='adam', precomp_enc_type='basic', reset_start_epoch=False, resume='', val_step=500, vocab_path='./vocab/', vse_mean_warmup_epochs=1, word_dim=300, workers=10)
2021-06-21 11:49:14,347 Model name '/public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased' not found in model shortcut name list (bert-base-uncased, bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, bert-base-multilingual-cased, bert-base-chinese, bert-base-german-cased, bert-large-uncased-whole-word-masking, bert-large-cased-whole-word-masking, bert-large-uncased-whole-word-masking-finetuned-squad, bert-large-cased-whole-word-masking-finetuned-squad, bert-base-cased-finetuned-mrpc). Assuming '/public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased' is a path or url to a directory containing tokenizer files.
2021-06-21 11:49:14,347 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/added_tokens.json. We won't load it.
2021-06-21 11:49:14,347 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/special_tokens_map.json. We won't load it.
2021-06-21 11:49:14,348 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/tokenizer_config.json. We won't load it.
2021-06-21 11:49:14,348 loading file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/vocab.txt
2021-06-21 11:49:14,348 loading file None
2021-06-21 11:49:14,348 loading file None
2021-06-21 11:49:14,348 loading file None
/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-vse/lib/encoders.py:48: UserWarning: nn.init.constant is now deprecated in favor of nn.init.constant_.
  nn.init.constant(self.W[1].weight, 0)
/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-vse/lib/encoders.py:49: UserWarning: nn.init.constant is now deprecated in favor of nn.init.constant_.
  nn.init.constant(self.W[1].bias, 0)
2021-06-21 11:49:45,802 loading configuration file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/config.json
2021-06-21 11:49:45,803 Model config {
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

2021-06-21 11:49:45,804 loading weights file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/pytorch_model.bin
2021-06-21 11:49:54,175 Use adam as the optimizer, with init lr 0.0005
2021-06-21 11:49:54,176 Image encoder is data paralleled now.
2021-06-21 11:49:54,176 runs/coco_butd_region_bert/log
2021-06-21 11:49:54,176 runs/coco_butd_region_bert
2021-06-21 11:49:54,178 image encoder trainable parameters: 20490144
2021-06-21 11:49:54,185 txt encoder trainable parameters: 137319072
/root/anaconda3/lib/python3.6/site-packages/torch/nn/modules/rnn.py:735: UserWarning: RNN module weights are not part of single contiguous chunk of memory. This means they need to be compacted at every call, possibly greatly increasing memory usage. To compact weights again call flatten_parameters(). (Triggered internally at  /pytorch/aten/src/ATen/native/cudnn/RNN.cpp:1264.)
  self.dropout, self.training, self.bidirectional, self.batch_first)
2021-06-21 11:53:26,462 Epoch: [0][199/4426]	Eit 200  lr 0.0005  Le 387.8069 (566.6723)	Time 0.920 (0.000)	Data 0.003 (0.000)	
2021-06-21 11:56:37,751 Epoch: [0][399/4426]	Eit 400  lr 0.0005  Le 261.5185 (435.7358)	Time 1.010 (0.000)	Data 0.002 (0.000)	
2021-06-21 11:59:47,094 Epoch: [0][599/4426]	Eit 600  lr 0.0005  Le 264.4520 (375.4538)	Time 0.988 (0.000)	Data 0.002 (0.000)	
2021-06-21 12:02:58,020 Epoch: [0][799/4426]	Eit 800  lr 0.0005  Le 235.6682 (338.6481)	Time 0.748 (0.000)	Data 0.002 (0.000)	
2021-06-21 12:06:07,663 Epoch: [0][999/4426]	Eit 1000  lr 0.0005  Le 185.0333 (313.5895)	Time 0.772 (0.000)	Data 0.002 (0.000)	
2021-06-21 12:09:15,541 Epoch: [0][1199/4426]	Eit 1200  lr 0.0005  Le 190.7135 (295.6032)	Time 1.038 (0.000)	Data 0.002 (0.000)	
2021-06-21 12:12:25,133 Epoch: [0][1399/4426]	Eit 1400  lr 0.0005  Le 156.3863 (280.7384)	Time 0.791 (0.000)	Data 0.002 (0.000)	
2021-06-21 12:15:34,378 Epoch: [0][1599/4426]	Eit 1600  lr 0.0005  Le 148.6849 (268.7804)	Time 1.085 (0.000)	Data 0.020 (0.000)	
2021-06-21 12:18:42,216 Epoch: [0][1799/4426]	Eit 1800  lr 0.0005  Le 156.4671 (259.2803)	Time 0.978 (0.000)	Data 0.002 (0.000)	
2021-06-21 12:21:50,554 Epoch: [0][1999/4426]	Eit 2000  lr 0.0005  Le 175.1576 (250.6233)	Time 0.919 (0.000)	Data 0.002 (0.000)	
2021-06-21 12:25:00,009 Epoch: [0][2199/4426]	Eit 2200  lr 0.0005  Le 166.7272 (243.8179)	Time 0.917 (0.000)	Data 0.002 (0.000)	
2021-06-21 12:28:09,014 Epoch: [0][2399/4426]	Eit 2400  lr 0.0005  Le 189.3710 (237.6272)	Time 0.867 (0.000)	Data 0.002 (0.000)	
2021-06-21 12:31:19,183 Epoch: [0][2599/4426]	Eit 2600  lr 0.0005  Le 128.0351 (231.9024)	Time 1.023 (0.000)	Data 0.004 (0.000)	
2021-06-21 12:34:28,025 Epoch: [0][2799/4426]	Eit 2800  lr 0.0005  Le 151.5873 (227.1506)	Time 1.017 (0.000)	Data 0.002 (0.000)	
2021-06-21 12:37:37,648 Epoch: [0][2999/4426]	Eit 3000  lr 0.0005  Le 133.3362 (222.3726)	Time 1.004 (0.000)	Data 0.002 (0.000)	
2021-06-21 12:40:46,527 Epoch: [0][3199/4426]	Eit 3200  lr 0.0005  Le 153.4745 (218.2213)	Time 0.875 (0.000)	Data 0.003 (0.000)	
2021-06-21 12:43:55,452 Epoch: [0][3399/4426]	Eit 3400  lr 0.0005  Le 170.1787 (214.4553)	Time 1.067 (0.000)	Data 0.002 (0.000)	
2021-06-21 12:47:05,379 Epoch: [0][3599/4426]	Eit 3600  lr 0.0005  Le 137.5500 (210.8212)	Time 0.935 (0.000)	Data 0.002 (0.000)	
2021-06-21 12:50:12,326 Epoch: [0][3799/4426]	Eit 3800  lr 0.0005  Le 105.7447 (207.7087)	Time 0.974 (0.000)	Data 0.002 (0.000)	
2021-06-21 12:53:22,076 Epoch: [0][3999/4426]	Eit 4000  lr 0.0005  Le 111.2899 (204.7355)	Time 1.141 (0.000)	Data 0.002 (0.000)	
2021-06-21 12:56:32,054 Epoch: [0][4199/4426]	Eit 4200  lr 0.0005  Le 157.2053 (202.0604)	Time 0.773 (0.000)	Data 0.002 (0.000)	
2021-06-21 12:59:43,976 Epoch: [0][4399/4426]	Eit 4400  lr 0.0005  Le 117.4107 (199.2256)	Time 0.922 (0.000)	Data 0.002 (0.000)	
2021-06-21 13:00:16,438 Test: [0/40]	Le 319.4269 (319.4267)	Time 7.730 (0.000)	
2021-06-21 13:00:31,812 calculate similarity time: 0.05461621284484863
2021-06-21 13:00:32,195 Image to text: 57.2, 87.5, 94.1, 1.0, 3.4
2021-06-21 13:00:32,510 Text to image: 46.1, 81.1, 92.2, 2.0, 4.3
2021-06-21 13:00:32,510 Current rsum is 458.2
2021-06-21 13:00:35,214 runs/coco_butd_region_bert/log
2021-06-21 13:00:35,214 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-21 13:00:35,215 image encoder trainable parameters: 20490144
2021-06-21 13:00:35,220 txt encoder trainable parameters: 137319072
2021-06-21 13:03:27,815 Epoch: [1][174/4426]	Eit 4600  lr 0.0005  Le 216.5433 (133.1432)	Time 1.049 (0.000)	Data 0.002 (0.000)	
2021-06-21 13:06:35,319 Epoch: [1][374/4426]	Eit 4800  lr 0.0005  Le 123.9803 (133.0964)	Time 0.928 (0.000)	Data 0.002 (0.000)	
2021-06-21 13:09:43,759 Epoch: [1][574/4426]	Eit 5000  lr 0.0005  Le 95.6103 (133.9341)	Time 1.136 (0.000)	Data 0.002 (0.000)	
2021-06-21 13:12:54,523 Epoch: [1][774/4426]	Eit 5200  lr 0.0005  Le 103.6219 (134.8040)	Time 0.762 (0.000)	Data 0.003 (0.000)	
2021-06-21 13:16:03,673 Epoch: [1][974/4426]	Eit 5400  lr 0.0005  Le 83.1920 (134.5181)	Time 0.891 (0.000)	Data 0.002 (0.000)	
2021-06-21 13:19:13,210 Epoch: [1][1174/4426]	Eit 5600  lr 0.0005  Le 98.9194 (134.2725)	Time 1.036 (0.000)	Data 0.025 (0.000)	
2021-06-21 13:22:23,024 Epoch: [1][1374/4426]	Eit 5800  lr 0.0005  Le 76.9330 (133.1593)	Time 0.934 (0.000)	Data 0.002 (0.000)	
2021-06-21 13:25:27,387 Epoch: [1][1574/4426]	Eit 6000  lr 0.0005  Le 107.9346 (132.1399)	Time 0.915 (0.000)	Data 0.002 (0.000)	
2021-06-21 13:28:31,623 Epoch: [1][1774/4426]	Eit 6200  lr 0.0005  Le 114.6692 (131.3692)	Time 0.907 (0.000)	Data 0.002 (0.000)	
2021-06-21 13:31:38,347 Epoch: [1][1974/4426]	Eit 6400  lr 0.0005  Le 114.0404 (131.4011)	Time 0.957 (0.000)	Data 0.002 (0.000)	
2021-06-21 13:34:46,581 Epoch: [1][2174/4426]	Eit 6600  lr 0.0005  Le 74.3425 (130.6333)	Time 0.919 (0.000)	Data 0.002 (0.000)	
2021-06-21 13:37:54,543 Epoch: [1][2374/4426]	Eit 6800  lr 0.0005  Le 105.7746 (130.4842)	Time 1.061 (0.000)	Data 0.002 (0.000)	
2021-06-21 13:41:04,368 Epoch: [1][2574/4426]	Eit 7000  lr 0.0005  Le 110.0911 (130.0173)	Time 0.985 (0.000)	Data 0.002 (0.000)	
2021-06-21 13:44:13,686 Epoch: [1][2774/4426]	Eit 7200  lr 0.0005  Le 88.4098 (129.6983)	Time 1.218 (0.000)	Data 0.002 (0.000)	
2021-06-21 13:47:21,367 Epoch: [1][2974/4426]	Eit 7400  lr 0.0005  Le 85.7462 (129.1488)	Time 0.840 (0.000)	Data 0.002 (0.000)	
2021-06-21 13:50:30,886 Epoch: [1][3174/4426]	Eit 7600  lr 0.0005  Le 145.3469 (128.7374)	Time 0.950 (0.000)	Data 0.002 (0.000)	
2021-06-21 13:53:39,737 Epoch: [1][3374/4426]	Eit 7800  lr 0.0005  Le 94.1092 (128.4198)	Time 0.980 (0.000)	Data 0.004 (0.000)	
2021-06-21 13:56:49,711 Epoch: [1][3574/4426]	Eit 8000  lr 0.0005  Le 94.9688 (128.1583)	Time 0.778 (0.000)	Data 0.002 (0.000)	
2021-06-21 13:59:58,070 Epoch: [1][3774/4426]	Eit 8200  lr 0.0005  Le 148.1431 (127.9161)	Time 0.740 (0.000)	Data 0.002 (0.000)	
2021-06-21 14:03:07,179 Epoch: [1][3974/4426]	Eit 8400  lr 0.0005  Le 128.7622 (127.5310)	Time 0.828 (0.000)	Data 0.002 (0.000)	
2021-06-21 14:06:15,269 Epoch: [1][4174/4426]	Eit 8600  lr 0.0005  Le 111.5861 (127.0901)	Time 0.972 (0.000)	Data 0.002 (0.000)	
2021-06-21 14:09:24,481 Epoch: [1][4374/4426]	Eit 8800  lr 0.0005  Le 117.9942 (126.9581)	Time 0.992 (0.000)	Data 0.002 (0.000)	
2021-06-21 14:10:20,260 Test: [0/40]	Le 296.2293 (296.2290)	Time 7.616 (0.000)	
2021-06-21 14:10:35,392 calculate similarity time: 0.06563425064086914
2021-06-21 14:10:35,936 Image to text: 61.3, 89.0, 95.8, 1.0, 3.3
2021-06-21 14:10:36,247 Text to image: 49.7, 83.5, 93.5, 2.0, 4.3
2021-06-21 14:10:36,247 Current rsum is 472.82000000000005
2021-06-21 14:10:39,776 runs/coco_butd_region_bert/log
2021-06-21 14:10:39,776 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-21 14:10:39,780 image encoder trainable parameters: 20490144
2021-06-21 14:10:39,792 txt encoder trainable parameters: 137319072
2021-06-21 14:13:09,449 Epoch: [2][149/4426]	Eit 9000  lr 0.0005  Le 73.0330 (111.9021)	Time 0.775 (0.000)	Data 0.002 (0.000)	
2021-06-21 14:16:16,951 Epoch: [2][349/4426]	Eit 9200  lr 0.0005  Le 172.1823 (113.6992)	Time 0.799 (0.000)	Data 0.002 (0.000)	
2021-06-21 14:19:26,523 Epoch: [2][549/4426]	Eit 9400  lr 0.0005  Le 77.0415 (113.6187)	Time 1.199 (0.000)	Data 0.002 (0.000)	
2021-06-21 14:22:35,623 Epoch: [2][749/4426]	Eit 9600  lr 0.0005  Le 160.4630 (112.1814)	Time 0.957 (0.000)	Data 0.002 (0.000)	
2021-06-21 14:25:44,810 Epoch: [2][949/4426]	Eit 9800  lr 0.0005  Le 100.1323 (111.8881)	Time 0.898 (0.000)	Data 0.002 (0.000)	
2021-06-21 14:28:54,149 Epoch: [2][1149/4426]	Eit 10000  lr 0.0005  Le 94.5419 (111.5489)	Time 0.945 (0.000)	Data 0.002 (0.000)	
2021-06-21 14:32:02,347 Epoch: [2][1349/4426]	Eit 10200  lr 0.0005  Le 124.2282 (111.5023)	Time 1.034 (0.000)	Data 0.002 (0.000)	
2021-06-21 14:35:11,610 Epoch: [2][1549/4426]	Eit 10400  lr 0.0005  Le 110.6272 (111.3871)	Time 0.976 (0.000)	Data 0.002 (0.000)	
2021-06-21 14:38:21,351 Epoch: [2][1749/4426]	Eit 10600  lr 0.0005  Le 116.2347 (111.2364)	Time 0.790 (0.000)	Data 0.002 (0.000)	
2021-06-21 14:41:28,280 Epoch: [2][1949/4426]	Eit 10800  lr 0.0005  Le 148.9480 (111.1170)	Time 1.028 (0.000)	Data 0.002 (0.000)	
2021-06-21 14:44:37,191 Epoch: [2][2149/4426]	Eit 11000  lr 0.0005  Le 105.1754 (110.7653)	Time 1.090 (0.000)	Data 0.002 (0.000)	
2021-06-21 14:47:45,329 Epoch: [2][2349/4426]	Eit 11200  lr 0.0005  Le 130.6399 (110.2960)	Time 0.885 (0.000)	Data 0.003 (0.000)	
2021-06-21 14:50:53,563 Epoch: [2][2549/4426]	Eit 11400  lr 0.0005  Le 99.4356 (110.2292)	Time 0.834 (0.000)	Data 0.002 (0.000)	
2021-06-21 14:54:00,556 Epoch: [2][2749/4426]	Eit 11600  lr 0.0005  Le 99.0838 (109.9727)	Time 1.009 (0.000)	Data 0.002 (0.000)	
2021-06-21 14:57:07,707 Epoch: [2][2949/4426]	Eit 11800  lr 0.0005  Le 102.5254 (110.0694)	Time 0.865 (0.000)	Data 0.002 (0.000)	
2021-06-21 15:00:15,142 Epoch: [2][3149/4426]	Eit 12000  lr 0.0005  Le 74.7387 (109.8058)	Time 0.831 (0.000)	Data 0.003 (0.000)	
2021-06-21 15:03:21,876 Epoch: [2][3349/4426]	Eit 12200  lr 0.0005  Le 86.7144 (109.7981)	Time 0.957 (0.000)	Data 0.002 (0.000)	
2021-06-21 15:06:31,045 Epoch: [2][3549/4426]	Eit 12400  lr 0.0005  Le 99.6080 (109.5160)	Time 0.907 (0.000)	Data 0.003 (0.000)	
2021-06-21 15:09:28,165 Epoch: [2][3749/4426]	Eit 12600  lr 0.0005  Le 68.5150 (109.5914)	Time 1.667 (0.000)	Data 0.002 (0.000)	
2021-06-21 15:12:37,301 Epoch: [2][3949/4426]	Eit 12800  lr 0.0005  Le 100.6042 (109.5879)	Time 0.822 (0.000)	Data 0.002 (0.000)	
2021-06-21 15:15:45,371 Epoch: [2][4149/4426]	Eit 13000  lr 0.0005  Le 117.1887 (109.3759)	Time 1.001 (0.000)	Data 0.002 (0.000)	
2021-06-21 15:18:55,272 Epoch: [2][4349/4426]	Eit 13200  lr 0.0005  Le 88.5011 (109.2192)	Time 1.051 (0.000)	Data 0.002 (0.000)	
2021-06-21 15:20:15,025 Test: [0/40]	Le 285.5242 (285.5239)	Time 7.541 (0.000)	
2021-06-21 15:20:30,084 calculate similarity time: 0.0579068660736084
2021-06-21 15:20:30,510 Image to text: 64.2, 90.1, 97.0, 1.0, 3.3
2021-06-21 15:20:30,862 Text to image: 52.1, 85.3, 94.3, 1.0, 3.8
2021-06-21 15:20:30,862 Current rsum is 483.03999999999996
2021-06-21 15:20:34,322 runs/coco_butd_region_bert/log
2021-06-21 15:20:34,322 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-21 15:20:34,325 image encoder trainable parameters: 20490144
2021-06-21 15:20:34,336 txt encoder trainable parameters: 137319072
2021-06-21 15:22:39,198 Epoch: [3][124/4426]	Eit 13400  lr 0.0005  Le 101.5244 (98.6190)	Time 1.011 (0.000)	Data 0.002 (0.000)	
2021-06-21 15:25:49,219 Epoch: [3][324/4426]	Eit 13600  lr 0.0005  Le 112.5418 (98.3088)	Time 0.864 (0.000)	Data 0.002 (0.000)	
2021-06-21 15:28:57,118 Epoch: [3][524/4426]	Eit 13800  lr 0.0005  Le 95.6703 (97.6744)	Time 0.762 (0.000)	Data 0.002 (0.000)	
2021-06-21 15:32:04,551 Epoch: [3][724/4426]	Eit 14000  lr 0.0005  Le 72.5511 (98.2326)	Time 1.042 (0.000)	Data 0.002 (0.000)	
2021-06-21 15:35:13,697 Epoch: [3][924/4426]	Eit 14200  lr 0.0005  Le 47.5720 (98.6918)	Time 1.000 (0.000)	Data 0.002 (0.000)	
2021-06-21 15:38:21,705 Epoch: [3][1124/4426]	Eit 14400  lr 0.0005  Le 60.7970 (98.6842)	Time 0.919 (0.000)	Data 0.002 (0.000)	
2021-06-21 15:41:32,044 Epoch: [3][1324/4426]	Eit 14600  lr 0.0005  Le 68.9096 (98.5696)	Time 0.846 (0.000)	Data 0.002 (0.000)	
2021-06-21 15:44:40,204 Epoch: [3][1524/4426]	Eit 14800  lr 0.0005  Le 66.1177 (98.8830)	Time 1.077 (0.000)	Data 0.002 (0.000)	
2021-06-21 15:47:46,659 Epoch: [3][1724/4426]	Eit 15000  lr 0.0005  Le 78.2143 (98.7503)	Time 0.887 (0.000)	Data 0.002 (0.000)	
2021-06-21 15:50:55,990 Epoch: [3][1924/4426]	Eit 15200  lr 0.0005  Le 102.9008 (99.0370)	Time 1.012 (0.000)	Data 0.002 (0.000)	
2021-06-21 15:54:05,870 Epoch: [3][2124/4426]	Eit 15400  lr 0.0005  Le 80.4485 (99.1224)	Time 1.008 (0.000)	Data 0.002 (0.000)	
2021-06-21 15:57:16,761 Epoch: [3][2324/4426]	Eit 15600  lr 0.0005  Le 109.2587 (99.0309)	Time 0.945 (0.000)	Data 0.002 (0.000)	
2021-06-21 16:00:25,953 Epoch: [3][2524/4426]	Eit 15800  lr 0.0005  Le 129.1686 (98.8652)	Time 0.892 (0.000)	Data 0.002 (0.000)	
2021-06-21 16:03:34,703 Epoch: [3][2724/4426]	Eit 16000  lr 0.0005  Le 128.2365 (98.8440)	Time 1.183 (0.000)	Data 0.002 (0.000)	
2021-06-21 16:06:42,280 Epoch: [3][2924/4426]	Eit 16200  lr 0.0005  Le 63.0362 (98.5807)	Time 0.988 (0.000)	Data 0.002 (0.000)	
2021-06-21 16:09:50,346 Epoch: [3][3124/4426]	Eit 16400  lr 0.0005  Le 73.8064 (98.4718)	Time 0.814 (0.000)	Data 0.002 (0.000)	
2021-06-21 16:13:00,661 Epoch: [3][3324/4426]	Eit 16600  lr 0.0005  Le 73.3114 (98.3497)	Time 0.757 (0.000)	Data 0.002 (0.000)	
2021-06-21 16:16:09,535 Epoch: [3][3524/4426]	Eit 16800  lr 0.0005  Le 64.7635 (98.3392)	Time 1.070 (0.000)	Data 0.002 (0.000)	
2021-06-21 16:19:17,851 Epoch: [3][3724/4426]	Eit 17000  lr 0.0005  Le 67.3951 (98.3352)	Time 0.810 (0.000)	Data 0.002 (0.000)	
2021-06-21 16:22:25,470 Epoch: [3][3924/4426]	Eit 17200  lr 0.0005  Le 76.1254 (98.0968)	Time 0.927 (0.000)	Data 0.002 (0.000)	
2021-06-21 16:25:35,510 Epoch: [3][4124/4426]	Eit 17400  lr 0.0005  Le 68.2755 (98.1279)	Time 0.990 (0.000)	Data 0.002 (0.000)	
2021-06-21 16:28:45,122 Epoch: [3][4324/4426]	Eit 17600  lr 0.0005  Le 76.3491 (98.0054)	Time 0.772 (0.000)	Data 0.002 (0.000)	
2021-06-21 16:30:30,608 Test: [0/40]	Le 286.7041 (286.7039)	Time 7.585 (0.000)	
2021-06-21 16:30:46,045 calculate similarity time: 0.08072853088378906
2021-06-21 16:30:46,567 Image to text: 65.5, 90.6, 96.3, 1.0, 3.0
2021-06-21 16:30:47,001 Text to image: 52.9, 86.1, 94.8, 1.0, 4.0
2021-06-21 16:30:47,001 Current rsum is 486.15999999999997
2021-06-21 16:30:50,478 runs/coco_butd_region_bert/log
2021-06-21 16:30:50,479 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-21 16:30:50,482 image encoder trainable parameters: 20490144
2021-06-21 16:30:50,498 txt encoder trainable parameters: 137319072
2021-06-21 16:32:32,750 Epoch: [4][99/4426]	Eit 17800  lr 0.0005  Le 83.5063 (96.2455)	Time 1.172 (0.000)	Data 0.002 (0.000)	
2021-06-21 16:35:42,056 Epoch: [4][299/4426]	Eit 18000  lr 0.0005  Le 55.5601 (92.3676)	Time 1.000 (0.000)	Data 0.002 (0.000)	
2021-06-21 16:38:51,257 Epoch: [4][499/4426]	Eit 18200  lr 0.0005  Le 113.6004 (92.6824)	Time 0.859 (0.000)	Data 0.002 (0.000)	
2021-06-21 16:42:00,249 Epoch: [4][699/4426]	Eit 18400  lr 0.0005  Le 91.5083 (92.3532)	Time 0.962 (0.000)	Data 0.002 (0.000)	
2021-06-21 16:45:08,348 Epoch: [4][899/4426]	Eit 18600  lr 0.0005  Le 94.5599 (92.3720)	Time 0.983 (0.000)	Data 0.002 (0.000)	
2021-06-21 16:48:17,466 Epoch: [4][1099/4426]	Eit 18800  lr 0.0005  Le 125.1476 (92.1098)	Time 0.917 (0.000)	Data 0.002 (0.000)	
2021-06-21 16:51:23,452 Epoch: [4][1299/4426]	Eit 19000  lr 0.0005  Le 108.9961 (91.8961)	Time 0.580 (0.000)	Data 0.002 (0.000)	
2021-06-21 16:54:22,367 Epoch: [4][1499/4426]	Eit 19200  lr 0.0005  Le 66.4941 (91.7266)	Time 1.091 (0.000)	Data 0.002 (0.000)	
2021-06-21 16:57:31,029 Epoch: [4][1699/4426]	Eit 19400  lr 0.0005  Le 70.3335 (91.4596)	Time 0.969 (0.000)	Data 0.002 (0.000)	
2021-06-21 17:00:39,863 Epoch: [4][1899/4426]	Eit 19600  lr 0.0005  Le 87.7579 (91.6767)	Time 0.937 (0.000)	Data 0.002 (0.000)	
2021-06-21 17:03:46,508 Epoch: [4][2099/4426]	Eit 19800  lr 0.0005  Le 90.4390 (91.7044)	Time 0.860 (0.000)	Data 0.002 (0.000)	
2021-06-21 17:06:54,631 Epoch: [4][2299/4426]	Eit 20000  lr 0.0005  Le 140.2012 (91.9510)	Time 1.046 (0.000)	Data 0.002 (0.000)	
2021-06-21 17:10:03,522 Epoch: [4][2499/4426]	Eit 20200  lr 0.0005  Le 114.2811 (91.8729)	Time 0.942 (0.000)	Data 0.002 (0.000)	
2021-06-21 17:13:12,066 Epoch: [4][2699/4426]	Eit 20400  lr 0.0005  Le 67.6862 (91.7137)	Time 0.890 (0.000)	Data 0.002 (0.000)	
2021-06-21 17:16:21,804 Epoch: [4][2899/4426]	Eit 20600  lr 0.0005  Le 118.9944 (91.6892)	Time 1.155 (0.000)	Data 0.002 (0.000)	
2021-06-21 17:19:31,857 Epoch: [4][3099/4426]	Eit 20800  lr 0.0005  Le 77.7197 (91.6171)	Time 1.062 (0.000)	Data 0.002 (0.000)	
2021-06-21 17:22:41,215 Epoch: [4][3299/4426]	Eit 21000  lr 0.0005  Le 79.5065 (91.6439)	Time 1.022 (0.000)	Data 0.003 (0.000)	
2021-06-21 17:25:52,294 Epoch: [4][3499/4426]	Eit 21200  lr 0.0005  Le 64.0433 (91.6539)	Time 1.034 (0.000)	Data 0.002 (0.000)	
2021-06-21 17:28:59,921 Epoch: [4][3699/4426]	Eit 21400  lr 0.0005  Le 106.0639 (91.7315)	Time 1.009 (0.000)	Data 0.002 (0.000)	
2021-06-21 17:32:07,569 Epoch: [4][3899/4426]	Eit 21600  lr 0.0005  Le 61.1220 (91.6851)	Time 0.808 (0.000)	Data 0.002 (0.000)	
2021-06-21 17:35:16,588 Epoch: [4][4099/4426]	Eit 21800  lr 0.0005  Le 121.0512 (91.6257)	Time 1.137 (0.000)	Data 0.002 (0.000)	
2021-06-21 17:38:25,525 Epoch: [4][4299/4426]	Eit 22000  lr 0.0005  Le 156.2776 (91.7471)	Time 1.065 (0.000)	Data 0.004 (0.000)	
2021-06-21 17:40:33,098 Test: [0/40]	Le 274.7731 (274.7729)	Time 7.782 (0.000)	
2021-06-21 17:40:48,645 calculate similarity time: 0.0649571418762207
2021-06-21 17:40:49,182 Image to text: 64.0, 92.0, 97.5, 1.0, 3.0
2021-06-21 17:40:49,511 Text to image: 53.9, 86.7, 95.1, 1.0, 4.0
2021-06-21 17:40:49,511 Current rsum is 489.14
2021-06-21 17:40:53,096 runs/coco_butd_region_bert/log
2021-06-21 17:40:53,096 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-21 17:40:53,099 image encoder trainable parameters: 20490144
2021-06-21 17:40:53,108 txt encoder trainable parameters: 137319072
2021-06-21 17:42:13,564 Epoch: [5][74/4426]	Eit 22200  lr 0.0005  Le 97.9517 (83.5557)	Time 1.136 (0.000)	Data 0.003 (0.000)	
2021-06-21 17:45:21,912 Epoch: [5][274/4426]	Eit 22400  lr 0.0005  Le 65.7208 (83.7694)	Time 0.925 (0.000)	Data 0.002 (0.000)	
2021-06-21 17:48:31,881 Epoch: [5][474/4426]	Eit 22600  lr 0.0005  Le 59.6629 (84.9147)	Time 1.129 (0.000)	Data 0.002 (0.000)	
2021-06-21 17:51:41,105 Epoch: [5][674/4426]	Eit 22800  lr 0.0005  Le 86.8686 (84.5832)	Time 0.956 (0.000)	Data 0.003 (0.000)	
2021-06-21 17:54:51,015 Epoch: [5][874/4426]	Eit 23000  lr 0.0005  Le 86.2392 (85.1485)	Time 0.803 (0.000)	Data 0.002 (0.000)	
2021-06-21 17:58:02,618 Epoch: [5][1074/4426]	Eit 23200  lr 0.0005  Le 71.9063 (85.4274)	Time 1.028 (0.000)	Data 0.002 (0.000)	
2021-06-21 18:01:12,138 Epoch: [5][1274/4426]	Eit 23400  lr 0.0005  Le 61.2348 (85.1250)	Time 0.886 (0.000)	Data 0.002 (0.000)	
2021-06-21 18:04:21,907 Epoch: [5][1474/4426]	Eit 23600  lr 0.0005  Le 95.4927 (85.3498)	Time 1.072 (0.000)	Data 0.002 (0.000)	
2021-06-21 18:07:31,795 Epoch: [5][1674/4426]	Eit 23800  lr 0.0005  Le 129.0106 (85.3244)	Time 1.091 (0.000)	Data 0.002 (0.000)	
2021-06-21 18:10:38,905 Epoch: [5][1874/4426]	Eit 24000  lr 0.0005  Le 126.2440 (85.2755)	Time 1.175 (0.000)	Data 0.002 (0.000)	
2021-06-21 18:13:47,536 Epoch: [5][2074/4426]	Eit 24200  lr 0.0005  Le 67.8419 (85.4625)	Time 0.884 (0.000)	Data 0.002 (0.000)	
2021-06-21 18:16:56,505 Epoch: [5][2274/4426]	Eit 24400  lr 0.0005  Le 83.7161 (85.7009)	Time 0.834 (0.000)	Data 0.003 (0.000)	
2021-06-21 18:20:08,125 Epoch: [5][2474/4426]	Eit 24600  lr 0.0005  Le 71.4487 (85.5890)	Time 0.741 (0.000)	Data 0.002 (0.000)	
2021-06-21 18:23:16,535 Epoch: [5][2674/4426]	Eit 24800  lr 0.0005  Le 89.3924 (85.4728)	Time 0.846 (0.000)	Data 0.002 (0.000)	
2021-06-21 18:26:24,844 Epoch: [5][2874/4426]	Eit 25000  lr 0.0005  Le 70.5098 (85.4517)	Time 1.062 (0.000)	Data 0.002 (0.000)	
2021-06-21 18:29:33,075 Epoch: [5][3074/4426]	Eit 25200  lr 0.0005  Le 104.8141 (85.5172)	Time 0.786 (0.000)	Data 0.017 (0.000)	
2021-06-21 18:32:40,584 Epoch: [5][3274/4426]	Eit 25400  lr 0.0005  Le 55.4024 (85.3067)	Time 0.800 (0.000)	Data 0.002 (0.000)	
2021-06-21 18:35:40,396 Epoch: [5][3474/4426]	Eit 25600  lr 0.0005  Le 78.4266 (85.4335)	Time 0.848 (0.000)	Data 0.002 (0.000)	
2021-06-21 18:38:49,921 Epoch: [5][3674/4426]	Eit 25800  lr 0.0005  Le 73.6247 (85.3500)	Time 1.023 (0.000)	Data 0.002 (0.000)	
2021-06-21 18:41:58,682 Epoch: [5][3874/4426]	Eit 26000  lr 0.0005  Le 72.3973 (85.1488)	Time 1.083 (0.000)	Data 0.002 (0.000)	
2021-06-21 18:45:08,975 Epoch: [5][4074/4426]	Eit 26200  lr 0.0005  Le 78.9364 (85.2115)	Time 0.916 (0.000)	Data 0.002 (0.000)	
2021-06-21 18:48:17,185 Epoch: [5][4274/4426]	Eit 26400  lr 0.0005  Le 77.3560 (85.1613)	Time 0.935 (0.000)	Data 0.002 (0.000)	
2021-06-21 18:50:50,405 Test: [0/40]	Le 273.5222 (273.5220)	Time 7.991 (0.000)	
2021-06-21 18:51:04,866 calculate similarity time: 0.07090473175048828
2021-06-21 18:51:05,371 Image to text: 66.8, 93.0, 97.2, 1.0, 2.9
2021-06-21 18:51:05,691 Text to image: 55.3, 87.5, 95.4, 1.0, 3.5
2021-06-21 18:51:05,691 Current rsum is 495.16
2021-06-21 18:51:09,139 runs/coco_butd_region_bert/log
2021-06-21 18:51:09,139 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-21 18:51:09,142 image encoder trainable parameters: 20490144
2021-06-21 18:51:09,152 txt encoder trainable parameters: 137319072
2021-06-21 18:52:03,820 Epoch: [6][49/4426]	Eit 26600  lr 0.0005  Le 88.9773 (79.6500)	Time 0.975 (0.000)	Data 0.002 (0.000)	
2021-06-21 18:55:14,716 Epoch: [6][249/4426]	Eit 26800  lr 0.0005  Le 64.0010 (80.0234)	Time 0.921 (0.000)	Data 0.002 (0.000)	
2021-06-21 18:58:23,197 Epoch: [6][449/4426]	Eit 27000  lr 0.0005  Le 43.0826 (81.2147)	Time 0.817 (0.000)	Data 0.002 (0.000)	
2021-06-21 19:01:33,143 Epoch: [6][649/4426]	Eit 27200  lr 0.0005  Le 90.2586 (80.5879)	Time 0.906 (0.000)	Data 0.002 (0.000)	
2021-06-21 19:04:40,694 Epoch: [6][849/4426]	Eit 27400  lr 0.0005  Le 141.0219 (81.0164)	Time 0.921 (0.000)	Data 0.003 (0.000)	
2021-06-21 19:07:48,911 Epoch: [6][1049/4426]	Eit 27600  lr 0.0005  Le 121.6234 (81.3053)	Time 0.925 (0.000)	Data 0.002 (0.000)	
2021-06-21 19:10:57,482 Epoch: [6][1249/4426]	Eit 27800  lr 0.0005  Le 51.0467 (80.7069)	Time 0.891 (0.000)	Data 0.002 (0.000)	
2021-06-21 19:14:07,477 Epoch: [6][1449/4426]	Eit 28000  lr 0.0005  Le 106.2192 (80.9182)	Time 0.940 (0.000)	Data 0.002 (0.000)	
2021-06-21 19:17:15,209 Epoch: [6][1649/4426]	Eit 28200  lr 0.0005  Le 51.1086 (80.8320)	Time 1.058 (0.000)	Data 0.002 (0.000)	
2021-06-21 19:20:23,115 Epoch: [6][1849/4426]	Eit 28400  lr 0.0005  Le 108.3498 (80.7832)	Time 0.939 (0.000)	Data 0.002 (0.000)	
2021-06-21 19:23:30,815 Epoch: [6][2049/4426]	Eit 28600  lr 0.0005  Le 68.5523 (80.4020)	Time 1.067 (0.000)	Data 0.002 (0.000)	
2021-06-21 19:26:39,763 Epoch: [6][2249/4426]	Eit 28800  lr 0.0005  Le 40.5616 (80.5149)	Time 1.172 (0.000)	Data 0.002 (0.000)	
2021-06-21 19:29:50,366 Epoch: [6][2449/4426]	Eit 29000  lr 0.0005  Le 77.5270 (80.4399)	Time 0.953 (0.000)	Data 0.002 (0.000)	
2021-06-21 19:33:00,113 Epoch: [6][2649/4426]	Eit 29200  lr 0.0005  Le 61.1787 (80.3214)	Time 0.868 (0.000)	Data 0.003 (0.000)	
2021-06-21 19:36:09,501 Epoch: [6][2849/4426]	Eit 29400  lr 0.0005  Le 68.7529 (80.2712)	Time 1.025 (0.000)	Data 0.002 (0.000)	
2021-06-21 19:39:19,387 Epoch: [6][3049/4426]	Eit 29600  lr 0.0005  Le 87.3206 (80.2366)	Time 0.841 (0.000)	Data 0.002 (0.000)	
2021-06-21 19:42:29,849 Epoch: [6][3249/4426]	Eit 29800  lr 0.0005  Le 98.6325 (80.1878)	Time 0.939 (0.000)	Data 0.002 (0.000)	
2021-06-21 19:45:41,142 Epoch: [6][3449/4426]	Eit 30000  lr 0.0005  Le 66.9606 (80.3155)	Time 1.142 (0.000)	Data 0.002 (0.000)	
2021-06-21 19:48:51,980 Epoch: [6][3649/4426]	Eit 30200  lr 0.0005  Le 102.8907 (80.5513)	Time 0.887 (0.000)	Data 0.002 (0.000)	
2021-06-21 19:52:00,628 Epoch: [6][3849/4426]	Eit 30400  lr 0.0005  Le 66.0357 (80.5607)	Time 1.149 (0.000)	Data 0.002 (0.000)	
2021-06-21 19:55:09,478 Epoch: [6][4049/4426]	Eit 30600  lr 0.0005  Le 66.7281 (80.5801)	Time 0.721 (0.000)	Data 0.002 (0.000)	
2021-06-21 19:58:19,254 Epoch: [6][4249/4426]	Eit 30800  lr 0.0005  Le 134.2290 (80.6242)	Time 1.074 (0.000)	Data 0.002 (0.000)	
2021-06-21 20:01:13,982 Test: [0/40]	Le 272.6459 (272.6457)	Time 7.496 (0.000)	
2021-06-21 20:01:29,761 calculate similarity time: 0.08271527290344238
2021-06-21 20:01:30,202 Image to text: 66.8, 92.3, 97.4, 1.0, 3.0
2021-06-21 20:01:30,650 Text to image: 56.4, 88.7, 95.8, 1.0, 3.6
2021-06-21 20:01:30,650 Current rsum is 497.4
2021-06-21 20:01:34,146 runs/coco_butd_region_bert/log
2021-06-21 20:01:34,147 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-21 20:01:34,150 image encoder trainable parameters: 20490144
2021-06-21 20:01:34,158 txt encoder trainable parameters: 137319072
2021-06-21 20:02:05,906 Epoch: [7][24/4426]	Eit 31000  lr 0.0005  Le 60.6585 (72.6221)	Time 1.119 (0.000)	Data 0.002 (0.000)	
2021-06-21 20:05:13,190 Epoch: [7][224/4426]	Eit 31200  lr 0.0005  Le 61.5538 (77.0339)	Time 0.812 (0.000)	Data 0.002 (0.000)	
2021-06-21 20:08:23,304 Epoch: [7][424/4426]	Eit 31400  lr 0.0005  Le 80.5830 (76.1535)	Time 1.004 (0.000)	Data 0.002 (0.000)	
2021-06-21 20:11:33,223 Epoch: [7][624/4426]	Eit 31600  lr 0.0005  Le 61.6866 (76.4485)	Time 1.007 (0.000)	Data 0.002 (0.000)	
2021-06-21 20:14:41,173 Epoch: [7][824/4426]	Eit 31800  lr 0.0005  Le 138.9062 (75.9647)	Time 0.735 (0.000)	Data 0.002 (0.000)	
2021-06-21 20:17:45,442 Epoch: [7][1024/4426]	Eit 32000  lr 0.0005  Le 93.5421 (76.3968)	Time 0.900 (0.000)	Data 0.002 (0.000)	
2021-06-21 20:20:49,884 Epoch: [7][1224/4426]	Eit 32200  lr 0.0005  Le 117.0682 (76.0281)	Time 1.173 (0.000)	Data 0.002 (0.000)	
2021-06-21 20:24:00,251 Epoch: [7][1424/4426]	Eit 32400  lr 0.0005  Le 63.1222 (76.0928)	Time 1.052 (0.000)	Data 0.002 (0.000)	
2021-06-21 20:27:08,994 Epoch: [7][1624/4426]	Eit 32600  lr 0.0005  Le 64.1890 (76.2050)	Time 0.870 (0.000)	Data 0.002 (0.000)	
2021-06-21 20:30:18,217 Epoch: [7][1824/4426]	Eit 32800  lr 0.0005  Le 76.5296 (76.2554)	Time 1.052 (0.000)	Data 0.002 (0.000)	
2021-06-21 20:33:27,289 Epoch: [7][2024/4426]	Eit 33000  lr 0.0005  Le 41.8402 (76.5746)	Time 0.866 (0.000)	Data 0.002 (0.000)	
2021-06-21 20:36:39,007 Epoch: [7][2224/4426]	Eit 33200  lr 0.0005  Le 43.9587 (76.8751)	Time 0.936 (0.000)	Data 0.002 (0.000)	
2021-06-21 20:39:48,911 Epoch: [7][2424/4426]	Eit 33400  lr 0.0005  Le 82.2125 (76.8494)	Time 0.886 (0.000)	Data 0.002 (0.000)	
2021-06-21 20:42:58,260 Epoch: [7][2624/4426]	Eit 33600  lr 0.0005  Le 95.6635 (77.3218)	Time 0.922 (0.000)	Data 0.002 (0.000)	
2021-06-21 20:46:06,451 Epoch: [7][2824/4426]	Eit 33800  lr 0.0005  Le 87.0498 (77.2153)	Time 1.021 (0.000)	Data 0.002 (0.000)	
2021-06-21 20:49:16,939 Epoch: [7][3024/4426]	Eit 34000  lr 0.0005  Le 87.6494 (77.3039)	Time 0.801 (0.000)	Data 0.002 (0.000)	
2021-06-21 20:52:26,511 Epoch: [7][3224/4426]	Eit 34200  lr 0.0005  Le 65.2046 (77.1838)	Time 0.872 (0.000)	Data 0.002 (0.000)	
2021-06-21 20:55:34,814 Epoch: [7][3424/4426]	Eit 34400  lr 0.0005  Le 46.6306 (77.0503)	Time 1.047 (0.000)	Data 0.002 (0.000)	
2021-06-21 20:58:44,685 Epoch: [7][3624/4426]	Eit 34600  lr 0.0005  Le 83.6604 (77.0870)	Time 0.771 (0.000)	Data 0.002 (0.000)	
2021-06-21 21:01:55,429 Epoch: [7][3824/4426]	Eit 34800  lr 0.0005  Le 84.8356 (77.2733)	Time 1.072 (0.000)	Data 0.002 (0.000)	
2021-06-21 21:05:06,295 Epoch: [7][4024/4426]	Eit 35000  lr 0.0005  Le 101.6799 (77.3261)	Time 1.049 (0.000)	Data 0.002 (0.000)	
2021-06-21 21:08:14,670 Epoch: [7][4224/4426]	Eit 35200  lr 0.0005  Le 83.9971 (77.2676)	Time 0.849 (0.000)	Data 0.017 (0.000)	
2021-06-21 21:11:22,072 Epoch: [7][4424/4426]	Eit 35400  lr 0.0005  Le 51.4252 (77.3184)	Time 0.890 (0.000)	Data 0.002 (0.000)	
2021-06-21 21:11:30,970 Test: [0/40]	Le 278.5592 (278.5590)	Time 7.574 (0.000)	
2021-06-21 21:11:45,835 calculate similarity time: 0.053781747817993164
2021-06-21 21:11:46,261 Image to text: 68.0, 92.7, 97.6, 1.0, 2.8
2021-06-21 21:11:46,571 Text to image: 56.8, 88.3, 95.8, 1.0, 3.7
2021-06-21 21:11:46,572 Current rsum is 499.2199999999999
2021-06-21 21:11:50,035 runs/coco_butd_region_bert/log
2021-06-21 21:11:50,036 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-21 21:11:50,039 image encoder trainable parameters: 20490144
2021-06-21 21:11:50,050 txt encoder trainable parameters: 137319072
2021-06-21 21:15:04,277 Epoch: [8][199/4426]	Eit 35600  lr 0.0005  Le 67.9246 (71.0624)	Time 1.073 (0.000)	Data 0.002 (0.000)	
2021-06-21 21:18:14,031 Epoch: [8][399/4426]	Eit 35800  lr 0.0005  Le 77.5857 (73.2887)	Time 0.855 (0.000)	Data 0.002 (0.000)	
2021-06-21 21:21:23,524 Epoch: [8][599/4426]	Eit 36000  lr 0.0005  Le 71.4009 (72.9974)	Time 1.193 (0.000)	Data 0.002 (0.000)	
2021-06-21 21:24:34,365 Epoch: [8][799/4426]	Eit 36200  lr 0.0005  Le 73.0697 (73.0444)	Time 0.803 (0.000)	Data 0.003 (0.000)	
2021-06-21 21:27:43,204 Epoch: [8][999/4426]	Eit 36400  lr 0.0005  Le 108.4496 (73.1909)	Time 0.985 (0.000)	Data 0.002 (0.000)	
2021-06-21 21:30:50,845 Epoch: [8][1199/4426]	Eit 36600  lr 0.0005  Le 70.8963 (73.6052)	Time 0.897 (0.000)	Data 0.002 (0.000)	
2021-06-21 21:34:00,525 Epoch: [8][1399/4426]	Eit 36800  lr 0.0005  Le 79.9822 (73.6537)	Time 0.795 (0.000)	Data 0.002 (0.000)	
2021-06-21 21:37:10,996 Epoch: [8][1599/4426]	Eit 37000  lr 0.0005  Le 51.2704 (73.5268)	Time 1.094 (0.000)	Data 0.004 (0.000)	
2021-06-21 21:40:21,925 Epoch: [8][1799/4426]	Eit 37200  lr 0.0005  Le 55.3878 (73.8686)	Time 0.903 (0.000)	Data 0.002 (0.000)	
2021-06-21 21:43:31,445 Epoch: [8][1999/4426]	Eit 37400  lr 0.0005  Le 49.2677 (74.0864)	Time 1.327 (0.000)	Data 0.004 (0.000)	
2021-06-21 21:46:39,996 Epoch: [8][2199/4426]	Eit 37600  lr 0.0005  Le 113.7564 (74.1398)	Time 0.868 (0.000)	Data 0.002 (0.000)	
2021-06-21 21:49:49,734 Epoch: [8][2399/4426]	Eit 37800  lr 0.0005  Le 59.9964 (73.9644)	Time 0.788 (0.000)	Data 0.002 (0.000)	
2021-06-21 21:52:59,888 Epoch: [8][2599/4426]	Eit 38000  lr 0.0005  Le 63.9603 (73.9815)	Time 0.891 (0.000)	Data 0.002 (0.000)	
2021-06-21 21:56:13,724 Epoch: [8][2799/4426]	Eit 38200  lr 0.0005  Le 68.1059 (74.2364)	Time 0.935 (0.000)	Data 0.003 (0.000)	
2021-06-21 21:59:23,025 Epoch: [8][2999/4426]	Eit 38400  lr 0.0005  Le 63.5869 (74.2307)	Time 0.849 (0.000)	Data 0.002 (0.000)	
2021-06-21 22:02:22,502 Epoch: [8][3199/4426]	Eit 38600  lr 0.0005  Le 58.8387 (74.1626)	Time 0.905 (0.000)	Data 0.002 (0.000)	
2021-06-21 22:05:34,707 Epoch: [8][3399/4426]	Eit 38800  lr 0.0005  Le 53.0536 (74.1287)	Time 0.833 (0.000)	Data 0.003 (0.000)	
2021-06-21 22:08:45,743 Epoch: [8][3599/4426]	Eit 39000  lr 0.0005  Le 52.7748 (74.2053)	Time 0.952 (0.000)	Data 0.011 (0.000)	
2021-06-21 22:11:55,033 Epoch: [8][3799/4426]	Eit 39200  lr 0.0005  Le 68.2307 (74.1874)	Time 1.040 (0.000)	Data 0.002 (0.000)	
2021-06-21 22:15:04,799 Epoch: [8][3999/4426]	Eit 39400  lr 0.0005  Le 103.9757 (74.1401)	Time 0.758 (0.000)	Data 0.002 (0.000)	
2021-06-21 22:18:15,671 Epoch: [8][4199/4426]	Eit 39600  lr 0.0005  Le 99.9242 (74.1171)	Time 0.967 (0.000)	Data 0.002 (0.000)	
2021-06-21 22:21:26,055 Epoch: [8][4399/4426]	Eit 39800  lr 0.0005  Le 74.7067 (74.1164)	Time 1.152 (0.000)	Data 0.004 (0.000)	
2021-06-21 22:21:58,970 Test: [0/40]	Le 265.3356 (265.3354)	Time 7.948 (0.000)	
2021-06-21 22:22:14,167 calculate similarity time: 0.06045699119567871
2021-06-21 22:22:14,579 Image to text: 68.4, 93.5, 97.5, 1.0, 2.5
2021-06-21 22:22:14,954 Text to image: 56.8, 88.8, 96.1, 1.0, 3.4
2021-06-21 22:22:14,955 Current rsum is 501.08
2021-06-21 22:22:18,922 runs/coco_butd_region_bert/log
2021-06-21 22:22:18,922 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-21 22:22:18,925 image encoder trainable parameters: 20490144
2021-06-21 22:22:18,936 txt encoder trainable parameters: 137319072
2021-06-21 22:25:12,035 Epoch: [9][174/4426]	Eit 40000  lr 0.0005  Le 68.7010 (70.4948)	Time 1.174 (0.000)	Data 0.003 (0.000)	
2021-06-21 22:28:20,826 Epoch: [9][374/4426]	Eit 40200  lr 0.0005  Le 69.3412 (70.4531)	Time 0.825 (0.000)	Data 0.003 (0.000)	
2021-06-21 22:31:33,642 Epoch: [9][574/4426]	Eit 40400  lr 0.0005  Le 66.1286 (70.4038)	Time 1.033 (0.000)	Data 0.002 (0.000)	
2021-06-21 22:34:44,963 Epoch: [9][774/4426]	Eit 40600  lr 0.0005  Le 109.6403 (70.5265)	Time 1.110 (0.000)	Data 0.002 (0.000)	
2021-06-21 22:37:55,836 Epoch: [9][974/4426]	Eit 40800  lr 0.0005  Le 57.9161 (70.2620)	Time 0.904 (0.000)	Data 0.002 (0.000)	
2021-06-21 22:41:04,704 Epoch: [9][1174/4426]	Eit 41000  lr 0.0005  Le 70.8226 (70.8907)	Time 1.027 (0.000)	Data 0.002 (0.000)	
2021-06-21 22:44:15,091 Epoch: [9][1374/4426]	Eit 41200  lr 0.0005  Le 81.7522 (71.1719)	Time 0.822 (0.000)	Data 0.003 (0.000)	
2021-06-21 22:47:25,237 Epoch: [9][1574/4426]	Eit 41400  lr 0.0005  Le 72.6525 (71.1456)	Time 0.894 (0.000)	Data 0.002 (0.000)	
2021-06-21 22:50:34,541 Epoch: [9][1774/4426]	Eit 41600  lr 0.0005  Le 65.2953 (71.0968)	Time 0.905 (0.000)	Data 0.002 (0.000)	
2021-06-21 22:53:44,921 Epoch: [9][1974/4426]	Eit 41800  lr 0.0005  Le 64.9321 (71.1102)	Time 0.825 (0.000)	Data 0.005 (0.000)	
2021-06-21 22:56:54,960 Epoch: [9][2174/4426]	Eit 42000  lr 0.0005  Le 65.7232 (71.1225)	Time 0.931 (0.000)	Data 0.002 (0.000)	
2021-06-21 23:00:05,501 Epoch: [9][2374/4426]	Eit 42200  lr 0.0005  Le 66.9897 (71.2616)	Time 1.296 (0.000)	Data 0.002 (0.000)	
2021-06-21 23:03:17,242 Epoch: [9][2574/4426]	Eit 42400  lr 0.0005  Le 46.2211 (71.2078)	Time 0.854 (0.000)	Data 0.002 (0.000)	
2021-06-21 23:06:28,146 Epoch: [9][2774/4426]	Eit 42600  lr 0.0005  Le 69.0574 (71.1559)	Time 0.891 (0.000)	Data 0.003 (0.000)	
2021-06-21 23:09:39,121 Epoch: [9][2974/4426]	Eit 42800  lr 0.0005  Le 53.6689 (71.1415)	Time 1.142 (0.000)	Data 0.002 (0.000)	
2021-06-21 23:12:49,368 Epoch: [9][3174/4426]	Eit 43000  lr 0.0005  Le 45.2807 (71.2931)	Time 1.272 (0.000)	Data 0.003 (0.000)	
2021-06-21 23:15:58,786 Epoch: [9][3374/4426]	Eit 43200  lr 0.0005  Le 55.6399 (71.4559)	Time 0.989 (0.000)	Data 0.002 (0.000)	
2021-06-21 23:19:08,874 Epoch: [9][3574/4426]	Eit 43400  lr 0.0005  Le 74.5940 (71.3569)	Time 0.796 (0.000)	Data 0.003 (0.000)	
2021-06-21 23:22:19,424 Epoch: [9][3774/4426]	Eit 43600  lr 0.0005  Le 49.2758 (71.3215)	Time 0.817 (0.000)	Data 0.003 (0.000)	
2021-06-21 23:25:28,897 Epoch: [9][3974/4426]	Eit 43800  lr 0.0005  Le 53.5745 (71.2522)	Time 1.011 (0.000)	Data 0.002 (0.000)	
2021-06-21 23:28:36,882 Epoch: [9][4174/4426]	Eit 44000  lr 0.0005  Le 78.2344 (71.3785)	Time 0.978 (0.000)	Data 0.002 (0.000)	
2021-06-21 23:31:45,852 Epoch: [9][4374/4426]	Eit 44200  lr 0.0005  Le 46.3120 (71.4492)	Time 0.932 (0.000)	Data 0.003 (0.000)	
2021-06-21 23:32:41,341 Test: [0/40]	Le 259.3271 (259.3269)	Time 7.881 (0.000)	
2021-06-21 23:32:56,405 calculate similarity time: 0.07367444038391113
2021-06-21 23:32:56,857 Image to text: 70.4, 93.0, 97.6, 1.0, 2.9
2021-06-21 23:32:57,205 Text to image: 57.5, 88.9, 96.0, 1.0, 3.5
2021-06-21 23:32:57,205 Current rsum is 503.32
2021-06-21 23:33:01,145 runs/coco_butd_region_bert/log
2021-06-21 23:33:01,145 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-21 23:33:01,147 image encoder trainable parameters: 20490144
2021-06-21 23:33:01,156 txt encoder trainable parameters: 137319072
2021-06-21 23:35:32,997 Epoch: [10][149/4426]	Eit 44400  lr 0.0005  Le 46.2130 (66.4102)	Time 0.978 (0.000)	Data 0.002 (0.000)	
2021-06-21 23:38:45,153 Epoch: [10][349/4426]	Eit 44600  lr 0.0005  Le 60.3394 (66.2618)	Time 0.742 (0.000)	Data 0.002 (0.000)	
2021-06-21 23:41:55,554 Epoch: [10][549/4426]	Eit 44800  lr 0.0005  Le 103.6135 (67.2531)	Time 0.882 (0.000)	Data 0.002 (0.000)	
2021-06-21 23:44:51,219 Epoch: [10][749/4426]	Eit 45000  lr 0.0005  Le 65.7547 (67.7723)	Time 1.043 (0.000)	Data 0.002 (0.000)	
2021-06-21 23:47:59,917 Epoch: [10][949/4426]	Eit 45200  lr 0.0005  Le 71.9150 (68.4139)	Time 0.946 (0.000)	Data 0.002 (0.000)	
2021-06-21 23:51:09,170 Epoch: [10][1149/4426]	Eit 45400  lr 0.0005  Le 117.4370 (68.4288)	Time 0.729 (0.000)	Data 0.002 (0.000)	
2021-06-21 23:54:20,494 Epoch: [10][1349/4426]	Eit 45600  lr 0.0005  Le 58.9704 (68.8809)	Time 0.816 (0.000)	Data 0.002 (0.000)	
2021-06-21 23:57:30,179 Epoch: [10][1549/4426]	Eit 45800  lr 0.0005  Le 46.6064 (68.8866)	Time 0.887 (0.000)	Data 0.003 (0.000)	
2021-06-22 00:00:40,257 Epoch: [10][1749/4426]	Eit 46000  lr 0.0005  Le 81.3562 (69.2175)	Time 0.916 (0.000)	Data 0.003 (0.000)	
2021-06-22 00:03:50,154 Epoch: [10][1949/4426]	Eit 46200  lr 0.0005  Le 134.3957 (69.2107)	Time 1.153 (0.000)	Data 0.003 (0.000)	
2021-06-22 00:07:00,047 Epoch: [10][2149/4426]	Eit 46400  lr 0.0005  Le 70.0728 (69.2395)	Time 1.011 (0.000)	Data 0.002 (0.000)	
2021-06-22 00:10:11,026 Epoch: [10][2349/4426]	Eit 46600  lr 0.0005  Le 51.8084 (69.2097)	Time 1.091 (0.000)	Data 0.003 (0.000)	
2021-06-22 00:13:22,505 Epoch: [10][2549/4426]	Eit 46800  lr 0.0005  Le 53.6632 (69.0739)	Time 1.090 (0.000)	Data 0.002 (0.000)	
2021-06-22 00:16:32,813 Epoch: [10][2749/4426]	Eit 47000  lr 0.0005  Le 60.2040 (69.1529)	Time 0.895 (0.000)	Data 0.002 (0.000)	
2021-06-22 00:19:45,170 Epoch: [10][2949/4426]	Eit 47200  lr 0.0005  Le 84.6527 (69.2521)	Time 0.920 (0.000)	Data 0.002 (0.000)	
2021-06-22 00:22:57,718 Epoch: [10][3149/4426]	Eit 47400  lr 0.0005  Le 68.3714 (69.1482)	Time 1.096 (0.000)	Data 0.002 (0.000)	
2021-06-22 00:26:09,599 Epoch: [10][3349/4426]	Eit 47600  lr 0.0005  Le 59.7987 (69.1599)	Time 1.031 (0.000)	Data 0.002 (0.000)	
2021-06-22 00:29:20,585 Epoch: [10][3549/4426]	Eit 47800  lr 0.0005  Le 90.9366 (69.1199)	Time 0.985 (0.000)	Data 0.002 (0.000)	
2021-06-22 00:32:30,922 Epoch: [10][3749/4426]	Eit 48000  lr 0.0005  Le 54.1632 (69.1006)	Time 0.835 (0.000)	Data 0.002 (0.000)	
2021-06-22 00:35:40,429 Epoch: [10][3949/4426]	Eit 48200  lr 0.0005  Le 52.9268 (69.1190)	Time 0.917 (0.000)	Data 0.002 (0.000)	
2021-06-22 00:38:48,700 Epoch: [10][4149/4426]	Eit 48400  lr 0.0005  Le 47.3175 (69.0508)	Time 0.855 (0.000)	Data 0.002 (0.000)	
2021-06-22 00:41:58,712 Epoch: [10][4349/4426]	Eit 48600  lr 0.0005  Le 61.1072 (69.1450)	Time 1.008 (0.000)	Data 0.002 (0.000)	
2021-06-22 00:43:18,856 Test: [0/40]	Le 265.9095 (265.9093)	Time 8.199 (0.000)	
2021-06-22 00:43:33,926 calculate similarity time: 0.06974434852600098
2021-06-22 00:43:34,460 Image to text: 68.0, 92.8, 97.6, 1.0, 2.7
2021-06-22 00:43:34,915 Text to image: 57.0, 89.2, 96.0, 1.0, 3.4
2021-06-22 00:43:34,916 Current rsum is 500.62
2021-06-22 00:43:36,997 runs/coco_butd_region_bert/log
2021-06-22 00:43:36,997 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-22 00:43:36,999 image encoder trainable parameters: 20490144
2021-06-22 00:43:37,006 txt encoder trainable parameters: 137319072
2021-06-22 00:45:44,587 Epoch: [11][124/4426]	Eit 48800  lr 0.0005  Le 64.2420 (66.5548)	Time 1.128 (0.000)	Data 0.003 (0.000)	
2021-06-22 00:48:53,709 Epoch: [11][324/4426]	Eit 49000  lr 0.0005  Le 62.9779 (66.1728)	Time 0.768 (0.000)	Data 0.002 (0.000)	
2021-06-22 00:52:05,963 Epoch: [11][524/4426]	Eit 49200  lr 0.0005  Le 74.5473 (66.3885)	Time 0.795 (0.000)	Data 0.003 (0.000)	
2021-06-22 00:55:15,231 Epoch: [11][724/4426]	Eit 49400  lr 0.0005  Le 63.2414 (66.4639)	Time 0.866 (0.000)	Data 0.002 (0.000)	
2021-06-22 00:58:25,534 Epoch: [11][924/4426]	Eit 49600  lr 0.0005  Le 54.2294 (65.7248)	Time 0.738 (0.000)	Data 0.002 (0.000)	
2021-06-22 01:01:34,325 Epoch: [11][1124/4426]	Eit 49800  lr 0.0005  Le 52.3203 (66.1050)	Time 0.883 (0.000)	Data 0.002 (0.000)	
2021-06-22 01:04:43,902 Epoch: [11][1324/4426]	Eit 50000  lr 0.0005  Le 75.8547 (66.3773)	Time 0.830 (0.000)	Data 0.002 (0.000)	
2021-06-22 01:07:52,883 Epoch: [11][1524/4426]	Eit 50200  lr 0.0005  Le 45.0124 (66.4308)	Time 1.024 (0.000)	Data 0.002 (0.000)	
2021-06-22 01:11:03,579 Epoch: [11][1724/4426]	Eit 50400  lr 0.0005  Le 48.7409 (66.7168)	Time 1.041 (0.000)	Data 0.003 (0.000)	
2021-06-22 01:14:13,134 Epoch: [11][1924/4426]	Eit 50600  lr 0.0005  Le 56.3870 (66.6775)	Time 0.731 (0.000)	Data 0.003 (0.000)	
2021-06-22 01:17:22,899 Epoch: [11][2124/4426]	Eit 50800  lr 0.0005  Le 62.0589 (66.7639)	Time 1.125 (0.000)	Data 0.003 (0.000)	
2021-06-22 01:20:32,756 Epoch: [11][2324/4426]	Eit 51000  lr 0.0005  Le 55.1679 (66.7086)	Time 0.822 (0.000)	Data 0.002 (0.000)	
2021-06-22 01:23:43,096 Epoch: [11][2524/4426]	Eit 51200  lr 0.0005  Le 43.2216 (66.5683)	Time 0.955 (0.000)	Data 0.002 (0.000)	
2021-06-22 01:26:51,305 Epoch: [11][2724/4426]	Eit 51400  lr 0.0005  Le 42.3707 (66.4624)	Time 0.766 (0.000)	Data 0.002 (0.000)	
2021-06-22 01:29:50,634 Epoch: [11][2924/4426]	Eit 51600  lr 0.0005  Le 51.7935 (66.6081)	Time 1.036 (0.000)	Data 0.002 (0.000)	
2021-06-22 01:32:57,529 Epoch: [11][3124/4426]	Eit 51800  lr 0.0005  Le 58.0543 (66.7215)	Time 0.905 (0.000)	Data 0.002 (0.000)	
2021-06-22 01:36:05,867 Epoch: [11][3324/4426]	Eit 52000  lr 0.0005  Le 71.7028 (66.7463)	Time 0.901 (0.000)	Data 0.002 (0.000)	
2021-06-22 01:39:17,719 Epoch: [11][3524/4426]	Eit 52200  lr 0.0005  Le 67.9283 (66.7899)	Time 0.958 (0.000)	Data 0.002 (0.000)	
2021-06-22 01:42:30,353 Epoch: [11][3724/4426]	Eit 52400  lr 0.0005  Le 56.1964 (66.8316)	Time 1.167 (0.000)	Data 0.002 (0.000)	
2021-06-22 01:45:39,527 Epoch: [11][3924/4426]	Eit 52600  lr 0.0005  Le 89.1124 (66.7806)	Time 1.122 (0.000)	Data 0.002 (0.000)	
2021-06-22 01:48:48,168 Epoch: [11][4124/4426]	Eit 52800  lr 0.0005  Le 75.1251 (66.7419)	Time 0.774 (0.000)	Data 0.002 (0.000)	
2021-06-22 01:52:00,440 Epoch: [11][4324/4426]	Eit 53000  lr 0.0005  Le 36.6141 (66.7335)	Time 0.926 (0.000)	Data 0.002 (0.000)	
2021-06-22 01:53:45,533 Test: [0/40]	Le 265.6987 (265.6985)	Time 8.335 (0.000)	
2021-06-22 01:54:00,851 calculate similarity time: 0.07077217102050781
2021-06-22 01:54:01,373 Image to text: 69.5, 94.0, 98.0, 1.0, 2.6
2021-06-22 01:54:01,822 Text to image: 58.0, 89.5, 96.0, 1.0, 3.4
2021-06-22 01:54:01,822 Current rsum is 504.96
2021-06-22 01:54:05,592 runs/coco_butd_region_bert/log
2021-06-22 01:54:05,593 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-22 01:54:05,595 image encoder trainable parameters: 20490144
2021-06-22 01:54:05,602 txt encoder trainable parameters: 137319072
2021-06-22 01:55:47,860 Epoch: [12][99/4426]	Eit 53200  lr 0.0005  Le 59.6485 (62.5665)	Time 0.807 (0.000)	Data 0.005 (0.000)	
2021-06-22 01:58:55,236 Epoch: [12][299/4426]	Eit 53400  lr 0.0005  Le 34.0682 (62.9957)	Time 0.746 (0.000)	Data 0.002 (0.000)	
2021-06-22 02:02:04,130 Epoch: [12][499/4426]	Eit 53600  lr 0.0005  Le 58.8291 (64.0303)	Time 1.104 (0.000)	Data 0.002 (0.000)	
2021-06-22 02:05:12,176 Epoch: [12][699/4426]	Eit 53800  lr 0.0005  Le 61.1003 (64.3202)	Time 1.145 (0.000)	Data 0.002 (0.000)	
2021-06-22 02:08:21,424 Epoch: [12][899/4426]	Eit 54000  lr 0.0005  Le 61.3071 (64.5923)	Time 0.990 (0.000)	Data 0.002 (0.000)	
2021-06-22 02:11:30,560 Epoch: [12][1099/4426]	Eit 54200  lr 0.0005  Le 48.4656 (64.7967)	Time 0.897 (0.000)	Data 0.002 (0.000)	
2021-06-22 02:14:38,746 Epoch: [12][1299/4426]	Eit 54400  lr 0.0005  Le 57.8399 (64.9542)	Time 1.012 (0.000)	Data 0.002 (0.000)	
2021-06-22 02:17:46,655 Epoch: [12][1499/4426]	Eit 54600  lr 0.0005  Le 53.5836 (65.0572)	Time 1.040 (0.000)	Data 0.002 (0.000)	
2021-06-22 02:20:54,532 Epoch: [12][1699/4426]	Eit 54800  lr 0.0005  Le 43.9582 (65.0459)	Time 1.026 (0.000)	Data 0.002 (0.000)	
2021-06-22 02:24:04,329 Epoch: [12][1899/4426]	Eit 55000  lr 0.0005  Le 69.3078 (65.0693)	Time 1.011 (0.000)	Data 0.002 (0.000)	
2021-06-22 02:27:10,898 Epoch: [12][2099/4426]	Eit 55200  lr 0.0005  Le 117.0896 (65.1018)	Time 1.090 (0.000)	Data 0.002 (0.000)	
2021-06-22 02:30:19,031 Epoch: [12][2299/4426]	Eit 55400  lr 0.0005  Le 40.5662 (65.2704)	Time 0.774 (0.000)	Data 0.002 (0.000)	
2021-06-22 02:33:28,424 Epoch: [12][2499/4426]	Eit 55600  lr 0.0005  Le 95.3831 (65.3499)	Time 0.891 (0.000)	Data 0.002 (0.000)	
2021-06-22 02:36:37,419 Epoch: [12][2699/4426]	Eit 55800  lr 0.0005  Le 37.9988 (65.5706)	Time 0.824 (0.000)	Data 0.002 (0.000)	
2021-06-22 02:39:47,919 Epoch: [12][2899/4426]	Eit 56000  lr 0.0005  Le 56.4542 (65.5301)	Time 0.972 (0.000)	Data 0.002 (0.000)	
2021-06-22 02:42:58,001 Epoch: [12][3099/4426]	Eit 56200  lr 0.0005  Le 36.3557 (65.3930)	Time 0.850 (0.000)	Data 0.003 (0.000)	
2021-06-22 02:46:07,006 Epoch: [12][3299/4426]	Eit 56400  lr 0.0005  Le 50.6754 (65.4481)	Time 0.899 (0.000)	Data 0.002 (0.000)	
2021-06-22 02:49:14,093 Epoch: [12][3499/4426]	Eit 56600  lr 0.0005  Le 55.7876 (65.4676)	Time 0.935 (0.000)	Data 0.002 (0.000)	
2021-06-22 02:52:23,806 Epoch: [12][3699/4426]	Eit 56800  lr 0.0005  Le 151.1027 (65.4794)	Time 1.077 (0.000)	Data 0.002 (0.000)	
2021-06-22 02:55:34,274 Epoch: [12][3899/4426]	Eit 57000  lr 0.0005  Le 79.4760 (65.5429)	Time 0.881 (0.000)	Data 0.002 (0.000)	
2021-06-22 02:58:44,675 Epoch: [12][4099/4426]	Eit 57200  lr 0.0005  Le 85.1998 (65.7490)	Time 0.837 (0.000)	Data 0.002 (0.000)	
2021-06-22 03:01:53,865 Epoch: [12][4299/4426]	Eit 57400  lr 0.0005  Le 69.0905 (65.7991)	Time 0.992 (0.000)	Data 0.002 (0.000)	
2021-06-22 03:04:00,689 Test: [0/40]	Le 267.5183 (267.5181)	Time 7.720 (0.000)	
2021-06-22 03:04:15,543 calculate similarity time: 0.07502031326293945
2021-06-22 03:04:16,070 Image to text: 69.9, 93.9, 97.9, 1.0, 2.5
2021-06-22 03:04:16,381 Text to image: 58.1, 89.7, 96.2, 1.0, 3.5
2021-06-22 03:04:16,381 Current rsum is 505.70000000000005
2021-06-22 03:04:20,065 runs/coco_butd_region_bert/log
2021-06-22 03:04:20,066 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-22 03:04:20,069 image encoder trainable parameters: 20490144
2021-06-22 03:04:20,079 txt encoder trainable parameters: 137319072
2021-06-22 03:05:39,924 Epoch: [13][74/4426]	Eit 57600  lr 0.0005  Le 43.8536 (63.4182)	Time 1.215 (0.000)	Data 0.002 (0.000)	
2021-06-22 03:08:48,439 Epoch: [13][274/4426]	Eit 57800  lr 0.0005  Le 53.7677 (62.6755)	Time 1.126 (0.000)	Data 0.002 (0.000)	
2021-06-22 03:11:45,905 Epoch: [13][474/4426]	Eit 58000  lr 0.0005  Le 81.8570 (63.2003)	Time 1.326 (0.000)	Data 0.002 (0.000)	
2021-06-22 03:14:53,093 Epoch: [13][674/4426]	Eit 58200  lr 0.0005  Le 36.2565 (62.7895)	Time 1.039 (0.000)	Data 0.002 (0.000)	
2021-06-22 03:18:00,463 Epoch: [13][874/4426]	Eit 58400  lr 0.0005  Le 36.4127 (62.4433)	Time 0.840 (0.000)	Data 0.002 (0.000)	
2021-06-22 03:21:08,417 Epoch: [13][1074/4426]	Eit 58600  lr 0.0005  Le 51.0453 (62.6546)	Time 0.885 (0.000)	Data 0.002 (0.000)	
2021-06-22 03:24:18,740 Epoch: [13][1274/4426]	Eit 58800  lr 0.0005  Le 41.5738 (62.5065)	Time 0.872 (0.000)	Data 0.003 (0.000)	
2021-06-22 03:27:28,318 Epoch: [13][1474/4426]	Eit 59000  lr 0.0005  Le 51.4863 (62.8634)	Time 0.997 (0.000)	Data 0.002 (0.000)	
2021-06-22 03:30:38,233 Epoch: [13][1674/4426]	Eit 59200  lr 0.0005  Le 64.3025 (62.7610)	Time 0.981 (0.000)	Data 0.002 (0.000)	
2021-06-22 03:33:48,215 Epoch: [13][1874/4426]	Eit 59400  lr 0.0005  Le 48.4077 (62.8771)	Time 1.075 (0.000)	Data 0.002 (0.000)	
2021-06-22 03:36:56,282 Epoch: [13][2074/4426]	Eit 59600  lr 0.0005  Le 35.3872 (62.9798)	Time 0.895 (0.000)	Data 0.002 (0.000)	
2021-06-22 03:40:03,988 Epoch: [13][2274/4426]	Eit 59800  lr 0.0005  Le 77.4476 (63.2219)	Time 0.854 (0.000)	Data 0.002 (0.000)	
2021-06-22 03:43:16,246 Epoch: [13][2474/4426]	Eit 60000  lr 0.0005  Le 77.5993 (63.4160)	Time 1.036 (0.000)	Data 0.002 (0.000)	
2021-06-22 03:46:25,814 Epoch: [13][2674/4426]	Eit 60200  lr 0.0005  Le 50.2399 (63.4362)	Time 0.835 (0.000)	Data 0.002 (0.000)	
2021-06-22 03:49:36,546 Epoch: [13][2874/4426]	Eit 60400  lr 0.0005  Le 57.9049 (63.5775)	Time 0.989 (0.000)	Data 0.002 (0.000)	
2021-06-22 03:52:45,772 Epoch: [13][3074/4426]	Eit 60600  lr 0.0005  Le 44.3955 (63.6281)	Time 1.204 (0.000)	Data 0.002 (0.000)	
2021-06-22 03:55:54,883 Epoch: [13][3274/4426]	Eit 60800  lr 0.0005  Le 51.7552 (63.6967)	Time 0.838 (0.000)	Data 0.002 (0.000)	
2021-06-22 03:59:02,878 Epoch: [13][3474/4426]	Eit 61000  lr 0.0005  Le 31.5371 (63.6165)	Time 0.844 (0.000)	Data 0.002 (0.000)	
2021-06-22 04:02:12,629 Epoch: [13][3674/4426]	Eit 61200  lr 0.0005  Le 34.3257 (63.6545)	Time 1.106 (0.000)	Data 0.002 (0.000)	
2021-06-22 04:05:20,317 Epoch: [13][3874/4426]	Eit 61400  lr 0.0005  Le 43.6946 (63.6217)	Time 0.837 (0.000)	Data 0.002 (0.000)	
2021-06-22 04:08:29,781 Epoch: [13][4074/4426]	Eit 61600  lr 0.0005  Le 48.8488 (63.6815)	Time 0.937 (0.000)	Data 0.002 (0.000)	
2021-06-22 04:11:39,674 Epoch: [13][4274/4426]	Eit 61800  lr 0.0005  Le 50.5223 (63.5705)	Time 0.987 (0.000)	Data 0.002 (0.000)	
2021-06-22 04:14:09,665 Test: [0/40]	Le 260.5487 (260.5485)	Time 7.831 (0.000)	
2021-06-22 04:14:24,496 calculate similarity time: 0.0729672908782959
2021-06-22 04:14:24,928 Image to text: 70.0, 93.3, 98.1, 1.0, 2.4
2021-06-22 04:14:25,238 Text to image: 58.5, 89.9, 96.4, 1.0, 3.2
2021-06-22 04:14:25,238 Current rsum is 506.14
2021-06-22 04:14:28,886 runs/coco_butd_region_bert/log
2021-06-22 04:14:28,886 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-22 04:14:28,888 image encoder trainable parameters: 20490144
2021-06-22 04:14:28,896 txt encoder trainable parameters: 137319072
2021-06-22 04:15:24,873 Epoch: [14][49/4426]	Eit 62000  lr 0.0005  Le 55.9285 (63.2996)	Time 0.782 (0.000)	Data 0.002 (0.000)	
2021-06-22 04:18:33,015 Epoch: [14][249/4426]	Eit 62200  lr 0.0005  Le 42.9309 (61.1138)	Time 0.940 (0.000)	Data 0.002 (0.000)	
2021-06-22 04:21:41,777 Epoch: [14][449/4426]	Eit 62400  lr 0.0005  Le 37.3539 (61.5744)	Time 1.131 (0.000)	Data 0.002 (0.000)	
2021-06-22 04:24:51,360 Epoch: [14][649/4426]	Eit 62600  lr 0.0005  Le 103.0603 (61.0866)	Time 1.082 (0.000)	Data 0.002 (0.000)	
2021-06-22 04:28:01,446 Epoch: [14][849/4426]	Eit 62800  lr 0.0005  Le 44.1909 (60.7616)	Time 0.689 (0.000)	Data 0.002 (0.000)	
2021-06-22 04:31:10,791 Epoch: [14][1049/4426]	Eit 63000  lr 0.0005  Le 76.6076 (60.8996)	Time 1.089 (0.000)	Data 0.002 (0.000)	
2021-06-22 04:34:21,028 Epoch: [14][1249/4426]	Eit 63200  lr 0.0005  Le 34.4450 (61.2182)	Time 0.782 (0.000)	Data 0.002 (0.000)	
2021-06-22 04:37:30,063 Epoch: [14][1449/4426]	Eit 63400  lr 0.0005  Le 72.6682 (61.1638)	Time 0.833 (0.000)	Data 0.002 (0.000)	
2021-06-22 04:40:39,547 Epoch: [14][1649/4426]	Eit 63600  lr 0.0005  Le 51.1852 (61.1167)	Time 0.846 (0.000)	Data 0.003 (0.000)	
2021-06-22 04:43:49,015 Epoch: [14][1849/4426]	Eit 63800  lr 0.0005  Le 35.7842 (61.1132)	Time 0.861 (0.000)	Data 0.002 (0.000)	
2021-06-22 04:46:57,287 Epoch: [14][2049/4426]	Eit 64000  lr 0.0005  Le 51.4435 (61.4742)	Time 0.955 (0.000)	Data 0.002 (0.000)	
2021-06-22 04:50:07,624 Epoch: [14][2249/4426]	Eit 64200  lr 0.0005  Le 106.1853 (61.6398)	Time 0.927 (0.000)	Data 0.002 (0.000)	
2021-06-22 04:53:12,969 Epoch: [14][2449/4426]	Eit 64400  lr 0.0005  Le 61.1170 (61.6209)	Time 0.594 (0.000)	Data 0.002 (0.000)	
2021-06-22 04:56:13,784 Epoch: [14][2649/4426]	Eit 64600  lr 0.0005  Le 86.2926 (61.6693)	Time 0.867 (0.000)	Data 0.002 (0.000)	
2021-06-22 04:59:22,790 Epoch: [14][2849/4426]	Eit 64800  lr 0.0005  Le 50.4941 (61.8029)	Time 1.080 (0.000)	Data 0.002 (0.000)	
2021-06-22 05:02:33,011 Epoch: [14][3049/4426]	Eit 65000  lr 0.0005  Le 64.0131 (61.7969)	Time 0.797 (0.000)	Data 0.002 (0.000)	
2021-06-22 05:05:43,151 Epoch: [14][3249/4426]	Eit 65200  lr 0.0005  Le 45.8464 (61.7464)	Time 0.918 (0.000)	Data 0.002 (0.000)	
2021-06-22 05:08:51,767 Epoch: [14][3449/4426]	Eit 65400  lr 0.0005  Le 52.7473 (61.7507)	Time 0.730 (0.000)	Data 0.002 (0.000)	
2021-06-22 05:12:00,772 Epoch: [14][3649/4426]	Eit 65600  lr 0.0005  Le 79.0198 (61.7842)	Time 0.929 (0.000)	Data 0.002 (0.000)	
2021-06-22 05:15:10,423 Epoch: [14][3849/4426]	Eit 65800  lr 0.0005  Le 46.2963 (62.0288)	Time 1.131 (0.000)	Data 0.002 (0.000)	
2021-06-22 05:18:20,381 Epoch: [14][4049/4426]	Eit 66000  lr 0.0005  Le 56.7238 (62.0619)	Time 0.991 (0.000)	Data 0.002 (0.000)	
2021-06-22 05:21:29,553 Epoch: [14][4249/4426]	Eit 66200  lr 0.0005  Le 62.5824 (62.0205)	Time 1.016 (0.000)	Data 0.002 (0.000)	
2021-06-22 05:24:23,972 Test: [0/40]	Le 259.8291 (259.8289)	Time 8.032 (0.000)	
2021-06-22 05:24:38,820 calculate similarity time: 0.06581950187683105
2021-06-22 05:24:39,248 Image to text: 70.5, 93.2, 97.5, 1.0, 3.2
2021-06-22 05:24:39,597 Text to image: 58.8, 89.5, 96.0, 1.0, 3.4
2021-06-22 05:24:39,597 Current rsum is 505.48
2021-06-22 05:24:41,341 runs/coco_butd_region_bert/log
2021-06-22 05:24:41,341 runs/coco_butd_region_bert
2021-06-22 05:24:41,341 Current epoch num is 15, decrease all lr by 10
2021-06-22 05:24:41,341 new lr 5e-05
2021-06-22 05:24:41,341 new lr 5e-06
2021-06-22 05:24:41,341 new lr 5e-05
Use VSE++ objective.
2021-06-22 05:24:41,343 image encoder trainable parameters: 20490144
2021-06-22 05:24:41,348 txt encoder trainable parameters: 137319072
2021-06-22 05:25:13,153 Epoch: [15][24/4426]	Eit 66400  lr 5e-05  Le 45.4967 (64.9263)	Time 1.085 (0.000)	Data 0.002 (0.000)	
2021-06-22 05:28:22,271 Epoch: [15][224/4426]	Eit 66600  lr 5e-05  Le 43.2170 (58.8135)	Time 0.989 (0.000)	Data 0.002 (0.000)	
2021-06-22 05:31:31,317 Epoch: [15][424/4426]	Eit 66800  lr 5e-05  Le 80.0449 (58.7729)	Time 0.926 (0.000)	Data 0.002 (0.000)	
2021-06-22 05:34:40,307 Epoch: [15][624/4426]	Eit 67000  lr 5e-05  Le 83.9605 (57.6085)	Time 0.935 (0.000)	Data 0.002 (0.000)	
2021-06-22 05:37:49,410 Epoch: [15][824/4426]	Eit 67200  lr 5e-05  Le 54.8131 (56.8080)	Time 0.824 (0.000)	Data 0.002 (0.000)	
2021-06-22 05:40:59,732 Epoch: [15][1024/4426]	Eit 67400  lr 5e-05  Le 65.3865 (56.4614)	Time 0.804 (0.000)	Data 0.002 (0.000)	
2021-06-22 05:44:11,133 Epoch: [15][1224/4426]	Eit 67600  lr 5e-05  Le 52.1709 (56.0538)	Time 0.827 (0.000)	Data 0.002 (0.000)	
2021-06-22 05:47:19,915 Epoch: [15][1424/4426]	Eit 67800  lr 5e-05  Le 64.7401 (55.9518)	Time 0.817 (0.000)	Data 0.002 (0.000)	
2021-06-22 05:50:29,294 Epoch: [15][1624/4426]	Eit 68000  lr 5e-05  Le 41.2924 (55.6349)	Time 1.045 (0.000)	Data 0.002 (0.000)	
2021-06-22 05:53:38,679 Epoch: [15][1824/4426]	Eit 68200  lr 5e-05  Le 41.7713 (55.5624)	Time 0.959 (0.000)	Data 0.002 (0.000)	
2021-06-22 05:56:47,317 Epoch: [15][2024/4426]	Eit 68400  lr 5e-05  Le 41.8819 (55.6041)	Time 1.121 (0.000)	Data 0.002 (0.000)	
2021-06-22 05:59:55,706 Epoch: [15][2224/4426]	Eit 68600  lr 5e-05  Le 37.2485 (55.3147)	Time 1.174 (0.000)	Data 0.002 (0.000)	
2021-06-22 06:03:03,878 Epoch: [15][2424/4426]	Eit 68800  lr 5e-05  Le 44.5962 (55.0870)	Time 1.033 (0.000)	Data 0.002 (0.000)	
2021-06-22 06:06:11,747 Epoch: [15][2624/4426]	Eit 69000  lr 5e-05  Le 83.2431 (54.9405)	Time 0.939 (0.000)	Data 0.002 (0.000)	
2021-06-22 06:09:23,250 Epoch: [15][2824/4426]	Eit 69200  lr 5e-05  Le 37.3668 (54.7574)	Time 1.053 (0.000)	Data 0.002 (0.000)	
2021-06-22 06:12:31,478 Epoch: [15][3024/4426]	Eit 69400  lr 5e-05  Le 75.4841 (54.5852)	Time 0.845 (0.000)	Data 0.002 (0.000)	
2021-06-22 06:15:42,661 Epoch: [15][3224/4426]	Eit 69600  lr 5e-05  Le 52.2065 (54.5293)	Time 0.929 (0.000)	Data 0.002 (0.000)	
2021-06-22 06:18:52,608 Epoch: [15][3424/4426]	Eit 69800  lr 5e-05  Le 53.0931 (54.3693)	Time 0.967 (0.000)	Data 0.002 (0.000)	
2021-06-22 06:22:00,413 Epoch: [15][3624/4426]	Eit 70000  lr 5e-05  Le 68.5054 (54.2322)	Time 1.155 (0.000)	Data 0.002 (0.000)	
2021-06-22 06:25:08,374 Epoch: [15][3824/4426]	Eit 70200  lr 5e-05  Le 50.1823 (54.2472)	Time 0.953 (0.000)	Data 0.003 (0.000)	
2021-06-22 06:28:16,485 Epoch: [15][4024/4426]	Eit 70400  lr 5e-05  Le 72.5827 (54.2318)	Time 0.831 (0.000)	Data 0.002 (0.000)	
2021-06-22 06:31:26,417 Epoch: [15][4224/4426]	Eit 70600  lr 5e-05  Le 50.2372 (54.1379)	Time 0.959 (0.000)	Data 0.002 (0.000)	
2021-06-22 06:34:34,060 Epoch: [15][4424/4426]	Eit 70800  lr 5e-05  Le 48.1668 (54.0826)	Time 0.738 (0.000)	Data 0.002 (0.000)	
2021-06-22 06:34:43,537 Test: [0/40]	Le 255.1167 (255.1165)	Time 8.078 (0.000)	
2021-06-22 06:34:58,634 calculate similarity time: 0.06778192520141602
2021-06-22 06:34:59,146 Image to text: 71.6, 94.3, 97.7, 1.0, 2.9
2021-06-22 06:34:59,485 Text to image: 60.4, 90.5, 96.6, 1.0, 3.3
2021-06-22 06:34:59,485 Current rsum is 511.08
2021-06-22 06:35:03,244 runs/coco_butd_region_bert/log
2021-06-22 06:35:03,244 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-22 06:35:03,247 image encoder trainable parameters: 20490144
2021-06-22 06:35:03,257 txt encoder trainable parameters: 137319072
2021-06-22 06:38:06,657 Epoch: [16][199/4426]	Eit 71000  lr 5e-05  Le 42.8160 (51.9373)	Time 1.000 (0.000)	Data 0.002 (0.000)	
2021-06-22 06:41:15,950 Epoch: [16][399/4426]	Eit 71200  lr 5e-05  Le 46.2761 (51.9180)	Time 0.848 (0.000)	Data 0.002 (0.000)	
2021-06-22 06:44:26,578 Epoch: [16][599/4426]	Eit 71400  lr 5e-05  Le 53.8856 (51.7568)	Time 0.753 (0.000)	Data 0.002 (0.000)	
2021-06-22 06:47:36,764 Epoch: [16][799/4426]	Eit 71600  lr 5e-05  Le 54.7564 (51.7982)	Time 1.074 (0.000)	Data 0.002 (0.000)	
2021-06-22 06:50:45,902 Epoch: [16][999/4426]	Eit 71800  lr 5e-05  Le 116.7760 (51.5957)	Time 1.038 (0.000)	Data 0.002 (0.000)	
2021-06-22 06:53:55,177 Epoch: [16][1199/4426]	Eit 72000  lr 5e-05  Le 88.5210 (51.6656)	Time 0.901 (0.000)	Data 0.002 (0.000)	
2021-06-22 06:57:04,413 Epoch: [16][1399/4426]	Eit 72200  lr 5e-05  Le 51.9651 (51.4124)	Time 0.939 (0.000)	Data 0.002 (0.000)	
2021-06-22 07:00:12,245 Epoch: [16][1599/4426]	Eit 72400  lr 5e-05  Le 60.6679 (51.4968)	Time 0.835 (0.000)	Data 0.002 (0.000)	
2021-06-22 07:03:19,863 Epoch: [16][1799/4426]	Eit 72600  lr 5e-05  Le 31.6858 (51.4313)	Time 0.763 (0.000)	Data 0.003 (0.000)	
2021-06-22 07:06:26,070 Epoch: [16][1999/4426]	Eit 72800  lr 5e-05  Le 83.6651 (51.1955)	Time 0.996 (0.000)	Data 0.002 (0.000)	
2021-06-22 07:09:33,815 Epoch: [16][2199/4426]	Eit 73000  lr 5e-05  Le 48.7311 (51.1839)	Time 0.834 (0.000)	Data 0.002 (0.000)	
2021-06-22 07:12:44,362 Epoch: [16][2399/4426]	Eit 73200  lr 5e-05  Le 24.5723 (51.1336)	Time 0.795 (0.000)	Data 0.002 (0.000)	
2021-06-22 07:15:57,845 Epoch: [16][2599/4426]	Eit 73400  lr 5e-05  Le 25.5590 (51.1150)	Time 0.899 (0.000)	Data 0.002 (0.000)	
2021-06-22 07:19:06,858 Epoch: [16][2799/4426]	Eit 73600  lr 5e-05  Le 32.6376 (51.0277)	Time 0.847 (0.000)	Data 0.002 (0.000)	
2021-06-22 07:22:16,044 Epoch: [16][2999/4426]	Eit 73800  lr 5e-05  Le 36.2352 (51.0092)	Time 0.965 (0.000)	Data 0.002 (0.000)	
2021-06-22 07:25:25,505 Epoch: [16][3199/4426]	Eit 74000  lr 5e-05  Le 36.8460 (51.0479)	Time 1.226 (0.000)	Data 0.002 (0.000)	
2021-06-22 07:28:33,387 Epoch: [16][3399/4426]	Eit 74200  lr 5e-05  Le 28.5466 (50.9360)	Time 0.818 (0.000)	Data 0.004 (0.000)	
2021-06-22 07:31:44,291 Epoch: [16][3599/4426]	Eit 74400  lr 5e-05  Le 96.2955 (50.9529)	Time 0.906 (0.000)	Data 0.002 (0.000)	
2021-06-22 07:34:54,889 Epoch: [16][3799/4426]	Eit 74600  lr 5e-05  Le 32.8207 (50.9998)	Time 0.880 (0.000)	Data 0.002 (0.000)	
2021-06-22 07:38:03,440 Epoch: [16][3999/4426]	Eit 74800  lr 5e-05  Le 33.5834 (50.8596)	Time 0.959 (0.000)	Data 0.002 (0.000)	
2021-06-22 07:41:13,388 Epoch: [16][4199/4426]	Eit 75000  lr 5e-05  Le 43.1045 (50.9799)	Time 1.031 (0.000)	Data 0.002 (0.000)	
2021-06-22 07:44:23,238 Epoch: [16][4399/4426]	Eit 75200  lr 5e-05  Le 70.2563 (50.9306)	Time 1.018 (0.000)	Data 0.003 (0.000)	
2021-06-22 07:44:56,142 Test: [0/40]	Le 253.2316 (253.2314)	Time 7.625 (0.000)	
2021-06-22 07:45:11,149 calculate similarity time: 0.06602144241333008
2021-06-22 07:45:11,639 Image to text: 71.3, 94.2, 97.9, 1.0, 2.8
2021-06-22 07:45:11,952 Text to image: 60.3, 90.7, 96.6, 1.0, 3.3
2021-06-22 07:45:11,952 Current rsum is 510.9
2021-06-22 07:45:13,497 runs/coco_butd_region_bert/log
2021-06-22 07:45:13,497 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-22 07:45:13,499 image encoder trainable parameters: 20490144
2021-06-22 07:45:13,504 txt encoder trainable parameters: 137319072
2021-06-22 07:48:05,301 Epoch: [17][174/4426]	Eit 75400  lr 5e-05  Le 41.4947 (49.3347)	Time 1.052 (0.000)	Data 0.002 (0.000)	
2021-06-22 07:51:13,979 Epoch: [17][374/4426]	Eit 75600  lr 5e-05  Le 51.1996 (49.8604)	Time 1.020 (0.000)	Data 0.002 (0.000)	
2021-06-22 07:54:26,132 Epoch: [17][574/4426]	Eit 75800  lr 5e-05  Le 29.2477 (49.8070)	Time 0.950 (0.000)	Data 0.002 (0.000)	
2021-06-22 07:57:33,039 Epoch: [17][774/4426]	Eit 76000  lr 5e-05  Le 24.0500 (49.7333)	Time 0.882 (0.000)	Data 0.002 (0.000)	
2021-06-22 08:00:43,077 Epoch: [17][974/4426]	Eit 76200  lr 5e-05  Le 33.9413 (50.0270)	Time 1.030 (0.000)	Data 0.002 (0.000)	
2021-06-22 08:03:52,151 Epoch: [17][1174/4426]	Eit 76400  lr 5e-05  Le 59.3990 (49.9512)	Time 0.859 (0.000)	Data 0.002 (0.000)	
2021-06-22 08:07:02,390 Epoch: [17][1374/4426]	Eit 76600  lr 5e-05  Le 33.4690 (49.8056)	Time 0.817 (0.000)	Data 0.002 (0.000)	
2021-06-22 08:10:11,985 Epoch: [17][1574/4426]	Eit 76800  lr 5e-05  Le 59.5367 (49.7148)	Time 0.869 (0.000)	Data 0.002 (0.000)	
2021-06-22 08:13:19,773 Epoch: [17][1774/4426]	Eit 77000  lr 5e-05  Le 48.3302 (49.7514)	Time 0.910 (0.000)	Data 0.002 (0.000)	
2021-06-22 08:16:28,198 Epoch: [17][1974/4426]	Eit 77200  lr 5e-05  Le 43.6527 (49.7376)	Time 0.795 (0.000)	Data 0.003 (0.000)	
2021-06-22 08:19:32,853 Epoch: [17][2174/4426]	Eit 77400  lr 5e-05  Le 38.7550 (49.6968)	Time 0.838 (0.000)	Data 0.002 (0.000)	
2021-06-22 08:22:35,260 Epoch: [17][2374/4426]	Eit 77600  lr 5e-05  Le 43.9595 (49.8802)	Time 0.878 (0.000)	Data 0.002 (0.000)	
2021-06-22 08:25:43,847 Epoch: [17][2574/4426]	Eit 77800  lr 5e-05  Le 39.1530 (49.9967)	Time 0.791 (0.000)	Data 0.002 (0.000)	
2021-06-22 08:28:52,834 Epoch: [17][2774/4426]	Eit 78000  lr 5e-05  Le 54.0046 (50.1363)	Time 1.263 (0.000)	Data 0.002 (0.000)	
2021-06-22 08:32:03,346 Epoch: [17][2974/4426]	Eit 78200  lr 5e-05  Le 31.4876 (50.0873)	Time 0.952 (0.000)	Data 0.002 (0.000)	
2021-06-22 08:35:13,787 Epoch: [17][3174/4426]	Eit 78400  lr 5e-05  Le 67.8285 (50.1100)	Time 0.956 (0.000)	Data 0.002 (0.000)	
2021-06-22 08:38:21,293 Epoch: [17][3374/4426]	Eit 78600  lr 5e-05  Le 63.9050 (49.9107)	Time 0.748 (0.000)	Data 0.002 (0.000)	
2021-06-22 08:41:30,193 Epoch: [17][3574/4426]	Eit 78800  lr 5e-05  Le 37.0025 (49.7446)	Time 1.022 (0.000)	Data 0.002 (0.000)	
2021-06-22 08:44:39,261 Epoch: [17][3774/4426]	Eit 79000  lr 5e-05  Le 58.1517 (49.8207)	Time 1.005 (0.000)	Data 0.002 (0.000)	
2021-06-22 08:47:47,812 Epoch: [17][3974/4426]	Eit 79200  lr 5e-05  Le 44.8801 (49.7670)	Time 1.112 (0.000)	Data 0.002 (0.000)	
2021-06-22 08:50:55,888 Epoch: [17][4174/4426]	Eit 79400  lr 5e-05  Le 35.0440 (49.6737)	Time 0.948 (0.000)	Data 0.002 (0.000)	
2021-06-22 08:54:06,308 Epoch: [17][4374/4426]	Eit 79600  lr 5e-05  Le 72.7877 (49.7253)	Time 0.980 (0.000)	Data 0.002 (0.000)	
2021-06-22 08:55:01,320 Test: [0/40]	Le 253.9631 (253.9629)	Time 7.829 (0.000)	
2021-06-22 08:55:16,241 calculate similarity time: 0.07686305046081543
2021-06-22 08:55:16,769 Image to text: 71.3, 94.4, 98.0, 1.0, 2.9
2021-06-22 08:55:17,205 Text to image: 60.4, 90.7, 96.4, 1.0, 3.4
2021-06-22 08:55:17,205 Current rsum is 511.21999999999997
2021-06-22 08:55:20,762 runs/coco_butd_region_bert/log
2021-06-22 08:55:20,762 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-22 08:55:20,764 image encoder trainable parameters: 20490144
2021-06-22 08:55:20,773 txt encoder trainable parameters: 137319072
2021-06-22 08:57:50,282 Epoch: [18][149/4426]	Eit 79800  lr 5e-05  Le 44.9311 (48.5791)	Time 0.903 (0.000)	Data 0.002 (0.000)	
2021-06-22 09:00:59,390 Epoch: [18][349/4426]	Eit 80000  lr 5e-05  Le 53.9820 (49.6958)	Time 1.060 (0.000)	Data 0.002 (0.000)	
2021-06-22 09:04:07,083 Epoch: [18][549/4426]	Eit 80200  lr 5e-05  Le 66.0238 (49.0722)	Time 0.967 (0.000)	Data 0.002 (0.000)	
2021-06-22 09:07:15,431 Epoch: [18][749/4426]	Eit 80400  lr 5e-05  Le 38.2764 (49.4603)	Time 0.894 (0.000)	Data 0.002 (0.000)	
2021-06-22 09:10:26,181 Epoch: [18][949/4426]	Eit 80600  lr 5e-05  Le 48.3649 (49.8045)	Time 1.079 (0.000)	Data 0.002 (0.000)	
2021-06-22 09:13:34,408 Epoch: [18][1149/4426]	Eit 80800  lr 5e-05  Le 48.6777 (49.5012)	Time 1.067 (0.000)	Data 0.002 (0.000)	
2021-06-22 09:16:44,460 Epoch: [18][1349/4426]	Eit 81000  lr 5e-05  Le 38.0989 (49.3390)	Time 1.097 (0.000)	Data 0.002 (0.000)	
2021-06-22 09:19:54,673 Epoch: [18][1549/4426]	Eit 81200  lr 5e-05  Le 66.3391 (49.0569)	Time 0.870 (0.000)	Data 0.002 (0.000)	
2021-06-22 09:23:03,224 Epoch: [18][1749/4426]	Eit 81400  lr 5e-05  Le 51.3903 (49.1486)	Time 0.764 (0.000)	Data 0.002 (0.000)	
2021-06-22 09:26:11,772 Epoch: [18][1949/4426]	Eit 81600  lr 5e-05  Le 42.5443 (49.1590)	Time 1.025 (0.000)	Data 0.002 (0.000)	
2021-06-22 09:29:19,001 Epoch: [18][2149/4426]	Eit 81800  lr 5e-05  Le 65.8325 (49.0510)	Time 0.850 (0.000)	Data 0.003 (0.000)	
2021-06-22 09:32:29,192 Epoch: [18][2349/4426]	Eit 82000  lr 5e-05  Le 49.4817 (49.0109)	Time 0.716 (0.000)	Data 0.002 (0.000)	
2021-06-22 09:35:39,760 Epoch: [18][2549/4426]	Eit 82200  lr 5e-05  Le 33.2999 (49.1133)	Time 0.986 (0.000)	Data 0.002 (0.000)	
2021-06-22 09:38:48,486 Epoch: [18][2749/4426]	Eit 82400  lr 5e-05  Le 39.3436 (49.0653)	Time 0.948 (0.000)	Data 0.002 (0.000)	
2021-06-22 09:41:57,407 Epoch: [18][2949/4426]	Eit 82600  lr 5e-05  Le 73.7105 (48.8356)	Time 0.916 (0.000)	Data 0.002 (0.000)	
2021-06-22 09:45:08,370 Epoch: [18][3149/4426]	Eit 82800  lr 5e-05  Le 73.5072 (48.9761)	Time 1.000 (0.000)	Data 0.002 (0.000)	
2021-06-22 09:48:17,397 Epoch: [18][3349/4426]	Eit 83000  lr 5e-05  Le 42.1435 (48.8387)	Time 1.063 (0.000)	Data 0.002 (0.000)	
2021-06-22 09:51:25,760 Epoch: [18][3549/4426]	Eit 83200  lr 5e-05  Le 58.4940 (48.7491)	Time 0.989 (0.000)	Data 0.002 (0.000)	
2021-06-22 09:54:35,501 Epoch: [18][3749/4426]	Eit 83400  lr 5e-05  Le 40.3394 (48.7681)	Time 1.006 (0.000)	Data 0.002 (0.000)	
2021-06-22 09:57:45,015 Epoch: [18][3949/4426]	Eit 83600  lr 5e-05  Le 37.2373 (48.7497)	Time 0.871 (0.000)	Data 0.002 (0.000)	
2021-06-22 10:00:55,306 Epoch: [18][4149/4426]	Eit 83800  lr 5e-05  Le 72.5412 (48.7779)	Time 0.930 (0.000)	Data 0.004 (0.000)	
2021-06-22 10:03:53,315 Epoch: [18][4349/4426]	Eit 84000  lr 5e-05  Le 50.3485 (48.7730)	Time 0.953 (0.000)	Data 0.003 (0.000)	
2021-06-22 10:05:15,494 Test: [0/40]	Le 255.5466 (255.5464)	Time 8.932 (0.000)	
2021-06-22 10:05:30,719 calculate similarity time: 0.054038047790527344
2021-06-22 10:05:31,264 Image to text: 71.0, 94.6, 97.9, 1.0, 2.9
2021-06-22 10:05:31,608 Text to image: 60.7, 90.9, 96.6, 1.0, 3.3
2021-06-22 10:05:31,608 Current rsum is 511.58
2021-06-22 10:05:35,563 runs/coco_butd_region_bert/log
2021-06-22 10:05:35,564 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-22 10:05:35,566 image encoder trainable parameters: 20490144
2021-06-22 10:05:35,577 txt encoder trainable parameters: 137319072
2021-06-22 10:07:43,100 Epoch: [19][124/4426]	Eit 84200  lr 5e-05  Le 41.4113 (45.2261)	Time 0.736 (0.000)	Data 0.002 (0.000)	
2021-06-22 10:10:54,117 Epoch: [19][324/4426]	Eit 84400  lr 5e-05  Le 24.4390 (45.6505)	Time 0.926 (0.000)	Data 0.003 (0.000)	
2021-06-22 10:14:04,493 Epoch: [19][524/4426]	Eit 84600  lr 5e-05  Le 80.9447 (45.4752)	Time 0.857 (0.000)	Data 0.002 (0.000)	
2021-06-22 10:17:16,561 Epoch: [19][724/4426]	Eit 84800  lr 5e-05  Le 39.4685 (46.1856)	Time 0.820 (0.000)	Data 0.002 (0.000)	
2021-06-22 10:20:27,181 Epoch: [19][924/4426]	Eit 85000  lr 5e-05  Le 77.1907 (46.2860)	Time 0.857 (0.000)	Data 0.003 (0.000)	
2021-06-22 10:23:35,852 Epoch: [19][1124/4426]	Eit 85200  lr 5e-05  Le 46.3582 (46.3773)	Time 0.936 (0.000)	Data 0.002 (0.000)	
2021-06-22 10:26:43,963 Epoch: [19][1324/4426]	Eit 85400  lr 5e-05  Le 53.1361 (46.7966)	Time 0.921 (0.000)	Data 0.002 (0.000)	
2021-06-22 10:29:52,448 Epoch: [19][1524/4426]	Eit 85600  lr 5e-05  Le 56.8201 (47.4105)	Time 1.008 (0.000)	Data 0.002 (0.000)	
2021-06-22 10:33:02,532 Epoch: [19][1724/4426]	Eit 85800  lr 5e-05  Le 47.1627 (47.6279)	Time 1.068 (0.000)	Data 0.003 (0.000)	
2021-06-22 10:36:12,901 Epoch: [19][1924/4426]	Eit 86000  lr 5e-05  Le 51.4895 (47.4390)	Time 1.077 (0.000)	Data 0.002 (0.000)	
2021-06-22 10:39:24,444 Epoch: [19][2124/4426]	Eit 86200  lr 5e-05  Le 42.4757 (47.3256)	Time 0.997 (0.000)	Data 0.004 (0.000)	
2021-06-22 10:42:32,869 Epoch: [19][2324/4426]	Eit 86400  lr 5e-05  Le 37.9586 (47.1956)	Time 0.765 (0.000)	Data 0.002 (0.000)	
2021-06-22 10:45:41,930 Epoch: [19][2524/4426]	Eit 86600  lr 5e-05  Le 52.2164 (47.1347)	Time 1.190 (0.000)	Data 0.002 (0.000)	
2021-06-22 10:48:49,742 Epoch: [19][2724/4426]	Eit 86800  lr 5e-05  Le 37.0497 (46.9936)	Time 0.880 (0.000)	Data 0.002 (0.000)	
2021-06-22 10:52:00,679 Epoch: [19][2924/4426]	Eit 87000  lr 5e-05  Le 32.2543 (47.1562)	Time 0.741 (0.000)	Data 0.002 (0.000)	
2021-06-22 10:55:10,398 Epoch: [19][3124/4426]	Eit 87200  lr 5e-05  Le 49.3264 (47.1124)	Time 1.264 (0.000)	Data 0.002 (0.000)	
2021-06-22 10:58:18,184 Epoch: [19][3324/4426]	Eit 87400  lr 5e-05  Le 52.7458 (47.1800)	Time 0.859 (0.000)	Data 0.002 (0.000)	
2021-06-22 11:01:27,157 Epoch: [19][3524/4426]	Eit 87600  lr 5e-05  Le 30.1503 (47.2405)	Time 1.096 (0.000)	Data 0.002 (0.000)	
2021-06-22 11:04:36,648 Epoch: [19][3724/4426]	Eit 87800  lr 5e-05  Le 87.0821 (47.2383)	Time 1.004 (0.000)	Data 0.002 (0.000)	
2021-06-22 11:07:45,570 Epoch: [19][3924/4426]	Eit 88000  lr 5e-05  Le 31.7051 (47.2735)	Time 1.004 (0.000)	Data 0.002 (0.000)	
2021-06-22 11:10:55,468 Epoch: [19][4124/4426]	Eit 88200  lr 5e-05  Le 35.6164 (47.2629)	Time 1.110 (0.000)	Data 0.002 (0.000)	
2021-06-22 11:14:03,865 Epoch: [19][4324/4426]	Eit 88400  lr 5e-05  Le 61.2601 (47.2764)	Time 1.079 (0.000)	Data 0.002 (0.000)	
2021-06-22 11:15:47,863 Test: [0/40]	Le 255.8163 (255.8161)	Time 7.705 (0.000)	
2021-06-22 11:16:02,682 calculate similarity time: 0.07921767234802246
2021-06-22 11:16:03,230 Image to text: 71.6, 95.0, 98.1, 1.0, 2.7
2021-06-22 11:16:03,540 Text to image: 60.9, 90.8, 96.7, 1.0, 3.3
2021-06-22 11:16:03,540 Current rsum is 513.08
2021-06-22 11:16:07,071 runs/coco_butd_region_bert/log
2021-06-22 11:16:07,071 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-22 11:16:07,074 image encoder trainable parameters: 20490144
2021-06-22 11:16:07,082 txt encoder trainable parameters: 137319072
2021-06-22 11:17:50,021 Epoch: [20][99/4426]	Eit 88600  lr 5e-05  Le 38.6125 (45.6139)	Time 0.992 (0.000)	Data 0.002 (0.000)	
2021-06-22 11:21:00,346 Epoch: [20][299/4426]	Eit 88800  lr 5e-05  Le 75.0090 (44.6256)	Time 0.864 (0.000)	Data 0.002 (0.000)	
2021-06-22 11:24:08,656 Epoch: [20][499/4426]	Eit 89000  lr 5e-05  Le 47.0703 (46.1585)	Time 0.907 (0.000)	Data 0.002 (0.000)	
2021-06-22 11:27:19,017 Epoch: [20][699/4426]	Eit 89200  lr 5e-05  Le 70.4773 (46.5248)	Time 0.938 (0.000)	Data 0.002 (0.000)	
2021-06-22 11:30:28,622 Epoch: [20][899/4426]	Eit 89400  lr 5e-05  Le 53.1603 (46.9384)	Time 0.846 (0.000)	Data 0.003 (0.000)	
2021-06-22 11:33:39,038 Epoch: [20][1099/4426]	Eit 89600  lr 5e-05  Le 49.6938 (46.8781)	Time 1.023 (0.000)	Data 0.002 (0.000)	
2021-06-22 11:36:48,320 Epoch: [20][1299/4426]	Eit 89800  lr 5e-05  Le 43.1264 (46.7402)	Time 1.092 (0.000)	Data 0.003 (0.000)	
2021-06-22 11:39:56,286 Epoch: [20][1499/4426]	Eit 90000  lr 5e-05  Le 53.0220 (46.6371)	Time 0.742 (0.000)	Data 0.002 (0.000)	
2021-06-22 11:43:05,018 Epoch: [20][1699/4426]	Eit 90200  lr 5e-05  Le 34.5557 (46.5397)	Time 1.233 (0.000)	Data 0.002 (0.000)	
2021-06-22 11:46:02,286 Epoch: [20][1899/4426]	Eit 90400  lr 5e-05  Le 61.5330 (46.6655)	Time 0.896 (0.000)	Data 0.002 (0.000)	
2021-06-22 11:49:09,932 Epoch: [20][2099/4426]	Eit 90600  lr 5e-05  Le 56.7351 (46.8432)	Time 1.047 (0.000)	Data 0.002 (0.000)	
2021-06-22 11:52:20,169 Epoch: [20][2299/4426]	Eit 90800  lr 5e-05  Le 36.2427 (46.9518)	Time 1.026 (0.000)	Data 0.003 (0.000)	
2021-06-22 11:55:29,603 Epoch: [20][2499/4426]	Eit 91000  lr 5e-05  Le 27.3781 (46.9259)	Time 0.829 (0.000)	Data 0.002 (0.000)	
2021-06-22 11:58:39,273 Epoch: [20][2699/4426]	Eit 91200  lr 5e-05  Le 53.9187 (46.9172)	Time 0.914 (0.000)	Data 0.002 (0.000)	
2021-06-22 12:01:51,695 Epoch: [20][2899/4426]	Eit 91400  lr 5e-05  Le 31.7286 (47.0201)	Time 0.974 (0.000)	Data 0.002 (0.000)	
2021-06-22 12:05:00,905 Epoch: [20][3099/4426]	Eit 91600  lr 5e-05  Le 94.6004 (46.9650)	Time 0.940 (0.000)	Data 0.002 (0.000)	
2021-06-22 12:08:12,591 Epoch: [20][3299/4426]	Eit 91800  lr 5e-05  Le 46.3715 (47.1123)	Time 0.899 (0.000)	Data 0.002 (0.000)	
2021-06-22 12:11:21,413 Epoch: [20][3499/4426]	Eit 92000  lr 5e-05  Le 37.1892 (47.0682)	Time 1.158 (0.000)	Data 0.002 (0.000)	
2021-06-22 12:14:32,567 Epoch: [20][3699/4426]	Eit 92200  lr 5e-05  Le 37.8393 (47.0866)	Time 0.873 (0.000)	Data 0.002 (0.000)	
2021-06-22 12:17:41,660 Epoch: [20][3899/4426]	Eit 92400  lr 5e-05  Le 48.5995 (47.1236)	Time 0.895 (0.000)	Data 0.002 (0.000)	
2021-06-22 12:20:51,390 Epoch: [20][4099/4426]	Eit 92600  lr 5e-05  Le 72.5161 (47.1125)	Time 0.934 (0.000)	Data 0.002 (0.000)	
2021-06-22 12:24:01,399 Epoch: [20][4299/4426]	Eit 92800  lr 5e-05  Le 38.2715 (47.0454)	Time 0.847 (0.000)	Data 0.002 (0.000)	
2021-06-22 12:26:08,977 Test: [0/40]	Le 253.3085 (253.3083)	Time 7.936 (0.000)	
2021-06-22 12:26:24,018 calculate similarity time: 0.07988834381103516
2021-06-22 12:26:24,565 Image to text: 71.6, 95.0, 97.9, 1.0, 2.5
2021-06-22 12:26:24,887 Text to image: 60.9, 91.0, 96.5, 1.0, 3.3
2021-06-22 12:26:24,888 Current rsum is 512.96
2021-06-22 12:26:26,359 runs/coco_butd_region_bert/log
2021-06-22 12:26:26,359 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-22 12:26:26,361 image encoder trainable parameters: 20490144
2021-06-22 12:26:26,366 txt encoder trainable parameters: 137319072
2021-06-22 12:27:45,942 Epoch: [21][74/4426]	Eit 93000  lr 5e-05  Le 39.9231 (47.9883)	Time 1.000 (0.000)	Data 0.002 (0.000)	
2021-06-22 12:30:54,523 Epoch: [21][274/4426]	Eit 93200  lr 5e-05  Le 41.0345 (46.5669)	Time 1.166 (0.000)	Data 0.002 (0.000)	
2021-06-22 12:34:03,970 Epoch: [21][474/4426]	Eit 93400  lr 5e-05  Le 27.9892 (46.3818)	Time 1.003 (0.000)	Data 0.002 (0.000)	
2021-06-22 12:37:13,121 Epoch: [21][674/4426]	Eit 93600  lr 5e-05  Le 61.4007 (46.2470)	Time 1.041 (0.000)	Data 0.002 (0.000)	
2021-06-22 12:40:21,088 Epoch: [21][874/4426]	Eit 93800  lr 5e-05  Le 60.1963 (46.3439)	Time 0.908 (0.000)	Data 0.002 (0.000)	
2021-06-22 12:43:30,775 Epoch: [21][1074/4426]	Eit 94000  lr 5e-05  Le 33.3307 (46.6278)	Time 0.959 (0.000)	Data 0.002 (0.000)	
2021-06-22 12:46:39,165 Epoch: [21][1274/4426]	Eit 94200  lr 5e-05  Le 32.3718 (47.0230)	Time 0.777 (0.000)	Data 0.002 (0.000)	
2021-06-22 12:49:48,330 Epoch: [21][1474/4426]	Eit 94400  lr 5e-05  Le 39.4187 (47.1680)	Time 0.846 (0.000)	Data 0.002 (0.000)	
2021-06-22 12:52:58,019 Epoch: [21][1674/4426]	Eit 94600  lr 5e-05  Le 38.9371 (46.8712)	Time 1.084 (0.000)	Data 0.002 (0.000)	
2021-06-22 12:56:08,471 Epoch: [21][1874/4426]	Eit 94800  lr 5e-05  Le 32.6554 (46.7748)	Time 1.005 (0.000)	Data 0.002 (0.000)	
2021-06-22 12:59:16,511 Epoch: [21][2074/4426]	Eit 95000  lr 5e-05  Le 55.8526 (46.6839)	Time 0.887 (0.000)	Data 0.003 (0.000)	
2021-06-22 13:02:27,704 Epoch: [21][2274/4426]	Eit 95200  lr 5e-05  Le 38.4368 (46.5588)	Time 0.873 (0.000)	Data 0.002 (0.000)	
2021-06-22 13:05:37,386 Epoch: [21][2474/4426]	Eit 95400  lr 5e-05  Le 45.7900 (46.5271)	Time 1.053 (0.000)	Data 0.002 (0.000)	
2021-06-22 13:08:47,268 Epoch: [21][2674/4426]	Eit 95600  lr 5e-05  Le 41.8183 (46.4995)	Time 0.836 (0.000)	Data 0.002 (0.000)	
2021-06-22 13:11:57,799 Epoch: [21][2874/4426]	Eit 95800  lr 5e-05  Le 67.1975 (46.5434)	Time 1.190 (0.000)	Data 0.002 (0.000)	
2021-06-22 13:15:08,024 Epoch: [21][3074/4426]	Eit 96000  lr 5e-05  Le 58.0698 (46.4924)	Time 1.028 (0.000)	Data 0.002 (0.000)	
2021-06-22 13:18:16,824 Epoch: [21][3274/4426]	Eit 96200  lr 5e-05  Le 55.8850 (46.4422)	Time 1.001 (0.000)	Data 0.002 (0.000)	
2021-06-22 13:21:24,533 Epoch: [21][3474/4426]	Eit 96400  lr 5e-05  Le 36.7385 (46.4388)	Time 0.817 (0.000)	Data 0.002 (0.000)	
2021-06-22 13:24:34,556 Epoch: [21][3674/4426]	Eit 96600  lr 5e-05  Le 22.0606 (46.4285)	Time 0.983 (0.000)	Data 0.002 (0.000)	
2021-06-22 13:27:43,394 Epoch: [21][3874/4426]	Eit 96800  lr 5e-05  Le 38.7319 (46.4529)	Time 1.019 (0.000)	Data 0.002 (0.000)	
2021-06-22 13:30:40,469 Epoch: [21][4074/4426]	Eit 97000  lr 5e-05  Le 53.4897 (46.3516)	Time 0.870 (0.000)	Data 0.002 (0.000)	
2021-06-22 13:33:49,744 Epoch: [21][4274/4426]	Eit 97200  lr 5e-05  Le 33.6654 (46.3625)	Time 0.959 (0.000)	Data 0.003 (0.000)	
2021-06-22 13:36:20,150 Test: [0/40]	Le 251.8209 (251.8207)	Time 7.469 (0.000)	
2021-06-22 13:36:35,060 calculate similarity time: 0.06978416442871094
2021-06-22 13:36:35,485 Image to text: 71.4, 95.4, 98.1, 1.0, 2.7
2021-06-22 13:36:35,924 Text to image: 61.0, 91.0, 96.8, 1.0, 3.2
2021-06-22 13:36:35,924 Current rsum is 513.74
2021-06-22 13:36:39,834 runs/coco_butd_region_bert/log
2021-06-22 13:36:39,834 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-22 13:36:39,836 image encoder trainable parameters: 20490144
2021-06-22 13:36:39,843 txt encoder trainable parameters: 137319072
2021-06-22 13:37:34,441 Epoch: [22][49/4426]	Eit 97400  lr 5e-05  Le 66.7716 (45.6848)	Time 0.778 (0.000)	Data 0.002 (0.000)	
2021-06-22 13:40:44,231 Epoch: [22][249/4426]	Eit 97600  lr 5e-05  Le 77.9376 (46.4956)	Time 0.994 (0.000)	Data 0.002 (0.000)	
2021-06-22 13:43:53,458 Epoch: [22][449/4426]	Eit 97800  lr 5e-05  Le 51.3843 (46.1906)	Time 0.840 (0.000)	Data 0.003 (0.000)	
2021-06-22 13:47:01,961 Epoch: [22][649/4426]	Eit 98000  lr 5e-05  Le 56.0591 (46.1874)	Time 0.952 (0.000)	Data 0.002 (0.000)	
2021-06-22 13:50:12,053 Epoch: [22][849/4426]	Eit 98200  lr 5e-05  Le 53.6820 (46.0328)	Time 1.089 (0.000)	Data 0.002 (0.000)	
2021-06-22 13:53:21,961 Epoch: [22][1049/4426]	Eit 98400  lr 5e-05  Le 57.5975 (46.0224)	Time 1.135 (0.000)	Data 0.002 (0.000)	
2021-06-22 13:56:32,124 Epoch: [22][1249/4426]	Eit 98600  lr 5e-05  Le 67.4804 (46.3444)	Time 0.845 (0.000)	Data 0.004 (0.000)	
2021-06-22 13:59:43,734 Epoch: [22][1449/4426]	Eit 98800  lr 5e-05  Le 34.0260 (46.3753)	Time 0.877 (0.000)	Data 0.003 (0.000)	
2021-06-22 14:02:54,653 Epoch: [22][1649/4426]	Eit 99000  lr 5e-05  Le 51.4072 (46.4179)	Time 0.936 (0.000)	Data 0.002 (0.000)	
2021-06-22 14:06:04,698 Epoch: [22][1849/4426]	Eit 99200  lr 5e-05  Le 32.5738 (46.3843)	Time 1.001 (0.000)	Data 0.010 (0.000)	
2021-06-22 14:09:17,220 Epoch: [22][2049/4426]	Eit 99400  lr 5e-05  Le 48.5533 (46.3014)	Time 0.863 (0.000)	Data 0.003 (0.000)	
2021-06-22 14:12:28,563 Epoch: [22][2249/4426]	Eit 99600  lr 5e-05  Le 29.8376 (46.0901)	Time 0.912 (0.000)	Data 0.002 (0.000)	
2021-06-22 14:15:38,448 Epoch: [22][2449/4426]	Eit 99800  lr 5e-05  Le 40.3643 (46.1229)	Time 0.838 (0.000)	Data 0.004 (0.000)	
2021-06-22 14:18:49,154 Epoch: [22][2649/4426]	Eit 100000  lr 5e-05  Le 60.1173 (46.1196)	Time 0.778 (0.000)	Data 0.011 (0.000)	
2021-06-22 14:22:00,193 Epoch: [22][2849/4426]	Eit 100200  lr 5e-05  Le 32.1687 (46.0496)	Time 0.912 (0.000)	Data 0.002 (0.000)	
2021-06-22 14:25:12,218 Epoch: [22][3049/4426]	Eit 100400  lr 5e-05  Le 40.5714 (46.0671)	Time 0.878 (0.000)	Data 0.002 (0.000)	
2021-06-22 14:28:23,674 Epoch: [22][3249/4426]	Eit 100600  lr 5e-05  Le 44.3119 (46.0405)	Time 1.179 (0.000)	Data 0.002 (0.000)	
2021-06-22 14:31:34,885 Epoch: [22][3449/4426]	Eit 100800  lr 5e-05  Le 60.8854 (46.0519)	Time 1.055 (0.000)	Data 0.002 (0.000)	
2021-06-22 14:34:45,402 Epoch: [22][3649/4426]	Eit 101000  lr 5e-05  Le 32.9257 (46.0615)	Time 0.744 (0.000)	Data 0.002 (0.000)	
2021-06-22 14:37:55,107 Epoch: [22][3849/4426]	Eit 101200  lr 5e-05  Le 31.9101 (45.9925)	Time 0.830 (0.000)	Data 0.002 (0.000)	
2021-06-22 14:41:03,150 Epoch: [22][4049/4426]	Eit 101400  lr 5e-05  Le 70.9796 (46.0152)	Time 0.954 (0.000)	Data 0.002 (0.000)	
2021-06-22 14:44:13,185 Epoch: [22][4249/4426]	Eit 101600  lr 5e-05  Le 70.5761 (46.0402)	Time 0.939 (0.000)	Data 0.002 (0.000)	
2021-06-22 14:47:09,343 Test: [0/40]	Le 252.6592 (252.6590)	Time 9.882 (0.000)	
2021-06-22 14:47:24,692 calculate similarity time: 0.07185530662536621
2021-06-22 14:47:25,105 Image to text: 71.1, 94.8, 98.2, 1.0, 2.7
2021-06-22 14:47:25,416 Text to image: 61.2, 91.1, 96.8, 1.0, 3.2
2021-06-22 14:47:25,416 Current rsum is 513.14
2021-06-22 14:47:26,917 runs/coco_butd_region_bert/log
2021-06-22 14:47:26,917 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-22 14:47:26,919 image encoder trainable parameters: 20490144
2021-06-22 14:47:26,924 txt encoder trainable parameters: 137319072
2021-06-22 14:47:59,960 Epoch: [23][24/4426]	Eit 101800  lr 5e-05  Le 51.3617 (41.2792)	Time 0.935 (0.000)	Data 0.002 (0.000)	
2021-06-22 14:51:08,553 Epoch: [23][224/4426]	Eit 102000  lr 5e-05  Le 30.8490 (45.0668)	Time 0.830 (0.000)	Data 0.002 (0.000)	
2021-06-22 14:54:16,830 Epoch: [23][424/4426]	Eit 102200  lr 5e-05  Le 38.9666 (45.5228)	Time 1.016 (0.000)	Data 0.002 (0.000)	
2021-06-22 14:57:25,346 Epoch: [23][624/4426]	Eit 102400  lr 5e-05  Le 42.6703 (45.7190)	Time 0.917 (0.000)	Data 0.002 (0.000)	
2021-06-22 15:00:37,007 Epoch: [23][824/4426]	Eit 102600  lr 5e-05  Le 58.1356 (46.3216)	Time 1.044 (0.000)	Data 0.002 (0.000)	
2021-06-22 15:03:45,460 Epoch: [23][1024/4426]	Eit 102800  lr 5e-05  Le 50.6172 (45.5483)	Time 1.016 (0.000)	Data 0.002 (0.000)	
2021-06-22 15:06:54,636 Epoch: [23][1224/4426]	Eit 103000  lr 5e-05  Le 42.6861 (45.4100)	Time 0.824 (0.000)	Data 0.002 (0.000)	
2021-06-22 15:10:02,671 Epoch: [23][1424/4426]	Eit 103200  lr 5e-05  Le 51.6352 (45.5243)	Time 0.890 (0.000)	Data 0.002 (0.000)	
2021-06-22 15:12:55,500 Epoch: [23][1624/4426]	Eit 103400  lr 5e-05  Le 38.7903 (45.4693)	Time 0.885 (0.000)	Data 0.002 (0.000)	
2021-06-22 15:16:04,581 Epoch: [23][1824/4426]	Eit 103600  lr 5e-05  Le 29.0174 (45.4263)	Time 1.052 (0.000)	Data 0.002 (0.000)	
2021-06-22 15:19:13,708 Epoch: [23][2024/4426]	Eit 103800  lr 5e-05  Le 48.1222 (45.5370)	Time 1.010 (0.000)	Data 0.002 (0.000)	
2021-06-22 15:22:25,088 Epoch: [23][2224/4426]	Eit 104000  lr 5e-05  Le 38.6176 (45.5231)	Time 0.824 (0.000)	Data 0.002 (0.000)	
2021-06-22 15:25:33,875 Epoch: [23][2424/4426]	Eit 104200  lr 5e-05  Le 42.9308 (45.4415)	Time 0.998 (0.000)	Data 0.002 (0.000)	
2021-06-22 15:28:42,865 Epoch: [23][2624/4426]	Eit 104400  lr 5e-05  Le 32.7084 (45.5252)	Time 1.010 (0.000)	Data 0.002 (0.000)	
2021-06-22 15:31:54,392 Epoch: [23][2824/4426]	Eit 104600  lr 5e-05  Le 27.5714 (45.4700)	Time 0.843 (0.000)	Data 0.002 (0.000)	
2021-06-22 15:35:04,660 Epoch: [23][3024/4426]	Eit 104800  lr 5e-05  Le 30.0349 (45.2548)	Time 0.901 (0.000)	Data 0.002 (0.000)	
2021-06-22 15:38:13,853 Epoch: [23][3224/4426]	Eit 105000  lr 5e-05  Le 39.3720 (45.1783)	Time 1.084 (0.000)	Data 0.002 (0.000)	
2021-06-22 15:41:24,581 Epoch: [23][3424/4426]	Eit 105200  lr 5e-05  Le 53.5848 (45.2341)	Time 0.831 (0.000)	Data 0.002 (0.000)	
2021-06-22 15:44:33,796 Epoch: [23][3624/4426]	Eit 105400  lr 5e-05  Le 64.6002 (45.1821)	Time 1.000 (0.000)	Data 0.002 (0.000)	
2021-06-22 15:47:44,280 Epoch: [23][3824/4426]	Eit 105600  lr 5e-05  Le 37.5321 (45.2711)	Time 1.074 (0.000)	Data 0.002 (0.000)	
2021-06-22 15:50:53,867 Epoch: [23][4024/4426]	Eit 105800  lr 5e-05  Le 43.4813 (45.2778)	Time 1.020 (0.000)	Data 0.002 (0.000)	
2021-06-22 15:54:01,790 Epoch: [23][4224/4426]	Eit 106000  lr 5e-05  Le 61.7087 (45.3767)	Time 1.069 (0.000)	Data 0.002 (0.000)	
2021-06-22 15:57:12,124 Epoch: [23][4424/4426]	Eit 106200  lr 5e-05  Le 46.2540 (45.4068)	Time 1.032 (0.000)	Data 0.002 (0.000)	
2021-06-22 15:57:22,993 Test: [0/40]	Le 255.6322 (255.6320)	Time 9.545 (0.000)	
2021-06-22 15:57:37,842 calculate similarity time: 0.06755781173706055
2021-06-22 15:57:38,341 Image to text: 72.2, 94.6, 98.2, 1.0, 2.8
2021-06-22 15:57:38,655 Text to image: 61.2, 90.9, 96.7, 1.0, 3.2
2021-06-22 15:57:38,655 Current rsum is 513.8000000000001
2021-06-22 15:57:42,385 runs/coco_butd_region_bert/log
2021-06-22 15:57:42,386 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-22 15:57:42,388 image encoder trainable parameters: 20490144
2021-06-22 15:57:42,402 txt encoder trainable parameters: 137319072
2021-06-22 16:01:00,476 Epoch: [24][199/4426]	Eit 106400  lr 5e-05  Le 39.2289 (46.4545)	Time 0.921 (0.000)	Data 0.002 (0.000)	
2021-06-22 16:04:10,132 Epoch: [24][399/4426]	Eit 106600  lr 5e-05  Le 28.8067 (45.5244)	Time 1.224 (0.000)	Data 0.002 (0.000)	
2021-06-22 16:07:20,850 Epoch: [24][599/4426]	Eit 106800  lr 5e-05  Le 49.2986 (45.3257)	Time 0.847 (0.000)	Data 0.002 (0.000)	
2021-06-22 16:10:29,140 Epoch: [24][799/4426]	Eit 107000  lr 5e-05  Le 46.7295 (45.4489)	Time 0.889 (0.000)	Data 0.002 (0.000)	
2021-06-22 16:13:37,583 Epoch: [24][999/4426]	Eit 107200  lr 5e-05  Le 26.1746 (45.2727)	Time 0.827 (0.000)	Data 0.002 (0.000)	
2021-06-22 16:16:47,070 Epoch: [24][1199/4426]	Eit 107400  lr 5e-05  Le 39.9746 (45.2414)	Time 0.844 (0.000)	Data 0.002 (0.000)	
2021-06-22 16:19:55,117 Epoch: [24][1399/4426]	Eit 107600  lr 5e-05  Le 42.2580 (45.2723)	Time 0.995 (0.000)	Data 0.002 (0.000)	
2021-06-22 16:23:02,626 Epoch: [24][1599/4426]	Eit 107800  lr 5e-05  Le 39.3862 (45.5042)	Time 0.984 (0.000)	Data 0.002 (0.000)	
2021-06-22 16:26:10,980 Epoch: [24][1799/4426]	Eit 108000  lr 5e-05  Le 34.8276 (45.2119)	Time 0.875 (0.000)	Data 0.002 (0.000)	
2021-06-22 16:29:20,234 Epoch: [24][1999/4426]	Eit 108200  lr 5e-05  Le 52.2250 (45.1865)	Time 0.857 (0.000)	Data 0.002 (0.000)	
2021-06-22 16:32:29,916 Epoch: [24][2199/4426]	Eit 108400  lr 5e-05  Le 30.5105 (45.0542)	Time 0.794 (0.000)	Data 0.002 (0.000)	
2021-06-22 16:35:37,890 Epoch: [24][2399/4426]	Eit 108600  lr 5e-05  Le 46.0649 (44.9681)	Time 0.850 (0.000)	Data 0.002 (0.000)	
2021-06-22 16:38:47,985 Epoch: [24][2599/4426]	Eit 108800  lr 5e-05  Le 72.7513 (45.0634)	Time 1.045 (0.000)	Data 0.002 (0.000)	
2021-06-22 16:41:56,284 Epoch: [24][2799/4426]	Eit 109000  lr 5e-05  Le 31.9486 (45.0778)	Time 0.801 (0.000)	Data 0.002 (0.000)	
2021-06-22 16:45:04,882 Epoch: [24][2999/4426]	Eit 109200  lr 5e-05  Le 69.2213 (45.0618)	Time 1.087 (0.000)	Data 0.002 (0.000)	
2021-06-22 16:48:15,005 Epoch: [24][3199/4426]	Eit 109400  lr 5e-05  Le 36.1225 (45.0224)	Time 1.028 (0.000)	Data 0.002 (0.000)	
2021-06-22 16:51:23,697 Epoch: [24][3399/4426]	Eit 109600  lr 5e-05  Le 20.8006 (45.0621)	Time 0.866 (0.000)	Data 0.002 (0.000)	
2021-06-22 16:54:29,459 Epoch: [24][3599/4426]	Eit 109800  lr 5e-05  Le 32.1164 (45.0751)	Time 0.952 (0.000)	Data 0.002 (0.000)	
2021-06-22 16:57:31,859 Epoch: [24][3799/4426]	Eit 110000  lr 5e-05  Le 33.8078 (45.1171)	Time 0.828 (0.000)	Data 0.002 (0.000)	
2021-06-22 17:00:41,858 Epoch: [24][3999/4426]	Eit 110200  lr 5e-05  Le 20.6967 (45.0788)	Time 0.973 (0.000)	Data 0.002 (0.000)	
2021-06-22 17:03:52,635 Epoch: [24][4199/4426]	Eit 110400  lr 5e-05  Le 48.8767 (45.0461)	Time 1.099 (0.000)	Data 0.002 (0.000)	
2021-06-22 17:07:02,536 Epoch: [24][4399/4426]	Eit 110600  lr 5e-05  Le 35.1107 (45.0648)	Time 0.722 (0.000)	Data 0.002 (0.000)	
2021-06-22 17:07:36,089 Test: [0/40]	Le 253.6674 (253.6672)	Time 8.570 (0.000)	
2021-06-22 17:07:50,951 calculate similarity time: 0.06948089599609375
2021-06-22 17:07:51,428 Image to text: 71.8, 94.7, 98.3, 1.0, 2.7
2021-06-22 17:07:51,739 Text to image: 60.9, 91.1, 96.6, 1.0, 3.2
2021-06-22 17:07:51,739 Current rsum is 513.5
You have new mail in /var/spool/mail/root
[root@gpu1 vse_infty-master-my-graph-gru-vse]# CUDA_VISIBLE_DEVICES=0,1 python3 eval.py --dataset coco --data_path ../data/coco
INFO:root:Evaluating runs/coco_butd_region_bert...
INFO:lib.evaluation:Namespace(backbone_lr_factor=0.01, backbone_path='', backbone_source='detector', backbone_warmup_epochs=5, batch_size=128, data_name='coco', data_path='../data/coco', embed_size=1024, embedding_warmup_epochs=2, grad_clip=2.0, img_dim=2048, input_scale_factor=1, learning_rate=0.0005, log_step=200, logger_name='runs/coco_butd_region_bert/log', lr_update=15, margin=0.2, max_violation=True, model_name='runs/coco_butd_region_bert', no_imgnorm=False, no_txtnorm=False, num_epochs=25, optim='adam', precomp_enc_type='basic', reset_start_epoch=False, resume='', val_step=500, vocab_path='./vocab/', vocab_size=30522, vse_mean_warmup_epochs=1, word_dim=300, workers=5)
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
INFO:lib.evaluation:Test: [0/196]	Le 294.1564 (294.1561)	Time 7.062 (0.000)	
INFO:lib.evaluation:Test: [10/196]	Le 325.8827 (348.3169)	Time 0.567 (0.000)	
INFO:lib.evaluation:Test: [20/196]	Le 312.5199 (346.9449)	Time 0.229 (0.000)	
INFO:lib.evaluation:Test: [30/196]	Le 421.2379 (354.4483)	Time 0.383 (0.000)	
INFO:lib.evaluation:Test: [40/196]	Le 346.0676 (347.5916)	Time 0.223 (0.000)	
INFO:lib.evaluation:Test: [50/196]	Le 270.4281 (348.1202)	Time 0.371 (0.000)	
INFO:lib.evaluation:Test: [60/196]	Le 357.0307 (348.2660)	Time 0.268 (0.000)	
INFO:lib.evaluation:Test: [70/196]	Le 405.9034 (345.2909)	Time 0.404 (0.000)	
INFO:lib.evaluation:Test: [80/196]	Le 379.0026 (349.7486)	Time 0.364 (0.000)	
INFO:lib.evaluation:Test: [90/196]	Le 322.2385 (351.1173)	Time 0.491 (0.000)	
INFO:lib.evaluation:Test: [100/196]	Le 338.6799 (353.2177)	Time 0.382 (0.000)	
INFO:lib.evaluation:Test: [110/196]	Le 311.5255 (351.2100)	Time 0.391 (0.000)	
INFO:lib.evaluation:Test: [120/196]	Le 507.7498 (349.3945)	Time 0.386 (0.000)	
INFO:lib.evaluation:Test: [130/196]	Le 290.1730 (348.9479)	Time 0.402 (0.000)	
INFO:lib.evaluation:Test: [140/196]	Le 479.8446 (350.7158)	Time 0.329 (0.000)	
INFO:lib.evaluation:Test: [150/196]	Le 243.2328 (348.0939)	Time 0.462 (0.000)	
INFO:lib.evaluation:Test: [160/196]	Le 307.3575 (346.9708)	Time 0.384 (0.000)	
INFO:lib.evaluation:Test: [170/196]	Le 407.5431 (346.1864)	Time 0.255 (0.000)	
INFO:lib.evaluation:Test: [180/196]	Le 320.6277 (344.5731)	Time 0.444 (0.000)	
INFO:lib.evaluation:Test: [190/196]	Le 251.2290 (347.3016)	Time 0.233 (0.000)	
INFO:lib.evaluation:Images: 5000, Captions: 25000
INFO:lib.evaluation:calculate similarity time: 0.1068720817565918
INFO:lib.evaluation:Image to text: 71.3, 95.5, 98.5, 1.0, 1.8
INFO:lib.evaluation:Text to image: 61.2, 90.3, 95.6, 1.0, 4.0
INFO:lib.evaluation:rsum: 512.4 ar: 88.4 ari: 82.4
INFO:lib.evaluation:calculate similarity time: 0.08903622627258301
INFO:lib.evaluation:Image to text: 72.2, 94.0, 97.4, 1.0, 2.1
INFO:lib.evaluation:Text to image: 59.3, 88.3, 95.0, 1.0, 4.1
INFO:lib.evaluation:rsum: 506.2 ar: 87.9 ari: 80.9
INFO:lib.evaluation:calculate similarity time: 0.08624386787414551
INFO:lib.evaluation:Image to text: 70.9, 93.2, 97.7, 1.0, 2.1
INFO:lib.evaluation:Text to image: 60.0, 89.4, 95.6, 1.0, 3.6
INFO:lib.evaluation:rsum: 506.8 ar: 87.3 ari: 81.7
INFO:lib.evaluation:calculate similarity time: 0.08415794372558594
INFO:lib.evaluation:Image to text: 71.6, 94.9, 98.1, 1.0, 1.9
INFO:lib.evaluation:Text to image: 58.2, 89.5, 96.1, 1.0, 3.3
INFO:lib.evaluation:rsum: 508.3 ar: 88.2 ari: 81.2
INFO:lib.evaluation:calculate similarity time: 0.06989789009094238
INFO:lib.evaluation:Image to text: 71.8, 94.9, 98.3, 1.0, 1.9
INFO:lib.evaluation:Text to image: 60.3, 90.2, 96.0, 1.0, 3.4
INFO:lib.evaluation:rsum: 511.4 ar: 88.3 ari: 82.1
INFO:lib.evaluation:-----------------------------------
INFO:lib.evaluation:Mean metrics: 
INFO:lib.evaluation:rsum: 509.0
INFO:lib.evaluation:Average i2t Recall: 88.0
INFO:lib.evaluation:Image to text: 71.6 94.5 98.0 1.0 2.0
INFO:lib.evaluation:Average t2i Recall: 81.7
INFO:lib.evaluation:Text to image: 59.8 89.5 95.7 1.0 3.7
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
INFO:lib.evaluation:Test: [0/196]	Le 294.1564 (294.1561)	Time 2.490 (0.000)	
INFO:lib.evaluation:Test: [10/196]	Le 325.8827 (348.3169)	Time 0.293 (0.000)	
INFO:lib.evaluation:Test: [20/196]	Le 312.5199 (346.9449)	Time 0.379 (0.000)	
INFO:lib.evaluation:Test: [30/196]	Le 421.2379 (354.4483)	Time 0.489 (0.000)	
INFO:lib.evaluation:Test: [40/196]	Le 346.0676 (347.5916)	Time 0.321 (0.000)	
INFO:lib.evaluation:Test: [50/196]	Le 270.4281 (348.1202)	Time 0.434 (0.000)	
INFO:lib.evaluation:Test: [60/196]	Le 357.0307 (348.2660)	Time 0.332 (0.000)	
INFO:lib.evaluation:Test: [70/196]	Le 405.9034 (345.2909)	Time 0.416 (0.000)	
INFO:lib.evaluation:Test: [80/196]	Le 379.0026 (349.7486)	Time 0.521 (0.000)	
INFO:lib.evaluation:Test: [90/196]	Le 322.2385 (351.1173)	Time 0.239 (0.000)	
INFO:lib.evaluation:Test: [100/196]	Le 338.6799 (353.2177)	Time 0.450 (0.000)	
INFO:lib.evaluation:Test: [110/196]	Le 311.5255 (351.2100)	Time 0.351 (0.000)	
INFO:lib.evaluation:Test: [120/196]	Le 507.7498 (349.3945)	Time 0.449 (0.000)	
INFO:lib.evaluation:Test: [130/196]	Le 290.1730 (348.9479)	Time 0.345 (0.000)	
INFO:lib.evaluation:Test: [140/196]	Le 479.8446 (350.7158)	Time 0.373 (0.000)	
INFO:lib.evaluation:Test: [150/196]	Le 243.2328 (348.0939)	Time 0.338 (0.000)	
INFO:lib.evaluation:Test: [160/196]	Le 307.3575 (346.9708)	Time 0.417 (0.000)	
INFO:lib.evaluation:Test: [170/196]	Le 407.5431 (346.1864)	Time 0.471 (0.000)	
INFO:lib.evaluation:Test: [180/196]	Le 320.6277 (344.5731)	Time 0.376 (0.000)	
INFO:lib.evaluation:Test: [190/196]	Le 251.2290 (347.3016)	Time 0.327 (0.000)	
INFO:lib.evaluation:Images: 5000, Captions: 25000
INFO:lib.evaluation:calculate similarity time: 0.8939573764801025
INFO:lib.evaluation:rsum: 396.6
INFO:lib.evaluation:Average i2t Recall: 71.0
INFO:lib.evaluation:Image to text: 48.1 77.7 87.3 2.0 5.7
INFO:lib.evaluation:Average t2i Recall: 61.2
INFO:lib.evaluation:Text to image: 36.3 67.6 79.6 2.0 14.2
INFO:root:Evaluating runs/coco_butd_grid_bert...
Traceback (most recent call last):
  File "eval.py", line 58, in <module>
    main()
  File "eval.py", line 46, in main
    evaluation.evalrank(model_path, data_path=opt.data_path, split='testall', fold5=True)
  File "/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-vse/lib/evaluation.py", line 196, in evalrank
    checkpoint = torch.load(model_path)
  File "/root/anaconda3/lib/python3.6/site-packages/torch/serialization.py", line 571, in load
    with _open_file_like(f, 'rb') as opened_file:
  File "/root/anaconda3/lib/python3.6/site-packages/torch/serialization.py", line 229, in _open_file_like
    return _open_file(name_or_buffer, mode)
  File "/root/anaconda3/lib/python3.6/site-packages/torch/serialization.py", line 210, in __init__
    super(_open_file, self).__init__(open(name, mode))
FileNotFoundError: [Errno 2] No such file or directory: 'runs/coco_butd_grid_bert/model_best.pth'

