[root@gpu1 vse_infty-master-my-graph-gru-unified-no-graph]# sh train_region.sh 

2021-06-19 16:20:17,282 Namespace(backbone_lr_factor=0.01, backbone_path='', backbone_source='detector', backbone_warmup_epochs=5, batch_size=128, data_name='coco', data_path='../data/coco', embed_size=1024, embedding_warmup_epochs=2, grad_clip=2.0, img_dim=2048, input_scale_factor=1, learning_rate=0.0005, log_step=200, logger_name='runs/coco_butd_region_bert/log', lr_update=15, margin=0.2, max_violation=False, model_name='runs/coco_butd_region_bert', no_imgnorm=False, no_txtnorm=False, num_epochs=25, optim='adam', precomp_enc_type='basic', reset_start_epoch=False, resume='', val_step=500, vocab_path='./vocab/', vse_mean_warmup_epochs=1, word_dim=300, workers=10)
2021-06-19 16:20:17,282 Model name '/public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased' not found in model shortcut name list (bert-base-uncased, bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, bert-base-multilingual-cased, bert-base-chinese, bert-base-german-cased, bert-large-uncased-whole-word-masking, bert-large-cased-whole-word-masking, bert-large-uncased-whole-word-masking-finetuned-squad, bert-large-cased-whole-word-masking-finetuned-squad, bert-base-cased-finetuned-mrpc). Assuming '/public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased' is a path or url to a directory containing tokenizer files.
2021-06-19 16:20:17,282 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/added_tokens.json. We won't load it.
2021-06-19 16:20:17,282 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/special_tokens_map.json. We won't load it.
2021-06-19 16:20:17,282 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/tokenizer_config.json. We won't load it.
2021-06-19 16:20:17,282 loading file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/vocab.txt
2021-06-19 16:20:17,282 loading file None
2021-06-19 16:20:17,283 loading file None
2021-06-19 16:20:17,283 loading file None
2021-06-19 16:20:46,342 loading configuration file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/config.json
2021-06-19 16:20:46,342 Model config {
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

2021-06-19 16:20:46,343 loading weights file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/pytorch_model.bin
2021-06-19 16:20:53,846 Use adam as the optimizer, with init lr 0.0005
2021-06-19 16:20:53,847 Image encoder is data paralleled now.
2021-06-19 16:20:53,847 runs/coco_butd_region_bert/log
2021-06-19 16:20:53,847 runs/coco_butd_region_bert
2021-06-19 16:20:53,848 image encoder trainable parameters: 3688352
2021-06-19 16:20:53,853 txt encoder trainable parameters: 120517280
/root/anaconda3/lib/python3.6/site-packages/torch/nn/modules/rnn.py:735: UserWarning: RNN module weights are not part of single contiguous chunk of memory. This means they need to be compacted at every call, possibly greatly increasing memory usage. To compact weights again call flatten_parameters(). (Triggered internally at  /pytorch/aten/src/ATen/native/cudnn/RNN.cpp:1264.)
  self.dropout, self.training, self.bidirectional, self.batch_first)
Traceback (most recent call last):
  File "train.py", line 269, in <module>
    main()
  File "train.py", line 99, in main
    train(opt, train_loader, model, epoch, val_loader)
  File "train.py", line 147, in train
    model.train_emb(images, captions, lengths, image_lengths=img_lengths)
  File "/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-unified-no-graph/lib/vse.py", line 186, in train_emb
    img_emb, cap_emb = self.forward_emb(images, captions, lengths, image_lengths=image_lengths)
  File "/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-unified-no-graph/lib/vse.py", line 168, in forward_emb
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
  File "/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-unified-no-graph/lib/encoders.py", line 287, in forward
    bert_emb = self.bert(x, bert_attention_mask)[0]  # B x N x D
  File "/root/anaconda3/lib/python3.6/site-packages/torch/nn/modules/module.py", line 722, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/root/anaconda3/lib/python3.6/site-packages/transformers/modeling_bert.py", line 625, in forward
    head_mask=head_mask)
  File "/root/anaconda3/lib/python3.6/site-packages/torch/nn/modules/module.py", line 722, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/root/anaconda3/lib/python3.6/site-packages/transformers/modeling_bert.py", line 346, in forward
    layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])
  File "/root/anaconda3/lib/python3.6/site-packages/torch/nn/modules/module.py", line 722, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/root/anaconda3/lib/python3.6/site-packages/transformers/modeling_bert.py", line 326, in forward
    intermediate_output = self.intermediate(attention_output)
  File "/root/anaconda3/lib/python3.6/site-packages/torch/nn/modules/module.py", line 722, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/root/anaconda3/lib/python3.6/site-packages/transformers/modeling_bert.py", line 298, in forward
    hidden_states = self.intermediate_act_fn(hidden_states)
  File "/root/anaconda3/lib/python3.6/site-packages/transformers/modeling_bert.py", line 126, in gelu
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
RuntimeError: CUDA out of memory. Tried to allocate 18.00 MiB (GPU 0; 22.38 GiB total capacity; 1.30 GiB already allocated; 9.56 MiB free; 1.33 GiB reserved in total by PyTorch)

[root@gpu1 vse_infty-master-my-graph-gru-unified-no-graph]# 
[root@gpu1 vse_infty-master-my-graph-gru-unified-no-graph]# sh train_region.sh 
2021-06-19 16:24:40,379 Namespace(backbone_lr_factor=0.01, backbone_path='', backbone_source='detector', backbone_warmup_epochs=5, batch_size=128, data_name='coco', data_path='../data/coco', embed_size=1024, embedding_warmup_epochs=2, grad_clip=2.0, img_dim=2048, input_scale_factor=1, learning_rate=0.0005, log_step=200, logger_name='runs/coco_butd_region_bert/log', lr_update=15, margin=0.2, max_violation=False, model_name='runs/coco_butd_region_bert', no_imgnorm=False, no_txtnorm=False, num_epochs=25, optim='adam', precomp_enc_type='basic', reset_start_epoch=False, resume='', val_step=500, vocab_path='./vocab/', vse_mean_warmup_epochs=1, word_dim=300, workers=10)
2021-06-19 16:24:40,379 Model name '/public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased' not found in model shortcut name list (bert-base-uncased, bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, bert-base-multilingual-cased, bert-base-chinese, bert-base-german-cased, bert-large-uncased-whole-word-masking, bert-large-cased-whole-word-masking, bert-large-uncased-whole-word-masking-finetuned-squad, bert-large-cased-whole-word-masking-finetuned-squad, bert-base-cased-finetuned-mrpc). Assuming '/public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased' is a path or url to a directory containing tokenizer files.
2021-06-19 16:24:40,390 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/added_tokens.json. We won't load it.
2021-06-19 16:24:40,390 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/special_tokens_map.json. We won't load it.
2021-06-19 16:24:40,391 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/tokenizer_config.json. We won't load it.
2021-06-19 16:24:40,391 loading file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/vocab.txt
2021-06-19 16:24:40,391 loading file None
2021-06-19 16:24:40,391 loading file None
2021-06-19 16:24:40,391 loading file None
2021-06-19 16:25:08,786 loading configuration file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/config.json
2021-06-19 16:25:08,787 Model config {
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

2021-06-19 16:25:08,789 loading weights file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/pytorch_model.bin
2021-06-19 16:25:15,551 Use adam as the optimizer, with init lr 0.0005
2021-06-19 16:25:15,552 Image encoder is data paralleled now.
2021-06-19 16:25:15,552 runs/coco_butd_region_bert/log
2021-06-19 16:25:15,552 runs/coco_butd_region_bert
2021-06-19 16:25:15,553 image encoder trainable parameters: 3688352
2021-06-19 16:25:15,557 txt encoder trainable parameters: 120517280
/root/anaconda3/lib/python3.6/site-packages/torch/nn/modules/rnn.py:735: UserWarning: RNN module weights are not part of single contiguous chunk of memory. This means they need to be compacted at every call, possibly greatly increasing memory usage. To compact weights again call flatten_parameters(). (Triggered internally at  /pytorch/aten/src/ATen/native/cudnn/RNN.cpp:1264.)
  self.dropout, self.training, self.bidirectional, self.batch_first)
2021-06-19 16:29:12,774 Epoch: [0][199/4426]	Eit 200  lr 0.0005  Le 10.1174 (10.1397)	Time 0.964 (0.000)	Data 0.002 (0.000)	
2021-06-19 16:32:59,122 Epoch: [0][399/4426]	Eit 400  lr 0.0005  Le 10.0863 (10.1224)	Time 1.191 (0.000)	Data 0.002 (0.000)	
2021-06-19 16:36:47,273 Epoch: [0][599/4426]	Eit 600  lr 0.0005  Le 10.0667 (10.1019)	Time 1.117 (0.000)	Data 0.002 (0.000)	
2021-06-19 16:40:34,573 Epoch: [0][799/4426]	Eit 800  lr 0.0005  Le 10.0513 (10.0855)	Time 1.186 (0.000)	Data 0.002 (0.000)	
2021-06-19 16:44:22,028 Epoch: [0][999/4426]	Eit 1000  lr 0.0005  Le 10.0234 (10.0727)	Time 1.190 (0.000)	Data 0.002 (0.000)	
2021-06-19 16:48:08,832 Epoch: [0][1199/4426]	Eit 1200  lr 0.0005  Le 10.0145 (10.0620)	Time 1.119 (0.000)	Data 0.002 (0.000)	
2021-06-19 16:51:55,554 Epoch: [0][1399/4426]	Eit 1400  lr 0.0005  Le 10.0074 (10.0535)	Time 1.174 (0.000)	Data 0.002 (0.000)	
2021-06-19 16:55:41,390 Epoch: [0][1599/4426]	Eit 1600  lr 0.0005  Le 9.9957 (10.0465)	Time 0.990 (0.000)	Data 0.002 (0.000)	
2021-06-19 16:59:29,388 Epoch: [0][1799/4426]	Eit 1800  lr 0.0005  Le 10.0034 (10.0404)	Time 1.237 (0.000)	Data 0.002 (0.000)	
2021-06-19 17:03:16,531 Epoch: [0][1999/4426]	Eit 2000  lr 0.0005  Le 9.9704 (10.0349)	Time 1.184 (0.000)	Data 0.002 (0.000)	
2021-06-19 17:07:03,988 Epoch: [0][2199/4426]	Eit 2200  lr 0.0005  Le 9.9981 (10.0303)	Time 1.100 (0.000)	Data 0.002 (0.000)	
2021-06-19 17:10:51,509 Epoch: [0][2399/4426]	Eit 2400  lr 0.0005  Le 9.9728 (10.0262)	Time 1.108 (0.000)	Data 0.002 (0.000)	
2021-06-19 17:14:38,558 Epoch: [0][2599/4426]	Eit 2600  lr 0.0005  Le 9.9689 (10.0226)	Time 1.230 (0.000)	Data 0.003 (0.000)	
2021-06-19 17:18:25,736 Epoch: [0][2799/4426]	Eit 2800  lr 0.0005  Le 9.9602 (10.0192)	Time 1.059 (0.000)	Data 0.002 (0.000)	
2021-06-19 17:22:13,763 Epoch: [0][2999/4426]	Eit 3000  lr 0.0005  Le 9.9559 (10.0162)	Time 0.939 (0.000)	Data 0.002 (0.000)	
2021-06-19 17:26:02,787 Epoch: [0][3199/4426]	Eit 3200  lr 0.0005  Le 9.9584 (10.0133)	Time 1.203 (0.000)	Data 0.002 (0.000)	
2021-06-19 17:29:52,565 Epoch: [0][3399/4426]	Eit 3400  lr 0.0005  Le 9.9612 (10.0105)	Time 1.090 (0.000)	Data 0.002 (0.000)	
2021-06-19 17:33:39,081 Epoch: [0][3599/4426]	Eit 3600  lr 0.0005  Le 9.9728 (10.0080)	Time 1.126 (0.000)	Data 0.002 (0.000)	
2021-06-19 17:37:26,926 Epoch: [0][3799/4426]	Eit 3800  lr 0.0005  Le 9.9642 (10.0057)	Time 1.060 (0.000)	Data 0.002 (0.000)	
2021-06-19 17:41:13,572 Epoch: [0][3999/4426]	Eit 4000  lr 0.0005  Le 9.9706 (10.0035)	Time 1.139 (0.000)	Data 0.002 (0.000)	
2021-06-19 17:45:00,559 Epoch: [0][4199/4426]	Eit 4200  lr 0.0005  Le 9.9646 (10.0014)	Time 1.143 (0.000)	Data 0.002 (0.000)	
2021-06-19 17:48:32,867 Epoch: [0][4399/4426]	Eit 4400  lr 0.0005  Le 9.9888 (9.9995)	Time 1.145 (0.000)	Data 0.002 (0.000)	
2021-06-19 17:49:10,289 Test: [0/40]	Le 10.1841 (10.1841)	Time 7.894 (0.000)	
2021-06-19 17:49:30,027 calculate similarity time: 0.06881022453308105
2021-06-19 17:49:30,508 Image to text: 73.4, 95.0, 98.2, 1.0, 2.4
2021-06-19 17:49:30,818 Text to image: 59.1, 88.4, 95.4, 1.0, 3.9
2021-06-19 17:49:30,819 Current rsum is 509.40000000000003
2021-06-19 17:49:33,089 runs/coco_butd_region_bert/log
2021-06-19 17:49:33,089 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-19 17:49:33,090 image encoder trainable parameters: 3688352
2021-06-19 17:49:33,094 txt encoder trainable parameters: 120517280
2021-06-19 17:52:59,455 Epoch: [1][174/4426]	Eit 4600  lr 0.0005  Le 9.9424 (9.9510)	Time 1.168 (0.000)	Data 0.002 (0.000)	
2021-06-19 17:56:46,188 Epoch: [1][374/4426]	Eit 4800  lr 0.0005  Le 9.9273 (9.9516)	Time 1.151 (0.000)	Data 0.002 (0.000)	
2021-06-19 18:00:30,959 Epoch: [1][574/4426]	Eit 5000  lr 0.0005  Le 9.9560 (9.9506)	Time 1.323 (0.000)	Data 0.002 (0.000)	
2021-06-19 18:04:19,364 Epoch: [1][774/4426]	Eit 5200  lr 0.0005  Le 9.9611 (9.9499)	Time 1.209 (0.000)	Data 0.002 (0.000)	
2021-06-19 18:08:04,093 Epoch: [1][974/4426]	Eit 5400  lr 0.0005  Le 9.9241 (9.9493)	Time 1.210 (0.000)	Data 0.002 (0.000)	
2021-06-19 18:11:51,668 Epoch: [1][1174/4426]	Eit 5600  lr 0.0005  Le 9.9617 (9.9487)	Time 1.086 (0.000)	Data 0.002 (0.000)	
2021-06-19 18:15:37,101 Epoch: [1][1374/4426]	Eit 5800  lr 0.0005  Le 9.9786 (9.9482)	Time 1.180 (0.000)	Data 0.002 (0.000)	
2021-06-19 18:19:23,843 Epoch: [1][1574/4426]	Eit 6000  lr 0.0005  Le 9.9484 (9.9480)	Time 1.167 (0.000)	Data 0.002 (0.000)	
2021-06-19 18:23:10,796 Epoch: [1][1774/4426]	Eit 6200  lr 0.0005  Le 9.9205 (9.9474)	Time 1.213 (0.000)	Data 0.002 (0.000)	
2021-06-19 18:26:57,904 Epoch: [1][1974/4426]	Eit 6400  lr 0.0005  Le 9.9522 (9.9470)	Time 1.271 (0.000)	Data 0.002 (0.000)	
2021-06-19 18:30:42,776 Epoch: [1][2174/4426]	Eit 6600  lr 0.0005  Le 9.9535 (9.9462)	Time 1.142 (0.000)	Data 0.002 (0.000)	
2021-06-19 18:34:28,940 Epoch: [1][2374/4426]	Eit 6800  lr 0.0005  Le 9.9306 (9.9456)	Time 1.069 (0.000)	Data 0.002 (0.000)	
2021-06-19 18:38:17,123 Epoch: [1][2574/4426]	Eit 7000  lr 0.0005  Le 9.9191 (9.9453)	Time 1.148 (0.000)	Data 0.002 (0.000)	
2021-06-19 18:42:04,790 Epoch: [1][2774/4426]	Eit 7200  lr 0.0005  Le 9.9515 (9.9450)	Time 1.034 (0.000)	Data 0.002 (0.000)	
2021-06-19 18:45:51,877 Epoch: [1][2974/4426]	Eit 7400  lr 0.0005  Le 9.9204 (9.9446)	Time 1.093 (0.000)	Data 0.002 (0.000)	
2021-06-19 18:49:36,096 Epoch: [1][3174/4426]	Eit 7600  lr 0.0005  Le 9.9541 (9.9443)	Time 1.201 (0.000)	Data 0.002 (0.000)	
2021-06-19 18:53:23,448 Epoch: [1][3374/4426]	Eit 7800  lr 0.0005  Le 9.8832 (9.9438)	Time 1.053 (0.000)	Data 0.002 (0.000)	
2021-06-19 18:57:09,137 Epoch: [1][3574/4426]	Eit 8000  lr 0.0005  Le 9.9743 (9.9436)	Time 1.129 (0.000)	Data 0.002 (0.000)	
2021-06-19 19:00:55,955 Epoch: [1][3774/4426]	Eit 8200  lr 0.0005  Le 9.9503 (9.9432)	Time 1.152 (0.000)	Data 0.002 (0.000)	
2021-06-19 19:04:41,602 Epoch: [1][3974/4426]	Eit 8400  lr 0.0005  Le 9.9456 (9.9430)	Time 1.155 (0.000)	Data 0.002 (0.000)	
2021-06-19 19:08:29,472 Epoch: [1][4174/4426]	Eit 8600  lr 0.0005  Le 9.9775 (9.9426)	Time 1.195 (0.000)	Data 0.002 (0.000)	
2021-06-19 19:12:15,169 Epoch: [1][4374/4426]	Eit 8800  lr 0.0005  Le 9.9223 (9.9422)	Time 1.161 (0.000)	Data 0.002 (0.000)	
2021-06-19 19:13:20,048 Test: [0/40]	Le 10.1883 (10.1883)	Time 7.839 (0.000)	
2021-06-19 19:13:37,142 calculate similarity time: 0.06729412078857422
2021-06-19 19:13:37,693 Image to text: 75.2, 95.9, 99.3, 1.0, 2.8
2021-06-19 19:13:38,052 Text to image: 61.4, 90.5, 96.2, 1.0, 4.0
2021-06-19 19:13:38,053 Current rsum is 518.52
2021-06-19 19:13:41,170 runs/coco_butd_region_bert/log
2021-06-19 19:13:41,170 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-19 19:13:41,171 image encoder trainable parameters: 3688352
2021-06-19 19:13:41,177 txt encoder trainable parameters: 120517280
2021-06-19 19:16:29,967 Epoch: [2][149/4426]	Eit 9000  lr 0.0005  Le 9.8998 (9.9242)	Time 1.270 (0.000)	Data 0.002 (0.000)	
2021-06-19 19:20:16,980 Epoch: [2][349/4426]	Eit 9200  lr 0.0005  Le 9.9168 (9.9271)	Time 0.864 (0.000)	Data 0.002 (0.000)	
2021-06-19 19:24:03,192 Epoch: [2][549/4426]	Eit 9400  lr 0.0005  Le 9.9666 (9.9269)	Time 1.182 (0.000)	Data 0.002 (0.000)	
2021-06-19 19:27:50,624 Epoch: [2][749/4426]	Eit 9600  lr 0.0005  Le 9.9321 (9.9260)	Time 1.103 (0.000)	Data 0.002 (0.000)	
2021-06-19 19:31:34,433 Epoch: [2][949/4426]	Eit 9800  lr 0.0005  Le 9.9044 (9.9254)	Time 0.902 (0.000)	Data 0.002 (0.000)	
2021-06-19 19:35:23,123 Epoch: [2][1149/4426]	Eit 10000  lr 0.0005  Le 9.9469 (9.9253)	Time 1.146 (0.000)	Data 0.002 (0.000)	
2021-06-19 19:39:10,441 Epoch: [2][1349/4426]	Eit 10200  lr 0.0005  Le 9.9567 (9.9253)	Time 1.067 (0.000)	Data 0.002 (0.000)	
2021-06-19 19:42:57,886 Epoch: [2][1549/4426]	Eit 10400  lr 0.0005  Le 9.9048 (9.9253)	Time 1.333 (0.000)	Data 0.007 (0.000)	
2021-06-19 19:46:43,831 Epoch: [2][1749/4426]	Eit 10600  lr 0.0005  Le 9.9204 (9.9254)	Time 1.099 (0.000)	Data 0.002 (0.000)	
2021-06-19 19:50:30,302 Epoch: [2][1949/4426]	Eit 10800  lr 0.0005  Le 9.9190 (9.9251)	Time 1.124 (0.000)	Data 0.002 (0.000)	
2021-06-19 19:54:17,862 Epoch: [2][2149/4426]	Eit 11000  lr 0.0005  Le 9.9272 (9.9251)	Time 1.200 (0.000)	Data 0.002 (0.000)	
2021-06-19 19:58:05,835 Epoch: [2][2349/4426]	Eit 11200  lr 0.0005  Le 9.9662 (9.9251)	Time 1.571 (0.000)	Data 0.002 (0.000)	
2021-06-19 20:01:51,200 Epoch: [2][2549/4426]	Eit 11400  lr 0.0005  Le 9.9252 (9.9251)	Time 1.134 (0.000)	Data 0.002 (0.000)	
2021-06-19 20:05:38,607 Epoch: [2][2749/4426]	Eit 11600  lr 0.0005  Le 9.9287 (9.9249)	Time 1.150 (0.000)	Data 0.002 (0.000)	
2021-06-19 20:09:23,168 Epoch: [2][2949/4426]	Eit 11800  lr 0.0005  Le 9.9467 (9.9246)	Time 1.010 (0.000)	Data 0.002 (0.000)	
2021-06-19 20:13:07,095 Epoch: [2][3149/4426]	Eit 12000  lr 0.0005  Le 9.9556 (9.9245)	Time 1.210 (0.000)	Data 0.002 (0.000)	
2021-06-19 20:16:53,829 Epoch: [2][3349/4426]	Eit 12200  lr 0.0005  Le 9.9559 (9.9244)	Time 1.128 (0.000)	Data 0.002 (0.000)	
2021-06-19 20:20:40,019 Epoch: [2][3549/4426]	Eit 12400  lr 0.0005  Le 9.9075 (9.9242)	Time 1.116 (0.000)	Data 0.002 (0.000)	
2021-06-19 20:24:26,454 Epoch: [2][3749/4426]	Eit 12600  lr 0.0005  Le 9.8956 (9.9240)	Time 1.179 (0.000)	Data 0.002 (0.000)	
2021-06-19 20:28:16,268 Epoch: [2][3949/4426]	Eit 12800  lr 0.0005  Le 9.9227 (9.9240)	Time 1.139 (0.000)	Data 0.002 (0.000)	
2021-06-19 20:32:03,275 Epoch: [2][4149/4426]	Eit 13000  lr 0.0005  Le 9.9057 (9.9238)	Time 1.061 (0.000)	Data 0.002 (0.000)	
2021-06-19 20:35:47,431 Epoch: [2][4349/4426]	Eit 13200  lr 0.0005  Le 9.9098 (9.9236)	Time 1.174 (0.000)	Data 0.002 (0.000)	
2021-06-19 20:37:20,521 Test: [0/40]	Le 10.1833 (10.1833)	Time 7.821 (0.000)	
2021-06-19 20:37:40,178 calculate similarity time: 0.06316113471984863
2021-06-19 20:37:40,676 Image to text: 78.4, 96.8, 99.2, 1.0, 2.5
2021-06-19 20:37:40,992 Text to image: 63.0, 91.2, 96.4, 1.0, 3.7
2021-06-19 20:37:40,992 Current rsum is 524.9599999999999
2021-06-19 20:37:43,906 runs/coco_butd_region_bert/log
2021-06-19 20:37:43,906 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-19 20:37:43,907 image encoder trainable parameters: 3688352
2021-06-19 20:37:43,917 txt encoder trainable parameters: 120517280
2021-06-19 20:40:10,777 Epoch: [3][124/4426]	Eit 13400  lr 0.0005  Le 9.9061 (9.9119)	Time 1.143 (0.000)	Data 0.002 (0.000)	
2021-06-19 20:43:42,798 Epoch: [3][324/4426]	Eit 13600  lr 0.0005  Le 9.9006 (9.9130)	Time 0.960 (0.000)	Data 0.002 (0.000)	
2021-06-19 20:47:28,818 Epoch: [3][524/4426]	Eit 13800  lr 0.0005  Le 9.9165 (9.9131)	Time 1.186 (0.000)	Data 0.002 (0.000)	
2021-06-19 20:51:13,524 Epoch: [3][724/4426]	Eit 14000  lr 0.0005  Le 9.8949 (9.9128)	Time 1.093 (0.000)	Data 0.002 (0.000)	
2021-06-19 20:54:57,258 Epoch: [3][924/4426]	Eit 14200  lr 0.0005  Le 9.9164 (9.9126)	Time 1.125 (0.000)	Data 0.002 (0.000)	
2021-06-19 20:58:43,779 Epoch: [3][1124/4426]	Eit 14400  lr 0.0005  Le 9.9044 (9.9133)	Time 1.141 (0.000)	Data 0.002 (0.000)	
2021-06-19 21:02:30,564 Epoch: [3][1324/4426]	Eit 14600  lr 0.0005  Le 9.9100 (9.9136)	Time 1.233 (0.000)	Data 0.002 (0.000)	
2021-06-19 21:06:16,438 Epoch: [3][1524/4426]	Eit 14800  lr 0.0005  Le 9.9274 (9.9134)	Time 1.155 (0.000)	Data 0.002 (0.000)	
2021-06-19 21:10:01,693 Epoch: [3][1724/4426]	Eit 15000  lr 0.0005  Le 9.9309 (9.9133)	Time 1.104 (0.000)	Data 0.002 (0.000)	
2021-06-19 21:13:49,662 Epoch: [3][1924/4426]	Eit 15200  lr 0.0005  Le 9.8970 (9.9132)	Time 1.217 (0.000)	Data 0.002 (0.000)	
2021-06-19 21:17:34,803 Epoch: [3][2124/4426]	Eit 15400  lr 0.0005  Le 9.9051 (9.9130)	Time 1.211 (0.000)	Data 0.002 (0.000)	
2021-06-19 21:21:21,165 Epoch: [3][2324/4426]	Eit 15600  lr 0.0005  Le 9.8813 (9.9128)	Time 0.938 (0.000)	Data 0.002 (0.000)	
2021-06-19 21:25:07,322 Epoch: [3][2524/4426]	Eit 15800  lr 0.0005  Le 9.9148 (9.9127)	Time 0.931 (0.000)	Data 0.002 (0.000)	
2021-06-19 21:28:54,303 Epoch: [3][2724/4426]	Eit 16000  lr 0.0005  Le 9.9056 (9.9128)	Time 1.095 (0.000)	Data 0.002 (0.000)	
2021-06-19 21:32:41,579 Epoch: [3][2924/4426]	Eit 16200  lr 0.0005  Le 9.9017 (9.9127)	Time 1.154 (0.000)	Data 0.002 (0.000)	
2021-06-19 21:36:25,399 Epoch: [3][3124/4426]	Eit 16400  lr 0.0005  Le 9.9073 (9.9126)	Time 1.094 (0.000)	Data 0.002 (0.000)	
2021-06-19 21:40:10,623 Epoch: [3][3324/4426]	Eit 16600  lr 0.0005  Le 9.8742 (9.9125)	Time 1.083 (0.000)	Data 0.002 (0.000)	
2021-06-19 21:43:57,698 Epoch: [3][3524/4426]	Eit 16800  lr 0.0005  Le 9.9098 (9.9124)	Time 1.223 (0.000)	Data 0.009 (0.000)	
2021-06-19 21:47:44,105 Epoch: [3][3724/4426]	Eit 17000  lr 0.0005  Le 9.9270 (9.9122)	Time 1.208 (0.000)	Data 0.002 (0.000)	
2021-06-19 21:51:28,400 Epoch: [3][3924/4426]	Eit 17200  lr 0.0005  Le 9.9027 (9.9123)	Time 1.153 (0.000)	Data 0.002 (0.000)	
2021-06-19 21:55:15,702 Epoch: [3][4124/4426]	Eit 17400  lr 0.0005  Le 9.9012 (9.9122)	Time 1.128 (0.000)	Data 0.002 (0.000)	
2021-06-19 21:59:01,823 Epoch: [3][4324/4426]	Eit 17600  lr 0.0005  Le 9.9212 (9.9121)	Time 1.179 (0.000)	Data 0.002 (0.000)	
2021-06-19 22:01:03,049 Test: [0/40]	Le 10.1871 (10.1871)	Time 8.086 (0.000)	
2021-06-19 22:01:22,512 calculate similarity time: 0.06892585754394531
2021-06-19 22:01:23,063 Image to text: 78.4, 97.2, 99.3, 1.0, 1.9
2021-06-19 22:01:23,376 Text to image: 63.8, 91.5, 96.4, 1.0, 3.5
2021-06-19 22:01:23,376 Current rsum is 526.6200000000001
2021-06-19 22:01:26,206 runs/coco_butd_region_bert/log
2021-06-19 22:01:26,207 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-19 22:01:26,208 image encoder trainable parameters: 3688352
2021-06-19 22:01:26,218 txt encoder trainable parameters: 120517280
2021-06-19 22:03:28,118 Epoch: [4][99/4426]	Eit 17800  lr 0.0005  Le 9.9069 (9.9008)	Time 1.122 (0.000)	Data 0.002 (0.000)	
2021-06-19 22:07:16,148 Epoch: [4][299/4426]	Eit 18000  lr 0.0005  Le 9.8755 (9.8999)	Time 1.265 (0.000)	Data 0.002 (0.000)	
2021-06-19 22:10:44,728 Epoch: [4][499/4426]	Eit 18200  lr 0.0005  Le 9.8897 (9.9009)	Time 1.143 (0.000)	Data 0.002 (0.000)	
2021-06-19 22:14:33,864 Epoch: [4][699/4426]	Eit 18400  lr 0.0005  Le 9.9266 (9.9017)	Time 1.168 (0.000)	Data 0.002 (0.000)	
2021-06-19 22:18:20,622 Epoch: [4][899/4426]	Eit 18600  lr 0.0005  Le 9.8867 (9.9022)	Time 1.178 (0.000)	Data 0.002 (0.000)	
2021-06-19 22:22:05,148 Epoch: [4][1099/4426]	Eit 18800  lr 0.0005  Le 9.9264 (9.9028)	Time 1.159 (0.000)	Data 0.002 (0.000)	
2021-06-19 22:25:53,929 Epoch: [4][1299/4426]	Eit 19000  lr 0.0005  Le 9.9012 (9.9030)	Time 1.163 (0.000)	Data 0.002 (0.000)	
2021-06-19 22:29:40,056 Epoch: [4][1499/4426]	Eit 19200  lr 0.0005  Le 9.8859 (9.9029)	Time 1.099 (0.000)	Data 0.002 (0.000)	
2021-06-19 22:33:25,997 Epoch: [4][1699/4426]	Eit 19400  lr 0.0005  Le 9.8869 (9.9027)	Time 1.170 (0.000)	Data 0.002 (0.000)	
2021-06-19 22:37:12,982 Epoch: [4][1899/4426]	Eit 19600  lr 0.0005  Le 9.9305 (9.9029)	Time 0.847 (0.000)	Data 0.002 (0.000)	
2021-06-19 22:41:01,760 Epoch: [4][2099/4426]	Eit 19800  lr 0.0005  Le 9.9294 (9.9029)	Time 1.194 (0.000)	Data 0.002 (0.000)	
2021-06-19 22:44:48,304 Epoch: [4][2299/4426]	Eit 20000  lr 0.0005  Le 9.8854 (9.9031)	Time 1.172 (0.000)	Data 0.014 (0.000)	
2021-06-19 22:48:34,680 Epoch: [4][2499/4426]	Eit 20200  lr 0.0005  Le 9.9189 (9.9032)	Time 1.104 (0.000)	Data 0.002 (0.000)	
2021-06-19 22:52:22,662 Epoch: [4][2699/4426]	Eit 20400  lr 0.0005  Le 9.9120 (9.9034)	Time 1.150 (0.000)	Data 0.002 (0.000)	
2021-06-19 22:56:09,030 Epoch: [4][2899/4426]	Eit 20600  lr 0.0005  Le 9.8837 (9.9036)	Time 1.139 (0.000)	Data 0.002 (0.000)	
2021-06-19 22:59:53,594 Epoch: [4][3099/4426]	Eit 20800  lr 0.0005  Le 9.9066 (9.9035)	Time 1.162 (0.000)	Data 0.005 (0.000)	
2021-06-19 23:03:38,312 Epoch: [4][3299/4426]	Eit 21000  lr 0.0005  Le 9.9269 (9.9035)	Time 1.168 (0.000)	Data 0.002 (0.000)	
2021-06-19 23:07:27,955 Epoch: [4][3499/4426]	Eit 21200  lr 0.0005  Le 9.9090 (9.9036)	Time 1.114 (0.000)	Data 0.002 (0.000)	
2021-06-19 23:11:13,412 Epoch: [4][3699/4426]	Eit 21400  lr 0.0005  Le 9.8789 (9.9035)	Time 1.206 (0.000)	Data 0.002 (0.000)	
2021-06-19 23:15:00,516 Epoch: [4][3899/4426]	Eit 21600  lr 0.0005  Le 9.8979 (9.9037)	Time 1.202 (0.000)	Data 0.002 (0.000)	
2021-06-19 23:18:47,311 Epoch: [4][4099/4426]	Eit 21800  lr 0.0005  Le 9.9331 (9.9036)	Time 1.209 (0.000)	Data 0.002 (0.000)	
2021-06-19 23:22:32,679 Epoch: [4][4299/4426]	Eit 22000  lr 0.0005  Le 9.8874 (9.9036)	Time 1.307 (0.000)	Data 0.002 (0.000)	
2021-06-19 23:25:02,398 Test: [0/40]	Le 10.1892 (10.1892)	Time 7.998 (0.000)	
2021-06-19 23:25:22,098 calculate similarity time: 0.0656428337097168
2021-06-19 23:25:22,476 Image to text: 80.2, 97.2, 99.1, 1.0, 1.9
2021-06-19 23:25:22,792 Text to image: 65.4, 92.0, 96.8, 1.0, 3.6
2021-06-19 23:25:22,792 Current rsum is 530.66
2021-06-19 23:25:25,663 runs/coco_butd_region_bert/log
2021-06-19 23:25:25,663 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-19 23:25:25,665 image encoder trainable parameters: 3688352
2021-06-19 23:25:25,674 txt encoder trainable parameters: 120517280
2021-06-19 23:26:57,922 Epoch: [5][74/4426]	Eit 22200  lr 0.0005  Le 9.8952 (9.8946)	Time 1.207 (0.000)	Data 0.002 (0.000)	
2021-06-19 23:30:43,832 Epoch: [5][274/4426]	Eit 22400  lr 0.0005  Le 9.8978 (9.8938)	Time 1.126 (0.000)	Data 0.002 (0.000)	
2021-06-19 23:34:29,025 Epoch: [5][474/4426]	Eit 22600  lr 0.0005  Le 9.8498 (9.8943)	Time 1.171 (0.000)	Data 0.002 (0.000)	
2021-06-19 23:38:12,804 Epoch: [5][674/4426]	Eit 22800  lr 0.0005  Le 9.9065 (9.8938)	Time 0.798 (0.000)	Data 0.002 (0.000)	
2021-06-19 23:41:45,640 Epoch: [5][874/4426]	Eit 23000  lr 0.0005  Le 9.9033 (9.8945)	Time 1.160 (0.000)	Data 0.002 (0.000)	
2021-06-19 23:45:34,802 Epoch: [5][1074/4426]	Eit 23200  lr 0.0005  Le 9.9121 (9.8947)	Time 1.168 (0.000)	Data 0.002 (0.000)	
2021-06-19 23:49:21,321 Epoch: [5][1274/4426]	Eit 23400  lr 0.0005  Le 9.8914 (9.8950)	Time 1.111 (0.000)	Data 0.002 (0.000)	
2021-06-19 23:53:07,781 Epoch: [5][1474/4426]	Eit 23600  lr 0.0005  Le 9.9030 (9.8950)	Time 1.102 (0.000)	Data 0.002 (0.000)	
2021-06-19 23:56:51,837 Epoch: [5][1674/4426]	Eit 23800  lr 0.0005  Le 9.8875 (9.8950)	Time 1.129 (0.000)	Data 0.002 (0.000)	
2021-06-20 00:00:39,338 Epoch: [5][1874/4426]	Eit 24000  lr 0.0005  Le 9.8957 (9.8949)	Time 1.118 (0.000)	Data 0.002 (0.000)	
2021-06-20 00:04:24,885 Epoch: [5][2074/4426]	Eit 24200  lr 0.0005  Le 9.8650 (9.8951)	Time 1.138 (0.000)	Data 0.002 (0.000)	
2021-06-20 00:08:10,585 Epoch: [5][2274/4426]	Eit 24400  lr 0.0005  Le 9.9125 (9.8951)	Time 1.286 (0.000)	Data 0.002 (0.000)	
2021-06-20 00:11:58,320 Epoch: [5][2474/4426]	Eit 24600  lr 0.0005  Le 9.8933 (9.8953)	Time 1.298 (0.000)	Data 0.002 (0.000)	
2021-06-20 00:15:43,364 Epoch: [5][2674/4426]	Eit 24800  lr 0.0005  Le 9.8901 (9.8953)	Time 1.154 (0.000)	Data 0.002 (0.000)	
2021-06-20 00:19:28,834 Epoch: [5][2874/4426]	Eit 25000  lr 0.0005  Le 9.9171 (9.8952)	Time 1.140 (0.000)	Data 0.002 (0.000)	
2021-06-20 00:23:16,859 Epoch: [5][3074/4426]	Eit 25200  lr 0.0005  Le 9.9023 (9.8952)	Time 1.160 (0.000)	Data 0.002 (0.000)	
2021-06-20 00:27:03,706 Epoch: [5][3274/4426]	Eit 25400  lr 0.0005  Le 9.8831 (9.8954)	Time 1.104 (0.000)	Data 0.002 (0.000)	
2021-06-20 00:30:48,209 Epoch: [5][3474/4426]	Eit 25600  lr 0.0005  Le 9.9040 (9.8955)	Time 1.149 (0.000)	Data 0.002 (0.000)	
2021-06-20 00:34:36,366 Epoch: [5][3674/4426]	Eit 25800  lr 0.0005  Le 9.8973 (9.8956)	Time 1.133 (0.000)	Data 0.002 (0.000)	
2021-06-20 00:38:25,196 Epoch: [5][3874/4426]	Eit 26000  lr 0.0005  Le 9.8564 (9.8956)	Time 1.170 (0.000)	Data 0.002 (0.000)	
2021-06-20 00:42:11,372 Epoch: [5][4074/4426]	Eit 26200  lr 0.0005  Le 9.8998 (9.8956)	Time 1.093 (0.000)	Data 0.002 (0.000)	
2021-06-20 00:46:00,243 Epoch: [5][4274/4426]	Eit 26400  lr 0.0005  Le 9.9043 (9.8957)	Time 1.174 (0.000)	Data 0.002 (0.000)	
2021-06-20 00:48:57,566 Test: [0/40]	Le 10.1881 (10.1881)	Time 8.360 (0.000)	
2021-06-20 00:49:17,223 calculate similarity time: 0.05665087699890137
2021-06-20 00:49:17,605 Image to text: 79.7, 97.3, 99.7, 1.0, 2.0
2021-06-20 00:49:17,919 Text to image: 65.4, 92.0, 96.7, 1.0, 3.7
2021-06-20 00:49:17,919 Current rsum is 530.8199999999999
2021-06-20 00:49:20,928 runs/coco_butd_region_bert/log
2021-06-20 00:49:20,928 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-20 00:49:20,930 image encoder trainable parameters: 3688352
2021-06-20 00:49:20,940 txt encoder trainable parameters: 120517280
2021-06-20 00:50:25,561 Epoch: [6][49/4426]	Eit 26600  lr 0.0005  Le 9.8944 (9.8843)	Time 1.193 (0.000)	Data 0.002 (0.000)	
2021-06-20 00:54:14,037 Epoch: [6][249/4426]	Eit 26800  lr 0.0005  Le 9.8851 (9.8887)	Time 1.265 (0.000)	Data 0.002 (0.000)	
2021-06-20 00:57:58,621 Epoch: [6][449/4426]	Eit 27000  lr 0.0005  Le 9.9061 (9.8889)	Time 1.228 (0.000)	Data 0.002 (0.000)	
2021-06-20 01:01:42,640 Epoch: [6][649/4426]	Eit 27200  lr 0.0005  Le 9.8937 (9.8887)	Time 1.186 (0.000)	Data 0.001 (0.000)	
2021-06-20 01:05:28,868 Epoch: [6][849/4426]	Eit 27400  lr 0.0005  Le 9.9287 (9.8894)	Time 0.913 (0.000)	Data 0.002 (0.000)	
2021-06-20 01:08:59,115 Epoch: [6][1049/4426]	Eit 27600  lr 0.0005  Le 9.8704 (9.8893)	Time 1.173 (0.000)	Data 0.002 (0.000)	
2021-06-20 01:12:46,009 Epoch: [6][1249/4426]	Eit 27800  lr 0.0005  Le 9.8616 (9.8893)	Time 1.217 (0.000)	Data 0.002 (0.000)	
2021-06-20 01:16:34,106 Epoch: [6][1449/4426]	Eit 28000  lr 0.0005  Le 9.8669 (9.8893)	Time 1.022 (0.000)	Data 0.002 (0.000)	
2021-06-20 01:20:20,145 Epoch: [6][1649/4426]	Eit 28200  lr 0.0005  Le 9.8764 (9.8892)	Time 1.019 (0.000)	Data 0.002 (0.000)	
2021-06-20 01:24:06,376 Epoch: [6][1849/4426]	Eit 28400  lr 0.0005  Le 9.8822 (9.8891)	Time 1.142 (0.000)	Data 0.002 (0.000)	
2021-06-20 01:27:50,806 Epoch: [6][2049/4426]	Eit 28600  lr 0.0005  Le 9.9240 (9.8890)	Time 1.141 (0.000)	Data 0.002 (0.000)	
2021-06-20 01:31:35,583 Epoch: [6][2249/4426]	Eit 28800  lr 0.0005  Le 9.8810 (9.8890)	Time 1.114 (0.000)	Data 0.002 (0.000)	
2021-06-20 01:35:22,758 Epoch: [6][2449/4426]	Eit 29000  lr 0.0005  Le 9.8713 (9.8892)	Time 1.246 (0.000)	Data 0.002 (0.000)	
2021-06-20 01:39:08,514 Epoch: [6][2649/4426]	Eit 29200  lr 0.0005  Le 9.8478 (9.8893)	Time 1.153 (0.000)	Data 0.002 (0.000)	
2021-06-20 01:42:55,079 Epoch: [6][2849/4426]	Eit 29400  lr 0.0005  Le 9.8753 (9.8893)	Time 1.161 (0.000)	Data 0.002 (0.000)	
2021-06-20 01:46:41,529 Epoch: [6][3049/4426]	Eit 29600  lr 0.0005  Le 9.8590 (9.8894)	Time 0.892 (0.000)	Data 0.002 (0.000)	
2021-06-20 01:50:26,842 Epoch: [6][3249/4426]	Eit 29800  lr 0.0005  Le 9.9050 (9.8893)	Time 1.175 (0.000)	Data 0.002 (0.000)	
2021-06-20 01:54:12,199 Epoch: [6][3449/4426]	Eit 30000  lr 0.0005  Le 9.9005 (9.8894)	Time 1.196 (0.000)	Data 0.002 (0.000)	
2021-06-20 01:57:58,238 Epoch: [6][3649/4426]	Eit 30200  lr 0.0005  Le 9.8871 (9.8894)	Time 1.142 (0.000)	Data 0.002 (0.000)	
2021-06-20 02:01:44,184 Epoch: [6][3849/4426]	Eit 30400  lr 0.0005  Le 9.9031 (9.8895)	Time 1.099 (0.000)	Data 0.002 (0.000)	
2021-06-20 02:05:28,037 Epoch: [6][4049/4426]	Eit 30600  lr 0.0005  Le 9.8963 (9.8895)	Time 1.153 (0.000)	Data 0.002 (0.000)	
2021-06-20 02:09:14,653 Epoch: [6][4249/4426]	Eit 30800  lr 0.0005  Le 9.9047 (9.8895)	Time 1.130 (0.000)	Data 0.002 (0.000)	
2021-06-20 02:12:41,240 Test: [0/40]	Le 10.1969 (10.1969)	Time 8.039 (0.000)	
2021-06-20 02:13:00,759 calculate similarity time: 0.07480120658874512
2021-06-20 02:13:01,291 Image to text: 81.6, 98.3, 99.4, 1.0, 1.9
2021-06-20 02:13:01,606 Text to image: 65.5, 92.3, 96.9, 1.0, 3.3
2021-06-20 02:13:01,606 Current rsum is 533.9599999999999
2021-06-20 02:13:04,639 runs/coco_butd_region_bert/log
2021-06-20 02:13:04,639 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-20 02:13:04,641 image encoder trainable parameters: 3688352
2021-06-20 02:13:04,651 txt encoder trainable parameters: 120517280
2021-06-20 02:13:40,438 Epoch: [7][24/4426]	Eit 31000  lr 0.0005  Le 9.8879 (9.8796)	Time 1.177 (0.000)	Data 0.002 (0.000)	
2021-06-20 02:17:29,053 Epoch: [7][224/4426]	Eit 31200  lr 0.0005  Le 9.8788 (9.8822)	Time 1.115 (0.000)	Data 0.002 (0.000)	
2021-06-20 02:21:18,364 Epoch: [7][424/4426]	Eit 31400  lr 0.0005  Le 9.8641 (9.8827)	Time 1.230 (0.000)	Data 0.002 (0.000)	
2021-06-20 02:25:05,378 Epoch: [7][624/4426]	Eit 31600  lr 0.0005  Le 9.8838 (9.8832)	Time 1.143 (0.000)	Data 0.002 (0.000)	
2021-06-20 02:28:50,847 Epoch: [7][824/4426]	Eit 31800  lr 0.0005  Le 9.8773 (9.8824)	Time 1.098 (0.000)	Data 0.002 (0.000)	
2021-06-20 02:32:38,735 Epoch: [7][1024/4426]	Eit 32000  lr 0.0005  Le 9.8886 (9.8824)	Time 1.170 (0.000)	Data 0.002 (0.000)	
2021-06-20 02:36:07,194 Epoch: [7][1224/4426]	Eit 32200  lr 0.0005  Le 9.8917 (9.8823)	Time 1.033 (0.000)	Data 0.002 (0.000)	
2021-06-20 02:39:54,208 Epoch: [7][1424/4426]	Eit 32400  lr 0.0005  Le 9.8682 (9.8822)	Time 1.166 (0.000)	Data 0.002 (0.000)	
2021-06-20 02:43:43,786 Epoch: [7][1624/4426]	Eit 32600  lr 0.0005  Le 9.9017 (9.8823)	Time 1.066 (0.000)	Data 0.002 (0.000)	
2021-06-20 02:47:28,688 Epoch: [7][1824/4426]	Eit 32800  lr 0.0005  Le 9.9078 (9.8824)	Time 1.122 (0.000)	Data 0.002 (0.000)	
2021-06-20 02:51:14,925 Epoch: [7][2024/4426]	Eit 33000  lr 0.0005  Le 9.9078 (9.8826)	Time 1.121 (0.000)	Data 0.002 (0.000)	
2021-06-20 02:55:00,869 Epoch: [7][2224/4426]	Eit 33200  lr 0.0005  Le 9.8926 (9.8828)	Time 0.947 (0.000)	Data 0.002 (0.000)	
2021-06-20 02:58:45,733 Epoch: [7][2424/4426]	Eit 33400  lr 0.0005  Le 9.9056 (9.8831)	Time 1.116 (0.000)	Data 0.002 (0.000)	
2021-06-20 03:02:31,954 Epoch: [7][2624/4426]	Eit 33600  lr 0.0005  Le 9.8893 (9.8833)	Time 0.956 (0.000)	Data 0.002 (0.000)	
2021-06-20 03:06:17,708 Epoch: [7][2824/4426]	Eit 33800  lr 0.0005  Le 9.8964 (9.8833)	Time 1.177 (0.000)	Data 0.002 (0.000)	
2021-06-20 03:10:03,858 Epoch: [7][3024/4426]	Eit 34000  lr 0.0005  Le 9.8643 (9.8835)	Time 0.987 (0.000)	Data 0.002 (0.000)	
2021-06-20 03:13:50,322 Epoch: [7][3224/4426]	Eit 34200  lr 0.0005  Le 9.8692 (9.8835)	Time 1.176 (0.000)	Data 0.002 (0.000)	
2021-06-20 03:17:35,396 Epoch: [7][3424/4426]	Eit 34400  lr 0.0005  Le 9.8570 (9.8837)	Time 1.084 (0.000)	Data 0.002 (0.000)	
2021-06-20 03:21:22,650 Epoch: [7][3624/4426]	Eit 34600  lr 0.0005  Le 9.9137 (9.8837)	Time 1.213 (0.000)	Data 0.002 (0.000)	
2021-06-20 03:25:08,072 Epoch: [7][3824/4426]	Eit 34800  lr 0.0005  Le 9.9108 (9.8838)	Time 1.139 (0.000)	Data 0.002 (0.000)	
2021-06-20 03:28:55,225 Epoch: [7][4024/4426]	Eit 35000  lr 0.0005  Le 9.8845 (9.8839)	Time 1.192 (0.000)	Data 0.002 (0.000)	
2021-06-20 03:32:39,949 Epoch: [7][4224/4426]	Eit 35200  lr 0.0005  Le 9.9034 (9.8840)	Time 1.170 (0.000)	Data 0.002 (0.000)	
2021-06-20 03:36:23,663 Epoch: [7][4424/4426]	Eit 35400  lr 0.0005  Le 9.8850 (9.8840)	Time 1.133 (0.000)	Data 0.002 (0.000)	
2021-06-20 03:36:32,817 Test: [0/40]	Le 10.1995 (10.1995)	Time 7.800 (0.000)	
2021-06-20 03:36:52,963 calculate similarity time: 0.08234524726867676
2021-06-20 03:36:53,370 Image to text: 81.3, 98.3, 99.6, 1.0, 2.4
2021-06-20 03:36:53,678 Text to image: 65.7, 92.4, 96.9, 1.0, 3.7
2021-06-20 03:36:53,678 Current rsum is 534.28
2021-06-20 03:36:56,743 runs/coco_butd_region_bert/log
2021-06-20 03:36:56,743 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-20 03:36:56,745 image encoder trainable parameters: 3688352
2021-06-20 03:36:56,755 txt encoder trainable parameters: 120517280
2021-06-20 03:40:48,629 Epoch: [8][199/4426]	Eit 35600  lr 0.0005  Le 9.8670 (9.8760)	Time 1.130 (0.000)	Data 0.002 (0.000)	
2021-06-20 03:44:33,480 Epoch: [8][399/4426]	Eit 35800  lr 0.0005  Le 9.8787 (9.8777)	Time 1.103 (0.000)	Data 0.002 (0.000)	
2021-06-20 03:48:17,859 Epoch: [8][599/4426]	Eit 36000  lr 0.0005  Le 9.8897 (9.8774)	Time 1.129 (0.000)	Data 0.002 (0.000)	
2021-06-20 03:52:04,524 Epoch: [8][799/4426]	Eit 36200  lr 0.0005  Le 9.8927 (9.8777)	Time 1.064 (0.000)	Data 0.002 (0.000)	
2021-06-20 03:55:50,238 Epoch: [8][999/4426]	Eit 36400  lr 0.0005  Le 9.8549 (9.8779)	Time 1.008 (0.000)	Data 0.002 (0.000)	
2021-06-20 03:59:34,718 Epoch: [8][1199/4426]	Eit 36600  lr 0.0005  Le 9.8417 (9.8778)	Time 1.209 (0.000)	Data 0.002 (0.000)	
2021-06-20 04:03:19,635 Epoch: [8][1399/4426]	Eit 36800  lr 0.0005  Le 9.8882 (9.8779)	Time 1.012 (0.000)	Data 0.002 (0.000)	
2021-06-20 04:06:48,481 Epoch: [8][1599/4426]	Eit 37000  lr 0.0005  Le 9.8756 (9.8782)	Time 1.113 (0.000)	Data 0.002 (0.000)	
2021-06-20 04:10:36,788 Epoch: [8][1799/4426]	Eit 37200  lr 0.0005  Le 9.8955 (9.8781)	Time 1.324 (0.000)	Data 0.002 (0.000)	
2021-06-20 04:14:22,870 Epoch: [8][1999/4426]	Eit 37400  lr 0.0005  Le 9.8955 (9.8782)	Time 1.118 (0.000)	Data 0.002 (0.000)	
2021-06-20 04:18:08,221 Epoch: [8][2199/4426]	Eit 37600  lr 0.0005  Le 9.8705 (9.8783)	Time 1.132 (0.000)	Data 0.002 (0.000)	
2021-06-20 04:21:55,490 Epoch: [8][2399/4426]	Eit 37800  lr 0.0005  Le 9.9025 (9.8786)	Time 1.101 (0.000)	Data 0.002 (0.000)	
2021-06-20 04:25:42,966 Epoch: [8][2599/4426]	Eit 38000  lr 0.0005  Le 9.9083 (9.8786)	Time 1.212 (0.000)	Data 0.002 (0.000)	
2021-06-20 04:29:28,217 Epoch: [8][2799/4426]	Eit 38200  lr 0.0005  Le 9.8771 (9.8787)	Time 1.255 (0.000)	Data 0.002 (0.000)	
2021-06-20 04:33:13,737 Epoch: [8][2999/4426]	Eit 38400  lr 0.0005  Le 9.8892 (9.8788)	Time 0.891 (0.000)	Data 0.002 (0.000)	
2021-06-20 04:36:57,834 Epoch: [8][3199/4426]	Eit 38600  lr 0.0005  Le 9.8608 (9.8789)	Time 1.118 (0.000)	Data 0.002 (0.000)	
2021-06-20 04:40:45,397 Epoch: [8][3399/4426]	Eit 38800  lr 0.0005  Le 9.8939 (9.8788)	Time 1.201 (0.000)	Data 0.002 (0.000)	
2021-06-20 04:44:29,707 Epoch: [8][3599/4426]	Eit 39000  lr 0.0005  Le 9.8609 (9.8789)	Time 1.141 (0.000)	Data 0.002 (0.000)	
2021-06-20 04:48:16,589 Epoch: [8][3799/4426]	Eit 39200  lr 0.0005  Le 9.8807 (9.8789)	Time 1.141 (0.000)	Data 0.002 (0.000)	
2021-06-20 04:52:03,172 Epoch: [8][3999/4426]	Eit 39400  lr 0.0005  Le 9.8546 (9.8791)	Time 1.261 (0.000)	Data 0.002 (0.000)	
2021-06-20 04:55:48,840 Epoch: [8][4199/4426]	Eit 39600  lr 0.0005  Le 9.8624 (9.8792)	Time 1.078 (0.000)	Data 0.002 (0.000)	
2021-06-20 04:59:34,169 Epoch: [8][4399/4426]	Eit 39800  lr 0.0005  Le 9.9010 (9.8792)	Time 1.110 (0.000)	Data 0.002 (0.000)	
2021-06-20 05:00:11,966 Test: [0/40]	Le 10.1918 (10.1918)	Time 8.193 (0.000)	
2021-06-20 05:00:31,710 calculate similarity time: 0.07684969902038574
2021-06-20 05:00:32,091 Image to text: 80.7, 97.5, 99.4, 1.0, 2.4
2021-06-20 05:00:32,415 Text to image: 66.3, 92.5, 96.9, 1.0, 3.4
2021-06-20 05:00:32,415 Current rsum is 533.32
2021-06-20 05:00:33,624 runs/coco_butd_region_bert/log
2021-06-20 05:00:33,624 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-20 05:00:33,625 image encoder trainable parameters: 3688352
2021-06-20 05:00:33,629 txt encoder trainable parameters: 120517280
2021-06-20 05:03:59,325 Epoch: [9][174/4426]	Eit 40000  lr 0.0005  Le 9.8667 (9.8726)	Time 1.321 (0.000)	Data 0.002 (0.000)	
2021-06-20 05:07:43,218 Epoch: [9][374/4426]	Eit 40200  lr 0.0005  Le 9.8702 (9.8722)	Time 1.123 (0.000)	Data 0.002 (0.000)	
2021-06-20 05:11:29,497 Epoch: [9][574/4426]	Eit 40400  lr 0.0005  Le 9.8719 (9.8731)	Time 1.285 (0.000)	Data 0.002 (0.000)	
2021-06-20 05:15:16,225 Epoch: [9][774/4426]	Eit 40600  lr 0.0005  Le 9.8533 (9.8726)	Time 1.167 (0.000)	Data 0.002 (0.000)	
2021-06-20 05:19:03,392 Epoch: [9][974/4426]	Eit 40800  lr 0.0005  Le 9.8799 (9.8731)	Time 1.426 (0.000)	Data 0.002 (0.000)	
2021-06-20 05:22:49,257 Epoch: [9][1174/4426]	Eit 41000  lr 0.0005  Le 9.9011 (9.8734)	Time 1.272 (0.000)	Data 0.002 (0.000)	
2021-06-20 05:26:36,021 Epoch: [9][1374/4426]	Eit 41200  lr 0.0005  Le 9.8682 (9.8736)	Time 1.197 (0.000)	Data 0.002 (0.000)	
2021-06-20 05:30:22,110 Epoch: [9][1574/4426]	Eit 41400  lr 0.0005  Le 9.9035 (9.8736)	Time 1.031 (0.000)	Data 0.002 (0.000)	
2021-06-20 05:33:51,163 Epoch: [9][1774/4426]	Eit 41600  lr 0.0005  Le 9.8760 (9.8737)	Time 1.140 (0.000)	Data 0.002 (0.000)	
2021-06-20 05:37:38,824 Epoch: [9][1974/4426]	Eit 41800  lr 0.0005  Le 9.8685 (9.8741)	Time 0.977 (0.000)	Data 0.002 (0.000)	
2021-06-20 05:41:24,697 Epoch: [9][2174/4426]	Eit 42000  lr 0.0005  Le 9.9183 (9.8744)	Time 1.102 (0.000)	Data 0.002 (0.000)	
2021-06-20 05:45:10,846 Epoch: [9][2374/4426]	Eit 42200  lr 0.0005  Le 9.8684 (9.8747)	Time 1.194 (0.000)	Data 0.002 (0.000)	
2021-06-20 05:48:58,745 Epoch: [9][2574/4426]	Eit 42400  lr 0.0005  Le 9.8574 (9.8748)	Time 1.221 (0.000)	Data 0.002 (0.000)	
2021-06-20 05:52:41,371 Epoch: [9][2774/4426]	Eit 42600  lr 0.0005  Le 9.8595 (9.8748)	Time 1.127 (0.000)	Data 0.002 (0.000)	
2021-06-20 05:56:27,234 Epoch: [9][2974/4426]	Eit 42800  lr 0.0005  Le 9.9020 (9.8748)	Time 1.039 (0.000)	Data 0.002 (0.000)	
2021-06-20 06:00:13,544 Epoch: [9][3174/4426]	Eit 43000  lr 0.0005  Le 9.8663 (9.8748)	Time 1.099 (0.000)	Data 0.002 (0.000)	
2021-06-20 06:03:59,310 Epoch: [9][3374/4426]	Eit 43200  lr 0.0005  Le 9.8818 (9.8750)	Time 1.139 (0.000)	Data 0.002 (0.000)	
2021-06-20 06:07:44,474 Epoch: [9][3574/4426]	Eit 43400  lr 0.0005  Le 9.9101 (9.8751)	Time 1.102 (0.000)	Data 0.002 (0.000)	
2021-06-20 06:11:28,840 Epoch: [9][3774/4426]	Eit 43600  lr 0.0005  Le 9.8978 (9.8753)	Time 1.108 (0.000)	Data 0.002 (0.000)	
2021-06-20 06:15:17,052 Epoch: [9][3974/4426]	Eit 43800  lr 0.0005  Le 9.8645 (9.8752)	Time 1.196 (0.000)	Data 0.002 (0.000)	
2021-06-20 06:19:03,644 Epoch: [9][4174/4426]	Eit 44000  lr 0.0005  Le 9.8961 (9.8753)	Time 1.260 (0.000)	Data 0.002 (0.000)	
2021-06-20 06:22:54,068 Epoch: [9][4374/4426]	Eit 44200  lr 0.0005  Le 9.9148 (9.8753)	Time 1.300 (0.000)	Data 0.002 (0.000)	
2021-06-20 06:23:59,966 Test: [0/40]	Le 10.1961 (10.1961)	Time 7.886 (0.000)	
2021-06-20 06:24:19,996 calculate similarity time: 0.08239555358886719
2021-06-20 06:24:20,525 Image to text: 81.2, 97.4, 99.3, 1.0, 2.0
2021-06-20 06:24:20,957 Text to image: 66.1, 92.3, 97.1, 1.0, 3.6
2021-06-20 06:24:20,957 Current rsum is 533.38
2021-06-20 06:24:22,190 runs/coco_butd_region_bert/log
2021-06-20 06:24:22,190 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-20 06:24:22,190 image encoder trainable parameters: 3688352
2021-06-20 06:24:22,195 txt encoder trainable parameters: 120517280
2021-06-20 06:27:18,557 Epoch: [10][149/4426]	Eit 44400  lr 0.0005  Le 9.8742 (9.8696)	Time 1.178 (0.000)	Data 0.002 (0.000)	
2021-06-20 06:31:04,871 Epoch: [10][349/4426]	Eit 44600  lr 0.0005  Le 9.8791 (9.8694)	Time 1.174 (0.000)	Data 0.002 (0.000)	
2021-06-20 06:34:48,236 Epoch: [10][549/4426]	Eit 44800  lr 0.0005  Le 9.8546 (9.8706)	Time 1.408 (0.000)	Data 0.002 (0.000)	
2021-06-20 06:38:34,798 Epoch: [10][749/4426]	Eit 45000  lr 0.0005  Le 9.8613 (9.8710)	Time 1.145 (0.000)	Data 0.002 (0.000)	
2021-06-20 06:42:22,201 Epoch: [10][949/4426]	Eit 45200  lr 0.0005  Le 9.8824 (9.8710)	Time 1.220 (0.000)	Data 0.002 (0.000)	
2021-06-20 06:46:08,974 Epoch: [10][1149/4426]	Eit 45400  lr 0.0005  Le 9.8731 (9.8708)	Time 1.102 (0.000)	Data 0.002 (0.000)	
2021-06-20 06:49:58,338 Epoch: [10][1349/4426]	Eit 45600  lr 0.0005  Le 9.8814 (9.8712)	Time 1.189 (0.000)	Data 0.002 (0.000)	
2021-06-20 06:53:45,433 Epoch: [10][1549/4426]	Eit 45800  lr 0.0005  Le 9.8775 (9.8710)	Time 1.015 (0.000)	Data 0.002 (0.000)	
2021-06-20 06:57:31,994 Epoch: [10][1749/4426]	Eit 46000  lr 0.0005  Le 9.8630 (9.8709)	Time 1.275 (0.000)	Data 0.002 (0.000)	
2021-06-20 07:00:59,118 Epoch: [10][1949/4426]	Eit 46200  lr 0.0005  Le 9.8528 (9.8711)	Time 1.175 (0.000)	Data 0.001 (0.000)	
2021-06-20 07:04:49,568 Epoch: [10][2149/4426]	Eit 46400  lr 0.0005  Le 9.8389 (9.8711)	Time 1.243 (0.000)	Data 0.002 (0.000)	
2021-06-20 07:08:36,665 Epoch: [10][2349/4426]	Eit 46600  lr 0.0005  Le 9.8892 (9.8713)	Time 1.044 (0.000)	Data 0.002 (0.000)	
2021-06-20 07:12:24,734 Epoch: [10][2549/4426]	Eit 46800  lr 0.0005  Le 9.8893 (9.8713)	Time 1.166 (0.000)	Data 0.002 (0.000)	
2021-06-20 07:16:12,713 Epoch: [10][2749/4426]	Eit 47000  lr 0.0005  Le 9.8882 (9.8713)	Time 1.032 (0.000)	Data 0.002 (0.000)	
2021-06-20 07:19:59,392 Epoch: [10][2949/4426]	Eit 47200  lr 0.0005  Le 9.8441 (9.8714)	Time 1.074 (0.000)	Data 0.002 (0.000)	
2021-06-20 07:23:45,193 Epoch: [10][3149/4426]	Eit 47400  lr 0.0005  Le 9.8923 (9.8714)	Time 1.111 (0.000)	Data 0.002 (0.000)	
2021-06-20 07:27:29,959 Epoch: [10][3349/4426]	Eit 47600  lr 0.0005  Le 9.8639 (9.8714)	Time 1.162 (0.000)	Data 0.002 (0.000)	
2021-06-20 07:31:18,054 Epoch: [10][3549/4426]	Eit 47800  lr 0.0005  Le 9.8710 (9.8714)	Time 1.192 (0.000)	Data 0.002 (0.000)	
2021-06-20 07:35:05,351 Epoch: [10][3749/4426]	Eit 48000  lr 0.0005  Le 9.8567 (9.8715)	Time 1.170 (0.000)	Data 0.002 (0.000)	
2021-06-20 07:38:52,833 Epoch: [10][3949/4426]	Eit 48200  lr 0.0005  Le 9.8462 (9.8715)	Time 1.145 (0.000)	Data 0.002 (0.000)	
2021-06-20 07:42:38,102 Epoch: [10][4149/4426]	Eit 48400  lr 0.0005  Le 9.8718 (9.8716)	Time 0.863 (0.000)	Data 0.002 (0.000)	
2021-06-20 07:46:25,348 Epoch: [10][4349/4426]	Eit 48600  lr 0.0005  Le 9.8967 (9.8717)	Time 1.125 (0.000)	Data 0.002 (0.000)	
2021-06-20 07:48:00,037 Test: [0/40]	Le 10.1981 (10.1981)	Time 8.157 (0.000)	
2021-06-20 07:48:19,882 calculate similarity time: 0.06751585006713867
2021-06-20 07:48:20,396 Image to text: 80.7, 97.4, 99.3, 1.0, 1.7
2021-06-20 07:48:20,743 Text to image: 66.0, 92.3, 96.9, 1.0, 3.4
2021-06-20 07:48:20,743 Current rsum is 532.62
2021-06-20 07:48:21,961 runs/coco_butd_region_bert/log
2021-06-20 07:48:21,961 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-20 07:48:21,962 image encoder trainable parameters: 3688352
2021-06-20 07:48:21,966 txt encoder trainable parameters: 120517280
2021-06-20 07:50:52,086 Epoch: [11][124/4426]	Eit 48800  lr 0.0005  Le 9.8589 (9.8661)	Time 1.273 (0.000)	Data 0.002 (0.000)	
2021-06-20 07:54:39,303 Epoch: [11][324/4426]	Eit 49000  lr 0.0005  Le 9.8592 (9.8670)	Time 1.211 (0.000)	Data 0.002 (0.000)	
2021-06-20 07:58:25,805 Epoch: [11][524/4426]	Eit 49200  lr 0.0005  Le 9.8518 (9.8650)	Time 1.079 (0.000)	Data 0.002 (0.000)	
2021-06-20 08:02:11,483 Epoch: [11][724/4426]	Eit 49400  lr 0.0005  Le 9.8986 (9.8651)	Time 1.132 (0.000)	Data 0.002 (0.000)	
2021-06-20 08:05:56,523 Epoch: [11][924/4426]	Eit 49600  lr 0.0005  Le 9.8467 (9.8653)	Time 1.084 (0.000)	Data 0.002 (0.000)	
2021-06-20 08:09:45,237 Epoch: [11][1124/4426]	Eit 49800  lr 0.0005  Le 9.9056 (9.8656)	Time 0.961 (0.000)	Data 0.002 (0.000)	
2021-06-20 08:13:30,372 Epoch: [11][1324/4426]	Eit 50000  lr 0.0005  Le 9.8550 (9.8655)	Time 1.108 (0.000)	Data 0.001 (0.000)	
2021-06-20 08:17:17,557 Epoch: [11][1524/4426]	Eit 50200  lr 0.0005  Le 9.9070 (9.8658)	Time 1.271 (0.000)	Data 0.002 (0.000)	
2021-06-20 08:21:03,216 Epoch: [11][1724/4426]	Eit 50400  lr 0.0005  Le 9.8699 (9.8661)	Time 1.284 (0.000)	Data 0.002 (0.000)	
2021-06-20 08:24:50,989 Epoch: [11][1924/4426]	Eit 50600  lr 0.0005  Le 9.8826 (9.8660)	Time 1.084 (0.000)	Data 0.001 (0.000)	
2021-06-20 08:28:32,147 Epoch: [11][2124/4426]	Eit 50800  lr 0.0005  Le 9.8700 (9.8661)	Time 0.980 (0.000)	Data 0.002 (0.000)	
2021-06-20 08:32:09,442 Epoch: [11][2324/4426]	Eit 51000  lr 0.0005  Le 9.8964 (9.8662)	Time 1.152 (0.000)	Data 0.002 (0.000)	
2021-06-20 08:35:57,402 Epoch: [11][2524/4426]	Eit 51200  lr 0.0005  Le 9.8778 (9.8665)	Time 1.333 (0.000)	Data 0.002 (0.000)	
2021-06-20 08:39:45,300 Epoch: [11][2724/4426]	Eit 51400  lr 0.0005  Le 9.8886 (9.8666)	Time 1.128 (0.000)	Data 0.002 (0.000)	
2021-06-20 08:43:31,593 Epoch: [11][2924/4426]	Eit 51600  lr 0.0005  Le 9.8918 (9.8667)	Time 1.176 (0.000)	Data 0.002 (0.000)	
2021-06-20 08:47:20,103 Epoch: [11][3124/4426]	Eit 51800  lr 0.0005  Le 9.8722 (9.8667)	Time 0.958 (0.000)	Data 0.002 (0.000)	
2021-06-20 08:51:06,988 Epoch: [11][3324/4426]	Eit 52000  lr 0.0005  Le 9.8760 (9.8669)	Time 1.105 (0.000)	Data 0.002 (0.000)	
2021-06-20 08:54:55,651 Epoch: [11][3524/4426]	Eit 52200  lr 0.0005  Le 9.8399 (9.8670)	Time 1.282 (0.000)	Data 0.002 (0.000)	
2021-06-20 08:58:39,451 Epoch: [11][3724/4426]	Eit 52400  lr 0.0005  Le 9.8743 (9.8671)	Time 0.998 (0.000)	Data 0.009 (0.000)	
2021-06-20 09:02:25,346 Epoch: [11][3924/4426]	Eit 52600  lr 0.0005  Le 9.8398 (9.8672)	Time 1.138 (0.000)	Data 0.002 (0.000)	
2021-06-20 09:06:14,386 Epoch: [11][4124/4426]	Eit 52800  lr 0.0005  Le 9.8634 (9.8674)	Time 1.148 (0.000)	Data 0.002 (0.000)	
2021-06-20 09:10:01,394 Epoch: [11][4324/4426]	Eit 53000  lr 0.0005  Le 9.8614 (9.8676)	Time 1.049 (0.000)	Data 0.002 (0.000)	
2021-06-20 09:12:02,247 Test: [0/40]	Le 10.1918 (10.1918)	Time 7.772 (0.000)	
2021-06-20 09:12:21,798 calculate similarity time: 0.07499146461486816
2021-06-20 09:12:22,312 Image to text: 80.7, 97.5, 99.6, 1.0, 2.0
2021-06-20 09:12:22,644 Text to image: 66.6, 92.6, 97.3, 1.0, 3.6
2021-06-20 09:12:22,644 Current rsum is 534.3599999999999
2021-06-20 09:12:25,587 runs/coco_butd_region_bert/log
2021-06-20 09:12:25,587 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-20 09:12:25,588 image encoder trainable parameters: 3688352
2021-06-20 09:12:25,617 txt encoder trainable parameters: 120517280
2021-06-20 09:14:25,539 Epoch: [12][99/4426]	Eit 53200  lr 0.0005  Le 9.8769 (9.8584)	Time 1.127 (0.000)	Data 0.002 (0.000)	
2021-06-20 09:18:11,417 Epoch: [12][299/4426]	Eit 53400  lr 0.0005  Le 9.8525 (9.8611)	Time 1.008 (0.000)	Data 0.002 (0.000)	
2021-06-20 09:21:59,434 Epoch: [12][499/4426]	Eit 53600  lr 0.0005  Le 9.8573 (9.8605)	Time 1.149 (0.000)	Data 0.002 (0.000)	
2021-06-20 09:25:46,526 Epoch: [12][699/4426]	Eit 53800  lr 0.0005  Le 9.8947 (9.8612)	Time 1.174 (0.000)	Data 0.002 (0.000)	
2021-06-20 09:29:30,961 Epoch: [12][899/4426]	Eit 54000  lr 0.0005  Le 9.8705 (9.8613)	Time 1.138 (0.000)	Data 0.002 (0.000)	
2021-06-20 09:33:18,632 Epoch: [12][1099/4426]	Eit 54200  lr 0.0005  Le 9.8411 (9.8620)	Time 1.182 (0.000)	Data 0.002 (0.000)	
2021-06-20 09:37:03,951 Epoch: [12][1299/4426]	Eit 54400  lr 0.0005  Le 9.8569 (9.8624)	Time 1.170 (0.000)	Data 0.002 (0.000)	
2021-06-20 09:40:50,968 Epoch: [12][1499/4426]	Eit 54600  lr 0.0005  Le 9.8680 (9.8627)	Time 1.093 (0.000)	Data 0.002 (0.000)	
2021-06-20 09:44:39,830 Epoch: [12][1699/4426]	Eit 54800  lr 0.0005  Le 9.8855 (9.8629)	Time 1.079 (0.000)	Data 0.002 (0.000)	
2021-06-20 09:48:26,209 Epoch: [12][1899/4426]	Eit 55000  lr 0.0005  Le 9.8276 (9.8629)	Time 1.114 (0.000)	Data 0.002 (0.000)	
2021-06-20 09:52:11,480 Epoch: [12][2099/4426]	Eit 55200  lr 0.0005  Le 9.8814 (9.8630)	Time 1.111 (0.000)	Data 0.002 (0.000)	
2021-06-20 09:55:59,765 Epoch: [12][2299/4426]	Eit 55400  lr 0.0005  Le 9.8413 (9.8633)	Time 1.128 (0.000)	Data 0.002 (0.000)	
2021-06-20 09:59:33,994 Epoch: [12][2499/4426]	Eit 55600  lr 0.0005  Le 9.8779 (9.8635)	Time 1.058 (0.000)	Data 0.002 (0.000)	
2021-06-20 10:03:15,194 Epoch: [12][2699/4426]	Eit 55800  lr 0.0005  Le 9.8794 (9.8636)	Time 1.162 (0.000)	Data 0.002 (0.000)	
2021-06-20 10:07:02,093 Epoch: [12][2899/4426]	Eit 56000  lr 0.0005  Le 9.8777 (9.8636)	Time 1.193 (0.000)	Data 0.002 (0.000)	
2021-06-20 10:10:50,748 Epoch: [12][3099/4426]	Eit 56200  lr 0.0005  Le 9.9139 (9.8638)	Time 1.160 (0.000)	Data 0.002 (0.000)	
2021-06-20 10:14:35,166 Epoch: [12][3299/4426]	Eit 56400  lr 0.0005  Le 9.8688 (9.8640)	Time 0.977 (0.000)	Data 0.002 (0.000)	
2021-06-20 10:18:21,131 Epoch: [12][3499/4426]	Eit 56600  lr 0.0005  Le 9.8592 (9.8643)	Time 0.965 (0.000)	Data 0.002 (0.000)	
2021-06-20 10:22:05,225 Epoch: [12][3699/4426]	Eit 56800  lr 0.0005  Le 9.8482 (9.8643)	Time 1.161 (0.000)	Data 0.002 (0.000)	
2021-06-20 10:25:49,609 Epoch: [12][3899/4426]	Eit 57000  lr 0.0005  Le 9.8673 (9.8643)	Time 1.093 (0.000)	Data 0.002 (0.000)	
2021-06-20 10:29:34,686 Epoch: [12][4099/4426]	Eit 57200  lr 0.0005  Le 9.8787 (9.8644)	Time 1.129 (0.000)	Data 0.002 (0.000)	
2021-06-20 10:33:21,892 Epoch: [12][4299/4426]	Eit 57400  lr 0.0005  Le 9.8876 (9.8645)	Time 1.139 (0.000)	Data 0.013 (0.000)	
2021-06-20 10:35:52,528 Test: [0/40]	Le 10.1979 (10.1979)	Time 8.032 (0.000)	
2021-06-20 10:36:12,326 calculate similarity time: 0.07267403602600098
2021-06-20 10:36:12,826 Image to text: 82.3, 97.8, 99.4, 1.0, 1.9
2021-06-20 10:36:13,149 Text to image: 66.1, 92.3, 97.1, 1.0, 3.4
2021-06-20 10:36:13,149 Current rsum is 534.94
2021-06-20 10:36:16,070 runs/coco_butd_region_bert/log
2021-06-20 10:36:16,070 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-20 10:36:16,072 image encoder trainable parameters: 3688352
2021-06-20 10:36:16,082 txt encoder trainable parameters: 120517280
2021-06-20 10:37:47,368 Epoch: [13][74/4426]	Eit 57600  lr 0.0005  Le 9.8482 (9.8597)	Time 1.021 (0.000)	Data 0.002 (0.000)	
2021-06-20 10:41:33,252 Epoch: [13][274/4426]	Eit 57800  lr 0.0005  Le 9.8735 (9.8602)	Time 1.119 (0.000)	Data 0.002 (0.000)	
2021-06-20 10:45:19,090 Epoch: [13][474/4426]	Eit 58000  lr 0.0005  Le 9.8354 (9.8601)	Time 1.132 (0.000)	Data 0.002 (0.000)	
2021-06-20 10:49:03,613 Epoch: [13][674/4426]	Eit 58200  lr 0.0005  Le 9.8373 (9.8603)	Time 1.411 (0.000)	Data 0.002 (0.000)	
2021-06-20 10:52:50,882 Epoch: [13][874/4426]	Eit 58400  lr 0.0005  Le 9.8740 (9.8606)	Time 1.114 (0.000)	Data 0.002 (0.000)	
2021-06-20 10:56:38,673 Epoch: [13][1074/4426]	Eit 58600  lr 0.0005  Le 9.8582 (9.8606)	Time 1.112 (0.000)	Data 0.002 (0.000)	
2021-06-20 11:00:23,763 Epoch: [13][1274/4426]	Eit 58800  lr 0.0005  Le 9.8504 (9.8604)	Time 1.137 (0.000)	Data 0.002 (0.000)	
2021-06-20 11:04:12,134 Epoch: [13][1474/4426]	Eit 59000  lr 0.0005  Le 9.8237 (9.8607)	Time 1.222 (0.000)	Data 0.002 (0.000)	
2021-06-20 11:07:59,889 Epoch: [13][1674/4426]	Eit 59200  lr 0.0005  Le 9.8409 (9.8610)	Time 1.186 (0.000)	Data 0.002 (0.000)	
2021-06-20 11:11:43,647 Epoch: [13][1874/4426]	Eit 59400  lr 0.0005  Le 9.8500 (9.8611)	Time 1.086 (0.000)	Data 0.002 (0.000)	
2021-06-20 11:15:28,202 Epoch: [13][2074/4426]	Eit 59600  lr 0.0005  Le 9.8516 (9.8610)	Time 1.137 (0.000)	Data 0.002 (0.000)	
2021-06-20 11:19:13,553 Epoch: [13][2274/4426]	Eit 59800  lr 0.0005  Le 9.8192 (9.8613)	Time 1.177 (0.000)	Data 0.002 (0.000)	
2021-06-20 11:23:01,196 Epoch: [13][2474/4426]	Eit 60000  lr 0.0005  Le 9.8664 (9.8612)	Time 1.157 (0.000)	Data 0.002 (0.000)	
2021-06-20 11:26:32,335 Epoch: [13][2674/4426]	Eit 60200  lr 0.0005  Le 9.8736 (9.8614)	Time 1.165 (0.000)	Data 0.002 (0.000)	
2021-06-20 11:30:18,440 Epoch: [13][2874/4426]	Eit 60400  lr 0.0005  Le 9.8786 (9.8615)	Time 1.179 (0.000)	Data 0.002 (0.000)	
2021-06-20 11:34:00,326 Epoch: [13][3074/4426]	Eit 60600  lr 0.0005  Le 9.8618 (9.8615)	Time 1.146 (0.000)	Data 0.002 (0.000)	
2021-06-20 11:37:45,565 Epoch: [13][3274/4426]	Eit 60800  lr 0.0005  Le 9.8650 (9.8616)	Time 1.164 (0.000)	Data 0.002 (0.000)	
2021-06-20 11:41:30,656 Epoch: [13][3474/4426]	Eit 61000  lr 0.0005  Le 9.8399 (9.8617)	Time 1.168 (0.000)	Data 0.002 (0.000)	
2021-06-20 11:45:17,563 Epoch: [13][3674/4426]	Eit 61200  lr 0.0005  Le 9.8716 (9.8618)	Time 1.252 (0.000)	Data 0.002 (0.000)	
2021-06-20 11:49:03,078 Epoch: [13][3874/4426]	Eit 61400  lr 0.0005  Le 9.8695 (9.8618)	Time 1.195 (0.000)	Data 0.002 (0.000)	
2021-06-20 11:52:49,705 Epoch: [13][4074/4426]	Eit 61600  lr 0.0005  Le 9.8392 (9.8619)	Time 1.158 (0.000)	Data 0.002 (0.000)	
2021-06-20 11:56:37,086 Epoch: [13][4274/4426]	Eit 61800  lr 0.0005  Le 9.8714 (9.8619)	Time 1.112 (0.000)	Data 0.002 (0.000)	
2021-06-20 11:59:36,964 Test: [0/40]	Le 10.1956 (10.1956)	Time 7.833 (0.000)	
2021-06-20 11:59:56,634 calculate similarity time: 0.07102417945861816
2021-06-20 11:59:57,201 Image to text: 82.5, 97.7, 99.8, 1.0, 2.2
2021-06-20 11:59:57,630 Text to image: 66.3, 92.8, 97.1, 1.0, 3.7
2021-06-20 11:59:57,630 Current rsum is 536.0999999999999
2021-06-20 12:00:00,540 runs/coco_butd_region_bert/log
2021-06-20 12:00:00,541 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-20 12:00:00,544 image encoder trainable parameters: 3688352
2021-06-20 12:00:00,554 txt encoder trainable parameters: 120517280
2021-06-20 12:01:04,765 Epoch: [14][49/4426]	Eit 62000  lr 0.0005  Le 9.8639 (9.8563)	Time 1.205 (0.000)	Data 0.002 (0.000)	
2021-06-20 12:04:49,568 Epoch: [14][249/4426]	Eit 62200  lr 0.0005  Le 9.8757 (9.8569)	Time 1.165 (0.000)	Data 0.002 (0.000)	
2021-06-20 12:08:34,523 Epoch: [14][449/4426]	Eit 62400  lr 0.0005  Le 9.8273 (9.8569)	Time 1.187 (0.000)	Data 0.002 (0.000)	
2021-06-20 12:12:20,001 Epoch: [14][649/4426]	Eit 62600  lr 0.0005  Le 9.8719 (9.8565)	Time 1.103 (0.000)	Data 0.002 (0.000)	
2021-06-20 12:16:06,920 Epoch: [14][849/4426]	Eit 62800  lr 0.0005  Le 9.9071 (9.8566)	Time 1.159 (0.000)	Data 0.002 (0.000)	
2021-06-20 12:19:54,992 Epoch: [14][1049/4426]	Eit 63000  lr 0.0005  Le 9.8617 (9.8565)	Time 1.144 (0.000)	Data 0.002 (0.000)	
2021-06-20 12:23:42,304 Epoch: [14][1249/4426]	Eit 63200  lr 0.0005  Le 9.8714 (9.8569)	Time 1.141 (0.000)	Data 0.002 (0.000)	
2021-06-20 12:27:29,050 Epoch: [14][1449/4426]	Eit 63400  lr 0.0005  Le 9.8513 (9.8570)	Time 0.999 (0.000)	Data 0.002 (0.000)	
2021-06-20 12:31:17,158 Epoch: [14][1649/4426]	Eit 63600  lr 0.0005  Le 9.9141 (9.8568)	Time 1.134 (0.000)	Data 0.002 (0.000)	
2021-06-20 12:35:00,133 Epoch: [14][1849/4426]	Eit 63800  lr 0.0005  Le 9.8583 (9.8572)	Time 0.886 (0.000)	Data 0.002 (0.000)	
2021-06-20 12:38:47,577 Epoch: [14][2049/4426]	Eit 64000  lr 0.0005  Le 9.8515 (9.8575)	Time 1.106 (0.000)	Data 0.002 (0.000)	
2021-06-20 12:42:33,738 Epoch: [14][2249/4426]	Eit 64200  lr 0.0005  Le 9.8581 (9.8576)	Time 1.197 (0.000)	Data 0.002 (0.000)	
2021-06-20 12:46:19,841 Epoch: [14][2449/4426]	Eit 64400  lr 0.0005  Le 9.8517 (9.8578)	Time 1.026 (0.000)	Data 0.002 (0.000)	
2021-06-20 12:50:05,823 Epoch: [14][2649/4426]	Eit 64600  lr 0.0005  Le 9.8692 (9.8579)	Time 1.107 (0.000)	Data 0.002 (0.000)	
2021-06-20 12:53:36,460 Epoch: [14][2849/4426]	Eit 64800  lr 0.0005  Le 9.8527 (9.8581)	Time 1.181 (0.000)	Data 0.002 (0.000)	
2021-06-20 12:57:24,968 Epoch: [14][3049/4426]	Eit 65000  lr 0.0005  Le 9.8661 (9.8582)	Time 1.230 (0.000)	Data 0.002 (0.000)	
2021-06-20 13:01:11,035 Epoch: [14][3249/4426]	Eit 65200  lr 0.0005  Le 9.8700 (9.8584)	Time 0.856 (0.000)	Data 0.002 (0.000)	
2021-06-20 13:04:56,346 Epoch: [14][3449/4426]	Eit 65400  lr 0.0005  Le 9.8756 (9.8584)	Time 1.108 (0.000)	Data 0.013 (0.000)	
2021-06-20 13:08:42,157 Epoch: [14][3649/4426]	Eit 65600  lr 0.0005  Le 9.8619 (9.8585)	Time 1.170 (0.000)	Data 0.002 (0.000)	
2021-06-20 13:12:28,361 Epoch: [14][3849/4426]	Eit 65800  lr 0.0005  Le 9.8636 (9.8587)	Time 1.254 (0.000)	Data 0.002 (0.000)	
2021-06-20 13:16:16,221 Epoch: [14][4049/4426]	Eit 66000  lr 0.0005  Le 9.8336 (9.8588)	Time 1.087 (0.000)	Data 0.002 (0.000)	
2021-06-20 13:20:02,339 Epoch: [14][4249/4426]	Eit 66200  lr 0.0005  Le 9.8783 (9.8588)	Time 1.222 (0.000)	Data 0.002 (0.000)	
2021-06-20 13:23:28,098 Test: [0/40]	Le 10.1932 (10.1931)	Time 7.890 (0.000)	
2021-06-20 13:23:47,766 calculate similarity time: 0.06494545936584473
2021-06-20 13:23:48,314 Image to text: 81.8, 97.9, 99.5, 1.0, 1.7
2021-06-20 13:23:48,738 Text to image: 66.6, 93.0, 97.0, 1.0, 3.5
2021-06-20 13:23:48,738 Current rsum is 535.72
2021-06-20 13:23:49,964 runs/coco_butd_region_bert/log
2021-06-20 13:23:49,965 runs/coco_butd_region_bert
2021-06-20 13:23:49,965 Current epoch num is 15, decrease all lr by 10
2021-06-20 13:23:49,965 new lr 5e-05
2021-06-20 13:23:49,965 new lr 5e-06
2021-06-20 13:23:49,965 new lr 5e-05
Use VSE++ objective.
2021-06-20 13:23:49,966 image encoder trainable parameters: 3688352
2021-06-20 13:23:49,971 txt encoder trainable parameters: 120517280
2021-06-20 13:24:25,086 Epoch: [15][24/4426]	Eit 66400  lr 5e-05  Le 9.8585 (9.8517)	Time 1.010 (0.000)	Data 0.002 (0.000)	
2021-06-20 13:28:10,986 Epoch: [15][224/4426]	Eit 66600  lr 5e-05  Le 9.8483 (9.8517)	Time 1.161 (0.000)	Data 0.001 (0.000)	
2021-06-20 13:31:55,949 Epoch: [15][424/4426]	Eit 66800  lr 5e-05  Le 9.8806 (9.8501)	Time 1.154 (0.000)	Data 0.002 (0.000)	
2021-06-20 13:35:40,728 Epoch: [15][624/4426]	Eit 67000  lr 5e-05  Le 9.8643 (9.8499)	Time 1.094 (0.000)	Data 0.002 (0.000)	
2021-06-20 13:39:27,044 Epoch: [15][824/4426]	Eit 67200  lr 5e-05  Le 9.8422 (9.8493)	Time 1.102 (0.000)	Data 0.002 (0.000)	
2021-06-20 13:43:14,773 Epoch: [15][1024/4426]	Eit 67400  lr 5e-05  Le 9.8628 (9.8490)	Time 1.010 (0.000)	Data 0.002 (0.000)	
2021-06-20 13:47:01,374 Epoch: [15][1224/4426]	Eit 67600  lr 5e-05  Le 9.8009 (9.8484)	Time 1.169 (0.000)	Data 0.002 (0.000)	
2021-06-20 13:50:48,359 Epoch: [15][1424/4426]	Eit 67800  lr 5e-05  Le 9.8396 (9.8481)	Time 1.170 (0.000)	Data 0.002 (0.000)	
2021-06-20 13:54:34,738 Epoch: [15][1624/4426]	Eit 68000  lr 5e-05  Le 9.8529 (9.8480)	Time 1.115 (0.000)	Data 0.002 (0.000)	
2021-06-20 13:58:20,001 Epoch: [15][1824/4426]	Eit 68200  lr 5e-05  Le 9.8162 (9.8476)	Time 0.994 (0.000)	Data 0.002 (0.000)	
2021-06-20 14:02:05,128 Epoch: [15][2024/4426]	Eit 68400  lr 5e-05  Le 9.8675 (9.8474)	Time 1.247 (0.000)	Data 0.002 (0.000)	
2021-06-20 14:05:50,435 Epoch: [15][2224/4426]	Eit 68600  lr 5e-05  Le 9.8581 (9.8471)	Time 1.244 (0.000)	Data 0.002 (0.000)	
2021-06-20 14:09:37,064 Epoch: [15][2424/4426]	Eit 68800  lr 5e-05  Le 9.8613 (9.8470)	Time 1.162 (0.000)	Data 0.002 (0.000)	
2021-06-20 14:13:23,582 Epoch: [15][2624/4426]	Eit 69000  lr 5e-05  Le 9.8563 (9.8470)	Time 1.132 (0.000)	Data 0.002 (0.000)	
2021-06-20 14:17:09,939 Epoch: [15][2824/4426]	Eit 69200  lr 5e-05  Le 9.8570 (9.8469)	Time 1.094 (0.000)	Data 0.002 (0.000)	
2021-06-20 14:20:55,608 Epoch: [15][3024/4426]	Eit 69400  lr 5e-05  Le 9.8360 (9.8468)	Time 1.219 (0.000)	Data 0.002 (0.000)	
2021-06-20 14:24:26,085 Epoch: [15][3224/4426]	Eit 69600  lr 5e-05  Le 9.8632 (9.8467)	Time 1.150 (0.000)	Data 0.002 (0.000)	
2021-06-20 14:28:14,301 Epoch: [15][3424/4426]	Eit 69800  lr 5e-05  Le 9.8814 (9.8466)	Time 1.079 (0.000)	Data 0.002 (0.000)	
2021-06-20 14:31:59,459 Epoch: [15][3624/4426]	Eit 70000  lr 5e-05  Le 9.8364 (9.8465)	Time 1.120 (0.000)	Data 0.002 (0.000)	
2021-06-20 14:35:47,147 Epoch: [15][3824/4426]	Eit 70200  lr 5e-05  Le 9.8055 (9.8463)	Time 1.157 (0.000)	Data 0.002 (0.000)	
2021-06-20 14:39:32,902 Epoch: [15][4024/4426]	Eit 70400  lr 5e-05  Le 9.8561 (9.8462)	Time 1.137 (0.000)	Data 0.002 (0.000)	
2021-06-20 14:43:19,698 Epoch: [15][4224/4426]	Eit 70600  lr 5e-05  Le 9.8195 (9.8461)	Time 1.067 (0.000)	Data 0.002 (0.000)	
2021-06-20 14:47:04,541 Epoch: [15][4424/4426]	Eit 70800  lr 5e-05  Le 9.8644 (9.8461)	Time 1.119 (0.000)	Data 0.002 (0.000)	
2021-06-20 14:47:13,921 Test: [0/40]	Le 10.1990 (10.1990)	Time 7.977 (0.000)	
2021-06-20 14:47:33,312 calculate similarity time: 0.08377623558044434
2021-06-20 14:47:33,809 Image to text: 81.5, 98.3, 99.7, 1.0, 1.7
2021-06-20 14:47:34,238 Text to image: 67.5, 93.2, 97.2, 1.0, 3.5
2021-06-20 14:47:34,238 Current rsum is 537.42
2021-06-20 14:47:37,160 runs/coco_butd_region_bert/log
2021-06-20 14:47:37,161 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-20 14:47:37,163 image encoder trainable parameters: 3688352
2021-06-20 14:47:37,172 txt encoder trainable parameters: 120517280
2021-06-20 14:51:26,589 Epoch: [16][199/4426]	Eit 71000  lr 5e-05  Le 9.8189 (9.8404)	Time 1.101 (0.000)	Data 0.002 (0.000)	
2021-06-20 14:55:13,883 Epoch: [16][399/4426]	Eit 71200  lr 5e-05  Le 9.8056 (9.8409)	Time 1.129 (0.000)	Data 0.002 (0.000)	
2021-06-20 14:58:58,613 Epoch: [16][599/4426]	Eit 71400  lr 5e-05  Le 9.8539 (9.8408)	Time 1.155 (0.000)	Data 0.002 (0.000)	
2021-06-20 15:02:43,730 Epoch: [16][799/4426]	Eit 71600  lr 5e-05  Le 9.8482 (9.8410)	Time 1.171 (0.000)	Data 0.002 (0.000)	
2021-06-20 15:06:31,093 Epoch: [16][999/4426]	Eit 71800  lr 5e-05  Le 9.8287 (9.8412)	Time 1.088 (0.000)	Data 0.002 (0.000)	
2021-06-20 15:10:19,319 Epoch: [16][1199/4426]	Eit 72000  lr 5e-05  Le 9.8262 (9.8410)	Time 1.211 (0.000)	Data 0.002 (0.000)	
2021-06-20 15:14:06,312 Epoch: [16][1399/4426]	Eit 72200  lr 5e-05  Le 9.8152 (9.8411)	Time 1.177 (0.000)	Data 0.002 (0.000)	
2021-06-20 15:17:51,220 Epoch: [16][1599/4426]	Eit 72400  lr 5e-05  Le 9.8201 (9.8413)	Time 1.301 (0.000)	Data 0.002 (0.000)	
2021-06-20 15:21:39,697 Epoch: [16][1799/4426]	Eit 72600  lr 5e-05  Le 9.8572 (9.8410)	Time 0.953 (0.000)	Data 0.002 (0.000)	
2021-06-20 15:25:26,393 Epoch: [16][1999/4426]	Eit 72800  lr 5e-05  Le 9.8478 (9.8412)	Time 1.094 (0.000)	Data 0.002 (0.000)	
2021-06-20 15:29:12,557 Epoch: [16][2199/4426]	Eit 73000  lr 5e-05  Le 9.8719 (9.8411)	Time 1.102 (0.000)	Data 0.002 (0.000)	
2021-06-20 15:33:00,983 Epoch: [16][2399/4426]	Eit 73200  lr 5e-05  Le 9.8365 (9.8410)	Time 0.882 (0.000)	Data 0.002 (0.000)	
2021-06-20 15:36:47,799 Epoch: [16][2599/4426]	Eit 73400  lr 5e-05  Le 9.8560 (9.8411)	Time 1.121 (0.000)	Data 0.002 (0.000)	
2021-06-20 15:40:35,879 Epoch: [16][2799/4426]	Eit 73600  lr 5e-05  Le 9.8289 (9.8408)	Time 1.135 (0.000)	Data 0.002 (0.000)	
2021-06-20 15:44:21,562 Epoch: [16][2999/4426]	Eit 73800  lr 5e-05  Le 9.8614 (9.8408)	Time 0.997 (0.000)	Data 0.002 (0.000)	
2021-06-20 15:48:05,568 Epoch: [16][3199/4426]	Eit 74000  lr 5e-05  Le 9.8441 (9.8406)	Time 1.238 (0.000)	Data 0.002 (0.000)	
2021-06-20 15:51:40,599 Epoch: [16][3399/4426]	Eit 74200  lr 5e-05  Le 9.8343 (9.8406)	Time 1.239 (0.000)	Data 0.002 (0.000)	
2021-06-20 15:55:28,907 Epoch: [16][3599/4426]	Eit 74400  lr 5e-05  Le 9.8240 (9.8406)	Time 0.924 (0.000)	Data 0.002 (0.000)	
2021-06-20 15:59:14,165 Epoch: [16][3799/4426]	Eit 74600  lr 5e-05  Le 9.8574 (9.8407)	Time 1.114 (0.000)	Data 0.002 (0.000)	
2021-06-20 16:02:59,370 Epoch: [16][3999/4426]	Eit 74800  lr 5e-05  Le 9.8255 (9.8406)	Time 1.105 (0.000)	Data 0.002 (0.000)	
2021-06-20 16:06:43,125 Epoch: [16][4199/4426]	Eit 75000  lr 5e-05  Le 9.8352 (9.8406)	Time 1.191 (0.000)	Data 0.002 (0.000)	
2021-06-20 16:10:28,370 Epoch: [16][4399/4426]	Eit 75200  lr 5e-05  Le 9.8365 (9.8407)	Time 1.149 (0.000)	Data 0.002 (0.000)	
2021-06-20 16:11:05,256 Test: [0/40]	Le 10.1972 (10.1972)	Time 7.762 (0.000)	
2021-06-20 16:11:24,767 calculate similarity time: 0.05548667907714844
2021-06-20 16:11:25,201 Image to text: 82.3, 98.0, 99.6, 1.0, 1.7
2021-06-20 16:11:25,511 Text to image: 67.7, 93.2, 97.1, 1.0, 3.4
2021-06-20 16:11:25,511 Current rsum is 537.92
2021-06-20 16:11:28,527 runs/coco_butd_region_bert/log
2021-06-20 16:11:28,528 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-20 16:11:28,529 image encoder trainable parameters: 3688352
2021-06-20 16:11:28,538 txt encoder trainable parameters: 120517280
2021-06-20 16:14:55,398 Epoch: [17][174/4426]	Eit 75400  lr 5e-05  Le 9.8572 (9.8357)	Time 1.157 (0.000)	Data 0.002 (0.000)	
2021-06-20 16:18:40,561 Epoch: [17][374/4426]	Eit 75600  lr 5e-05  Le 9.8544 (9.8374)	Time 0.999 (0.000)	Data 0.004 (0.000)	
2021-06-20 16:22:25,313 Epoch: [17][574/4426]	Eit 75800  lr 5e-05  Le 9.8154 (9.8377)	Time 1.174 (0.000)	Data 0.002 (0.000)	
2021-06-20 16:26:10,110 Epoch: [17][774/4426]	Eit 76000  lr 5e-05  Le 9.8382 (9.8382)	Time 1.053 (0.000)	Data 0.002 (0.000)	
2021-06-20 16:29:55,955 Epoch: [17][974/4426]	Eit 76200  lr 5e-05  Le 9.8476 (9.8385)	Time 1.048 (0.000)	Data 0.002 (0.000)	
2021-06-20 16:33:41,450 Epoch: [17][1174/4426]	Eit 76400  lr 5e-05  Le 9.8337 (9.8385)	Time 1.177 (0.000)	Data 0.002 (0.000)	
2021-06-20 16:37:26,565 Epoch: [17][1374/4426]	Eit 76600  lr 5e-05  Le 9.8500 (9.8384)	Time 1.101 (0.000)	Data 0.002 (0.000)	
2021-06-20 16:41:13,987 Epoch: [17][1574/4426]	Eit 76800  lr 5e-05  Le 9.8183 (9.8386)	Time 1.104 (0.000)	Data 0.002 (0.000)	
2021-06-20 16:45:02,119 Epoch: [17][1774/4426]	Eit 77000  lr 5e-05  Le 9.8554 (9.8386)	Time 1.090 (0.000)	Data 0.002 (0.000)	
2021-06-20 16:48:46,849 Epoch: [17][1974/4426]	Eit 77200  lr 5e-05  Le 9.8180 (9.8386)	Time 0.880 (0.000)	Data 0.002 (0.000)	
2021-06-20 16:52:31,992 Epoch: [17][2174/4426]	Eit 77400  lr 5e-05  Le 9.8455 (9.8387)	Time 1.129 (0.000)	Data 0.002 (0.000)	
2021-06-20 16:56:16,271 Epoch: [17][2374/4426]	Eit 77600  lr 5e-05  Le 9.8036 (9.8387)	Time 1.276 (0.000)	Data 0.002 (0.000)	
2021-06-20 17:00:01,556 Epoch: [17][2574/4426]	Eit 77800  lr 5e-05  Le 9.8600 (9.8387)	Time 1.143 (0.000)	Data 0.002 (0.000)	
2021-06-20 17:03:48,030 Epoch: [17][2774/4426]	Eit 78000  lr 5e-05  Le 9.8444 (9.8385)	Time 1.260 (0.000)	Data 0.002 (0.000)	
2021-06-20 17:07:33,683 Epoch: [17][2974/4426]	Eit 78200  lr 5e-05  Le 9.8323 (9.8385)	Time 1.081 (0.000)	Data 0.002 (0.000)	
2021-06-20 17:11:18,773 Epoch: [17][3174/4426]	Eit 78400  lr 5e-05  Le 9.8190 (9.8386)	Time 1.144 (0.000)	Data 0.002 (0.000)	
2021-06-20 17:15:05,217 Epoch: [17][3374/4426]	Eit 78600  lr 5e-05  Le 9.8369 (9.8386)	Time 1.016 (0.000)	Data 0.002 (0.000)	
2021-06-20 17:18:32,129 Epoch: [17][3574/4426]	Eit 78800  lr 5e-05  Le 9.8532 (9.8385)	Time 0.966 (0.000)	Data 0.002 (0.000)	
2021-06-20 17:22:23,177 Epoch: [17][3774/4426]	Eit 79000  lr 5e-05  Le 9.8439 (9.8385)	Time 1.161 (0.000)	Data 0.002 (0.000)	
2021-06-20 17:26:10,496 Epoch: [17][3974/4426]	Eit 79200  lr 5e-05  Le 9.8464 (9.8386)	Time 1.139 (0.000)	Data 0.002 (0.000)	
2021-06-20 17:29:56,535 Epoch: [17][4174/4426]	Eit 79400  lr 5e-05  Le 9.8240 (9.8385)	Time 0.894 (0.000)	Data 0.002 (0.000)	
2021-06-20 17:33:44,940 Epoch: [17][4374/4426]	Eit 79600  lr 5e-05  Le 9.8513 (9.8386)	Time 1.135 (0.000)	Data 0.002 (0.000)	
2021-06-20 17:34:50,273 Test: [0/40]	Le 10.2001 (10.2001)	Time 7.806 (0.000)	
2021-06-20 17:35:09,986 calculate similarity time: 0.06099891662597656
2021-06-20 17:35:10,414 Image to text: 82.5, 98.1, 99.6, 1.0, 1.8
2021-06-20 17:35:10,729 Text to image: 67.8, 93.2, 97.1, 1.0, 3.4
2021-06-20 17:35:10,729 Current rsum is 538.26
2021-06-20 17:35:13,761 runs/coco_butd_region_bert/log
2021-06-20 17:35:13,762 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-20 17:35:13,763 image encoder trainable parameters: 3688352
2021-06-20 17:35:13,772 txt encoder trainable parameters: 120517280
2021-06-20 17:38:10,086 Epoch: [18][149/4426]	Eit 79800  lr 5e-05  Le 9.8326 (9.8364)	Time 1.162 (0.000)	Data 0.002 (0.000)	
2021-06-20 17:41:55,996 Epoch: [18][349/4426]	Eit 80000  lr 5e-05  Le 9.8254 (9.8367)	Time 1.208 (0.000)	Data 0.002 (0.000)	
2021-06-20 17:45:43,743 Epoch: [18][549/4426]	Eit 80200  lr 5e-05  Le 9.8277 (9.8366)	Time 1.431 (0.000)	Data 0.002 (0.000)	
2021-06-20 17:49:28,957 Epoch: [18][749/4426]	Eit 80400  lr 5e-05  Le 9.8684 (9.8368)	Time 1.066 (0.000)	Data 0.002 (0.000)	
2021-06-20 17:53:15,125 Epoch: [18][949/4426]	Eit 80600  lr 5e-05  Le 9.8462 (9.8371)	Time 1.182 (0.000)	Data 0.002 (0.000)	
2021-06-20 17:57:02,680 Epoch: [18][1149/4426]	Eit 80800  lr 5e-05  Le 9.8058 (9.8366)	Time 1.141 (0.000)	Data 0.007 (0.000)	
2021-06-20 18:00:48,602 Epoch: [18][1349/4426]	Eit 81000  lr 5e-05  Le 9.8287 (9.8363)	Time 0.869 (0.000)	Data 0.002 (0.000)	
2021-06-20 18:04:34,242 Epoch: [18][1549/4426]	Eit 81200  lr 5e-05  Le 9.8185 (9.8364)	Time 1.221 (0.000)	Data 0.002 (0.000)	
2021-06-20 18:08:20,329 Epoch: [18][1749/4426]	Eit 81400  lr 5e-05  Le 9.8651 (9.8364)	Time 1.247 (0.000)	Data 0.002 (0.000)	
2021-06-20 18:12:07,491 Epoch: [18][1949/4426]	Eit 81600  lr 5e-05  Le 9.8300 (9.8361)	Time 1.137 (0.000)	Data 0.002 (0.000)	
2021-06-20 18:15:54,095 Epoch: [18][2149/4426]	Eit 81800  lr 5e-05  Le 9.8213 (9.8363)	Time 1.198 (0.000)	Data 0.002 (0.000)	
2021-06-20 18:19:40,835 Epoch: [18][2349/4426]	Eit 82000  lr 5e-05  Le 9.8135 (9.8361)	Time 1.077 (0.000)	Data 0.002 (0.000)	
2021-06-20 18:23:25,516 Epoch: [18][2549/4426]	Eit 82200  lr 5e-05  Le 9.8310 (9.8362)	Time 1.127 (0.000)	Data 0.002 (0.000)	
2021-06-20 18:27:10,964 Epoch: [18][2749/4426]	Eit 82400  lr 5e-05  Le 9.8445 (9.8363)	Time 1.172 (0.000)	Data 0.002 (0.000)	
2021-06-20 18:30:55,694 Epoch: [18][2949/4426]	Eit 82600  lr 5e-05  Le 9.8171 (9.8364)	Time 1.059 (0.000)	Data 0.002 (0.000)	
2021-06-20 18:34:42,005 Epoch: [18][3149/4426]	Eit 82800  lr 5e-05  Le 9.8256 (9.8362)	Time 1.165 (0.000)	Data 0.002 (0.000)	
2021-06-20 18:38:28,691 Epoch: [18][3349/4426]	Eit 83000  lr 5e-05  Le 9.8304 (9.8362)	Time 1.074 (0.000)	Data 0.002 (0.000)	
2021-06-20 18:42:16,368 Epoch: [18][3549/4426]	Eit 83200  lr 5e-05  Le 9.8204 (9.8362)	Time 0.974 (0.000)	Data 0.002 (0.000)	
2021-06-20 18:46:03,422 Epoch: [18][3749/4426]	Eit 83400  lr 5e-05  Le 9.8306 (9.8361)	Time 1.172 (0.000)	Data 0.002 (0.000)	
2021-06-20 18:49:30,649 Epoch: [18][3949/4426]	Eit 83600  lr 5e-05  Le 9.8987 (9.8362)	Time 1.438 (0.000)	Data 0.002 (0.000)	
2021-06-20 18:53:19,058 Epoch: [18][4149/4426]	Eit 83800  lr 5e-05  Le 9.7952 (9.8361)	Time 1.159 (0.000)	Data 0.002 (0.000)	
2021-06-20 18:57:04,921 Epoch: [18][4349/4426]	Eit 84000  lr 5e-05  Le 9.8530 (9.8361)	Time 1.083 (0.000)	Data 0.002 (0.000)	
2021-06-20 18:58:39,600 Test: [0/40]	Le 10.1999 (10.1999)	Time 8.059 (0.000)	
2021-06-20 18:58:59,324 calculate similarity time: 0.055832624435424805
2021-06-20 18:58:59,760 Image to text: 82.7, 98.1, 99.7, 1.0, 1.7
2021-06-20 18:59:00,072 Text to image: 68.1, 93.1, 97.2, 1.0, 3.5
2021-06-20 18:59:00,072 Current rsum is 538.92
2021-06-20 18:59:03,042 runs/coco_butd_region_bert/log
2021-06-20 18:59:03,042 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-20 18:59:03,044 image encoder trainable parameters: 3688352
2021-06-20 18:59:03,055 txt encoder trainable parameters: 120517280
2021-06-20 19:01:32,048 Epoch: [19][124/4426]	Eit 84200  lr 5e-05  Le 9.8345 (9.8333)	Time 1.099 (0.000)	Data 0.002 (0.000)	
2021-06-20 19:05:20,891 Epoch: [19][324/4426]	Eit 84400  lr 5e-05  Le 9.8133 (9.8349)	Time 0.939 (0.000)	Data 0.002 (0.000)	
2021-06-20 19:09:06,652 Epoch: [19][524/4426]	Eit 84600  lr 5e-05  Le 9.8712 (9.8347)	Time 1.113 (0.000)	Data 0.002 (0.000)	
2021-06-20 19:12:52,242 Epoch: [19][724/4426]	Eit 84800  lr 5e-05  Le 9.8654 (9.8345)	Time 0.909 (0.000)	Data 0.002 (0.000)	
2021-06-20 19:16:38,450 Epoch: [19][924/4426]	Eit 85000  lr 5e-05  Le 9.8309 (9.8351)	Time 1.137 (0.000)	Data 0.002 (0.000)	
2021-06-20 19:20:25,532 Epoch: [19][1124/4426]	Eit 85200  lr 5e-05  Le 9.8442 (9.8354)	Time 1.185 (0.000)	Data 0.002 (0.000)	
2021-06-20 19:24:11,313 Epoch: [19][1324/4426]	Eit 85400  lr 5e-05  Le 9.8029 (9.8352)	Time 1.391 (0.000)	Data 0.002 (0.000)	
2021-06-20 19:27:56,882 Epoch: [19][1524/4426]	Eit 85600  lr 5e-05  Le 9.8145 (9.8354)	Time 1.197 (0.000)	Data 0.002 (0.000)	
2021-06-20 19:31:43,253 Epoch: [19][1724/4426]	Eit 85800  lr 5e-05  Le 9.8531 (9.8355)	Time 1.114 (0.000)	Data 0.002 (0.000)	
2021-06-20 19:35:27,046 Epoch: [19][1924/4426]	Eit 86000  lr 5e-05  Le 9.8588 (9.8354)	Time 1.074 (0.000)	Data 0.002 (0.000)	
2021-06-20 19:39:13,825 Epoch: [19][2124/4426]	Eit 86200  lr 5e-05  Le 9.8530 (9.8355)	Time 1.079 (0.000)	Data 0.002 (0.000)	
2021-06-20 19:42:59,322 Epoch: [19][2324/4426]	Eit 86400  lr 5e-05  Le 9.8560 (9.8357)	Time 0.997 (0.000)	Data 0.002 (0.000)	
2021-06-20 19:46:46,628 Epoch: [19][2524/4426]	Eit 86600  lr 5e-05  Le 9.8116 (9.8357)	Time 1.186 (0.000)	Data 0.002 (0.000)	
2021-06-20 19:50:32,015 Epoch: [19][2724/4426]	Eit 86800  lr 5e-05  Le 9.8674 (9.8358)	Time 1.121 (0.000)	Data 0.002 (0.000)	
2021-06-20 19:54:18,857 Epoch: [19][2924/4426]	Eit 87000  lr 5e-05  Le 9.8572 (9.8359)	Time 1.309 (0.000)	Data 0.002 (0.000)	
2021-06-20 19:58:06,976 Epoch: [19][3124/4426]	Eit 87200  lr 5e-05  Le 9.8478 (9.8359)	Time 1.078 (0.000)	Data 0.002 (0.000)	
2021-06-20 20:01:51,043 Epoch: [19][3324/4426]	Eit 87400  lr 5e-05  Le 9.8179 (9.8359)	Time 0.942 (0.000)	Data 0.002 (0.000)	
2021-06-20 20:05:36,494 Epoch: [19][3524/4426]	Eit 87600  lr 5e-05  Le 9.8418 (9.8358)	Time 1.223 (0.000)	Data 0.002 (0.000)	
2021-06-20 20:09:20,898 Epoch: [19][3724/4426]	Eit 87800  lr 5e-05  Le 9.8066 (9.8357)	Time 1.071 (0.000)	Data 0.002 (0.000)	
2021-06-20 20:13:09,644 Epoch: [19][3924/4426]	Eit 88000  lr 5e-05  Le 9.8121 (9.8358)	Time 1.188 (0.000)	Data 0.004 (0.000)	
2021-06-20 20:16:42,686 Epoch: [19][4124/4426]	Eit 88200  lr 5e-05  Le 9.8147 (9.8358)	Time 1.051 (0.000)	Data 0.002 (0.000)	
2021-06-20 20:20:29,828 Epoch: [19][4324/4426]	Eit 88400  lr 5e-05  Le 9.8384 (9.8357)	Time 1.139 (0.000)	Data 0.002 (0.000)	
2021-06-20 20:22:31,860 Test: [0/40]	Le 10.2004 (10.2004)	Time 8.206 (0.000)	
2021-06-20 20:22:51,665 calculate similarity time: 0.07403564453125
2021-06-20 20:22:52,191 Image to text: 82.0, 98.3, 99.6, 1.0, 1.7
2021-06-20 20:22:52,500 Text to image: 67.9, 93.3, 97.0, 1.0, 3.5
2021-06-20 20:22:52,500 Current rsum is 538.16
2021-06-20 20:22:53,775 runs/coco_butd_region_bert/log
2021-06-20 20:22:53,775 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-20 20:22:53,776 image encoder trainable parameters: 3688352
2021-06-20 20:22:53,780 txt encoder trainable parameters: 120517280
2021-06-20 20:24:55,317 Epoch: [20][99/4426]	Eit 88600  lr 5e-05  Le 9.8405 (9.8335)	Time 1.169 (0.000)	Data 0.002 (0.000)	
2021-06-20 20:28:41,540 Epoch: [20][299/4426]	Eit 88800  lr 5e-05  Le 9.8297 (9.8341)	Time 1.110 (0.000)	Data 0.002 (0.000)	
2021-06-20 20:32:26,223 Epoch: [20][499/4426]	Eit 89000  lr 5e-05  Le 9.8208 (9.8333)	Time 1.027 (0.000)	Data 0.002 (0.000)	
2021-06-20 20:36:13,885 Epoch: [20][699/4426]	Eit 89200  lr 5e-05  Le 9.8484 (9.8337)	Time 1.173 (0.000)	Data 0.002 (0.000)	
2021-06-20 20:40:00,256 Epoch: [20][899/4426]	Eit 89400  lr 5e-05  Le 9.8300 (9.8339)	Time 1.185 (0.000)	Data 0.002 (0.000)	
2021-06-20 20:43:44,734 Epoch: [20][1099/4426]	Eit 89600  lr 5e-05  Le 9.8502 (9.8341)	Time 1.113 (0.000)	Data 0.002 (0.000)	
2021-06-20 20:47:32,191 Epoch: [20][1299/4426]	Eit 89800  lr 5e-05  Le 9.8374 (9.8340)	Time 1.118 (0.000)	Data 0.002 (0.000)	
2021-06-20 20:51:18,375 Epoch: [20][1499/4426]	Eit 90000  lr 5e-05  Le 9.8284 (9.8340)	Time 1.109 (0.000)	Data 0.002 (0.000)	
2021-06-20 20:55:02,117 Epoch: [20][1699/4426]	Eit 90200  lr 5e-05  Le 9.8610 (9.8336)	Time 1.112 (0.000)	Data 0.002 (0.000)	
2021-06-20 20:58:50,356 Epoch: [20][1899/4426]	Eit 90400  lr 5e-05  Le 9.8483 (9.8337)	Time 1.037 (0.000)	Data 0.002 (0.000)	
2021-06-20 21:02:35,871 Epoch: [20][2099/4426]	Eit 90600  lr 5e-05  Le 9.8304 (9.8337)	Time 1.160 (0.000)	Data 0.002 (0.000)	
2021-06-20 21:06:19,558 Epoch: [20][2299/4426]	Eit 90800  lr 5e-05  Le 9.8428 (9.8339)	Time 1.172 (0.000)	Data 0.003 (0.000)	
2021-06-20 21:10:05,429 Epoch: [20][2499/4426]	Eit 91000  lr 5e-05  Le 9.8298 (9.8338)	Time 1.184 (0.000)	Data 0.002 (0.000)	
2021-06-20 21:13:51,509 Epoch: [20][2699/4426]	Eit 91200  lr 5e-05  Le 9.8108 (9.8338)	Time 1.116 (0.000)	Data 0.005 (0.000)	
2021-06-20 21:17:38,938 Epoch: [20][2899/4426]	Eit 91400  lr 5e-05  Le 9.8475 (9.8338)	Time 1.085 (0.000)	Data 0.002 (0.000)	
2021-06-20 21:21:30,481 Epoch: [20][3099/4426]	Eit 91600  lr 5e-05  Le 9.8318 (9.8338)	Time 1.181 (0.000)	Data 0.002 (0.000)	
2021-06-20 21:25:16,187 Epoch: [20][3299/4426]	Eit 91800  lr 5e-05  Le 9.8167 (9.8339)	Time 1.120 (0.000)	Data 0.007 (0.000)	
2021-06-20 21:29:01,851 Epoch: [20][3499/4426]	Eit 92000  lr 5e-05  Le 9.8411 (9.8340)	Time 1.104 (0.000)	Data 0.002 (0.000)	
2021-06-20 21:32:46,842 Epoch: [20][3699/4426]	Eit 92200  lr 5e-05  Le 9.8117 (9.8339)	Time 1.134 (0.000)	Data 0.002 (0.000)	
2021-06-20 21:36:33,743 Epoch: [20][3899/4426]	Eit 92400  lr 5e-05  Le 9.8087 (9.8339)	Time 1.163 (0.000)	Data 0.002 (0.000)	
2021-06-20 21:40:19,362 Epoch: [20][4099/4426]	Eit 92600  lr 5e-05  Le 9.8411 (9.8341)	Time 0.927 (0.000)	Data 0.002 (0.000)	
2021-06-20 21:43:49,612 Epoch: [20][4299/4426]	Eit 92800  lr 5e-05  Le 9.8608 (9.8340)	Time 1.256 (0.000)	Data 0.002 (0.000)	
2021-06-20 21:46:22,167 Test: [0/40]	Le 10.1986 (10.1986)	Time 8.067 (0.000)	
2021-06-20 21:46:41,607 calculate similarity time: 0.08174347877502441
2021-06-20 21:46:42,099 Image to text: 82.0, 98.3, 99.5, 1.0, 1.7
2021-06-20 21:46:42,410 Text to image: 67.9, 93.3, 97.1, 1.0, 3.5
2021-06-20 21:46:42,411 Current rsum is 538.0
2021-06-20 21:46:43,618 runs/coco_butd_region_bert/log
2021-06-20 21:46:43,618 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-20 21:46:43,619 image encoder trainable parameters: 3688352
2021-06-20 21:46:43,623 txt encoder trainable parameters: 120517280
2021-06-20 21:48:16,557 Epoch: [21][74/4426]	Eit 93000  lr 5e-05  Le 9.8387 (9.8332)	Time 1.125 (0.000)	Data 0.002 (0.000)	
2021-06-20 21:52:04,056 Epoch: [21][274/4426]	Eit 93200  lr 5e-05  Le 9.8441 (9.8319)	Time 0.969 (0.000)	Data 0.002 (0.000)	
2021-06-20 21:55:51,462 Epoch: [21][474/4426]	Eit 93400  lr 5e-05  Le 9.8416 (9.8319)	Time 1.162 (0.000)	Data 0.002 (0.000)	
2021-06-20 21:59:37,534 Epoch: [21][674/4426]	Eit 93600  lr 5e-05  Le 9.8443 (9.8331)	Time 1.140 (0.000)	Data 0.002 (0.000)	
2021-06-20 22:03:23,084 Epoch: [21][874/4426]	Eit 93800  lr 5e-05  Le 9.8575 (9.8328)	Time 1.224 (0.000)	Data 0.002 (0.000)	
2021-06-20 22:07:08,991 Epoch: [21][1074/4426]	Eit 94000  lr 5e-05  Le 9.8707 (9.8329)	Time 1.255 (0.000)	Data 0.002 (0.000)	
2021-06-20 22:10:57,177 Epoch: [21][1274/4426]	Eit 94200  lr 5e-05  Le 9.8458 (9.8329)	Time 1.149 (0.000)	Data 0.002 (0.000)	
2021-06-20 22:14:44,208 Epoch: [21][1474/4426]	Eit 94400  lr 5e-05  Le 9.8192 (9.8327)	Time 1.377 (0.000)	Data 0.002 (0.000)	
2021-06-20 22:18:30,548 Epoch: [21][1674/4426]	Eit 94600  lr 5e-05  Le 9.8175 (9.8325)	Time 1.174 (0.000)	Data 0.002 (0.000)	
2021-06-20 22:22:16,558 Epoch: [21][1874/4426]	Eit 94800  lr 5e-05  Le 9.8018 (9.8325)	Time 1.094 (0.000)	Data 0.002 (0.000)	
2021-06-20 22:26:02,544 Epoch: [21][2074/4426]	Eit 95000  lr 5e-05  Le 9.8162 (9.8325)	Time 1.313 (0.000)	Data 0.002 (0.000)	
2021-06-20 22:29:47,611 Epoch: [21][2274/4426]	Eit 95200  lr 5e-05  Le 9.7765 (9.8324)	Time 1.151 (0.000)	Data 0.006 (0.000)	
2021-06-20 22:33:36,342 Epoch: [21][2474/4426]	Eit 95400  lr 5e-05  Le 9.8272 (9.8324)	Time 1.141 (0.000)	Data 0.002 (0.000)	
2021-06-20 22:37:22,131 Epoch: [21][2674/4426]	Eit 95600  lr 5e-05  Le 9.8363 (9.8324)	Time 1.200 (0.000)	Data 0.002 (0.000)	
2021-06-20 22:41:06,960 Epoch: [21][2874/4426]	Eit 95800  lr 5e-05  Le 9.8106 (9.8324)	Time 1.127 (0.000)	Data 0.002 (0.000)	
2021-06-20 22:44:51,194 Epoch: [21][3074/4426]	Eit 96000  lr 5e-05  Le 9.8345 (9.8325)	Time 0.877 (0.000)	Data 0.002 (0.000)	
2021-06-20 22:48:38,138 Epoch: [21][3274/4426]	Eit 96200  lr 5e-05  Le 9.8140 (9.8325)	Time 1.162 (0.000)	Data 0.002 (0.000)	
2021-06-20 22:52:23,407 Epoch: [21][3474/4426]	Eit 96400  lr 5e-05  Le 9.8345 (9.8327)	Time 0.974 (0.000)	Data 0.002 (0.000)	
2021-06-20 22:56:08,215 Epoch: [21][3674/4426]	Eit 96600  lr 5e-05  Le 9.8465 (9.8327)	Time 1.048 (0.000)	Data 0.002 (0.000)	
2021-06-20 22:59:53,657 Epoch: [21][3874/4426]	Eit 96800  lr 5e-05  Le 9.8578 (9.8327)	Time 1.264 (0.000)	Data 0.002 (0.000)	
2021-06-20 23:03:40,783 Epoch: [21][4074/4426]	Eit 97000  lr 5e-05  Le 9.8201 (9.8328)	Time 1.133 (0.000)	Data 0.002 (0.000)	
2021-06-20 23:07:28,400 Epoch: [21][4274/4426]	Eit 97200  lr 5e-05  Le 9.8464 (9.8328)	Time 1.145 (0.000)	Data 0.002 (0.000)	
2021-06-20 23:10:27,369 Test: [0/40]	Le 10.1992 (10.1992)	Time 8.111 (0.000)	
2021-06-20 23:10:44,825 calculate similarity time: 0.06711244583129883
2021-06-20 23:10:45,342 Image to text: 82.6, 98.2, 99.6, 1.0, 1.6
2021-06-20 23:10:45,687 Text to image: 68.0, 93.2, 97.1, 1.0, 3.4
2021-06-20 23:10:45,687 Current rsum is 538.64
2021-06-20 23:10:46,898 runs/coco_butd_region_bert/log
2021-06-20 23:10:46,899 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-20 23:10:46,899 image encoder trainable parameters: 3688352
2021-06-20 23:10:46,904 txt encoder trainable parameters: 120517280
2021-06-20 23:11:41,712 Epoch: [22][49/4426]	Eit 97400  lr 5e-05  Le 9.8225 (9.8295)	Time 1.105 (0.000)	Data 0.002 (0.000)	
2021-06-20 23:15:29,166 Epoch: [22][249/4426]	Eit 97600  lr 5e-05  Le 9.8231 (9.8301)	Time 1.170 (0.000)	Data 0.002 (0.000)	
2021-06-20 23:19:16,056 Epoch: [22][449/4426]	Eit 97800  lr 5e-05  Le 9.8036 (9.8296)	Time 1.176 (0.000)	Data 0.002 (0.000)	
2021-06-20 23:23:00,674 Epoch: [22][649/4426]	Eit 98000  lr 5e-05  Le 9.8508 (9.8309)	Time 1.097 (0.000)	Data 0.002 (0.000)	
2021-06-20 23:26:46,278 Epoch: [22][849/4426]	Eit 98200  lr 5e-05  Le 9.8223 (9.8310)	Time 1.111 (0.000)	Data 0.002 (0.000)	
2021-06-20 23:30:32,169 Epoch: [22][1049/4426]	Eit 98400  lr 5e-05  Le 9.8372 (9.8312)	Time 1.124 (0.000)	Data 0.002 (0.000)	
2021-06-20 23:34:20,098 Epoch: [22][1249/4426]	Eit 98600  lr 5e-05  Le 9.8510 (9.8314)	Time 1.128 (0.000)	Data 0.002 (0.000)	
2021-06-20 23:38:06,814 Epoch: [22][1449/4426]	Eit 98800  lr 5e-05  Le 9.8157 (9.8316)	Time 1.129 (0.000)	Data 0.003 (0.000)	
2021-06-20 23:41:54,964 Epoch: [22][1649/4426]	Eit 99000  lr 5e-05  Le 9.8352 (9.8317)	Time 1.071 (0.000)	Data 0.002 (0.000)	
2021-06-20 23:45:42,292 Epoch: [22][1849/4426]	Eit 99200  lr 5e-05  Le 9.8395 (9.8318)	Time 1.205 (0.000)	Data 0.002 (0.000)	
2021-06-20 23:49:27,609 Epoch: [22][2049/4426]	Eit 99400  lr 5e-05  Le 9.8264 (9.8319)	Time 1.142 (0.000)	Data 0.002 (0.000)	
2021-06-20 23:53:15,147 Epoch: [22][2249/4426]	Eit 99600  lr 5e-05  Le 9.8125 (9.8317)	Time 1.114 (0.000)	Data 0.002 (0.000)	
2021-06-20 23:57:01,594 Epoch: [22][2449/4426]	Eit 99800  lr 5e-05  Le 9.8344 (9.8317)	Time 1.070 (0.000)	Data 0.002 (0.000)	
2021-06-21 00:00:48,067 Epoch: [22][2649/4426]	Eit 100000  lr 5e-05  Le 9.8029 (9.8317)	Time 1.076 (0.000)	Data 0.002 (0.000)	
2021-06-21 00:04:35,042 Epoch: [22][2849/4426]	Eit 100200  lr 5e-05  Le 9.8260 (9.8318)	Time 1.121 (0.000)	Data 0.002 (0.000)	
2021-06-21 00:08:22,843 Epoch: [22][3049/4426]	Eit 100400  lr 5e-05  Le 9.8286 (9.8319)	Time 1.090 (0.000)	Data 0.002 (0.000)	
2021-06-21 00:12:08,122 Epoch: [22][3249/4426]	Eit 100600  lr 5e-05  Le 9.7955 (9.8318)	Time 1.104 (0.000)	Data 0.002 (0.000)	
2021-06-21 00:15:53,457 Epoch: [22][3449/4426]	Eit 100800  lr 5e-05  Le 9.8530 (9.8318)	Time 1.090 (0.000)	Data 0.002 (0.000)	
2021-06-21 00:19:40,090 Epoch: [22][3649/4426]	Eit 101000  lr 5e-05  Le 9.8434 (9.8319)	Time 1.043 (0.000)	Data 0.002 (0.000)	
2021-06-21 00:23:26,050 Epoch: [22][3849/4426]	Eit 101200  lr 5e-05  Le 9.8379 (9.8318)	Time 1.138 (0.000)	Data 0.001 (0.000)	
2021-06-21 00:27:10,935 Epoch: [22][4049/4426]	Eit 101400  lr 5e-05  Le 9.8382 (9.8318)	Time 1.126 (0.000)	Data 0.002 (0.000)	
2021-06-21 00:30:58,797 Epoch: [22][4249/4426]	Eit 101600  lr 5e-05  Le 9.8364 (9.8318)	Time 1.123 (0.000)	Data 0.002 (0.000)	
2021-06-21 00:34:27,583 Test: [0/40]	Le 10.2009 (10.2009)	Time 8.149 (0.000)	
2021-06-21 00:34:47,583 calculate similarity time: 0.10831570625305176
2021-06-21 00:34:48,081 Image to text: 82.7, 98.4, 99.7, 1.0, 1.7
2021-06-21 00:34:48,416 Text to image: 68.0, 93.1, 97.2, 1.0, 3.5
2021-06-21 00:34:48,416 Current rsum is 539.1
2021-06-21 00:34:51,454 runs/coco_butd_region_bert/log
2021-06-21 00:34:51,454 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-21 00:34:51,456 image encoder trainable parameters: 3688352
2021-06-21 00:34:51,466 txt encoder trainable parameters: 120517280
2021-06-21 00:35:27,662 Epoch: [23][24/4426]	Eit 101800  lr 5e-05  Le 9.8163 (9.8325)	Time 1.069 (0.000)	Data 0.002 (0.000)	
2021-06-21 00:39:04,372 Epoch: [23][224/4426]	Eit 102000  lr 5e-05  Le 9.7873 (9.8305)	Time 1.202 (0.000)	Data 0.002 (0.000)	
2021-06-21 00:42:42,739 Epoch: [23][424/4426]	Eit 102200  lr 5e-05  Le 9.8427 (9.8306)	Time 1.120 (0.000)	Data 0.002 (0.000)	
2021-06-21 00:46:31,801 Epoch: [23][624/4426]	Eit 102400  lr 5e-05  Le 9.8235 (9.8303)	Time 1.168 (0.000)	Data 0.002 (0.000)	
2021-06-21 00:50:14,201 Epoch: [23][824/4426]	Eit 102600  lr 5e-05  Le 9.8198 (9.8305)	Time 1.039 (0.000)	Data 0.002 (0.000)	
2021-06-21 00:53:59,036 Epoch: [23][1024/4426]	Eit 102800  lr 5e-05  Le 9.8172 (9.8307)	Time 1.123 (0.000)	Data 0.002 (0.000)	
2021-06-21 00:57:43,851 Epoch: [23][1224/4426]	Eit 103000  lr 5e-05  Le 9.8412 (9.8305)	Time 1.172 (0.000)	Data 0.002 (0.000)	
2021-06-21 01:01:28,465 Epoch: [23][1424/4426]	Eit 103200  lr 5e-05  Le 9.8162 (9.8304)	Time 1.142 (0.000)	Data 0.002 (0.000)	
2021-06-21 01:05:15,079 Epoch: [23][1624/4426]	Eit 103400  lr 5e-05  Le 9.8372 (9.8306)	Time 1.137 (0.000)	Data 0.002 (0.000)	
2021-06-21 01:09:01,206 Epoch: [23][1824/4426]	Eit 103600  lr 5e-05  Le 9.8231 (9.8305)	Time 1.148 (0.000)	Data 0.002 (0.000)	
2021-06-21 01:12:48,415 Epoch: [23][2024/4426]	Eit 103800  lr 5e-05  Le 9.8195 (9.8306)	Time 1.125 (0.000)	Data 0.002 (0.000)	
2021-06-21 01:16:34,294 Epoch: [23][2224/4426]	Eit 104000  lr 5e-05  Le 9.8465 (9.8305)	Time 1.183 (0.000)	Data 0.002 (0.000)	
2021-06-21 01:20:19,847 Epoch: [23][2424/4426]	Eit 104200  lr 5e-05  Le 9.8151 (9.8307)	Time 1.163 (0.000)	Data 0.002 (0.000)	
2021-06-21 01:24:07,378 Epoch: [23][2624/4426]	Eit 104400  lr 5e-05  Le 9.8246 (9.8307)	Time 1.133 (0.000)	Data 0.002 (0.000)	
2021-06-21 01:27:56,828 Epoch: [23][2824/4426]	Eit 104600  lr 5e-05  Le 9.8508 (9.8306)	Time 1.356 (0.000)	Data 0.002 (0.000)	
2021-06-21 01:31:42,156 Epoch: [23][3024/4426]	Eit 104800  lr 5e-05  Le 9.8174 (9.8305)	Time 1.172 (0.000)	Data 0.002 (0.000)	
2021-06-21 01:35:27,769 Epoch: [23][3224/4426]	Eit 105000  lr 5e-05  Le 9.8378 (9.8305)	Time 0.889 (0.000)	Data 0.002 (0.000)	
2021-06-21 01:39:12,403 Epoch: [23][3424/4426]	Eit 105200  lr 5e-05  Le 9.8221 (9.8306)	Time 1.143 (0.000)	Data 0.002 (0.000)	
2021-06-21 01:43:00,353 Epoch: [23][3624/4426]	Eit 105400  lr 5e-05  Le 9.8406 (9.8308)	Time 1.150 (0.000)	Data 0.002 (0.000)	
2021-06-21 01:46:44,997 Epoch: [23][3824/4426]	Eit 105600  lr 5e-05  Le 9.8449 (9.8309)	Time 1.172 (0.000)	Data 0.002 (0.000)	
2021-06-21 01:50:31,726 Epoch: [23][4024/4426]	Eit 105800  lr 5e-05  Le 9.8004 (9.8309)	Time 1.225 (0.000)	Data 0.002 (0.000)	
2021-06-21 01:54:18,713 Epoch: [23][4224/4426]	Eit 106000  lr 5e-05  Le 9.8273 (9.8309)	Time 1.079 (0.000)	Data 0.002 (0.000)	
2021-06-21 01:58:07,235 Epoch: [23][4424/4426]	Eit 106200  lr 5e-05  Le 9.8185 (9.8309)	Time 1.254 (0.000)	Data 0.002 (0.000)	
2021-06-21 01:58:16,938 Test: [0/40]	Le 10.1995 (10.1995)	Time 8.347 (0.000)	
2021-06-21 01:58:36,212 calculate similarity time: 0.0756680965423584
2021-06-21 01:58:36,759 Image to text: 82.4, 98.3, 99.6, 1.0, 1.6
2021-06-21 01:58:37,173 Text to image: 67.7, 93.0, 97.1, 1.0, 3.5
2021-06-21 01:58:37,173 Current rsum is 538.14
2021-06-21 01:58:38,403 runs/coco_butd_region_bert/log
2021-06-21 01:58:38,403 runs/coco_butd_region_bert
Use VSE++ objective.
2021-06-21 01:58:38,403 image encoder trainable parameters: 3688352
2021-06-21 01:58:38,408 txt encoder trainable parameters: 120517280
2021-06-21 02:02:30,323 Epoch: [24][199/4426]	Eit 106400  lr 5e-05  Le 9.8204 (9.8308)	Time 1.151 (0.000)	Data 0.002 (0.000)	
2021-06-21 02:06:18,525 Epoch: [24][399/4426]	Eit 106600  lr 5e-05  Le 9.8239 (9.8312)	Time 1.051 (0.000)	Data 0.002 (0.000)	
2021-06-21 02:09:48,403 Epoch: [24][599/4426]	Eit 106800  lr 5e-05  Le 9.8264 (9.8309)	Time 1.125 (0.000)	Data 0.002 (0.000)	
2021-06-21 02:13:33,807 Epoch: [24][799/4426]	Eit 107000  lr 5e-05  Le 9.8325 (9.8303)	Time 1.148 (0.000)	Data 0.002 (0.000)	
2021-06-21 02:17:20,414 Epoch: [24][999/4426]	Eit 107200  lr 5e-05  Le 9.8302 (9.8299)	Time 1.097 (0.000)	Data 0.002 (0.000)	
2021-06-21 02:21:07,377 Epoch: [24][1199/4426]	Eit 107400  lr 5e-05  Le 9.8052 (9.8300)	Time 0.913 (0.000)	Data 0.002 (0.000)	
2021-06-21 02:24:51,794 Epoch: [24][1399/4426]	Eit 107600  lr 5e-05  Le 9.8277 (9.8300)	Time 1.173 (0.000)	Data 0.002 (0.000)	
2021-06-21 02:28:37,716 Epoch: [24][1599/4426]	Eit 107800  lr 5e-05  Le 9.8166 (9.8298)	Time 1.067 (0.000)	Data 0.002 (0.000)	
2021-06-21 02:32:23,825 Epoch: [24][1799/4426]	Eit 108000  lr 5e-05  Le 9.8384 (9.8301)	Time 1.127 (0.000)	Data 0.002 (0.000)	
2021-06-21 02:36:11,977 Epoch: [24][1999/4426]	Eit 108200  lr 5e-05  Le 9.8410 (9.8301)	Time 1.235 (0.000)	Data 0.002 (0.000)	
2021-06-21 02:40:00,209 Epoch: [24][2199/4426]	Eit 108400  lr 5e-05  Le 9.8218 (9.8301)	Time 1.063 (0.000)	Data 0.002 (0.000)	
2021-06-21 02:43:48,507 Epoch: [24][2399/4426]	Eit 108600  lr 5e-05  Le 9.8437 (9.8300)	Time 1.130 (0.000)	Data 0.002 (0.000)	
2021-06-21 02:47:33,035 Epoch: [24][2599/4426]	Eit 108800  lr 5e-05  Le 9.8357 (9.8301)	Time 1.153 (0.000)	Data 0.002 (0.000)	
2021-06-21 02:51:22,535 Epoch: [24][2799/4426]	Eit 109000  lr 5e-05  Le 9.8172 (9.8303)	Time 1.071 (0.000)	Data 0.002 (0.000)	
2021-06-21 02:55:08,249 Epoch: [24][2999/4426]	Eit 109200  lr 5e-05  Le 9.8299 (9.8303)	Time 1.136 (0.000)	Data 0.002 (0.000)	
2021-06-21 02:58:55,860 Epoch: [24][3199/4426]	Eit 109400  lr 5e-05  Le 9.8393 (9.8301)	Time 0.991 (0.000)	Data 0.002 (0.000)	
2021-06-21 03:02:42,170 Epoch: [24][3399/4426]	Eit 109600  lr 5e-05  Le 9.7955 (9.8301)	Time 0.949 (0.000)	Data 0.002 (0.000)	
2021-06-21 03:06:28,158 Epoch: [24][3599/4426]	Eit 109800  lr 5e-05  Le 9.8646 (9.8301)	Time 1.290 (0.000)	Data 0.002 (0.000)	
2021-06-21 03:10:13,597 Epoch: [24][3799/4426]	Eit 110000  lr 5e-05  Le 9.8322 (9.8301)	Time 1.173 (0.000)	Data 0.002 (0.000)	
2021-06-21 03:13:59,703 Epoch: [24][3999/4426]	Eit 110200  lr 5e-05  Le 9.8372 (9.8301)	Time 1.160 (0.000)	Data 0.002 (0.000)	
2021-06-21 03:17:46,991 Epoch: [24][4199/4426]	Eit 110400  lr 5e-05  Le 9.8109 (9.8301)	Time 1.056 (0.000)	Data 0.002 (0.000)	
2021-06-21 03:21:33,861 Epoch: [24][4399/4426]	Eit 110600  lr 5e-05  Le 9.8274 (9.8302)	Time 1.260 (0.000)	Data 0.002 (0.000)	
2021-06-21 03:22:11,528 Test: [0/40]	Le 10.2011 (10.2011)	Time 8.232 (0.000)	
2021-06-21 03:22:31,338 calculate similarity time: 0.07097744941711426
2021-06-21 03:22:31,858 Image to text: 82.2, 98.4, 99.7, 1.0, 1.6
2021-06-21 03:22:32,289 Text to image: 67.7, 93.1, 97.0, 1.0, 3.5
2021-06-21 03:22:32,290 Current rsum is 538.04
You have new mail in /var/spool/mail/root
[root@gpu1 vse_infty-master-my-graph-gru-unified-no-graph]# CUDA_VISIBLE_DEVICES=0,1 python3 eval.py --dataset coco --data_path ../data/coco
INFO:root:Evaluating runs/coco_butd_region_bert...

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
/root/anaconda3/lib/python3.6/site-packages/torch/nn/modules/rnn.py:735: UserWarning: RNN module weights are not part of single contiguous chunk of memory. This means they need to be compacted at every call, possibly greatly increasing memory usage. To compact weights again call flatten_parameters(). (Triggered internally at  /pytorch/aten/src/ATen/native/cudnn/RNN.cpp:1264.)
  self.dropout, self.training, self.bidirectional, self.batch_first)
INFO:lib.evaluation:Test: [0/196]	Le 10.2127 (10.2127)	Time 6.990 (0.000)	
INFO:lib.evaluation:Test: [10/196]	Le 10.2123 (10.2085)	Time 0.582 (0.000)	
INFO:lib.evaluation:Test: [20/196]	Le 10.2104 (10.2054)	Time 0.431 (0.000)	
INFO:lib.evaluation:Test: [30/196]	Le 10.2187 (10.2069)	Time 0.428 (0.000)	
INFO:lib.evaluation:Test: [40/196]	Le 10.2148 (10.2053)	Time 0.426 (0.000)	
INFO:lib.evaluation:Test: [50/196]	Le 10.2068 (10.2049)	Time 0.448 (0.000)	
INFO:lib.evaluation:Test: [60/196]	Le 10.2007 (10.2054)	Time 0.426 (0.000)	
INFO:lib.evaluation:Test: [70/196]	Le 10.2162 (10.2048)	Time 0.459 (0.000)	
INFO:lib.evaluation:Test: [80/196]	Le 10.2113 (10.2054)	Time 0.529 (0.000)	
INFO:lib.evaluation:Test: [90/196]	Le 10.2061 (10.2059)	Time 0.449 (0.000)	
INFO:lib.evaluation:Test: [100/196]	Le 10.2116 (10.2064)	Time 0.419 (0.000)	
INFO:lib.evaluation:Test: [110/196]	Le 10.1953 (10.2063)	Time 0.399 (0.000)	
INFO:lib.evaluation:Test: [120/196]	Le 10.2410 (10.2068)	Time 0.449 (0.000)	
INFO:lib.evaluation:Test: [130/196]	Le 10.1964 (10.2069)	Time 0.429 (0.000)	
INFO:lib.evaluation:Test: [140/196]	Le 10.2180 (10.2073)	Time 0.416 (0.000)	
INFO:lib.evaluation:Test: [150/196]	Le 10.1953 (10.2073)	Time 0.418 (0.000)	
INFO:lib.evaluation:Test: [160/196]	Le 10.2133 (10.2074)	Time 0.446 (0.000)	
INFO:lib.evaluation:Test: [170/196]	Le 10.2001 (10.2071)	Time 0.454 (0.000)	
INFO:lib.evaluation:Test: [180/196]	Le 10.1960 (10.2067)	Time 0.427 (0.000)	
INFO:lib.evaluation:Test: [190/196]	Le 10.1890 (10.2071)	Time 0.263 (0.000)	
INFO:lib.evaluation:Images: 5000, Captions: 25000
INFO:lib.evaluation:calculate similarity time: 0.09393119812011719
INFO:lib.evaluation:Image to text: 82.9, 97.5, 99.1, 1.0, 1.5
INFO:lib.evaluation:Text to image: 66.3, 92.1, 96.3, 1.0, 3.9
INFO:lib.evaluation:rsum: 534.2 ar: 93.2 ari: 84.9
INFO:lib.evaluation:calculate similarity time: 0.09306502342224121
INFO:lib.evaluation:Image to text: 79.6, 96.0, 98.6, 1.0, 1.8
INFO:lib.evaluation:Text to image: 66.4, 91.3, 96.0, 1.0, 3.9
INFO:lib.evaluation:rsum: 527.9 ar: 91.4 ari: 84.6
INFO:lib.evaluation:calculate similarity time: 0.09298038482666016
INFO:lib.evaluation:Image to text: 81.5, 96.8, 98.9, 1.0, 1.6
INFO:lib.evaluation:Text to image: 66.4, 91.2, 96.2, 1.0, 3.8
INFO:lib.evaluation:rsum: 531.0 ar: 92.4 ari: 84.6
INFO:lib.evaluation:calculate similarity time: 0.09844350814819336
INFO:lib.evaluation:Image to text: 79.5, 95.9, 99.0, 1.0, 1.6
INFO:lib.evaluation:Text to image: 63.9, 91.2, 96.7, 1.0, 3.5
INFO:lib.evaluation:rsum: 526.3 ar: 91.5 ari: 84.0
INFO:lib.evaluation:calculate similarity time: 0.09001827239990234
INFO:lib.evaluation:Image to text: 82.1, 96.7, 98.8, 1.0, 1.6
INFO:lib.evaluation:Text to image: 66.0, 92.2, 96.5, 1.0, 3.5
INFO:lib.evaluation:rsum: 532.3 ar: 92.5 ari: 84.9
INFO:lib.evaluation:-----------------------------------
INFO:lib.evaluation:Mean metrics: 
INFO:lib.evaluation:rsum: 530.3
INFO:lib.evaluation:Average i2t Recall: 92.2
INFO:lib.evaluation:Image to text: 81.1 96.6 98.9 1.0 1.6
INFO:lib.evaluation:Average t2i Recall: 84.6
INFO:lib.evaluation:Text to image: 65.8 91.6 96.4 1.0 3.7
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
INFO:lib.evaluation:Test: [0/196]	Le 10.2127 (10.2127)	Time 2.291 (0.000)	
INFO:lib.evaluation:Test: [10/196]	Le 10.2123 (10.2085)	Time 0.421 (0.000)	
INFO:lib.evaluation:Test: [20/196]	Le 10.2104 (10.2054)	Time 0.416 (0.000)	
INFO:lib.evaluation:Test: [30/196]	Le 10.2187 (10.2069)	Time 0.450 (0.000)	
INFO:lib.evaluation:Test: [40/196]	Le 10.2148 (10.2053)	Time 0.412 (0.000)	
INFO:lib.evaluation:Test: [50/196]	Le 10.2068 (10.2049)	Time 0.436 (0.000)	
INFO:lib.evaluation:Test: [60/196]	Le 10.2007 (10.2054)	Time 0.418 (0.000)	
INFO:lib.evaluation:Test: [70/196]	Le 10.2162 (10.2048)	Time 0.344 (0.000)	
INFO:lib.evaluation:Test: [80/196]	Le 10.2113 (10.2054)	Time 0.547 (0.000)	
INFO:lib.evaluation:Test: [90/196]	Le 10.2061 (10.2059)	Time 0.444 (0.000)	
INFO:lib.evaluation:Test: [100/196]	Le 10.2116 (10.2064)	Time 0.437 (0.000)	
INFO:lib.evaluation:Test: [110/196]	Le 10.1953 (10.2063)	Time 0.455 (0.000)	
INFO:lib.evaluation:Test: [120/196]	Le 10.2410 (10.2068)	Time 0.471 (0.000)	
INFO:lib.evaluation:Test: [130/196]	Le 10.1964 (10.2069)	Time 0.443 (0.000)	
INFO:lib.evaluation:Test: [140/196]	Le 10.2180 (10.2073)	Time 0.410 (0.000)	
INFO:lib.evaluation:Test: [150/196]	Le 10.1953 (10.2073)	Time 0.409 (0.000)	
INFO:lib.evaluation:Test: [160/196]	Le 10.2133 (10.2074)	Time 0.421 (0.000)	
INFO:lib.evaluation:Test: [170/196]	Le 10.2001 (10.2071)	Time 0.555 (0.000)	
INFO:lib.evaluation:Test: [180/196]	Le 10.1960 (10.2067)	Time 0.425 (0.000)	
INFO:lib.evaluation:Test: [190/196]	Le 10.1890 (10.2071)	Time 0.413 (0.000)	
INFO:lib.evaluation:Images: 5000, Captions: 25000
INFO:lib.evaluation:calculate similarity time: 0.9775807857513428
INFO:lib.evaluation:rsum: 439.7
INFO:lib.evaluation:Average i2t Recall: 79.5
INFO:lib.evaluation:Image to text: 59.6 86.3 92.7 1.0 3.9
INFO:lib.evaluation:Average t2i Recall: 67.0
INFO:lib.evaluation:Text to image: 43.5 73.8 83.9 2.0 14.4
INFO:root:Evaluating runs/coco_butd_grid_bert...
Traceback (most recent call last):
  File "eval.py", line 58, in <module>
    main()
  File "eval.py", line 46, in main
    evaluation.evalrank(model_path, data_path=opt.data_path, split='testall', fold5=True)
  File "/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-unified-no-graph/lib/evaluation.py", line 196, in evalrank
    checkpoint = torch.load(model_path)
  File "/root/anaconda3/lib/python3.6/site-packages/torch/serialization.py", line 571, in load
    with _open_file_like(f, 'rb') as opened_file:
  File "/root/anaconda3/lib/python3.6/site-packages/torch/serialization.py", line 229, in _open_file_like
    return _open_file(name_or_buffer, mode)
  File "/root/anaconda3/lib/python3.6/site-packages/torch/serialization.py", line 210, in __init__
    super(_open_file, self).__init__(open(name, mode))
FileNotFoundError: [Errno 2] No such file or directory: 'runs/coco_butd_grid_bert/model_best.pth'
You have new mail in /var/spool/mail/root
[root@gpu1 vse_infty-master-my-graph-gru-unified-no-graph]# 
[root@gpu1 vse_infty-master-my-graph-gru-unified-no-graph]# 

