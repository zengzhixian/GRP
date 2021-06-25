[root@gpu1 vse_infty-master-my-graph-gru-unified-meanpooling]# sh train_grid.sh 
2021-06-22 22:40:16,744 Namespace(backbone_lr_factor=0.01, backbone_path='/tmp/data/weights/original_updown_backbone.pth', backbone_source='detector', backbone_warmup_epochs=0, batch_size=128, data_name='coco', data_path='/tmp/data/coco', embed_size=1024, embedding_warmup_epochs=1, grad_clip=2.0, img_dim=2048, input_scale_factor=2.0, learning_rate=0.0005, log_step=200, logger_name='runs/coco_butd_grid_bert/log', lr_update=15, margin=0.2, max_violation=False, model_name='runs/coco_butd_grid_bert', no_imgnorm=False, no_txtnorm=False, num_epochs=25, optim='adam', precomp_enc_type='backbone', reset_start_epoch=False, resume='', val_step=500, vocab_path='./vocab/', vse_mean_warmup_epochs=1, word_dim=300, workers=20)
2021-06-22 22:40:16,744 Model name '/public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased' not found in model shortcut name list (bert-base-uncased, bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, bert-base-multilingual-cased, bert-base-chinese, bert-base-german-cased, bert-large-uncased-whole-word-masking, bert-large-cased-whole-word-masking, bert-large-uncased-whole-word-masking-finetuned-squad, bert-large-cased-whole-word-masking-finetuned-squad, bert-base-cased-finetuned-mrpc). Assuming '/public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased' is a path or url to a directory containing tokenizer files.
2021-06-22 22:40:16,745 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/added_tokens.json. We won't load it.
2021-06-22 22:40:16,745 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/special_tokens_map.json. We won't load it.
2021-06-22 22:40:16,745 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/tokenizer_config.json. We won't load it.
2021-06-22 22:40:16,745 loading file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/vocab.txt
2021-06-22 22:40:16,745 loading file None
2021-06-22 22:40:16,745 loading file None
2021-06-22 22:40:16,745 loading file None
Traceback (most recent call last):
  File "train.py", line 269, in <module>
    main()
  File "train.py", line 37, in main
    opt.data_path, opt.data_name, tokenizer, opt.batch_size, opt.workers, opt)
  File "/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-unified-meanpooling/lib/datasets/image_caption.py", line 352, in get_loaders
    batch_size, True, workers)
  File "/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-unified-meanpooling/lib/datasets/image_caption.py", line 340, in get_loader
    dset = RawImageDataset(data_path, data_name, data_split, tokenizer, opt, train)
  File "/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-unified-meanpooling/lib/datasets/image_caption.py", line 38, in __init__
    with open(loc_mapping, 'r') as f_mapping:
FileNotFoundError: [Errno 2] No such file or directory: '/tmp/data/coco/id_mapping.json'
[root@gpu1 vse_infty-master-my-graph-gru-unified-meanpooling]# ^C
[root@gpu1 vse_infty-master-my-graph-gru-unified-meanpooling]# ^C
[root@gpu1 vse_infty-master-my-graph-gru-unified-meanpooling]# ^C
[root@gpu1 vse_infty-master-my-graph-gru-unified-meanpooling]# ^C
[root@gpu1 vse_infty-master-my-graph-gru-unified-meanpooling]# sh train_region.sh 
2021-06-22 22:40:25,635 Namespace(backbone_lr_factor=0.01, backbone_path='', backbone_source='detector', backbone_warmup_epochs=5, batch_size=128, data_name='f30k', data_path='../data/f30k', embed_size=1024, embedding_warmup_epochs=2, grad_clip=2.0, img_dim=2048, input_scale_factor=1, learning_rate=0.0005, log_step=200, logger_name='runs/f30k_butd_region_bert/log', lr_update=15, margin=0.2, max_violation=False, model_name='runs/f30k_butd_region_bert', no_imgnorm=False, no_txtnorm=False, num_epochs=25, optim='adam', precomp_enc_type='basic', reset_start_epoch=False, resume='', val_step=500, vocab_path='./vocab/', vse_mean_warmup_epochs=1, word_dim=300, workers=10)
2021-06-22 22:40:25,635 Model name '/public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased' not found in model shortcut name list (bert-base-uncased, bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, bert-base-multilingual-cased, bert-base-chinese, bert-base-german-cased, bert-large-uncased-whole-word-masking, bert-large-cased-whole-word-masking, bert-large-uncased-whole-word-masking-finetuned-squad, bert-large-cased-whole-word-masking-finetuned-squad, bert-base-cased-finetuned-mrpc). Assuming '/public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased' is a path or url to a directory containing tokenizer files.
2021-06-22 22:40:25,635 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/added_tokens.json. We won't load it.
2021-06-22 22:40:25,635 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/special_tokens_map.json. We won't load it.
2021-06-22 22:40:25,635 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/tokenizer_config.json. We won't load it.
2021-06-22 22:40:25,635 loading file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/vocab.txt
2021-06-22 22:40:25,635 loading file None
2021-06-22 22:40:25,635 loading file None
2021-06-22 22:40:25,635 loading file None
/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-unified-meanpooling/lib/encoders.py:48: UserWarning: nn.init.constant is now deprecated in favor of nn.init.constant_.
  nn.init.constant(self.W[1].weight, 0)
/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-unified-meanpooling/lib/encoders.py:49: UserWarning: nn.init.constant is now deprecated in favor of nn.init.constant_.
  nn.init.constant(self.W[1].bias, 0)
2021-06-22 22:40:35,417 loading configuration file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/config.json
2021-06-22 22:40:35,418 Model config {
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

2021-06-22 22:40:35,418 loading weights file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/pytorch_model.bin
2021-06-22 22:40:44,379 Use adam as the optimizer, with init lr 0.0005
2021-06-22 22:40:44,380 Image encoder is data paralleled now.
2021-06-22 22:40:44,381 runs/f30k_butd_region_bert/log
2021-06-22 22:40:44,381 runs/f30k_butd_region_bert
2021-06-22 22:40:44,383 image encoder trainable parameters: 20490144
2021-06-22 22:40:44,393 txt encoder trainable parameters: 137319072
/root/anaconda3/lib/python3.6/site-packages/torch/nn/modules/rnn.py:735: UserWarning: RNN module weights are not part of single contiguous chunk of memory. This means they need to be compacted at every call, possibly greatly increasing memory usage. To compact weights again call flatten_parameters(). (Triggered internally at  /pytorch/aten/src/ATen/native/cudnn/RNN.cpp:1264.)
  self.dropout, self.training, self.bidirectional, self.batch_first)
2021-06-22 22:47:27,153 Epoch: [0][199/1133]	Eit 200  lr 0.0005  Le 10.1354 (10.1549)	Time 2.159 (0.000)	Data 0.002 (0.000)	
2021-06-22 22:53:42,297 Epoch: [0][399/1133]	Eit 400  lr 0.0005  Le 10.1218 (10.1419)	Time 1.744 (0.000)	Data 0.002 (0.000)	
2021-06-22 23:00:11,406 Epoch: [0][599/1133]	Eit 600  lr 0.0005  Le 10.1202 (10.1345)	Time 1.738 (0.000)	Data 0.002 (0.000)	
2021-06-22 23:06:38,559 Epoch: [0][799/1133]	Eit 800  lr 0.0005  Le 10.0794 (10.1274)	Time 1.609 (0.000)	Data 0.002 (0.000)	
2021-06-22 23:13:06,940 Epoch: [0][999/1133]	Eit 1000  lr 0.0005  Le 10.0828 (10.1204)	Time 1.801 (0.000)	Data 0.003 (0.000)	
2021-06-22 23:17:12,697 Test: [0/40]	Le 10.1698 (10.1698)	Time 3.896 (0.000)	
2021-06-22 23:17:40,431 calculate similarity time: 0.09253168106079102
2021-06-22 23:17:40,952 Image to text: 46.7, 72.7, 83.0, 2.0, 9.1
2021-06-22 23:17:41,381 Text to image: 35.4, 65.3, 76.4, 3.0, 15.3
2021-06-22 23:17:41,381 Current rsum is 379.46000000000004
2021-06-22 23:17:44,189 runs/f30k_butd_region_bert/log
2021-06-22 23:17:44,189 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-22 23:17:44,191 image encoder trainable parameters: 20490144
2021-06-22 23:17:44,196 txt encoder trainable parameters: 137319072
2021-06-22 23:19:58,363 Epoch: [1][67/1133]	Eit 1200  lr 0.0005  Le 10.0697 (10.0661)	Time 1.971 (0.000)	Data 0.002 (0.000)	
Traceback (most recent call last):
  File "train.py", line 269, in <module>
    main()
  File "train.py", line 99, in main
    train(opt, train_loader, model, epoch, val_loader)
  File "train.py", line 147, in train
    model.train_emb(images, captions, lengths, image_lengths=img_lengths)
  File "/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-unified-meanpooling/lib/vse.py", line 186, in train_emb
    img_emb, cap_emb = self.forward_emb(images, captions, lengths, image_lengths=image_lengths)
  File "/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-unified-meanpooling/lib/vse.py", line 168, in forward_emb
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
  File "/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-unified-meanpooling/lib/encoders.py", line 302, in forward
    GCN_img_emd = self.Rs_GCN_2(GCN_img_emd)
  File "/root/anaconda3/lib/python3.6/site-packages/torch/nn/modules/module.py", line 722, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-unified-meanpooling/lib/encoders.py", line 89, in forward
    v_star = W_y + v
RuntimeError: CUDA out of memory. Tried to allocate 22.00 MiB (GPU 0; 22.38 GiB total capacity; 8.57 GiB already allocated; 14.56 MiB free; 9.31 GiB reserved in total by PyTorch)

[root@gpu1 vse_infty-master-my-graph-gru-unified-meanpooling]# sh train_region.sh 
2021-06-23 08:23:24,389 Namespace(backbone_lr_factor=0.01, backbone_path='', backbone_source='detector', backbone_warmup_epochs=5, batch_size=128, data_name='f30k', data_path='../data/f30k', embed_size=1024, embedding_warmup_epochs=2, grad_clip=2.0, img_dim=2048, input_scale_factor=1, learning_rate=0.0005, log_step=200, logger_name='runs/f30k_butd_region_bert/log', lr_update=15, margin=0.2, max_violation=False, model_name='runs/f30k_butd_region_bert', no_imgnorm=False, no_txtnorm=False, num_epochs=25, optim='adam', precomp_enc_type='basic', reset_start_epoch=False, resume='', val_step=500, vocab_path='./vocab/', vse_mean_warmup_epochs=1, word_dim=300, workers=10)
2021-06-23 08:23:24,389 Model name '/public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased' not found in model shortcut name list (bert-base-uncased, bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, bert-base-multilingual-cased, bert-base-chinese, bert-base-german-cased, bert-large-uncased-whole-word-masking, bert-large-cased-whole-word-masking, bert-large-uncased-whole-word-masking-finetuned-squad, bert-large-cased-whole-word-masking-finetuned-squad, bert-base-cased-finetuned-mrpc). Assuming '/public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased' is a path or url to a directory containing tokenizer files.
2021-06-23 08:23:24,389 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/added_tokens.json. We won't load it.
2021-06-23 08:23:24,389 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/special_tokens_map.json. We won't load it.
2021-06-23 08:23:24,389 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/tokenizer_config.json. We won't load it.
2021-06-23 08:23:24,389 loading file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/vocab.txt
2021-06-23 08:23:24,389 loading file None
2021-06-23 08:23:24,389 loading file None
2021-06-23 08:23:24,389 loading file None
/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-unified-meanpooling/lib/encoders.py:48: UserWarning: nn.init.constant is now deprecated in favor of nn.init.constant_.
  nn.init.constant(self.W[1].weight, 0)
/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-unified-meanpooling/lib/encoders.py:49: UserWarning: nn.init.constant is now deprecated in favor of nn.init.constant_.
  nn.init.constant(self.W[1].bias, 0)
2021-06-23 08:23:32,582 loading configuration file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/config.json
2021-06-23 08:23:32,583 Model config {
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

2021-06-23 08:23:32,584 loading weights file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/pytorch_model.bin
2021-06-23 08:23:39,344 Use adam as the optimizer, with init lr 0.0005
2021-06-23 08:23:39,345 Image encoder is data paralleled now.
2021-06-23 08:23:39,345 runs/f30k_butd_region_bert/log
2021-06-23 08:23:39,345 runs/f30k_butd_region_bert
2021-06-23 08:23:39,347 image encoder trainable parameters: 20490144
2021-06-23 08:23:39,351 txt encoder trainable parameters: 137319072
/root/anaconda3/lib/python3.6/site-packages/torch/nn/modules/rnn.py:735: UserWarning: RNN module weights are not part of single contiguous chunk of memory. This means they need to be compacted at every call, possibly greatly increasing memory usage. To compact weights again call flatten_parameters(). (Triggered internally at  /pytorch/aten/src/ATen/native/cudnn/RNN.cpp:1264.)
  self.dropout, self.training, self.bidirectional, self.batch_first)
2021-06-23 08:26:37,856 Epoch: [0][199/1133]	Eit 200  lr 0.0005  Le 10.1338 (10.1507)	Time 1.060 (0.000)	Data 0.002 (0.000)	
2021-06-23 08:31:08,780 Epoch: [0][399/1133]	Eit 400  lr 0.0005  Le 10.1229 (10.1390)	Time 1.739 (0.000)	Data 0.002 (0.000)	
2021-06-23 08:36:48,431 Epoch: [0][599/1133]	Eit 600  lr 0.0005  Le 10.1252 (10.1315)	Time 1.979 (0.000)	Data 0.002 (0.000)	
2021-06-23 08:42:31,750 Epoch: [0][799/1133]	Eit 800  lr 0.0005  Le 10.0878 (10.1242)	Time 1.769 (0.000)	Data 0.002 (0.000)	
2021-06-23 08:48:21,963 Epoch: [0][999/1133]	Eit 1000  lr 0.0005  Le 10.0931 (10.1176)	Time 1.880 (0.000)	Data 0.002 (0.000)	
2021-06-23 08:52:11,063 Test: [0/40]	Le 10.1814 (10.1814)	Time 3.701 (0.000)	
2021-06-23 08:52:34,831 calculate similarity time: 0.0821084976196289
2021-06-23 08:52:35,300 Image to text: 48.5, 76.6, 85.7, 2.0, 7.8
2021-06-23 08:52:35,611 Text to image: 34.8, 65.5, 76.9, 3.0, 14.8
2021-06-23 08:52:35,612 Current rsum is 387.92
2021-06-23 08:52:40,091 runs/f30k_butd_region_bert/log
2021-06-23 08:52:40,092 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-23 08:52:40,095 image encoder trainable parameters: 20490144
2021-06-23 08:52:40,103 txt encoder trainable parameters: 137319072
2021-06-23 08:54:38,797 Epoch: [1][67/1133]	Eit 1200  lr 0.0005  Le 10.0526 (10.0638)	Time 1.476 (0.000)	Data 0.002 (0.000)	
2021-06-23 09:00:22,484 Epoch: [1][267/1133]	Eit 1400  lr 0.0005  Le 10.0695 (10.0626)	Time 1.941 (0.000)	Data 0.002 (0.000)	
2021-06-23 09:06:07,253 Epoch: [1][467/1133]	Eit 1600  lr 0.0005  Le 10.0661 (10.0582)	Time 1.895 (0.000)	Data 0.003 (0.000)	
2021-06-23 09:11:51,006 Epoch: [1][667/1133]	Eit 1800  lr 0.0005  Le 10.0552 (10.0542)	Time 1.864 (0.000)	Data 0.002 (0.000)	
2021-06-23 09:17:34,149 Epoch: [1][867/1133]	Eit 2000  lr 0.0005  Le 10.0311 (10.0502)	Time 2.064 (0.000)	Data 0.002 (0.000)	
2021-06-23 09:23:16,207 Epoch: [1][1067/1133]	Eit 2200  lr 0.0005  Le 10.0283 (10.0464)	Time 1.538 (0.000)	Data 0.002 (0.000)	
2021-06-23 09:25:09,480 Test: [0/40]	Le 10.1852 (10.1852)	Time 3.690 (0.000)	
2021-06-23 09:25:32,691 calculate similarity time: 0.0738534927368164
2021-06-23 09:25:33,241 Image to text: 61.1, 85.6, 91.3, 1.0, 4.5
2021-06-23 09:25:33,570 Text to image: 44.6, 74.6, 83.7, 2.0, 11.0
2021-06-23 09:25:33,570 Current rsum is 440.98
2021-06-23 09:25:37,011 runs/f30k_butd_region_bert/log
2021-06-23 09:25:37,011 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-23 09:25:37,014 image encoder trainable parameters: 20490144
2021-06-23 09:25:37,021 txt encoder trainable parameters: 137319072
2021-06-23 09:29:33,554 Epoch: [2][135/1133]	Eit 2400  lr 0.0005  Le 9.9759 (10.0070)	Time 1.729 (0.000)	Data 0.002 (0.000)	
2021-06-23 09:35:15,519 Epoch: [2][335/1133]	Eit 2600  lr 0.0005  Le 10.0167 (10.0050)	Time 1.631 (0.000)	Data 0.002 (0.000)	
2021-06-23 09:40:55,532 Epoch: [2][535/1133]	Eit 2800  lr 0.0005  Le 9.9885 (10.0035)	Time 1.924 (0.000)	Data 0.002 (0.000)	
2021-06-23 09:46:18,577 Epoch: [2][735/1133]	Eit 3000  lr 0.0005  Le 9.9680 (10.0025)	Time 1.703 (0.000)	Data 0.002 (0.000)	
2021-06-23 09:52:04,708 Epoch: [2][935/1133]	Eit 3200  lr 0.0005  Le 10.0173 (10.0008)	Time 1.558 (0.000)	Data 0.002 (0.000)	
2021-06-23 09:57:46,772 Test: [0/40]	Le 10.1828 (10.1828)	Time 3.599 (0.000)	
2021-06-23 09:58:10,072 calculate similarity time: 0.07004499435424805
2021-06-23 09:58:10,500 Image to text: 66.3, 88.5, 94.2, 1.0, 3.6
2021-06-23 09:58:10,883 Text to image: 48.5, 77.6, 86.1, 2.0, 9.4
2021-06-23 09:58:10,883 Current rsum is 461.22
2021-06-23 09:58:14,603 runs/f30k_butd_region_bert/log
2021-06-23 09:58:14,603 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-23 09:58:14,605 image encoder trainable parameters: 20490144
2021-06-23 09:58:14,612 txt encoder trainable parameters: 137319072
2021-06-23 09:58:25,211 Epoch: [3][3/1133]	Eit 3400  lr 0.0005  Le 9.9380 (9.9545)	Time 1.924 (0.000)	Data 0.002 (0.000)	
2021-06-23 10:04:08,550 Epoch: [3][203/1133]	Eit 3600  lr 0.0005  Le 9.9981 (9.9719)	Time 1.885 (0.000)	Data 0.003 (0.000)	
2021-06-23 10:09:49,301 Epoch: [3][403/1133]	Eit 3800  lr 0.0005  Le 9.9580 (9.9701)	Time 1.430 (0.000)	Data 0.012 (0.000)	
2021-06-23 10:15:33,595 Epoch: [3][603/1133]	Eit 4000  lr 0.0005  Le 9.9746 (9.9697)	Time 2.041 (0.000)	Data 0.002 (0.000)	
2021-06-23 10:21:14,029 Epoch: [3][803/1133]	Eit 4200  lr 0.0005  Le 9.9784 (9.9693)	Time 1.853 (0.000)	Data 0.002 (0.000)	
2021-06-23 10:26:54,366 Epoch: [3][1003/1133]	Eit 4400  lr 0.0005  Le 9.9643 (9.9682)	Time 1.317 (0.000)	Data 0.002 (0.000)	
2021-06-23 10:30:34,676 Test: [0/40]	Le 10.1859 (10.1859)	Time 3.636 (0.000)	
2021-06-23 10:30:58,256 calculate similarity time: 0.06240105628967285
2021-06-23 10:30:58,720 Image to text: 66.3, 88.3, 93.4, 1.0, 4.0
2021-06-23 10:30:59,030 Text to image: 50.5, 79.2, 86.9, 1.0, 9.2
2021-06-23 10:30:59,031 Current rsum is 464.6
2021-06-23 10:31:02,546 runs/f30k_butd_region_bert/log
2021-06-23 10:31:02,546 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-23 10:31:02,549 image encoder trainable parameters: 20490144
2021-06-23 10:31:02,561 txt encoder trainable parameters: 137319072
2021-06-23 10:33:10,075 Epoch: [4][71/1133]	Eit 4600  lr 0.0005  Le 9.9257 (9.9409)	Time 1.705 (0.000)	Data 0.002 (0.000)	
2021-06-23 10:38:52,442 Epoch: [4][271/1133]	Eit 4800  lr 0.0005  Le 9.9616 (9.9423)	Time 1.872 (0.000)	Data 0.002 (0.000)	
2021-06-23 10:44:33,998 Epoch: [4][471/1133]	Eit 5000  lr 0.0005  Le 9.9568 (9.9431)	Time 1.843 (0.000)	Data 0.002 (0.000)	
2021-06-23 10:50:15,367 Epoch: [4][671/1133]	Eit 5200  lr 0.0005  Le 9.9429 (9.9430)	Time 2.037 (0.000)	Data 0.002 (0.000)	
2021-06-23 10:55:56,555 Epoch: [4][871/1133]	Eit 5400  lr 0.0005  Le 9.9374 (9.9432)	Time 1.915 (0.000)	Data 0.002 (0.000)	
2021-06-23 11:01:33,672 Epoch: [4][1071/1133]	Eit 5600  lr 0.0005  Le 9.8985 (9.9430)	Time 1.598 (0.000)	Data 0.002 (0.000)	
2021-06-23 11:02:55,763 Test: [0/40]	Le 10.1922 (10.1922)	Time 3.715 (0.000)	
2021-06-23 11:03:19,517 calculate similarity time: 0.06644392013549805
2021-06-23 11:03:20,025 Image to text: 68.8, 90.0, 95.6, 1.0, 3.4
2021-06-23 11:03:20,512 Text to image: 51.7, 80.8, 87.8, 1.0, 8.2
2021-06-23 11:03:20,512 Current rsum is 474.72
2021-06-23 11:03:24,094 runs/f30k_butd_region_bert/log
2021-06-23 11:03:24,094 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-23 11:03:24,098 image encoder trainable parameters: 20490144
2021-06-23 11:03:24,112 txt encoder trainable parameters: 137319072
2021-06-23 11:07:29,357 Epoch: [5][139/1133]	Eit 5800  lr 0.0005  Le 9.9109 (9.9221)	Time 1.743 (0.000)	Data 0.002 (0.000)	
2021-06-23 11:13:09,918 Epoch: [5][339/1133]	Eit 6000  lr 0.0005  Le 9.9269 (9.9233)	Time 1.782 (0.000)	Data 0.002 (0.000)	
2021-06-23 11:18:55,340 Epoch: [5][539/1133]	Eit 6200  lr 0.0005  Le 9.9563 (9.9238)	Time 1.763 (0.000)	Data 0.002 (0.000)	
2021-06-23 11:24:35,470 Epoch: [5][739/1133]	Eit 6400  lr 0.0005  Le 9.9785 (9.9242)	Time 1.380 (0.000)	Data 0.002 (0.000)	
2021-06-23 11:30:19,322 Epoch: [5][939/1133]	Eit 6600  lr 0.0005  Le 9.9499 (9.9247)	Time 1.541 (0.000)	Data 0.002 (0.000)	
2021-06-23 11:36:01,149 Test: [0/40]	Le 10.1844 (10.1843)	Time 3.827 (0.000)	
2021-06-23 11:36:24,654 calculate similarity time: 0.09616684913635254
2021-06-23 11:36:25,099 Image to text: 70.1, 90.6, 95.2, 1.0, 3.9
2021-06-23 11:36:25,414 Text to image: 52.9, 81.1, 88.3, 1.0, 8.6
2021-06-23 11:36:25,414 Current rsum is 478.21999999999997
2021-06-23 11:36:28,834 runs/f30k_butd_region_bert/log
2021-06-23 11:36:28,834 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-23 11:36:28,836 image encoder trainable parameters: 20490144
2021-06-23 11:36:28,844 txt encoder trainable parameters: 137319072
2021-06-23 11:36:45,228 Epoch: [6][7/1133]	Eit 6800  lr 0.0005  Le 9.8943 (9.9061)	Time 1.722 (0.000)	Data 0.002 (0.000)	
2021-06-23 11:42:26,717 Epoch: [6][207/1133]	Eit 7000  lr 0.0005  Le 9.9069 (9.9056)	Time 1.655 (0.000)	Data 0.002 (0.000)	
2021-06-23 11:48:13,177 Epoch: [6][407/1133]	Eit 7200  lr 0.0005  Le 9.8954 (9.9078)	Time 1.745 (0.000)	Data 0.003 (0.000)	
2021-06-23 11:53:53,105 Epoch: [6][607/1133]	Eit 7400  lr 0.0005  Le 9.9485 (9.9089)	Time 1.398 (0.000)	Data 0.002 (0.000)	
2021-06-23 11:59:34,726 Epoch: [6][807/1133]	Eit 7600  lr 0.0005  Le 9.8856 (9.9093)	Time 2.254 (0.000)	Data 0.003 (0.000)	
2021-06-23 12:05:22,427 Epoch: [6][1007/1133]	Eit 7800  lr 0.0005  Le 9.9089 (9.9092)	Time 1.534 (0.000)	Data 0.002 (0.000)	
2021-06-23 12:09:04,850 Test: [0/40]	Le 10.1869 (10.1869)	Time 4.135 (0.000)	
2021-06-23 12:09:29,286 calculate similarity time: 0.09146857261657715
2021-06-23 12:09:29,693 Image to text: 69.7, 91.3, 95.2, 1.0, 3.3
2021-06-23 12:09:30,004 Text to image: 52.6, 81.7, 88.6, 1.0, 8.2
2021-06-23 12:09:30,004 Current rsum is 479.04
2021-06-23 12:09:33,659 runs/f30k_butd_region_bert/log
2021-06-23 12:09:33,659 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-23 12:09:33,662 image encoder trainable parameters: 20490144
2021-06-23 12:09:33,673 txt encoder trainable parameters: 137319072
2021-06-23 12:11:46,962 Epoch: [7][75/1133]	Eit 8000  lr 0.0005  Le 9.9117 (9.8956)	Time 1.433 (0.000)	Data 0.008 (0.000)	
2021-06-23 12:17:35,532 Epoch: [7][275/1133]	Eit 8200  lr 0.0005  Le 9.8881 (9.8935)	Time 2.168 (0.000)	Data 0.003 (0.000)	
2021-06-23 12:22:54,843 Epoch: [7][475/1133]	Eit 8400  lr 0.0005  Le 9.8883 (9.8955)	Time 1.969 (0.000)	Data 0.002 (0.000)	
2021-06-23 12:28:39,760 Epoch: [7][675/1133]	Eit 8600  lr 0.0005  Le 9.9102 (9.8954)	Time 1.407 (0.000)	Data 0.002 (0.000)	
2021-06-23 12:34:22,383 Epoch: [7][875/1133]	Eit 8800  lr 0.0005  Le 9.9000 (9.8959)	Time 1.332 (0.000)	Data 0.002 (0.000)	
2021-06-23 12:40:12,638 Epoch: [7][1075/1133]	Eit 9000  lr 0.0005  Le 9.8729 (9.8962)	Time 1.927 (0.000)	Data 0.009 (0.000)	
2021-06-23 12:41:54,302 Test: [0/40]	Le 10.1949 (10.1949)	Time 4.205 (0.000)	
2021-06-23 12:42:19,720 calculate similarity time: 0.07376217842102051
2021-06-23 12:42:20,302 Image to text: 69.4, 89.9, 94.3, 1.0, 3.8
2021-06-23 12:42:20,758 Text to image: 51.7, 81.0, 88.3, 1.0, 9.2
2021-06-23 12:42:20,758 Current rsum is 474.56000000000006
2021-06-23 12:42:22,771 runs/f30k_butd_region_bert/log
2021-06-23 12:42:22,771 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-23 12:42:22,774 image encoder trainable parameters: 20490144
2021-06-23 12:42:22,785 txt encoder trainable parameters: 137319072
2021-06-23 12:46:35,126 Epoch: [8][143/1133]	Eit 9200  lr 0.0005  Le 9.8469 (9.8810)	Time 1.497 (0.000)	Data 0.002 (0.000)	
2021-06-23 12:52:18,528 Epoch: [8][343/1133]	Eit 9400  lr 0.0005  Le 9.9421 (9.8840)	Time 1.847 (0.000)	Data 0.002 (0.000)	
2021-06-23 12:58:00,372 Epoch: [8][543/1133]	Eit 9600  lr 0.0005  Le 9.9065 (9.8852)	Time 1.442 (0.000)	Data 0.003 (0.000)	
2021-06-23 13:03:44,279 Epoch: [8][743/1133]	Eit 9800  lr 0.0005  Le 9.9221 (9.8852)	Time 1.812 (0.000)	Data 0.002 (0.000)	
2021-06-23 13:09:28,033 Epoch: [8][943/1133]	Eit 10000  lr 0.0005  Le 9.9115 (9.8857)	Time 1.402 (0.000)	Data 0.002 (0.000)	
2021-06-23 13:14:56,421 Test: [0/40]	Le 10.1978 (10.1978)	Time 3.583 (0.000)	
2021-06-23 13:15:19,947 calculate similarity time: 0.07182884216308594
2021-06-23 13:15:20,446 Image to text: 69.3, 90.3, 94.6, 1.0, 3.5
2021-06-23 13:15:20,770 Text to image: 52.8, 80.8, 88.4, 1.0, 9.1
2021-06-23 13:15:20,771 Current rsum is 476.22
2021-06-23 13:15:22,329 runs/f30k_butd_region_bert/log
2021-06-23 13:15:22,329 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-23 13:15:22,331 image encoder trainable parameters: 20490144
2021-06-23 13:15:22,337 txt encoder trainable parameters: 137319072
2021-06-23 13:15:46,631 Epoch: [9][11/1133]	Eit 10200  lr 0.0005  Le 9.9035 (9.8714)	Time 1.667 (0.000)	Data 0.002 (0.000)	
2021-06-23 13:21:23,075 Epoch: [9][211/1133]	Eit 10400  lr 0.0005  Le 9.8899 (9.8729)	Time 2.011 (0.000)	Data 0.004 (0.000)	
2021-06-23 13:27:11,698 Epoch: [9][411/1133]	Eit 10600  lr 0.0005  Le 9.8495 (9.8744)	Time 1.755 (0.000)	Data 0.002 (0.000)	
2021-06-23 13:32:52,788 Epoch: [9][611/1133]	Eit 10800  lr 0.0005  Le 9.9180 (9.8751)	Time 1.916 (0.000)	Data 0.002 (0.000)	
2021-06-23 13:38:06,826 Epoch: [9][811/1133]	Eit 11000  lr 0.0005  Le 9.8769 (9.8754)	Time 1.898 (0.000)	Data 0.002 (0.000)	
2021-06-23 13:43:51,876 Epoch: [9][1011/1133]	Eit 11200  lr 0.0005  Le 9.8428 (9.8757)	Time 1.381 (0.000)	Data 0.002 (0.000)	
2021-06-23 13:47:21,423 Test: [0/40]	Le 10.1935 (10.1935)	Time 3.863 (0.000)	
2021-06-23 13:47:46,178 calculate similarity time: 0.06514668464660645
2021-06-23 13:47:46,615 Image to text: 70.6, 90.7, 95.9, 1.0, 3.3
2021-06-23 13:47:46,946 Text to image: 54.2, 81.9, 88.7, 1.0, 8.9
2021-06-23 13:47:46,946 Current rsum is 482.04
2021-06-23 13:47:50,554 runs/f30k_butd_region_bert/log
2021-06-23 13:47:50,555 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-23 13:47:50,558 image encoder trainable parameters: 20490144
2021-06-23 13:47:50,569 txt encoder trainable parameters: 137319072
2021-06-23 13:50:12,907 Epoch: [10][79/1133]	Eit 11400  lr 0.0005  Le 9.8473 (9.8625)	Time 1.798 (0.000)	Data 0.002 (0.000)	
2021-06-23 13:55:55,447 Epoch: [10][279/1133]	Eit 11600  lr 0.0005  Le 9.8516 (9.8636)	Time 1.930 (0.000)	Data 0.002 (0.000)	
2021-06-23 14:01:37,148 Epoch: [10][479/1133]	Eit 11800  lr 0.0005  Le 9.8515 (9.8655)	Time 1.391 (0.000)	Data 0.002 (0.000)	
2021-06-23 14:07:21,277 Epoch: [10][679/1133]	Eit 12000  lr 0.0005  Le 9.8611 (9.8658)	Time 2.152 (0.000)	Data 0.002 (0.000)	
2021-06-23 14:13:04,587 Epoch: [10][879/1133]	Eit 12200  lr 0.0005  Le 9.8707 (9.8666)	Time 1.879 (0.000)	Data 0.005 (0.000)	
2021-06-23 14:18:45,947 Epoch: [10][1079/1133]	Eit 12400  lr 0.0005  Le 9.9046 (9.8671)	Time 2.096 (0.000)	Data 0.002 (0.000)	
2021-06-23 14:20:19,445 Test: [0/40]	Le 10.1862 (10.1862)	Time 4.291 (0.000)	
2021-06-23 14:20:43,775 calculate similarity time: 0.07200336456298828
2021-06-23 14:20:44,217 Image to text: 70.9, 90.8, 95.2, 1.0, 3.7
2021-06-23 14:20:44,544 Text to image: 54.2, 81.0, 88.0, 1.0, 9.3
2021-06-23 14:20:44,544 Current rsum is 479.99999999999994
2021-06-23 14:20:46,104 runs/f30k_butd_region_bert/log
2021-06-23 14:20:46,104 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-23 14:20:46,106 image encoder trainable parameters: 20490144
2021-06-23 14:20:46,112 txt encoder trainable parameters: 137319072
2021-06-23 14:25:03,081 Epoch: [11][147/1133]	Eit 12600  lr 0.0005  Le 9.8623 (9.8572)	Time 2.071 (0.000)	Data 0.002 (0.000)	
2021-06-23 14:30:43,459 Epoch: [11][347/1133]	Eit 12800  lr 0.0005  Le 9.9014 (9.8581)	Time 1.442 (0.000)	Data 0.002 (0.000)	
2021-06-23 14:36:25,662 Epoch: [11][547/1133]	Eit 13000  lr 0.0005  Le 9.8595 (9.8581)	Time 1.995 (0.000)	Data 0.002 (0.000)	
2021-06-23 14:42:09,845 Epoch: [11][747/1133]	Eit 13200  lr 0.0005  Le 9.8694 (9.8593)	Time 1.302 (0.000)	Data 0.002 (0.000)	
2021-06-23 14:47:45,730 Epoch: [11][947/1133]	Eit 13400  lr 0.0005  Le 9.8583 (9.8600)	Time 1.671 (0.000)	Data 0.002 (0.000)	
2021-06-23 14:53:04,712 Test: [0/40]	Le 10.1986 (10.1986)	Time 3.685 (0.000)	
2021-06-23 14:53:28,419 calculate similarity time: 0.06986784934997559
2021-06-23 14:53:28,962 Image to text: 69.9, 90.2, 94.7, 1.0, 3.7
2021-06-23 14:53:29,396 Text to image: 53.9, 81.4, 88.3, 1.0, 9.7
2021-06-23 14:53:29,396 Current rsum is 478.32000000000005
2021-06-23 14:53:30,880 runs/f30k_butd_region_bert/log
2021-06-23 14:53:30,880 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-23 14:53:30,882 image encoder trainable parameters: 20490144
2021-06-23 14:53:30,888 txt encoder trainable parameters: 137319072
2021-06-23 14:53:52,896 Epoch: [12][15/1133]	Eit 13600  lr 0.0005  Le 9.8727 (9.8481)	Time 0.807 (0.000)	Data 0.002 (0.000)	
2021-06-23 14:59:20,993 Epoch: [12][215/1133]	Eit 13800  lr 0.0005  Le 9.8128 (9.8477)	Time 1.878 (0.000)	Data 0.002 (0.000)	
2021-06-23 15:05:04,354 Epoch: [12][415/1133]	Eit 14000  lr 0.0005  Le 9.8545 (9.8496)	Time 2.021 (0.000)	Data 0.002 (0.000)	
2021-06-23 15:10:41,069 Epoch: [12][615/1133]	Eit 14200  lr 0.0005  Le 9.8586 (9.8512)	Time 1.879 (0.000)	Data 0.002 (0.000)	
2021-06-23 15:16:22,053 Epoch: [12][815/1133]	Eit 14400  lr 0.0005  Le 9.8634 (9.8519)	Time 1.454 (0.000)	Data 0.002 (0.000)	
2021-06-23 15:22:01,554 Epoch: [12][1015/1133]	Eit 14600  lr 0.0005  Le 9.8548 (9.8525)	Time 1.942 (0.000)	Data 0.002 (0.000)	
2021-06-23 15:25:27,518 Test: [0/40]	Le 10.2034 (10.2034)	Time 3.663 (0.000)	
2021-06-23 15:25:50,827 calculate similarity time: 0.06763410568237305
2021-06-23 15:25:51,229 Image to text: 71.1, 91.0, 95.6, 1.0, 3.2
2021-06-23 15:25:51,540 Text to image: 54.2, 80.6, 87.6, 1.0, 9.2
2021-06-23 15:25:51,540 Current rsum is 480.14
2021-06-23 15:25:52,989 runs/f30k_butd_region_bert/log
2021-06-23 15:25:52,989 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-23 15:25:52,991 image encoder trainable parameters: 20490144
2021-06-23 15:25:52,996 txt encoder trainable parameters: 137319072
2021-06-23 15:28:19,757 Epoch: [13][83/1133]	Eit 14800  lr 0.0005  Le 9.8062 (9.8410)	Time 2.195 (0.000)	Data 0.002 (0.000)	
2021-06-23 15:34:04,916 Epoch: [13][283/1133]	Eit 15000  lr 0.0005  Le 9.8530 (9.8431)	Time 1.390 (0.000)	Data 0.002 (0.000)	
2021-06-23 15:39:49,292 Epoch: [13][483/1133]	Eit 15200  lr 0.0005  Le 9.8467 (9.8447)	Time 1.820 (0.000)	Data 0.002 (0.000)	
2021-06-23 15:45:31,698 Epoch: [13][683/1133]	Eit 15400  lr 0.0005  Le 9.8294 (9.8452)	Time 1.879 (0.000)	Data 0.002 (0.000)	
2021-06-23 15:51:11,018 Epoch: [13][883/1133]	Eit 15600  lr 0.0005  Le 9.8893 (9.8459)	Time 1.857 (0.000)	Data 0.003 (0.000)	
2021-06-23 15:56:52,484 Epoch: [13][1083/1133]	Eit 15800  lr 0.0005  Le 9.8395 (9.8465)	Time 1.907 (0.000)	Data 0.003 (0.000)	
2021-06-23 15:58:20,068 Test: [0/40]	Le 10.1889 (10.1888)	Time 3.527 (0.000)	
2021-06-23 15:58:43,811 calculate similarity time: 0.08434939384460449
2021-06-23 15:58:44,301 Image to text: 70.0, 90.6, 95.1, 1.0, 3.1
2021-06-23 15:58:44,613 Text to image: 54.8, 80.7, 87.9, 1.0, 9.8
2021-06-23 15:58:44,613 Current rsum is 479.1
2021-06-23 15:58:46,103 runs/f30k_butd_region_bert/log
2021-06-23 15:58:46,103 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-23 15:58:46,105 image encoder trainable parameters: 20490144
2021-06-23 15:58:46,110 txt encoder trainable parameters: 137319072
2021-06-23 16:03:09,943 Epoch: [14][151/1133]	Eit 16000  lr 0.0005  Le 9.8197 (9.8376)	Time 1.302 (0.000)	Data 0.002 (0.000)	
2021-06-23 16:08:50,332 Epoch: [14][351/1133]	Eit 16200  lr 0.0005  Le 9.8697 (9.8377)	Time 1.535 (0.000)	Data 0.002 (0.000)	
2021-06-23 16:14:05,045 Epoch: [14][551/1133]	Eit 16400  lr 0.0005  Le 9.8630 (9.8385)	Time 1.588 (0.000)	Data 0.002 (0.000)	
2021-06-23 16:19:43,325 Epoch: [14][751/1133]	Eit 16600  lr 0.0005  Le 9.8335 (9.8395)	Time 1.346 (0.000)	Data 0.002 (0.000)	
2021-06-23 16:25:24,750 Epoch: [14][951/1133]	Eit 16800  lr 0.0005  Le 9.8452 (9.8402)	Time 2.218 (0.000)	Data 0.002 (0.000)	
2021-06-23 16:30:36,629 Test: [0/40]	Le 10.1945 (10.1944)	Time 3.647 (0.000)	
2021-06-23 16:31:01,420 calculate similarity time: 0.07606220245361328
2021-06-23 16:31:01,930 Image to text: 70.6, 91.1, 95.7, 1.0, 3.1
2021-06-23 16:31:02,260 Text to image: 55.1, 80.4, 87.7, 1.0, 10.4
2021-06-23 16:31:02,260 Current rsum is 480.58
2021-06-23 16:31:03,725 runs/f30k_butd_region_bert/log
2021-06-23 16:31:03,726 runs/f30k_butd_region_bert
2021-06-23 16:31:03,726 Current epoch num is 15, decrease all lr by 10
2021-06-23 16:31:03,726 new lr 5e-05
2021-06-23 16:31:03,726 new lr 5e-06
2021-06-23 16:31:03,726 new lr 5e-05
Use VSE++ objective.
2021-06-23 16:31:03,728 image encoder trainable parameters: 20490144
2021-06-23 16:31:03,733 txt encoder trainable parameters: 137319072
2021-06-23 16:31:41,412 Epoch: [15][19/1133]	Eit 17000  lr 5e-05  Le 9.7934 (9.8264)	Time 1.852 (0.000)	Data 0.003 (0.000)	
2021-06-23 16:37:24,575 Epoch: [15][219/1133]	Eit 17200  lr 5e-05  Le 9.8305 (9.8250)	Time 1.695 (0.000)	Data 0.002 (0.000)	
2021-06-23 16:43:06,354 Epoch: [15][419/1133]	Eit 17400  lr 5e-05  Le 9.8160 (9.8225)	Time 1.710 (0.000)	Data 0.002 (0.000)	
2021-06-23 16:48:45,596 Epoch: [15][619/1133]	Eit 17600  lr 5e-05  Le 9.8064 (9.8202)	Time 1.886 (0.000)	Data 0.002 (0.000)	
2021-06-23 16:54:24,909 Epoch: [15][819/1133]	Eit 17800  lr 5e-05  Le 9.8086 (9.8194)	Time 1.855 (0.000)	Data 0.002 (0.000)	
2021-06-23 17:00:07,347 Epoch: [15][1019/1133]	Eit 18000  lr 5e-05  Le 9.8546 (9.8185)	Time 1.847 (0.000)	Data 0.002 (0.000)	
2021-06-23 17:03:22,385 Test: [0/40]	Le 10.1907 (10.1906)	Time 3.847 (0.000)	
2021-06-23 17:03:46,632 calculate similarity time: 0.10212874412536621
2021-06-23 17:03:47,171 Image to text: 72.6, 92.7, 96.5, 1.0, 2.6
2021-06-23 17:03:47,609 Text to image: 56.3, 81.3, 88.3, 1.0, 9.5
2021-06-23 17:03:47,609 Current rsum is 487.78
2021-06-23 17:03:51,704 runs/f30k_butd_region_bert/log
2021-06-23 17:03:51,704 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-23 17:03:51,706 image encoder trainable parameters: 20490144
2021-06-23 17:03:51,714 txt encoder trainable parameters: 137319072
2021-06-23 17:06:24,121 Epoch: [16][87/1133]	Eit 18200  lr 5e-05  Le 9.8351 (9.8144)	Time 1.417 (0.000)	Data 0.002 (0.000)	
2021-06-23 17:12:07,637 Epoch: [16][287/1133]	Eit 18400  lr 5e-05  Le 9.8372 (9.8133)	Time 1.937 (0.000)	Data 0.002 (0.000)	
2021-06-23 17:17:53,023 Epoch: [16][487/1133]	Eit 18600  lr 5e-05  Le 9.7976 (9.8136)	Time 1.952 (0.000)	Data 0.003 (0.000)	
2021-06-23 17:23:30,946 Epoch: [16][687/1133]	Eit 18800  lr 5e-05  Le 9.8350 (9.8128)	Time 1.383 (0.000)	Data 0.002 (0.000)	
2021-06-23 17:28:56,700 Epoch: [16][887/1133]	Eit 19000  lr 5e-05  Le 9.8240 (9.8128)	Time 1.836 (0.000)	Data 0.003 (0.000)	
2021-06-23 17:34:27,197 Epoch: [16][1087/1133]	Eit 19200  lr 5e-05  Le 9.8166 (9.8128)	Time 1.943 (0.000)	Data 0.003 (0.000)	
2021-06-23 17:35:46,834 Test: [0/40]	Le 10.1924 (10.1924)	Time 3.405 (0.000)	
2021-06-23 17:36:10,324 calculate similarity time: 0.07311081886291504
2021-06-23 17:36:10,806 Image to text: 72.4, 91.9, 96.1, 1.0, 2.6
2021-06-23 17:36:11,124 Text to image: 56.7, 81.7, 88.3, 1.0, 9.3
2021-06-23 17:36:11,124 Current rsum is 487.09999999999997
2021-06-23 17:36:12,604 runs/f30k_butd_region_bert/log
2021-06-23 17:36:12,605 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-23 17:36:12,606 image encoder trainable parameters: 20490144
2021-06-23 17:36:12,611 txt encoder trainable parameters: 137319072
2021-06-23 17:40:41,259 Epoch: [17][155/1133]	Eit 19400  lr 5e-05  Le 9.8162 (9.8107)	Time 1.759 (0.000)	Data 0.003 (0.000)	
2021-06-23 17:46:23,552 Epoch: [17][355/1133]	Eit 19600  lr 5e-05  Le 9.8171 (9.8082)	Time 1.665 (0.000)	Data 0.002 (0.000)	
2021-06-23 17:52:02,535 Epoch: [17][555/1133]	Eit 19800  lr 5e-05  Le 9.8039 (9.8085)	Time 1.592 (0.000)	Data 0.002 (0.000)	
2021-06-23 17:57:43,751 Epoch: [17][755/1133]	Eit 20000  lr 5e-05  Le 9.8019 (9.8089)	Time 1.451 (0.000)	Data 0.002 (0.000)	
2021-06-23 18:03:29,041 Epoch: [17][955/1133]	Eit 20200  lr 5e-05  Le 9.8196 (9.8088)	Time 1.665 (0.000)	Data 0.002 (0.000)	
2021-06-23 18:08:31,502 Test: [0/40]	Le 10.1936 (10.1936)	Time 3.674 (0.000)	
2021-06-23 18:08:55,301 calculate similarity time: 0.06900572776794434
2021-06-23 18:08:55,840 Image to text: 72.5, 91.8, 95.8, 1.0, 2.7
2021-06-23 18:08:56,185 Text to image: 56.7, 81.4, 88.5, 1.0, 9.2
2021-06-23 18:08:56,186 Current rsum is 486.59999999999997
2021-06-23 18:08:57,655 runs/f30k_butd_region_bert/log
2021-06-23 18:08:57,656 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-23 18:08:57,657 image encoder trainable parameters: 20490144
2021-06-23 18:08:57,663 txt encoder trainable parameters: 137319072
2021-06-23 18:09:41,043 Epoch: [18][23/1133]	Eit 20400  lr 5e-05  Le 9.7960 (9.8043)	Time 1.743 (0.000)	Data 0.002 (0.000)	
2021-06-23 18:15:22,774 Epoch: [18][223/1133]	Eit 20600  lr 5e-05  Le 9.8140 (9.8064)	Time 1.988 (0.000)	Data 0.002 (0.000)	
2021-06-23 18:21:03,245 Epoch: [18][423/1133]	Eit 20800  lr 5e-05  Le 9.7859 (9.8066)	Time 1.887 (0.000)	Data 0.002 (0.000)	
2021-06-23 18:26:44,277 Epoch: [18][623/1133]	Eit 21000  lr 5e-05  Le 9.7997 (9.8064)	Time 1.766 (0.000)	Data 0.002 (0.000)	
2021-06-23 18:32:24,115 Epoch: [18][823/1133]	Eit 21200  lr 5e-05  Le 9.7976 (9.8065)	Time 1.733 (0.000)	Data 0.002 (0.000)	
2021-06-23 18:38:02,567 Epoch: [18][1023/1133]	Eit 21400  lr 5e-05  Le 9.8189 (9.8063)	Time 1.883 (0.000)	Data 0.002 (0.000)	
2021-06-23 18:41:11,319 Test: [0/40]	Le 10.1952 (10.1952)	Time 3.605 (0.000)	
2021-06-23 18:41:34,889 calculate similarity time: 0.08556962013244629
2021-06-23 18:41:35,314 Image to text: 71.2, 92.3, 96.5, 1.0, 2.6
2021-06-23 18:41:35,745 Text to image: 56.4, 81.5, 88.2, 1.0, 9.5
2021-06-23 18:41:35,746 Current rsum is 486.06000000000006
2021-06-23 18:41:37,346 runs/f30k_butd_region_bert/log
2021-06-23 18:41:37,346 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-23 18:41:37,348 image encoder trainable parameters: 20490144
2021-06-23 18:41:37,354 txt encoder trainable parameters: 137319072
2021-06-23 18:44:16,511 Epoch: [19][91/1133]	Eit 21600  lr 5e-05  Le 9.8056 (9.8051)	Time 2.101 (0.000)	Data 0.009 (0.000)	
2021-06-23 18:49:28,971 Epoch: [19][291/1133]	Eit 21800  lr 5e-05  Le 9.8197 (9.8047)	Time 1.876 (0.000)	Data 0.003 (0.000)	
2021-06-23 18:55:08,860 Epoch: [19][491/1133]	Eit 22000  lr 5e-05  Le 9.7862 (9.8043)	Time 1.709 (0.000)	Data 0.002 (0.000)	
2021-06-23 19:00:48,813 Epoch: [19][691/1133]	Eit 22200  lr 5e-05  Le 9.8005 (9.8043)	Time 1.824 (0.000)	Data 0.002 (0.000)	
2021-06-23 19:06:26,880 Epoch: [19][891/1133]	Eit 22400  lr 5e-05  Le 9.7982 (9.8044)	Time 1.992 (0.000)	Data 0.002 (0.000)	
2021-06-23 19:12:08,078 Epoch: [19][1091/1133]	Eit 22600  lr 5e-05  Le 9.8080 (9.8045)	Time 1.411 (0.000)	Data 0.002 (0.000)	
2021-06-23 19:13:21,422 Test: [0/40]	Le 10.1966 (10.1966)	Time 3.691 (0.000)	
2021-06-23 19:13:45,032 calculate similarity time: 0.058144330978393555
2021-06-23 19:13:45,545 Image to text: 72.2, 92.8, 96.3, 1.0, 2.7
2021-06-23 19:13:45,870 Text to image: 56.8, 81.7, 88.3, 1.0, 9.5
2021-06-23 19:13:45,870 Current rsum is 488.1
2021-06-23 19:13:49,447 runs/f30k_butd_region_bert/log
2021-06-23 19:13:49,447 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-23 19:13:49,450 image encoder trainable parameters: 20490144
2021-06-23 19:13:49,457 txt encoder trainable parameters: 137319072
2021-06-23 19:18:22,914 Epoch: [20][159/1133]	Eit 22800  lr 5e-05  Le 9.7684 (9.8027)	Time 1.818 (0.000)	Data 0.005 (0.000)	
2021-06-23 19:24:04,428 Epoch: [20][359/1133]	Eit 23000  lr 5e-05  Le 9.7955 (9.8032)	Time 1.690 (0.000)	Data 0.002 (0.000)	
2021-06-23 19:29:42,254 Epoch: [20][559/1133]	Eit 23200  lr 5e-05  Le 9.8177 (9.8032)	Time 1.987 (0.000)	Data 0.002 (0.000)	
2021-06-23 19:35:23,878 Epoch: [20][759/1133]	Eit 23400  lr 5e-05  Le 9.7928 (9.8029)	Time 1.812 (0.000)	Data 0.002 (0.000)	
2021-06-23 19:41:04,569 Epoch: [20][959/1133]	Eit 23600  lr 5e-05  Le 9.7949 (9.8030)	Time 1.744 (0.000)	Data 0.003 (0.000)	
2021-06-23 19:45:58,557 Test: [0/40]	Le 10.1945 (10.1945)	Time 3.553 (0.000)	
2021-06-23 19:46:22,644 calculate similarity time: 0.0750124454498291
2021-06-23 19:46:23,086 Image to text: 72.3, 92.5, 96.2, 1.0, 2.7
2021-06-23 19:46:23,518 Text to image: 56.7, 81.7, 88.2, 1.0, 9.6
2021-06-23 19:46:23,518 Current rsum is 487.53999999999996
2021-06-23 19:46:25,061 runs/f30k_butd_region_bert/log
2021-06-23 19:46:25,061 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-23 19:46:25,062 image encoder trainable parameters: 20490144
2021-06-23 19:46:25,068 txt encoder trainable parameters: 137319072
2021-06-23 19:47:17,162 Epoch: [21][27/1133]	Eit 23800  lr 5e-05  Le 9.8149 (9.8024)	Time 2.092 (0.000)	Data 0.002 (0.000)	
2021-06-23 19:52:57,516 Epoch: [21][227/1133]	Eit 24000  lr 5e-05  Le 9.8032 (9.8014)	Time 1.838 (0.000)	Data 0.002 (0.000)	
2021-06-23 19:58:35,144 Epoch: [21][427/1133]	Eit 24200  lr 5e-05  Le 9.7946 (9.8014)	Time 2.017 (0.000)	Data 0.002 (0.000)	
2021-06-23 20:04:05,378 Epoch: [21][627/1133]	Eit 24400  lr 5e-05  Le 9.8152 (9.8010)	Time 0.806 (0.000)	Data 0.002 (0.000)	
2021-06-23 20:09:38,534 Epoch: [21][827/1133]	Eit 24600  lr 5e-05  Le 9.8148 (9.8010)	Time 1.940 (0.000)	Data 0.003 (0.000)	
2021-06-23 20:15:26,519 Epoch: [21][1027/1133]	Eit 24800  lr 5e-05  Le 9.7876 (9.8010)	Time 1.387 (0.000)	Data 0.002 (0.000)	
2021-06-23 20:18:29,966 Test: [0/40]	Le 10.1933 (10.1933)	Time 3.884 (0.000)	
2021-06-23 20:18:53,910 calculate similarity time: 0.07108688354492188
2021-06-23 20:18:54,427 Image to text: 72.9, 92.2, 96.5, 1.0, 2.7
2021-06-23 20:18:54,885 Text to image: 56.9, 82.0, 88.5, 1.0, 9.6
2021-06-23 20:18:54,885 Current rsum is 489.02000000000004
2021-06-23 20:18:58,616 runs/f30k_butd_region_bert/log
2021-06-23 20:18:58,616 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-23 20:18:58,619 image encoder trainable parameters: 20490144
2021-06-23 20:18:58,630 txt encoder trainable parameters: 137319072
2021-06-23 20:21:45,835 Epoch: [22][95/1133]	Eit 25000  lr 5e-05  Le 9.8101 (9.7995)	Time 1.506 (0.000)	Data 0.002 (0.000)	
2021-06-23 20:27:32,618 Epoch: [22][295/1133]	Eit 25200  lr 5e-05  Le 9.8059 (9.7995)	Time 1.896 (0.000)	Data 0.002 (0.000)	
2021-06-23 20:33:11,187 Epoch: [22][495/1133]	Eit 25400  lr 5e-05  Le 9.8205 (9.7996)	Time 1.667 (0.000)	Data 0.003 (0.000)	
2021-06-23 20:38:55,765 Epoch: [22][695/1133]	Eit 25600  lr 5e-05  Le 9.8087 (9.7995)	Time 1.643 (0.000)	Data 0.003 (0.000)	
2021-06-23 20:44:36,895 Epoch: [22][895/1133]	Eit 25800  lr 5e-05  Le 9.8019 (9.7994)	Time 2.109 (0.000)	Data 0.002 (0.000)	
2021-06-23 20:50:25,835 Epoch: [22][1095/1133]	Eit 26000  lr 5e-05  Le 9.7934 (9.7995)	Time 1.891 (0.000)	Data 0.003 (0.000)	
2021-06-23 20:51:30,660 Test: [0/40]	Le 10.1962 (10.1962)	Time 3.769 (0.000)	
2021-06-23 20:51:55,620 calculate similarity time: 0.06472659111022949
2021-06-23 20:51:56,119 Image to text: 72.9, 91.3, 96.6, 1.0, 2.8
2021-06-23 20:51:56,518 Text to image: 56.5, 81.3, 88.2, 1.0, 9.9
2021-06-23 20:51:56,518 Current rsum is 486.8
2021-06-23 20:51:58,208 runs/f30k_butd_region_bert/log
2021-06-23 20:51:58,208 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-23 20:51:58,210 image encoder trainable parameters: 20490144
2021-06-23 20:51:58,215 txt encoder trainable parameters: 137319072
2021-06-23 20:56:43,814 Epoch: [23][163/1133]	Eit 26200  lr 5e-05  Le 9.7969 (9.8003)	Time 1.458 (0.000)	Data 0.011 (0.000)	
2021-06-23 21:02:30,534 Epoch: [23][363/1133]	Eit 26400  lr 5e-05  Le 9.8093 (9.7996)	Time 1.866 (0.000)	Data 0.003 (0.000)	
2021-06-23 21:08:16,342 Epoch: [23][563/1133]	Eit 26600  lr 5e-05  Le 9.7939 (9.7991)	Time 1.443 (0.000)	Data 0.003 (0.000)	
2021-06-23 21:13:57,282 Epoch: [23][763/1133]	Eit 26800  lr 5e-05  Le 9.8070 (9.7988)	Time 1.676 (0.000)	Data 0.002 (0.000)	
2021-06-23 21:19:38,578 Epoch: [23][963/1133]	Eit 27000  lr 5e-05  Le 9.8253 (9.7987)	Time 1.473 (0.000)	Data 0.003 (0.000)	
2021-06-23 21:24:03,078 Test: [0/40]	Le 10.1961 (10.1961)	Time 3.970 (0.000)	
2021-06-23 21:24:28,154 calculate similarity time: 0.06942892074584961
2021-06-23 21:24:28,695 Image to text: 72.5, 92.1, 96.6, 1.0, 2.8
2021-06-23 21:24:29,005 Text to image: 56.6, 81.4, 88.1, 1.0, 10.1
2021-06-23 21:24:29,005 Current rsum is 487.28
2021-06-23 21:24:31,076 runs/f30k_butd_region_bert/log
2021-06-23 21:24:31,076 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-23 21:24:31,078 image encoder trainable parameters: 20490144
2021-06-23 21:24:31,083 txt encoder trainable parameters: 137319072
2021-06-23 21:25:29,593 Epoch: [24][31/1133]	Eit 27200  lr 5e-05  Le 9.7834 (9.7935)	Time 1.863 (0.000)	Data 0.002 (0.000)	
2021-06-23 21:31:13,845 Epoch: [24][231/1133]	Eit 27400  lr 5e-05  Le 9.7924 (9.7953)	Time 1.804 (0.000)	Data 0.004 (0.000)	
2021-06-23 21:37:05,994 Epoch: [24][431/1133]	Eit 27600  lr 5e-05  Le 9.7805 (9.7960)	Time 1.317 (0.000)	Data 0.004 (0.000)	
2021-06-23 21:42:51,738 Epoch: [24][631/1133]	Eit 27800  lr 5e-05  Le 9.8073 (9.7963)	Time 1.772 (0.000)	Data 0.003 (0.000)	
2021-06-23 21:48:35,450 Epoch: [24][831/1133]	Eit 28000  lr 5e-05  Le 9.7908 (9.7964)	Time 1.300 (0.000)	Data 0.002 (0.000)	
2021-06-23 21:54:21,008 Epoch: [24][1031/1133]	Eit 28200  lr 5e-05  Le 9.8178 (9.7969)	Time 1.766 (0.000)	Data 0.002 (0.000)	
2021-06-23 21:57:24,206 Test: [0/40]	Le 10.1947 (10.1946)	Time 4.020 (0.000)	
2021-06-23 21:57:48,921 calculate similarity time: 0.07719588279724121
2021-06-23 21:57:49,409 Image to text: 72.8, 92.2, 96.5, 1.0, 2.7
2021-06-23 21:57:49,776 Text to image: 56.5, 82.0, 88.0, 1.0, 9.9
2021-06-23 21:57:49,776 Current rsum is 488.02
You have new mail in /var/spool/mail/root
[root@gpu1 vse_infty-master-my-graph-gru-unified-meanpooling]# CUDA_VISIBLE_DEVICES=0,1 python3 eval.py --dataset f30k --data_path ../data/f30k
INFO:root:Evaluating runs/f30k_butd_region_bert...
INFO:lib.evaluation:Namespace(backbone_lr_factor=0.01, backbone_path='', backbone_source='detector', backbone_warmup_epochs=5, batch_size=128, data_name='f30k', data_path='../data/f30k', embed_size=1024, embedding_warmup_epochs=2, grad_clip=2.0, img_dim=2048, input_scale_factor=1, learning_rate=0.0005, log_step=200, logger_name='runs/f30k_butd_region_bert/log', lr_update=15, margin=0.2, max_violation=True, model_name='runs/f30k_butd_region_bert', no_imgnorm=False, no_txtnorm=False, num_epochs=25, optim='adam', precomp_enc_type='basic', reset_start_epoch=False, resume='', val_step=500, vocab_path='./vocab/', vocab_size=30522, vse_mean_warmup_epochs=1, word_dim=300, workers=5)
INFO:transformers.tokenization_utils:loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-unified-meanpooling/lib/encoders.py:48: UserWarning: nn.init.constant is now deprecated in favor of nn.init.constant_.
  nn.init.constant(self.W[1].weight, 0)
/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-unified-meanpooling/lib/encoders.py:49: UserWarning: nn.init.constant is now deprecated in favor of nn.init.constant_.
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
INFO:lib.evaluation:Test: [0/40]	Le 10.2073 (10.2073)	Time 5.695 (0.000)	
INFO:lib.evaluation:Test: [10/40]	Le 10.2175 (10.2094)	Time 0.623 (0.000)	
INFO:lib.evaluation:Test: [20/40]	Le 10.1964 (10.2083)	Time 0.606 (0.000)	
INFO:lib.evaluation:Test: [30/40]	Le 10.2155 (10.2114)	Time 0.881 (0.000)	
INFO:lib.evaluation:Images: 1000, Captions: 5000
INFO:lib.evaluation:calculate similarity time: 0.0849609375
INFO:lib.evaluation:rsum: 489.1
INFO:lib.evaluation:Average i2t Recall: 87.1
INFO:lib.evaluation:Image to text: 74.0 91.2 96.1 1.0 3.1
INFO:lib.evaluation:Average t2i Recall: 75.9
INFO:lib.evaluation:Text to image: 55.7 82.7 89.4 1.0 8.3
INFO:root:Evaluating runs/release_weights/f30k_butd_grid_bert...
Traceback (most recent call last):
  File "eval.py", line 58, in <module>
    main()
  File "eval.py", line 54, in main
    evaluation.evalrank(model_path, data_path=opt.data_path, split='test', fold5=False, save_path=save_path)
  File "/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-unified-meanpooling/lib/evaluation.py", line 196, in evalrank
    checkpoint = torch.load(model_path)
  File "/root/anaconda3/lib/python3.6/site-packages/torch/serialization.py", line 571, in load
    with _open_file_like(f, 'rb') as opened_file:
  File "/root/anaconda3/lib/python3.6/site-packages/torch/serialization.py", line 229, in _open_file_like
    return _open_file(name_or_buffer, mode)
  File "/root/anaconda3/lib/python3.6/site-packages/torch/serialization.py", line 210, in __init__
    super(_open_file, self).__init__(open(name, mode))
FileNotFoundError: [Errno 2] No such file or directory: 'runs/release_weights/f30k_butd_grid_bert/model_best.pth'

