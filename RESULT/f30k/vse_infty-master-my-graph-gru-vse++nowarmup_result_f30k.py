[root@gpu1 vse_infty-master-my-graph-gru-vse++nowarmup]# sh train_region.sh 
2021-06-24 09:07:06,355 Namespace(backbone_lr_factor=0.01, backbone_path='', backbone_source='detector', backbone_warmup_epochs=5, batch_size=128, data_name='f30k', data_path='../data/f30k', embed_size=1024, embedding_warmup_epochs=2, grad_clip=2.0, img_dim=2048, input_scale_factor=1, learning_rate=0.0005, log_step=200, logger_name='runs/f30k_butd_region_bert/log', lr_update=15, margin=0.2, max_violation=False, model_name='runs/f30k_butd_region_bert', no_imgnorm=False, no_txtnorm=False, num_epochs=25, optim='adam', precomp_enc_type='basic', reset_start_epoch=False, resume='', val_step=500, vocab_path='./vocab/', vse_mean_warmup_epochs=1, word_dim=300, workers=10)
2021-06-24 09:07:06,356 Model name '/public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased' not found in model shortcut name list (bert-base-uncased, bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, bert-base-multilingual-cased, bert-base-chinese, bert-base-german-cased, bert-large-uncased-whole-word-masking, bert-large-cased-whole-word-masking, bert-large-uncased-whole-word-masking-finetuned-squad, bert-large-cased-whole-word-masking-finetuned-squad, bert-base-cased-finetuned-mrpc). Assuming '/public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased' is a path or url to a directory containing tokenizer files.
2021-06-24 09:07:06,356 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/added_tokens.json. We won't load it.
2021-06-24 09:07:06,356 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/special_tokens_map.json. We won't load it.
2021-06-24 09:07:06,356 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/tokenizer_config.json. We won't load it.
2021-06-24 09:07:06,356 loading file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/vocab.txt
2021-06-24 09:07:06,356 loading file None
2021-06-24 09:07:06,356 loading file None
2021-06-24 09:07:06,356 loading file None
/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-vse++nowarmup/lib/encoders.py:48: UserWarning: nn.init.constant is now deprecated in favor of nn.init.constant_.
  nn.init.constant(self.W[1].weight, 0)
/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-vse++nowarmup/lib/encoders.py:49: UserWarning: nn.init.constant is now deprecated in favor of nn.init.constant_.
  nn.init.constant(self.W[1].bias, 0)
2021-06-24 09:07:14,741 loading configuration file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/config.json
2021-06-24 09:07:14,742 Model config {
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

2021-06-24 09:07:14,742 loading weights file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/pytorch_model.bin
2021-06-24 09:07:20,954 Use adam as the optimizer, with init lr 0.0005
2021-06-24 09:07:20,955 Image encoder is data paralleled now.
2021-06-24 09:07:20,955 runs/f30k_butd_region_bert/log
2021-06-24 09:07:20,955 runs/f30k_butd_region_bert
2021-06-24 09:07:20,956 image encoder trainable parameters: 20490144
2021-06-24 09:07:20,961 txt encoder trainable parameters: 137319072
/root/anaconda3/lib/python3.6/site-packages/torch/nn/modules/rnn.py:735: UserWarning: RNN module weights are not part of single contiguous chunk of memory. This means they need to be compacted at every call, possibly greatly increasing memory usage. To compact weights again call flatten_parameters(). (Triggered internally at  /pytorch/aten/src/ATen/native/cudnn/RNN.cpp:1264.)
  self.dropout, self.training, self.bidirectional, self.batch_first)
2021-06-24 09:09:56,982 Epoch: [0][199/1133]	Eit 200  lr 0.0005  Le 51.4751 (53.3722)	Time 0.785 (0.000)	Data 0.002 (0.000)	
2021-06-24 09:13:14,914 Epoch: [0][399/1133]	Eit 400  lr 0.0005  Le 51.2482 (52.3429)	Time 0.924 (0.000)	Data 0.002 (0.000)	
2021-06-24 09:16:54,170 Epoch: [0][599/1133]	Eit 600  lr 0.0005  Le 51.2232 (51.9722)	Time 0.883 (0.000)	Data 0.002 (0.000)	
2021-06-24 09:20:35,911 Epoch: [0][799/1133]	Eit 800  lr 0.0005  Le 51.2063 (51.7829)	Time 1.105 (0.000)	Data 0.002 (0.000)	
2021-06-24 09:24:19,226 Epoch: [0][999/1133]	Eit 1000  lr 0.0005  Le 51.2010 (51.6679)	Time 1.194 (0.000)	Data 0.003 (0.000)	
2021-06-24 09:26:50,423 Test: [0/40]	Le 51.2269 (51.2269)	Time 4.134 (0.000)	
2021-06-24 09:27:07,640 calculate similarity time: 0.06697559356689453
2021-06-24 09:27:08,125 Image to text: 29.0, 52.6, 66.1, 5.0, 28.2
2021-06-24 09:27:08,563 Text to image: 19.7, 47.3, 60.6, 6.0, 31.5
2021-06-24 09:27:08,563 Current rsum is 275.28
2021-06-24 09:27:11,712 runs/f30k_butd_region_bert/log
2021-06-24 09:27:11,712 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 09:27:11,714 image encoder trainable parameters: 20490144
2021-06-24 09:27:11,720 txt encoder trainable parameters: 137319072
2021-06-24 09:28:30,174 Epoch: [1][67/1133]	Eit 1200  lr 0.0005  Le 51.2022 (51.1997)	Time 0.995 (0.000)	Data 0.002 (0.000)	
2021-06-24 09:32:15,386 Epoch: [1][267/1133]	Eit 1400  lr 0.0005  Le 51.1908 (51.1963)	Time 1.163 (0.000)	Data 0.002 (0.000)	
2021-06-24 09:35:56,136 Epoch: [1][467/1133]	Eit 1600  lr 0.0005  Le 51.1985 (51.1906)	Time 1.045 (0.000)	Data 0.002 (0.000)	
2021-06-24 09:39:38,200 Epoch: [1][667/1133]	Eit 1800  lr 0.0005  Le 50.0902 (51.1240)	Time 1.147 (0.000)	Data 0.002 (0.000)	
2021-06-24 09:43:21,212 Epoch: [1][867/1133]	Eit 2000  lr 0.0005  Le 44.4737 (50.3012)	Time 1.329 (0.000)	Data 0.002 (0.000)	
2021-06-24 09:47:07,688 Epoch: [1][1067/1133]	Eit 2200  lr 0.0005  Le 45.1829 (49.1854)	Time 0.907 (0.000)	Data 0.002 (0.000)	
2021-06-24 09:48:23,472 Test: [0/40]	Le 58.6352 (58.6351)	Time 3.684 (0.000)	
2021-06-24 09:48:40,644 calculate similarity time: 0.06524538993835449
2021-06-24 09:48:41,179 Image to text: 53.0, 78.6, 87.2, 1.0, 6.8
2021-06-24 09:48:41,502 Text to image: 39.1, 69.1, 78.8, 2.0, 12.7
2021-06-24 09:48:41,502 Current rsum is 405.82
2021-06-24 09:48:45,046 runs/f30k_butd_region_bert/log
2021-06-24 09:48:45,046 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 09:48:45,049 image encoder trainable parameters: 20490144
2021-06-24 09:48:45,061 txt encoder trainable parameters: 137319072
2021-06-24 09:51:22,537 Epoch: [2][135/1133]	Eit 2400  lr 0.0005  Le 41.5012 (40.4115)	Time 0.969 (0.000)	Data 0.002 (0.000)	
2021-06-24 09:55:11,082 Epoch: [2][335/1133]	Eit 2600  lr 0.0005  Le 39.3419 (39.7078)	Time 1.200 (0.000)	Data 0.002 (0.000)	
2021-06-24 09:58:59,299 Epoch: [2][535/1133]	Eit 2800  lr 0.0005  Le 39.0596 (39.1807)	Time 1.049 (0.000)	Data 0.002 (0.000)	
2021-06-24 10:02:45,447 Epoch: [2][735/1133]	Eit 3000  lr 0.0005  Le 32.1449 (38.6679)	Time 1.263 (0.000)	Data 0.002 (0.000)	
2021-06-24 10:06:30,365 Epoch: [2][935/1133]	Eit 3200  lr 0.0005  Le 31.4111 (38.1834)	Time 0.972 (0.000)	Data 0.002 (0.000)	
2021-06-24 10:10:12,899 Test: [0/40]	Le 58.6551 (58.6551)	Time 3.410 (0.000)	
2021-06-24 10:10:29,193 calculate similarity time: 0.059152841567993164
2021-06-24 10:10:29,688 Image to text: 64.8, 87.8, 93.5, 1.0, 3.6
2021-06-24 10:10:30,140 Text to image: 47.2, 76.8, 85.1, 2.0, 9.6
2021-06-24 10:10:30,140 Current rsum is 455.14
2021-06-24 10:10:33,769 runs/f30k_butd_region_bert/log
2021-06-24 10:10:33,769 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 10:10:33,772 image encoder trainable parameters: 20490144
2021-06-24 10:10:33,781 txt encoder trainable parameters: 137319072
2021-06-24 10:10:41,218 Epoch: [3][3/1133]	Eit 3400  lr 0.0005  Le 36.0010 (34.3422)	Time 1.028 (0.000)	Data 0.002 (0.000)	
2021-06-24 10:14:26,097 Epoch: [3][203/1133]	Eit 3600  lr 0.0005  Le 36.1095 (32.8615)	Time 0.945 (0.000)	Data 0.002 (0.000)	
2021-06-24 10:18:10,739 Epoch: [3][403/1133]	Eit 3800  lr 0.0005  Le 27.5800 (32.6197)	Time 0.946 (0.000)	Data 0.002 (0.000)	
2021-06-24 10:21:55,035 Epoch: [3][603/1133]	Eit 4000  lr 0.0005  Le 35.6481 (32.5405)	Time 0.988 (0.000)	Data 0.003 (0.000)	
2021-06-24 10:25:40,812 Epoch: [3][803/1133]	Eit 4200  lr 0.0005  Le 34.0348 (32.3761)	Time 1.318 (0.000)	Data 0.004 (0.000)	
2021-06-24 10:29:24,902 Epoch: [3][1003/1133]	Eit 4400  lr 0.0005  Le 29.1027 (32.2058)	Time 1.248 (0.000)	Data 0.002 (0.000)	
2021-06-24 10:31:52,717 Test: [0/40]	Le 59.4214 (59.4213)	Time 3.587 (0.000)	
2021-06-24 10:32:09,533 calculate similarity time: 0.07029366493225098
2021-06-24 10:32:09,978 Image to text: 68.5, 91.1, 95.6, 1.0, 3.3
2021-06-24 10:32:10,298 Text to image: 52.0, 79.8, 87.4, 1.0, 7.9
2021-06-24 10:32:10,298 Current rsum is 474.49999999999994
2021-06-24 10:32:13,837 runs/f30k_butd_region_bert/log
2021-06-24 10:32:13,837 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 10:32:13,841 image encoder trainable parameters: 20490144
2021-06-24 10:32:13,851 txt encoder trainable parameters: 137319072
2021-06-24 10:33:36,780 Epoch: [4][71/1133]	Eit 4600  lr 0.0005  Le 26.9475 (28.7764)	Time 0.898 (0.000)	Data 0.002 (0.000)	
2021-06-24 10:37:14,776 Epoch: [4][271/1133]	Eit 4800  lr 0.0005  Le 31.4216 (28.9654)	Time 1.082 (0.000)	Data 0.002 (0.000)	
2021-06-24 10:41:00,047 Epoch: [4][471/1133]	Eit 5000  lr 0.0005  Le 26.0985 (28.8137)	Time 1.203 (0.000)	Data 0.002 (0.000)	
2021-06-24 10:44:44,731 Epoch: [4][671/1133]	Eit 5200  lr 0.0005  Le 25.1795 (28.7998)	Time 1.141 (0.000)	Data 0.002 (0.000)	
2021-06-24 10:48:33,138 Epoch: [4][871/1133]	Eit 5400  lr 0.0005  Le 29.8740 (28.7557)	Time 1.044 (0.000)	Data 0.002 (0.000)	
2021-06-24 10:52:16,394 Epoch: [4][1071/1133]	Eit 5600  lr 0.0005  Le 27.8403 (28.6447)	Time 0.969 (0.000)	Data 0.002 (0.000)	
2021-06-24 10:53:26,804 Test: [0/40]	Le 59.2504 (59.2504)	Time 3.425 (0.000)	
2021-06-24 10:53:43,569 calculate similarity time: 0.060784339904785156
2021-06-24 10:53:44,060 Image to text: 73.9, 92.0, 95.7, 1.0, 2.7
2021-06-24 10:53:44,382 Text to image: 53.1, 81.4, 88.1, 1.0, 7.4
2021-06-24 10:53:44,382 Current rsum is 484.3
2021-06-24 10:53:47,981 runs/f30k_butd_region_bert/log
2021-06-24 10:53:47,982 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 10:53:47,985 image encoder trainable parameters: 20490144
2021-06-24 10:53:47,997 txt encoder trainable parameters: 137319072
2021-06-24 10:56:27,519 Epoch: [5][139/1133]	Eit 5800  lr 0.0005  Le 25.0502 (25.9052)	Time 0.956 (0.000)	Data 0.002 (0.000)	
2021-06-24 11:00:15,358 Epoch: [5][339/1133]	Eit 6000  lr 0.0005  Le 24.6494 (26.1100)	Time 1.154 (0.000)	Data 0.002 (0.000)	
2021-06-24 11:03:58,754 Epoch: [5][539/1133]	Eit 6200  lr 0.0005  Le 25.2557 (26.2693)	Time 0.900 (0.000)	Data 0.002 (0.000)	
2021-06-24 11:07:39,021 Epoch: [5][739/1133]	Eit 6400  lr 0.0005  Le 24.3406 (26.1561)	Time 0.987 (0.000)	Data 0.002 (0.000)	
2021-06-24 11:11:25,012 Epoch: [5][939/1133]	Eit 6600  lr 0.0005  Le 25.5549 (26.0838)	Time 0.906 (0.000)	Data 0.002 (0.000)	
2021-06-24 11:15:05,586 Test: [0/40]	Le 59.3358 (59.3358)	Time 3.652 (0.000)	
2021-06-24 11:15:22,272 calculate similarity time: 0.06992959976196289
2021-06-24 11:15:22,852 Image to text: 73.0, 91.8, 95.5, 1.0, 3.2
2021-06-24 11:15:23,307 Text to image: 55.0, 82.1, 88.7, 1.0, 7.9
2021-06-24 11:15:23,307 Current rsum is 486.12
2021-06-24 11:15:26,930 runs/f30k_butd_region_bert/log
2021-06-24 11:15:26,931 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 11:15:26,934 image encoder trainable parameters: 20490144
2021-06-24 11:15:26,946 txt encoder trainable parameters: 137319072
2021-06-24 11:15:38,897 Epoch: [6][7/1133]	Eit 6800  lr 0.0005  Le 20.6061 (23.0342)	Time 1.016 (0.000)	Data 0.002 (0.000)	
2021-06-24 11:19:20,259 Epoch: [6][207/1133]	Eit 7000  lr 0.0005  Le 28.0614 (23.9690)	Time 1.032 (0.000)	Data 0.002 (0.000)	
2021-06-24 11:23:01,428 Epoch: [6][407/1133]	Eit 7200  lr 0.0005  Le 23.9288 (24.0526)	Time 0.989 (0.000)	Data 0.003 (0.000)	
2021-06-24 11:26:45,452 Epoch: [6][607/1133]	Eit 7400  lr 0.0005  Le 21.0971 (24.0235)	Time 1.248 (0.000)	Data 0.002 (0.000)	
2021-06-24 11:30:35,677 Epoch: [6][807/1133]	Eit 7600  lr 0.0005  Le 23.0554 (23.9216)	Time 0.970 (0.000)	Data 0.002 (0.000)	
2021-06-24 11:34:23,258 Epoch: [6][1007/1133]	Eit 7800  lr 0.0005  Le 21.0905 (23.9222)	Time 0.638 (0.000)	Data 0.002 (0.000)	
2021-06-24 11:36:43,388 Test: [0/40]	Le 59.4704 (59.4704)	Time 3.805 (0.000)	
2021-06-24 11:36:59,889 calculate similarity time: 0.06654119491577148
2021-06-24 11:37:00,400 Image to text: 74.2, 92.4, 96.5, 1.0, 2.5
2021-06-24 11:37:00,729 Text to image: 56.0, 82.2, 88.9, 1.0, 7.4
2021-06-24 11:37:00,730 Current rsum is 490.24
2021-06-24 11:37:04,257 runs/f30k_butd_region_bert/log
2021-06-24 11:37:04,258 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 11:37:04,263 image encoder trainable parameters: 20490144
2021-06-24 11:37:04,274 txt encoder trainable parameters: 137319072
2021-06-24 11:38:34,765 Epoch: [7][75/1133]	Eit 8000  lr 0.0005  Le 21.7904 (21.8593)	Time 1.385 (0.000)	Data 0.002 (0.000)	
2021-06-24 11:42:22,458 Epoch: [7][275/1133]	Eit 8200  lr 0.0005  Le 23.7865 (21.9985)	Time 1.048 (0.000)	Data 0.003 (0.000)	
2021-06-24 11:46:08,518 Epoch: [7][475/1133]	Eit 8400  lr 0.0005  Le 20.3650 (21.9471)	Time 0.948 (0.000)	Data 0.002 (0.000)	
2021-06-24 11:49:54,323 Epoch: [7][675/1133]	Eit 8600  lr 0.0005  Le 23.7991 (22.1055)	Time 0.975 (0.000)	Data 0.002 (0.000)	
2021-06-24 11:53:36,152 Epoch: [7][875/1133]	Eit 8800  lr 0.0005  Le 17.7819 (22.1538)	Time 1.222 (0.000)	Data 0.002 (0.000)	
2021-06-24 11:57:21,383 Epoch: [7][1075/1133]	Eit 9000  lr 0.0005  Le 21.7239 (22.0859)	Time 0.963 (0.000)	Data 0.002 (0.000)	
2021-06-24 11:58:27,139 Test: [0/40]	Le 58.6753 (58.6752)	Time 3.339 (0.000)	
2021-06-24 11:58:43,732 calculate similarity time: 0.07053923606872559
2021-06-24 11:58:44,225 Image to text: 72.5, 91.7, 95.8, 1.0, 2.9
2021-06-24 11:58:44,677 Text to image: 55.1, 82.4, 88.7, 1.0, 7.4
2021-06-24 11:58:44,677 Current rsum is 486.26
2021-06-24 11:58:46,142 runs/f30k_butd_region_bert/log
2021-06-24 11:58:46,142 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 11:58:46,144 image encoder trainable parameters: 20490144
2021-06-24 11:58:46,150 txt encoder trainable parameters: 137319072
2021-06-24 12:01:29,654 Epoch: [8][143/1133]	Eit 9200  lr 0.0005  Le 19.8540 (20.5464)	Time 1.321 (0.000)	Data 0.002 (0.000)	
2021-06-24 12:05:09,851 Epoch: [8][343/1133]	Eit 9400  lr 0.0005  Le 21.6384 (20.7452)	Time 1.414 (0.000)	Data 0.002 (0.000)	
2021-06-24 12:08:57,344 Epoch: [8][543/1133]	Eit 9600  lr 0.0005  Le 19.8861 (20.6219)	Time 1.100 (0.000)	Data 0.002 (0.000)	
2021-06-24 12:12:42,103 Epoch: [8][743/1133]	Eit 9800  lr 0.0005  Le 19.5563 (20.6196)	Time 1.193 (0.000)	Data 0.002 (0.000)	
2021-06-24 12:16:27,391 Epoch: [8][943/1133]	Eit 10000  lr 0.0005  Le 19.8304 (20.6948)	Time 1.113 (0.000)	Data 0.002 (0.000)	
2021-06-24 12:20:03,497 Test: [0/40]	Le 59.1709 (59.1708)	Time 3.355 (0.000)	
2021-06-24 12:20:20,334 calculate similarity time: 0.06210589408874512
2021-06-24 12:20:20,880 Image to text: 74.4, 92.3, 96.9, 1.0, 2.7
2021-06-24 12:20:21,194 Text to image: 56.2, 83.0, 89.4, 1.0, 7.4
2021-06-24 12:20:21,194 Current rsum is 492.28000000000003
2021-06-24 12:20:24,839 runs/f30k_butd_region_bert/log
2021-06-24 12:20:24,839 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 12:20:24,842 image encoder trainable parameters: 20490144
2021-06-24 12:20:24,850 txt encoder trainable parameters: 137319072
2021-06-24 12:20:40,858 Epoch: [9][11/1133]	Eit 10200  lr 0.0005  Le 19.3765 (19.5712)	Time 0.994 (0.000)	Data 0.002 (0.000)	
2021-06-24 12:24:22,888 Epoch: [9][211/1133]	Eit 10400  lr 0.0005  Le 19.3760 (19.0466)	Time 0.965 (0.000)	Data 0.002 (0.000)	
2021-06-24 12:28:10,982 Epoch: [9][411/1133]	Eit 10600  lr 0.0005  Le 20.6426 (19.2524)	Time 1.170 (0.000)	Data 0.008 (0.000)	
2021-06-24 12:31:54,539 Epoch: [9][611/1133]	Eit 10800  lr 0.0005  Le 19.2357 (19.2626)	Time 1.327 (0.000)	Data 0.002 (0.000)	
2021-06-24 12:35:37,711 Epoch: [9][811/1133]	Eit 11000  lr 0.0005  Le 18.7154 (19.3091)	Time 0.935 (0.000)	Data 0.003 (0.000)	
2021-06-24 12:39:23,196 Epoch: [9][1011/1133]	Eit 11200  lr 0.0005  Le 21.4739 (19.3438)	Time 1.192 (0.000)	Data 0.008 (0.000)	
2021-06-24 12:41:41,174 Test: [0/40]	Le 59.5135 (59.5135)	Time 3.434 (0.000)	
2021-06-24 12:41:58,242 calculate similarity time: 0.08176231384277344
2021-06-24 12:41:58,713 Image to text: 75.6, 93.6, 96.8, 1.0, 2.4
2021-06-24 12:41:59,031 Text to image: 56.5, 82.3, 89.0, 1.0, 7.4
2021-06-24 12:41:59,031 Current rsum is 493.82
2021-06-24 12:42:02,675 runs/f30k_butd_region_bert/log
2021-06-24 12:42:02,675 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 12:42:02,677 image encoder trainable parameters: 20490144
2021-06-24 12:42:02,686 txt encoder trainable parameters: 137319072
2021-06-24 12:43:34,092 Epoch: [10][79/1133]	Eit 11400  lr 0.0005  Le 19.3605 (17.5104)	Time 1.250 (0.000)	Data 0.002 (0.000)	
2021-06-24 12:47:20,339 Epoch: [10][279/1133]	Eit 11600  lr 0.0005  Le 16.0390 (17.7085)	Time 1.027 (0.000)	Data 0.003 (0.000)	
2021-06-24 12:51:07,437 Epoch: [10][479/1133]	Eit 11800  lr 0.0005  Le 22.0444 (17.9521)	Time 0.934 (0.000)	Data 0.002 (0.000)	
2021-06-24 12:54:55,511 Epoch: [10][679/1133]	Eit 12000  lr 0.0005  Le 18.9667 (18.1410)	Time 1.257 (0.000)	Data 0.002 (0.000)	
2021-06-24 12:58:39,855 Epoch: [10][879/1133]	Eit 12200  lr 0.0005  Le 17.2601 (18.1604)	Time 0.937 (0.000)	Data 0.002 (0.000)	
2021-06-24 13:02:18,995 Epoch: [10][1079/1133]	Eit 12400  lr 0.0005  Le 18.6177 (18.2567)	Time 0.981 (0.000)	Data 0.003 (0.000)	
2021-06-24 13:03:20,083 Test: [0/40]	Le 59.3328 (59.3327)	Time 3.362 (0.000)	
2021-06-24 13:03:36,933 calculate similarity time: 0.058940887451171875
2021-06-24 13:03:37,421 Image to text: 73.1, 92.2, 96.8, 1.0, 2.4
2021-06-24 13:03:37,780 Text to image: 56.7, 83.2, 89.5, 1.0, 7.1
2021-06-24 13:03:37,780 Current rsum is 491.5
2021-06-24 13:03:39,244 runs/f30k_butd_region_bert/log
2021-06-24 13:03:39,244 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 13:03:39,246 image encoder trainable parameters: 20490144
2021-06-24 13:03:39,251 txt encoder trainable parameters: 137319072
2021-06-24 13:06:27,803 Epoch: [11][147/1133]	Eit 12600  lr 0.0005  Le 15.9103 (16.8726)	Time 0.856 (0.000)	Data 0.002 (0.000)	
2021-06-24 13:10:13,841 Epoch: [11][347/1133]	Eit 12800  lr 0.0005  Le 18.4416 (17.0720)	Time 0.992 (0.000)	Data 0.002 (0.000)	
2021-06-24 13:13:58,605 Epoch: [11][547/1133]	Eit 13000  lr 0.0005  Le 16.4472 (17.3142)	Time 1.222 (0.000)	Data 0.002 (0.000)	
2021-06-24 13:17:43,277 Epoch: [11][747/1133]	Eit 13200  lr 0.0005  Le 15.0527 (17.2680)	Time 1.038 (0.000)	Data 0.002 (0.000)	
2021-06-24 13:21:29,361 Epoch: [11][947/1133]	Eit 13400  lr 0.0005  Le 17.2515 (17.3409)	Time 1.205 (0.000)	Data 0.002 (0.000)	
2021-06-24 13:24:59,860 Test: [0/40]	Le 59.3768 (59.3768)	Time 3.251 (0.000)	
2021-06-24 13:25:16,753 calculate similarity time: 0.06144118309020996
2021-06-24 13:25:17,272 Image to text: 74.0, 92.6, 96.1, 1.0, 2.4
2021-06-24 13:25:17,611 Text to image: 56.5, 83.4, 89.5, 1.0, 7.3
2021-06-24 13:25:17,611 Current rsum is 492.2
2021-06-24 13:25:19,074 runs/f30k_butd_region_bert/log
2021-06-24 13:25:19,074 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 13:25:19,076 image encoder trainable parameters: 20490144
2021-06-24 13:25:19,081 txt encoder trainable parameters: 137319072
2021-06-24 13:25:39,814 Epoch: [12][15/1133]	Eit 13600  lr 0.0005  Le 16.6946 (16.3244)	Time 0.974 (0.000)	Data 0.002 (0.000)	
2021-06-24 13:29:25,545 Epoch: [12][215/1133]	Eit 13800  lr 0.0005  Le 16.2634 (16.1545)	Time 0.708 (0.000)	Data 0.002 (0.000)	
2021-06-24 13:33:09,654 Epoch: [12][415/1133]	Eit 14000  lr 0.0005  Le 18.7714 (16.2786)	Time 0.924 (0.000)	Data 0.002 (0.000)	
2021-06-24 13:36:52,604 Epoch: [12][615/1133]	Eit 14200  lr 0.0005  Le 14.7740 (16.3873)	Time 1.012 (0.000)	Data 0.002 (0.000)	
2021-06-24 13:40:38,828 Epoch: [12][815/1133]	Eit 14400  lr 0.0005  Le 17.6686 (16.3814)	Time 1.284 (0.000)	Data 0.011 (0.000)	
2021-06-24 13:44:21,208 Epoch: [12][1015/1133]	Eit 14600  lr 0.0005  Le 16.1926 (16.4104)	Time 0.839 (0.000)	Data 0.002 (0.000)	
2021-06-24 13:46:34,747 Test: [0/40]	Le 60.1136 (60.1135)	Time 3.435 (0.000)	
2021-06-24 13:46:51,023 calculate similarity time: 0.06881904602050781
2021-06-24 13:46:51,583 Image to text: 74.6, 93.3, 96.8, 1.0, 2.6
2021-06-24 13:46:51,929 Text to image: 58.1, 83.1, 89.6, 1.0, 7.4
2021-06-24 13:46:51,930 Current rsum is 495.49999999999994
2021-06-24 13:46:55,569 runs/f30k_butd_region_bert/log
2021-06-24 13:46:55,570 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 13:46:55,573 image encoder trainable parameters: 20490144
2021-06-24 13:46:55,584 txt encoder trainable parameters: 137319072
2021-06-24 13:48:33,468 Epoch: [13][83/1133]	Eit 14800  lr 0.0005  Le 14.5195 (14.9224)	Time 1.078 (0.000)	Data 0.002 (0.000)	
2021-06-24 13:52:18,683 Epoch: [13][283/1133]	Eit 15000  lr 0.0005  Le 19.7150 (15.4699)	Time 0.945 (0.000)	Data 0.002 (0.000)	
2021-06-24 13:56:02,283 Epoch: [13][483/1133]	Eit 15200  lr 0.0005  Le 14.5460 (15.5373)	Time 0.955 (0.000)	Data 0.002 (0.000)	
2021-06-24 13:59:45,584 Epoch: [13][683/1133]	Eit 15400  lr 0.0005  Le 14.1337 (15.5867)	Time 1.366 (0.000)	Data 0.003 (0.000)	
2021-06-24 14:03:30,746 Epoch: [13][883/1133]	Eit 15600  lr 0.0005  Le 16.3868 (15.6194)	Time 1.254 (0.000)	Data 0.002 (0.000)	
2021-06-24 14:07:17,014 Epoch: [13][1083/1133]	Eit 15800  lr 0.0005  Le 16.1415 (15.6875)	Time 1.009 (0.000)	Data 0.003 (0.000)	
2021-06-24 14:08:14,098 Test: [0/40]	Le 58.9946 (58.9946)	Time 3.615 (0.000)	
2021-06-24 14:08:30,394 calculate similarity time: 0.06804823875427246
2021-06-24 14:08:30,846 Image to text: 76.5, 92.9, 96.7, 1.0, 2.3
2021-06-24 14:08:31,165 Text to image: 57.3, 83.2, 89.2, 1.0, 7.5
2021-06-24 14:08:31,166 Current rsum is 495.74
2021-06-24 14:08:34,789 runs/f30k_butd_region_bert/log
2021-06-24 14:08:34,789 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 14:08:34,793 image encoder trainable parameters: 20490144
2021-06-24 14:08:34,806 txt encoder trainable parameters: 137319072
2021-06-24 14:11:28,208 Epoch: [14][151/1133]	Eit 16000  lr 0.0005  Le 12.5510 (14.6075)	Time 1.341 (0.000)	Data 0.002 (0.000)	
2021-06-24 14:15:15,431 Epoch: [14][351/1133]	Eit 16200  lr 0.0005  Le 14.4321 (14.7502)	Time 1.361 (0.000)	Data 0.002 (0.000)	
2021-06-24 14:19:02,265 Epoch: [14][551/1133]	Eit 16400  lr 0.0005  Le 13.2336 (14.8238)	Time 1.271 (0.000)	Data 0.002 (0.000)	
2021-06-24 14:22:45,692 Epoch: [14][751/1133]	Eit 16600  lr 0.0005  Le 15.1391 (14.8976)	Time 0.982 (0.000)	Data 0.003 (0.000)	
2021-06-24 14:26:30,190 Epoch: [14][951/1133]	Eit 16800  lr 0.0005  Le 14.9001 (14.9840)	Time 1.051 (0.000)	Data 0.002 (0.000)	
2021-06-24 14:29:53,206 Test: [0/40]	Le 59.7750 (59.7750)	Time 3.698 (0.000)	
2021-06-24 14:30:09,805 calculate similarity time: 0.06038236618041992
2021-06-24 14:30:10,391 Image to text: 75.4, 93.5, 97.1, 1.0, 2.5
2021-06-24 14:30:10,832 Text to image: 58.2, 83.1, 89.7, 1.0, 7.3
2021-06-24 14:30:10,832 Current rsum is 497.08000000000004
2021-06-24 14:30:14,486 runs/f30k_butd_region_bert/log
2021-06-24 14:30:14,486 runs/f30k_butd_region_bert
2021-06-24 14:30:14,487 Current epoch num is 15, decrease all lr by 10
2021-06-24 14:30:14,487 new lr 5e-05
2021-06-24 14:30:14,487 new lr 5e-06
2021-06-24 14:30:14,487 new lr 5e-05
Use VSE++ objective.
2021-06-24 14:30:14,489 image encoder trainable parameters: 20490144
2021-06-24 14:30:14,499 txt encoder trainable parameters: 137319072
2021-06-24 14:30:41,141 Epoch: [15][19/1133]	Eit 17000  lr 5e-05  Le 11.9358 (13.1148)	Time 0.977 (0.000)	Data 0.002 (0.000)	
2021-06-24 14:34:27,858 Epoch: [15][219/1133]	Eit 17200  lr 5e-05  Le 10.6488 (13.2282)	Time 1.174 (0.000)	Data 0.002 (0.000)	
2021-06-24 14:38:15,750 Epoch: [15][419/1133]	Eit 17400  lr 5e-05  Le 11.7931 (12.9897)	Time 1.421 (0.000)	Data 0.002 (0.000)	
2021-06-24 14:41:59,475 Epoch: [15][619/1133]	Eit 17600  lr 5e-05  Le 8.8043 (12.8215)	Time 1.039 (0.000)	Data 0.002 (0.000)	
2021-06-24 14:45:43,801 Epoch: [15][819/1133]	Eit 17800  lr 5e-05  Le 15.4452 (12.7114)	Time 0.903 (0.000)	Data 0.002 (0.000)	
2021-06-24 14:49:31,001 Epoch: [15][1019/1133]	Eit 18000  lr 5e-05  Le 9.5429 (12.5753)	Time 1.296 (0.000)	Data 0.002 (0.000)	
2021-06-24 14:51:39,161 Test: [0/40]	Le 59.6914 (59.6913)	Time 3.283 (0.000)	
2021-06-24 14:51:55,551 calculate similarity time: 0.06841826438903809
2021-06-24 14:51:56,102 Image to text: 78.0, 94.1, 97.1, 1.0, 2.3
2021-06-24 14:51:56,430 Text to image: 59.2, 83.9, 89.9, 1.0, 7.2
2021-06-24 14:51:56,430 Current rsum is 502.12
2021-06-24 14:52:00,078 runs/f30k_butd_region_bert/log
2021-06-24 14:52:00,078 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 14:52:00,081 image encoder trainable parameters: 20490144
2021-06-24 14:52:00,092 txt encoder trainable parameters: 137319072
2021-06-24 14:53:41,797 Epoch: [16][87/1133]	Eit 18200  lr 5e-05  Le 13.1769 (11.8756)	Time 1.288 (0.000)	Data 0.002 (0.000)	
2021-06-24 14:57:23,406 Epoch: [16][287/1133]	Eit 18400  lr 5e-05  Le 9.9930 (11.9134)	Time 1.244 (0.000)	Data 0.002 (0.000)	
2021-06-24 15:01:07,589 Epoch: [16][487/1133]	Eit 18600  lr 5e-05  Le 7.5707 (11.9069)	Time 1.090 (0.000)	Data 0.003 (0.000)	
2021-06-24 15:04:53,588 Epoch: [16][687/1133]	Eit 18800  lr 5e-05  Le 12.2337 (11.7967)	Time 1.376 (0.000)	Data 0.002 (0.000)	
2021-06-24 15:08:38,853 Epoch: [16][887/1133]	Eit 19000  lr 5e-05  Le 10.9758 (11.7627)	Time 0.957 (0.000)	Data 0.002 (0.000)	
2021-06-24 15:12:24,262 Epoch: [16][1087/1133]	Eit 19200  lr 5e-05  Le 8.7132 (11.7364)	Time 1.229 (0.000)	Data 0.002 (0.000)	
2021-06-24 15:13:18,725 Test: [0/40]	Le 59.9599 (59.9599)	Time 3.600 (0.000)	
2021-06-24 15:13:35,714 calculate similarity time: 0.0690608024597168
2021-06-24 15:13:36,216 Image to text: 76.8, 94.4, 97.3, 1.0, 2.3
2021-06-24 15:13:36,535 Text to image: 59.5, 83.5, 90.1, 1.0, 7.3
2021-06-24 15:13:36,535 Current rsum is 501.6
2021-06-24 15:13:38,039 runs/f30k_butd_region_bert/log
2021-06-24 15:13:38,039 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 15:13:38,041 image encoder trainable parameters: 20490144
2021-06-24 15:13:38,046 txt encoder trainable parameters: 137319072
2021-06-24 15:16:35,211 Epoch: [17][155/1133]	Eit 19400  lr 5e-05  Le 9.1478 (11.6283)	Time 1.122 (0.000)	Data 0.002 (0.000)	
2021-06-24 15:20:21,083 Epoch: [17][355/1133]	Eit 19600  lr 5e-05  Le 11.3886 (11.4606)	Time 1.015 (0.000)	Data 0.002 (0.000)	
2021-06-24 15:24:09,113 Epoch: [17][555/1133]	Eit 19800  lr 5e-05  Le 8.6023 (11.3813)	Time 1.445 (0.000)	Data 0.002 (0.000)	
2021-06-24 15:27:50,254 Epoch: [17][755/1133]	Eit 20000  lr 5e-05  Le 10.4413 (11.3639)	Time 1.268 (0.000)	Data 0.002 (0.000)	
2021-06-24 15:31:37,777 Epoch: [17][955/1133]	Eit 20200  lr 5e-05  Le 9.0837 (11.3464)	Time 1.152 (0.000)	Data 0.002 (0.000)	
2021-06-24 15:34:58,958 Test: [0/40]	Le 59.6598 (59.6597)	Time 3.452 (0.000)	
2021-06-24 15:35:15,742 calculate similarity time: 0.0652763843536377
2021-06-24 15:35:16,326 Image to text: 76.4, 94.2, 97.4, 1.0, 2.3
2021-06-24 15:35:16,778 Text to image: 59.4, 83.7, 89.9, 1.0, 7.4
2021-06-24 15:35:16,778 Current rsum is 501.02
2021-06-24 15:35:18,247 runs/f30k_butd_region_bert/log
2021-06-24 15:35:18,247 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 15:35:18,249 image encoder trainable parameters: 20490144
2021-06-24 15:35:18,255 txt encoder trainable parameters: 137319072
2021-06-24 15:35:48,511 Epoch: [18][23/1133]	Eit 20400  lr 5e-05  Le 9.3108 (10.4300)	Time 1.343 (0.000)	Data 0.002 (0.000)	
2021-06-24 15:39:29,330 Epoch: [18][223/1133]	Eit 20600  lr 5e-05  Le 7.5177 (11.0235)	Time 1.112 (0.000)	Data 0.002 (0.000)	
2021-06-24 15:43:16,037 Epoch: [18][423/1133]	Eit 20800  lr 5e-05  Le 9.2643 (11.0883)	Time 0.951 (0.000)	Data 0.002 (0.000)	
2021-06-24 15:47:00,308 Epoch: [18][623/1133]	Eit 21000  lr 5e-05  Le 9.7513 (11.1036)	Time 1.029 (0.000)	Data 0.002 (0.000)	
2021-06-24 15:50:48,487 Epoch: [18][823/1133]	Eit 21200  lr 5e-05  Le 14.3996 (11.0995)	Time 1.192 (0.000)	Data 0.002 (0.000)	
2021-06-24 15:54:31,264 Epoch: [18][1023/1133]	Eit 21400  lr 5e-05  Le 7.3458 (11.0536)	Time 1.013 (0.000)	Data 0.002 (0.000)	
2021-06-24 15:56:39,767 Test: [0/40]	Le 59.7715 (59.7714)	Time 3.723 (0.000)	
2021-06-24 15:56:56,441 calculate similarity time: 0.07471561431884766
2021-06-24 15:56:56,948 Image to text: 76.5, 94.1, 97.5, 1.0, 2.2
2021-06-24 15:56:57,443 Text to image: 59.1, 83.6, 89.7, 1.0, 7.5
2021-06-24 15:56:57,443 Current rsum is 500.46
2021-06-24 15:56:58,961 runs/f30k_butd_region_bert/log
2021-06-24 15:56:58,961 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 15:56:58,962 image encoder trainable parameters: 20490144
2021-06-24 15:56:58,968 txt encoder trainable parameters: 137319072
2021-06-24 15:58:46,683 Epoch: [19][91/1133]	Eit 21600  lr 5e-05  Le 11.7757 (10.9647)	Time 0.963 (0.000)	Data 0.002 (0.000)	
2021-06-24 16:02:32,786 Epoch: [19][291/1133]	Eit 21800  lr 5e-05  Le 9.5826 (10.8704)	Time 1.183 (0.000)	Data 0.002 (0.000)	
2021-06-24 16:06:17,698 Epoch: [19][491/1133]	Eit 22000  lr 5e-05  Le 8.0510 (10.8532)	Time 0.977 (0.000)	Data 0.002 (0.000)	
2021-06-24 16:10:03,956 Epoch: [19][691/1133]	Eit 22200  lr 5e-05  Le 10.3247 (10.8527)	Time 1.079 (0.000)	Data 0.004 (0.000)	
2021-06-24 16:13:45,952 Epoch: [19][891/1133]	Eit 22400  lr 5e-05  Le 8.7835 (10.8315)	Time 0.972 (0.000)	Data 0.002 (0.000)	
2021-06-24 16:17:33,021 Epoch: [19][1091/1133]	Eit 22600  lr 5e-05  Le 12.6595 (10.8180)	Time 1.148 (0.000)	Data 0.002 (0.000)	
2021-06-24 16:18:20,353 Test: [0/40]	Le 59.8203 (59.8203)	Time 3.552 (0.000)	
2021-06-24 16:18:36,787 calculate similarity time: 0.06385231018066406
2021-06-24 16:18:37,212 Image to text: 76.9, 93.8, 97.6, 1.0, 2.2
2021-06-24 16:18:37,528 Text to image: 59.2, 83.8, 89.8, 1.0, 7.4
2021-06-24 16:18:37,528 Current rsum is 501.0199999999999
2021-06-24 16:18:39,012 runs/f30k_butd_region_bert/log
2021-06-24 16:18:39,012 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 16:18:39,014 image encoder trainable parameters: 20490144
2021-06-24 16:18:39,019 txt encoder trainable parameters: 137319072
2021-06-24 16:21:42,727 Epoch: [20][159/1133]	Eit 22800  lr 5e-05  Le 10.5395 (10.5064)	Time 1.291 (0.000)	Data 0.003 (0.000)	
2021-06-24 16:25:23,975 Epoch: [20][359/1133]	Eit 23000  lr 5e-05  Le 11.1186 (10.7059)	Time 1.358 (0.000)	Data 0.002 (0.000)	
2021-06-24 16:29:12,405 Epoch: [20][559/1133]	Eit 23200  lr 5e-05  Le 10.1725 (10.6613)	Time 1.304 (0.000)	Data 0.002 (0.000)	
2021-06-24 16:32:55,027 Epoch: [20][759/1133]	Eit 23400  lr 5e-05  Le 13.3923 (10.6214)	Time 1.277 (0.000)	Data 0.003 (0.000)	
2021-06-24 16:36:39,405 Epoch: [20][959/1133]	Eit 23600  lr 5e-05  Le 11.5711 (10.6122)	Time 1.294 (0.000)	Data 0.002 (0.000)	
2021-06-24 16:39:56,102 Test: [0/40]	Le 59.8696 (59.8695)	Time 3.643 (0.000)	
2021-06-24 16:40:12,417 calculate similarity time: 0.06066083908081055
2021-06-24 16:40:12,872 Image to text: 77.5, 94.6, 97.3, 1.0, 2.2
2021-06-24 16:40:13,186 Text to image: 59.4, 83.7, 89.9, 1.0, 7.5
2021-06-24 16:40:13,186 Current rsum is 502.34
2021-06-24 16:40:16,811 runs/f30k_butd_region_bert/log
2021-06-24 16:40:16,811 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 16:40:16,815 image encoder trainable parameters: 20490144
2021-06-24 16:40:16,825 txt encoder trainable parameters: 137319072
2021-06-24 16:40:51,618 Epoch: [21][27/1133]	Eit 23800  lr 5e-05  Le 9.8444 (10.2104)	Time 0.943 (0.000)	Data 0.002 (0.000)	
2021-06-24 16:44:37,320 Epoch: [21][227/1133]	Eit 24000  lr 5e-05  Le 9.0895 (10.3982)	Time 1.228 (0.000)	Data 0.002 (0.000)	
2021-06-24 16:48:24,762 Epoch: [21][427/1133]	Eit 24200  lr 5e-05  Le 14.2086 (10.4383)	Time 1.078 (0.000)	Data 0.002 (0.000)	
2021-06-24 16:52:02,846 Epoch: [21][627/1133]	Eit 24400  lr 5e-05  Le 8.5813 (10.4696)	Time 1.305 (0.000)	Data 0.002 (0.000)	
2021-06-24 16:55:52,331 Epoch: [21][827/1133]	Eit 24600  lr 5e-05  Le 11.6287 (10.4360)	Time 1.304 (0.000)	Data 0.002 (0.000)	
2021-06-24 16:59:36,679 Epoch: [21][1027/1133]	Eit 24800  lr 5e-05  Le 12.0634 (10.4570)	Time 0.932 (0.000)	Data 0.002 (0.000)	
2021-06-24 17:01:36,343 Test: [0/40]	Le 59.6888 (59.6888)	Time 3.270 (0.000)	
2021-06-24 17:01:53,553 calculate similarity time: 0.06814408302307129
2021-06-24 17:01:54,013 Image to text: 78.0, 94.3, 97.1, 1.0, 2.2
2021-06-24 17:01:54,331 Text to image: 59.3, 84.1, 89.6, 1.0, 7.4
2021-06-24 17:01:54,331 Current rsum is 502.4599999999999
2021-06-24 17:01:58,115 runs/f30k_butd_region_bert/log
2021-06-24 17:01:58,115 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 17:01:58,118 image encoder trainable parameters: 20490144
2021-06-24 17:01:58,129 txt encoder trainable parameters: 137319072
2021-06-24 17:03:47,361 Epoch: [22][95/1133]	Eit 25000  lr 5e-05  Le 10.1926 (10.2770)	Time 0.942 (0.000)	Data 0.002 (0.000)	
2021-06-24 17:07:31,754 Epoch: [22][295/1133]	Eit 25200  lr 5e-05  Le 12.7044 (10.2879)	Time 1.090 (0.000)	Data 0.002 (0.000)	
2021-06-24 17:11:14,520 Epoch: [22][495/1133]	Eit 25400  lr 5e-05  Le 8.8927 (10.3089)	Time 1.152 (0.000)	Data 0.002 (0.000)	
2021-06-24 17:15:02,086 Epoch: [22][695/1133]	Eit 25600  lr 5e-05  Le 11.1041 (10.3199)	Time 0.978 (0.000)	Data 0.002 (0.000)	
2021-06-24 17:18:47,272 Epoch: [22][895/1133]	Eit 25800  lr 5e-05  Le 11.3852 (10.3438)	Time 1.308 (0.000)	Data 0.003 (0.000)	
2021-06-24 17:22:30,159 Epoch: [22][1095/1133]	Eit 26000  lr 5e-05  Le 7.3613 (10.3543)	Time 1.251 (0.000)	Data 0.003 (0.000)	
2021-06-24 17:23:14,150 Test: [0/40]	Le 59.8962 (59.8962)	Time 3.217 (0.000)	
2021-06-24 17:23:31,206 calculate similarity time: 0.08703088760375977
2021-06-24 17:23:31,700 Image to text: 77.3, 94.0, 97.6, 1.0, 2.2
2021-06-24 17:23:32,045 Text to image: 59.1, 83.9, 89.8, 1.0, 7.5
2021-06-24 17:23:32,045 Current rsum is 501.73999999999995
2021-06-24 17:23:33,566 runs/f30k_butd_region_bert/log
2021-06-24 17:23:33,567 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 17:23:33,568 image encoder trainable parameters: 20490144
2021-06-24 17:23:33,573 txt encoder trainable parameters: 137319072
2021-06-24 17:26:40,930 Epoch: [23][163/1133]	Eit 26200  lr 5e-05  Le 8.0510 (9.9800)	Time 1.361 (0.000)	Data 0.003 (0.000)	
2021-06-24 17:30:26,040 Epoch: [23][363/1133]	Eit 26400  lr 5e-05  Le 12.4616 (10.0754)	Time 1.094 (0.000)	Data 0.002 (0.000)	
2021-06-24 17:34:08,824 Epoch: [23][563/1133]	Eit 26600  lr 5e-05  Le 10.9703 (10.1303)	Time 1.001 (0.000)	Data 0.002 (0.000)	
2021-06-24 17:37:52,691 Epoch: [23][763/1133]	Eit 26800  lr 5e-05  Le 8.6207 (10.1421)	Time 1.210 (0.000)	Data 0.002 (0.000)	
2021-06-24 17:41:35,779 Epoch: [23][963/1133]	Eit 27000  lr 5e-05  Le 8.4329 (10.1174)	Time 1.240 (0.000)	Data 0.002 (0.000)	
2021-06-24 17:44:50,367 Test: [0/40]	Le 59.8467 (59.8466)	Time 3.467 (0.000)	
2021-06-24 17:45:07,129 calculate similarity time: 0.058571815490722656
2021-06-24 17:45:07,688 Image to text: 76.7, 93.2, 97.7, 1.0, 2.2
2021-06-24 17:45:08,008 Text to image: 59.0, 83.7, 89.7, 1.0, 7.6
2021-06-24 17:45:08,008 Current rsum is 500.04
2021-06-24 17:45:09,523 runs/f30k_butd_region_bert/log
2021-06-24 17:45:09,524 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 17:45:09,525 image encoder trainable parameters: 20490144
2021-06-24 17:45:09,530 txt encoder trainable parameters: 137319072
2021-06-24 17:45:48,815 Epoch: [24][31/1133]	Eit 27200  lr 5e-05  Le 12.8330 (10.3317)	Time 1.158 (0.000)	Data 0.002 (0.000)	
2021-06-24 17:49:27,923 Epoch: [24][231/1133]	Eit 27400  lr 5e-05  Le 9.2828 (9.8623)	Time 1.204 (0.000)	Data 0.002 (0.000)	
2021-06-24 17:53:10,173 Epoch: [24][431/1133]	Eit 27600  lr 5e-05  Le 9.6497 (9.9794)	Time 1.210 (0.000)	Data 0.002 (0.000)	
2021-06-24 17:56:54,115 Epoch: [24][631/1133]	Eit 27800  lr 5e-05  Le 12.8636 (10.0412)	Time 1.256 (0.000)	Data 0.002 (0.000)	
2021-06-24 18:00:40,403 Epoch: [24][831/1133]	Eit 28000  lr 5e-05  Le 10.4098 (10.0656)	Time 0.978 (0.000)	Data 0.002 (0.000)	
2021-06-24 18:04:29,917 Epoch: [24][1031/1133]	Eit 28200  lr 5e-05  Le 8.1337 (10.0732)	Time 1.056 (0.000)	Data 0.002 (0.000)	
2021-06-24 18:06:25,897 Test: [0/40]	Le 60.0168 (60.0167)	Time 3.472 (0.000)	
2021-06-24 18:06:42,883 calculate similarity time: 0.061276912689208984
2021-06-24 18:06:43,432 Image to text: 76.6, 94.0, 97.4, 1.0, 2.2
2021-06-24 18:06:43,881 Text to image: 58.9, 83.9, 89.9, 1.0, 7.5
2021-06-24 18:06:43,881 Current rsum is 500.7
[root@gpu1 vse_infty-master-my-graph-gru-vse++nowarmup]# CUDA_VISIBLE_DEVICES=0,1 python3 eval.py --dataset f30k --data_path ../data/f30k
INFO:root:Evaluating runs/f30k_butd_region_bert...
INFO:lib.evaluation:Namespace(backbone_lr_factor=0.01, backbone_path='', backbone_source='detector', backbone_warmup_epochs=5, batch_size=128, data_name='f30k', data_path='../data/f30k', embed_size=1024, embedding_warmup_epochs=2, grad_clip=2.0, img_dim=2048, input_scale_factor=1, learning_rate=0.0005, log_step=200, logger_name='runs/f30k_butd_region_bert/log', lr_update=15, margin=0.2, max_violation=True, model_name='runs/f30k_butd_region_bert', no_imgnorm=False, no_txtnorm=False, num_epochs=25, optim='adam', precomp_enc_type='basic', reset_start_epoch=False, resume='', val_step=500, vocab_path='./vocab/', vocab_size=30522, vse_mean_warmup_epochs=1, word_dim=300, workers=5)
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
INFO:lib.evaluation:Test: [0/40]	Le 60.8406 (60.8405)	Time 5.428 (0.000)	
INFO:lib.evaluation:Test: [10/40]	Le 62.0278 (61.0316)	Time 0.345 (0.000)	
INFO:lib.evaluation:Test: [20/40]	Le 59.5577 (60.8642)	Time 0.266 (0.000)	
INFO:lib.evaluation:Test: [30/40]	Le 62.8166 (61.2280)	Time 0.436 (0.000)	
INFO:lib.evaluation:Images: 1000, Captions: 5000
INFO:lib.evaluation:calculate similarity time: 0.06293678283691406
INFO:lib.evaluation:rsum: 504.3
INFO:lib.evaluation:Average i2t Recall: 90.0
INFO:lib.evaluation:Image to text: 78.3 94.3 97.4 1.0 2.4
INFO:lib.evaluation:Average t2i Recall: 78.1
INFO:lib.evaluation:Text to image: 59.3 84.6 90.5 1.0 6.9
INFO:root:Evaluating runs/release_weights/f30k_butd_grid_bert...
Traceback (most recent call last):
  File "eval.py", line 58, in <module>
    main()
  File "eval.py", line 54, in main
    evaluation.evalrank(model_path, data_path=opt.data_path, split='test', fold5=False, save_path=save_path)
  File "/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-vse++nowarmup/lib/evaluation.py", line 196, in evalrank
    checkpoint = torch.load(model_path)
  File "/root/anaconda3/lib/python3.6/site-packages/torch/serialization.py", line 571, in load
    with _open_file_like(f, 'rb') as opened_file:
  File "/root/anaconda3/lib/python3.6/site-packages/torch/serialization.py", line 229, in _open_file_like
    return _open_file(name_or_buffer, mode)
  File "/root/anaconda3/lib/python3.6/site-packages/torch/serialization.py", line 210, in __init__
    super(_open_file, self).__init__(open(name, mode))
FileNotFoundError: [Errno 2] No such file or directory: 'runs/release_weights/f30k_butd_grid_bert/model_best.pth'
[root@gpu1 vse_infty-master-my-graph-gru-vse++nowarmup]# 

