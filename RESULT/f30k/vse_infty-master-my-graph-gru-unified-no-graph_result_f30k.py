[root@gpu1 vse_infty-master-my-graph-gru-unified-no-graph]# sh train_region.sh 
2021-06-24 19:55:39,370 Namespace(backbone_lr_factor=0.01, backbone_path='', backbone_source='detector', backbone_warmup_epochs=5, batch_size=128, data_name='f30k', data_path='../data/f30k', embed_size=1024, embedding_warmup_epochs=2, grad_clip=2.0, img_dim=2048, input_scale_factor=1, learning_rate=0.0005, log_step=200, logger_name='runs/f30k_butd_region_bert/log', lr_update=15, margin=0.2, max_violation=False, model_name='runs/f30k_butd_region_bert', no_imgnorm=False, no_txtnorm=False, num_epochs=25, optim='adam', precomp_enc_type='basic', reset_start_epoch=False, resume='', val_step=500, vocab_path='./vocab/', vse_mean_warmup_epochs=1, word_dim=300, workers=10)
2021-06-24 19:55:39,370 Model name '/public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased' not found in model shortcut name list (bert-base-uncased, bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, bert-base-multilingual-cased, bert-base-chinese, bert-base-german-cased, bert-large-uncased-whole-word-masking, bert-large-cased-whole-word-masking, bert-large-uncased-whole-word-masking-finetuned-squad, bert-large-cased-whole-word-masking-finetuned-squad, bert-base-cased-finetuned-mrpc). Assuming '/public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased' is a path or url to a directory containing tokenizer files.
2021-06-24 19:55:39,370 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/added_tokens.json. We won't load it.
2021-06-24 19:55:39,370 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/special_tokens_map.json. We won't load it.
2021-06-24 19:55:39,371 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/tokenizer_config.json. We won't load it.
2021-06-24 19:55:39,371 loading file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/vocab.txt
2021-06-24 19:55:39,371 loading file None
2021-06-24 19:55:39,371 loading file None
2021-06-24 19:55:39,371 loading file None
2021-06-24 19:55:47,311 loading configuration file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/config.json
2021-06-24 19:55:47,312 Model config {
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

2021-06-24 19:55:47,312 loading weights file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/pytorch_model.bin
2021-06-24 19:55:53,460 Use adam as the optimizer, with init lr 0.0005
2021-06-24 19:55:53,461 Image encoder is data paralleled now.
2021-06-24 19:55:53,461 runs/f30k_butd_region_bert/log
2021-06-24 19:55:53,461 runs/f30k_butd_region_bert
2021-06-24 19:55:53,461 image encoder trainable parameters: 3688352
2021-06-24 19:55:53,465 txt encoder trainable parameters: 120517280
/root/anaconda3/lib/python3.6/site-packages/torch/nn/modules/rnn.py:735: UserWarning: RNN module weights are not part of single contiguous chunk of memory. This means they need to be compacted at every call, possibly greatly increasing memory usage. To compact weights again call flatten_parameters(). (Triggered internally at  /pytorch/aten/src/ATen/native/cudnn/RNN.cpp:1264.)
  self.dropout, self.training, self.bidirectional, self.batch_first)
2021-06-24 20:00:13,487 Epoch: [0][199/1133]	Eit 200  lr 0.0005  Le 10.1351 (10.1543)	Time 1.348 (0.000)	Data 0.002 (0.000)	
2021-06-24 20:05:18,545 Epoch: [0][399/1133]	Eit 400  lr 0.0005  Le 10.1131 (10.1420)	Time 1.522 (0.000)	Data 0.002 (0.000)	
2021-06-24 20:10:00,882 Epoch: [0][599/1133]	Eit 600  lr 0.0005  Le 10.1212 (10.1342)	Time 0.707 (0.000)	Data 0.002 (0.000)	
2021-06-24 20:14:40,804 Epoch: [0][799/1133]	Eit 800  lr 0.0005  Le 10.0774 (10.1253)	Time 1.505 (0.000)	Data 0.002 (0.000)	
2021-06-24 20:19:38,663 Epoch: [0][999/1133]	Eit 1000  lr 0.0005  Le 10.0573 (10.1151)	Time 1.558 (0.000)	Data 0.002 (0.000)	
2021-06-24 20:23:01,823 Test: [0/40]	Le 10.1842 (10.1842)	Time 3.676 (0.000)	
2021-06-24 20:23:25,904 calculate similarity time: 0.09039759635925293
2021-06-24 20:23:26,306 Image to text: 61.1, 84.2, 89.7, 1.0, 5.1
2021-06-24 20:23:26,632 Text to image: 43.9, 72.6, 82.0, 2.0, 10.8
2021-06-24 20:23:26,632 Current rsum is 433.46
2021-06-24 20:23:29,059 runs/f30k_butd_region_bert/log
2021-06-24 20:23:29,059 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 20:23:29,060 image encoder trainable parameters: 3688352
2021-06-24 20:23:29,065 txt encoder trainable parameters: 120517280
2021-06-24 20:25:13,078 Epoch: [1][67/1133]	Eit 1200  lr 0.0005  Le 10.0097 (10.0326)	Time 1.400 (0.000)	Data 0.002 (0.000)	
2021-06-24 20:30:13,524 Epoch: [1][267/1133]	Eit 1400  lr 0.0005  Le 10.0397 (10.0275)	Time 1.433 (0.000)	Data 0.002 (0.000)	
2021-06-24 20:35:00,896 Epoch: [1][467/1133]	Eit 1600  lr 0.0005  Le 10.0071 (10.0229)	Time 1.329 (0.000)	Data 0.002 (0.000)	
2021-06-24 20:40:01,595 Epoch: [1][667/1133]	Eit 1800  lr 0.0005  Le 9.9626 (10.0179)	Time 1.296 (0.000)	Data 0.002 (0.000)	
2021-06-24 20:45:05,541 Epoch: [1][867/1133]	Eit 2000  lr 0.0005  Le 9.9826 (10.0136)	Time 1.435 (0.000)	Data 0.001 (0.000)	
2021-06-24 20:50:06,930 Epoch: [1][1067/1133]	Eit 2200  lr 0.0005  Le 10.0026 (10.0099)	Time 1.422 (0.000)	Data 0.002 (0.000)	
2021-06-24 20:51:46,241 Test: [0/40]	Le 10.1864 (10.1864)	Time 3.656 (0.000)	
2021-06-24 20:52:09,997 calculate similarity time: 0.06923842430114746
2021-06-24 20:52:10,542 Image to text: 75.1, 92.2, 95.2, 1.0, 3.0
2021-06-24 20:52:10,872 Text to image: 53.5, 80.8, 88.2, 1.0, 7.9
2021-06-24 20:52:10,873 Current rsum is 484.88
2021-06-24 20:52:13,849 runs/f30k_butd_region_bert/log
2021-06-24 20:52:13,850 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 20:52:13,851 image encoder trainable parameters: 3688352
2021-06-24 20:52:13,860 txt encoder trainable parameters: 120517280
2021-06-24 20:55:28,723 Epoch: [2][135/1133]	Eit 2400  lr 0.0005  Le 9.9691 (9.9687)	Time 1.461 (0.000)	Data 0.001 (0.000)	
2021-06-24 21:00:30,134 Epoch: [2][335/1133]	Eit 2600  lr 0.0005  Le 9.9641 (9.9673)	Time 1.277 (0.000)	Data 0.002 (0.000)	
2021-06-24 21:05:30,915 Epoch: [2][535/1133]	Eit 2800  lr 0.0005  Le 9.9548 (9.9650)	Time 1.575 (0.000)	Data 0.002 (0.000)	
2021-06-24 21:10:30,614 Epoch: [2][735/1133]	Eit 3000  lr 0.0005  Le 9.9496 (9.9646)	Time 1.561 (0.000)	Data 0.002 (0.000)	
2021-06-24 21:15:31,249 Epoch: [2][935/1133]	Eit 3200  lr 0.0005  Le 9.9255 (9.9633)	Time 1.387 (0.000)	Data 0.002 (0.000)	
2021-06-24 21:20:15,391 Test: [0/40]	Le 10.1784 (10.1784)	Time 3.520 (0.000)	
2021-06-24 21:20:38,973 calculate similarity time: 0.07006335258483887
2021-06-24 21:20:39,523 Image to text: 74.2, 92.4, 96.4, 1.0, 2.4
2021-06-24 21:20:39,861 Text to image: 56.8, 83.4, 89.7, 1.0, 6.9
2021-06-24 21:20:39,861 Current rsum is 492.98
2021-06-24 21:20:42,935 runs/f30k_butd_region_bert/log
2021-06-24 21:20:42,935 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 21:20:42,936 image encoder trainable parameters: 3688352
2021-06-24 21:20:42,945 txt encoder trainable parameters: 120517280
2021-06-24 21:20:51,603 Epoch: [3][3/1133]	Eit 3400  lr 0.0005  Le 9.9560 (9.9394)	Time 1.595 (0.000)	Data 0.002 (0.000)	
2021-06-24 21:25:51,529 Epoch: [3][203/1133]	Eit 3600  lr 0.0005  Le 9.9371 (9.9389)	Time 1.491 (0.000)	Data 0.002 (0.000)	
2021-06-24 21:30:54,314 Epoch: [3][403/1133]	Eit 3800  lr 0.0005  Le 9.9270 (9.9392)	Time 1.480 (0.000)	Data 0.002 (0.000)	
2021-06-24 21:35:58,503 Epoch: [3][603/1133]	Eit 4000  lr 0.0005  Le 9.9450 (9.9393)	Time 1.563 (0.000)	Data 0.002 (0.000)	
2021-06-24 21:40:47,591 Epoch: [3][803/1133]	Eit 4200  lr 0.0005  Le 9.9189 (9.9380)	Time 1.343 (0.000)	Data 0.002 (0.000)	
2021-06-24 21:45:51,320 Epoch: [3][1003/1133]	Eit 4400  lr 0.0005  Le 9.9602 (9.9373)	Time 1.644 (0.000)	Data 0.002 (0.000)	
2021-06-24 21:49:08,289 Test: [0/40]	Le 10.1862 (10.1862)	Time 3.807 (0.000)	
2021-06-24 21:49:32,727 calculate similarity time: 0.07306528091430664
2021-06-24 21:49:33,248 Image to text: 77.2, 93.9, 97.6, 1.0, 2.2
2021-06-24 21:49:33,679 Text to image: 58.2, 84.2, 90.3, 1.0, 6.9
2021-06-24 21:49:33,679 Current rsum is 501.38000000000005
2021-06-24 21:49:36,670 runs/f30k_butd_region_bert/log
2021-06-24 21:49:36,670 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 21:49:36,672 image encoder trainable parameters: 3688352
2021-06-24 21:49:36,680 txt encoder trainable parameters: 120517280
2021-06-24 21:51:26,969 Epoch: [4][71/1133]	Eit 4600  lr 0.0005  Le 9.9021 (9.9183)	Time 1.634 (0.000)	Data 0.002 (0.000)	
2021-06-24 21:56:28,516 Epoch: [4][271/1133]	Eit 4800  lr 0.0005  Le 9.8922 (9.9172)	Time 1.556 (0.000)	Data 0.002 (0.000)	
2021-06-24 22:01:28,754 Epoch: [4][471/1133]	Eit 5000  lr 0.0005  Le 9.9261 (9.9177)	Time 1.549 (0.000)	Data 0.002 (0.000)	
2021-06-24 22:06:18,324 Epoch: [4][671/1133]	Eit 5200  lr 0.0005  Le 9.9294 (9.9185)	Time 1.335 (0.000)	Data 0.002 (0.000)	
2021-06-24 22:11:20,969 Epoch: [4][871/1133]	Eit 5400  lr 0.0005  Le 9.8902 (9.9182)	Time 1.463 (0.000)	Data 0.002 (0.000)	
2021-06-24 22:16:23,293 Epoch: [4][1071/1133]	Eit 5600  lr 0.0005  Le 9.9462 (9.9181)	Time 1.244 (0.000)	Data 0.002 (0.000)	
2021-06-24 22:17:58,301 Test: [0/40]	Le 10.1855 (10.1855)	Time 3.681 (0.000)	
2021-06-24 22:18:22,013 calculate similarity time: 0.061753273010253906
2021-06-24 22:18:22,571 Image to text: 78.6, 94.7, 97.7, 1.0, 2.3
2021-06-24 22:18:22,897 Text to image: 60.0, 84.9, 90.8, 1.0, 6.6
2021-06-24 22:18:22,897 Current rsum is 506.65999999999997
2021-06-24 22:18:25,818 runs/f30k_butd_region_bert/log
2021-06-24 22:18:25,818 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 22:18:25,819 image encoder trainable parameters: 3688352
2021-06-24 22:18:25,828 txt encoder trainable parameters: 120517280
2021-06-24 22:22:03,239 Epoch: [5][139/1133]	Eit 5800  lr 0.0005  Le 9.8835 (9.8988)	Time 1.550 (0.000)	Data 0.002 (0.000)	
2021-06-24 22:26:51,897 Epoch: [5][339/1133]	Eit 6000  lr 0.0005  Le 9.9098 (9.9011)	Time 1.488 (0.000)	Data 0.002 (0.000)	
2021-06-24 22:31:54,508 Epoch: [5][539/1133]	Eit 6200  lr 0.0005  Le 9.9244 (9.9018)	Time 1.519 (0.000)	Data 0.002 (0.000)	
2021-06-24 22:36:55,820 Epoch: [5][739/1133]	Eit 6400  lr 0.0005  Le 9.8923 (9.9021)	Time 1.441 (0.000)	Data 0.002 (0.000)	
2021-06-24 22:41:59,023 Epoch: [5][939/1133]	Eit 6600  lr 0.0005  Le 9.9046 (9.9023)	Time 1.552 (0.000)	Data 0.002 (0.000)	
2021-06-24 22:46:52,773 Test: [0/40]	Le 10.1822 (10.1822)	Time 3.520 (0.000)	
2021-06-24 22:47:16,574 calculate similarity time: 0.08623862266540527
2021-06-24 22:47:17,121 Image to text: 80.2, 94.9, 97.8, 1.0, 2.1
2021-06-24 22:47:17,450 Text to image: 60.5, 84.8, 90.8, 1.0, 6.6
2021-06-24 22:47:17,450 Current rsum is 509.08000000000004
2021-06-24 22:47:20,389 runs/f30k_butd_region_bert/log
2021-06-24 22:47:20,390 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 22:47:20,392 image encoder trainable parameters: 3688352
2021-06-24 22:47:20,401 txt encoder trainable parameters: 120517280
2021-06-24 22:47:31,415 Epoch: [6][7/1133]	Eit 6800  lr 0.0005  Le 9.8809 (9.8756)	Time 1.389 (0.000)	Data 0.004 (0.000)	
2021-06-24 22:52:26,866 Epoch: [6][207/1133]	Eit 7000  lr 0.0005  Le 9.8702 (9.8861)	Time 1.585 (0.000)	Data 0.002 (0.000)	
2021-06-24 22:57:30,025 Epoch: [6][407/1133]	Eit 7200  lr 0.0005  Le 9.9057 (9.8881)	Time 1.466 (0.000)	Data 0.002 (0.000)	
2021-06-24 23:02:26,513 Epoch: [6][607/1133]	Eit 7400  lr 0.0005  Le 9.9249 (9.8883)	Time 1.635 (0.000)	Data 0.002 (0.000)	
2021-06-24 23:07:25,117 Epoch: [6][807/1133]	Eit 7600  lr 0.0005  Le 9.8727 (9.8883)	Time 1.569 (0.000)	Data 0.002 (0.000)	
2021-06-24 23:12:14,681 Epoch: [6][1007/1133]	Eit 7800  lr 0.0005  Le 9.8627 (9.8891)	Time 1.597 (0.000)	Data 0.002 (0.000)	
2021-06-24 23:15:24,809 Test: [0/40]	Le 10.1810 (10.1810)	Time 3.685 (0.000)	
2021-06-24 23:15:48,993 calculate similarity time: 0.09480094909667969
2021-06-24 23:15:49,519 Image to text: 80.5, 95.9, 98.3, 1.0, 1.9
2021-06-24 23:15:49,951 Text to image: 60.0, 84.7, 90.9, 1.0, 6.7
2021-06-24 23:15:49,951 Current rsum is 510.36
2021-06-24 23:15:53,012 runs/f30k_butd_region_bert/log
2021-06-24 23:15:53,013 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 23:15:53,014 image encoder trainable parameters: 3688352
2021-06-24 23:15:53,023 txt encoder trainable parameters: 120517280
2021-06-24 23:17:50,397 Epoch: [7][75/1133]	Eit 8000  lr 0.0005  Le 9.8653 (9.8772)	Time 1.296 (0.000)	Data 0.002 (0.000)	
2021-06-24 23:22:49,205 Epoch: [7][275/1133]	Eit 8200  lr 0.0005  Le 9.8763 (9.8776)	Time 1.454 (0.000)	Data 0.010 (0.000)	
2021-06-24 23:27:51,331 Epoch: [7][475/1133]	Eit 8400  lr 0.0005  Le 9.8693 (9.8784)	Time 1.564 (0.000)	Data 0.002 (0.000)	
2021-06-24 23:32:47,140 Epoch: [7][675/1133]	Eit 8600  lr 0.0005  Le 9.8798 (9.8797)	Time 1.041 (0.000)	Data 0.002 (0.000)	
2021-06-24 23:37:43,299 Epoch: [7][875/1133]	Eit 8800  lr 0.0005  Le 9.8657 (9.8799)	Time 1.587 (0.000)	Data 0.002 (0.000)	
2021-06-24 23:42:44,307 Epoch: [7][1075/1133]	Eit 9000  lr 0.0005  Le 9.8872 (9.8797)	Time 1.639 (0.000)	Data 0.002 (0.000)	
2021-06-24 23:44:11,678 Test: [0/40]	Le 10.1824 (10.1824)	Time 3.632 (0.000)	
2021-06-24 23:44:35,750 calculate similarity time: 0.08089494705200195
2021-06-24 23:44:36,300 Image to text: 78.5, 94.9, 97.7, 1.0, 2.1
2021-06-24 23:44:36,645 Text to image: 60.4, 84.9, 91.1, 1.0, 6.6
2021-06-24 23:44:36,645 Current rsum is 507.58000000000004
2021-06-24 23:44:37,862 runs/f30k_butd_region_bert/log
2021-06-24 23:44:37,862 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 23:44:37,863 image encoder trainable parameters: 3688352
2021-06-24 23:44:37,868 txt encoder trainable parameters: 120517280
2021-06-24 23:48:18,669 Epoch: [8][143/1133]	Eit 9200  lr 0.0005  Le 9.8841 (9.8668)	Time 1.818 (0.000)	Data 0.002 (0.000)	
2021-06-24 23:53:20,045 Epoch: [8][343/1133]	Eit 9400  lr 0.0005  Le 9.8780 (9.8689)	Time 1.317 (0.000)	Data 0.002 (0.000)	
2021-06-24 23:58:06,440 Epoch: [8][543/1133]	Eit 9600  lr 0.0005  Le 9.8706 (9.8686)	Time 1.215 (0.000)	Data 0.002 (0.000)	
2021-06-25 00:03:07,126 Epoch: [8][743/1133]	Eit 9800  lr 0.0005  Le 9.8901 (9.8690)	Time 1.430 (0.000)	Data 0.002 (0.000)	
2021-06-25 00:08:07,758 Epoch: [8][943/1133]	Eit 10000  lr 0.0005  Le 9.8842 (9.8697)	Time 1.422 (0.000)	Data 0.002 (0.000)	
2021-06-25 00:12:54,074 Test: [0/40]	Le 10.1831 (10.1830)	Time 3.643 (0.000)	
2021-06-25 00:13:18,286 calculate similarity time: 0.06699275970458984
2021-06-25 00:13:18,786 Image to text: 79.9, 96.0, 98.3, 1.0, 1.9
2021-06-25 00:13:19,129 Text to image: 60.2, 84.8, 90.9, 1.0, 6.6
2021-06-25 00:13:19,129 Current rsum is 510.02
2021-06-25 00:13:20,354 runs/f30k_butd_region_bert/log
2021-06-25 00:13:20,354 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-25 00:13:20,354 image encoder trainable parameters: 3688352
2021-06-25 00:13:20,359 txt encoder trainable parameters: 120517280
2021-06-25 00:13:41,201 Epoch: [9][11/1133]	Eit 10200  lr 0.0005  Le 9.8587 (9.8566)	Time 1.297 (0.000)	Data 0.002 (0.000)	
2021-06-25 00:18:30,374 Epoch: [9][211/1133]	Eit 10400  lr 0.0005  Le 9.8460 (9.8584)	Time 1.335 (0.000)	Data 0.002 (0.000)	
2021-06-25 00:23:32,958 Epoch: [9][411/1133]	Eit 10600  lr 0.0005  Le 9.8371 (9.8585)	Time 1.413 (0.000)	Data 0.002 (0.000)	
2021-06-25 00:28:34,246 Epoch: [9][611/1133]	Eit 10800  lr 0.0005  Le 9.8333 (9.8595)	Time 1.669 (0.000)	Data 0.002 (0.000)	
2021-06-25 00:33:36,871 Epoch: [9][811/1133]	Eit 11000  lr 0.0005  Le 9.8403 (9.8600)	Time 1.386 (0.000)	Data 0.002 (0.000)	
2021-06-25 00:38:40,665 Epoch: [9][1011/1133]	Eit 11200  lr 0.0005  Le 9.8873 (9.8612)	Time 1.268 (0.000)	Data 0.002 (0.000)	
2021-06-25 00:41:33,897 Test: [0/40]	Le 10.1831 (10.1831)	Time 3.560 (0.000)	
2021-06-25 00:41:58,509 calculate similarity time: 0.0730741024017334
2021-06-25 00:41:59,034 Image to text: 81.3, 96.0, 98.1, 1.0, 1.9
2021-06-25 00:41:59,466 Text to image: 61.5, 85.4, 91.2, 1.0, 6.4
2021-06-25 00:41:59,467 Current rsum is 513.5600000000001
2021-06-25 00:42:02,704 runs/f30k_butd_region_bert/log
2021-06-25 00:42:02,705 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-25 00:42:02,706 image encoder trainable parameters: 3688352
2021-06-25 00:42:02,716 txt encoder trainable parameters: 120517280
2021-06-25 00:44:05,818 Epoch: [10][79/1133]	Eit 11400  lr 0.0005  Le 9.8510 (9.8541)	Time 1.744 (0.000)	Data 0.002 (0.000)	
2021-06-25 00:49:08,223 Epoch: [10][279/1133]	Eit 11600  lr 0.0005  Le 9.8628 (9.8524)	Time 1.625 (0.000)	Data 0.002 (0.000)	
2021-06-25 00:54:07,481 Epoch: [10][479/1133]	Eit 11800  lr 0.0005  Le 9.8850 (9.8540)	Time 1.728 (0.000)	Data 0.002 (0.000)	
2021-06-25 00:59:10,139 Epoch: [10][679/1133]	Eit 12000  lr 0.0005  Le 9.8470 (9.8533)	Time 1.554 (0.000)	Data 0.002 (0.000)	
2021-06-25 01:03:57,540 Epoch: [10][879/1133]	Eit 12200  lr 0.0005  Le 9.8338 (9.8535)	Time 1.485 (0.000)	Data 0.002 (0.000)	
2021-06-25 01:09:01,921 Epoch: [10][1079/1133]	Eit 12400  lr 0.0005  Le 9.8665 (9.8542)	Time 1.411 (0.000)	Data 0.001 (0.000)	
2021-06-25 01:10:23,142 Test: [0/40]	Le 10.1820 (10.1820)	Time 3.521 (0.000)	
2021-06-25 01:10:47,469 calculate similarity time: 0.06988048553466797
2021-06-25 01:10:48,025 Image to text: 80.4, 95.6, 97.7, 1.0, 2.3
2021-06-25 01:10:48,343 Text to image: 61.1, 85.6, 90.9, 1.0, 7.2
2021-06-25 01:10:48,343 Current rsum is 511.38
2021-06-25 01:10:49,565 runs/f30k_butd_region_bert/log
2021-06-25 01:10:49,566 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-25 01:10:49,566 image encoder trainable parameters: 3688352
2021-06-25 01:10:49,571 txt encoder trainable parameters: 120517280
2021-06-25 01:14:35,075 Epoch: [11][147/1133]	Eit 12600  lr 0.0005  Le 9.8207 (9.8422)	Time 1.461 (0.000)	Data 0.002 (0.000)	
2021-06-25 01:19:36,894 Epoch: [11][347/1133]	Eit 12800  lr 0.0005  Le 9.8208 (9.8449)	Time 1.421 (0.000)	Data 0.002 (0.000)	
2021-06-25 01:24:38,335 Epoch: [11][547/1133]	Eit 13000  lr 0.0005  Le 9.8434 (9.8459)	Time 1.686 (0.000)	Data 0.001 (0.000)	
2021-06-25 01:29:28,817 Epoch: [11][747/1133]	Eit 13200  lr 0.0005  Le 9.8301 (9.8467)	Time 1.443 (0.000)	Data 0.002 (0.000)	
2021-06-25 01:34:28,213 Epoch: [11][947/1133]	Eit 13400  lr 0.0005  Le 9.8260 (9.8473)	Time 1.520 (0.000)	Data 0.002 (0.000)	
2021-06-25 01:39:08,568 Test: [0/40]	Le 10.1823 (10.1823)	Time 3.507 (0.000)	
2021-06-25 01:39:31,876 calculate similarity time: 0.07480168342590332
2021-06-25 01:39:32,384 Image to text: 79.5, 96.0, 98.2, 1.0, 2.1
2021-06-25 01:39:32,717 Text to image: 60.5, 85.1, 91.1, 1.0, 6.9
2021-06-25 01:39:32,717 Current rsum is 510.41999999999996
2021-06-25 01:39:33,920 runs/f30k_butd_region_bert/log
2021-06-25 01:39:33,920 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-25 01:39:33,921 image encoder trainable parameters: 3688352
2021-06-25 01:39:33,925 txt encoder trainable parameters: 120517280
2021-06-25 01:40:00,757 Epoch: [12][15/1133]	Eit 13600  lr 0.0005  Le 9.8262 (9.8403)	Time 1.536 (0.000)	Data 0.002 (0.000)	
2021-06-25 01:45:04,612 Epoch: [12][215/1133]	Eit 13800  lr 0.0005  Le 9.8467 (9.8404)	Time 1.525 (0.000)	Data 0.002 (0.000)	
2021-06-25 01:49:58,755 Epoch: [12][415/1133]	Eit 14000  lr 0.0005  Le 9.8191 (9.8401)	Time 1.532 (0.000)	Data 0.002 (0.000)	
2021-06-25 01:55:00,009 Epoch: [12][615/1133]	Eit 14200  lr 0.0005  Le 9.8478 (9.8410)	Time 1.502 (0.000)	Data 0.002 (0.000)	
2021-06-25 02:00:02,339 Epoch: [12][815/1133]	Eit 14400  lr 0.0005  Le 9.8437 (9.8418)	Time 1.522 (0.000)	Data 0.004 (0.000)	
2021-06-25 02:05:05,145 Epoch: [12][1015/1133]	Eit 14600  lr 0.0005  Le 9.8408 (9.8425)	Time 1.694 (0.000)	Data 0.002 (0.000)	
2021-06-25 02:08:05,243 Test: [0/40]	Le 10.1852 (10.1852)	Time 3.543 (0.000)	
2021-06-25 02:08:28,846 calculate similarity time: 0.08057284355163574
2021-06-25 02:08:29,398 Image to text: 80.8, 95.4, 97.9, 1.0, 2.0
2021-06-25 02:08:29,720 Text to image: 60.9, 85.1, 90.7, 1.0, 7.4
2021-06-25 02:08:29,720 Current rsum is 510.68000000000006
2021-06-25 02:08:30,957 runs/f30k_butd_region_bert/log
2021-06-25 02:08:30,957 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-25 02:08:30,958 image encoder trainable parameters: 3688352
2021-06-25 02:08:30,962 txt encoder trainable parameters: 120517280
2021-06-25 02:10:27,335 Epoch: [13][83/1133]	Eit 14800  lr 0.0005  Le 9.8358 (9.8327)	Time 1.446 (0.000)	Data 0.002 (0.000)	
2021-06-25 02:15:30,904 Epoch: [13][283/1133]	Eit 15000  lr 0.0005  Le 9.8374 (9.8347)	Time 1.430 (0.000)	Data 0.002 (0.000)	
2021-06-25 02:20:32,030 Epoch: [13][483/1133]	Eit 15200  lr 0.0005  Le 9.8444 (9.8351)	Time 1.395 (0.000)	Data 0.002 (0.000)	
2021-06-25 02:25:33,837 Epoch: [13][683/1133]	Eit 15400  lr 0.0005  Le 9.8378 (9.8353)	Time 1.469 (0.000)	Data 0.002 (0.000)	
2021-06-25 02:30:35,438 Epoch: [13][883/1133]	Eit 15600  lr 0.0005  Le 9.8287 (9.8362)	Time 1.747 (0.000)	Data 0.001 (0.000)	
2021-06-25 02:35:25,909 Epoch: [13][1083/1133]	Eit 15800  lr 0.0005  Le 9.8224 (9.8368)	Time 1.401 (0.000)	Data 0.002 (0.000)	
2021-06-25 02:36:42,002 Test: [0/40]	Le 10.1826 (10.1826)	Time 3.545 (0.000)	
2021-06-25 02:37:06,242 calculate similarity time: 0.08281970024108887
2021-06-25 02:37:06,796 Image to text: 79.8, 95.4, 98.1, 1.0, 2.4
2021-06-25 02:37:07,127 Text to image: 60.8, 85.5, 90.9, 1.0, 7.4
2021-06-25 02:37:07,127 Current rsum is 510.48
2021-06-25 02:37:08,377 runs/f30k_butd_region_bert/log
2021-06-25 02:37:08,377 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-25 02:37:08,378 image encoder trainable parameters: 3688352
2021-06-25 02:37:08,383 txt encoder trainable parameters: 120517280
2021-06-25 02:41:02,459 Epoch: [14][151/1133]	Eit 16000  lr 0.0005  Le 9.8019 (9.8240)	Time 1.627 (0.000)	Data 0.002 (0.000)	
2021-06-25 02:46:01,060 Epoch: [14][351/1133]	Eit 16200  lr 0.0005  Le 9.8200 (9.8280)	Time 1.538 (0.000)	Data 0.002 (0.000)	
2021-06-25 02:51:07,772 Epoch: [14][551/1133]	Eit 16400  lr 0.0005  Le 9.8451 (9.8292)	Time 1.518 (0.000)	Data 0.002 (0.000)	
2021-06-25 02:55:57,640 Epoch: [14][751/1133]	Eit 16600  lr 0.0005  Le 9.8355 (9.8305)	Time 1.380 (0.000)	Data 0.002 (0.000)	
2021-06-25 03:00:57,033 Epoch: [14][951/1133]	Eit 16800  lr 0.0005  Le 9.8387 (9.8308)	Time 1.562 (0.000)	Data 0.002 (0.000)	
2021-06-25 03:05:32,980 Test: [0/40]	Le 10.1821 (10.1821)	Time 3.713 (0.000)	
2021-06-25 03:05:57,086 calculate similarity time: 0.06974482536315918
2021-06-25 03:05:57,602 Image to text: 78.6, 95.9, 98.1, 1.0, 2.3
2021-06-25 03:05:57,932 Text to image: 60.4, 84.7, 90.3, 1.0, 7.4
2021-06-25 03:05:57,932 Current rsum is 508.08000000000004
2021-06-25 03:05:59,152 runs/f30k_butd_region_bert/log
2021-06-25 03:05:59,153 runs/f30k_butd_region_bert
2021-06-25 03:05:59,153 Current epoch num is 15, decrease all lr by 10
2021-06-25 03:05:59,153 new lr 5e-05
2021-06-25 03:05:59,153 new lr 5e-06
2021-06-25 03:05:59,153 new lr 5e-05
Use VSE++ objective.
2021-06-25 03:05:59,154 image encoder trainable parameters: 3688352
2021-06-25 03:05:59,158 txt encoder trainable parameters: 120517280
2021-06-25 03:06:33,582 Epoch: [15][19/1133]	Eit 17000  lr 5e-05  Le 9.8388 (9.8266)	Time 1.389 (0.000)	Data 0.002 (0.000)	
2021-06-25 03:11:35,681 Epoch: [15][219/1133]	Eit 17200  lr 5e-05  Le 9.8129 (9.8212)	Time 1.529 (0.000)	Data 0.002 (0.000)	
2021-06-25 03:16:37,982 Epoch: [15][419/1133]	Eit 17400  lr 5e-05  Le 9.8167 (9.8179)	Time 1.448 (0.000)	Data 0.002 (0.000)	
2021-06-25 03:21:30,509 Epoch: [15][619/1133]	Eit 17600  lr 5e-05  Le 9.8223 (9.8166)	Time 1.316 (0.000)	Data 0.002 (0.000)	
2021-06-25 03:26:32,010 Epoch: [15][819/1133]	Eit 17800  lr 5e-05  Le 9.8066 (9.8160)	Time 1.482 (0.000)	Data 0.002 (0.000)	
2021-06-25 03:31:32,807 Epoch: [15][1019/1133]	Eit 18000  lr 5e-05  Le 9.8144 (9.8157)	Time 1.292 (0.000)	Data 0.002 (0.000)	
2021-06-25 03:34:24,244 Test: [0/40]	Le 10.1829 (10.1829)	Time 3.613 (0.000)	
2021-06-25 03:34:48,426 calculate similarity time: 0.07198357582092285
2021-06-25 03:34:48,979 Image to text: 81.6, 95.4, 98.7, 1.0, 2.3
2021-06-25 03:34:49,335 Text to image: 61.2, 85.8, 91.2, 1.0, 7.2
2021-06-25 03:34:49,335 Current rsum is 513.96
2021-06-25 03:34:52,473 runs/f30k_butd_region_bert/log
2021-06-25 03:34:52,473 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-25 03:34:52,474 image encoder trainable parameters: 3688352
2021-06-25 03:34:52,480 txt encoder trainable parameters: 120517280
2021-06-25 03:37:06,614 Epoch: [16][87/1133]	Eit 18200  lr 5e-05  Le 9.8168 (9.8109)	Time 1.455 (0.000)	Data 0.003 (0.000)	
2021-06-25 03:41:56,442 Epoch: [16][287/1133]	Eit 18400  lr 5e-05  Le 9.8087 (9.8103)	Time 1.959 (0.000)	Data 0.002 (0.000)	
2021-06-25 03:46:56,095 Epoch: [16][487/1133]	Eit 18600  lr 5e-05  Le 9.8182 (9.8094)	Time 1.616 (0.000)	Data 0.002 (0.000)	
2021-06-25 03:51:58,549 Epoch: [16][687/1133]	Eit 18800  lr 5e-05  Le 9.8160 (9.8088)	Time 1.477 (0.000)	Data 0.002 (0.000)	
2021-06-25 03:56:57,453 Epoch: [16][887/1133]	Eit 19000  lr 5e-05  Le 9.8189 (9.8084)	Time 1.445 (0.000)	Data 0.002 (0.000)	
2021-06-25 04:01:59,292 Epoch: [16][1087/1133]	Eit 19200  lr 5e-05  Le 9.8002 (9.8082)	Time 0.786 (0.000)	Data 0.002 (0.000)	
2021-06-25 04:02:57,758 Test: [0/40]	Le 10.1820 (10.1820)	Time 3.760 (0.000)	
2021-06-25 04:03:21,394 calculate similarity time: 0.0666799545288086
2021-06-25 04:03:21,910 Image to text: 81.5, 95.9, 98.1, 1.0, 2.2
2021-06-25 04:03:22,299 Text to image: 61.9, 85.8, 91.1, 1.0, 7.0
2021-06-25 04:03:22,299 Current rsum is 514.24
2021-06-25 04:03:25,616 runs/f30k_butd_region_bert/log
2021-06-25 04:03:25,617 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-25 04:03:25,618 image encoder trainable parameters: 3688352
2021-06-25 04:03:25,629 txt encoder trainable parameters: 120517280
2021-06-25 04:07:23,075 Epoch: [17][155/1133]	Eit 19400  lr 5e-05  Le 9.8056 (9.8054)	Time 1.273 (0.000)	Data 0.002 (0.000)	
2021-06-25 04:12:23,577 Epoch: [17][355/1133]	Eit 19600  lr 5e-05  Le 9.7985 (9.8049)	Time 1.482 (0.000)	Data 0.002 (0.000)	
2021-06-25 04:17:24,929 Epoch: [17][555/1133]	Eit 19800  lr 5e-05  Le 9.8219 (9.8053)	Time 1.499 (0.000)	Data 0.002 (0.000)	
2021-06-25 04:22:29,889 Epoch: [17][755/1133]	Eit 20000  lr 5e-05  Le 9.8045 (9.8052)	Time 1.445 (0.000)	Data 0.002 (0.000)	
2021-06-25 04:27:24,968 Epoch: [17][955/1133]	Eit 20200  lr 5e-05  Le 9.7867 (9.8054)	Time 1.708 (0.000)	Data 0.002 (0.000)	
2021-06-25 04:31:53,028 Test: [0/40]	Le 10.1819 (10.1819)	Time 3.649 (0.000)	
2021-06-25 04:32:16,885 calculate similarity time: 0.07130670547485352
2021-06-25 04:32:17,441 Image to text: 80.9, 96.2, 98.1, 1.0, 2.1
2021-06-25 04:32:17,764 Text to image: 61.8, 86.0, 91.2, 1.0, 7.1
2021-06-25 04:32:17,765 Current rsum is 514.1
2021-06-25 04:32:18,982 runs/f30k_butd_region_bert/log
2021-06-25 04:32:18,983 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-25 04:32:18,983 image encoder trainable parameters: 3688352
2021-06-25 04:32:18,988 txt encoder trainable parameters: 120517280
2021-06-25 04:32:58,604 Epoch: [18][23/1133]	Eit 20400  lr 5e-05  Le 9.8055 (9.8030)	Time 1.506 (0.000)	Data 0.002 (0.000)	
2021-06-25 04:37:57,905 Epoch: [18][223/1133]	Eit 20600  lr 5e-05  Le 9.7982 (9.8028)	Time 1.466 (0.000)	Data 0.002 (0.000)	
2021-06-25 04:42:57,448 Epoch: [18][423/1133]	Eit 20800  lr 5e-05  Le 9.8146 (9.8033)	Time 1.332 (0.000)	Data 0.002 (0.000)	
2021-06-25 04:47:48,319 Epoch: [18][623/1133]	Eit 21000  lr 5e-05  Le 9.8207 (9.8035)	Time 1.488 (0.000)	Data 0.002 (0.000)	
2021-06-25 04:52:50,907 Epoch: [18][823/1133]	Eit 21200  lr 5e-05  Le 9.8146 (9.8034)	Time 1.436 (0.000)	Data 0.002 (0.000)	
2021-06-25 04:57:52,032 Epoch: [18][1023/1133]	Eit 21400  lr 5e-05  Le 9.7864 (9.8034)	Time 1.518 (0.000)	Data 0.002 (0.000)	
2021-06-25 05:00:36,889 Test: [0/40]	Le 10.1818 (10.1818)	Time 3.388 (0.000)	
2021-06-25 05:01:01,005 calculate similarity time: 0.06667494773864746
2021-06-25 05:01:01,476 Image to text: 79.8, 96.1, 98.0, 1.0, 2.2
2021-06-25 05:01:01,925 Text to image: 61.8, 86.0, 91.0, 1.0, 7.1
2021-06-25 05:01:01,926 Current rsum is 512.64
2021-06-25 05:01:03,206 runs/f30k_butd_region_bert/log
2021-06-25 05:01:03,207 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-25 05:01:03,207 image encoder trainable parameters: 3688352
2021-06-25 05:01:03,211 txt encoder trainable parameters: 120517280
2021-06-25 05:03:24,028 Epoch: [19][91/1133]	Eit 21600  lr 5e-05  Le 9.7921 (9.8043)	Time 1.752 (0.000)	Data 0.001 (0.000)	
2021-06-25 05:08:25,133 Epoch: [19][291/1133]	Eit 21800  lr 5e-05  Le 9.7947 (9.8027)	Time 1.497 (0.000)	Data 0.002 (0.000)	
2021-06-25 05:13:13,690 Epoch: [19][491/1133]	Eit 22000  lr 5e-05  Le 9.7922 (9.8028)	Time 1.431 (0.000)	Data 0.001 (0.000)	
2021-06-25 05:18:14,035 Epoch: [19][691/1133]	Eit 22200  lr 5e-05  Le 9.8115 (9.8025)	Time 1.474 (0.000)	Data 0.002 (0.000)	
2021-06-25 05:23:15,291 Epoch: [19][891/1133]	Eit 22400  lr 5e-05  Le 9.7858 (9.8022)	Time 1.652 (0.000)	Data 0.002 (0.000)	
2021-06-25 05:28:18,054 Epoch: [19][1091/1133]	Eit 22600  lr 5e-05  Le 9.8191 (9.8020)	Time 1.640 (0.000)	Data 0.002 (0.000)	
2021-06-25 05:29:22,092 Test: [0/40]	Le 10.1817 (10.1817)	Time 3.529 (0.000)	
2021-06-25 05:29:46,498 calculate similarity time: 0.07878327369689941
2021-06-25 05:29:46,995 Image to text: 80.4, 96.4, 98.3, 1.0, 2.1
2021-06-25 05:29:47,425 Text to image: 61.9, 86.0, 91.0, 1.0, 7.2
2021-06-25 05:29:47,425 Current rsum is 513.94
2021-06-25 05:29:48,665 runs/f30k_butd_region_bert/log
2021-06-25 05:29:48,665 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-25 05:29:48,666 image encoder trainable parameters: 3688352
2021-06-25 05:29:48,670 txt encoder trainable parameters: 120517280
2021-06-25 05:32:54,660 Epoch: [20][159/1133]	Eit 22800  lr 5e-05  Le 9.8107 (9.7995)	Time 0.712 (0.000)	Data 0.009 (0.000)	
2021-06-25 05:35:23,294 Epoch: [20][359/1133]	Eit 23000  lr 5e-05  Le 9.8043 (9.8000)	Time 0.694 (0.000)	Data 0.002 (0.000)	
2021-06-25 05:37:51,464 Epoch: [20][559/1133]	Eit 23200  lr 5e-05  Le 9.8074 (9.7999)	Time 0.673 (0.000)	Data 0.002 (0.000)	
2021-06-25 05:40:16,440 Epoch: [20][759/1133]	Eit 23400  lr 5e-05  Le 9.8114 (9.7997)	Time 0.690 (0.000)	Data 0.002 (0.000)	
2021-06-25 05:42:42,175 Epoch: [20][959/1133]	Eit 23600  lr 5e-05  Le 9.7905 (9.7996)	Time 0.676 (0.000)	Data 0.002 (0.000)	
2021-06-25 05:44:52,876 Test: [0/40]	Le 10.1821 (10.1821)	Time 3.896 (0.000)	
2021-06-25 05:45:05,375 calculate similarity time: 0.05379652976989746
2021-06-25 05:45:05,766 Image to text: 80.2, 95.8, 98.2, 1.0, 2.1
2021-06-25 05:45:06,074 Text to image: 61.6, 86.1, 90.9, 1.0, 7.3
2021-06-25 05:45:06,074 Current rsum is 512.78
2021-06-25 05:45:07,444 runs/f30k_butd_region_bert/log
2021-06-25 05:45:07,445 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-25 05:45:07,445 image encoder trainable parameters: 3688352
2021-06-25 05:45:07,450 txt encoder trainable parameters: 120517280
2021-06-25 05:45:32,150 Epoch: [21][27/1133]	Eit 23800  lr 5e-05  Le 9.8201 (9.7972)	Time 0.818 (0.000)	Data 0.002 (0.000)	
2021-06-25 05:48:01,537 Epoch: [21][227/1133]	Eit 24000  lr 5e-05  Le 9.7820 (9.7987)	Time 0.726 (0.000)	Data 0.002 (0.000)	
2021-06-25 05:50:29,169 Epoch: [21][427/1133]	Eit 24200  lr 5e-05  Le 9.8101 (9.7989)	Time 0.705 (0.000)	Data 0.002 (0.000)	
2021-06-25 05:52:57,332 Epoch: [21][627/1133]	Eit 24400  lr 5e-05  Le 9.7897 (9.7991)	Time 0.677 (0.000)	Data 0.002 (0.000)	
2021-06-25 05:55:24,043 Epoch: [21][827/1133]	Eit 24600  lr 5e-05  Le 9.8105 (9.7986)	Time 0.696 (0.000)	Data 0.002 (0.000)	
2021-06-25 05:57:49,521 Epoch: [21][1027/1133]	Eit 24800  lr 5e-05  Le 9.7886 (9.7984)	Time 0.735 (0.000)	Data 0.002 (0.000)	
2021-06-25 05:59:09,184 Test: [0/40]	Le 10.1821 (10.1821)	Time 3.316 (0.000)	
2021-06-25 05:59:21,719 calculate similarity time: 0.06439542770385742
2021-06-25 05:59:22,191 Image to text: 80.0, 96.3, 98.2, 1.0, 2.2
2021-06-25 05:59:22,512 Text to image: 61.8, 85.9, 91.0, 1.0, 7.2
2021-06-25 05:59:22,512 Current rsum is 513.14
2021-06-25 05:59:23,966 runs/f30k_butd_region_bert/log
2021-06-25 05:59:23,966 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-25 05:59:23,967 image encoder trainable parameters: 3688352
2021-06-25 05:59:23,971 txt encoder trainable parameters: 120517280
2021-06-25 06:00:39,024 Epoch: [22][95/1133]	Eit 25000  lr 5e-05  Le 9.8035 (9.7969)	Time 0.683 (0.000)	Data 0.002 (0.000)	
2021-06-25 06:03:05,885 Epoch: [22][295/1133]	Eit 25200  lr 5e-05  Le 9.8173 (9.7967)	Time 0.678 (0.000)	Data 0.002 (0.000)	
2021-06-25 06:05:34,942 Epoch: [22][495/1133]	Eit 25400  lr 5e-05  Le 9.7852 (9.7962)	Time 0.834 (0.000)	Data 0.002 (0.000)	
2021-06-25 06:08:02,508 Epoch: [22][695/1133]	Eit 25600  lr 5e-05  Le 9.8057 (9.7963)	Time 0.722 (0.000)	Data 0.002 (0.000)	
2021-06-25 06:10:30,289 Epoch: [22][895/1133]	Eit 25800  lr 5e-05  Le 9.7844 (9.7967)	Time 0.776 (0.000)	Data 0.002 (0.000)	
2021-06-25 06:12:57,447 Epoch: [22][1095/1133]	Eit 26000  lr 5e-05  Le 9.8079 (9.7970)	Time 0.700 (0.000)	Data 0.003 (0.000)	
2021-06-25 06:13:27,738 Test: [0/40]	Le 10.1821 (10.1821)	Time 3.928 (0.000)	
2021-06-25 06:13:40,022 calculate similarity time: 0.05146503448486328
2021-06-25 06:13:40,526 Image to text: 79.3, 96.3, 98.3, 1.0, 2.1
2021-06-25 06:13:40,883 Text to image: 61.9, 86.1, 91.2, 1.0, 7.3
2021-06-25 06:13:40,883 Current rsum is 513.1199999999999
2021-06-25 06:13:42,086 runs/f30k_butd_region_bert/log
2021-06-25 06:13:42,087 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-25 06:13:42,088 image encoder trainable parameters: 3688352
2021-06-25 06:13:42,093 txt encoder trainable parameters: 120517280
2021-06-25 06:15:45,491 Epoch: [23][163/1133]	Eit 26200  lr 5e-05  Le 9.8131 (9.7964)	Time 0.689 (0.000)	Data 0.002 (0.000)	
2021-06-25 06:18:12,759 Epoch: [23][363/1133]	Eit 26400  lr 5e-05  Le 9.7834 (9.7966)	Time 0.732 (0.000)	Data 0.003 (0.000)	
2021-06-25 06:20:40,864 Epoch: [23][563/1133]	Eit 26600  lr 5e-05  Le 9.7962 (9.7963)	Time 0.669 (0.000)	Data 0.002 (0.000)	
2021-06-25 06:23:08,054 Epoch: [23][763/1133]	Eit 26800  lr 5e-05  Le 9.8073 (9.7968)	Time 0.727 (0.000)	Data 0.002 (0.000)	
2021-06-25 06:25:35,582 Epoch: [23][963/1133]	Eit 27000  lr 5e-05  Le 9.7844 (9.7969)	Time 0.713 (0.000)	Data 0.002 (0.000)	
2021-06-25 06:27:42,155 Test: [0/40]	Le 10.1823 (10.1823)	Time 3.501 (0.000)	
2021-06-25 06:27:54,507 calculate similarity time: 0.06130051612854004
2021-06-25 06:27:55,008 Image to text: 79.7, 96.2, 98.3, 1.0, 2.2
2021-06-25 06:27:55,369 Text to image: 61.7, 86.0, 91.1, 1.0, 7.3
2021-06-25 06:27:55,369 Current rsum is 512.98
2021-06-25 06:27:56,640 runs/f30k_butd_region_bert/log
2021-06-25 06:27:56,641 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-25 06:27:56,641 image encoder trainable parameters: 3688352
2021-06-25 06:27:56,646 txt encoder trainable parameters: 120517280
2021-06-25 06:28:22,942 Epoch: [24][31/1133]	Eit 27200  lr 5e-05  Le 9.7984 (9.7922)	Time 0.730 (0.000)	Data 0.002 (0.000)	
2021-06-25 06:30:49,845 Epoch: [24][231/1133]	Eit 27400  lr 5e-05  Le 9.7903 (9.7944)	Time 0.762 (0.000)	Data 0.002 (0.000)	
2021-06-25 06:33:16,782 Epoch: [24][431/1133]	Eit 27600  lr 5e-05  Le 9.8106 (9.7949)	Time 0.818 (0.000)	Data 0.002 (0.000)	
2021-06-25 06:35:43,348 Epoch: [24][631/1133]	Eit 27800  lr 5e-05  Le 9.8084 (9.7953)	Time 0.669 (0.000)	Data 0.002 (0.000)	
2021-06-25 06:38:09,827 Epoch: [24][831/1133]	Eit 28000  lr 5e-05  Le 9.7864 (9.7951)	Time 0.725 (0.000)	Data 0.002 (0.000)	
2021-06-25 06:40:36,137 Epoch: [24][1031/1133]	Eit 28200  lr 5e-05  Le 9.7954 (9.7953)	Time 0.724 (0.000)	Data 0.002 (0.000)	
2021-06-25 06:41:52,988 Test: [0/40]	Le 10.1838 (10.1838)	Time 3.836 (0.000)	
2021-06-25 06:42:05,279 calculate similarity time: 0.04723644256591797
2021-06-25 06:42:05,780 Image to text: 80.4, 96.4, 98.2, 1.0, 2.1
2021-06-25 06:42:06,094 Text to image: 61.8, 85.7, 90.8, 1.0, 7.4
2021-06-25 06:42:06,094 Current rsum is 513.3
You have new mail in /var/spool/mail/root
[root@gpu1 vse_infty-master-my-graph-gru-unified-no-graph]# CUDA_VISIBLE_DEVICES=0,1 python3 eval.py --dataset f30k --data_path ../data/f30k
INFO:root:Evaluating runs/f30k_butd_region_bert...
INFO:lib.evaluation:Namespace(backbone_lr_factor=0.01, backbone_path='', backbone_source='detector', backbone_warmup_epochs=5, batch_size=128, data_name='f30k', data_path='../data/f30k', embed_size=1024, embedding_warmup_epochs=2, grad_clip=2.0, img_dim=2048, input_scale_factor=1, learning_rate=0.0005, log_step=200, logger_name='runs/f30k_butd_region_bert/log', lr_update=15, margin=0.2, max_violation=True, model_name='runs/f30k_butd_region_bert', no_imgnorm=False, no_txtnorm=False, num_epochs=25, optim='adam', precomp_enc_type='basic', reset_start_epoch=False, resume='', val_step=500, vocab_path='./vocab/', vocab_size=30522, vse_mean_warmup_epochs=1, word_dim=300, workers=5)
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
INFO:lib.evaluation:Test: [0/40]	Le 10.1991 (10.1991)	Time 5.053 (0.000)	
INFO:lib.evaluation:Test: [10/40]	Le 10.1922 (10.1935)	Time 0.343 (0.000)	
INFO:lib.evaluation:Test: [20/40]	Le 10.1938 (10.1942)	Time 0.302 (0.000)	
INFO:lib.evaluation:Test: [30/40]	Le 10.1994 (10.1958)	Time 0.412 (0.000)	
INFO:lib.evaluation:Images: 1000, Captions: 5000
INFO:lib.evaluation:calculate similarity time: 0.06474995613098145
INFO:lib.evaluation:rsum: 514.1
INFO:lib.evaluation:Average i2t Recall: 91.5
INFO:lib.evaluation:Image to text: 81.2 95.3 97.9 1.0 2.4
INFO:lib.evaluation:Average t2i Recall: 79.9
INFO:lib.evaluation:Text to image: 61.7 86.3 91.7 1.0 6.2
INFO:root:Evaluating runs/release_weights/f30k_butd_grid_bert...
Traceback (most recent call last):
  File "eval.py", line 58, in <module>
    main()
  File "eval.py", line 54, in main
    evaluation.evalrank(model_path, data_path=opt.data_path, split='test', fold5=False, save_path=save_path)
  File "/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-unified-no-graph/lib/evaluation.py", line 196, in evalrank
    checkpoint = torch.load(model_path)
  File "/root/anaconda3/lib/python3.6/site-packages/torch/serialization.py", line 571, in load
    with _open_file_like(f, 'rb') as opened_file:
  File "/root/anaconda3/lib/python3.6/site-packages/torch/serialization.py", line 229, in _open_file_like
    return _open_file(name_or_buffer, mode)
  File "/root/anaconda3/lib/python3.6/site-packages/torch/serialization.py", line 210, in __init__
    super(_open_file, self).__init__(open(name, mode))
FileNotFoundError: [Errno 2] No such file or directory: 'runs/release_weights/f30k_butd_grid_bert/model_best.pth'
[root@gpu1 vse_infty-master-my-graph-gru-unified-no-graph]# 

