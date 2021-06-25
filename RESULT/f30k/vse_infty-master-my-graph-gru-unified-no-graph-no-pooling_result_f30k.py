[root@gpu1 vse_infty-master-my-graph-gru-unified-no-graph-no-pooling]# sh train_region.sh 
2021-06-24 09:10:09,841 Namespace(backbone_lr_factor=0.01, backbone_path='', backbone_source='detector', backbone_warmup_epochs=5, batch_size=128, data_name='f30k', data_path='../data/f30k', embed_size=1024, embedding_warmup_epochs=2, grad_clip=2.0, img_dim=2048, input_scale_factor=1, learning_rate=0.0005, log_step=200, logger_name='runs/f30k_butd_region_bert/log', lr_update=15, margin=0.2, max_violation=False, model_name='runs/f30k_butd_region_bert', no_imgnorm=False, no_txtnorm=False, num_epochs=25, optim='adam', precomp_enc_type='basic', reset_start_epoch=False, resume='', val_step=500, vocab_path='./vocab/', vse_mean_warmup_epochs=1, word_dim=300, workers=10)
2021-06-24 09:10:09,841 Model name '/public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased' not found in model shortcut name list (bert-base-uncased, bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, bert-base-multilingual-cased, bert-base-chinese, bert-base-german-cased, bert-large-uncased-whole-word-masking, bert-large-cased-whole-word-masking, bert-large-uncased-whole-word-masking-finetuned-squad, bert-large-cased-whole-word-masking-finetuned-squad, bert-base-cased-finetuned-mrpc). Assuming '/public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased' is a path or url to a directory containing tokenizer files.
2021-06-24 09:10:09,841 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/added_tokens.json. We won't load it.
2021-06-24 09:10:09,841 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/special_tokens_map.json. We won't load it.
2021-06-24 09:10:09,841 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/tokenizer_config.json. We won't load it.
2021-06-24 09:10:09,841 loading file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/vocab.txt
2021-06-24 09:10:09,842 loading file None
2021-06-24 09:10:09,842 loading file None
2021-06-24 09:10:09,842 loading file None
2021-06-24 09:10:18,913 loading configuration file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/config.json
2021-06-24 09:10:18,913 Model config {
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

2021-06-24 09:10:18,914 loading weights file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/pytorch_model.bin
2021-06-24 09:10:27,216 Use adam as the optimizer, with init lr 0.0005
2021-06-24 09:10:27,217 Image encoder is data paralleled now.
2021-06-24 09:10:27,218 runs/f30k_butd_region_bert/log
2021-06-24 09:10:27,218 runs/f30k_butd_region_bert
2021-06-24 09:10:27,218 image encoder trainable parameters: 3688352
2021-06-24 09:10:27,224 txt encoder trainable parameters: 120517280
/root/anaconda3/lib/python3.6/site-packages/torch/nn/modules/rnn.py:735: UserWarning: RNN module weights are not part of single contiguous chunk of memory. This means they need to be compacted at every call, possibly greatly increasing memory usage. To compact weights again call flatten_parameters(). (Triggered internally at  /pytorch/aten/src/ATen/native/cudnn/RNN.cpp:1264.)
  self.dropout, self.training, self.bidirectional, self.batch_first)
2021-06-24 09:15:49,957 Epoch: [0][199/1133]	Eit 200  lr 0.0005  Le 10.1359 (10.1558)	Time 1.774 (0.000)	Data 0.003 (0.000)	
2021-06-24 09:21:06,593 Epoch: [0][399/1133]	Eit 400  lr 0.0005  Le 10.1242 (10.1432)	Time 1.716 (0.000)	Data 0.002 (0.000)	
2021-06-24 09:26:14,944 Epoch: [0][599/1133]	Eit 600  lr 0.0005  Le 10.1258 (10.1359)	Time 1.286 (0.000)	Data 0.002 (0.000)	
2021-06-24 09:31:05,846 Epoch: [0][799/1133]	Eit 800  lr 0.0005  Le 10.1091 (10.1297)	Time 1.420 (0.000)	Data 0.002 (0.000)	
2021-06-24 09:36:08,435 Epoch: [0][999/1133]	Eit 1000  lr 0.0005  Le 10.0855 (10.1230)	Time 1.701 (0.000)	Data 0.002 (0.000)	
2021-06-24 09:39:26,768 Test: [0/40]	Le 10.1718 (10.1718)	Time 3.913 (0.000)	
2021-06-24 09:39:49,167 calculate similarity time: 0.08240270614624023
2021-06-24 09:39:49,576 Image to text: 47.7, 74.3, 83.1, 2.0, 9.2
2021-06-24 09:39:49,886 Text to image: 34.0, 65.9, 76.0, 3.0, 15.7
2021-06-24 09:39:49,886 Current rsum is 381.06
2021-06-24 09:39:52,138 runs/f30k_butd_region_bert/log
2021-06-24 09:39:52,138 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 09:39:52,139 image encoder trainable parameters: 3688352
2021-06-24 09:39:52,143 txt encoder trainable parameters: 120517280
2021-06-24 09:41:37,129 Epoch: [1][67/1133]	Eit 1200  lr 0.0005  Le 10.0776 (10.0708)	Time 1.377 (0.000)	Data 0.001 (0.000)	
2021-06-24 09:46:34,841 Epoch: [1][267/1133]	Eit 1400  lr 0.0005  Le 10.0511 (10.0694)	Time 1.300 (0.000)	Data 0.002 (0.000)	
2021-06-24 09:51:21,529 Epoch: [1][467/1133]	Eit 1600  lr 0.0005  Le 10.0647 (10.0652)	Time 1.195 (0.000)	Data 0.001 (0.000)	
2021-06-24 09:56:28,029 Epoch: [1][667/1133]	Eit 1800  lr 0.0005  Le 10.0642 (10.0615)	Time 1.466 (0.000)	Data 0.002 (0.000)	
2021-06-24 10:01:27,481 Epoch: [1][867/1133]	Eit 2000  lr 0.0005  Le 10.0226 (10.0572)	Time 1.652 (0.000)	Data 0.002 (0.000)	
2021-06-24 10:06:26,075 Epoch: [1][1067/1133]	Eit 2200  lr 0.0005  Le 10.0579 (10.0535)	Time 1.574 (0.000)	Data 0.002 (0.000)	
2021-06-24 10:08:04,189 Test: [0/40]	Le 10.1846 (10.1846)	Time 3.706 (0.000)	
2021-06-24 10:08:27,117 calculate similarity time: 0.07876896858215332
2021-06-24 10:08:27,626 Image to text: 59.2, 84.8, 89.7, 1.0, 4.6
2021-06-24 10:08:27,940 Text to image: 43.7, 74.1, 82.3, 2.0, 10.4
2021-06-24 10:08:27,940 Current rsum is 433.84
2021-06-24 10:08:30,888 runs/f30k_butd_region_bert/log
2021-06-24 10:08:30,888 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 10:08:30,889 image encoder trainable parameters: 3688352
2021-06-24 10:08:30,896 txt encoder trainable parameters: 120517280
2021-06-24 10:11:44,584 Epoch: [2][135/1133]	Eit 2400  lr 0.0005  Le 10.0266 (10.0140)	Time 1.656 (0.000)	Data 0.002 (0.000)	
2021-06-24 10:16:50,523 Epoch: [2][335/1133]	Eit 2600  lr 0.0005  Le 9.9985 (10.0125)	Time 1.924 (0.000)	Data 0.002 (0.000)	
2021-06-24 10:21:54,026 Epoch: [2][535/1133]	Eit 2800  lr 0.0005  Le 10.0120 (10.0103)	Time 1.381 (0.000)	Data 0.002 (0.000)	
2021-06-24 10:26:55,562 Epoch: [2][735/1133]	Eit 3000  lr 0.0005  Le 10.0051 (10.0094)	Time 1.486 (0.000)	Data 0.002 (0.000)	
2021-06-24 10:31:52,211 Epoch: [2][935/1133]	Eit 3200  lr 0.0005  Le 9.9854 (10.0083)	Time 0.588 (0.000)	Data 0.002 (0.000)	
2021-06-24 10:36:42,469 Test: [0/40]	Le 10.1871 (10.1871)	Time 3.345 (0.000)	
2021-06-24 10:37:04,995 calculate similarity time: 0.0676114559173584
2021-06-24 10:37:05,471 Image to text: 63.1, 85.9, 93.6, 1.0, 3.8
2021-06-24 10:37:05,913 Text to image: 48.1, 76.6, 85.4, 2.0, 9.8
2021-06-24 10:37:05,913 Current rsum is 452.74
2021-06-24 10:37:08,894 runs/f30k_butd_region_bert/log
2021-06-24 10:37:08,895 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 10:37:08,897 image encoder trainable parameters: 3688352
2021-06-24 10:37:08,908 txt encoder trainable parameters: 120517280
2021-06-24 10:37:17,757 Epoch: [3][3/1133]	Eit 3400  lr 0.0005  Le 9.9747 (9.9724)	Time 1.324 (0.000)	Data 0.001 (0.000)	
2021-06-24 10:42:17,971 Epoch: [3][203/1133]	Eit 3600  lr 0.0005  Le 9.9998 (9.9775)	Time 1.427 (0.000)	Data 0.002 (0.000)	
2021-06-24 10:47:18,408 Epoch: [3][403/1133]	Eit 3800  lr 0.0005  Le 9.9779 (9.9775)	Time 1.268 (0.000)	Data 0.002 (0.000)	
2021-06-24 10:52:20,122 Epoch: [3][603/1133]	Eit 4000  lr 0.0005  Le 9.9797 (9.9769)	Time 1.565 (0.000)	Data 0.002 (0.000)	
2021-06-24 10:57:13,327 Epoch: [3][803/1133]	Eit 4200  lr 0.0005  Le 10.0005 (9.9768)	Time 1.660 (0.000)	Data 0.002 (0.000)	
2021-06-24 11:02:15,935 Epoch: [3][1003/1133]	Eit 4400  lr 0.0005  Le 9.9703 (9.9767)	Time 1.406 (0.000)	Data 0.002 (0.000)	
2021-06-24 11:05:33,854 Test: [0/40]	Le 10.1945 (10.1945)	Time 3.646 (0.000)	
2021-06-24 11:05:56,189 calculate similarity time: 0.07927751541137695
2021-06-24 11:05:56,587 Image to text: 64.6, 87.2, 93.6, 1.0, 3.6
2021-06-24 11:05:57,025 Text to image: 47.9, 78.1, 86.1, 2.0, 9.3
2021-06-24 11:05:57,025 Current rsum is 457.53999999999996
2021-06-24 11:06:00,052 runs/f30k_butd_region_bert/log
2021-06-24 11:06:00,052 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 11:06:00,053 image encoder trainable parameters: 3688352
2021-06-24 11:06:00,060 txt encoder trainable parameters: 120517280
2021-06-24 11:07:51,447 Epoch: [4][71/1133]	Eit 4600  lr 0.0005  Le 9.9004 (9.9518)	Time 1.452 (0.000)	Data 0.002 (0.000)	
2021-06-24 11:12:50,416 Epoch: [4][271/1133]	Eit 4800  lr 0.0005  Le 9.9699 (9.9523)	Time 1.562 (0.000)	Data 0.002 (0.000)	
2021-06-24 11:17:39,502 Epoch: [4][471/1133]	Eit 5000  lr 0.0005  Le 9.9101 (9.9529)	Time 1.424 (0.000)	Data 0.002 (0.000)	
2021-06-24 11:22:44,487 Epoch: [4][671/1133]	Eit 5200  lr 0.0005  Le 9.9709 (9.9534)	Time 1.695 (0.000)	Data 0.002 (0.000)	
2021-06-24 11:27:44,961 Epoch: [4][871/1133]	Eit 5400  lr 0.0005  Le 9.9487 (9.9527)	Time 1.689 (0.000)	Data 0.002 (0.000)	
2021-06-24 11:32:51,237 Epoch: [4][1071/1133]	Eit 5600  lr 0.0005  Le 9.9572 (9.9525)	Time 1.159 (0.000)	Data 0.002 (0.000)	
2021-06-24 11:34:24,183 Test: [0/40]	Le 10.1886 (10.1886)	Time 3.583 (0.000)	
2021-06-24 11:34:45,890 calculate similarity time: 0.061948537826538086
2021-06-24 11:34:46,279 Image to text: 65.1, 88.1, 93.5, 1.0, 3.6
2021-06-24 11:34:46,592 Text to image: 49.1, 78.7, 86.5, 2.0, 9.5
2021-06-24 11:34:46,592 Current rsum is 461.06
2021-06-24 11:34:49,552 runs/f30k_butd_region_bert/log
2021-06-24 11:34:49,552 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 11:34:49,554 image encoder trainable parameters: 3688352
2021-06-24 11:34:49,564 txt encoder trainable parameters: 120517280
2021-06-24 11:38:07,520 Epoch: [5][139/1133]	Eit 5800  lr 0.0005  Le 9.9331 (9.9338)	Time 1.375 (0.000)	Data 0.002 (0.000)	
2021-06-24 11:43:12,699 Epoch: [5][339/1133]	Eit 6000  lr 0.0005  Le 9.9408 (9.9346)	Time 1.283 (0.000)	Data 0.002 (0.000)	
2021-06-24 11:48:12,714 Epoch: [5][539/1133]	Eit 6200  lr 0.0005  Le 9.9744 (9.9354)	Time 1.766 (0.000)	Data 0.002 (0.000)	
2021-06-24 11:53:13,147 Epoch: [5][739/1133]	Eit 6400  lr 0.0005  Le 9.9314 (9.9361)	Time 1.245 (0.000)	Data 0.002 (0.000)	
2021-06-24 11:58:17,616 Epoch: [5][939/1133]	Eit 6600  lr 0.0005  Le 9.9410 (9.9359)	Time 1.454 (0.000)	Data 0.002 (0.000)	
2021-06-24 12:03:00,617 Test: [0/40]	Le 10.1835 (10.1835)	Time 3.993 (0.000)	
2021-06-24 12:03:22,640 calculate similarity time: 0.07957792282104492
2021-06-24 12:03:23,163 Image to text: 68.6, 88.7, 94.7, 1.0, 3.1
2021-06-24 12:03:23,477 Text to image: 50.4, 79.5, 86.4, 1.0, 9.6
2021-06-24 12:03:23,477 Current rsum is 468.32
2021-06-24 12:03:26,326 runs/f30k_butd_region_bert/log
2021-06-24 12:03:26,326 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 12:03:26,328 image encoder trainable parameters: 3688352
2021-06-24 12:03:26,338 txt encoder trainable parameters: 120517280
2021-06-24 12:03:40,977 Epoch: [6][7/1133]	Eit 6800  lr 0.0005  Le 9.9203 (9.9213)	Time 1.327 (0.000)	Data 0.002 (0.000)	
2021-06-24 12:08:40,373 Epoch: [6][207/1133]	Eit 7000  lr 0.0005  Le 9.9188 (9.9168)	Time 1.295 (0.000)	Data 0.002 (0.000)	
2021-06-24 12:13:41,336 Epoch: [6][407/1133]	Eit 7200  lr 0.0005  Le 9.9193 (9.9185)	Time 1.368 (0.000)	Data 0.005 (0.000)	
2021-06-24 12:18:41,119 Epoch: [6][607/1133]	Eit 7400  lr 0.0005  Le 9.9579 (9.9193)	Time 1.301 (0.000)	Data 0.002 (0.000)	
2021-06-24 12:23:31,316 Epoch: [6][807/1133]	Eit 7600  lr 0.0005  Le 9.9428 (9.9200)	Time 1.608 (0.000)	Data 0.002 (0.000)	
2021-06-24 12:28:34,405 Epoch: [6][1007/1133]	Eit 7800  lr 0.0005  Le 9.9156 (9.9202)	Time 1.454 (0.000)	Data 0.002 (0.000)	
2021-06-24 12:31:46,866 Test: [0/40]	Le 10.1880 (10.1880)	Time 3.678 (0.000)	
2021-06-24 12:32:09,469 calculate similarity time: 0.08169984817504883
2021-06-24 12:32:10,022 Image to text: 69.0, 90.3, 94.9, 1.0, 3.2
2021-06-24 12:32:10,365 Text to image: 50.9, 79.1, 86.6, 1.0, 9.4
2021-06-24 12:32:10,365 Current rsum is 470.78
2021-06-24 12:32:13,233 runs/f30k_butd_region_bert/log
2021-06-24 12:32:13,233 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 12:32:13,234 image encoder trainable parameters: 3688352
2021-06-24 12:32:13,241 txt encoder trainable parameters: 120517280
2021-06-24 12:34:09,718 Epoch: [7][75/1133]	Eit 8000  lr 0.0005  Le 9.8831 (9.9034)	Time 1.573 (0.000)	Data 0.002 (0.000)	
2021-06-24 12:39:16,675 Epoch: [7][275/1133]	Eit 8200  lr 0.0005  Le 9.8931 (9.9056)	Time 2.026 (0.000)	Data 0.002 (0.000)	
2021-06-24 12:44:05,813 Epoch: [7][475/1133]	Eit 8400  lr 0.0005  Le 9.9231 (9.9057)	Time 1.387 (0.000)	Data 0.002 (0.000)	
2021-06-24 12:49:06,057 Epoch: [7][675/1133]	Eit 8600  lr 0.0005  Le 9.9132 (9.9065)	Time 1.469 (0.000)	Data 0.002 (0.000)	
2021-06-24 12:54:09,613 Epoch: [7][875/1133]	Eit 8800  lr 0.0005  Le 9.9160 (9.9071)	Time 1.713 (0.000)	Data 0.002 (0.000)	
2021-06-24 12:59:09,403 Epoch: [7][1075/1133]	Eit 9000  lr 0.0005  Le 9.9043 (9.9076)	Time 1.545 (0.000)	Data 0.002 (0.000)	
2021-06-24 13:00:35,852 Test: [0/40]	Le 10.1863 (10.1863)	Time 3.794 (0.000)	
2021-06-24 13:00:58,631 calculate similarity time: 0.07528495788574219
2021-06-24 13:00:59,118 Image to text: 67.9, 88.7, 95.6, 1.0, 4.0
2021-06-24 13:00:59,431 Text to image: 50.7, 79.3, 86.2, 1.0, 9.7
2021-06-24 13:00:59,431 Current rsum is 468.44000000000005
2021-06-24 13:01:00,623 runs/f30k_butd_region_bert/log
2021-06-24 13:01:00,623 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 13:01:00,624 image encoder trainable parameters: 3688352
2021-06-24 13:01:00,631 txt encoder trainable parameters: 120517280
2021-06-24 13:04:29,677 Epoch: [8][143/1133]	Eit 9200  lr 0.0005  Le 9.8584 (9.8917)	Time 1.370 (0.000)	Data 0.002 (0.000)	
2021-06-24 13:09:34,121 Epoch: [8][343/1133]	Eit 9400  lr 0.0005  Le 9.8961 (9.8939)	Time 1.367 (0.000)	Data 0.002 (0.000)	
2021-06-24 13:14:36,504 Epoch: [8][543/1133]	Eit 9600  lr 0.0005  Le 9.8790 (9.8941)	Time 2.144 (0.000)	Data 0.002 (0.000)	
2021-06-24 13:19:40,872 Epoch: [8][743/1133]	Eit 9800  lr 0.0005  Le 9.8719 (9.8946)	Time 1.539 (0.000)	Data 0.004 (0.000)	
2021-06-24 13:24:47,277 Epoch: [8][943/1133]	Eit 10000  lr 0.0005  Le 9.9268 (9.8956)	Time 1.426 (0.000)	Data 0.002 (0.000)	
2021-06-24 13:29:27,162 Test: [0/40]	Le 10.1912 (10.1912)	Time 3.682 (0.000)	
2021-06-24 13:29:49,407 calculate similarity time: 0.0652153491973877
2021-06-24 13:29:49,833 Image to text: 65.8, 89.0, 95.4, 1.0, 3.8
2021-06-24 13:29:50,144 Text to image: 50.5, 78.9, 86.7, 1.0, 10.3
2021-06-24 13:29:50,145 Current rsum is 466.26
2021-06-24 13:29:51,300 runs/f30k_butd_region_bert/log
2021-06-24 13:29:51,300 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 13:29:51,301 image encoder trainable parameters: 3688352
2021-06-24 13:29:51,305 txt encoder trainable parameters: 120517280
2021-06-24 13:30:11,110 Epoch: [9][11/1133]	Eit 10200  lr 0.0005  Le 9.9078 (9.8838)	Time 1.685 (0.000)	Data 0.002 (0.000)	
2021-06-24 13:35:14,283 Epoch: [9][211/1133]	Eit 10400  lr 0.0005  Le 9.8807 (9.8824)	Time 1.262 (0.000)	Data 0.002 (0.000)	
2021-06-24 13:40:14,560 Epoch: [9][411/1133]	Eit 10600  lr 0.0005  Le 9.8810 (9.8830)	Time 1.608 (0.000)	Data 0.002 (0.000)	
2021-06-24 13:45:15,053 Epoch: [9][611/1133]	Eit 10800  lr 0.0005  Le 9.8938 (9.8837)	Time 1.321 (0.000)	Data 0.002 (0.000)	
2021-06-24 13:50:09,091 Epoch: [9][811/1133]	Eit 11000  lr 0.0005  Le 9.8820 (9.8847)	Time 1.311 (0.000)	Data 0.002 (0.000)	
2021-06-24 13:55:08,999 Epoch: [9][1011/1133]	Eit 11200  lr 0.0005  Le 9.8662 (9.8853)	Time 1.776 (0.000)	Data 0.002 (0.000)	
2021-06-24 13:58:11,301 Test: [0/40]	Le 10.1880 (10.1880)	Time 3.453 (0.000)	
2021-06-24 13:58:33,929 calculate similarity time: 0.07770085334777832
2021-06-24 13:58:34,486 Image to text: 66.5, 89.3, 93.2, 1.0, 4.0
2021-06-24 13:58:34,882 Text to image: 51.0, 78.9, 85.7, 1.0, 10.3
2021-06-24 13:58:34,882 Current rsum is 464.59999999999997
2021-06-24 13:58:36,056 runs/f30k_butd_region_bert/log
2021-06-24 13:58:36,056 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 13:58:36,057 image encoder trainable parameters: 3688352
2021-06-24 13:58:36,061 txt encoder trainable parameters: 120517280
2021-06-24 14:00:40,360 Epoch: [10][79/1133]	Eit 11400  lr 0.0005  Le 9.8645 (9.8748)	Time 1.698 (0.000)	Data 0.002 (0.000)	
2021-06-24 14:05:46,196 Epoch: [10][279/1133]	Eit 11600  lr 0.0005  Le 9.8840 (9.8750)	Time 1.459 (0.000)	Data 0.002 (0.000)	
2021-06-24 14:10:36,198 Epoch: [10][479/1133]	Eit 11800  lr 0.0005  Le 9.8942 (9.8762)	Time 1.641 (0.000)	Data 0.002 (0.000)	
2021-06-24 14:15:39,592 Epoch: [10][679/1133]	Eit 12000  lr 0.0005  Le 9.8731 (9.8767)	Time 1.438 (0.000)	Data 0.002 (0.000)	
2021-06-24 14:20:43,178 Epoch: [10][879/1133]	Eit 12200  lr 0.0005  Le 9.8841 (9.8765)	Time 1.492 (0.000)	Data 0.002 (0.000)	
2021-06-24 14:25:46,405 Epoch: [10][1079/1133]	Eit 12400  lr 0.0005  Le 9.8644 (9.8771)	Time 1.802 (0.000)	Data 0.002 (0.000)	
2021-06-24 14:27:07,415 Test: [0/40]	Le 10.1915 (10.1915)	Time 3.658 (0.000)	
2021-06-24 14:27:29,606 calculate similarity time: 0.09554409980773926
2021-06-24 14:27:30,103 Image to text: 67.4, 90.5, 94.2, 1.0, 3.3
2021-06-24 14:27:30,438 Text to image: 50.9, 79.2, 86.5, 1.0, 10.1
2021-06-24 14:27:30,438 Current rsum is 468.74
2021-06-24 14:27:31,606 runs/f30k_butd_region_bert/log
2021-06-24 14:27:31,607 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 14:27:31,607 image encoder trainable parameters: 3688352
2021-06-24 14:27:31,612 txt encoder trainable parameters: 120517280
2021-06-24 14:31:03,449 Epoch: [11][147/1133]	Eit 12600  lr 0.0005  Le 9.8710 (9.8653)	Time 1.796 (0.000)	Data 0.002 (0.000)	
2021-06-24 14:36:08,645 Epoch: [11][347/1133]	Eit 12800  lr 0.0005  Le 9.8772 (9.8665)	Time 1.709 (0.000)	Data 0.002 (0.000)	
2021-06-24 14:41:11,604 Epoch: [11][547/1133]	Eit 13000  lr 0.0005  Le 9.8578 (9.8676)	Time 1.603 (0.000)	Data 0.002 (0.000)	
2021-06-24 14:46:09,494 Epoch: [11][747/1133]	Eit 13200  lr 0.0005  Le 9.8673 (9.8682)	Time 1.306 (0.000)	Data 0.002 (0.000)	
2021-06-24 14:51:12,154 Epoch: [11][947/1133]	Eit 13400  lr 0.0005  Le 9.8666 (9.8684)	Time 1.378 (0.000)	Data 0.002 (0.000)	
2021-06-24 14:55:41,923 Test: [0/40]	Le 10.1902 (10.1902)	Time 3.492 (0.000)	
2021-06-24 14:56:03,607 calculate similarity time: 0.08888459205627441
2021-06-24 14:56:04,067 Image to text: 67.1, 88.2, 94.7, 1.0, 3.8
2021-06-24 14:56:04,378 Text to image: 50.2, 78.8, 86.1, 1.0, 10.4
2021-06-24 14:56:04,378 Current rsum is 465.12
2021-06-24 14:56:05,539 runs/f30k_butd_region_bert/log
2021-06-24 14:56:05,539 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 14:56:05,540 image encoder trainable parameters: 3688352
2021-06-24 14:56:05,544 txt encoder trainable parameters: 120517280
2021-06-24 14:56:32,395 Epoch: [12][15/1133]	Eit 13600  lr 0.0005  Le 9.8906 (9.8576)	Time 1.286 (0.000)	Data 0.002 (0.000)	
2021-06-24 15:01:34,677 Epoch: [12][215/1133]	Eit 13800  lr 0.0005  Le 9.8531 (9.8578)	Time 1.384 (0.000)	Data 0.002 (0.000)	
2021-06-24 15:06:41,285 Epoch: [12][415/1133]	Eit 14000  lr 0.0005  Le 9.8464 (9.8593)	Time 1.284 (0.000)	Data 0.002 (0.000)	
2021-06-24 15:11:40,849 Epoch: [12][615/1133]	Eit 14200  lr 0.0005  Le 9.8539 (9.8607)	Time 1.462 (0.000)	Data 0.002 (0.000)	
2021-06-24 15:16:34,068 Epoch: [12][815/1133]	Eit 14400  lr 0.0005  Le 9.8530 (9.8611)	Time 1.468 (0.000)	Data 0.001 (0.000)	
2021-06-24 15:21:35,018 Epoch: [12][1015/1133]	Eit 14600  lr 0.0005  Le 9.8702 (9.8617)	Time 1.780 (0.000)	Data 0.002 (0.000)	
2021-06-24 15:24:32,171 Test: [0/40]	Le 10.1876 (10.1876)	Time 3.427 (0.000)	
2021-06-24 15:24:53,990 calculate similarity time: 0.07210350036621094
2021-06-24 15:24:54,479 Image to text: 66.7, 88.3, 94.5, 1.0, 3.5
2021-06-24 15:24:54,791 Text to image: 50.3, 78.6, 85.8, 1.0, 10.5
2021-06-24 15:24:54,792 Current rsum is 464.18000000000006
2021-06-24 15:24:55,970 runs/f30k_butd_region_bert/log
2021-06-24 15:24:55,970 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 15:24:55,971 image encoder trainable parameters: 3688352
2021-06-24 15:24:55,975 txt encoder trainable parameters: 120517280
2021-06-24 15:27:05,446 Epoch: [13][83/1133]	Eit 14800  lr 0.0005  Le 9.8673 (9.8536)	Time 1.316 (0.000)	Data 0.002 (0.000)	
2021-06-24 15:32:03,173 Epoch: [13][283/1133]	Eit 15000  lr 0.0005  Le 9.8406 (9.8526)	Time 1.399 (0.000)	Data 0.002 (0.000)	
2021-06-24 15:36:57,516 Epoch: [13][483/1133]	Eit 15200  lr 0.0005  Le 9.8298 (9.8529)	Time 1.419 (0.000)	Data 0.002 (0.000)	
2021-06-24 15:42:02,064 Epoch: [13][683/1133]	Eit 15400  lr 0.0005  Le 9.8616 (9.8536)	Time 1.313 (0.000)	Data 0.002 (0.000)	
2021-06-24 15:47:02,545 Epoch: [13][883/1133]	Eit 15600  lr 0.0005  Le 9.8392 (9.8541)	Time 1.349 (0.000)	Data 0.002 (0.000)	
2021-06-24 15:52:04,981 Epoch: [13][1083/1133]	Eit 15800  lr 0.0005  Le 9.8822 (9.8551)	Time 1.537 (0.000)	Data 0.002 (0.000)	
2021-06-24 15:53:21,205 Test: [0/40]	Le 10.1911 (10.1911)	Time 3.639 (0.000)	
2021-06-24 15:53:42,690 calculate similarity time: 0.056601762771606445
2021-06-24 15:53:43,085 Image to text: 66.3, 88.6, 93.5, 1.0, 3.7
2021-06-24 15:53:43,398 Text to image: 50.2, 78.4, 86.0, 1.0, 10.2
2021-06-24 15:53:43,398 Current rsum is 462.9599999999999
2021-06-24 15:53:44,580 runs/f30k_butd_region_bert/log
2021-06-24 15:53:44,580 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 15:53:44,581 image encoder trainable parameters: 3688352
2021-06-24 15:53:44,585 txt encoder trainable parameters: 120517280
2021-06-24 15:57:23,207 Epoch: [14][151/1133]	Eit 16000  lr 0.0005  Le 9.8349 (9.8438)	Time 1.319 (0.000)	Data 0.001 (0.000)	
2021-06-24 16:02:24,966 Epoch: [14][351/1133]	Eit 16200  lr 0.0005  Le 9.8358 (9.8459)	Time 1.649 (0.000)	Data 0.002 (0.000)	
2021-06-24 16:07:28,683 Epoch: [14][551/1133]	Eit 16400  lr 0.0005  Le 9.8380 (9.8473)	Time 1.262 (0.000)	Data 0.002 (0.000)	
2021-06-24 16:12:30,502 Epoch: [14][751/1133]	Eit 16600  lr 0.0005  Le 9.8442 (9.8476)	Time 1.678 (0.000)	Data 0.001 (0.000)	
2021-06-24 16:17:30,934 Epoch: [14][951/1133]	Eit 16800  lr 0.0005  Le 9.8598 (9.8479)	Time 1.337 (0.000)	Data 0.011 (0.000)	
2021-06-24 16:21:57,326 Test: [0/40]	Le 10.1973 (10.1973)	Time 3.738 (0.000)	
2021-06-24 16:22:19,331 calculate similarity time: 0.07146120071411133
2021-06-24 16:22:19,830 Image to text: 67.6, 89.0, 94.2, 1.0, 3.6
2021-06-24 16:22:20,152 Text to image: 50.5, 78.1, 86.1, 1.0, 10.1
2021-06-24 16:22:20,152 Current rsum is 465.47999999999996
2021-06-24 16:22:21,349 runs/f30k_butd_region_bert/log
2021-06-24 16:22:21,349 runs/f30k_butd_region_bert
2021-06-24 16:22:21,349 Current epoch num is 15, decrease all lr by 10
2021-06-24 16:22:21,349 new lr 5e-05
2021-06-24 16:22:21,349 new lr 5e-06
2021-06-24 16:22:21,350 new lr 5e-05
Use VSE++ objective.
2021-06-24 16:22:21,350 image encoder trainable parameters: 3688352
2021-06-24 16:22:21,355 txt encoder trainable parameters: 120517280
2021-06-24 16:22:54,328 Epoch: [15][19/1133]	Eit 17000  lr 5e-05  Le 9.8619 (9.8428)	Time 1.693 (0.000)	Data 0.002 (0.000)	
2021-06-24 16:27:58,074 Epoch: [15][219/1133]	Eit 17200  lr 5e-05  Le 9.8155 (9.8324)	Time 1.250 (0.000)	Data 0.002 (0.000)	
2021-06-24 16:33:02,364 Epoch: [15][419/1133]	Eit 17400  lr 5e-05  Le 9.8445 (9.8303)	Time 1.909 (0.000)	Data 0.002 (0.000)	
2021-06-24 16:38:03,511 Epoch: [15][619/1133]	Eit 17600  lr 5e-05  Le 9.8439 (9.8294)	Time 1.302 (0.000)	Data 0.002 (0.000)	
2021-06-24 16:42:53,826 Epoch: [15][819/1133]	Eit 17800  lr 5e-05  Le 9.8312 (9.8286)	Time 1.794 (0.000)	Data 0.002 (0.000)	
2021-06-24 16:47:54,610 Epoch: [15][1019/1133]	Eit 18000  lr 5e-05  Le 9.8199 (9.8281)	Time 1.463 (0.000)	Data 0.002 (0.000)	
2021-06-24 16:50:49,728 Test: [0/40]	Le 10.1951 (10.1951)	Time 3.600 (0.000)	
2021-06-24 16:51:11,548 calculate similarity time: 0.06847715377807617
2021-06-24 16:51:12,046 Image to text: 68.8, 90.2, 95.0, 1.0, 3.2
2021-06-24 16:51:12,355 Text to image: 52.4, 79.8, 87.0, 1.0, 9.7
2021-06-24 16:51:12,355 Current rsum is 473.2
2021-06-24 16:51:15,307 runs/f30k_butd_region_bert/log
2021-06-24 16:51:15,307 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 16:51:15,309 image encoder trainable parameters: 3688352
2021-06-24 16:51:15,320 txt encoder trainable parameters: 120517280
2021-06-24 16:53:29,160 Epoch: [16][87/1133]	Eit 18200  lr 5e-05  Le 9.8274 (9.8210)	Time 1.473 (0.000)	Data 0.002 (0.000)	
2021-06-24 16:58:30,268 Epoch: [16][287/1133]	Eit 18400  lr 5e-05  Le 9.8177 (9.8202)	Time 1.503 (0.000)	Data 0.002 (0.000)	
2021-06-24 17:03:19,964 Epoch: [16][487/1133]	Eit 18600  lr 5e-05  Le 9.8425 (9.8200)	Time 1.276 (0.000)	Data 0.002 (0.000)	
2021-06-24 17:08:22,595 Epoch: [16][687/1133]	Eit 18800  lr 5e-05  Le 9.8251 (9.8203)	Time 1.350 (0.000)	Data 0.002 (0.000)	
2021-06-24 17:13:27,729 Epoch: [16][887/1133]	Eit 19000  lr 5e-05  Le 9.8209 (9.8202)	Time 1.683 (0.000)	Data 0.002 (0.000)	
2021-06-24 17:18:29,478 Epoch: [16][1087/1133]	Eit 19200  lr 5e-05  Le 9.8095 (9.8201)	Time 1.389 (0.000)	Data 0.002 (0.000)	
2021-06-24 17:19:39,349 Test: [0/40]	Le 10.1962 (10.1961)	Time 3.573 (0.000)	
2021-06-24 17:20:01,634 calculate similarity time: 0.0709683895111084
2021-06-24 17:20:02,207 Image to text: 68.7, 90.2, 94.9, 1.0, 3.3
2021-06-24 17:20:02,532 Text to image: 51.9, 79.2, 86.8, 1.0, 9.7
2021-06-24 17:20:02,532 Current rsum is 471.72
2021-06-24 17:20:03,757 runs/f30k_butd_region_bert/log
2021-06-24 17:20:03,758 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 17:20:03,758 image encoder trainable parameters: 3688352
2021-06-24 17:20:03,763 txt encoder trainable parameters: 120517280
2021-06-24 17:23:50,054 Epoch: [17][155/1133]	Eit 19400  lr 5e-05  Le 9.8240 (9.8178)	Time 1.594 (0.000)	Data 0.002 (0.000)	
2021-06-24 17:28:53,016 Epoch: [17][355/1133]	Eit 19600  lr 5e-05  Le 9.8324 (9.8168)	Time 1.358 (0.000)	Data 0.002 (0.000)	
2021-06-24 17:34:01,764 Epoch: [17][555/1133]	Eit 19800  lr 5e-05  Le 9.8276 (9.8162)	Time 1.780 (0.000)	Data 0.002 (0.000)	
2021-06-24 17:39:05,724 Epoch: [17][755/1133]	Eit 20000  lr 5e-05  Le 9.8060 (9.8160)	Time 1.446 (0.000)	Data 0.002 (0.000)	
2021-06-24 17:44:09,241 Epoch: [17][955/1133]	Eit 20200  lr 5e-05  Le 9.8128 (9.8160)	Time 1.692 (0.000)	Data 0.002 (0.000)	
2021-06-24 17:48:30,324 Test: [0/40]	Le 10.1932 (10.1932)	Time 3.358 (0.000)	
2021-06-24 17:48:52,696 calculate similarity time: 0.0796349048614502
2021-06-24 17:48:53,198 Image to text: 69.2, 89.9, 95.5, 1.0, 3.1
2021-06-24 17:48:53,511 Text to image: 51.8, 78.9, 86.7, 1.0, 9.9
2021-06-24 17:48:53,511 Current rsum is 472.02000000000004
2021-06-24 17:48:54,718 runs/f30k_butd_region_bert/log
2021-06-24 17:48:54,718 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 17:48:54,719 image encoder trainable parameters: 3688352
2021-06-24 17:48:54,724 txt encoder trainable parameters: 120517280
2021-06-24 17:49:32,691 Epoch: [18][23/1133]	Eit 20400  lr 5e-05  Le 9.8491 (9.8207)	Time 1.381 (0.000)	Data 0.002 (0.000)	
2021-06-24 17:54:37,269 Epoch: [18][223/1133]	Eit 20600  lr 5e-05  Le 9.8000 (9.8165)	Time 1.838 (0.000)	Data 0.002 (0.000)	
2021-06-24 17:59:38,538 Epoch: [18][423/1133]	Eit 20800  lr 5e-05  Le 9.8081 (9.8152)	Time 1.815 (0.000)	Data 0.002 (0.000)	
2021-06-24 18:04:41,872 Epoch: [18][623/1133]	Eit 21000  lr 5e-05  Le 9.8176 (9.8151)	Time 1.439 (0.000)	Data 0.002 (0.000)	
2021-06-24 18:08:04,110 Epoch: [18][823/1133]	Eit 21200  lr 5e-05  Le 9.8224 (9.8152)	Time 0.663 (0.000)	Data 0.002 (0.000)	
2021-06-24 18:10:24,828 Epoch: [18][1023/1133]	Eit 21400  lr 5e-05  Le 9.8008 (9.8150)	Time 0.642 (0.000)	Data 0.002 (0.000)	
2021-06-24 18:11:43,986 Test: [0/40]	Le 10.1953 (10.1953)	Time 3.249 (0.000)	
2021-06-24 18:11:54,732 calculate similarity time: 0.05164837837219238
2021-06-24 18:11:55,235 Image to text: 68.8, 90.7, 94.9, 1.0, 3.2
2021-06-24 18:11:55,597 Text to image: 52.0, 78.9, 86.1, 1.0, 10.0
2021-06-24 18:11:55,597 Current rsum is 471.43999999999994
2021-06-24 18:11:56,532 runs/f30k_butd_region_bert/log
2021-06-24 18:11:56,532 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 18:11:56,533 image encoder trainable parameters: 3688352
2021-06-24 18:11:56,537 txt encoder trainable parameters: 120517280
2021-06-24 18:13:02,502 Epoch: [19][91/1133]	Eit 21600  lr 5e-05  Le 9.8041 (9.8123)	Time 0.669 (0.000)	Data 0.002 (0.000)	
2021-06-24 18:15:19,607 Epoch: [19][291/1133]	Eit 21800  lr 5e-05  Le 9.8123 (9.8120)	Time 0.641 (0.000)	Data 0.002 (0.000)	
2021-06-24 18:17:35,415 Epoch: [19][491/1133]	Eit 22000  lr 5e-05  Le 9.7901 (9.8116)	Time 0.612 (0.000)	Data 0.002 (0.000)	
2021-06-24 18:19:50,242 Epoch: [19][691/1133]	Eit 22200  lr 5e-05  Le 9.8253 (9.8119)	Time 0.653 (0.000)	Data 0.002 (0.000)	
2021-06-24 18:22:05,027 Epoch: [19][891/1133]	Eit 22400  lr 5e-05  Le 9.7949 (9.8121)	Time 0.650 (0.000)	Data 0.002 (0.000)	
2021-06-24 18:24:20,269 Epoch: [19][1091/1133]	Eit 22600  lr 5e-05  Le 9.8087 (9.8122)	Time 0.571 (0.000)	Data 0.002 (0.000)	
2021-06-24 18:24:51,048 Test: [0/40]	Le 10.1947 (10.1947)	Time 3.711 (0.000)	
2021-06-24 18:25:01,694 calculate similarity time: 0.04859423637390137
2021-06-24 18:25:02,197 Image to text: 69.5, 89.5, 94.7, 1.0, 3.1
2021-06-24 18:25:02,515 Text to image: 51.8, 78.5, 86.2, 1.0, 10.0
2021-06-24 18:25:02,515 Current rsum is 470.20000000000005
2021-06-24 18:25:03,615 runs/f30k_butd_region_bert/log
2021-06-24 18:25:03,615 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 18:25:03,616 image encoder trainable parameters: 3688352
2021-06-24 18:25:03,620 txt encoder trainable parameters: 120517280
2021-06-24 18:26:54,385 Epoch: [20][159/1133]	Eit 22800  lr 5e-05  Le 9.8099 (9.8073)	Time 0.599 (0.000)	Data 0.002 (0.000)	
2021-06-24 18:29:07,243 Epoch: [20][359/1133]	Eit 23000  lr 5e-05  Le 9.8108 (9.8093)	Time 0.626 (0.000)	Data 0.002 (0.000)	
2021-06-24 18:31:19,980 Epoch: [20][559/1133]	Eit 23200  lr 5e-05  Le 9.8189 (9.8095)	Time 0.636 (0.000)	Data 0.002 (0.000)	
2021-06-24 18:33:33,070 Epoch: [20][759/1133]	Eit 23400  lr 5e-05  Le 9.8053 (9.8095)	Time 0.746 (0.000)	Data 0.002 (0.000)	
2021-06-24 18:35:46,113 Epoch: [20][959/1133]	Eit 23600  lr 5e-05  Le 9.8015 (9.8098)	Time 0.613 (0.000)	Data 0.002 (0.000)	
2021-06-24 18:37:45,570 Test: [0/40]	Le 10.1929 (10.1929)	Time 3.911 (0.000)	
2021-06-24 18:37:56,198 calculate similarity time: 0.058385372161865234
2021-06-24 18:37:56,700 Image to text: 69.4, 89.3, 94.5, 1.0, 3.1
2021-06-24 18:37:57,073 Text to image: 51.8, 78.7, 86.1, 1.0, 10.2
2021-06-24 18:37:57,073 Current rsum is 469.84000000000003
2021-06-24 18:37:58,315 runs/f30k_butd_region_bert/log
2021-06-24 18:37:58,315 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 18:37:58,316 image encoder trainable parameters: 3688352
2021-06-24 18:37:58,320 txt encoder trainable parameters: 120517280
2021-06-24 18:38:20,482 Epoch: [21][27/1133]	Eit 23800  lr 5e-05  Le 9.8464 (9.8145)	Time 0.661 (0.000)	Data 0.002 (0.000)	
2021-06-24 18:40:37,483 Epoch: [21][227/1133]	Eit 24000  lr 5e-05  Le 9.8220 (9.8097)	Time 0.723 (0.000)	Data 0.002 (0.000)	
2021-06-24 18:42:54,376 Epoch: [21][427/1133]	Eit 24200  lr 5e-05  Le 9.8131 (9.8097)	Time 0.673 (0.000)	Data 0.002 (0.000)	
2021-06-24 18:45:11,526 Epoch: [21][627/1133]	Eit 24400  lr 5e-05  Le 9.8231 (9.8101)	Time 0.650 (0.000)	Data 0.002 (0.000)	
2021-06-24 18:47:26,770 Epoch: [21][827/1133]	Eit 24600  lr 5e-05  Le 9.8029 (9.8097)	Time 0.685 (0.000)	Data 0.002 (0.000)	
2021-06-24 18:49:38,964 Epoch: [21][1027/1133]	Eit 24800  lr 5e-05  Le 9.8032 (9.8094)	Time 0.637 (0.000)	Data 0.002 (0.000)	
2021-06-24 18:50:51,638 Test: [0/40]	Le 10.1966 (10.1966)	Time 3.806 (0.000)	
2021-06-24 18:51:02,486 calculate similarity time: 0.06209921836853027
2021-06-24 18:51:02,960 Image to text: 69.4, 90.1, 94.9, 1.0, 3.1
2021-06-24 18:51:03,286 Text to image: 52.3, 78.9, 85.9, 1.0, 10.1
2021-06-24 18:51:03,286 Current rsum is 471.53999999999996
2021-06-24 18:51:04,601 runs/f30k_butd_region_bert/log
2021-06-24 18:51:04,601 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 18:51:04,602 image encoder trainable parameters: 3688352
2021-06-24 18:51:04,609 txt encoder trainable parameters: 120517280
2021-06-24 18:52:13,318 Epoch: [22][95/1133]	Eit 25000  lr 5e-05  Le 9.7840 (9.8079)	Time 0.688 (0.000)	Data 0.002 (0.000)	
2021-06-24 18:54:28,300 Epoch: [22][295/1133]	Eit 25200  lr 5e-05  Le 9.7985 (9.8069)	Time 0.658 (0.000)	Data 0.002 (0.000)	
2021-06-24 18:56:39,879 Epoch: [22][495/1133]	Eit 25400  lr 5e-05  Le 9.8302 (9.8073)	Time 0.664 (0.000)	Data 0.002 (0.000)	
2021-06-24 18:58:54,780 Epoch: [22][695/1133]	Eit 25600  lr 5e-05  Le 9.8029 (9.8072)	Time 0.707 (0.000)	Data 0.002 (0.000)	
2021-06-24 19:01:09,140 Epoch: [22][895/1133]	Eit 25800  lr 5e-05  Le 9.8317 (9.8072)	Time 0.638 (0.000)	Data 0.004 (0.000)	
2021-06-24 19:03:22,507 Epoch: [22][1095/1133]	Eit 26000  lr 5e-05  Le 9.8073 (9.8075)	Time 0.596 (0.000)	Data 0.002 (0.000)	
2021-06-24 19:03:51,822 Test: [0/40]	Le 10.1932 (10.1932)	Time 3.819 (0.000)	
2021-06-24 19:04:02,584 calculate similarity time: 0.06382393836975098
2021-06-24 19:04:03,100 Image to text: 69.4, 89.2, 94.8, 1.0, 3.3
2021-06-24 19:04:03,464 Text to image: 52.1, 78.7, 86.4, 1.0, 10.4
2021-06-24 19:04:03,464 Current rsum is 470.65999999999997
2021-06-24 19:04:04,730 runs/f30k_butd_region_bert/log
2021-06-24 19:04:04,730 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 19:04:04,730 image encoder trainable parameters: 3688352
2021-06-24 19:04:04,735 txt encoder trainable parameters: 120517280
2021-06-24 19:05:58,321 Epoch: [23][163/1133]	Eit 26200  lr 5e-05  Le 9.8061 (9.8077)	Time 0.650 (0.000)	Data 0.002 (0.000)	
2021-06-24 19:08:10,584 Epoch: [23][363/1133]	Eit 26400  lr 5e-05  Le 9.8018 (9.8074)	Time 0.653 (0.000)	Data 0.002 (0.000)	
2021-06-24 19:10:23,594 Epoch: [23][563/1133]	Eit 26600  lr 5e-05  Le 9.8067 (9.8073)	Time 0.619 (0.000)	Data 0.002 (0.000)	
2021-06-24 19:12:36,009 Epoch: [23][763/1133]	Eit 26800  lr 5e-05  Le 9.8034 (9.8071)	Time 0.624 (0.000)	Data 0.002 (0.000)	
2021-06-24 19:14:49,961 Epoch: [23][963/1133]	Eit 27000  lr 5e-05  Le 9.8030 (9.8070)	Time 0.608 (0.000)	Data 0.002 (0.000)	
2021-06-24 19:16:44,769 Test: [0/40]	Le 10.1953 (10.1953)	Time 3.829 (0.000)	
2021-06-24 19:16:56,071 calculate similarity time: 0.04910540580749512
2021-06-24 19:16:56,575 Image to text: 69.2, 89.8, 95.1, 1.0, 3.0
2021-06-24 19:16:56,904 Text to image: 51.8, 78.5, 85.8, 1.0, 10.5
2021-06-24 19:16:56,904 Current rsum is 470.26
2021-06-24 19:16:58,310 runs/f30k_butd_region_bert/log
2021-06-24 19:16:58,311 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-24 19:16:58,311 image encoder trainable parameters: 3688352
2021-06-24 19:16:58,316 txt encoder trainable parameters: 120517280
2021-06-24 19:17:23,712 Epoch: [24][31/1133]	Eit 27200  lr 5e-05  Le 9.8140 (9.8046)	Time 0.792 (0.000)	Data 0.002 (0.000)	
2021-06-24 19:19:41,007 Epoch: [24][231/1133]	Eit 27400  lr 5e-05  Le 9.8230 (9.8051)	Time 0.654 (0.000)	Data 0.002 (0.000)	
2021-06-24 19:22:00,217 Epoch: [24][431/1133]	Eit 27600  lr 5e-05  Le 9.7989 (9.8050)	Time 0.625 (0.000)	Data 0.002 (0.000)	
2021-06-24 19:24:18,445 Epoch: [24][631/1133]	Eit 27800  lr 5e-05  Le 9.7996 (9.8053)	Time 0.666 (0.000)	Data 0.002 (0.000)	
2021-06-24 19:26:32,608 Epoch: [24][831/1133]	Eit 28000  lr 5e-05  Le 9.8174 (9.8057)	Time 0.655 (0.000)	Data 0.003 (0.000)	
2021-06-24 19:28:47,499 Epoch: [24][1031/1133]	Eit 28200  lr 5e-05  Le 9.7995 (9.8057)	Time 0.775 (0.000)	Data 0.002 (0.000)	
2021-06-24 19:29:58,403 Test: [0/40]	Le 10.1946 (10.1946)	Time 3.768 (0.000)	
2021-06-24 19:30:09,006 calculate similarity time: 0.049706220626831055
2021-06-24 19:30:09,547 Image to text: 69.0, 89.3, 95.4, 1.0, 3.1
2021-06-24 19:30:09,997 Text to image: 51.9, 78.6, 86.0, 1.0, 10.3
2021-06-24 19:30:09,998 Current rsum is 470.26000000000005
[root@gpu1 vse_infty-master-my-graph-gru-unified-no-graph-no-pooling]# CUDA_VISIBLE_DEVICES=0,1 python3 eval.py --dataset f30k --data_path ../data/f30k
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
INFO:lib.evaluation:Test: [0/40]	Le 10.2181 (10.2181)	Time 4.384 (0.000)	
INFO:lib.evaluation:Test: [10/40]	Le 10.2104 (10.2096)	Time 0.320 (0.000)	
INFO:lib.evaluation:Test: [20/40]	Le 10.1940 (10.2077)	Time 0.262 (0.000)	
INFO:lib.evaluation:Test: [30/40]	Le 10.2180 (10.2110)	Time 0.371 (0.000)	
INFO:lib.evaluation:Images: 1000, Captions: 5000
INFO:lib.evaluation:calculate similarity time: 0.08050870895385742
INFO:lib.evaluation:rsum: 473.8
INFO:lib.evaluation:Average i2t Recall: 84.9
INFO:lib.evaluation:Image to text: 68.7 90.8 95.2 1.0 3.4
INFO:lib.evaluation:Average t2i Recall: 73.0
INFO:lib.evaluation:Text to image: 51.8 80.1 87.3 1.0 9.8
INFO:root:Evaluating runs/release_weights/f30k_butd_grid_bert...
Traceback (most recent call last):
  File "eval.py", line 58, in <module>
    main()
  File "eval.py", line 54, in main
    evaluation.evalrank(model_path, data_path=opt.data_path, split='test', fold5=False, save_path=save_path)
  File "/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-unified-no-graph-no-pooling/lib/evaluation.py", line 196, in evalrank
    checkpoint = torch.load(model_path)
  File "/root/anaconda3/lib/python3.6/site-packages/torch/serialization.py", line 571, in load
    with _open_file_like(f, 'rb') as opened_file:
  File "/root/anaconda3/lib/python3.6/site-packages/torch/serialization.py", line 229, in _open_file_like
    return _open_file(name_or_buffer, mode)
  File "/root/anaconda3/lib/python3.6/site-packages/torch/serialization.py", line 210, in __init__
    super(_open_file, self).__init__(open(name, mode))
FileNotFoundError: [Errno 2] No such file or directory: 'runs/release_weights/f30k_butd_grid_bert/model_best.pth'
[root@gpu1 vse_infty-master-my-graph-gru-unified-no-graph-no-pooling]# 

