[root@gpu1 vse_infty-master-my-graph-gru-unified-3loss]# sh train_region.sh 
2021-06-04 09:00:42,971 Namespace(backbone_lr_factor=0.01, backbone_path='', backbone_source='detector', backbone_warmup_epochs=5, batch_size=128, data_name='f30k', data_path='../data/f30k', embed_size=1024, embedding_warmup_epochs=2, grad_clip=2.0, img_dim=2048, input_scale_factor=1, learning_rate=0.0005, log_step=200, logger_name='runs/f30k_butd_region_bert/log', lr_update=15, margin=0.2, max_violation=False, model_name='runs/f30k_butd_region_bert', no_imgnorm=False, no_txtnorm=False, num_epochs=25, optim='adam', precomp_enc_type='basic', reset_start_epoch=False, resume='', val_step=500, vocab_path='./vocab/', vse_mean_warmup_epochs=1, word_dim=300, workers=10)
2021-06-04 09:00:42,971 Model name '/public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased' not found in model shortcut name list (bert-base-uncased, bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, bert-base-multilingual-cased, bert-base-chinese, bert-base-german-cased, bert-large-uncased-whole-word-masking, bert-large-cased-whole-word-masking, bert-large-uncased-whole-word-masking-finetuned-squad, bert-large-cased-whole-word-masking-finetuned-squad, bert-base-cased-finetuned-mrpc). Assuming '/public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased' is a path or url to a directory containing tokenizer files.
2021-06-04 09:00:42,971 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/added_tokens.json. We won't load it.
2021-06-04 09:00:42,971 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/special_tokens_map.json. We won't load it.
2021-06-04 09:00:42,971 Didn't find file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/tokenizer_config.json. We won't load it.
2021-06-04 09:00:42,971 loading file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/vocab.txt
2021-06-04 09:00:42,972 loading file None
2021-06-04 09:00:42,972 loading file None
2021-06-04 09:00:42,972 loading file None
/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-unified-3loss/lib/encoders.py:48: UserWarning: nn.init.constant is now deprecated in favor of nn.init.constant_.
  nn.init.constant(self.W[1].weight, 0)
/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-unified-3loss/lib/encoders.py:49: UserWarning: nn.init.constant is now deprecated in favor of nn.init.constant_.
  nn.init.constant(self.W[1].bias, 0)
2021-06-04 09:00:50,262 loading configuration file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/config.json
2021-06-04 09:00:50,262 Model config {
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

2021-06-04 09:00:50,263 loading weights file /public/ZZX/SCAN_dataset/vse_infty-master/bert-base-uncased/pytorch_model.bin
2021-06-04 09:00:56,567 Use adam as the optimizer, with init lr 0.0005
2021-06-04 09:00:56,568 Image encoder is data paralleled now.
2021-06-04 09:00:56,568 runs/f30k_butd_region_bert/log
2021-06-04 09:00:56,568 runs/f30k_butd_region_bert
2021-06-04 09:00:56,570 image encoder trainable parameters: 20490144
2021-06-04 09:00:56,577 txt encoder trainable parameters: 137319072
/root/anaconda3/lib/python3.6/site-packages/torch/nn/modules/rnn.py:735: UserWarning: RNN module weights are not part of single contiguous chunk of memory. This means they need to be compacted at every call, possibly greatly increasing memory usage. To compact weights again call flatten_parameters(). (Triggered internally at  /pytorch/aten/src/ATen/native/cudnn/RNN.cpp:1264.)
  self.dropout, self.training, self.bidirectional, self.batch_first)
2021-06-04 09:05:16,161 Epoch: [0][199/1133]	Eit 200  lr 0.0005  Le 29.5763 (29.6146)	Time 1.153 (0.000)	Data 0.002 (0.000)	
2021-06-04 09:09:23,151 Epoch: [0][399/1133]	Eit 400  lr 0.0005  Le 29.5632 (29.5882)	Time 1.304 (0.000)	Data 0.002 (0.000)	
2021-06-04 09:13:27,825 Epoch: [0][599/1133]	Eit 600  lr 0.0005  Le 29.5414 (29.5688)	Time 1.085 (0.000)	Data 0.002 (0.000)	
2021-06-04 09:17:30,255 Epoch: [0][799/1133]	Eit 800  lr 0.0005  Le 29.4962 (29.5531)	Time 1.392 (0.000)	Data 0.002 (0.000)	
2021-06-04 09:21:30,266 Epoch: [0][999/1133]	Eit 1000  lr 0.0005  Le 29.4890 (29.5395)	Time 1.196 (0.000)	Data 0.002 (0.000)	
2021-06-04 09:24:15,331 Test: [0/40]	Le 30.0897 (30.0897)	Time 4.098 (0.000)	
2021-06-04 09:24:36,282 calculate similarity time: 0.06224226951599121
2021-06-04 09:24:36,677 Image to text: 68.9, 88.8, 94.3, 1.0, 3.7
2021-06-04 09:24:37,007 Text to image: 50.0, 76.8, 85.5, 2.0, 8.8
2021-06-04 09:24:37,007 Current rsum is 464.26
2021-06-04 09:24:39,645 runs/f30k_butd_region_bert/log
2021-06-04 09:24:39,645 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-04 09:24:39,649 image encoder trainable parameters: 20490144
2021-06-04 09:24:39,661 txt encoder trainable parameters: 137319072
2021-06-04 09:26:06,335 Epoch: [1][67/1133]	Eit 1200  lr 0.0005  Le 29.4742 (29.4518)	Time 1.459 (0.000)	Data 0.002 (0.000)	
2021-06-04 09:30:13,566 Epoch: [1][267/1133]	Eit 1400  lr 0.0005  Le 29.4311 (29.4466)	Time 1.264 (0.000)	Data 0.002 (0.000)	
2021-06-04 09:34:19,127 Epoch: [1][467/1133]	Eit 1600  lr 0.0005  Le 29.4320 (29.4434)	Time 1.184 (0.000)	Data 0.002 (0.000)	
2021-06-04 09:38:23,993 Epoch: [1][667/1133]	Eit 1800  lr 0.0005  Le 29.4149 (29.4405)	Time 1.146 (0.000)	Data 0.003 (0.000)	
2021-06-04 09:42:26,390 Epoch: [1][867/1133]	Eit 2000  lr 0.0005  Le 29.4128 (29.4373)	Time 1.386 (0.000)	Data 0.002 (0.000)	
2021-06-04 09:46:31,783 Epoch: [1][1067/1133]	Eit 2200  lr 0.0005  Le 29.4125 (29.4345)	Time 1.245 (0.000)	Data 0.002 (0.000)	
2021-06-04 09:47:52,723 Test: [0/40]	Le 30.0819 (30.0818)	Time 3.141 (0.000)	
2021-06-04 09:48:12,846 calculate similarity time: 0.04874873161315918
2021-06-04 09:48:13,351 Image to text: 74.2, 91.2, 95.2, 1.0, 3.1
2021-06-04 09:48:13,696 Text to image: 54.7, 81.0, 88.3, 1.0, 7.4
2021-06-04 09:48:13,696 Current rsum is 484.6
2021-06-04 09:48:16,943 runs/f30k_butd_region_bert/log
2021-06-04 09:48:16,944 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-04 09:48:16,948 image encoder trainable parameters: 20490144
2021-06-04 09:48:16,960 txt encoder trainable parameters: 137319072
2021-06-04 09:51:07,893 Epoch: [2][135/1133]	Eit 2400  lr 0.0005  Le 29.3931 (29.3988)	Time 1.260 (0.000)	Data 0.002 (0.000)	
2021-06-04 09:55:13,428 Epoch: [2][335/1133]	Eit 2600  lr 0.0005  Le 29.3878 (29.3976)	Time 1.271 (0.000)	Data 0.002 (0.000)	
2021-06-04 09:59:18,651 Epoch: [2][535/1133]	Eit 2800  lr 0.0005  Le 29.3630 (29.3975)	Time 1.199 (0.000)	Data 0.002 (0.000)	
2021-06-04 10:03:22,100 Epoch: [2][735/1133]	Eit 3000  lr 0.0005  Le 29.3983 (29.3970)	Time 1.153 (0.000)	Data 0.002 (0.000)	
2021-06-04 10:07:24,753 Epoch: [2][935/1133]	Eit 3200  lr 0.0005  Le 29.3987 (29.3961)	Time 1.342 (0.000)	Data 0.002 (0.000)	
2021-06-04 10:11:29,388 Test: [0/40]	Le 30.0853 (30.0853)	Time 3.859 (0.000)	
2021-06-04 10:11:49,861 calculate similarity time: 0.05554056167602539
2021-06-04 10:11:50,294 Image to text: 79.4, 94.5, 97.7, 1.0, 2.0
2021-06-04 10:11:50,609 Text to image: 58.2, 84.3, 90.6, 1.0, 6.7
2021-06-04 10:11:50,609 Current rsum is 504.74000000000007
2021-06-04 10:11:54,315 runs/f30k_butd_region_bert/log
2021-06-04 10:11:54,315 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-04 10:11:54,318 image encoder trainable parameters: 20490144
2021-06-04 10:11:54,329 txt encoder trainable parameters: 137319072
2021-06-04 10:12:02,454 Epoch: [3][3/1133]	Eit 3400  lr 0.0005  Le 29.3573 (29.3681)	Time 1.192 (0.000)	Data 0.002 (0.000)	
2021-06-04 10:16:07,587 Epoch: [3][203/1133]	Eit 3600  lr 0.0005  Le 29.3845 (29.3690)	Time 1.267 (0.000)	Data 0.002 (0.000)	
2021-06-04 10:20:11,601 Epoch: [3][403/1133]	Eit 3800  lr 0.0005  Le 29.3763 (29.3704)	Time 1.163 (0.000)	Data 0.002 (0.000)	
2021-06-04 10:24:15,314 Epoch: [3][603/1133]	Eit 4000  lr 0.0005  Le 29.4020 (29.3705)	Time 1.307 (0.000)	Data 0.002 (0.000)	
2021-06-04 10:28:19,409 Epoch: [3][803/1133]	Eit 4200  lr 0.0005  Le 29.3400 (29.3701)	Time 1.204 (0.000)	Data 0.002 (0.000)	
2021-06-04 10:32:25,421 Epoch: [3][1003/1133]	Eit 4400  lr 0.0005  Le 29.3467 (29.3693)	Time 1.345 (0.000)	Data 0.002 (0.000)	
2021-06-04 10:35:07,464 Test: [0/40]	Le 30.0840 (30.0840)	Time 3.985 (0.000)	
2021-06-04 10:35:27,897 calculate similarity time: 0.06380915641784668
2021-06-04 10:35:28,398 Image to text: 78.4, 94.7, 97.9, 1.0, 2.5
2021-06-04 10:35:28,737 Text to image: 58.0, 84.4, 91.0, 1.0, 6.3
2021-06-04 10:35:28,737 Current rsum is 504.4
2021-06-04 10:35:30,149 runs/f30k_butd_region_bert/log
2021-06-04 10:35:30,149 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-04 10:35:30,150 image encoder trainable parameters: 20490144
2021-06-04 10:35:30,155 txt encoder trainable parameters: 137319072
2021-06-04 10:37:02,275 Epoch: [4][71/1133]	Eit 4600  lr 0.0005  Le 29.3377 (29.3468)	Time 1.401 (0.000)	Data 0.002 (0.000)	
2021-06-04 10:41:04,851 Epoch: [4][271/1133]	Eit 4800  lr 0.0005  Le 29.3522 (29.3492)	Time 1.172 (0.000)	Data 0.002 (0.000)	
2021-06-04 10:45:09,833 Epoch: [4][471/1133]	Eit 5000  lr 0.0005  Le 29.3579 (29.3498)	Time 1.283 (0.000)	Data 0.003 (0.000)	
2021-06-04 10:49:16,211 Epoch: [4][671/1133]	Eit 5200  lr 0.0005  Le 29.3687 (29.3509)	Time 1.222 (0.000)	Data 0.002 (0.000)	
2021-06-04 10:53:19,737 Epoch: [4][871/1133]	Eit 5400  lr 0.0005  Le 29.3281 (29.3509)	Time 1.261 (0.000)	Data 0.002 (0.000)	
2021-06-04 10:57:21,673 Epoch: [4][1071/1133]	Eit 5600  lr 0.0005  Le 29.3169 (29.3511)	Time 1.113 (0.000)	Data 0.002 (0.000)	
2021-06-04 10:58:39,103 Test: [0/40]	Le 30.0868 (30.0868)	Time 4.135 (0.000)	
2021-06-04 10:59:00,421 calculate similarity time: 0.0614008903503418
2021-06-04 10:59:00,945 Image to text: 79.3, 94.3, 97.2, 1.0, 2.1
2021-06-04 10:59:01,257 Text to image: 61.0, 86.0, 91.7, 1.0, 6.1
2021-06-04 10:59:01,257 Current rsum is 509.52
2021-06-04 10:59:04,656 runs/f30k_butd_region_bert/log
2021-06-04 10:59:04,657 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-04 10:59:04,661 image encoder trainable parameters: 20490144
2021-06-04 10:59:04,673 txt encoder trainable parameters: 137319072
2021-06-04 11:01:59,084 Epoch: [5][139/1133]	Eit 5800  lr 0.0005  Le 29.3173 (29.3344)	Time 1.178 (0.000)	Data 0.002 (0.000)	
2021-06-04 11:06:02,412 Epoch: [5][339/1133]	Eit 6000  lr 0.0005  Le 29.3015 (29.3335)	Time 1.156 (0.000)	Data 0.002 (0.000)	
2021-06-04 11:10:06,883 Epoch: [5][539/1133]	Eit 6200  lr 0.0005  Le 29.3533 (29.3352)	Time 1.244 (0.000)	Data 0.002 (0.000)	
2021-06-04 11:14:09,033 Epoch: [5][739/1133]	Eit 6400  lr 0.0005  Le 29.3160 (29.3355)	Time 1.212 (0.000)	Data 0.002 (0.000)	
2021-06-04 11:18:12,409 Epoch: [5][939/1133]	Eit 6600  lr 0.0005  Le 29.3338 (29.3355)	Time 1.236 (0.000)	Data 0.002 (0.000)	
2021-06-04 11:22:09,067 Test: [0/40]	Le 30.0865 (30.0864)	Time 3.884 (0.000)	
2021-06-04 11:22:29,274 calculate similarity time: 0.04607105255126953
2021-06-04 11:22:29,802 Image to text: 79.6, 95.6, 98.3, 1.0, 1.9
2021-06-04 11:22:30,238 Text to image: 60.6, 85.4, 91.3, 1.0, 6.5
2021-06-04 11:22:30,238 Current rsum is 510.78
2021-06-04 11:22:34,238 runs/f30k_butd_region_bert/log
2021-06-04 11:22:34,239 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-04 11:22:34,242 image encoder trainable parameters: 20490144
2021-06-04 11:22:34,253 txt encoder trainable parameters: 137319072
2021-06-04 11:22:47,733 Epoch: [6][7/1133]	Eit 6800  lr 0.0005  Le 29.3440 (29.3176)	Time 1.169 (0.000)	Data 0.002 (0.000)	
2021-06-04 11:26:50,457 Epoch: [6][207/1133]	Eit 7000  lr 0.0005  Le 29.3049 (29.3216)	Time 1.344 (0.000)	Data 0.002 (0.000)	
2021-06-04 11:30:52,450 Epoch: [6][407/1133]	Eit 7200  lr 0.0005  Le 29.3337 (29.3222)	Time 1.234 (0.000)	Data 0.002 (0.000)	
2021-06-04 11:34:58,602 Epoch: [6][607/1133]	Eit 7400  lr 0.0005  Le 29.3020 (29.3222)	Time 1.283 (0.000)	Data 0.003 (0.000)	
2021-06-04 11:39:05,989 Epoch: [6][807/1133]	Eit 7600  lr 0.0005  Le 29.3416 (29.3230)	Time 1.225 (0.000)	Data 0.002 (0.000)	
2021-06-04 11:43:07,382 Epoch: [6][1007/1133]	Eit 7800  lr 0.0005  Le 29.3118 (29.3232)	Time 1.278 (0.000)	Data 0.002 (0.000)	
2021-06-04 11:45:41,799 Test: [0/40]	Le 30.0779 (30.0779)	Time 4.233 (0.000)	
2021-06-04 11:46:02,400 calculate similarity time: 0.06472969055175781
2021-06-04 11:46:02,928 Image to text: 80.8, 95.1, 98.0, 1.0, 2.0
2021-06-04 11:46:03,259 Text to image: 61.2, 85.3, 91.1, 1.0, 6.4
2021-06-04 11:46:03,259 Current rsum is 511.48
2021-06-04 11:46:06,892 runs/f30k_butd_region_bert/log
2021-06-04 11:46:06,893 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-04 11:46:06,898 image encoder trainable parameters: 20490144
2021-06-04 11:46:06,908 txt encoder trainable parameters: 137319072
2021-06-04 11:47:41,436 Epoch: [7][75/1133]	Eit 8000  lr 0.0005  Le 29.3017 (29.3045)	Time 1.158 (0.000)	Data 0.002 (0.000)	
2021-06-04 11:51:45,225 Epoch: [7][275/1133]	Eit 8200  lr 0.0005  Le 29.2962 (29.3085)	Time 1.098 (0.000)	Data 0.002 (0.000)	
2021-06-04 11:55:49,533 Epoch: [7][475/1133]	Eit 8400  lr 0.0005  Le 29.2919 (29.3089)	Time 1.220 (0.000)	Data 0.002 (0.000)	
2021-06-04 11:59:52,301 Epoch: [7][675/1133]	Eit 8600  lr 0.0005  Le 29.3188 (29.3101)	Time 1.254 (0.000)	Data 0.002 (0.000)	
2021-06-04 12:03:55,457 Epoch: [7][875/1133]	Eit 8800  lr 0.0005  Le 29.3096 (29.3109)	Time 1.166 (0.000)	Data 0.002 (0.000)	
2021-06-04 12:07:59,235 Epoch: [7][1075/1133]	Eit 9000  lr 0.0005  Le 29.3217 (29.3119)	Time 1.252 (0.000)	Data 0.002 (0.000)	
2021-06-04 12:09:10,613 Test: [0/40]	Le 30.0822 (30.0822)	Time 3.714 (0.000)	
2021-06-04 12:09:32,127 calculate similarity time: 0.06296229362487793
2021-06-04 12:09:32,522 Image to text: 80.5, 95.3, 98.3, 1.0, 2.1
2021-06-04 12:09:32,834 Text to image: 61.5, 85.4, 91.3, 1.0, 6.4
2021-06-04 12:09:32,834 Current rsum is 512.28
2021-06-04 12:09:36,448 runs/f30k_butd_region_bert/log
2021-06-04 12:09:36,448 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-04 12:09:36,451 image encoder trainable parameters: 20490144
2021-06-04 12:09:36,461 txt encoder trainable parameters: 137319072
2021-06-04 12:12:33,942 Epoch: [8][143/1133]	Eit 9200  lr 0.0005  Le 29.3114 (29.3000)	Time 1.122 (0.000)	Data 0.002 (0.000)	
2021-06-04 12:16:38,022 Epoch: [8][343/1133]	Eit 9400  lr 0.0005  Le 29.3038 (29.3011)	Time 1.098 (0.000)	Data 0.002 (0.000)	
2021-06-04 12:20:43,142 Epoch: [8][543/1133]	Eit 9600  lr 0.0005  Le 29.3045 (29.3023)	Time 1.241 (0.000)	Data 0.002 (0.000)	
2021-06-04 12:24:49,360 Epoch: [8][743/1133]	Eit 9800  lr 0.0005  Le 29.3228 (29.3023)	Time 1.320 (0.000)	Data 0.002 (0.000)	
2021-06-04 12:28:55,481 Epoch: [8][943/1133]	Eit 10000  lr 0.0005  Le 29.3133 (29.3030)	Time 1.502 (0.000)	Data 0.002 (0.000)	
2021-06-04 12:32:47,833 Test: [0/40]	Le 30.0817 (30.0817)	Time 3.820 (0.000)	
2021-06-04 12:33:09,092 calculate similarity time: 0.06130671501159668
2021-06-04 12:33:09,614 Image to text: 80.0, 95.7, 97.9, 1.0, 2.1
2021-06-04 12:33:09,930 Text to image: 61.2, 86.1, 91.5, 1.0, 6.3
2021-06-04 12:33:09,930 Current rsum is 512.4399999999999
2021-06-04 12:33:13,346 runs/f30k_butd_region_bert/log
2021-06-04 12:33:13,347 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-04 12:33:13,350 image encoder trainable parameters: 20490144
2021-06-04 12:33:13,362 txt encoder trainable parameters: 137319072
2021-06-04 12:33:31,302 Epoch: [9][11/1133]	Eit 10200  lr 0.0005  Le 29.2898 (29.2891)	Time 1.130 (0.000)	Data 0.002 (0.000)	
2021-06-04 12:37:37,513 Epoch: [9][211/1133]	Eit 10400  lr 0.0005  Le 29.3131 (29.2900)	Time 1.246 (0.000)	Data 0.002 (0.000)	
2021-06-04 12:41:41,067 Epoch: [9][411/1133]	Eit 10600  lr 0.0005  Le 29.3186 (29.2926)	Time 1.193 (0.000)	Data 0.002 (0.000)	
2021-06-04 12:45:44,873 Epoch: [9][611/1133]	Eit 10800  lr 0.0005  Le 29.2937 (29.2931)	Time 1.249 (0.000)	Data 0.002 (0.000)	
2021-06-04 12:49:49,514 Epoch: [9][811/1133]	Eit 11000  lr 0.0005  Le 29.2729 (29.2942)	Time 1.170 (0.000)	Data 0.002 (0.000)	
2021-06-04 12:53:49,327 Epoch: [9][1011/1133]	Eit 11200  lr 0.0005  Le 29.2813 (29.2949)	Time 1.180 (0.000)	Data 0.002 (0.000)	
2021-06-04 12:56:20,059 Test: [0/40]	Le 30.0774 (30.0773)	Time 4.009 (0.000)	
2021-06-04 12:56:41,489 calculate similarity time: 0.06213212013244629
2021-06-04 12:56:41,992 Image to text: 81.6, 95.5, 98.5, 1.0, 2.0
2021-06-04 12:56:42,312 Text to image: 61.9, 85.9, 91.3, 1.0, 6.8
2021-06-04 12:56:42,312 Current rsum is 514.82
2021-06-04 12:56:45,796 runs/f30k_butd_region_bert/log
2021-06-04 12:56:45,797 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-04 12:56:45,800 image encoder trainable parameters: 20490144
2021-06-04 12:56:45,811 txt encoder trainable parameters: 137319072
2021-06-04 12:58:24,436 Epoch: [10][79/1133]	Eit 11400  lr 0.0005  Le 29.2644 (29.2824)	Time 1.131 (0.000)	Data 0.002 (0.000)	
2021-06-04 13:02:28,422 Epoch: [10][279/1133]	Eit 11600  lr 0.0005  Le 29.2902 (29.2840)	Time 1.227 (0.000)	Data 0.002 (0.000)	
2021-06-04 13:06:31,873 Epoch: [10][479/1133]	Eit 11800  lr 0.0005  Le 29.2894 (29.2846)	Time 1.177 (0.000)	Data 0.002 (0.000)	
2021-06-04 13:10:33,834 Epoch: [10][679/1133]	Eit 12000  lr 0.0005  Le 29.2931 (29.2858)	Time 1.143 (0.000)	Data 0.002 (0.000)	
2021-06-04 13:14:36,943 Epoch: [10][879/1133]	Eit 12200  lr 0.0005  Le 29.2641 (29.2863)	Time 1.201 (0.000)	Data 0.002 (0.000)	
2021-06-04 13:18:39,972 Epoch: [10][1079/1133]	Eit 12400  lr 0.0005  Le 29.3087 (29.2870)	Time 1.224 (0.000)	Data 0.002 (0.000)	
2021-06-04 13:19:44,553 Test: [0/40]	Le 30.0777 (30.0777)	Time 3.374 (0.000)	
2021-06-04 13:20:05,124 calculate similarity time: 0.056427717208862305
2021-06-04 13:20:05,512 Image to text: 82.0, 96.2, 98.3, 1.0, 2.0
2021-06-04 13:20:05,826 Text to image: 61.4, 85.6, 91.6, 1.0, 6.9
2021-06-04 13:20:05,826 Current rsum is 515.08
2021-06-04 13:20:09,418 runs/f30k_butd_region_bert/log
2021-06-04 13:20:09,418 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-04 13:20:09,422 image encoder trainable parameters: 20490144
2021-06-04 13:20:09,434 txt encoder trainable parameters: 137319072
2021-06-04 13:23:12,418 Epoch: [11][147/1133]	Eit 12600  lr 0.0005  Le 29.2673 (29.2752)	Time 1.282 (0.000)	Data 0.003 (0.000)	
2021-06-04 13:27:16,228 Epoch: [11][347/1133]	Eit 12800  lr 0.0005  Le 29.2514 (29.2773)	Time 1.197 (0.000)	Data 0.002 (0.000)	
2021-06-04 13:31:21,039 Epoch: [11][547/1133]	Eit 13000  lr 0.0005  Le 29.2750 (29.2787)	Time 1.185 (0.000)	Data 0.002 (0.000)	
2021-06-04 13:35:24,801 Epoch: [11][747/1133]	Eit 13200  lr 0.0005  Le 29.2528 (29.2798)	Time 1.173 (0.000)	Data 0.002 (0.000)	
2021-06-04 13:39:24,486 Epoch: [11][947/1133]	Eit 13400  lr 0.0005  Le 29.2712 (29.2809)	Time 1.158 (0.000)	Data 0.002 (0.000)	
2021-06-04 13:43:12,665 Test: [0/40]	Le 30.0752 (30.0752)	Time 3.898 (0.000)	
2021-06-04 13:43:33,362 calculate similarity time: 0.06336617469787598
2021-06-04 13:43:33,866 Image to text: 81.6, 96.0, 98.2, 1.0, 1.8
2021-06-04 13:43:34,228 Text to image: 61.6, 85.9, 91.3, 1.0, 6.7
2021-06-04 13:43:34,228 Current rsum is 514.6800000000001
2021-06-04 13:43:35,983 runs/f30k_butd_region_bert/log
2021-06-04 13:43:35,984 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-04 13:43:35,987 image encoder trainable parameters: 20490144
2021-06-04 13:43:35,995 txt encoder trainable parameters: 137319072
2021-06-04 13:43:58,948 Epoch: [12][15/1133]	Eit 13600  lr 0.0005  Le 29.2743 (29.2669)	Time 1.230 (0.000)	Data 0.003 (0.000)	
2021-06-04 13:48:01,125 Epoch: [12][215/1133]	Eit 13800  lr 0.0005  Le 29.2873 (29.2717)	Time 1.241 (0.000)	Data 0.002 (0.000)	
2021-06-04 13:52:04,408 Epoch: [12][415/1133]	Eit 14000  lr 0.0005  Le 29.2724 (29.2730)	Time 1.255 (0.000)	Data 0.002 (0.000)	
2021-06-04 13:56:06,361 Epoch: [12][615/1133]	Eit 14200  lr 0.0005  Le 29.2741 (29.2741)	Time 1.180 (0.000)	Data 0.002 (0.000)	
2021-06-04 14:00:06,815 Epoch: [12][815/1133]	Eit 14400  lr 0.0005  Le 29.2443 (29.2747)	Time 1.114 (0.000)	Data 0.002 (0.000)	
2021-06-04 14:04:13,338 Epoch: [12][1015/1133]	Eit 14600  lr 0.0005  Le 29.2490 (29.2754)	Time 1.147 (0.000)	Data 0.011 (0.000)	
2021-06-04 14:06:38,872 Test: [0/40]	Le 30.0735 (30.0735)	Time 4.015 (0.000)	
2021-06-04 14:06:59,476 calculate similarity time: 0.06284236907958984
2021-06-04 14:06:59,985 Image to text: 80.2, 95.9, 98.1, 1.0, 1.8
2021-06-04 14:07:00,340 Text to image: 61.5, 85.2, 91.2, 1.0, 6.7
2021-06-04 14:07:00,340 Current rsum is 512.08
2021-06-04 14:07:01,721 runs/f30k_butd_region_bert/log
2021-06-04 14:07:01,721 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-04 14:07:01,723 image encoder trainable parameters: 20490144
2021-06-04 14:07:01,728 txt encoder trainable parameters: 137319072
2021-06-04 14:08:47,879 Epoch: [13][83/1133]	Eit 14800  lr 0.0005  Le 29.2682 (29.2661)	Time 1.324 (0.000)	Data 0.002 (0.000)	
2021-06-04 14:12:54,354 Epoch: [13][283/1133]	Eit 15000  lr 0.0005  Le 29.2614 (29.2678)	Time 1.178 (0.000)	Data 0.002 (0.000)	
2021-06-04 14:17:00,652 Epoch: [13][483/1133]	Eit 15200  lr 0.0005  Le 29.2808 (29.2676)	Time 1.239 (0.000)	Data 0.002 (0.000)	
2021-06-04 14:21:02,046 Epoch: [13][683/1133]	Eit 15400  lr 0.0005  Le 29.2584 (29.2692)	Time 1.178 (0.000)	Data 0.002 (0.000)	
2021-06-04 14:25:07,395 Epoch: [13][883/1133]	Eit 15600  lr 0.0005  Le 29.2710 (29.2699)	Time 1.161 (0.000)	Data 0.010 (0.000)	
2021-06-04 14:29:10,610 Epoch: [13][1083/1133]	Eit 15800  lr 0.0005  Le 29.2930 (29.2703)	Time 1.226 (0.000)	Data 0.002 (0.000)	
2021-06-04 14:30:13,689 Test: [0/40]	Le 30.0718 (30.0718)	Time 4.006 (0.000)	
2021-06-04 14:30:34,181 calculate similarity time: 0.04924798011779785
2021-06-04 14:30:34,687 Image to text: 80.3, 95.8, 98.4, 1.0, 2.0
2021-06-04 14:30:35,009 Text to image: 61.8, 85.9, 90.9, 1.0, 7.1
2021-06-04 14:30:35,009 Current rsum is 513.0799999999999
2021-06-04 14:30:36,387 runs/f30k_butd_region_bert/log
2021-06-04 14:30:36,388 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-04 14:30:36,389 image encoder trainable parameters: 20490144
2021-06-04 14:30:36,394 txt encoder trainable parameters: 137319072
2021-06-04 14:33:43,679 Epoch: [14][151/1133]	Eit 16000  lr 0.0005  Le 29.2563 (29.2597)	Time 1.208 (0.000)	Data 0.002 (0.000)	
2021-06-04 14:37:46,440 Epoch: [14][351/1133]	Eit 16200  lr 0.0005  Le 29.2889 (29.2616)	Time 1.178 (0.000)	Data 0.002 (0.000)	
2021-06-04 14:41:53,869 Epoch: [14][551/1133]	Eit 16400  lr 0.0005  Le 29.2731 (29.2634)	Time 1.244 (0.000)	Data 0.002 (0.000)	
2021-06-04 14:45:58,001 Epoch: [14][751/1133]	Eit 16600  lr 0.0005  Le 29.2602 (29.2643)	Time 1.319 (0.000)	Data 0.002 (0.000)	
2021-06-04 14:50:01,074 Epoch: [14][951/1133]	Eit 16800  lr 0.0005  Le 29.2643 (29.2653)	Time 1.194 (0.000)	Data 0.002 (0.000)	
2021-06-04 14:53:44,626 Test: [0/40]	Le 30.0710 (30.0709)	Time 3.922 (0.000)	
2021-06-04 14:54:05,696 calculate similarity time: 0.05904555320739746
2021-06-04 14:54:06,219 Image to text: 81.3, 95.8, 98.5, 1.0, 1.8
2021-06-04 14:54:06,596 Text to image: 61.5, 85.2, 90.9, 1.0, 7.2
2021-06-04 14:54:06,596 Current rsum is 513.24
2021-06-04 14:54:08,132 runs/f30k_butd_region_bert/log
2021-06-04 14:54:08,132 runs/f30k_butd_region_bert
2021-06-04 14:54:08,133 Current epoch num is 15, decrease all lr by 10
2021-06-04 14:54:08,133 new lr 5e-05
2021-06-04 14:54:08,133 new lr 5e-06
2021-06-04 14:54:08,133 new lr 5e-05
Use VSE++ objective.
2021-06-04 14:54:08,137 image encoder trainable parameters: 20490144
2021-06-04 14:54:08,148 txt encoder trainable parameters: 137319072
2021-06-04 14:54:36,525 Epoch: [15][19/1133]	Eit 17000  lr 5e-05  Le 29.2540 (29.2546)	Time 1.236 (0.000)	Data 0.002 (0.000)	
2021-06-04 14:58:40,992 Epoch: [15][219/1133]	Eit 17200  lr 5e-05  Le 29.2492 (29.2522)	Time 1.191 (0.000)	Data 0.002 (0.000)	
2021-06-04 15:02:47,209 Epoch: [15][419/1133]	Eit 17400  lr 5e-05  Le 29.2441 (29.2504)	Time 1.344 (0.000)	Data 0.002 (0.000)	
2021-06-04 15:06:50,440 Epoch: [15][619/1133]	Eit 17600  lr 5e-05  Le 29.2445 (29.2496)	Time 1.222 (0.000)	Data 0.002 (0.000)	
2021-06-04 15:10:57,571 Epoch: [15][819/1133]	Eit 17800  lr 5e-05  Le 29.2598 (29.2487)	Time 1.315 (0.000)	Data 0.002 (0.000)	
2021-06-04 15:14:58,976 Epoch: [15][1019/1133]	Eit 18000  lr 5e-05  Le 29.2329 (29.2480)	Time 1.276 (0.000)	Data 0.002 (0.000)	
2021-06-04 15:17:17,719 Test: [0/40]	Le 30.0715 (30.0715)	Time 4.147 (0.000)	
2021-06-04 15:17:38,068 calculate similarity time: 0.06728410720825195
2021-06-04 15:17:38,596 Image to text: 82.5, 96.2, 98.4, 1.0, 1.7
2021-06-04 15:17:38,920 Text to image: 63.1, 85.7, 91.5, 1.0, 7.0
2021-06-04 15:17:38,920 Current rsum is 517.3800000000001
2021-06-04 15:17:42,456 runs/f30k_butd_region_bert/log
2021-06-04 15:17:42,457 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-04 15:17:42,460 image encoder trainable parameters: 20490144
2021-06-04 15:17:42,474 txt encoder trainable parameters: 137319072
2021-06-04 15:19:33,734 Epoch: [16][87/1133]	Eit 18200  lr 5e-05  Le 29.2445 (29.2395)	Time 1.224 (0.000)	Data 0.002 (0.000)	
2021-06-04 15:23:35,485 Epoch: [16][287/1133]	Eit 18400  lr 5e-05  Le 29.2728 (29.2410)	Time 1.481 (0.000)	Data 0.002 (0.000)	
2021-06-04 15:27:36,534 Epoch: [16][487/1133]	Eit 18600  lr 5e-05  Le 29.2303 (29.2410)	Time 1.151 (0.000)	Data 0.002 (0.000)	
2021-06-04 15:31:39,365 Epoch: [16][687/1133]	Eit 18800  lr 5e-05  Le 29.2336 (29.2407)	Time 1.280 (0.000)	Data 0.007 (0.000)	
2021-06-04 15:35:51,907 Epoch: [16][887/1133]	Eit 19000  lr 5e-05  Le 29.2477 (29.2406)	Time 1.097 (0.000)	Data 0.002 (0.000)	
2021-06-04 15:39:56,172 Epoch: [16][1087/1133]	Eit 19200  lr 5e-05  Le 29.2363 (29.2405)	Time 1.485 (0.000)	Data 0.002 (0.000)	
2021-06-04 15:40:53,257 Test: [0/40]	Le 30.0708 (30.0707)	Time 3.972 (0.000)	
2021-06-04 15:41:13,746 calculate similarity time: 0.06298279762268066
2021-06-04 15:41:14,136 Image to text: 82.9, 96.3, 98.7, 1.0, 1.8
2021-06-04 15:41:14,449 Text to image: 63.6, 85.8, 91.5, 1.0, 7.0
2021-06-04 15:41:14,449 Current rsum is 518.76
2021-06-04 15:41:17,894 runs/f30k_butd_region_bert/log
2021-06-04 15:41:17,895 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-04 15:41:17,899 image encoder trainable parameters: 20490144
2021-06-04 15:41:17,910 txt encoder trainable parameters: 137319072
2021-06-04 15:44:30,739 Epoch: [17][155/1133]	Eit 19400  lr 5e-05  Le 29.2107 (29.2363)	Time 1.216 (0.000)	Data 0.004 (0.000)	
2021-06-04 15:48:36,904 Epoch: [17][355/1133]	Eit 19600  lr 5e-05  Le 29.2256 (29.2376)	Time 1.282 (0.000)	Data 0.002 (0.000)	
2021-06-04 15:52:42,024 Epoch: [17][555/1133]	Eit 19800  lr 5e-05  Le 29.2077 (29.2383)	Time 1.191 (0.000)	Data 0.002 (0.000)	
2021-06-04 15:56:42,475 Epoch: [17][755/1133]	Eit 20000  lr 5e-05  Le 29.2299 (29.2387)	Time 1.121 (0.000)	Data 0.002 (0.000)	
2021-06-04 16:00:43,509 Epoch: [17][955/1133]	Eit 20200  lr 5e-05  Le 29.2437 (29.2383)	Time 1.148 (0.000)	Data 0.002 (0.000)	
2021-06-04 16:04:17,349 Test: [0/40]	Le 30.0726 (30.0726)	Time 3.762 (0.000)	
2021-06-04 16:04:37,683 calculate similarity time: 0.06253480911254883
2021-06-04 16:04:38,075 Image to text: 82.6, 96.9, 98.6, 1.0, 1.8
2021-06-04 16:04:38,387 Text to image: 63.3, 85.8, 91.4, 1.0, 7.1
2021-06-04 16:04:38,387 Current rsum is 518.54
2021-06-04 16:04:39,907 runs/f30k_butd_region_bert/log
2021-06-04 16:04:39,907 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-04 16:04:39,908 image encoder trainable parameters: 20490144
2021-06-04 16:04:39,913 txt encoder trainable parameters: 137319072
2021-06-04 16:05:12,228 Epoch: [18][23/1133]	Eit 20400  lr 5e-05  Le 29.2148 (29.2349)	Time 1.218 (0.000)	Data 0.002 (0.000)	
2021-06-04 16:09:16,276 Epoch: [18][223/1133]	Eit 20600  lr 5e-05  Le 29.2398 (29.2376)	Time 1.215 (0.000)	Data 0.002 (0.000)	
2021-06-04 16:13:18,879 Epoch: [18][423/1133]	Eit 20800  lr 5e-05  Le 29.2487 (29.2366)	Time 1.149 (0.000)	Data 0.002 (0.000)	
2021-06-04 16:17:21,448 Epoch: [18][623/1133]	Eit 21000  lr 5e-05  Le 29.2102 (29.2357)	Time 1.224 (0.000)	Data 0.002 (0.000)	
2021-06-04 16:21:23,521 Epoch: [18][823/1133]	Eit 21200  lr 5e-05  Le 29.2710 (29.2356)	Time 1.125 (0.000)	Data 0.002 (0.000)	
2021-06-04 16:25:25,122 Epoch: [18][1023/1133]	Eit 21400  lr 5e-05  Le 29.2359 (29.2354)	Time 1.221 (0.000)	Data 0.002 (0.000)	
2021-06-04 16:27:39,200 Test: [0/40]	Le 30.0728 (30.0728)	Time 3.950 (0.000)	
2021-06-04 16:27:59,837 calculate similarity time: 0.06393146514892578
2021-06-04 16:28:00,232 Image to text: 83.1, 96.4, 98.8, 1.0, 1.8
2021-06-04 16:28:00,562 Text to image: 63.5, 85.8, 91.3, 1.0, 7.2
2021-06-04 16:28:00,562 Current rsum is 518.9200000000001
2021-06-04 16:28:04,103 runs/f30k_butd_region_bert/log
2021-06-04 16:28:04,104 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-04 16:28:04,108 image encoder trainable parameters: 20490144
2021-06-04 16:28:04,120 txt encoder trainable parameters: 137319072
2021-06-04 16:30:02,886 Epoch: [19][91/1133]	Eit 21600  lr 5e-05  Le 29.2222 (29.2326)	Time 1.238 (0.000)	Data 0.002 (0.000)	
2021-06-04 16:34:05,539 Epoch: [19][291/1133]	Eit 21800  lr 5e-05  Le 29.2237 (29.2335)	Time 1.118 (0.000)	Data 0.002 (0.000)	
2021-06-04 16:38:04,225 Epoch: [19][491/1133]	Eit 22000  lr 5e-05  Le 29.2370 (29.2340)	Time 1.437 (0.000)	Data 0.002 (0.000)	
2021-06-04 16:42:03,058 Epoch: [19][691/1133]	Eit 22200  lr 5e-05  Le 29.2392 (29.2342)	Time 1.451 (0.000)	Data 0.002 (0.000)	
2021-06-04 16:45:59,458 Epoch: [19][891/1133]	Eit 22400  lr 5e-05  Le 29.2095 (29.2341)	Time 1.166 (0.000)	Data 0.002 (0.000)	
2021-06-04 16:49:57,770 Epoch: [19][1091/1133]	Eit 22600  lr 5e-05  Le 29.2390 (29.2340)	Time 1.226 (0.000)	Data 0.002 (0.000)	
2021-06-04 16:50:48,573 Test: [0/40]	Le 30.0716 (30.0715)	Time 3.636 (0.000)	
2021-06-04 16:51:08,926 calculate similarity time: 0.06409573554992676
2021-06-04 16:51:09,447 Image to text: 82.9, 96.2, 98.7, 1.0, 1.8
2021-06-04 16:51:09,759 Text to image: 63.1, 85.8, 91.3, 1.0, 7.2
2021-06-04 16:51:09,759 Current rsum is 518.04
2021-06-04 16:51:11,124 runs/f30k_butd_region_bert/log
2021-06-04 16:51:11,125 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-04 16:51:11,126 image encoder trainable parameters: 20490144
2021-06-04 16:51:11,131 txt encoder trainable parameters: 137319072
2021-06-04 16:54:27,271 Epoch: [20][159/1133]	Eit 22800  lr 5e-05  Le 29.2343 (29.2337)	Time 1.215 (0.000)	Data 0.002 (0.000)	
2021-06-04 16:58:29,278 Epoch: [20][359/1133]	Eit 23000  lr 5e-05  Le 29.2230 (29.2322)	Time 1.165 (0.000)	Data 0.002 (0.000)	
2021-06-04 17:02:30,667 Epoch: [20][559/1133]	Eit 23200  lr 5e-05  Le 29.2275 (29.2325)	Time 1.110 (0.000)	Data 0.002 (0.000)	
2021-06-04 17:06:42,628 Epoch: [20][759/1133]	Eit 23400  lr 5e-05  Le 29.2512 (29.2326)	Time 1.138 (0.000)	Data 0.002 (0.000)	
2021-06-04 17:10:44,243 Epoch: [20][959/1133]	Eit 23600  lr 5e-05  Le 29.2150 (29.2323)	Time 1.150 (0.000)	Data 0.003 (0.000)	
2021-06-04 17:14:17,285 Test: [0/40]	Le 30.0713 (30.0712)	Time 3.949 (0.000)	
2021-06-04 17:14:37,996 calculate similarity time: 0.06194496154785156
2021-06-04 17:14:38,514 Image to text: 82.7, 96.3, 98.7, 1.0, 1.8
2021-06-04 17:14:38,827 Text to image: 63.2, 85.8, 91.1, 1.0, 7.3
2021-06-04 17:14:38,827 Current rsum is 517.8199999999999
2021-06-04 17:14:40,318 runs/f30k_butd_region_bert/log
2021-06-04 17:14:40,319 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-04 17:14:40,320 image encoder trainable parameters: 20490144
2021-06-04 17:14:40,325 txt encoder trainable parameters: 137319072
2021-06-04 17:15:18,865 Epoch: [21][27/1133]	Eit 23800  lr 5e-05  Le 29.1987 (29.2268)	Time 1.191 (0.000)	Data 0.002 (0.000)	
2021-06-04 17:19:25,182 Epoch: [21][227/1133]	Eit 24000  lr 5e-05  Le 29.2436 (29.2304)	Time 1.095 (0.000)	Data 0.002 (0.000)	
2021-06-04 17:23:26,057 Epoch: [21][427/1133]	Eit 24200  lr 5e-05  Le 29.2311 (29.2301)	Time 1.153 (0.000)	Data 0.002 (0.000)	
2021-06-04 17:27:26,595 Epoch: [21][627/1133]	Eit 24400  lr 5e-05  Le 29.2147 (29.2307)	Time 1.250 (0.000)	Data 0.002 (0.000)	
2021-06-04 17:31:29,123 Epoch: [21][827/1133]	Eit 24600  lr 5e-05  Le 29.2275 (29.2309)	Time 1.160 (0.000)	Data 0.002 (0.000)	
2021-06-04 17:35:32,529 Epoch: [21][1027/1133]	Eit 24800  lr 5e-05  Le 29.2306 (29.2308)	Time 1.315 (0.000)	Data 0.002 (0.000)	
2021-06-04 17:37:40,390 Test: [0/40]	Le 30.0730 (30.0729)	Time 4.140 (0.000)	
2021-06-04 17:38:00,761 calculate similarity time: 0.05881381034851074
2021-06-04 17:38:01,281 Image to text: 83.2, 96.1, 98.4, 1.0, 1.9
2021-06-04 17:38:01,726 Text to image: 63.1, 86.0, 91.4, 1.0, 7.2
2021-06-04 17:38:01,726 Current rsum is 518.22
2021-06-04 17:38:03,323 runs/f30k_butd_region_bert/log
2021-06-04 17:38:03,323 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-04 17:38:03,325 image encoder trainable parameters: 20490144
2021-06-04 17:38:03,330 txt encoder trainable parameters: 137319072
2021-06-04 17:40:04,904 Epoch: [22][95/1133]	Eit 25000  lr 5e-05  Le 29.2231 (29.2292)	Time 1.147 (0.000)	Data 0.002 (0.000)	
2021-06-04 17:44:08,752 Epoch: [22][295/1133]	Eit 25200  lr 5e-05  Le 29.2211 (29.2296)	Time 1.130 (0.000)	Data 0.002 (0.000)	
2021-06-04 17:48:08,854 Epoch: [22][495/1133]	Eit 25400  lr 5e-05  Le 29.2244 (29.2296)	Time 1.561 (0.000)	Data 0.002 (0.000)	
2021-06-04 17:52:09,636 Epoch: [22][695/1133]	Eit 25600  lr 5e-05  Le 29.2340 (29.2294)	Time 1.389 (0.000)	Data 0.002 (0.000)	
2021-06-04 17:56:09,289 Epoch: [22][895/1133]	Eit 25800  lr 5e-05  Le 29.2265 (29.2296)	Time 1.139 (0.000)	Data 0.002 (0.000)	
2021-06-04 18:00:07,933 Epoch: [22][1095/1133]	Eit 26000  lr 5e-05  Le 29.2304 (29.2298)	Time 1.135 (0.000)	Data 0.002 (0.000)	
2021-06-04 18:00:55,799 Test: [0/40]	Le 30.0716 (30.0716)	Time 4.065 (0.000)	
2021-06-04 18:01:16,185 calculate similarity time: 0.04932451248168945
2021-06-04 18:01:16,701 Image to text: 82.2, 96.0, 98.7, 1.0, 1.8
2021-06-04 18:01:17,059 Text to image: 62.9, 86.0, 91.0, 1.0, 7.4
2021-06-04 18:01:17,059 Current rsum is 516.76
2021-06-04 18:01:18,652 runs/f30k_butd_region_bert/log
2021-06-04 18:01:18,652 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-04 18:01:18,654 image encoder trainable parameters: 20490144
2021-06-04 18:01:18,660 txt encoder trainable parameters: 137319072
2021-06-04 18:04:42,630 Epoch: [23][163/1133]	Eit 26200  lr 5e-05  Le 29.2192 (29.2284)	Time 1.349 (0.000)	Data 0.002 (0.000)	
2021-06-04 18:08:46,195 Epoch: [23][363/1133]	Eit 26400  lr 5e-05  Le 29.2487 (29.2292)	Time 1.286 (0.000)	Data 0.002 (0.000)	
2021-06-04 18:12:47,363 Epoch: [23][563/1133]	Eit 26600  lr 5e-05  Le 29.2512 (29.2292)	Time 1.280 (0.000)	Data 0.003 (0.000)	
2021-06-04 18:16:48,554 Epoch: [23][763/1133]	Eit 26800  lr 5e-05  Le 29.2243 (29.2291)	Time 1.190 (0.000)	Data 0.002 (0.000)	
2021-06-04 18:20:49,305 Epoch: [23][963/1133]	Eit 27000  lr 5e-05  Le 29.2249 (29.2290)	Time 1.241 (0.000)	Data 0.002 (0.000)	
2021-06-04 18:24:11,963 Test: [0/40]	Le 30.0712 (30.0712)	Time 3.917 (0.000)	
2021-06-04 18:24:32,180 calculate similarity time: 0.05258488655090332
2021-06-04 18:24:32,705 Image to text: 81.9, 96.5, 98.7, 1.0, 1.8
2021-06-04 18:24:33,049 Text to image: 62.9, 85.9, 91.2, 1.0, 7.5
2021-06-04 18:24:33,049 Current rsum is 517.1800000000001
2021-06-04 18:24:34,420 runs/f30k_butd_region_bert/log
2021-06-04 18:24:34,421 runs/f30k_butd_region_bert
Use VSE++ objective.
2021-06-04 18:24:34,422 image encoder trainable parameters: 20490144
2021-06-04 18:24:34,427 txt encoder trainable parameters: 137319072
2021-06-04 18:25:15,704 Epoch: [24][31/1133]	Eit 27200  lr 5e-05  Le 29.2202 (29.2263)	Time 1.102 (0.000)	Data 0.002 (0.000)	
2021-06-04 18:29:14,076 Epoch: [24][231/1133]	Eit 27400  lr 5e-05  Le 29.2161 (29.2277)	Time 1.168 (0.000)	Data 0.002 (0.000)	
2021-06-04 18:33:11,866 Epoch: [24][431/1133]	Eit 27600  lr 5e-05  Le 29.2170 (29.2274)	Time 1.154 (0.000)	Data 0.002 (0.000)	
2021-06-04 18:37:10,792 Epoch: [24][631/1133]	Eit 27800  lr 5e-05  Le 29.2161 (29.2280)	Time 1.164 (0.000)	Data 0.003 (0.000)	
2021-06-04 18:41:10,781 Epoch: [24][831/1133]	Eit 28000  lr 5e-05  Le 29.2377 (29.2282)	Time 1.171 (0.000)	Data 0.002 (0.000)	
2021-06-04 18:45:10,068 Epoch: [24][1031/1133]	Eit 28200  lr 5e-05  Le 29.2244 (29.2280)	Time 1.198 (0.000)	Data 0.002 (0.000)	
2021-06-04 18:47:14,280 Test: [0/40]	Le 30.0715 (30.0715)	Time 3.891 (0.000)	
2021-06-04 18:47:34,769 calculate similarity time: 0.07794666290283203
2021-06-04 18:47:35,275 Image to text: 82.4, 95.9, 98.4, 1.0, 1.9
2021-06-04 18:47:35,588 Text to image: 63.5, 85.9, 91.1, 1.0, 7.4
2021-06-04 18:47:35,588 Current rsum is 517.24
[root@gpu1 vse_infty-master-my-graph-gru-unified-3loss]# CUDA_VISIBLE_DEVICES=0,1 python3 eval.py --dataset f30k --data_path ../data/f30k
INFO:root:Evaluating runs/f30k_butd_region_bert...
INFO:lib.evaluation:Namespace(backbone_lr_factor=0.01, backbone_path='', backbone_source='detector', backbone_warmup_epochs=5, batch_size=128, data_name='f30k', data_path='../data/f30k', embed_size=1024, embedding_warmup_epochs=2, grad_clip=2.0, img_dim=2048, input_scale_factor=1, learning_rate=0.0005, log_step=200, logger_name='runs/f30k_butd_region_bert/log', lr_update=15, margin=0.2, max_violation=True, model_name='runs/f30k_butd_region_bert', no_imgnorm=False, no_txtnorm=False, num_epochs=25, optim='adam', precomp_enc_type='basic', reset_start_epoch=False, resume='', val_step=500, vocab_path='./vocab/', vocab_size=30522, vse_mean_warmup_epochs=1, word_dim=300, workers=5)
INFO:transformers.tokenization_utils:loading file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt from cache at /root/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084
/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-unified-3loss/lib/encoders.py:48: UserWarning: nn.init.constant is now deprecated in favor of nn.init.constant_.
  nn.init.constant(self.W[1].weight, 0)
/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-unified-3loss/lib/encoders.py:49: UserWarning: nn.init.constant is now deprecated in favor of nn.init.constant_.
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
INFO:lib.evaluation:Test: [0/40]	Le 30.0686 (30.0685)	Time 5.818 (0.000)	
INFO:lib.evaluation:Test: [10/40]	Le 30.0718 (30.0680)	Time 0.554 (0.000)	
INFO:lib.evaluation:Test: [20/40]	Le 30.0659 (30.0726)	Time 0.514 (0.000)	
INFO:lib.evaluation:Test: [30/40]	Le 30.0752 (30.0721)	Time 0.765 (0.000)	
INFO:lib.evaluation:Images: 1000, Captions: 5000
INFO:lib.evaluation:calculate similarity time: 0.07845711708068848
INFO:lib.evaluation:rsum: 521.0
INFO:lib.evaluation:Average i2t Recall: 92.8
INFO:lib.evaluation:Image to text: 82.9 97.0 98.5 1.0 1.9
INFO:lib.evaluation:Average t2i Recall: 80.9
INFO:lib.evaluation:Text to image: 63.2 87.1 92.2 1.0 5.7
INFO:root:Evaluating runs/release_weights/f30k_butd_grid_bert...
Traceback (most recent call last):
  File "eval.py", line 58, in <module>
    main()
  File "eval.py", line 54, in main
    evaluation.evalrank(model_path, data_path=opt.data_path, split='test', fold5=False, save_path=save_path)
  File "/public/ZZX/SCAN_dataset/vse_infty-master-my-graph-gru-unified-3loss/lib/evaluation.py", line 196, in evalrank
    checkpoint = torch.load(model_path)
  File "/root/anaconda3/lib/python3.6/site-packages/torch/serialization.py", line 571, in load
    with _open_file_like(f, 'rb') as opened_file:
  File "/root/anaconda3/lib/python3.6/site-packages/torch/serialization.py", line 229, in _open_file_like
    return _open_file(name_or_buffer, mode)
  File "/root/anaconda3/lib/python3.6/site-packages/torch/serialization.py", line 210, in __init__
    super(_open_file, self).__init__(open(name, mode))
FileNotFoundError: [Errno 2] No such file or directory: 'runs/release_weights/f30k_butd_grid_bert/model_best.pth'

