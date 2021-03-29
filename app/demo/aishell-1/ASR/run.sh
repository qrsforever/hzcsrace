python3 train.py hparams/train.yaml \
	--output_folder /data/tmp/sb/aishell-1 \
	--data_folder /data/datasets/asr/AISHELL-1/ \
	--data_folder_rirs /data/datasets/asr/noises/ \
	--tokenizer_file /data/pretrained/asr/aishell-1/tokenizer/5000_unigram.model
