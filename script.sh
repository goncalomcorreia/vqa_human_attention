nice python vqa_human_attention/src/scripts/hsan_deepfix.py --changes "'model_name'":"'no_supervision_baseline'" "'lambda'":"0."
nice python vqa_human_attention/src/scripts/tuning/hsan_no_deepfix.py
nice python vqa_human_attention/src/scripts/tuning/hsan_deepfix_par.py
nice python vqa_human_attention/src/scripts/hsan_deepfix.py --changes "'model_name'":"'hsan_deepfix'" "'lambda'":"0.2"
