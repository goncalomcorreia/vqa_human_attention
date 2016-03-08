# data folder

This is the foler to put your data. Refer to
```src/data_provision_att_vqa.py``` to get the format of the data.

```
trainval_feat.h5: a sparse matrix for image features. Refer to load_image_feat function in data_provision_att_vqa.py
train/val1/val2.pkl: data splits pickle files. The pickle file consists of several lists:
    question_id:
    image_id: the index used to get image feature from trainval_feat.h5
    question: the index of quesiton words
    answer,
    answer_counter,
    answer_label
```
