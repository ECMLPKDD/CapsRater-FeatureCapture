**CapsRater + FeatureCapture Model**

Install Python 3.6

Install packages in requirements.txt

Launch BERT server using command:
```
 bert-serving-start -model_dir models/uncased_L-12_H-768_A-12/ -num_worker=5 -port 8190 -max_seq_len=NONE
```
 
Quickstart
```
 python ./main.py
```
