# ディレクトリ構成
 
|ー　leukemia・・・画像が格納してある  
|　　　|ー　LK172 pres 45 10 34 19 p65_6_focused _ R6 _ R7 _ normal cells _ cd 19 pos _ R2 _ R1_minh   
|　　　|ー　LK172 pres 45 10 34 19 p65_6_R5 _ R7 _ R10 _ blasts _ cd 19 pos _ R2 _ R1_minh  
|　　　|ー　LK172 pres 45 10 34 19 p65_6_R5 _ R7 _ R10 _ blasts_highSSC_granulocytes _ cd 19 pos _ R2 _ R1_minh  
|　　　|ー　LK172 pres 45 10 34 19 p65_6_R5 _ R7 _ R10 _ blasts_highSSC_middle_ugly _ cd 19 pos _ R2 _ R1    
|　　　|ー　LK172 pres 45 10 34 19 p65_6_R5 _ R7 _ R10 _ blasts_highSSC_upper_dead _ cd 19 pos _ R2 _ R1_minh   
|ー　result・・・学習状況、混合行列が保存される  
|ー　generate_data_xxx.py・・・学習用データ、テストデータを作成するファイル  
|ー　leukemia_xxx.py・・・学習とモデル評価を行うファイル
 
# 使い方
 
1.　generate_data_xxx.pyを実行する  
2.　leukemia_xxx.pyを実行する
 
# 注意点
 
事前に空のresultフォルダを作成しておかないと結果が保存できない。
 
# Requirement
 
pip freezeの結果を列挙する
 
* absl-py==0.10.0
* astunparse==1.6.3
* cachetools==4.1.1
* certifi==2020.6.20
* chardet==3.0.4
* gast==0.3.3
* google-auth==1.22.1
* google-auth-oauthlib==0.4.1
* google-pasta==0.2.0
* grpcio==1.33.1
* h5py==2.10.0
* idna==2.10
* Keras==2.4.3
* Keras-Preprocessing==1.1.2
* Markdown==3.3.2
* numpy==1.18.5
* oauthlib==3.1.0
* opt-einsum==3.3.0
* protobuf==3.13.0
* pyasn1==0.4.8
* pyasn1-modules==0.2.8
* PyYAML==5.3.1
* requests==2.24.0
* requests-oauthlib==1.3.0
* rsa==4.6
* scipy==1.5.3
* six==1.15.0
* tensorboard==2.3.0
* tensorboard-plugin-wit==1.7.0
* tensorflow==2.3.1
* tensorflow-estimator==2.3.0
* termcolor==1.1.0
* urllib3==1.25.11
* Werkzeug==1.0.1
* wrapt==1.12.1
