# ディレクトリ構成
 
|ー　NumbersResult・・・過去の当選結果のスクレイピング結果が保存されるディレクトリ     
|ー　scraping_numbers4_result.py・・・過去の当選結果をスクレイピングするファイル  
|ー　numbers_predictor.py・・・学習とモデル評価を行うファイル
 
# 使い方
 
1.　scraping_numbers4_result.pyを実行する  
2.　numbers_predictor.pyを実行する
 
# 注意点
 
事前に空のNumbersResultフォルダを作成しておかないとスクレイピング結果が保存できない。
 
# Requirement

1.最新のGoogleChrome及びChromeDriverをインストールし、ルートディレクトリに配置しておく  
詳しくはこちらを参照:https://qiita.com/infra_yk/items/5336c8f526530d2ef17f

2.pip install -r requirements.txtを実施し、下記パッケージをインストールする 
* beautifulsoup4==4.9.3  
* lxml==4.6.1  
* numpy==1.19.3  
* pandas==1.1.3  
* python-dateutil==2.8.1  
* pytz==2020.1  
* selenium==3.141.0  
* six==1.15.0  
* soupsieve==2.0.1  
* urllib3==1.25.11  
