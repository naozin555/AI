from bs4 import BeautifulSoup
import urllib.request
from selenium import webdriver
import time
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd

headers = {
    'Uset-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/69.0.3497.100 Safari/537.36'
}

today = datetime.today()
count = 1

cols = ['回別', '抽選日', '当選数字', 'ストレート口数', 'ストレート賞金額', 'ボックス口数', 'ボックス賞金額',
        'セットのストレート口数', 'セットのストレート賞金額', 'セットのボックス口数', 'セットのボックス賞金額',
        'ミニ口数', 'ミニ賞金額', '販売実績額']
df = pd.DataFrame(index=[], columns=cols)

while True:
    # Numbers3の場合
    # base_url = 'https://www.mizuhobank.co.jp/retail/takarakuji/numbers/numbers3/index.html?year={0}&month={1}'

    # Numbers4の場合
    base_url = 'https://www.mizuhobank.co.jp/retail/takarakuji/numbers/numbers4/index.html?year={0}&month={1}'
    previous_month = today - relativedelta(months=count)
    year = datetime.strftime(previous_month, '%Y')
    month = datetime.strftime(previous_month, '%m')

    url = base_url.format(year, month)
    print(url)

    driver = webdriver.Chrome(executable_path="./chromedriver")
    driver.get(url)
    time.sleep(1)
    html = driver.page_source
    driver.quit()

    req = urllib.request.Request(url, headers=headers)
    soup = BeautifulSoup(html, features="lxml")
    tables = soup.findAll(attrs={"class": "typeTK"})

    for table in tables:
        trs = table.findAll('tr')
        kai = trs[0].findAll('th')[1].text
        date = trs[1].find('td').text
        hit = trs[2].find('td').find('strong').text
        st = trs[3].findAll('td')[0].text
        st_price = trs[3].findAll('td')[1].find('strong').text
        box = trs[4].findAll('td')[0].text
        box_price = trs[4].findAll('td')[1].find('strong').text
        set_st = trs[5].findAll('td')[0].text
        set_st_price = trs[5].findAll('td')[1].find('strong').text
        set_box = trs[6].findAll('td')[0].text
        set_box_price = trs[6].findAll('td')[1].find('strong').text
        all_price = trs[7].find('td').text

        # Numbers3の場合
        # mini = trs[7].findAll('td')[0].text
        # mini_price = trs[7].findAll('td')[1].find('strong').text
        # all_price = trs[8].find('td').text

        # Numbers3の場合
        # df = df.append({'回別': kai, '抽選日': date, '当選数字': hit, 'ストレート口数': st, 'ストレート賞金額': st_price,
        #                 'ボックス口数': box, 'ボックス賞金額': box_price,  'セットのストレート口数': set_st,
        #                 'セットのストレート賞金額': set_st_price, 'セットのボックス口数': set_box,
        #                 'セットのボックス賞金額': set_box_price, 'ミニ口数': mini, 'ミニ賞金額': mini_price,
        #                 '販売実績額': all_price}, ignore_index=True)

        # Numbers4の場合
        df = df.append({'回別': kai, '抽選日': date, '当選数字': hit, 'ストレート口数': st, 'ストレート賞金額': st_price,
                        'ボックス口数': box, 'ボックス賞金額': box_price, 'セットのストレート口数': set_st,
                        'セットのストレート賞金額': set_st_price, 'セットのボックス口数': set_box,
                        'セットのボックス賞金額': set_box_price, '販売実績額': all_price}, ignore_index=True)
        # print(kai, date, hit,
        #       st, st_price,
        #       box, box_price,
        #       set_st, set_st_price,
        #       set_box, set_box_price,
        #       mini, mini_price,
        #       all_price)
    count = count + 1
    if count > 12:
        break

df.set_index("抽選日", inplace=True)
pd.set_option('display.max_columns', 300)
print(df)
df.to_csv("NumbersResult/Numbers4_result.csv")
