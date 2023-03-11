import string
import pandas as pd
import json

with open("/Users/chuameiyun/Documents/2023 AI Projects/singlish-language-model/dataset/smsCorpus_en_2015.03.09_all.json") as f:
    msg = json.load(f)

text_messages = [i['text'] for i in msg['smsCorpus']['message']]

text = [m.get('$') for m in text_messages]
df = pd.DataFrame(text, columns=['messages'],  index=None)

def process(text):
    text = str(text).lower().replace('\n', ' ')\
          .replace('-', ' ').replace(':', ' ')\
          .replace(',', '').replace('"', '') \
          .replace("...", ".").replace("..", ".") \
          .replace("!", ".").replace("?", "") \
          .replace(";", ".").replace(":", " ")

    text = "".join(v for v in text if v not in string.punctuation).lower()
    text = text.encode("utf8").decode("ascii",'ignore')

    text = " ".join(text.split())
    return text

df.messages = df.messages.apply(process)

df.head(20)




