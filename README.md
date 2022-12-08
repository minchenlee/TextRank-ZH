# TextRank
這個程式實作了 TextRank 對中文文本摘要的演算法，**並不能**做關鍵字抽取，斷詞的部分使用 CKIP-tagger，而 PageRank 計算的部分則使用 NetworkX 來達成。  
句子間相似度的計算除了 TextRank 原本的 term overlap 方法，還有 TF-IDF cosine similarity 和 BM25 共三種方法可以使用。


## dependency
numpy >= 1.7.1  
networkx == 1.9.1  
scipy == 1.8.0  
ckiptagger == 0.2.1    

ckiptagger 需要 tensorflow>=1.13.1，**建議重建一個虛擬環境**，建議安裝方式如下：  
1. 先安裝 tensorflow：https://www.tensorflow.org/install
2. 再安裝 ckiptagger 的最小版本：
```
$ pip install -U ckiptagger
```
3. ckiptagger 所需的 model data 由於檔案太大需另外下載，可從[這裡](https://drive.google.com/drive/folders/105IKCb88evUyLKlLondvDBoh7Dy_I1tm)下載，解壓後將 /data 底下的所有檔案放入 [/ckip_model_data](TextRank/ckip_model_data) 即可。
4. 請確認 scipy == 1.8.0，版本錯誤的話 ckiptagger 會一直報錯。

## compatibility
Python 3.9.12 ✅

## 範例  
詳見 [demo.py](TextRank/demo.py)、[demo.ipynb](TextRank/demo.ipynb)。

demo.py:

```python
# 創建 TextRank object
tr = TextRank()

# 輸入要分析的文本
tr.import_document(folder_name = 'text', file_name = 'bnext_news_72945')

# 分析
# method: 有 'overlap', 'tfidf', 'bm25' 三種，不指定的話預設使用 'overlap'，也就是 TextRank 的原始方法。
tr.analyze(method = 'overlap')  

# 輸出結果
tr.print_top_n_result(num = 10)
```

輸出結果如下：
```
0.047042273000621596 FCC主席Jessica Rosenworcel指出，這些廠商的電信與監控產品會對國家與國民安全構成不可接受（unacceptable）的風險，不過FCC實際上並未全面禁止所有產品，如果海能達、海康威視與大華科技三間公司能夠證明，他們旗下用於公共安全、政府設施安全、重要實體監管以及其他國家安全的設備，是銷售給一般消費者，或具有嚴格的安全機制，有可能仍會獲得FCC的許可

0.045434314287353096 11/25，美國聯邦傳播委員會（Federal Communications Commission，FCC）以國家安全風險為由，宣布禁止華為（Huawei）、中興（ZTE）與海能達（Hytera）、海康威視（Hikvision）與大華科技（Dahua）等中國大廠在美販售或進口的通訊與監控設備

0.042824866284834 同年11月，美國總統拜登簽署《安全設備法》（Secure Equipment Act），要求FCC不可審核或是同意授權任何構成國安風險的設備申請，尤其針對名單上的華為、中興等中國廠商，此法不僅可以解決先前非聯邦資金採購設備的問題，亦被外界視為重挫中國電信與科技大廠的重大立法案

0.03968425884102916 同時，FCC專員Carr也呼籲政府處理「華為漏洞」，並指出先前已被替換的網路設備都是由聯邦出資購買，因此仍有企業鑽漏洞，私下購買一樣不安全的產品來使用，而這樣的行為還是會導致風險發生

0.038034638305993275 綜合上述例子，可見由美、英、加、澳、紐五國組成的「五眼聯盟（Five Eyes Alliance，FVEY）」皆出於安全與風險考量，紛紛提出措施防備中國公司的電信公司、設備，而相繼出手的大國可能都會對華為、中興等廠商造成不少影響

0.037703583796656746 雖然過去FCC推出黑名單、點明哪些廠商的產品有風險，還禁止企業使用聯邦資金購買這些設備，但是在過去幾年中，FCC依然繼續許可這些設備在市面上販售，只要這類商品獲得授權章，廠商就能持續進貨到美國，賣給其他自掏腰包的買家

0.03700768816139613 FCC的新管制政策啟動後，華為、中興、海能達、海康威視與大華科技將無法在美進口與販售其電信設備、影像監視器，預期將讓銷售情況雪上加霜，並且連帶波及到這些廠商背後供應商， 如台灣的晶相光、矽力–KY等 

0.03628856244152909 作為進一步打擊中國電信與科技公司的行動，FFC近日表示不再授權黑名單（covered list）內的公司在美販售或進口其設備，華為、中興、海能達、海康威視（Hikvision）與大華科技（Dahua）均列名其中

0.03615397287203028 英國近日也考量安全風險，仿效美國採取類似措施，禁止各部門在「敏感（Sensitive）」的政府網站上安裝中國公司出產的監控設備

0.03573004947011017 根據Reuters報導，海康威視聲稱自己的產品不會危及美國國安，並指出對想要保護自己、家庭或財產的美國小公司、地方政府、學校單位與消費者來說，FCC的決定只會造成負擔，而不會真正保護到國家安全
```

## 使用方法
### 方法一
[範例](#範例)中使用的便是方法一。

### 方法二
```python
# 創建 TextRank object
tr = TextRank()

# 一行解決方案，一次完成第一種方法的所有功能。
tr.lazy_start(folder_name = 'text', file_name = 'bnext_news_72945', num = 10, method = 'overlap')
```
lazy_start() 可以一行直接達成文本的輸入、分析和結果輸出。而如果想要輸出更多結果可以用：
```python
# 輸出更多的句子數
tr.print_top_n_result(15)
```
### 取得 sentence score
如果要直接取得每個句子的分數，可以用：
```python
tr.sentence_score_list
```
會返還一個包含所有句子和其分數的 list，按照分數遞減，部分輸出結果如下：
```
[[0.047042273000621596,
  'FCC主席Jessica Rosenworcel指出，這些廠商的電信與監控產品會對國家與國民安全構成不可接受（unacceptable）的風險，不過FCC實際上並未全面禁止所有產品，如果海能達、海康威視與大華科技三間公司能夠證明，他們旗下用於公共安全、政府設施安全、重要實體監管以及其他國家安全的設備，是銷售給一般消費者，或具有嚴格的安全機制，有可能仍會獲得FCC的許可'],
 [0.045434314287353096,
  '11/25，美國聯邦傳播委員會（Federal Communications Commission，FCC）以國家安全風險為由，宣布禁止華為（Huawei）、中興（ZTE）與海能達（Hytera）、海康威視（Hikvision）與大華科技（Dahua）等中國大廠在美販售或進口的通訊與監控設備'],
 [0.042824866284834,
  '同年11月，美國總統拜登簽署《安全設備法》（Secure Equipment Act），要求FCC不可審核或是同意授權任何構成國安風險的設備申請，尤其針對名單上的華為、中興等中國廠商，此法不僅可以解決先前非聯邦資金採購設備的問題，亦被外界視為重挫中國電信與科技大廠的重大立法案'],
  ...
 [0.011903209468160434, '不只有美國！英、加、紐澳也出手制裁'],
 [0.01103909625035826, '從川普到拜登，FCC一步步打擊中國大廠'],
 [0.008993094340096087, '所以要真正杜絕這個問題，必須要從禁止授權開始'],
 [0.00880625614551769, '管制啟動後，哪些廠商會受影響？哪些廠商受惠？']]
```

## 細節
### file
|                      file name                          |              descrption              |
|---------------------------------------------------------|--------------------------------------|
|[demo.py](demo.py)                                       |為範例檔案                             |
|[demo.ipynb](demo.ipynb)                                 |為範例檔案                             |
|[similarity_compute.py](similarity_compute.py)           |存放計算 similarity 的 funtion         |
|[TextRank_sentence_summa.py](TextRank_sentence_summa.py) |存放 TextRank 這個 class               |
|[utilities.py](utilities.py)                             |存放檔案讀取、斷句、document frequency 計算、term frequency 計算、idf value 計算、stop word remove、TF-IDF 計算等 function。|
|[text](./text)                                           |存放要被摘要的文檔                       |
|[ckip_model_data](./ckip_model_data)                     |存放 ckip-tagger 所需的 model data     |


### About TextRank() class
#### attribute
```python
self.sentence_list = []             # 存有每一個句子的 list
self.word_sentence_list = []        # 存有每一個句子斷詞完的 term list
self.pos_sentence_list = []         # 存有每一個句子斷詞完的 term 的詞性的 list
self.sentence_num = 0               # 文檔共有的句子總數
self.textrank_score_dict = dict()   # 存有每個句子的 id 和 text rank 分數的 dict，{sentence_id: score}
self.sentence_score_list = []       # 存有每個句子的文字內容和 text rank 分數的 list，[score, sentence_text]

# tf-idf method
self.sentence_tfidf_dict = dict()   # 存有每個句子的 tf-idf value 的 dict
```

### Todo
加入詞性過濾功能。






