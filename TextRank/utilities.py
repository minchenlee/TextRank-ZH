import os
import math
import numpy as np

# CKIP 斷詞、詞性標註工具的初始化
from ckiptagger import data_utils, construct_dictionary, WS, POS, NER
ws = WS("./ckip_model_data")
pos = POS("./ckip_model_data")
ner = NER("./ckip_model_data")


# import 文本，回傳一串 string
def import_txt_file(folder_name, file_name):
    if '.txt' not in file_name:
        file_name += '.txt'
    
    path = os.getcwd() + '/' + folder_name + '/' + file_name
    
    with open(path, 'r') as f:
        document_text = f.read()
        
    return document_text


# 將文本進行斷句，回傳一個 list
def get_sentence_list(document_text):
    document_text = document_text.replace('\n', '。')
    document_text = document_text.replace('；', '。')
    sentence_list = document_text.split('。')
    sentence_list = list(filter(None, sentence_list))
    
    return sentence_list


# 使用 CKIP 進行斷詞和詞性標註，輸入一個 list，回傳兩個 list
def sentence_tokenizer(sentence_list):
    word_sentence_list = ws(sentence_list)
    word_sentence_list = list(filter(None, word_sentence_list))
    pos_sentence_list = pos(word_sentence_list)
    
    return word_sentence_list, pos_sentence_list


# 去除 stop word 和 punctuation，filter() 會用到
def is_stopword(word):
    stop_word_list = ['，', '（', '）', '「', '」', '、', '《', '》', '⋯⋯', '〈', '〉', '！','？', '：']
    
    for stop_word in stop_word_list:
        if word == stop_word:
            return True

    return False


# 移除 stop word 和 punctuation
def remove_stop_words(word_sentence_list, pos_sentence_list):
    for i, word_sentence in enumerate(word_sentence_list):
        pos_sentence = pos_sentence_list[i]
        del_token_list = list(filter(is_stopword, word_sentence))
        
        word_sentence, pos_sentence = zip(*((word_sentence, pos_sentence) 
                                            for word_sentence, pos_sentence in zip(word_sentence, pos_sentence)
                                            if word_sentence not in del_token_list))
        
        word_sentence_list[i] = list(word_sentence)
        pos_sentence_list[i] = list(pos_sentence)
        
    return word_sentence_list, pos_sentence_list


# indexing 每一個 term-frequency，然後存入另一個 dictionary，輸入一個 dict，回傳兩個 dict
def making_term_index_dict(dictionary):  
    """
    Parameters
    ----------
    dictionary : dict
        key: value = term: document_requency
    
    Returns
    -------
    index_dictionary : dict
        key: value = index: [term, document_frequency]
        
    index2term_dict : dict
        key: value = term: index
    """
    temp = dict()
    index2term_dict = dict()  # term 對應 index 的 dictionary
    
    term_count = 0
    for k, v in dictionary.items():
        term_count += 1
        temp[str(term_count)] = [k, v]
        index2term_dict[k] = term_count
        
        
    index_dictionary = temp
    return index_dictionary, index2term_dict


 # 將 term 按照 term 的字母順序或 index 值進行排序  
def sorting(dictionary, sortwith): 
    """
    Parameters
    ----------
    dictionary : dict
        key: value = term: count
    sortwith : int
        == 0: 按照字母順序排序
        == 1: 按照 index 值排序

    Returns
    -------
    dictionary : dict
        sorted 過後的 dictionary
    """
    if sortwith == 0:  # 按字母排序
        dictionary = {k: v for k, v in sorted(
            dictionary.items(), key=lambda item: item[0])}
    
    elif sortwith == 1:  # 按 index 值排序
        dictionary = {k: v for k, v in sorted(
            dictionary.items(), key=lambda item: int(item[0]))}
        
    return dictionary


# sentence (document) frequency 計算
# 會用到 sorting(), making_term_index_dict()
def document_frequency(TextRank_object):
    """ 
    Returns
    -------
    document_frequency_dict : dict
        key: value = index: [term, value]
    
    index2term_dict : dict
        key: value = term: index
    """
    term_sentence_posting = dict()  # 存 term t 在哪些 sentence 中出現過的 posting list
    sentence_frequency_dict = dict()  #  存 term t 在多少個 sentence 中出現過
    

    # 把 document 的 term 加入 collection 之中
    # intial_termize() 會回傳 preprocessing 好的 term list
    for sentence_id, word_sentence in enumerate(TextRank_object.word_sentence_list):
        for term in word_sentence:
            if term in term_sentence_posting:
                term_sentence_posting[term].append(sentence_id)

                # 轉換為 set 來消除重複出現的 term 
                term_sentence_posting[term] = set(term_sentence_posting[term])
                term_sentence_posting[term] = list(term_sentence_posting[term])

            elif term not in term_sentence_posting:
                # 如果 term 第一次出現，創造裝有 sentence_id 的 list
                term_sentence_posting[term] = [sentence_id]

            # 更新 sentence_frequency_dict 中 term 的 df 的數值。
            sentence_frequency_dict[term] = len(term_sentence_posting[term])

    # 排序
    sentence_frequency_dict = sorting(sentence_frequency_dict, 0)
    
    # 給予每一個 term index
    sentence_frequency_dict, index2term_dict = (
        making_term_index_dict(sentence_frequency_dict))
    
    return sentence_frequency_dict, index2term_dict


# 計算 TF-IDF 和 BM25 方法所需的 term frequency, idf value, sentence frequency, term2index 的 dict
def get_all_necessity(TextRank_object):
    """
    Parameters 
    ----------
    textrank object
    
    Returns
    -------
    sentence_tf_dict         存放所有 sentence 中每個 term 的 term frequency
    idf_dict                 存放整個 document 的 term 的 idf 值
    sentence_frequency_dict  存放 sentence frequency
    index2term_dict          存放 term to index 的 dict
    
    """
    sentence_tf_dict = dict()  # 用來存每一個 sentence 中每一個 term 的 term frequency
    idf_dict = dict()  # 用來存 document 中每一個 term 的 inverse document frequency
    sentence_frequency_dict, index2term_dict = document_frequency(TextRank_object)
    
    # 計算 term frequency 和 inverse document frequency
    for sentence_id, word_sentence in enumerate(TextRank_object.word_sentence_list):
        sentence_term_list = []
        term_count_dict = dict()  # 用來存每個 term 在每個 sentence 中的出現次數
        sentence_tf_dict_temp = dict()
        
        # 計算 sentence 中每一個的 term 出現次數
        for term in word_sentence:
            if term in term_count_dict:
                term_count_dict[term] += 1

            elif term not in term_count_dict:
                term_count_dict[term] = 1
            
            if term not in sentence_term_list:
                sentence_term_list.append(term)  # 紀錄整個 document 中不重複的 term
        
        # 計算 term frequency
        for term in word_sentence:
            tf = term_count_dict[term]/len(word_sentence)
            sentence_tf_dict_temp[term] = tf
        
        # 將 sentence 的 term frequency 存入 dict 中
        sentence_tf_dict[sentence_id] = sentence_tf_dict_temp
        
        # 計算 document 中每一個 term 的 idf
        for term in sentence_term_list:
            # 取得 term 的 index
            index = str(index2term_dict[term])

            # 計算 idf(但在這裡 document 其實是 sentence)
            df = sentence_frequency_dict[index][1]
            inverse_df = np.log10(TextRank_object.sentence_num/df)
            
            if inverse_df < 0: # why?
                break
            
            idf_dict[term] = inverse_df
            
    return sentence_tf_dict, idf_dict, sentence_frequency_dict, index2term_dict


# 計算每個 document 中的 term 的 TF-IDF, 會用到 get_sentence_tf_and_idf()
def TF_IDF(TextRank_object):
    sentence_tfidf_dict = dict()  # 用來存每一個 sentence 中每一個字的 tfidf 值
    
    for sentence_id, word_sentence in enumerate(TextRank_object.word_sentence_list):
        sentence_term_list = []
        tfidf_dict = dict() # tfidf_dict 用來存每個 sentence 中的 term 的 TF-IDF
        
        # 呼叫 get_sentence_tf_and_idf(TextRank_object) 取得相關資料
        sentence_tf_dict, idf_dict, sentence_frequency_dict, index2term_dict = get_all_necessity(TextRank_object)
        sentence_term_list = list(sentence_tf_dict[sentence_id].keys())
        
        # 計算 tf-idf
        for term in sentence_term_list:
            # 取得 term 的 index
            index = str(index2term_dict[term])

            # 計算 idf(但在這裡 document 其實是 sentence)
            df = sentence_frequency_dict[index][1]
            inverse_df = np.log10(TextRank_object.sentence_num/df)

            # 計算 tf
            term_frequency = sentence_tf_dict[sentence_id][term]

            # 計算 tf-idf
            tfidf = term_frequency*inverse_df
            tfidf_dict[index] = tfidf

            if inverse_df < 0:
                break
        
        # 將沒出現的 term 的 tf-idf 值設為 0
        for i in range(len(sentence_frequency_dict) + 1):
            if str(i) not in tfidf_dict:
                tfidf_dict[str(i)] = 0
            
        tfidf_dict = sorting(tfidf_dict, 1)
        sentence_tfidf_dict[sentence_id] = tfidf_dict
        
    return sentence_tfidf_dict

    
