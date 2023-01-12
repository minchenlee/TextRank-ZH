# 引入套件
import os
import math
import pickle
import numpy as np
import networkx as nx

import utilities as ut
import similarity_compute as scomp

class TextRank():
    def __init__(self):
        # sentence extraction
        self.sentence_list = []
        self.word_sentence_list = []
        self.pos_sentence_list = []
        self.sentence_num = 0
        self.textrank_score_dict = dict()
        self.sentence_score_list = []
        self.sentence_matrix = []

        # tf-idf methodcds
        self.sentence_tfidf_dict = dict()

        # 詞性英文縮寫與中文轉換
        with open('pos_translation_dict.pickle', 'rb') as f:
            self.pos_translation_dict = pickle.load(f)


        # keyword extraction
        self.document_token_list = []
        self.encoded_token_list = []
        self.document_token_dict = dict()
        self.document_token_matrix = []
        self.token_score_dict = dict()
        self.token_score_list = []


    # 讀入文本
    def import_document(self, folder_name, file_name):
        self.sentence_list = ut.get_sentence_list(ut.import_txt_file(folder_name, file_name))
        self.sentence_num = len(self.sentence_list)
    

    # 顯示詞性分析
    def generate_word_pos_pairs(self):
        assert len(self.word_sentence_list) == len(self.pos_sentence_list)

        for sent_index, sentence in enumerate(self.word_sentence_list):
            for word_index, word in enumerate(sentence):
                pos = self.pos_sentence_list[sent_index][word_index]
                pos_ch = self.pos_translation_dict[pos]

                print(f'{word}({pos})({pos_ch})')

    
    # 斷句和預處理
    def tokenize_and_preprocessing(self, pos_type_list):
        word_sentence_list, pos_sentence_list = ut.sentence_tokenizer(self.sentence_list)
        self.word_sentence_list, self.pos_sentence_list = \
            ut.run_pre_processing(word_sentence_list, pos_sentence_list, pos_type_list)


    # for sentence extraction 分析，進行預處理、建立 graph、跑 PageRank
    def analyze(self, method = 'overlap', pos_type_list = [], similarity_threshold = 0):

        # 進行斷句和預處理
        self.tokenize_and_preprocessing(pos_type_list)
        
        # 建立空的 Graph Matrix
        n = self.sentence_num
        self.sentence_matrix = np.zeros((n,n))

        if method == 'tfidf':
            self.sentence_tfidf_dict = ut.TF_IDF(self)
        
        # 填值進入 Graph Matrix
        for index_1, sentence_1 in enumerate(self.word_sentence_list):
            for index_2, sentence_2 in enumerate(self.word_sentence_list):

                self.sentence_matrix[index_1, index_2] = scomp.compute_similarity(self, sentence_1, sentence_2, 
                                                                           index_1, index_2, method)

                if index_1 == index_2:
                    self.sentence_matrix[index_1, index_2] = 0

                if self.sentence_matrix[index_1, index_2] < similarity_threshold:
                    self.sentence_matrix[index_1, index_2] = 0
                    
        # 用 networkx 套件跑 PageRank
        G = nx.DiGraph(self.sentence_matrix)
        result = (nx.pagerank(G, alpha=0.85))
        result = dict(sorted(result.items(), key=lambda item: item[1], reverse = True))
        self.textrank_score_dict = result

        # 重置 sentence_score_list
        self.sentence_score_list = []

        for index, value in self.textrank_score_dict.items():
            self.sentence_score_list.append([value, self.sentence_list[index]])


    # for keyword extraction
    def keyword_analyze(self, window_length = 5, pos_type_list = []):
        # 進行斷句和預處理
        self.tokenize_and_preprocessing(pos_type_list)

        # 合併每一個 sentence 的 word list。
        self.document_token_list = []
        for sentence in self.word_sentence_list:
            self.document_token_list += sentence

        # 建立 document_token_dict。
        for index, token in enumerate(list(set(self.document_token_list))):
            self.document_token_dict[token] = index
            self.document_token_dict[index] = token


        # 建立編碼後的 document token list
        for token in self.document_token_list:
            self.encoded_token_list.append(self.document_token_dict[token])

        # 建立空的 Graph Matrix
        n = int(len(self.document_token_dict)/2)
        self.document_token_matrix = np.zeros((n,n))

        # 滑動 window 經過整個 merge list，看是否有相鄰。
        for i in range(len(self.encoded_token_list)):
            window_head = i
            window_tail = i + window_length
            
            window = self.encoded_token_list[window_head: window_tail]
            if len(window) < window_length:
                break
            
            # 填值進入到 matrix 中
            for token_1 in enumerate(window):
                for token_2 in enumerate(window):
                    self.document_token_matrix[token_1, token_2] = 1
                    
                    if token_1 == token_2:
                        self.document_token_matrix[token_1, token_2] = 0


        # 跑 page rank
        G = nx.DiGraph(self.document_token_matrix)
        result = (nx.pagerank(G, alpha=0.85))
        result = dict(sorted(result.items(), key=lambda item: item[1], reverse = True))
        self.token_score_dict = result

        self.token_score_list = []
        for index, value in result.items():
            self.token_score_list.append([value, self.document_token_dict[index]])

    
    def print_top_n_result(self, result_type, num: int):

        if result_type == 'sentence':
            count = 0
            for index, value in self.textrank_score_dict.items():
                if count >= num:
                    break

                print(value, self.sentence_list[index])
                print()
                count += 1

        if result_type == 'keyword':
            count = 0
            for index, value in self.token_score_dict.items():
                if count >= num:
                    break

                print(value, self.document_token_dict[index])
                print()
                count += 1


    
    def lazy_start(self, folder_name, file_name, num: int, method = 'overlap'):
        self.import_document(folder_name, file_name)
        self.analyze(method)
        self.print_top_n_result(num)
