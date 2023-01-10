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
        self.sentence_list = []
        self.word_sentence_list = []
        self.pos_sentence_list = []
        self.sentence_num = 0
        self.textrank_score_dict = dict()
        self.sentence_score_list = []

        # tf-idf method
        self.sentence_tfidf_dict = dict()

        # 詞性英文縮寫與中文轉換
        with open('pos_translation_dict.pickle', 'rb') as f:
            self.pos_translation_dict = pickle.load(f)


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
                    
    
    # 分析，進行預處理、建立 graph、跑 PageRank
    def analyze(self, method = 'overlap', pos_type_list = []):
        word_sentence_list, pos_sentence_list = ut.sentence_tokenizer(self.sentence_list)
        self.word_sentence_list, self.pos_sentence_list = \
            ut.run_pre_processing(word_sentence_list, pos_sentence_list, pos_type_list)
        
        
        # 建立空的 Graph Matrix
        n = self.sentence_num
        sentence_matrix = np.zeros((n,n))
        
        if method == 'overlap':
            pass
            
        elif method == 'tfidf':
            self.sentence_tfidf_dict = ut.TF_IDF(self)
            
        elif method == 'bm25':
            pass
        
        
        # 填值進入 Graph Matrix
        for index_1, sentence_1 in enumerate(word_sentence_list):
            for index_2, sentence_2 in enumerate(word_sentence_list):
                if index_1 == index_2:
                    sentence_matrix[index_1, index_2] = 0

                else:
                    sentence_matrix[index_1, index_2] = scomp.compute_similarity(self, sentence_1, sentence_2, 
                                                                           index_1, index_2, method)
                    
        # 用 networkx 套件跑 PageRank
        G = nx.DiGraph(sentence_matrix)
        result = (nx.pagerank(G, alpha=0.85))
        result = dict(sorted(result.items(), key=lambda item: item[1], reverse = True))
        self.textrank_score_dict = result

        for index, value in self.textrank_score_dict.items():
            self.sentence_score_list.append([value, self.sentence_list[index]])

    
    def print_top_n_result(self, num: int):
        count = 0
        for index, value in self.textrank_score_dict.items():
            if count >= num:
                break

            print(value, self.sentence_list[index])
            print()
            count += 1

    
    def lazy_start(self, folder_name, file_name, num: int, method = 'overlap'):
        self.import_document(folder_name, file_name)
        self.analyze(method)
        self.print_top_n_result(num)
