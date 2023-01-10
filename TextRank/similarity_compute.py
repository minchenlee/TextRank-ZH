import math
import numpy as np
import utilities as ut

# 計算相似度
def compute_similarity(TextRank_object, sentence_1, sentence_2, index_1, index_2, method):
    # textrank overlap 
    if method == 'overlap':
        words_set = list(set(sentence_1 + sentence_2))

        sen_1_vec = [sentence_1.count(word) for word in words_set]
        sen_2_vec = [sentence_2.count(word) for word in words_set]

        common_vec = [sen_1_vec[i]*sen_2_vec[i] for i in range(len(sen_1_vec))]

        if len(common_vec) == 0:
            return 0

        nume = sum([1 for num in common_vec if num > 0.])
        deno = math.log(len(sen_1_vec)) + math.log(len(sen_2_vec))

        if deno == 0:
            return 0

        sim_value = nume/deno
    
    # tf-idf cosine similarity
    elif method == 'tfidf':
        tfidf_dict = TextRank_object.sentence_tfidf_dict
        
        vec_x = np.array(list(tfidf_dict[index_1].values()))
        vec_y = np.array(list(tfidf_dict[index_2].values()))
        cos_sim = np.dot(vec_x, vec_y) / (np.linalg.norm(vec_x) * np.linalg.norm(vec_y))
        sim_value = cos_sim
        
    # BM25
    elif method == 'bm25':
        sentence_tf_dict, idf_dict, sentence_frequency_dict, index2term_dict = ut.get_all_necessity(TextRank_object)
        
        k1 = 1.2
        k2 = 2
        b = 0.75

        sentence_length_sum = 0
        for i, sentence in enumerate(TextRank_object.word_sentence_list):
            sentence_length_sum += len(sentence)
        
        avg_sentence_length = sentence_length_sum/TextRank_object.sentence_num
        
        sim_value = 0
        for term in list(set(sentence_1)):
            idf_value = idf_dict[term]
            
            # query term frequency
            tf_value_1 = sentence_tf_dict[index_1][term]  
            
            # document(sentence) term frequency
            tf_value_2 = sentence_tf_dict[index_2][term] if term in sentence_tf_dict[index_2] else 0
            sentence_length = len(list(sentence_tf_dict[index_2].keys()))
            
            K1 = (k1 + 1) * tf_value_2 / k1 * ((1-b) + b * (sentence_length/avg_sentence_length)) + tf_value_2
            K2 = (k2 + 1) * tf_value_1 / k2 + tf_value_1
            
            sim_value += idf_value * K1 * K2
        
    return sim_value

