from TextRank_sentence_summa import TextRank

# 兩種使用方法

# 第一種
# 創建 TextRank object
tr = TextRank()

# 輸入要分析的文本
tr.import_document(folder_name = 'text', file_name = 'bnext_news_72945')

# 分析
# method: 有 'overlap', 'tfidf', 'bm25' 三種，不指定的話預設使用 'overlap'，也就是 TextRank 的原始方法。
tr.analyze(method = 'overlap')  

# 輸出結果
tr.print_top_n_result(num = 10)


# 第二種
# 創建 TextRank object
tr = TextRank()

# 一行解決方案，一次完成第一種方法的所有功能。
tr.lazy_start(folder_name = 'text', file_name = 'bnext_news_72945', num = 10, method = 'overlap')

# 輸出更多的句子數
tr.print_top_n_result(15)

# 取得 sentence score
tr.sentence_score_list
