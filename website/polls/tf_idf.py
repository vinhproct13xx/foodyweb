import numpy as np
import math

class tf_idf_cal:
    def lowercase_data(item):
        '''
        Word tokenize for all field in a item
        '''
        for key in item.keys():
            item[key] = str(item[key])
            item[key] = item[key].lower()
        return item

    def count_unigram(text):
        '''
        Đếm số lần xuất hiện mỗi từ trong một văn bản
        '''
        counter = {}
        words = text.split()
        vocabs = set(words)
        for vocab in vocabs:
            if not vocab.isdigit():
                counter[vocab] = words.count(vocab)
        return counter

    def count_word_in_dataset(items):
        '''
         Thống kê số lần xuất hiện từng từ trên toàn bộ dataset
        '''
        InputCounter = {}

        for item in items:
            for word in item['unigram_input'].keys():
                if word in InputCounter.keys():
                    InputCounter[word] += 1
                else:
                    InputCounter[word] = 1

        return InputCounter

    def tfidf(doc_len, corpus_len, doc_counter, corpus_counter, k=2):
        '''
        Công thức tính tf-idf cho 1 document, doc_counter là 1 biến
        lưu số lần suất hiện từng từ trong 1 document, corpus_counter
         là biến lưu số lần xuất hiện từng từ trong toàn bộ dataset.
          Kết quả trả về là 1 vector cho document.
        '''

        vector_len = len(corpus_counter)
        tfidf_vector = np.zeros((vector_len,))
        for i, key in enumerate(corpus_counter.keys()):
            if key in doc_counter.keys():
                tf = (k+1)*doc_counter[key]/(k+doc_counter[key])
                idf = math.log((corpus_len+1)/(corpus_counter[key]))
                tfidf_vector[i] = tf*idf
        return tfidf_vector
