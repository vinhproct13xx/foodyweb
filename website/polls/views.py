import django
from django.shortcuts import render
from django.http import HttpResponse
from .models import Customers
from .models import Comments
from .models import Similarity
from .models import SearchTrain
from .tf_idf import tf_idf_cal
from django.shortcuts import get_object_or_404
import os
import pandas as pd
import psycopg2
import sqlite3
import logging
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import coo_matrix
from datetime import datetime
from website import settings
from tqdm import tqdm
import sklearn
from sklearn.decomposition import TruncatedSVD

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "website.settings")
django.setup()

def go(request):
    search_trains = SearchTrain.objects.values('id','input')
    query = 'gần đây có cửa hàng nào gần nhất không'
    query = query.lower()
    newItems = []
    for item in tqdm(search_trains):
        newItem = tf_idf_cal.lowercase_data(item)
        newItem['unigram_input'] = tf_idf_cal.count_unigram(newItem['input'])
        newItems.append(newItem)
    newItems.append({'id':'query','input': query, 'unigram_input':tf_idf_cal.count_unigram(query)})
    # Tính vector cho description_document cho từng sản phẩm
    inputCounter = tf_idf_cal.count_word_in_dataset(newItems)
    tfidf_vectors = []
    corpus_len = len(newItems)
    for item in tqdm(newItems):
        doc_len = len(item['input'])
        tfidf_vectors.append(
            tf_idf_cal.tfidf(doc_len, corpus_len, item['unigram_input'], inputCounter)
        )
    query_vector = tfidf_vectors[len(tfidf_vectors)-1]
    query_vector = np.reshape(query_vector, (1,-1))
    # search
    sim_maxtrix = sklearn.metrics.pairwise.cosine_similarity(query_vector, tfidf_vectors)
    sim_maxtrix = np.reshape(sim_maxtrix, (-1,))

    idx = (-sim_maxtrix).argsort()[:20]
    for _id in idx:
        print(_id, sim_maxtrix[_id])
        print(newItems[_id]['id'].upper())

    # return tfidf_vectors


def index(request):
    latest_customer_list = Customers.objects.order_by('id')[:10]
    context = {'latest_customer_list': latest_customer_list}
    return render(request, 'polls/index.html', context)


# Create your views here.

def detail(request, question_id):
    custormer = get_object_or_404(Customers, pk=question_id)
    return render(request, 'polls.detail.html', {'customer': custormer})


def results(request, question_id):
    response = "You're looking at the question %s."
    return HttpResponse(response % question_id)


def vote(request, question_id):
    return HttpResponse("You're voting on question %s." % question_id)


def voice(request):
    return render(request, 'polls/test.html', None)


def test(request):
    return render(request, 'polls/test.html', None)


def runalgorithm(request):
    all_ratings = load_all_ratings()
    ItemSimilarityMatrixBuilder(min_overlap=10, min_sim=0.0).build(all_ratings)
    return HttpResponse("Done.")


def load_all_ratings(min_rating=1):
    columns = ['owner', 'resid', 'avgrating']

    ratings_data = Comments.objects.all().values('owner', 'resid', 'avgrating')

    ratings = pd.DataFrame.from_records(ratings_data, columns=columns)
    ctm_count = ratings[['owner', 'resid']].groupby('owner').count()
    ctm_count = ctm_count.reset_index()
    owners = ctm_count[ctm_count['resid'] > min_rating]['owner']
    ratings = ratings[ratings['owner'].isin(owners)]
    ratings['avgrating'] = ratings['avgrating'].astype(float)
    return ratings


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger('Item simialarity calculator')


class ItemSimilarityMatrixBuilder(object):

    def __init__(self, min_overlap=2, min_sim=0.2):
        self.min_overlap = min_overlap
        self.min_sim = min_sim
        self.db = settings.DATABASES['default']['ENGINE']

    def build(self, ratings, save=True):
        logger.debug("Calculating similarities ... using {} ratings".format(len(ratings)))
        start_time = datetime.now()

        logger.debug("Creating ratings matrix")

        ratings['avgrating'] = ratings['avgrating'].astype(float)
        ratings['avg'] = ratings['avgrating'].transform(lambda x: normalize(x))

        ratings['avg'] = ratings['avg'].astype(float)
        ratings['owner'] = ratings['owner'].astype('category')
        ratings['resid'] = ratings['resid'].astype('category')

        coo = coo_matrix((ratings['avg'].astype(float),
                          (ratings['resid'].cat.codes.copy(),
                           ratings['owner'].cat.codes.copy())))

        logger.debug("Calculating overlaps between the items")
        overlap_matrix = coo.astype(bool).astype(int).dot(coo.transpose().astype(bool).astype(int))

        number_of_overlaps = (overlap_matrix > self.min_overlap).count_nonzero()
        logger.debug("Overlap matrix leaves {} out of {} with {}".format(number_of_overlaps,
                                                                         overlap_matrix.count_nonzero(),
                                                                         self.min_overlap))

        logger.debug("Rating matrix (size {}x{}) finished, in {} seconds".format(coo.shape[0],
                                                                                 coo.shape[1],
                                                                                 datetime.now() - start_time))

        sparsity_level = 1 - (ratings.shape[0] / (coo.shape[0] * coo.shape[1]))
        logger.debug("Sparsity level is {}".format(sparsity_level))

        start_time = datetime.now()
        cor = cosine_similarity(coo, dense_output=False)
        # cor = rp.corr(method='pearson', min_periods=self.min_overlap)
        # cor = (cosine(rp.T))

        cor = cor.multiply(cor > self.min_sim)
        cor = cor.multiply(overlap_matrix > self.min_overlap)

        res = dict(enumerate(ratings['resid'].cat.categories))

        logger.debug('Correlation is finished, done in {} seconds'.format(datetime.now() - start_time))
        if save:

            start_time = datetime.now()
            logger.debug('save starting')

            self._save_with_django(cor, res)

            logger.debug('save finished, done in {} seconds'.format(datetime.now() - start_time))
        return cor, res

    def _save_similarities(self, sm, index, created=datetime.now()):
        logger.debug('1')
        start_time = datetime.now()

        logger.debug('truncating table in {} seconds'.format(datetime.now() - start_time))
        sims = []
        no_saved = 0
        start_time = datetime.now()
        coo = coo_matrix(sm)
        csr = coo.tocsr()

        logger.debug('instantiation of coo_matrix in {} seconds'.format(datetime.now() - start_time))

        query = "insert into similarity (created, source, target, similarity) values %s;"

        conn = self._get_conn()
        cur = conn.cursor()

        cur.execute('truncate table similarity')

        logger.debug('{} similarities to save'.format(coo.count_nonzero()))
        xs, ys = coo.nonzero()
        for x, y in tqdm(zip(xs, ys), leave=True):

            if x == y:
                continue

            sim = csr[x, y]

            if sim < self.min_sim:
                continue

            if len(sims) == 500000:
                psycopg2.extras.execute_values(cur, query, sims)
                sims = []
                logger.debug("{} saved in {}".format(no_saved,
                                                     datetime.now() - start_time))

            new_similarity = (str(created), index[x], index[y], sim)
            no_saved += 1
            sims.append(new_similarity)

        psycopg2.extras.execute_values(cur, query, sims, template=None, page_size=1000)
        conn.commit()
        logger.debug('{} Similarity items saved, done in {} seconds'.format(no_saved, datetime.now() - start_time))

    @staticmethod
    def _get_conn():
        if settings.DATABASES['default']['ENGINE'] == 'django.db.backends.mysql':
            dbUsername = settings.DATABASES['default']['root']
            dbPassword = settings.DATABASES['default']['']
            dbName = settings.DATABASES['default']['foody']
            conn_str = "dbname='foody' user='root' password=''".format(dbName,
                                                              dbUsername,
                                                              dbPassword)
            conn = psycopg2.connect(conn_str)
        elif settings.DATABASES['default']['ENGINE'] == 'django.db.backends.mysql':
            dbName = settings.DATABASES['default']['foody']
            conn = sqlite3.connect(dbName)

        return conn

    def _save_with_django(self, sm, index, created=datetime.now()):
        logger.debug('2')
        start_time = datetime.now()
        Similarity.objects.all().delete()
        logger.info(f'truncating table in {datetime.now() - start_time} seconds')
        sims = []
        no_saved = 0
        start_time = datetime.now()
        coo = coo_matrix(sm)
        csr = coo.tocsr()

        logger.debug(f'instantiation of coo_matrix in {datetime.now() - start_time} seconds')
        logger.debug(f'{coo.count_nonzero()} similarities to save')
        xs, ys = coo.nonzero()
        for x, y in zip(xs, ys):
            if x == y:
                continue

            sim = csr[x, y]

            if sim < self.min_sim:
                continue

            if len(sims) == 500000:

                Similarity.objects.bulk_create(sims)
                sims = []
                logger.debug(f"{no_saved} saved in {datetime.now() - start_time}")

            new_similarity = Similarity(
                source=index[x],
                target=index[y],
                created=created,
                similarity=sim
            )
            no_saved += 1
            sims.append(new_similarity)

        Similarity.objects.bulk_create(sims)
        logger.info('{} Similarity items saved, done in {} seconds'.format(no_saved, datetime.now() - start_time))


def normalize(x):
    x = x.astype(float)
    x_sum = x.sum()
    x_num = x.astype(bool).sum()
    x_mean = x_sum / x_num

    if x_num == 1 or x.std() == 0:
        return 0.0
    return (x - x_mean) / (x.max() - x.min())

