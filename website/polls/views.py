import django
from django.shortcuts import render
from django.http import HttpResponse
from .models import Customers
import os
import pandas as pd
import logging
from polls.models import Comments
from polls.neighborhood_based_recommender import NeighborhoodBasedRecs
from polls.item_similarity_matrix_builder import ItemSimilarityMatrixBuilder
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "website.settings")
django.setup()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger('Item simialarity calculator')


def index(request):
    latest_customer_list = Customers.objects.order_by('id')[:10]
    context = {'latest_customer_list': latest_customer_list}
    return render(request, 'polls/index.html', context)


def voice(request):
    return render(request, 'polls/test.html', None)


def runalgorithm(request):
    all_ratings = load_all_ratings()
    ItemSimilarityMatrixBuilder(min_overlap=2, min_sim=0.0).build(all_ratings)
    return HttpResponse("Done.")


def runnbh(request, owner):
    sorted_items = NeighborhoodBasedRecs(neighborhood_size=15, min_sim=0.0).recommend_items(owner, 6)
    logger.debug("show results")
    logger.debug(len(sorted_items))
    for x in sorted_items:
        logger.debug(x)
    return HttpResponse("Done run neighborhood based recommender.")


def run_predict_score(request, owner, resid):
    result = NeighborhoodBasedRecs(neighborhood_size=5, min_sim=0.0).predict_score(owner, resid)
    return HttpResponse(result)


def getSimItem(owner):
    return NeighborhoodBasedRecs(neighborhood_size=15, min_sim=0.0).recommend_items(owner, 6)


def predict_score(owner, res):
    return NeighborhoodBasedRecs(neighborhood_size=15, min_sim=0.0).predict_score(owner, res)


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


def normalize(x):
    x = x.astype(float)
    x_sum = x.sum()
    x_num = x.astype(bool).sum()
    x_mean = x_sum / x_num

    if x_num == 1 or x.std() == 0:
        return 0.0
    return (x - x_mean) / (x.max() - x.min())



