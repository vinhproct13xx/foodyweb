from .models import Comments
from .models import Similarity
from django.db.models import Q
import time

from decimal import Decimal


class NeighborhoodBasedRecs:

    def __init__(self, neighborhood_size=15, min_sim=0.0):
        self.neighborhood_size = neighborhood_size
        self.min_sim = min_sim
        self.max_candidates = 100

    def recommend_items(self, user_id, num=6):

        active_user_items = Comments.objects.filter(owner_id=user_id).order_by('-avgrating')[:100]
        # return active_user_items.values()
        return self.recommend_items_by_ratings(self, user_id, active_user_items.values())

    def recommend_items_by_ratings(self, user_id, active_user_items, num=6, min_sim = 0.0, max_candidates=100,neighborhood_size=15):

        if len(active_user_items) == 0:
            return {}

        start = time.time()
        res_ids = {res['resid_id']: res['avgrating'] for res in active_user_items}
        user_mean = sum(res_ids.values()) / len(res_ids)

        candidate_items = Similarity.objects.filter(Q(source__in=res_ids.keys())
                                                    & ~Q(target__in=res_ids.keys())
                                                    & Q(similarity__gt= min_sim)
                                                    )
        candidate_items = candidate_items.order_by('-similarity')[:max_candidates]

        recs = dict()
        for candidate in candidate_items:
            target = candidate.target
            pre = 0
            sim_sum = 0

            rated_items = [i for i in candidate_items if i.target == target][:neighborhood_size]
            if len(rated_items) > 0:
                for sim_item in rated_items:
                    r = Decimal(res_ids[sim_item.source] - user_mean)
                    pre += sim_item.similarity * r
                    sim_sum += sim_item.similarity
                if sim_sum > 0:
                    recs[target] = {'prediction': Decimal(user_mean) + pre / sim_sum,
                                    'sim_items': [r.source for r in rated_items]}

        sorted_items = sorted(recs.items(), key=lambda item: -float(item[1]['prediction']))[:num]
        return sorted_items

    # def predict_score(self, user_id, item_id):
    #
    #     user_items = Rating.objects.filter(user_id=user_id)
    #     user_items = user_items.exclude(movie_id=item_id).order_by('-rating')[:100]
    #     movie_ids = {movie.movie_id: movie.rating for movie in user_items}
    #
    #     return self.predict_score_by_ratings(item_id, movie_ids)
    #
    # def predict_score_by_ratings(self, item_id, movie_ids):
    #     top = Decimal(0.0)
    #     bottom = Decimal(0.0)
    #     ids = movie_ids.keys()
    #     mc = self.max_candidates
    #     candidate_items = (Similarity.objects.filter(source__in= ids)
    #                                          .exclude(source=item_id)
    #                                          .filter(target=item_id))
    #     candidate_items = candidate_items.distinct().order_by('-similarity')[:mc]
    #
    #     if len(candidate_items) == 0:
    #         return 0
    #
    #     for sim_item in candidate_items:
    #         r = movie_ids[sim_item.source]
    #         top += sim_item.similarity * r
    #         bottom += sim_item.similarity
    #
    #     return Decimal(top/bottom)
