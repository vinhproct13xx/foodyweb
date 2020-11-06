from .models import Comments
from .models import Similarity
from django.db.models import Q
import time
from decimal import Decimal


class NeighborhoodBasedRecs:

    def __init__(self, neighborhood_size=15, min_sim=0.0):
        self.neighborhood_size = neighborhood_size
        self.min_sim = min_sim
        self.max_candidates = 6

    def recommend_items(self, user_id, num=6):
        active_user_items = Comments.objects.filter(owner=user_id).order_by('-avgrating')[:100]
        return self.recommend_items_by_ratings(user_id, active_user_items.values())

    def recommend_items_by_ratings(self, user_id, active_user_items, num=6):

        if len(active_user_items) == 0:
            return {}

        start = time.time()
        res_ids = {res['resid']: res['avgrating'] for res in active_user_items}

        user_mean = sum(res_ids.values()) / len(res_ids)

        candidate_items = Similarity.objects.filter(Q(source__in=res_ids.keys())
                                                    & ~Q(target__in=res_ids.keys())
                                                    & Q(similarity__gt=self.min_sim)
                                                    )
        candidate_items = candidate_items.order_by('-similarity')[:self.max_candidates]
        recs = dict()
        for candidate in candidate_items:
            target = candidate.target
            pre = 0
            sim_sum = 0

            rated_items = [i for i in candidate_items if i.target == target][:self.neighborhood_size]
            if len(rated_items) > 0:
                for sim_item in rated_items:
                    index = (int)(sim_item.source)
                    r = Decimal(res_ids[index] - user_mean)
                    pre += sim_item.similarity * r
                    sim_sum += sim_item.similarity
                if sim_sum > 0:
                    recs[target] = {'prediction': Decimal(user_mean) + pre / sim_sum,
                                    'sim_items': [r.source for r in rated_items]}

        sorted_items = sorted(recs.items(), key=lambda item: -float(item[1]['prediction']))[:num]
        return sorted_items

    def predict_score(self, user_id, item_id):
        owner_items = Comments.objects.filter(owner=user_id)
        owner_items = owner_items.exclude(resid=item_id).order_by('-avgrating')[:100]
        res_ids = {res.resid: res.avgrating for res in owner_items}

        return self.predict_score_by_ratings(item_id, res_ids)

    def predict_score_by_ratings(self, item_id, res_ids):
        top = Decimal(0.0)
        bottom = Decimal(0.0)
        ids = res_ids.keys()
        mc = self.max_candidates
        candidate_items = (Similarity.objects.filter(source__in=ids)
                                             .exclude(source=item_id)
                                             .filter(target=item_id))
        candidate_items = candidate_items.distinct().order_by('-similarity')[:mc]

        if len(candidate_items) == 0:
            return 0

        for sim_item in candidate_items:
            r = res_ids[sim_item.source]
            top += float(sim_item.similarity) * float(r)
            bottom += float(sim_item.similarity)

        return Decimal(top/bottom)