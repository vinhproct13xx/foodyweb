import django
import os
import logging
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import coo_matrix
from datetime import datetime
from website import settings
from polls.models import Similarity
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "website.settings")
django.setup()

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
        ratings['avg'] = ratings['avgrating'].astype(float)
        # ratings['avg'] = ratings.groupby('owner')['avgrating'].transform(lambda x: normalize(x))

        # ratings['avg'] = ratings['avgrating'].replace(np.nan, -1)
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