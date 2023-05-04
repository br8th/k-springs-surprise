from surprise import Dataset
from surprise import Reader

from collections import defaultdict
import pandas as pd
from data.db import Product, ProductImage, CustomerRating
from data.db import session

class RatingsRepository:
    
    iid_to_name = {}
    name_to_iid = {}
    
    def load_customer_ratings(self):
        products = pd.DataFrame(item.json() for item in Product.find_all())
        ratings = pd.DataFrame(item.json() for item in CustomerRating.find_all())

        r = pd.merge(products, ratings, left_on='id', right_on='product_id', sort=True)[[
            'customer_id', 'product_id', 'rating']]

        # print('merged stuff: ')
        # print(r.head())
        reader = Reader(rating_scale=(1, 5), line_format = "user item rating")
        ratings_dataset = Dataset.load_from_df(r, reader)

        self.iid_to_name = {}
        self.name_to_iid = {}

        for item in Product.find_all():
            self.iid_to_name[item.id] = item.title
            self.name_to_iid[item.title] = item.id

        return ratings_dataset

    def get_item_rankings(self):
        ratings = defaultdict(int)
        rankings = defaultdict(int)

        for item in CustomerRating.find_all():
            ratings[item.product_id] += 1

        rank = 1
        for iid, ratings_count in sorted(ratings.items(), key = lambda x: x[1], reverse = True):
            rankings[iid] = rank
            rank += 1

        return rankings
    
    def get_product_name(self, iid):
        if iid in self.iid_to_name:
            return self.iid_to_name[iid]
        else:
            return ""
        
    def get_product_id(self, product_name):
        if product_name in self.name_to_iid:
            return self.name_to_iid[product_name]
        else:
            return 0
    
    @staticmethod
    def get_products_details(product_ids):
        # sql = 'SELECT products.id,title, image, price FROM products INNER JOIN product_image ON ' \
        #     'products.id=product_image.product_id WHERE products.id IN(%s);' % ','.join(str(x) for x in product_ids)
        return [product.json() | image.json() for product, image in Product.find_with_image(product_ids)]