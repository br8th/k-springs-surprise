from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm import Session
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
engine = create_engine("mysql+pymysql://kyalo-db:secret@localhost:3306/k_springs", echo=False)
session = Session(engine)

class CustomerRating(Base):
    __tablename__ = 'customer_ratings'
    id = Column(Integer, primary_key=True)
    rating = Column(Integer)
    customer_id = Column(Integer, ForeignKey('customers.id'))
    product_id = Column(Integer, ForeignKey('products.id'))

    @classmethod
    def find_all(cls):
        return session.query(CustomerRating);

    def json(self):
        return {'rating': self.rating, 'customer_id': self.customer_id, 'product_id': self.product_id}

class Product(Base):

    __tablename__ = 'products'
    id = Column(Integer, primary_key=True)
    title = Column(String())
    price = Column(Integer)

    def __init__(self, title):
        self.title = title

    @classmethod
    def find_all(cls):
        return session.query(Product)

    @classmethod
    def find_with_image(cls, product_ids):
        return session.query(Product, ProductImage).join(ProductImage).where(Product.id.in_(product_ids)).all()

    def json(self):
        return { 'title': self.title, 'id': self.id, 'price': self.price }

class ProductImage(Base):

    __tablename__ = 'product_image'
    id = Column(Integer, primary_key=True)
    image = Column(String())
    product_id = Column(Integer, ForeignKey('products.id'))

    def json(self):
        return {'image': 'http://localhost:5500/uploads/' + self.image}
        
