from sqlalchemy import create_engine, MetaData, Table, Column, BigInteger, String, Integer, Text, DateTime, Boolean, Enum, Float
from sqlalchemy.sql import func
import enum
import os
from dotenv import load_dotenv

load_dotenv() #환경변수 로드

DATABASE_URL = os.getenv("DATABASE_URL")  
engine = create_engine(DATABASE_URL)
metadata = MetaData()

# Enum
class DeliveryStatus(enum.Enum):
    BEFORE_DELIVERY = "BEFORE_DELIVERY"
    COMPLETED = "COMPLETED"
    DELIVERY_IN_PROGRESS = "DELIVERY_IN_PROGRESS"
    EXCHANGING = "EXCHANGING"
    REFUNDING = "REFUNDING"

# coupon table
coupon = Table('coupon', metadata,
    Column('coupon_id', BigInteger, primary_key=True, autoincrement=True),
    Column('coupon_name', String(255), nullable=False),
    Column('created_date', DateTime(6), default=func.now()),
    Column('creation_user', BigInteger),
    Column('deleted_status', Integer, default=0),
    Column('discount_rate', Integer),
    Column('ended_date', DateTime(6)),
    Column('modified_date', DateTime(6), default=func.now(), onupdate=func.now()),
    Column('started_date', DateTime(6)),
    Column('used_date', DateTime(6)),
    Column('user_id', BigInteger)
)

# item table
item = Table('item', metadata,
    Column('item_id', BigInteger, primary_key=True, autoincrement=True),
    Column('created_date', DateTime(6), default=func.now()),
    Column('creation_user', String(100)),
    Column('deleted_status', Boolean, default=False),
    Column('description', Text),
    Column('item_image_url', String(500)),
    Column('item_name', String(255), nullable=False),
    Column('item_price', Integer),
    Column('modification_user', String(100)),
    Column('modified_date', DateTime(6), default=func.now(), onupdate=func.now())
)

# item option table
item_option = Table('item_option', metadata,
    Column('i_option_id', BigInteger, primary_key=True, autoincrement=True),
    Column('created_date', DateTime(6), default=func.now()),
    Column('creation_user', String(255)),
    Column('deleted_status', Boolean, default=False),
    Column('modification_user', String(255)),
    Column('modified_date', DateTime(6), default=func.now(), onupdate=func.now()),
    Column('i_option_name', String(255)),
    Column('i_option_quantity', Integer),
    Column('item_id', BigInteger)
)

# purchase detail table
purchase_detail = Table('purchase_detail', metadata,
    Column('purchase_id', BigInteger, primary_key=True, autoincrement=True),
    Column('delivery_status', Enum(DeliveryStatus), default=DeliveryStatus.BEFORE_DELIVERY),
    Column('purchase_date', DateTime(6), default=func.now()),
    Column('quantity', Integer),
    Column('user_id', BigInteger)
)

# purchase item table
purchase_item = Table('purchase_item', metadata,
    Column('option_id', BigInteger, primary_key=True),
    Column('purchase_id', BigInteger, primary_key=True)
)

# review table
review = Table('review', metadata,
    Column('review_id', BigInteger, primary_key=True, autoincrement=True),
    Column('comment', Text),
    Column('created_date', DateTime(6), default=func.now()),
    Column('creation_user', String(255)),
    Column('deleted_status', Integer, default=0),
    Column('item_id', BigInteger),
    Column('modificated_date', DateTime(6), default=func.now(), onupdate=func.now()),
    Column('purchase_id', BigInteger),
    Column('review_score', Integer)
)

# user table
user = Table('user', metadata,
    Column('user_id', BigInteger, primary_key=True, autoincrement=True),
    Column('address_detail', String(255)),
    Column('address_road', String(255)),
    Column('created_date', DateTime(6), default=func.now()),
    Column('deleted_status', Boolean, default=False),
    Column('email', String(255), unique=True, nullable=False),
    Column('modified_date', DateTime(6), default=func.now(), onupdate=func.now()),
    Column('password', String(255), nullable=False),
    Column('user_role', String(50), default='USER')
)
