from schema.extract import extract
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv('OPEN_AI_KEY')

def test_extract_ps5_products():
    class Listings(BaseModel):
        title:str
        subtitle:str
        price:float
        seller:str
        seller_number_of_sales:str
        seller_satisfaction_percentage:float

    url = 'https://www.ebay.com/sch/i.html?_nkw=ps5&_sacat=0&_from=R40&_trksid=p2334524.m570.l1313&_odkw=predators&_osacat=0'

    model = OpenAI(api_key=api_key)
    listings_df = extract(url=url, schema=Listings, model=model)

    listings_df.to_csv('ps5_listing.csv')