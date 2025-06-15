from schema.extract import extract
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv('OPEN_AI_KEY')

def test_extract_house_listings():
    class Listings(BaseModel):
        location:str
        house_type:str
        number_of_bedrooms:int
        number_of_bathroioms:int
        estage_agent:str

    url = 'https://www.rightmove.co.uk/property-to-rent/find.html?searchLocation=Manchester%2C+Greater+Manchester&useLocationIdentifier=true&locationIdentifier=REGION%5E904&radius=1.0&propertyTypes=detached%2Csemi-detached%2Cterraced&maxDaysSinceAdded=1&_includeLetAgreed=on'

    model = OpenAI(api_key=api_key)
    listings_df = extract(url=url, schema=Listings, model=model)

    listings_df.to_csv('house_listings.csv')