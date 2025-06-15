import requests, json
from openai import OpenAI
from pydantic import BaseModel
from pydantic import ValidationError
import pandas as pd


def get_html(url:str)->str|None:
    try:
        resp = requests.get(url)
        return resp.text
    except:
        print(f'Invalid url: {url}')
        return None
    
def create_prompt(schema:BaseModel, data_source:str):
    if data_source is None or data_source == '':
        raise ValueError('Data source must be specificed and cannot be empty string')

    if not issubclass(schema, BaseModel) or len(schema.model_fields) == 0:
        raise ValueError('Schema must be a pydantic model and cannot be empty')

    prompt = f"""
    Your job is to extract the following JSON schema from the given data source.
    Return the data as an array of JSON objects with the schema fields and their values

    Schema
    ______
    {schema.model_json_schema()}

    Data Source
    _____
    {data_source}
    """
    return prompt

def parse_model_response_as_json(response:str)->list[dict]|None:
    try:
        parsed_json = json.loads(response)
        if isinstance(parsed_json, dict):
            return [parsed_json]
        return parsed_json
    except json.JSONDecodeError as e:
        print(f'Invalid JSON: {e}')
        return None

def validate_extracted_json_objects(schema:BaseModel, json_objects:list[dict]) -> bool:
    for json_obj in json_objects:
        try:
            schema(**json_obj)
        except ValidationError as e:
            print(f'Validation error: {e}')
            return False
    return True

def json_objects_to_dataframe(json_objects:list[dict]) -> pd.DataFrame:
    return pd.DataFrame.from_records(json_objects)

def extract(model:OpenAI, schema:BaseModel, url:str) -> pd.DataFrame|None:
    """
    Extracts structured data from a web source aligning with the schema passed
    """
    html = get_html(url)
    prompt = create_prompt(schema, html)

    model_response = model.responses.create(
        model='o4-mini-2025-04-16',
        input=prompt
    ).output_text
    
    parsed_json = parse_model_response_as_json(model_response)
    if parsed_json is None:
        return None

    if validate_extracted_json_objects(schema, parsed_json):
        dataframe = json_objects_to_dataframe(parsed_json)

    return dataframe
