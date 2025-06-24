import pytest
import pandas as pd
from openai import OpenAI
from schema.extract import (
    get_html, 
    clean_html,
    create_prompt, 
    parse_model_response_as_json,
    validate_extracted_json_objects,
    json_objects_to_dataframe,
    extract,
)

#Env
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('OPEN_AI_KEY')


#Test Inputs
from pydantic import BaseModel, Field
class Empty(BaseModel):
    ...

class People(BaseModel):
    name: str
    age: int
data_source = 'This a story about Jack and Jill. Jack was 5 years old and Jill was 4 1/2.'

extracted_json_objects_correct = [{"name":"Caine", "age":25}]
extracted_json_objects_incorrect = [{"first_name":"Caine", "age":25}]

cl_url = 'https://www.topendsports.com/sport/soccer/list-league-uefa.htm'
class ChampionsLeagueFinal(BaseModel):
        year:int = Field(description='The year the final was played')
        winner:str
        runner_up:str 
        score:str = Field(description='The final score after extra time not including penalties', pattern=r'\d+-\d+')

#Tests
def test_get_html_success():
    html = get_html(url='http://topendsports.com/sport/soccer/list-league-uefa.htm')
    assert isinstance(html, str)

def test_get_html_failure():
    html = get_html(url='http://incorrect-url.com')
    assert html is None

def test_clean_html():
    import regex as re

    with open('tests/fixtures/numpy.html') as f:
        html = f.read()

    print(len(html))
    
    html = clean_html(html)

    with open('tests/fixtures/numpy.txt', 'w') as f:
        f.write(html)


    print(len(html))

    assert len(re.findall(r'<.*>.+<\\>', html)) == 0 #tags are removed
    assert "Install Documentation Learn Community About Us" not in html #nav bar removed
    assert html.split('\n')[0] == 'NumPy' #body selected as root node

def test_create_prompt_success():
    prompt = create_prompt(schema=People, data_source=data_source)

    assert isinstance(prompt, str)
    assert data_source in prompt

def test_create_prompt_no_data_source():
    with pytest.raises(ValueError):
        prompt = create_prompt(schema=People, data_source=None)

def test_create_prompt_empty_data_source():
    with pytest.raises(ValueError):
        prompt = create_prompt(schema=People, data_source='')

def test_create_prompt_no_schema():
    with pytest.raises(ValueError):
        prompt = create_prompt(schema=None, data_source=data_source)

def test_create_prompt_empty_schema():
    with pytest.raises(ValueError):
        prompt = create_prompt(schema=Empty, data_source=data_source)

def test_parse_model_response_success():
    response = '[{"name":"caine", "age":25}, {"name":"char", "age":26}]'
    parsed_json = parse_model_response_as_json(response)
    
    assert isinstance(parsed_json, list)
    assert len(parsed_json) == 2
    assert parsed_json[0]['name'] == 'caine'

def test_parse_model_response_success_single_dict():
    response = '{"name":"caine", "age":25}'
    parsed_json = parse_model_response_as_json(response)
    
    assert isinstance(parsed_json, list)
    assert len(parsed_json) == 1
    assert parsed_json[0]['name'] == 'caine'

def test_parse_model_response_failure():
    response = '[{name:caine, age:25}, {name:char, age:26}]'
    parsed_json = parse_model_response_as_json(response)
    assert parsed_json is None

def test_validate_extracted_json_objects_success():
    valid = validate_extracted_json_objects(People, json_objects=extracted_json_objects_correct)
    assert valid

def test_validate_extracted_json_objects_failure():
    valid = validate_extracted_json_objects(schema=People, json_objects=extracted_json_objects_incorrect)
    assert not valid

def test_json_objects_to_dataframe_sucess():
    dataframe = json_objects_to_dataframe(extracted_json_objects_correct)

    assert isinstance(dataframe, pd.DataFrame)
    assert "name" in dataframe.columns.to_list()
    assert "age" in dataframe.columns.to_list()

# def test_extract_success():
#     model = OpenAI(api_key=api_key)
#     cl_finals_df = extract(model=model, schema=ChampionsLeagueFinal, url=cl_url)

#     assert isinstance(cl_finals_df, pd.DataFrame)
#     assert len(cl_finals_df.columns) == 4