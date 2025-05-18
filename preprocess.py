import json
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from llm_helper import llm
import re


def clean_text(text):
    return re.sub(r'[\ud800-\udfff]', '', text)

def process_post(raw_file_path, processed_file_path="processed_posts.json"):
    enriched_post = []
    with open(raw_file_path, encoding="utf-8", errors="ignore") as file:
        posts = json.load(file)
        for post in posts:
            cleaned_text = clean_text(post['text'])
            metadata = extract_metadata(cleaned_text)
            post_with_metadata = post | metadata
            post_with_metadata["text"] = cleaned_text 
            enriched_post.append(post_with_metadata)


    unified_tags = get_unified_tags(enriched_post)

    for post in enriched_post:
        current_tags = post['tags']
        new_tags = {unified_tags.get(tag, tag) for tag in current_tags}
        post['tags'] =list(new_tags)

    with open(processed_file_path, 'w', encoding="utf-8") as file:
        json.dump(enriched_post, file, indent=4, ensure_ascii=False)


def get_unified_tags(post_with_metadata):
    unique_tags = set()
    for post in post_with_metadata:
        unique_tags.update(post['tags'])

    unique_tags_list = ', '.join(unique_tags)

    template = '''I will give you a list of tags. You need to unify tags with the following requirements,
    1. Tags are unified and merged to create a shorter list. 
       Example 1: "Jobseekers", "Job Hunting" can be all merged into a single tag "Job Search". 
       Example 2: "Motivation", "Inspiration", "Drive" can be mapped to "Motivation"
       Example 3: "Personal Growth", "Personal Development", "Self Improvement" can be mapped to "Self Improvement"
       Example 4: "Scam Alert", "Job Scam" etc. can be mapped to "Scams"
    2. Each tag should be follow title case convention. example: "Motivation", "Job Search"
    3. Output should be a JSON object, No preamble
    3. Output should have mapping of original tag and the unified tag. 
       For example: {{"Jobseekers": "Job Search",  "Job Hunting": "Job Search", "Motivation": "Motivation}}
    4. DO not give any preamble.  Return Valid JSON Only
    
    : 
    {tags}
    '''

    pt = PromptTemplate.from_template(template)
    chain = pt | llm
    response = chain.invoke(input={'tags':unique_tags_list})

    try:
        json_parser = JsonOutputParser()
        content = clean_text(response.content) 
        res = json_parser.parse(response.content)
    except OutputParserException:
        raise OutputParserException("Content is too big. Unable to parse jobs.")
    
    return res




def extract_metadata(post):
    template = """
    You are given a linkedin post. your task is to extract number of line in the post, language of the post and tags from the post.
    1. return a JSON. No preamble
    2. The JSON object shoule have exactly 3 keys: line_count, language, tag
    3. tags is an array of text tags. extract maximum two tags.
    4. language should be English or Hinglish(Hinglsih meand English + Hindi)
    5. DO not give any preamble
    
    
    Here is the actual post you need to perform this task {post}"""


    pt = PromptTemplate.from_template(template)
    chain = pt | llm
    response = chain.invoke(input={'post': post})

    try:
        json_parser = JsonOutputParser()
        content = clean_text(response.content)
        res = json_parser.parse(response.content)
        res["tags"] = res.pop("tag", [])  # Rename 'tag' key to 'tags'
        return res
    except OutputParserException:
        raise OutputParserException("Content is too big. Unable to parse jobs.")
    
    return res



if __name__ == "__main__":
    process_post("data/raw_post.json", "data/processed_post.json")