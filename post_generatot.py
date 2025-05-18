from llm_helper import llm
from few_shot import FewShotPosts



few_shot = FewShotPosts()
def get_length_str(length):
    if length == "Short":
        return "1 to 5 lines"
    if length == "Medium":
        return "5 to 10 lines"
    if length == "Long":
        return "10 to 20 lines"

def get_prompt(length, language, tag):
    length_str = get_length_str(length)
    prompt = f'''
    Generate a LinkedIn post using the below information. No preamble.

    1) Topic: {tag}
    2) Length: {length_str}
    3) Language: {language}
    If Language is Hinglish then it means it is a mix of Hindi and English. 
    The script for the generated post should always be English.
    4)The  post should not contain any preamble. strictly post only.
    5) The post should be engaging and should not be generic.
    '''

    examples = few_shot.get_filtered_posts(length, language, tag)

    if len(examples) > 0:
        prompt +="6) Use the writing style as per the following examples."
        for i, post in enumerate(examples):
            post_text = post['text']
            prompt += f"\n\nExample {i+1}: \n\n{post_text}"

            if i == 2:
                break
    return prompt


def generate_post(length, language, tag):
    prompt = get_prompt(length, language, tag)
    response = llm.invoke(prompt)
    return response.content



if __name__ == "__main__":
    post = generate_post("Short", "English", "Motivation")
    print(post)