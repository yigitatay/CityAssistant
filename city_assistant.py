import openai
import os
from streaming_handler import StreamingHandler
import os
import openai
import numpy as np
import pandas as pd
import faiss
import textstat
import json
import re
import time
import datetime


openai.api_key = os.getenv("OPENAI_API_KEY")
context_df = pd.read_csv('./data/dc_service_requests.csv')

def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.embeddings.create(
        input=[text],
        model=model
    )
    return np.array(response.data[0].embedding)

def res_estimate_helper(res_estimate):
    resolution_estimate = res_estimate.split(' ')[0]
    bd_or_cd = res_estimate.split(' ')[1]
    resolution_estimate += ' business days' if bd_or_cd == 'bd' else ' calendar days'
    return resolution_estimate

faiss_index = faiss.read_index("dc_requests.faiss")

def search_dc_requests(query: str, top_k: int = 3):
    query_vec = get_embedding(query).astype(np.float32).reshape(1, -1)
    # Search FAISS index
    distances, indices = faiss_index.search(query_vec, top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        row_data = context_df.iloc[idx]
        dist = distances[0][rank]
        results.append({
            "request_type": row_data["request_type"],
            "department": row_data["department"],
            "resolution_estimate": res_estimate_helper(row_data["resolution_estimate"]),
            "description": row_data["Description"],
            "distance": float(dist),
        })
    return results

def generate_response(user_query, handler):
    # 1. Retrieve relevant requests from context
    dc_results = search_dc_requests(user_query, top_k=3)
    
    # 2. Create context string
    context_lines = []
    for res in dc_results:
        context_lines.append(
            f"Request Type: {res['request_type']}\n"
            f"Department: {res['department']}\n"
            f"Resolution Estimate: {res['resolution_estimate']}\n"
            f"Description: {res['description']}\n"
            f"Distance: {res['distance']}\n"
            "----"
        )
    context_str = "\n".join(context_lines)
    
    # 3. Build the final prompt
    SYSTEM_PROMPT = "You are a QA system that assists with resident inquiries and service requests in Washington D.C."
    FINAL_PROMPT = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context from Washington D.C. service requests:\n{context_str}\n\n"
        f"User's question: {user_query}\n"
        f"Provide clear, concise, and legally compliant responses."
        f"Make sure your response is easily readable and understandable by a layman." 
        f"If the answer doesn't belong to one of the request types, state that you're not sure."
        f"Answer format:"
        f"- The content of your answer"
        f"- Used request type: The request type you used"
    )
    
    # 4. Call API
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": FINAL_PROMPT},
        ],
        temperature=0.3,
    )
    answer = response.choices[0].message.content
    return answer


def reprompt_for_correctness(query, ai_response, context_info):
    """
    Calls the LLM again to check whether the AI's answer is correct 
    given the context_info (e.g., the row from dc_service_requests).
    Returns a dict with "is_correct" and "revised_answer" or similar fields.
    
    For demonstration, we do a ChatCompletion call that we parse.
    In production, you might want more robust JSON parsing or error handling.
    """

    system_prompt = "You are a QA system verifying correctness of the AI’s response."
    user_prompt = f"""
User Query: {query}

AI Response:
{ai_response.split('Used request type:')[0]}

Relevant Context (from official data):
{context_info}

Task:
1. Check if the AI's response is factually correct and consistent with the context.
2. If incorrect or incomplete, propose a corrected version.
3. Return your findings in the following JSON format:
{{
  "is_correct": true or false,
  "revised_answer": "text"
}}
"""

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,
    )

    content = response.choices[0].message.content.strip()

    # Attempt to parse a JSON-like structure from the content
    # We'll do a simple regex to find a JSON block, then use Python's `json` if well-formed
    try:
        # find a JSON substring
        json_match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)
            return parsed
        else:
            return {
                "is_correct": False,
                "revised_answer": "Could not parse JSON from LLM response"
            }
    except Exception as e:
        return {
            "is_correct": False,
            "revised_answer": f"Error parsing LLM output: {str(e)}"
        }

def evaluate_response_with_rules(query, ai_response, request_type):
    """
    Checks if the AI response obeys known rules from context_df and overall readability guidelines.
    Returns a dictionary with flags, metrics, and/or suggested corrections.
    """
    evaluation_result = {
        "flesch_reading_ease": None, 
        "gunning_fog": None,
        "potential_request_types": None,
        "rt_complete": True,
        "resolution_estimate_complete": True,
        "complete": True
    }

    # 1. Check readability
    evaluation_result['flesch_reading_ease'] = textstat.flesch_reading_ease(ai_response)
    evaluation_result['gunning_fog'] = textstat.gunning_fog(ai_response)
    
    # 2. Check if request type is in the context
    search_matches = search_dc_requests(query, top_k=3)
    potential_request_types = [match['request_type'].lower() for match in search_matches]
    if request_type.lower() not in potential_request_types:
        evaluation_result.update({"rt_complete": False, "resolution_estimate_complete": False, "complete": False})
        index_of_req = -1
    else:
        index_of_req = potential_request_types.index(request_type.lower())

    evaluation_result["potential_request_types"] = potential_request_types

    # 3. Check if the estimated resolution time is stated in the answer
    known_resolution_estimates = []
    for elem in search_matches:
        known_resolution_estimates.append(elem["resolution_estimate"])
    
    if index_of_req > -1:
        ai_resolution_estimate = known_resolution_estimates[index_of_req]
        ai_resolution_days = ai_resolution_estimate.split(' ')[0]
        if (ai_resolution_days not in ai_response or 
        (ai_resolution_days == '1' and not any(x in ai_response for x in ['1', 'one']))):
            evaluation_result.update({"resolution_estimate_complete": False, "complete": False})
    

    # 4. Re-Prompt to Check Correctness
    context_lines = []
    for res in search_matches:
        context_lines.append(
            f"Request Type: {res['request_type']}\n"
            f"Department: {res['department']}\n"
            f"Resolution Estimate: {res['resolution_estimate']}\n"
            f"Description: {res['description']}\n"
            f"Distance: {res['distance']}\n"
            "----"
        )
    context_str = "\n".join(context_lines)
    correctness_check = reprompt_for_correctness(query, ai_response, context_str)
    
    evaluation_result["is_correct"] = correctness_check.get("is_correct")
    evaluation_result["revised_answer"] = correctness_check.get("revised_answer")

    return evaluation_result

def print_eval_results(eval_result):
    print("\nEVALUATION OF AI RESPONSE:\n")

    print("READABILITY:")
    print(f"Flesch reading ease score: {eval_result['flesch_reading_ease']}")
    print(f"Gunning fog index: {eval_result['gunning_fog']}")

    print(f"\nPotential request types: {eval_result['potential_request_types']}")

    print("\nCOMPLETENESS:")
    if not eval_result["complete"]:
        print("Flagged Response: The AI response is not complete.")
        if not eval_result['resolution_estimate_complete']:
            print("The response does not include the estimated time for resolution of the request.")
        if not eval_result['rt_complete']:
            print("The response does not state which request type the request belongs to. The question might not be in the context.")
    else:
        print('AI response is complete.')

    print("\nCORRECTNESS:")
    if not eval_result['is_correct']:
        print("The answer might not be correct. Here's the revised response:")
        print(eval_result['revised_answer'])
    else:
        print('AI response seems to be correct.')


class CityAssistant:
    def __init__(self, chat_display):
        self.chat_display = chat_display
        self.streaming_handler = StreamingHandler(chat_display)
        self.results_df = pd.DataFrame(columns=['question', 'ai_response', 'flesch_reading_ease', 'gunning_fog', 'rt_complete', 're_complete', 'complete', 'correct', 'revised_answer'])

    def run(self, query):
        ai_response = generate_response(query, handler=self.streaming_handler)
        for word in ai_response.split():
            self.streaming_handler.on_llm_new_token(word + " ")
            time.sleep(0.02)  # Simulate delay

        try:
            request_type = ai_response.split("Used request type: ")[1]
        except:
            request_type = 'Not found'
        eval_result = evaluate_response_with_rules(query, ai_response, request_type)
        print_eval_results(eval_result)
        self.append_to_df(eval_result=eval_result, question=query, ai_answer=ai_response)
        self.streaming_handler.on_llm_end()
        return ai_response
    
    def append_to_df(self, eval_result, question, ai_answer):
        flesch_re, gunning_fog = eval_result['flesch_reading_ease'], eval_result['gunning_fog']
        rt_complete, re_complete, complete = eval_result['rt_complete'], eval_result['resolution_estimate_complete'], eval_result['complete']
        correct = eval_result['is_correct']
        revised_answer = eval_result['revised_answer'] if not correct else 'Correct'

        cur_dict = {
            'question': question,
            'ai_response': ai_answer,
            'flesch_reading_ease': flesch_re,
            'gunning_fog': gunning_fog,
            'rt_complete': rt_complete,
            're_complete': re_complete,
            'complete': complete,
            'correct': correct,
            'revised_answer': revised_answer
        }

        dict_df = pd.DataFrame([cur_dict])

        self.results_df = pd.concat([self.results_df, dict_df], ignore_index=True)
    
    def save_df(self):
        current_datetime = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        self.results_df.to_csv(f'./city_assistant/evals/eval_{current_datetime}.csv')

    