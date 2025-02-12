from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from summarizer.summarizer_prompt import summarizer_prompt
from dotenv import load_dotenv

import os 

load_dotenv()

def create_chain_from_template(template,input_variables, temperature=0.5):
    prompt = PromptTemplate(template=template,
                            input_variables=input_variables)
     
    chain = LLMChain(llm=OpenAI(temperature=temperature),
                     prompt=prompt)
    
    return chain

def write_to_txt(answer,name):
    folder_path = "city_assistant/summarizer_llm_reports"
    
    file_path = os.path.join(folder_path, name)
    with open(file_path, "w") as file:
        file.write(answer)

def summarizer_llm(file_path, chat_conversation):
    chain = create_chain_from_template(template=summarizer_prompt,
                                        input_variables=['conversation'])
    
    answer = chain.predict(conversation=chat_conversation)

    write_to_txt(answer=answer,name=file_path.split('/')[-1])
    
    return answer