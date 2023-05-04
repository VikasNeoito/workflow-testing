# this is the old code block. keeping this for reference 
'''
from fastapi import FastAPI

from api.routes.api import router as api_router
from core.events import create_start_app_handler
from core.config import API_PREFIX, DEBUG, PROJECT_NAME, VERSION


def get_application() -> FastAPI:
    application = FastAPI(title=PROJECT_NAME, debug=DEBUG, version=VERSION)
    application.include_router(api_router, prefix=API_PREFIX)
    pre_load = False
    if pre_load:
        application.add_event_handler("startup", create_start_app_handler(application))
    return application


app = get_application()

'''
# end of block

import os
from fastapi import FastAPI, File, UploadFile
from dotenv import load_dotenv
from typing import Union
from pydantic import BaseModel
from enum import Enum
import uvicorn

load_dotenv()



from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

openapi_key = os.getenv("openapi_key")

app = FastAPI()

def start():
    """Launched with `poetry run start` at root level"""
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

# chat = ChatOpenAI(temperature=0)
chat = ChatOpenAI(temperature=0.7,model_name='gpt-3.5-turbo',streaming=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=True)

class BaseUpload(BaseModel):
    input_string: str

class EndpointType(str,Enum):
    GENERATE_CODE = "generate_code_template"
    TEST_CODE = "get_test_prompt"
    GENERATE_DESC = "generate_description"

def get_test_prompt(language: str, test_code_file: str):
    with open(test_code_file) as f:
        human_test_code = f.read()

    def prompt_template():
        template = "You are an AI assistant who helps programmers to write high quality code in variety of programming languages. The output will only contain valid code without any description, text, greetings, or note. Exactly follow the given output structure for generating the output"
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        assistant_topic_language_prompt = AIMessagePromptTemplate.from_template("Which is the required programming language?")
        human_template = "{language}"
        user_language_prompt = HumanMessagePromptTemplate.from_template(human_template)
        assistant_out_structure_prompt = AIMessagePromptTemplate.from_template("What is the expected output structure?")
        user_output_struct_prompt = HumanMessagePromptTemplate.from_template("Here is the requested format for the output code: \n <code output here> ")
        assistant_req_prompt = AIMessagePromptTemplate.from_template("Describe the requirement precisely")
        user_input_test_code_prompt = HumanMessagePromptTemplate.from_template("Write testing code for the below function to verify the correctness of code.\n {human_test_code}")
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, assistant_topic_language_prompt, user_language_prompt, assistant_out_structure_prompt, user_output_struct_prompt, assistant_req_prompt, user_input_test_code_prompt])

        return chat_prompt

    chain = LLMChain(llm=chat, prompt=prompt_template(), verbose=True)
    response = chain.run(language=language, human_test_code=human_test_code)
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@",response)
    response = response.split("```")[1]

    with open('output_test_case.txt', 'w') as file:
        file.write(response)



def generate_code_template(language:str, input_string:str):

    def prompt_template():
        template = "You are an AI assistant who helps programmers to write high quality code in variety of programming languages. The output will only contain valid code without any description, text, greetings, or note. Exactly follow the given output structure for generating the output"
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        assistant_topic_language_prompt = AIMessagePromptTemplate.from_template("Which is the required programming language?")
        human_template = "{language}"
        user_language_prompt = HumanMessagePromptTemplate.from_template(human_template)
        assistant_out_structure_prompt = AIMessagePromptTemplate.from_template("What is the expected output structure?")
        user_output_struct_prompt = HumanMessagePromptTemplate.from_template("Here is the requested format for the output code: \n <code output here> ")
        assistant_req_prompt = AIMessagePromptTemplate.from_template("Describe the requirement precisely")
        user_input_test_code_prompt = HumanMessagePromptTemplate.from_template("{input_string}")
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,assistant_topic_language_prompt ,user_language_prompt,assistant_out_structure_prompt, user_output_struct_prompt, assistant_req_prompt, user_input_test_code_prompt])
        return chat_prompt

    chain = LLMChain(llm=chat, prompt=prompt_template(), verbose=True)
    response = chain.run(language=language,input_string=input_string)

    response = response.split("```")[1]

    with open('output_function.txt', 'w') as file:
        file.write(response)



def generate_description(language: str, test_code_file: str):
    with open(test_code_file) as f:
        human_test_code = f.read()

    def prompt_template():
        template = "As an AI assistant, your role is to aid programmers in crafting explanations for a block of code in a specified programming language. The output will solely consist of a legitimate explanation in the form of DocStrings in the given programming language."
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        assistant_topic_language_prompt = AIMessagePromptTemplate.from_template("Which is the required programming language?")
        human_template = "{language}"
        user_language_prompt = HumanMessagePromptTemplate.from_template(human_template)
        assistant_out_structure_prompt = AIMessagePromptTemplate.from_template("what should I do with the programming language?")
        user_output_struct_prompt = HumanMessagePromptTemplate.from_template("we will give a code block, and you have to give explanation for code as doc strings in the provided language. Never output any other texts or description with the Doc String.")
        # assistant_req_prompt = AIMessagePromptTemplate.from_template("SAMPLE_DOC_STRING: \n --- \n /**  \n *[bar description] \n * @param  {[type]} foo [description] \n * @return {[type]}     [description] \n */")
        assistant_req_code = AIMessagePromptTemplate.from_template("Give me the code and I will generate the function description as doc string in the given programming language. Never add any examples or code in Doc Strings.")
        # human_test_code = "{human_test_code}"
        user_input_test_code_prompt = HumanMessagePromptTemplate.from_template("CODE.\n {human_test_code}")
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, assistant_topic_language_prompt, user_language_prompt, assistant_out_structure_prompt, user_output_struct_prompt, assistant_req_code, user_input_test_code_prompt])

        return chat_prompt

    chain = LLMChain(llm=chat, prompt=prompt_template(), verbose=True)
    response = chain.run(language=language, human_test_code=human_test_code)

    response = response.split("```")[1]

    with open('output_desc.txt', 'w') as file:
        file.write(response)



@app.post("/{endpoint_type}")
async def handle_request(
    endpoint_type: EndpointType,
    language:str,
    test_code: UploadFile = File(None),
    input_string: str = "",
):
   
    if endpoint_type == EndpointType.GENERATE_CODE:

    
        generate_code_template(language,input_string)

        with open("output_function.txt") as f:
            response_text = f.read()

        return {"response": response_text}
    


    if endpoint_type == EndpointType.TEST_CODE:
        with open("test_code.txt", "wb") as f:
            contents = await test_code.read()
            f.write(contents)

        get_test_prompt(language, "test_code.txt")

        with open("output_test_case.txt") as f:
            response_text = f.read()

        return {"response": response_text}

    
    
    if endpoint_type == EndpointType.GENERATE_DESC:

        with open("test_code.txt", "wb") as f:
            contents = await test_code.read()
            f.write(contents)

        generate_description(language, "test_code.txt")

        with open("output_desc.txt") as f:
            response_text = f.read()

        return {"response": response_text}


    






