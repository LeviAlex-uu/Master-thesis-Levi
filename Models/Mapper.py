import os
import sys
import math
import openai
import codecs
import random

import numpy as np
import pandas as pd

from tqdm import tqdm
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv
from dataReader import getLabels

def validateApiKey(api_key: str):
    """
    Validates a given OPENAI_API_KEY.
    |
    api_key:    OPENAI secret key
    """
    client = OpenAI(api_key=api_key)
    try:
        client.models.list()
    except openai.AuthenticationError:
        print('Given key was invalid.')
        exit()
    else:
        return True

def createVectorStore(
    client,
    path: str = './VectorStore',
    store_name: str = 'ontology library',
):
    """
    Creates a vector store using files found in the given path.
    |
    client:     OPENAI client
    path:       Path to folder containing .txt files that will be stored
    store_name: Name of the vector store
    """
    # 1. Create vector store
    vector_store = client.beta.vector_stores.create(
        name=store_name,
    )

    # 2. Load files to vector store
    file_streams = [open(f'{path}/{f}', 'rb') for f in os.listdir(path)]

    file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store.id,
        files = file_streams
    )

    print(file_batch.status)
    print(file_batch.file_counts)

    # 3. Save vector store id
    f = open('id_vector_store.txt', 'w', encoding='utf-8')
    f.write(vector_store.id)
    f.close()
    

def createAssistant(
    assitant_prompt: str,
    api_key: str,
    model_type: str = 'gpt-4o',
    identifier: str = 'levi',
):
    """
    Creates a GPT assistant using the given prompt and model type.
    |
    assistant_prompt:   Input prompt for the assistant containing the assignment and behavior the assistant should follow
    api_key:            OPENAI secret key
    model_type:         Version of GPT the assistant will use
    identifier:         Id used to store the id of an assistant in a .txt file
    """
    client = OpenAI(api_key=api_key)
    # 1. Create assistant
    assistant = client.beta.assistants.create(
        name=f'Semantic Mapping Assistant {identifier}',
        instructions=assitant_prompt,
        model=model_type,
        tools=[{'type': 'file_search'}],
    )

    # 2. Load vector store id
    file = open('id_vector_store.txt', 'r')
    vector_store_id = file.read()

    # 3. Update assistant
    assistant = client.beta.assistants.update(
        assistant_id=assistant.id,
        tool_resources={'file_search': {'vector_store_ids': [vector_store_id]}}
    )

    # 4. Save assitant id
    f = open(f'id_assistant_{identifier}.txt', 'w')
    f.write(assistant.id)
    f.close() 


def saveOutput(
    messages: list,
    file_path: str = './Results/outputGPT.txt',
    fine_tuned: bool = False,
):
    """
    Saves the messages output from the GPT model in the file specified by 'file_path'
    |
    messages:   List of messages from the rungpt function.
    file_path:  Path to output file  
    """
    # 1. Gather all values of the messages 
    d = []
    if fine_tuned: 
        for message in messages:
            d.append(message[0].choices[0].message.content)

    else:
        for message in messages:
            n = -math.floor(len(message.data)/2)
            for m in message.data[:n]:
                d.append(m.content[0].text.value)
        
    output = '\n'.join(d)

    # 2. Check if output encoding is UTF-8 
    try:
        s = codecs.decode(codecs.encode(output, 'utf-8', 'replace'), 'utf-8')
    except:
        s = None

    # 3. Write output to file
    f = open(file_path, 'w', encoding='utf-8')
    f.write(s)
    f.close()

   
def batchData(
    data: list,
    batch_size: int = 10,
):
    """
    Create batches from a list of data
    |
    data:       Data
    batch_size: Size of each batch
    """
    result = []
    for i in range(0,len(data),batch_size):
        result.append(data[i:i+batch_size])
    return result


def generateFewshots(
    num_shots: int = 3,
    #context: bool = False,
):
    """
    Generates a prompt for few-shot prompting
    |
    num_shots:  Number of examples
    """
    _,labels,_,matches,_,_,ontologies,_ = getLabels()

    data = [(x,y,z) for x,y,z in zip(labels,matches,ontologies)]
    
    prompt = 'Here are some example mappings:\n'
    
    for x,y,z in random.sample(data, k=num_shots):
        prompt += f'Input: {x} \nAnwser: {{Label: {x} , Match: {y}, Ontology: {z}, Score: {1}}}\n'

    return prompt

def createOntologyContext(path : str = "./VecStore2"):
    context  = []
    
    for i,file in enumerate(os.listdir(path)):
        f = open(f"{path}/{file}", 'r', encoding='utf-8')
        lines = f.readlines()[1:-1]
        lines = [x for x in lines if x[:6].strip() == 'rdfs']
        context.append(f"Ontology {i+1}, {file.replace('.txt', '')}:")
        context.append(''.join(lines))

    context = '\n'.join(context)
    
    f = open('test.txt', 'w', encoding='utf-8')
    f.write(context)
    f.close()

def finetune(
    api_key: str,
):
    client  = OpenAI(api_key=api_key)

    client.fine_tuning.jobs.create(
        training_file='file-R3tIdjm3ifhZ5jlEztVd77uN',
        validation_file='file-U2mfp52P1p0KSEqJrHJqy7Zv',
        model='gpt-3.5-turbo',
    )

def runGPT(
    assistant_identifier : str,
    request_prompt: str,
    api_key: str,
    batches: list = [],
    few_shot: bool = False,
):
    """
    Runs the assistant using a request prompt
    |
    assistant_identifier:   Identifier used to call the right assistant
    request_prompt:         Prompt which is added to the messages of the model and excuted when the model is run
    api_key:                OPENAI secret key
    batches:                Batched data
    few_shot:               
    """
    # 1. Create client
    client = OpenAI(api_key=api_key)

    # 2. Create a thread
    thread = client.beta.threads.create()


    for batch in batches:
        label = '\n' + '\n'.join(batch)
        request_prompt = request_prompt.replace('$', label)  # Prompt contains $ character which is replaced by the labels

        if few_shot:
            examples = generateFewshots(5)
        else:
            examples = ''

        request_prompt = request_prompt.replace('@', examples)      # Prompt contains @ character which is replaced by the few-shot examples

                # 3. Add instructions to the thread
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role='user',
            content=request_prompt,
        )

    # 4. Run thread
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant_identifier,
    )

    # 5. Return results
    if run.status == 'completed': 
        messages = client.beta.threads.messages.list(
            thread_id=thread.id
        )
        #print(messages)
    else:
        print(run.status)

    return messages


def runGPTft(
    assistant_identifier : str,
    assistant_prompt : str,
    request_prompt: str,
    api_key: str,
    batches: list = [],
    few_shot: bool = False,
):
    # 1. Create client
    client = OpenAI(api_key=api_key)

    messages = []
    for batch in batches:
        data = [{"role":"system", "content":assistant_prompt}]

        label = '\n' + '\n'.join(batch)
        request_prompt = request_prompt.replace('$', label)  # Prompt contains $ character which is replaced by the labels
        
        if few_shot:
            examples = generateFewshots(5)
        else:
            examples = ''

        request_prompt = request_prompt.replace('@', examples)      # Prompt contains @ character which is replaced by the few-shot examples
        
        d = [{"role":"user", "content": request_prompt}]
        # 2. 
        completion = client.chat.completions.create(
            model=assistant_identifier,
            messages= data + d,
        )
        messages.append(completion)

    return messages


if __name__ == "__main__":
    # Load environment, if not present create .env file and create a variable called OPENAI_API_KEY
    load_dotenv()

    setup_file = open('./Prompts/AssistantSetupFineTuning.txt', 'r')
    assistant_prompt = ' '.join(setup_file.readlines()).replace('\n', '')

    prompt_file = open('./Prompts/TaskPrompt.txt', 'r')
    request_prompt = ' '.join(prompt_file.readlines()).replace('\n', '')

    context_file = open('./test.txt', 'r')
    context = context_file.read()

    id_file = open('./Assistants/id_assistant_levi_4.txt', 'r')
    assistant_id = id_file.read()

    key = os.environ['OPENAI_API_KEY']

    mapping = getLabels()
    #temp = [f"{x}; {y}" for x,y in zip([x.split(':')[0] for x in mapping[0]],mapping[1])]
    batches = batchData(mapping[1])
    
    messages = []
    assistant_prompt = assistant_prompt.replace('@', '').replace('$', context)

    #createVectorStore(client=OpenAI(api_key=key),path='./VecStore2',store_name='onology library small')
    #createAssistant(assistant_prompt,key, model_type='gpt-3.5-turbo', identifier='levi_3.5_V1.1')
    #finetune(key)
    #print(generateFewshots())
    #createOntologyContext()
    #print(batches[0])
    
    
    if validateApiKey(key):
        print(f'Given key: {key[:3] }...{ key[-4:]}')
                
        for i in tqdm(range(0,len(batches),1)):
            messages.append(runGPT(assistant_id, request_prompt, key, batches=batches[i:i+1], few_shot=True))
    
    saveOutput(messages, file_path='./Results/outputGPT4_zeroshot_4.txt')