import torch 
import os
import pandas
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType, PeftModel, PeftConfig
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModel, AutoTokenizer,AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context.\n"
        "아래는 작업을 설명하는 명령어와 추가적 맥락을 제공하는 입력이 짝을 이루는 예제입니다.\n\n"
        "Write a response that appropriately completes the request.\n요청을 적절히 완료하는 응답을 작성하세요.\n\n"
        "### Input(입력):\n{input}\n\n### Instruction(명령어):\n{instruction}\n\n### Response(응답):"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task.\n"
        "아래는 작업을 설명하는 명령어입니다.\n\n"
        "Write a response that appropriately completes the request.\n명령어에 따른 요청을 적절히 완료하는 응답을 작성하세요.\n\n"
        "### Instruction(명령어):\n{instruction}\n\n### Response(응답):"
    ),
}

checkpoint='models/KoAlpaca-30B-LoRA'
sentence_transformer_path="KDHyun08/TAACO_STS"
embedder = SentenceTransformer(sentence_transformer_path)



def data_prep_from_txt(file_name : str, o_data : dict = None) :
    '''
    Here we assume the data is a text file and follows the format :
    
    ■ location 
    
    
    [데이터]
    summary....
    
    [질의응답]
    Q1: question 1
    A1: answer1 
    .
    .
    .
    
    '''
    with open(file_name,'r') as fil :
        data=fil.read()
    data=data.replace('n\n\n\n\n\n\n\n\n\n','')
    data=data.split("■")
    #print(data)
    data=[data[i].split('[질의응답]') for i in range(len(data))]
    for i in range(len(data)) :
        try :
            data[i][1]=data[i][1].split('\nQ')
        except Exception as e  :
            continue
    for i in range(len(data)) :
        try :
            for j in range(len(data[i][1])) :
                data[i][1][j]=data[i][1][j].split('\nA')
        except Exception as e  :
            continue
    data=data[1:]
    #print(data)
    for i in range(len(data)) :
        data[i][1]=data[i][1][1:]
    
    dataf = {} if not o_data else o_data
    for i in range(len(data)):
        dataf[data[i][0].split('\n')[0][1:]] = {
               'context' : data[i][0].split('[데이터]')[1] ,
               'question' : [data[i][1][j][0].split(':')[1].replace('\n','')  for j in range(len(data[i][1]))],
               'answer' : [data[i][1][j][1].split(':')[1].replace('\n','')  for j in range(len(data[i][1]))],
                }  
        
    return dataf





def get_context(question : str, kD : dict) : #카테고리를 반환
    global category
    corpus = list(kD.keys())
    hits = util.semantic_search(embedder.encode(question),embedder.encode(corpus))
    print("카테고리 hits: ", hits)


    #유사도가 0.5이상이면 hits내용(카테고리), 이하면 빈 문자열 반환
    category = corpus[hits[0][0]['corpus_id']] if hits[0][0]['score'] > 0.5 else ''
    return category




def get_fitting_mod(data,question,size,tokenizer,hist,prev) :   #일치하는 질문을 찾지 못했을때    
    #print(data['context'].split('\n\n')) 해당 카테고리의 context내용(단락별로 리스트)
    
    hits=util.semantic_search(embedder.encode(question),embedder.encode(data['context'].split('\n\n')))
    hitsp=util.semantic_search(embedder.encode(prev),embedder.encode(data['context'].split('\n\n')))
    x = PROMPT_DICT['prompt_input'].format(instruction=question, input=data['context']+hist)   
    print("x의 내용 출력: ",x)
    print("prev의 내용 출력:\n",prev)
    print("hist의 내용 출력: \n",hist)
    print("hits의 내용 출력: \n",hits)    #context의 어느 단락이랑 가장 유사한지
    print("hitsp의 내용 출력: \n",hitsp)      #이전질문은 context의 어느 단락이랑 가장 유사한지
    if hits[0][0]['score']<0.4 :
        return ''       #유사한 단락이 없으면 빈문자열
    length = len(hits[0])    #단락 개수
    while (1):
        tok_size =  tokenizer.encode(x,return_tensors="pt").shape[1]
        print("size: ",tok_size,"\n")
        if(tok_size<=size):     #size = 2048-128
            break
        
        length-=1
        x=PROMPT_DICT['prompt_input'].format(instruction=question, input=data['context'].split('\n\n')[hitsp[0][0]['corpus_id']]+'\n\n'+'\n\n'.join([data['context'].split('\n\n')[hits[0][i]['corpus_id']] for i in range(length)])+'\n\n'+hist)
        print("[",length,"(l)]\n",x)
    return x


def get_answer(data,question) :     #data['question']에서 입력질문과 가장 비슷한 거 반환
    hits=util.semantic_search(embedder.encode(question),embedder.encode(data['question']))
    l = len(hits[0])
    print("get_ans함수의 hit 내용: ",hits)
    return (data['answer'][hits[0][0]['corpus_id']],hits[0][0]['score'])  




def make_convo( model,tokenizer, kd : dict) :
    global prev_quest
    category=''
    hist1="Below is an instruction that describes a task.\n아래는 작업을 설명하는 명령어입니다.\n\nWrite a response that appropriately completes the request.\n명령어에 따른 요청을 적절히 완료하는 응답을 작성하세요.\n"
    hist2="Below is an instruction that describes a task, paired with an input that provides further context.\n아래는 작업을 설명하는 명령어와 추가적 맥락을 제공하는 입력이 짝을 이루는 예제입니다.\n\nWrite a response that appropriately completes the request.\n요청을 적절히 완료하는 응답을 작성하세요.\n\n### Input(입력):\n"
    hist=''
    user_input=''
    context=''
    tok=False
    i=0
    x=('',0)
    da=''
    while True :
        user_input=input('>>') 
        if user_input.lower() == 'bye' :
            break
        n_context= get_context(user_input,kd)
        print("n_context 내용: ",n_context)
        if n_context != category:   #2개 정도로 늘려서 다시 테스트해보기
            #prev_quest=''
            hist='' #질문-응답 쌍
        if n_context!='' :
            context=n_context
        if context != '' :
            x=get_answer(data=kd[context],question=user_input)
            print("make_conv함수의 x내용: ",x)      #매치되는 (answer,정확도)
        if i==0: 
            model_inp=hist1+"\n\n### Instruction(명령어):\n"+user_input+"\n\n### Response(응답):" 
            nhist=hist+"\n\n### Instruction(명령어):\n"+user_input+"\n\n### Response(응답):"
        else :
            model_inp=hist2+"#####\nPrevious_chat\n\n"+hist+"\n#####\n"+"\n\n### Instruction(명령어):\n"+user_input+"\n\n### Response(응답):" 
            nhist=hist+"\n\n### Instruction(명령어):\n"+user_input+"\n\n### Response(응답):"

        if x[1]>0.5 :           #반환된 답의 정확도가 0.5이상일 경우
            gen_text=x[0]
        
        
        else :  #x[1]<=0.5
            if context != '' :
                #if first query
                if prev_quest=='':
                    print("bot: 해당 질문에 답을 할 수 없습니다. 감사합니다.")
                    continue
                da=get_fitting_mod(kd[context],user_input,1024-128,tokenizer,hist,prev_quest)
                print("da 내용 : ",da)
            mi=model_inp if da=='' else da
            input_ids=tokenizer.encode(mi, return_tensors="pt").to(model.device)    #peft 모델로 답변 생성
            gen_tokens = model.generate(
                input_ids=input_ids, 
                max_new_tokens=128, #생성 토큰의 개수
                num_return_sequences=1, 
                temperature=0.5,        #이전에 나왔던 토큰의 확률을 0으로 바꿈 --> 반복적인 생성 제거
                no_repeat_ngram_size=6,
                do_sample=False,     #할때마다 다른 응답이 나오게 됨
                top_k=50,
                top_p=0.90,
            )
            gen_text = tokenizer.decode(gen_tokens[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        hist=nhist+gen_text
        prev_quest=gen_text
        i=i+1
        category = n_context
        print(f'bot: {gen_text}')



if __name__=='__main__' :
    prev_quest=''
    category=''
    path_dir='data' #replace with the path to the txt data files
    df={}
    for fil in os.listdir(path_dir) :
        if not fil.endswith('txt') :
            continue
        df=data_prep_from_txt(path_dir+'/'+fil,df)
    config=PeftConfig.from_pretrained(checkpoint)
    tokenizer = LlamaTokenizer.from_pretrained(config._name_or_path)
    model = LlamaForCausalLM.from_pretrained(config._name_or_path,device_map='auto',load_in_8bit=True)
    model = PeftModel.from_pretrained(model, checkpoint, device_map={'':0})
    

    make_convo(model,tokenizer,df)