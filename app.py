import streamlit as st
# from img_classification import teachable_machine_classification
from PIL import Image, ImageOps
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

from transformers import BertForQuestionAnswering
from transformers import pipeline
from transformers import AutoTokenizer

model = BertForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")
tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")
nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

def show_messages(text):
    messages_str = [
        f"{_['role']}: {_['content']}" for _ in st.session_state["messages"][1:]
    ]
    text.text_area("聊天记录", value=str("\n".join(messages_str)), height=400)


# with open('./config.yaml') as file:
#     config = yaml.load(file, Loader=SafeLoader)

# authenticator = stauth.Authenticate(
#     config['credentials'],
#     config['cookie']['name'],
#     config['cookie']['key'],
#     config['cookie']['expiry_days'],
#     config['preauthorized']
# )
# name, authentication_status, username = authenticator.login('Login', 'main')
# print("preauthetification")

# if authentication_status:
# print("authentificated")
# authenticator.logout('Logout', 'main')
st.title("BERT Q/A")
st.write("教程[link](https://www.youtube.com/watch?v=scJsty_DR3o")
st.write("BERT模型[link](https://huggingface.co/deepset/bert-base-cased-squad2")


BASE_PROMPT = [{"role": "system", "content": "You are a helpful assistant."}]

if "messages" not in st.session_state:
    st.session_state["messages"] = BASE_PROMPT

st.header("BERT 聊天机器人")

text = st.empty()
show_messages(text)

prompt_context = st.text_input("上下文", value="这里输入...")
prompt_question = st.text_input("关于上下文的问题", value="这里输入...")

if st.button("发送"):
    with st.spinner("Generating response..."):
        st.session_state["messages"] += [{"role": "user", "content": prompt_context+'\n'+prompt_question}]
        # response = openai.ChatCompletion.create(
        #     model="gpt-3.5-turbo", messages=st.session_state["messages"]
        # )
        # message_response = response["choices"][0]["message"]["content"]
        message_response = nlp({
                                'question':prompt_question,
                                 'context': prompt_context
                            })
        st.session_state["messages"] += [
            {"role": "system", "content": message_response['answer']}
        ]
        show_messages(text)
        pass

if st.button("清除"):
    st.session_state["messages"] = BASE_PROMPT
    show_messages(text)


# elif authentication_status == False:
#     st.error('Username/password is incorrect')
# elif authentication_status == None:
#     st.warning('Please enter your username and password')
