from phon_lib import ImageRetrieval
from phon_lib import OPENAI_CLIEN, Agent_module, rag_system, qa_bot, sum_rag
from phon_lib import sugar_rag, mith_phon_text
from phon_lib import RAI_ML

from glob import glob

import gradio as gr
import re



def extract_raiG(text):
    pattern = r'G\d+'  # หา substring ที่ขึ้นต้นด้วย 'G' ตามด้วยตัวเลขหลายตัว
    match = re.search(pattern, text)
    extracted_id = match.group(0) if match else None
    return extracted_id

def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)

def add_message(history, message):
    for x in message["files"]:
        history.append(((x,), None))
    if message["text"] is not None:
        history.append((message["text"], None))
    return history, gr.MultimodalTextbox(value=None, interactive=False)

def bot(history):
    global img_rev
    global rager
    global rag_bot,sum_bot,chater
    global rai_status

    question = ""
    try:
        if history[-2][1] is None: #check if there is an image
          #do img rag
          cls_img = img_rev(history[-2][0][0])
          question += f'รูปต่อไปนี้คือรูปของอ้อยที่เป็นโรค{cls_img}\n'
    except:
        pass

    if 'G3100' in history[-1][0]:
        #call api to load data and predict and then extract result
        rai_interest = extract_raiG(history[-1][0])
        if rai_interest is None:
            question += 'ไม่มีรายละเอียดของไร่นั้น'
        else:
            res_status = rai_status(rai_interest)
            question += res_status
        


    question += history[-1][0]
    text_rag = rager(rag_bot(question),3)
    text_rag = sum_bot(text_rag)
    msg = f"rag: {text_rag}\nquestion: {question}"
    res = chater(msg)
    history[-1][1] = res
    return history



with gr.Blocks(fill_height=True) as demo:
    chatbot = gr.Chatbot(
        elem_id="chatbot",
        bubble_full_width=False,
        scale=1,
    )

    chat_input = gr.MultimodalTextbox(interactive=True,
                                      file_count="multiple",
                                      placeholder="Enter message or upload file...", show_label=False)

    chat_msg = chat_input.submit(add_message, [chatbot, chat_input], [chatbot, chat_input])
    bot_msg = chat_msg.then(bot, chatbot, chatbot, api_name="bot_response")
    bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

    chatbot.like(print_like_dislike, None, None)

if __name__ == "__main__":
    img_rev = ImageRetrieval('sugar_cane_desease')
    rager = sugar_rag('การปลูกอ้อย.pdf', mith_phon_text, 16)
    rag_bot = Agent_module('rag_bot', rag_system, OPENAI_CLIEN)
    chater = Agent_module('chat_bot', qa_bot, OPENAI_CLIEN)
    sum_bot = Agent_module('sum_bot', sum_rag, OPENAI_CLIEN)
    rai_status = RAI_ML()
    demo.launch(share=True, debug=True)