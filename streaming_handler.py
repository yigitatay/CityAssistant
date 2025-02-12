from langchain.callbacks.base import BaseCallbackHandler
import tkinter as tk

class StreamingHandler(BaseCallbackHandler):
    def __init__(self, chat_display):
        self.cur_response = ''
        self.finished_streaming = False
        self.chat_display = chat_display
        self.beginning_index = 0.0

    def on_llm_new_token(self, token, **kwargs):
        if self.cur_response == '':
            self.beginning_index = self.chat_display.index(tk.END)
            self.chat_display.insert(tk.END, 'CityAssistant: ', 'blue')
        self.chat_display.insert(tk.END, token, 'blue')
        self.chat_display.see(tk.END)
        self.cur_response = self.cur_response + ' ' + token
    
    def on_llm_end(self, **kwargs):
        self.cur_response = ''
        self.chat_display.delete(str(float(self.beginning_index)-1.0), 'end')