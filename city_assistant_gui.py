import tkinter as tk
from city_assistant import CityAssistant
import re
import datetime
import threading
import os
from summarizer.llm_summarizer import summarizer_llm


class ChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("City Assistant")
        self.root.configure(bg="#222")  # Set background color to dark gray

        # Initialize a list to store chat history
        self.chat_history = []

        # Create a text widget to display chat messages
        self.chat_display = tk.Text(root, width=70, height=50, bg='white')
        self.chat_display.pack()

        # Create an entry widget for user input
        self.user_input = tk.Entry(root, width=50)
        self.user_input.pack()

        # Create a send button to send user messages
        self.send_button = tk.Button(root, text="Send", command=self.send_message, bg="#444", fg="black")
        self.send_button.pack()

        # Initialize the chat with a welcome message
        initial_message = "City Asisstant: Hello, I am here to help with anything regarding residential issues. Ask me a question!"
        self.chat_history.append(initial_message)
        self.display_message(initial_message, color="blue")

        # Set a flag to track whether the model has been selected
        self.model_selected = False
        self.model_selection_attempts = 0  # Count user's attempts to enter the model name

        # Track whether the user input is enabled
        self.user_input_enabled = True

        self.reported = False

        # Track whether the user should be able to send a message 
        self.send_enabled = True

        # Bind the close event to the save_and_exit function
        self.root.protocol("WM_DELETE_WINDOW", self.save_and_exit)

        # Bind the Enter key to call the send_message function
        self.user_input.bind("<Return>", lambda event: self.send_message())
    
         # Initialize the City Assistant chatbot
        self.city_assistant = CityAssistant(self.chat_display)

    def save_and_exit(self):
        if not self.reported:
            self.save_chat_history(report=False)  # Save chat history before exiting
        self.root.destroy()  # Close the application

    def send_message(self):
        if self.send_enabled:
            user_message = self.user_input.get()
            if user_message and len(user_message) > 0:
                self.chat_history.append(f"User: {user_message}")
                self.display_message(f"User: {user_message}", color="black")  # Set User message to black

                # Clear the user input field
                self.user_input.delete(0, tk.END)

                processing_message = "City Assistant: ..."
                self.processing_index = self.display_message(processing_message, color='blue')

                threading.Thread(target=self.process_message, args=(user_message,)).start()

            else:
                message = "City Assistant: I am sorry, you need to write me a message. What can I assist you with?"
                self.display_message(message, color='blue')
                self.chat_history.append(message)

    def process_message(self, user_message):
        self.send_button['state'] = 'disabled'
        self.send_enabled = False
        bot_message = ''
        customer_agent = False
    
        # Delete the "City Assistant: ..." message
        start_index = int(self.processing_index.split('.')[0]) - 4.0
        end_index = start_index + 2.0
        self.chat_display.delete(str(start_index), str(end_index))
        # Get the bot's response based on user input
        bot_message = self.bot_answer(user_message)

        if "not sure" in bot_message.lower() or "not applicable" in bot_message.lower():
            customer_agent = True
            

        self.send_button['state'] = 'active'
        self.send_enabled = True
        self.display_message(bot_message, color='blue')
        self.chat_history.append(bot_message)

        
        if customer_agent:
            self.reported = True
            self.save_chat_history(report=True)


    def initialize_chat(self):
        # Initialize the City Assistant chatbot
        self.city_assistant = CityAssistant(self.chat_display)

    def bot_answer(self, user_input):
        # Use the City Assistant instance to get a response based on user input
        bot_response = self.city_assistant.run(user_input)
        return f"City Assistant: {bot_response}"

    def display_message(self, message, color="black"):
        self.chat_display.tag_configure(color, foreground=color)

        # Check if the message is from City Assistant
        if message.startswith("City Assistant:"):
            report_button = tk.Button(self.root, text="Report", bg="#FF3131", fg="#000000", relief=tk.FLAT, font=("Arial", 12), command=lambda message=message: self.report_message())
            report_button.pack(pady=1)

            report_button.config(borderwidth=1, relief="solid")  # Adjust border width and relief
            report_button.config(highlightbackground="#F1F1F1", highlightcolor="#F1F1F1", highlightthickness=1, activebackground="#C0C0C0", activeforeground="#000000")            
            self.chat_display.window_create(tk.END, window=report_button)


        self.chat_display.insert(tk.END, message + "\n", color)
        self.chat_display.insert(tk.END, "\n\n")  # Add some spacing

        last_index = self.chat_display.index(tk.END)
        self.chat_display.see(tk.END)
        return last_index


    def report_message(self):
        # Mark the message as "FLAGGED"
        if "(FLAGGED)" not in self.chat_history[-1]:
            self.chat_history[-1] = "(FLAGGED) " + self.chat_history[-1]


    def save_chat_history(self, report=False):
        # Save chat history to a text file
        current_datetime = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        if not (os.path.exists("./city_assistant/chat_history")):
            os.makedirs("./city_assistant/chat_history")
        if report:
            file_path = f"./city_assistant/chat_history/{current_datetime}_reported.txt"
            with open(file_path, "w") as file:
                msg = ''
                for message in self.chat_history:
                    file.write(message + "\n")
                    msg = msg + "\n" + message
                summarizer_llm(file_path, msg)
        else:
            with open(f"./city_assistant/chat_history/{current_datetime}.txt", "w") as file:
                for message in self.chat_history:
                    file.write(message + "\n")


def run():
    root = tk.Tk()
    app = ChatApp(root)
    root.mainloop()
