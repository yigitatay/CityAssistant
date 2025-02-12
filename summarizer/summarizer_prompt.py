summarizer_prompt = """

I will give you a conversation between an user and a chatbot named City Assistant where the chatbot is unable to solve the user's problem.
Summarize what is the problem in couple of sentences so the customer agent can understand what went wrong.

{conversation}
"""