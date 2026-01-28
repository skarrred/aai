#p3
#Creating a chatbot using advanced techniques like transformer models.

from transformers import pipeline

# Load QA model (explicitly choose a strong one)
question_answerer = pipeline("question-answering", model="deepset/roberta-base-squad2")

# General knowledge context
GK_CONTEXT = """
The current president of the United States is Joe Biden. The capital of India is New Delhi.
Mars is known as the Red Planet.
'Pride and Prejudice' was written by Jane Austen. The largest mammal in the world is the blue whale.
"""

def ask_gk_question(question, context=GK_CONTEXT):
    """
    Answers a general knowledge question based on a fixed context.
    Returns the best answer with confidence score.
    """
    result = question_answerer(question=question, context=context, top_k=1)
    return result["answer"], round(result["score"] * 100, 2)

# Chatbot Loop
print("GK Chatbot: Ask me a general knowledge question! Type 'exit' to end.\n")

while True:
    user_input = input("You: ")
    if user_input.strip().lower() == "exit":
        print("GK Chatbot: Goodbye!")
        break

    try:
        answer, confidence = ask_gk_question(user_input)
        print(f"GK Chatbot: {answer} (confidence: {confidence}%)\n")
    except Exception as e:
        print("GK Chatbot: Sorry, I couldn't process that question.\n")
