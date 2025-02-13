import streamlit as st
import nltk
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

#load pre-traning hugging face model
chatbot = pipeline("text-generation",model="distilgpt2")

#define healthcare-specific response logic
def healthcare_chatbot(user_input):
    #rule base respond
    if "symptom" in user_input:
        return "Please consult doctor for accurate advice"
    elif "appointment" in user_input:
        return "Would you like to schedule appointment with the doctor?"
    elif "medication" in user_input:
        return "It's important to take prescribed medicines regularly. If you have concerns, consults your doctor."
    else:
        #for other input
        response = chatbot(user_input,max_length = 500,num_return_sequences=1)
        return response[0]["generated_text"]
    
#streamlit web app interface
def main():
    st.title("Healthcare Assistant Chatbot")
    user_input = st.text_input("How can I assist you today?")
    #chatbot respond
    if st.button("Submit"):
        if user_input:
            st.write("User: ",user_input)
            with st.spinner("Processing your query,Please wait........"):
                response = healthcare_chatbot(user_input)
            st.write("Healthcare Assistant: ",response)
        else:
            st.write("Please enter a message to get a response.")

if __name__ == "__main__":
    main()