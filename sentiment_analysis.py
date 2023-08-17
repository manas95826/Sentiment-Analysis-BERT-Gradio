import gradio as gr
from transformers import pipeline

senti_pipeline = pipeline("sentiment-analysis")

result= senti_pipeline("I am extremely happy to share this video with all of you")
result

sentiment_label = result[0]['label']
sentiment_score = result[0]['score']

formatted_output = f"This sentiment is {sentiment_label} with the probability {sentiment_score*100:.2f}%"
print(formatted_output)

app_inputs = gr.inputs.Textbox(lines=2, placeholder="Enter title here...")

def res(app_inputs):
  result= senti_pipeline(app_inputs)
  sentiment_label = result[0]['label']
  sentiment_score = result[0]['score']

  formatted_output = f"This sentiment is {sentiment_label} with the probability {sentiment_score*100:.2f}%"
  return formatted_output

interface = gr.Interface(fn=res, 
                        inputs=app_inputs,
                         outputs=gr.outputs.Textbox(label="Sentiment Analysis Result"),
         title="Sup, I'm a Sentiment Analyzer Babe",
    description="Enter a text and see the sentiment analysis result!")

interface.launch(share=True)

