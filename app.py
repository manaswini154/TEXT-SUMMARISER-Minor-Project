from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re   
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI(title = "Text Summariser", version = 1.0, description = "A simple text summarisation application using T5 model.")


# Model and tokenizer loading
model = T5ForConditionalGeneration.from_pretrained(r"C:/Users/Manaswini/OneDrive/Desktop/Text_summariser/saved_summary_model")
tokeniser = T5Tokenizer.from_pretrained(r"C:/Users/Manaswini/OneDrive/Desktop/Text_summariser/saved_summary_model")


# device configuration
if torch.cuda.is_available():
  device = torch.device("cuda")
elif torch.backends.mps.is_available():
  device = torch.device("mps")
else:
  device = torch.device("cpu")

model.to(device)
# templating
templates = Jinja2Templates(directory=".")

#schema for request body
class DialogueInput(BaseModel):
    dialogue: str

def clean_data(text):
    text = re.sub(r"\r\n", ' ', text)
    text = re.sub(r"<.*?", ' ', text)
    text = re.sub(r"\s+", ' ', text)
    text = text.strip().lower()
    return text

def summarise_dialogue(dialogue : str) -> str:
  #clean
  dialogue = clean_data(dialogue)

  #tokenize
  inputs = tokeniser(
      dialogue, padding = "max_length", max_length = 512, truncation = True, return_tensors = "pt"
  ).to(device)

  model.to(device)

  #generate summary
  target = model.generate(
      input_ids = inputs['input_ids'],
      attention_mask = inputs['attention_mask'],
      max_length = 150, num_beams = 4,
      early_stopping = True
                          )
  #decoded output
  summary = tokeniser.decode(target[0], skip_special_tokens = True)

  return summary

# API Endpoints
@app.post("/summarise/")
async def summarise(dailogue: DialogueInput):
    summary = summarise_dialogue(dailogue.dialogue)
    return {"summary": summary}
       

@app.get("/", response_class = HTMLResponse)
async def home (request: Request):
    return templates.TemplateResponse( request = request, name = "index.html")
