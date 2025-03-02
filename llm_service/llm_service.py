from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login


import re
import json


app = FastAPI()
token = '' # Get token from huggingface.co
login(token=token)

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,                   # Enable 4-bit loading
    bnb_8bit_compute_dtype="bfloat16",   # Compute in bfloat16 (better for speed & stability)
    bnb_8bit_use_double_quant=True,      # Reduces memory usage further
    bnb_8bit_quant_type="nf4"            # "nf4" recommended for best balance of accuracy/performance
)
modelName = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(modelName)
model = AutoModelForCausalLM.from_pretrained(modelName, 
        token=token,
        quantization_config=bnb_config,
        device_map="auto"
        )

tokenizer = AutoTokenizer.from_pretrained(modelName)

class GenerateRequest(BaseModel):
    prompt: str

@app.post("/generate")
def generate(request: GenerateRequest):
    prompt = request.prompt
    system_prompt = "You are a helpful assistant."
    fprompt = f"<s>[SYS]{system_prompt}[/SYS]\n[USR]{prompt}[/USR]\n[ASS]"
    inputs = tokenizer(fprompt, return_tensors="pt").to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=1000, temperature=0.2)
    return {"response": tokenizer.decode(outputs[0], skip_special_tokens=True)}


# Pydantic model for the request body
class AttributeExtractionRequest(BaseModel):
    text: str  # Unstructured Jira story text

# Pydantic model for the response
class AttributeExtractionResponse(BaseModel):
    title: Optional[str] = ''
    story_points: Optional[float] = ''
    description: Optional[str] = ''
    acceptance_criteria: Optional[List[str]] = []
    team: Optional[str] = ''
    issue_type: Optional[str] = ''
    priority: Optional[str] = ''
    labels: Optional[List[str]] = []
    components: Optional[List[str]] = []
    status: Optional[str] = ''
    created_date: Optional[str] = '' 
    updated_date: Optional[str] = ''
    assignee: Optional[str] = ''

@app.post("/extract_attributes")
def extract_attributes(request: AttributeExtractionRequest):
    """
    Extracts structured Jira attributes from unstructured text using the local Mistral LLM.
    """
    prompt = (
        """
        Extract the following fields from the Jira story below:
        - title: Title
        - description: Description
        - acceptance_criteria: Acceptance Criteria (as a list)
        - team: Team
        - story_points: Story Points
        - issue_type: Issue Type (Bug, Story or Spike)
        - priority: Priority (High, Medium, Low)
        - labels: Labels (as a list)
        - components: Components (as a list)
        - status: Status (Backlog, In Progress, Done, etc.)
        - created_date: Created Date (ISO format if mentioned)
        - updated_date: Updated Date (ISO format if mentioned)
        - assignee: Assignee (name or identifier)

        Jira Story:
        """ + request.text.strip() + "\n\n" +
        "Respond with a JSON object containing these fields. If any field is not mentioned, use null or an empty list."
    )

    # Call to the local Mistral LLM API (running on localhost at port 8006 as an example)
    try:
        response = requests.post("http://localhost:8005/generate", json={"prompt": prompt}).json()["response"]
        # Regex pattern to extract JSON block
        pattern = r'\{\s*[^}]*\}'

        # Find all JSON-like objects in the response
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            json_str = matches[-1]  # Get the last JSON block
            try:
                    extracted_json = json.loads(json_str)
                    print("Extracted JSON:", extracted_json)
            except json.JSONDecodeError:
                print("Error: Found JSON block is not valid.")
        else:
            print("No JSON object found.")


        return extracted_json
        
        # Assuming the Mistral model returns a JSON string in the 'generated_text' field
        extracted_attributes = response.json().get("generated_text", "{}")
        attributes = eval(extracted_attributes)  # Caution: Use json.loads if you can guarantee valid JSON
    
    except Exception as e:
        return {"error": f"Failed to extract attributes: {str(e)}"}

