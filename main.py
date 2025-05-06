from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rag import answer_query

app = FastAPI()

class ChatRequest(BaseModel):
    question: str

@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chatbot</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            #chatbox { width: 100%; height: 300px; border: 1px solid #ccc; padding: 10px; overflow-y: scroll; margin-bottom: 10px; }
            #userinput { width: 80%; padding: 10px; }
            #sendbtn { padding: 10px 20px; }
        </style>
    </head>
    <body>
        <h2>Chatbot</h2>
        <div id="chatbox"></div>
        <input type="text" id="userinput" placeholder="Type your question..." />
        <button id="sendbtn">Send</button>
        <script>
            const chatbox = document.getElementById('chatbox');
            const userinput = document.getElementById('userinput');
            const sendbtn = document.getElementById('sendbtn');
            function appendMessage(sender, text) {
                chatbox.innerHTML += `<b>${sender}:</b> ${text}<br/>`;
                chatbox.scrollTop = chatbox.scrollHeight;
            }
            sendbtn.onclick = async function() {
                const question = userinput.value;
                if (!question) return;
                appendMessage('You', question);
                userinput.value = '';
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question })
                });
                const data = await response.json();
                appendMessage('Bot', data.answer?.answer || data.answer || 'No response');
            };
            userinput.addEventListener('keydown', function(e) {
                if (e.key === 'Enter') sendbtn.click();
            });
        </script>
    </body>
    </html>
    """

@app.post("/chat")
def chat(request: ChatRequest):
    result = answer_query(request.question)
    print(result)
    return {"answer": result["answer"].content} 