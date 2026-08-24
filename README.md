# **Enterprise Real-Time Voice AI Agent**

A production-grade, full-duplex conversational voice agent built for seamless restaurant order processing. This system achieves human-like conversational fluidity by decoupling audio processing from LLM logic, allowing for instant interruptions (barge-in) and highly reliable state transitions.

## **Overview**

This project serves as the interactive voice backend and frontend for an automated ordering system. It listens to customer audio in real-time, manages a complex transactional state (greetings, cart management, checkout, and fulfillment), and streams zero-latency voice responses directly back to the browser.

## **Tech Stack**

* **Backend:** Python, FastAPI, WebSockets  
*   
* **State Management:** LangGraph (State Machine Architecture)  
*   
* **LLM Inference:** Groq API (Llama-3 / GPT-OSS-20b)  
*   
* **Speech-to-Text (STT):** Deepgram Listen (Live Streaming)  
*   
* **Text-to-Speech (TTS):** Deepgram Speak (Aura Linear16 PCM)  
*   
* **Frontend:** HTML5, CSS3, Vanilla JavaScript, Web Audio API

## **Architecture Evolution**

### **The Previous Approach: Sequential Flat-Prompting**

Initially, the architecture relied on a flat, sequential prompt structure where a single conversation history array was fed into the LLM.

* **The Problem:** Tool execution (like adding items to the cart) ran as detached background tasks parallel to the audio generation. This created severe race conditions where the agent would speak confirmations before the database actually updated.  
    
* **State Pollution:** Global variables were used for the cart and conversation history, which caused memory leaks and cross-contamination between different concurrent callers.  
* **Latency:** Audio was buffered as complete MP3 sentences before being dispatched to the frontend, creating unnatural pauses.


### **The Current Approach: Decoupled State Machine & Zero-Latency Streaming**

The system has been completely re-engineered to separate the conversational intelligence from the audio transport layer.

* **LangGraph State Machine (The Brain):** The conversation is now strictly divided into phases (Greeting, Order, Payment, Fulfillment, Info, Complete). A LangGraph state dictionary isolates memory per WebSocket connection. The graph executes JSON tool calls synchronously *before* the agent speaks, ensuring 100% accurate cart calculations.  
    
* **Dual-Trigger Barge-In:** Deepgram's STT is utilized in a dual-trigger loop. *Interim results* (the millisecond a user speaks) instantly halt the frontend audio playback and cancel backend generation. *Final results* wait for the user to finish their sentence before advancing the LangGraph state.  
    
* **Raw PCM Audio Streaming (The Mouth):** MP3 buffering has been entirely replaced. The backend requests linear16 PCM audio chunks at 24kHz and pipes the raw bytes instantly through the WebSocket. The frontend utilizes the Web Audio API to seamlessly schedule and stitch these float32 frames together for true zero-latency playback.  
* 

## **Local Setup & Installation Guide**

Follow these steps to run the voice agent on your local machine.

### **1\. Prerequisites**

You will need Python 3.10+ installed on your system, along with active API keys for Deepgram and Groq.


### **2\. Clone the Repository**

Open your terminal and clone the repository to your local machine:

Bash  
git clone https://github.com/your-username/voice-ai-order-processing-agent.git  
cd voice-ai-order-processing-agent

### **3\. Set Up a Virtual Environment**

Create and activate a Python virtual environment to keep dependencies isolated:

Bash  
python \-m venv .venv  
\# On Windows:  
.venv\\Scripts\\activate  
\# On macOS/Linux:  
source .venv/bin/activate

### **4\. Install Dependencies**

Install the required packages using pip:

Bash  
pip install fastapi uvicorn deepgram-sdk groq langgraph python-dotenv numpy

### **5\. Environment Variables**

Create a .env file in the root directory of the project and add your API keys:

Code snippet  
DEEPGRAM\_API\_KEY=your\_deepgram\_api\_key\_here  
GROQ\_API\_KEY=your\_groq\_api\_key\_here

### **6\. Run the Server**

Launch the FastAPI application using Uvicorn:

Bash  
uvicorn src.server:app \--host 0.0.0.0 \--port 8000 \--reload

### **7\. Connect the Client**

Open your web browser and navigate to:  
http://localhost:8000  
Click **Connect Audio**, allow microphone permissions, and start speaking with the agent\!  
