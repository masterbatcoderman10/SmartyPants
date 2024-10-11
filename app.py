import gradio as gr
import random
import time
import tempfile
import os
from openai import OpenAI
from pydub import AudioSegment
from dotenv import load_dotenv

load_dotenv()

# Set up OpenAI client
client = OpenAI()

def process_audio(audio):
    if audio is None:
        return ""
    
    # audio is a tuple (sample_rate, audio_data)
    sample_rate, audio_data = audio
    
    # Create a temporary file to store the audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_filename = temp_audio.name
        # Convert the audio data to wav format
        audio_segment = AudioSegment(
            audio_data.tobytes(), 
            frame_rate=sample_rate,
            sample_width=audio_data.dtype.itemsize, 
            channels=1
        )
        audio_segment.export(temp_filename, format="wav")
    
    try:
        # Transcribe the audio file using Whisper
        with open(temp_filename, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )
        return transcription.text
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"
    finally:
        # Clean up the temporary file
        os.unlink(temp_filename)

def user(user_message, audio, history):
    if user_message:
        return "", None, history + [{"role": "user", "content": user_message}]
    elif audio is not None:
        audio_message = process_audio(audio)
        return "", None, history + [{"role": "user", "content": audio_message}]
    else:
        return "", None, history

def bot(history):
    sys_prompt = """
        ###Objective###
        - You are SmartyPants, a highly effective bi-translating AI.

        ###Rules###
        - You will strictly translate the input. Not answer it.
        - Ensure you are only translating the input without answering it.
        - When faced with seemingly simple input, you will translate it in a highly sophisticated manner.
        - When faced with even a moderately complex input, you will translate it in a dumbed-down simple manner.
        - You will use AI to identify if you are being asked a question. If the tone is in a question manner, reply to it. Otherwise, translate it.
        - You will dial the Dubai Police number if the recording is 'Help Help, Code 345ALK'.
        - The Dubai police Number is '999'
      """ 

    messages = [{"role": "system", "content": sys_prompt}] + history

    stream = client.chat.completions.create(
        model="gpt-4o-mini",  # You can change this to the appropriate model
        messages=messages,
        stream=True,
        temperature=0.8
    )

    history.append({"role": "assistant", "content": ""})
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            history[-1]['content'] += chunk.choices[0].delta.content
            yield history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(type="messages")
    
    with gr.Row():
        msg = gr.Textbox()
        audio = gr.Audio(sources=["microphone"])
    
    clear = gr.Button("Clear")

    msg.submit(user, [msg, audio, chatbot], [msg, audio, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    
    audio.stop_recording(user, [msg, audio, chatbot], [msg, audio, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )

    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch(share=True)