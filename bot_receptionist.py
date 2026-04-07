"""
Day 3 - Pipecat Local Bot: AI Receptionist (Project 5 extended)
================================================================
Same Pipecat pipeline, but with a full restaurant receptionist persona
and function calling for: hours, location, menu, reservation intent.

How to run:
  python bot_receptionist.py

This is a preview of what Day 4 builds over the phone.
"""

import os
import json
import asyncio
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioParams
from pipecat.frames.frames import LLMMessagesFrame

load_dotenv(Path(__file__).parent.parent.parent / ".env")

SYSTEM_PROMPT = """You are a friendly AI receptionist for Mario's Italian Kitchen.
Answer calls, help with reservations, and answer common questions.

Rules:
- Speak in short, natural sentences (this is a phone call).
- Be warm, helpful, and professional.
- Never use bullet points or lists — speak conversationally.
- Collect: date, time, party size, and name for reservations.
- Available dinner times: 5:30 PM, 6:00 PM, 6:30 PM, 7:00 PM, 7:30 PM, 8:00 PM

Use your tools to answer questions about hours and location accurately."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_business_hours",
            "description": "Get the restaurant's business hours",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_location",
            "description": "Get the restaurant's address and location",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": "Check table availability for a date and time",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "The date (e.g., March 15)"},
                    "time": {"type": "string", "description": "Desired time (e.g., 7:00 PM)"},
                    "party_size": {"type": "integer", "description": "Number of guests"},
                },
                "required": ["date", "time", "party_size"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "confirm_reservation",
            "description": "Confirm and save a reservation",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "date": {"type": "string"},
                    "time": {"type": "string"},
                    "party_size": {"type": "integer"},
                },
                "required": ["name", "date", "time", "party_size"],
            },
        },
    },
]


async def handle_tool_call(function_name: str, args: dict) -> str:
    if function_name == "get_business_hours":
        return json.dumps({
            "monday_friday": "5:00 PM - 10:00 PM",
            "saturday": "4:00 PM - 11:00 PM",
            "sunday": "4:00 PM - 9:00 PM",
            "note": "Closed on major holidays",
        })
    elif function_name == "get_location":
        return json.dumps({
            "address": "123 Main Street, San Francisco, CA 94102",
            "neighborhood": "Downtown",
            "parking": "Street parking and Union Square garage nearby",
            "transit": "2 blocks from Powell Street BART/Muni",
        })
    elif function_name == "check_availability":
        # Mock — always available for demo
        date = args.get("date", "")
        time_str = args.get("time", "")
        size = args.get("party_size", 2)
        return json.dumps({
            "available": True,
            "date": date,
            "time": time_str,
            "party_size": size,
            "message": f"We have availability for {size} guests on {date} at {time_str}.",
        })
    elif function_name == "confirm_reservation":
        name = args.get("name", "")
        date = args.get("date", "")
        time_str = args.get("time", "")
        size = args.get("party_size", 2)
        conf_number = f"RES{datetime.now().strftime('%H%M%S')}"
        print(f"\n  ✅ RESERVATION SAVED: {name} | {size} people | {date} at {time_str} | Conf# {conf_number}")
        return json.dumps({
            "confirmed": True,
            "confirmation_number": conf_number,
            "name": name,
            "date": date,
            "time": time_str,
            "party_size": size,
        })
    return json.dumps({"error": "unknown function"})


async def run_receptionist():
    transport = LocalAudioTransport(
        LocalAudioParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.6)),
        )
    )

    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        live_options={"model": "nova-2", "language": "en-US", "smart_format": True},
    )

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        params=OpenAILLMService.InputParams(temperature=0.7, max_tokens=150),
    )

    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id="21m00Tcm4TlvDq8ikWAM",
        model="eleven_turbo_v2_5",
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    context = OpenAILLMContext(messages, tools=TOOLS)
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline([
        transport.input(),
        stt,
        context_aggregator.user(),
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=True),
    )

    @transport.event_handler("on_client_connected")
    async def on_connected(transport, client):
        await task.queue_frames([
            LLMMessagesFrame([{
                "role": "system",
                "content": "Greet the caller warmly as Mario's Italian Kitchen receptionist. One sentence.",
            }])
        ])

    print("\n" + "="*55)
    print("  Mario's Italian Kitchen — AI Receptionist")
    print("  Speak to make a reservation or ask questions.")
    print("  Ctrl+C to stop.")
    print("="*55 + "\n")

    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(run_receptionist())
