"""
Day 3 - Pipecat Local Voice Bot
================================
Real-time voice AI bot using your microphone and speakers.
Pipeline: Mic → VAD → Deepgram STT → OpenAI LLM → OpenAI TTS → Speaker

How to run:
  source ~/venv-pipecat/bin/activate
  python bot.py

Press Ctrl+C to stop.
"""

import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator,
    LLMUserResponseAggregator,
)
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.openai import OpenAILLMService, OpenAITTSService
from pipecat.transports.local.audio import LocalAudioTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.vad.silero import SileroVADAnalyzer
from pipecat.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMMessagesFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import Frame


class TranscriptLogger(FrameProcessor):
    """Logs transcripts and filters out short noise/echo fragments."""
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, TranscriptionFrame):
            text = frame.text.strip()
            # Ignore very short transcripts (likely echo or noise)
            if len(text.split()) < 3:
                print(f"[ignored short transcript: '{text}']")
                return   # drop frame, don't send to LLM
            print(f"\n>>> YOU SAID: '{text}'\n")
        await self.push_frame(frame, direction)

load_dotenv(Path(__file__).parent.parent.parent / ".env")

SYSTEM_PROMPT = """You are a friendly voice assistant for Mario's Italian Kitchen.
Keep ALL responses to 1-2 short sentences — this is a spoken phone conversation.
Be warm and natural. Never use bullet points or lists."""


async def run_bot():
    transport = LocalAudioTransport(
        TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_audio_passthrough=True,   # send audio to Deepgram even during VAD
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=1.2)),
        )
    )

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
    )

    tts = OpenAITTSService(
        api_key=os.getenv("OPENAI_API_KEY"),
        voice="nova",
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    context = OpenAILLMContext(messages)

    user_aggregator      = LLMUserResponseAggregator(messages)
    assistant_aggregator = LLMAssistantResponseAggregator(messages)

    pipeline = Pipeline([
        transport.input(),
        stt,
        TranscriptLogger(),
        user_aggregator,
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=True),
    )

    # Send greeting when bot starts
    await task.queue_frames([
        LLMMessagesFrame(messages + [{
            "role": "user",
            "content": "Greet me warmly. One sentence only.",
        }])
    ])

    print("\n" + "="*55)
    print("  Pipecat Voice Bot — Listening...")
    print("  Speak into your microphone.")
    print("  Press Ctrl+C to quit.")
    print("="*55 + "\n")

    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(run_bot())
