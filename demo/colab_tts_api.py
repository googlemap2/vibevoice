#!/usr/bin/env python
"""
Colab-friendly FastAPI server for VibeVoice preset-voice TTS.

Typical usage in Google Colab:
  !pip install -e .[streamingtts] fastapi uvicorn pyngrok
  !python demo/colab_tts_api.py --model_path microsoft/VibeVoice-Realtime-0.5B --port 8000 --public
"""

import argparse
import base64
import copy
import glob
import io
import os
import threading
import time
import wave
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor

try:
    from pyngrok import ngrok  # type: ignore
except Exception:  # pragma: no cover
    ngrok = None


def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int = 24000) -> bytes:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())
    return buffer.getvalue()


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text to synthesize")
    voice: str = Field(default="en-Carter_man", description="Preset voice name")
    cfg_scale: float = Field(default=1.5, ge=1.0, le=3.0)
    return_base64: bool = Field(default=False)


class VibeVoiceColabService:
    def __init__(self, model_path: str, device: str, inference_steps: int = 5):
        self.model_path = model_path
        self.device = device
        self.inference_steps = inference_steps
        self.sample_rate = 24000
        self.voice_presets = self._load_voice_presets()
        self.voice_cache: Dict[str, object] = {}
        self.lock = threading.Lock()

        print(f"[startup] Loading processor from {self.model_path}")
        self.processor = VibeVoiceStreamingProcessor.from_pretrained(self.model_path)

        if self.device == "cuda":
            dtype = torch.bfloat16
            attn = "flash_attention_2"
            device_map = "cuda"
        else:
            dtype = torch.float32
            attn = "sdpa"
            device_map = "cpu"

        print(f"[startup] Loading model on {self.device} ({dtype}, {attn})")
        try:
            self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
                device_map=device_map,
                attn_implementation=attn,
            )
        except Exception:
            if self.device == "cuda":
                print("[startup] flash_attention_2 failed, fallback to sdpa")
                self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="cuda",
                    attn_implementation="sdpa",
                )
            else:
                raise

        self.model.eval()
        self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)
        print(f"[startup] Ready. {len(self.voice_presets)} voices available.")

    def _load_voice_presets(self) -> Dict[str, str]:
        voices_dir = Path(__file__).parent / "voices" / "streaming_model"
        if not voices_dir.exists():
            raise RuntimeError(f"Voices directory not found: {voices_dir}")

        voices: Dict[str, str] = {}
        for pt_file in glob.glob(str(voices_dir / "**" / "*.pt"), recursive=True):
            key = Path(pt_file).stem
            voices[key] = os.path.abspath(pt_file)

        if not voices:
            raise RuntimeError(f"No voice presets found in {voices_dir}")

        return dict(sorted(voices.items()))

    def _get_prompt_cache(self, voice: str):
        if voice not in self.voice_presets:
            raise ValueError(f"Unknown voice: {voice}")
        if voice not in self.voice_cache:
            self.voice_cache[voice] = torch.load(
                self.voice_presets[voice],
                map_location=self.device,
                weights_only=False,
            )
        return self.voice_cache[voice]

    def tts(self, text: str, voice: str, cfg_scale: float = 1.5) -> np.ndarray:
        prompt_cache = self._get_prompt_cache(voice)
        inputs = self.processor.process_input_with_cached_prompt(
            text=text.strip(),
            cached_prompt=prompt_cache,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=float(cfg_scale),
                tokenizer=self.processor.tokenizer,
                generation_config={"do_sample": False},
                verbose=False,
                all_prefilled_outputs=copy.deepcopy(prompt_cache),
            )

        audio = outputs.speech_outputs[0]
        if torch.is_tensor(audio):
            audio = audio.detach().cpu().to(torch.float32).numpy()
        return np.asarray(audio, dtype=np.float32).reshape(-1)


def create_app(service: VibeVoiceColabService) -> FastAPI:
    app = FastAPI(title="VibeVoice Colab API", version="1.0")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "device": service.device,
            "model_path": service.model_path,
            "voices": len(service.voice_presets),
        }

    @app.get("/voices")
    def voices():
        return {"voices": list(service.voice_presets.keys())}

    @app.post("/tts")
    def tts(req: TTSRequest):
        if not req.text.strip():
            raise HTTPException(status_code=400, detail="text must not be empty")
        if req.voice not in service.voice_presets:
            raise HTTPException(
                status_code=400,
                detail=f"voice must be one of: {', '.join(service.voice_presets.keys())}",
            )

        started = time.time()
        try:
            with service.lock:
                audio = service.tts(req.text, req.voice, req.cfg_scale)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"TTS failed: {exc}") from exc

        wav_bytes = audio_to_wav_bytes(audio, sample_rate=service.sample_rate)
        elapsed = round(time.time() - started, 3)

        if req.return_base64:
            return JSONResponse(
                {
                    "voice": req.voice,
                    "sample_rate": service.sample_rate,
                    "duration_sec": round(len(audio) / service.sample_rate, 3),
                    "latency_sec": elapsed,
                    "audio_base64": base64.b64encode(wav_bytes).decode("utf-8"),
                }
            )

        headers = {
            "X-Voice": req.voice,
            "X-Sample-Rate": str(service.sample_rate),
            "X-Latency-Sec": str(elapsed),
        }
        return Response(content=wav_bytes, media_type="audio/wav", headers=headers)

    return app


def open_public_tunnel(port: int, authtoken: Optional[str]) -> Optional[str]:
    if ngrok is None:
        print("[ngrok] pyngrok is not installed. Skipping public tunnel.")
        return None
    if authtoken:
        ngrok.set_auth_token(authtoken)
    tunnel = ngrok.connect(port, "http")
    public_url = tunnel.public_url
    print(f"[ngrok] Public URL: {public_url}")
    return public_url


def parse_args():
    parser = argparse.ArgumentParser(description="VibeVoice Colab FastAPI server")
    parser.add_argument(
        "--model_path",
        type=str,
        default="microsoft/VibeVoice-Realtime-0.5B",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument("--inference_steps", type=int, default=5)
    parser.add_argument(
        "--public",
        action="store_true",
        help="Open public tunnel with ngrok",
    )
    parser.add_argument(
        "--ngrok_authtoken",
        type=str,
        default=None,
        help="Optional ngrok authtoken",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    service = VibeVoiceColabService(
        model_path=args.model_path,
        device=device,
        inference_steps=args.inference_steps,
    )
    app = create_app(service)

    if args.public:
        open_public_tunnel(args.port, args.ngrok_authtoken)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
