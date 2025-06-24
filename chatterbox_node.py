import os
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Optional
import logging

# Import directly from the chatterbox package
from .local_chatterbox.chatterbox.tts import ChatterboxTTS
from .local_chatterbox.chatterbox.vc import ChatterboxVC

from comfy.utils import ProgressBar

# Configure logging for better diagnostics
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Monkey patch torch.load to use MPS or CPU if map_location is not specified
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        # Determine the appropriate device (MPS for Mac, else CPU)
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        kwargs['map_location'] = torch.device(device)
    return original_torch_load(*args, **kwargs)

torch.load = patched_torch_load


class AudioNodeBase:
    """Base class for audio nodes with common utilities."""
    
    @staticmethod
    def create_empty_tensor(audio, frame_rate, height, width, channels=None):
        """Create an empty tensor with dimensions based on audio duration."""
        audio_duration = audio['waveform'].shape[-1] / audio['sample_rate']
        num_frames = int(audio_duration * frame_rate)
        if channels is None:
            return torch.zeros((num_frames, height, width), dtype=torch.float32)
        else:
            return torch.zeros((num_frames, height, width, channels), dtype=torch.float32)

# Text-to-Speech node
class FL_ChatterboxTTSNode(AudioNodeBase):
    """
    ComfyUI node for Chatterbox Text-to-Speech functionality.
    """
    _tts_model = None
    _tts_device = None
    
    @staticmethod
    def chunk_text(text, min_words=80, max_words=110):
        """
        Split text into chunks using intelligent sentence-aware algorithm.
        
        This method creates chunks that:
        1. Are between min_words and max_words in length
        2. End at sentence boundaries (., !, ?) when possible
        3. Fall back to hard word limits for very long sentences
        
        Args:
            text: The input text to split.
            min_words: Minimum words per chunk before looking for sentence breaks.
            max_words: Maximum words per chunk (hard limit for stability).
            
        Returns:
            List of text chunks with enhanced logging.
        """
        if not text or not text.strip():
            return ["You need to add some text for me to talk."]
        
        # Tokenize the text into words
        words = text.split()
        total_words = len(words)
        
        if total_words <= max_words:
            logger.info(f"Text is short enough ({total_words} words <= {max_words}), returning as single chunk")
            return [text]
        
        chunks = []
        current_position = 0
        chunk_number = 1
        
        logger.info(f"=== Smart Chunking Algorithm Started ===")
        logger.info(f"Total words: {total_words}, Window: {min_words}-{max_words} words")
        
        while current_position < total_words:
            # Determine the boundaries for this chunk
            chunk_start = current_position
            min_end = min(chunk_start + min_words, total_words)
            max_end = min(chunk_start + max_words, total_words)
            
            # If we're near the end of the text, take everything remaining
            if max_end >= total_words:
                chunk_words = words[chunk_start:]
                chunk_text = " ".join(chunk_words)
                chunks.append(chunk_text)
                
                logger.info(f"Chunk {chunk_number} (final): {len(chunk_words)} words - took remaining text")
                logger.info(f"Chunk {chunk_number} text preview: {chunk_text[:100]}...")
                break
            
            # Look for the best sentence break within our window
            best_split_position = None
            
            # Search backwards from max_end to min_end for sentence endings
            for pos in range(max_end - 1, min_end - 1, -1):  # Search backwards
                word = words[pos]
                if word.endswith(('.', '!', '?')):
                    best_split_position = pos + 1  # Include the word with punctuation
                    break
            
            # Determine final split position and reason
            if best_split_position is not None:
                # Found a good sentence break
                split_position = best_split_position
                chunk_words = words[chunk_start:split_position]
                split_reason = f"sentence break at '{words[split_position-1]}'"
            else:
                # No sentence break found, use hard limit
                split_position = max_end
                chunk_words = words[chunk_start:split_position]
                split_reason = f"hard limit (no sentence break in window)"
                logger.warning(f"Chunk {chunk_number}: No sentence break found in {min_words}-{max_words} word window, using hard split")
            
            # Create the chunk
            chunk_text = " ".join(chunk_words)
            chunks.append(chunk_text)
            
            # Enhanced logging for this chunk
            chunk_word_count = len(chunk_words)
            chunk_char_count = len(chunk_text)
            logger.info(f"Chunk {chunk_number}: {chunk_word_count} words, {chunk_char_count} chars - {split_reason}")
            logger.info(f"Chunk {chunk_number} text preview: {chunk_text[:100]}...")
            
            # Move to the next chunk
            current_position = split_position
            chunk_number += 1
        
        logger.info(f"=== Smart Chunking Complete: {len(chunks)} chunks created ===")
        
        return chunks
    
    @staticmethod
    def validate_audio_segment(audio_tensor, chunk_index, sample_rate=16000):
        """
        Validate an audio segment before concatenation to prevent corruption.
        
        Args:
            audio_tensor: The audio tensor to validate
            chunk_index: Index of the chunk for logging
            sample_rate: Expected sample rate
            
        Returns:
            Tuple of (is_valid, duration_seconds, error_message)
        """
        try:
            if audio_tensor is None:
                return False, 0.0, f"Chunk {chunk_index}: Audio tensor is None"
            
            if not isinstance(audio_tensor, torch.Tensor):
                return False, 0.0, f"Chunk {chunk_index}: Audio is not a torch.Tensor"
            
            if audio_tensor.numel() == 0:
                return False, 0.0, f"Chunk {chunk_index}: Audio tensor is empty"
            
            if torch.isnan(audio_tensor).any() or torch.isinf(audio_tensor).any():
                return False, 0.0, f"Chunk {chunk_index}: Audio contains NaN or Inf values"
            
            # Calculate duration
            if len(audio_tensor.shape) == 1:
                num_samples = audio_tensor.shape[0]
            elif len(audio_tensor.shape) == 2:
                num_samples = audio_tensor.shape[1]  # Assuming shape is [channels, samples]
            else:
                return False, 0.0, f"Chunk {chunk_index}: Unexpected audio tensor shape {audio_tensor.shape}"
            
            duration = num_samples / sample_rate
            
            # Check if duration is reasonable (not too short or too long)
            if duration < 0.1:  # Less than 100ms might be problematic
                return False, duration, f"Chunk {chunk_index}: Audio too short ({duration:.2f}s)"
            
            if duration > 60.0:  # Longer than 60s is suspicious
                return False, duration, f"Chunk {chunk_index}: Audio suspiciously long ({duration:.2f}s)"
            
            return True, duration, None
            
        except Exception as e:
            return False, 0.0, f"Chunk {chunk_index}: Validation error - {str(e)}"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Hello, this is a test."}),
                "exaggeration": ("FLOAT", {"default": 0.5, "min": 0.25, "max": 2.0, "step": 0.05}),
                "cfg_weight": ("FLOAT", {"default": 0.5, "min": 0.2, "max": 1.0, "step": 0.05}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.05, "max": 5.0, "step": 0.05}),
            },
            "optional": {
                "audio_prompt": ("AUDIO",),
                "use_cpu": ("BOOLEAN", {"default": False}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "message")
    FUNCTION = "generate_speech"
    CATEGORY = "ChatterBox"
    
    def generate_speech(self, text, exaggeration, cfg_weight, temperature, audio_prompt=None, use_cpu=False, keep_model_loaded=False):
        """
        Generate speech from text with enhanced chunking and diagnostics.
        
        Args:
            text: The text to convert to speech.
            exaggeration: Controls emotion intensity (0.25-2.0).
            cfg_weight: Controls pace/classifier-free guidance (0.2-1.0).
            temperature: Controls randomness in generation (0.05-5.0).
            audio_prompt: AUDIO object containing the reference voice for TTS voice cloning.
            use_cpu: If True, forces CPU usage even if CUDA is available.
            keep_model_loaded: If True, keeps the model loaded in memory after generation.
            
        Returns:
            Tuple of (audio, message)
        """
        # Split text into chunks with reduced size for better stability
        text_chunks = self.chunk_text(text, min_words=80, max_words=110)
        total_chunks = len(text_chunks)
        
        # Determine device to use
        device = "cpu" if use_cpu else ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        if use_cpu:
            message = "Using CPU for inference (GPU disabled)"
        elif torch.backends.mps.is_available() and device == "mps":
             message = "Using MPS (Mac GPU) for inference"
        elif torch.cuda.is_available() and device == "cuda":
             message = "Using CUDA (NVIDIA GPU) for inference"
        else:
            message = f"Using {device} for inference" # Should be CPU if no GPU found
        
        # Enhanced logging for diagnostics
        logger.info(f"=== ChatterBox TTS Processing Started ===")
        logger.info(f"Original text length: {len(text)} characters, {len(text.split())} words")
        logger.info(f"Text split into {total_chunks} chunks (80-110 words each)")
        message += f"\nText split into {total_chunks} chunks (80-110 words each)"
        
        # Log each chunk for debugging
        for i, chunk in enumerate(text_chunks):
            chunk_words = len(chunk.split())
            chunk_chars = len(chunk)
            logger.info(f"Chunk {i+1}/{total_chunks}: {chunk_words} words, {chunk_chars} chars")
            logger.info(f"Chunk {i+1} text: {chunk}")
        
        # Create temporary files for any audio inputs
        import tempfile
        temp_files = []
        
        # Create a temporary file for the audio prompt if provided
        audio_prompt_path = None
        if audio_prompt is not None:
            try:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_prompt:
                    audio_prompt_path = temp_prompt.name
                    temp_files.append(audio_prompt_path)
                
                # Save the audio prompt to the temporary file
                prompt_waveform = audio_prompt['waveform'].squeeze(0)
                torchaudio.save(audio_prompt_path, prompt_waveform, audio_prompt['sample_rate'])
                
                # Enhanced audio prompt logging
                prompt_duration = prompt_waveform.shape[-1] / audio_prompt['sample_rate']
                logger.info(f"Audio prompt: {audio_prompt_path}")
                logger.info(f"Audio prompt duration: {prompt_duration:.2f}s, sample rate: {audio_prompt['sample_rate']}Hz")
                message += f"\nUsing provided audio prompt for voice cloning"
                message += f"\nAudio prompt duration: {prompt_duration:.2f}s"
                
                # Debug: Check if the file exists and has content
                if os.path.exists(audio_prompt_path):
                    file_size = os.path.getsize(audio_prompt_path)
                    logger.info(f"Audio prompt file created: {file_size} bytes")
                    message += f"\nAudio prompt file created successfully: {file_size} bytes"
                else:
                    logger.warning("Audio prompt file was not created properly")
                    message += f"\nWarning: Audio prompt file was not created properly"
            except Exception as e:
                logger.error(f"Error creating audio prompt file: {str(e)}")
                message += f"\nError creating audio prompt file: {str(e)}"
                audio_prompt_path = None
        
        tts_model = None
        audio_segments = []
        valid_segments = []
        audio_data = {"waveform": torch.zeros((1, 2, 1)), "sample_rate": 16000} # Initialize with empty audio
        pbar = ProgressBar(100)
        
        try:
            # Load the TTS model or reuse if loaded and device matches
            if FL_ChatterboxTTSNode._tts_model is not None and FL_ChatterboxTTSNode._tts_device == device:
                tts_model = FL_ChatterboxTTSNode._tts_model
                logger.info(f"Reusing loaded TTS model on {device}")
                message += f"\nReusing loaded TTS model on {device}..."
            else:
                if FL_ChatterboxTTSNode._tts_model is not None:
                    logger.info("Unloading previous TTS model (device mismatch)")
                    message += f"\nUnloading previous TTS model (device mismatch)..."
                    FL_ChatterboxTTSNode._tts_model = None
                    FL_ChatterboxTTSNode._tts_device = None

                logger.info(f"Loading TTS model on {device}")
                message += f"\nLoading TTS model on {device}..."
                pbar.update_absolute(10) # Indicate model loading started
                tts_model = ChatterboxTTS.from_pretrained(device=device)
                pbar.update_absolute(20) # Indicate model loading finished

                if keep_model_loaded:
                    FL_ChatterboxTTSNode._tts_model = tts_model
                    FL_ChatterboxTTSNode._tts_device = device
                    logger.info("Model will be kept loaded in memory")
                    message += "\nModel will be kept loaded in memory."
                else:
                    logger.info("Model will be unloaded after use")
                    message += "\nModel will be unloaded after use."

            # Process each text chunk with enhanced logging
            for i, chunk in enumerate(text_chunks):
                chunk_progress = 20 + int((i / total_chunks) * 70)  # Progress from 20% to 90%
                pbar.update_absolute(chunk_progress)
                
                logger.info(f"=== Processing chunk {i+1}/{total_chunks} ===")
                logger.info(f"Chunk text: {chunk}")
                
                chunk_preview = chunk[:50] + "..." if len(chunk) > 50 else chunk
                message += f"\nProcessing chunk {i+1}/{total_chunks}: {chunk_preview}"
                
                try:
                    # Generate speech for this chunk
                    wav = tts_model.generate(
                        text=chunk,
                        audio_prompt_path=audio_prompt_path,
                        exaggeration=exaggeration,
                        cfg_weight=cfg_weight,
                        temperature=temperature,
                    )
                    
                    # Validate the generated audio segment
                    is_valid, duration, error_msg = self.validate_audio_segment(wav, i+1, tts_model.sr)
                    
                    if is_valid:
                        logger.info(f"Chunk {i+1} generated successfully: {duration:.2f}s, shape: {wav.shape}")
                        message += f" -> {duration:.2f}s ✓"
                        audio_segments.append(wav)
                        valid_segments.append(i+1)
                    else:
                        logger.warning(f"Chunk {i+1} validation failed: {error_msg}")
                        message += f" -> FAILED: {error_msg}"
                        # Skip this segment to prevent corruption
                        
                except Exception as chunk_error:
                    logger.error(f"Error generating chunk {i+1}: {str(chunk_error)}")
                    message += f" -> ERROR: {str(chunk_error)}"
                    # Continue with next chunk
            
            pbar.update_absolute(90) # Indicate all chunks processed
            
            # Concatenate valid audio segments with safety checks
            if audio_segments:
                logger.info(f"Concatenating {len(audio_segments)} valid segments (skipped {total_chunks - len(audio_segments)} failed segments)")
                
                try:
                    concatenated_wav = torch.cat(audio_segments, dim=1)  # Concatenate along time dimension
                    total_duration = concatenated_wav.shape[1] / tts_model.sr
                    
                    audio_data = {
                        "waveform": concatenated_wav.unsqueeze(0),  # Add batch dimension
                        "sample_rate": tts_model.sr
                    }
                    
                    logger.info(f"Final audio: {total_duration:.2f}s, {len(valid_segments)}/{total_chunks} chunks")
                    message += f"\nSpeech generated successfully from {len(valid_segments)}/{total_chunks} chunks"
                    message += f"\nFinal duration: {total_duration:.2f}s"
                    
                    if len(valid_segments) < total_chunks:
                        skipped_chunks = [i+1 for i in range(total_chunks) if (i+1) not in valid_segments]
                        message += f"\nSkipped failed chunks: {skipped_chunks}"
                        
                except Exception as concat_error:
                    logger.error(f"Error during concatenation: {str(concat_error)}")
                    message += f"\nError during concatenation: {str(concat_error)}"
            else:
                logger.error("No valid audio segments generated")
                message += f"\nNo valid audio segments generated from {total_chunks} chunks"
            
            return (audio_data, message)
            
        except RuntimeError as e:
            # Check for CUDA or MPS errors and attempt fallback to CPU
            error_str = str(e)
            fallback_to_cpu = False
            if "CUDA" in error_str and device == "cuda":
                logger.warning("CUDA error detected, falling back to CPU")
                message += "\nCUDA error detected during TTS. Falling back to CPU..."
                fallback_to_cpu = True
            elif "MPS" in error_str and device == "mps":
                logger.warning("MPS error detected, falling back to CPU")
                message += "\nMPS error detected during TTS. Falling back to CPU..."
                fallback_to_cpu = True

            if fallback_to_cpu:
                device = "cpu"
                # Unload previous model if it exists
                if FL_ChatterboxTTSNode._tts_model is not None:
                    logger.info("Unloading previous TTS model for CPU fallback")
                    message += f"\nUnloading previous TTS model for CPU fallback..."
                    FL_ChatterboxTTSNode._tts_model = None
                    FL_ChatterboxTTSNode._tts_device = None

                logger.info("Loading TTS model on CPU for fallback")
                message += f"\nLoading TTS model on {device}..."
                pbar.update_absolute(10) # Indicate model loading started (fallback)
                tts_model = ChatterboxTTS.from_pretrained(device=device)
                pbar.update_absolute(20) # Indicate model loading finished (fallback)
                
                # Retry processing all chunks with CPU and same validation
                audio_segments = []
                valid_segments = []
                for i, chunk in enumerate(text_chunks):
                    chunk_progress = 20 + int((i / total_chunks) * 70)  # Progress from 20% to 90%
                    pbar.update_absolute(chunk_progress)
                    
                    logger.info(f"CPU fallback: Processing chunk {i+1}/{total_chunks}")
                    
                    try:
                        wav = tts_model.generate(
                            text=chunk,
                            audio_prompt_path=audio_prompt_path,
                            exaggeration=exaggeration,
                            cfg_weight=cfg_weight,
                            temperature=temperature,
                        )
                        
                        # Validate the generated audio segment
                        is_valid, duration, error_msg = self.validate_audio_segment(wav, i+1, tts_model.sr)
                        
                        if is_valid:
                            logger.info(f"CPU fallback chunk {i+1}: {duration:.2f}s ✓")
                            audio_segments.append(wav)
                            valid_segments.append(i+1)
                        else:
                            logger.warning(f"CPU fallback chunk {i+1} failed: {error_msg}")
                            
                    except Exception as chunk_error:
                        logger.error(f"CPU fallback chunk {i+1} error: {str(chunk_error)}")
                
                pbar.update_absolute(90) # Indicate all chunks processed (fallback)
                
                # Concatenate all valid audio segments
                if audio_segments:
                    try:
                        concatenated_wav = torch.cat(audio_segments, dim=1)  # Concatenate along time dimension
                        total_duration = concatenated_wav.shape[1] / tts_model.sr
                        
                        audio_data = {
                            "waveform": concatenated_wav.unsqueeze(0),  # Add batch dimension
                            "sample_rate": tts_model.sr
                        }
                        
                        logger.info(f"CPU fallback success: {total_duration:.2f}s from {len(valid_segments)}/{total_chunks} chunks")
                        message += f"\nSpeech generated successfully after fallback from {len(valid_segments)}/{total_chunks} chunks."
                        message += f"\nFinal duration: {total_duration:.2f}s"
                        
                    except Exception as concat_error:
                        logger.error(f"CPU fallback concatenation error: {str(concat_error)}")
                        message += f"\nError during CPU fallback concatenation: {str(concat_error)}"
                
                return (audio_data, message)
            else:
                logger.error(f"TTS error (no fallback): {str(e)}")
                message += f"\nError during TTS: {str(e)}"
                return (audio_data, message)
        except Exception as e:
            logger.error(f"Unexpected TTS error: {str(e)}")
            message += f"\nAn unexpected error occurred during TTS: {str(e)}"
            return (audio_data, message)
        finally:
            # Clean up all temporary files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    logger.info(f"Cleaned up temp file: {temp_file}")
            # Safe VRAM cleanup: Only remove reference, let Python GC handle cleanup
            if not keep_model_loaded and FL_ChatterboxTTSNode._tts_model is not None:
                logger.info("Unloading TTS model as keep_model_loaded is False")
                message += "\nUnloading TTS model as keep_model_loaded is False."
                FL_ChatterboxTTSNode._tts_model = None
                FL_ChatterboxTTSNode._tts_device = None
                # Gentle cache cleanup (no forced CPU movement)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

        logger.info("=== ChatterBox TTS Processing Complete ===")
        pbar.update_absolute(100) # Ensure progress bar completes on success or error
        return (audio_data, message)



# Voice Conversion node
class FL_ChatterboxVCNode(AudioNodeBase):
    """
    ComfyUI node for Chatterbox Voice Conversion functionality.
    """
    _vc_model = None
    _vc_device = None
    

    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_audio": ("AUDIO",),
                "target_voice": ("AUDIO",),
            },
            "optional": {
                "use_cpu": ("BOOLEAN", {"default": False}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "message")
    FUNCTION = "convert_voice"
    CATEGORY = "ChatterBox"
    
    def convert_voice(self, input_audio, target_voice, use_cpu=False, keep_model_loaded=False):
        """
        Convert the voice in an audio file to match a target voice.
        
        Args:
            input_audio: AUDIO object containing the audio to convert.
            target_voice: AUDIO object containing the target voice.
            use_cpu: If True, forces CPU usage even if CUDA is available.
            keep_model_loaded: If True, keeps the model loaded in memory after conversion.
            
        Returns:
            Tuple of (audio, message)
        """
        # Determine device to use
        device = "cpu" if use_cpu else ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        if use_cpu:
            message = "Using CPU for inference (GPU disabled)"
        elif torch.backends.mps.is_available() and device == "mps":
             message = "Using MPS (Mac GPU) for inference"
        elif torch.cuda.is_available() and device == "cuda":
             message = "Using CUDA (NVIDIA GPU) for inference"
        else:
            message = f"Using {device} for inference" # Should be CPU if no GPU found
        
        # Create temporary files for the audio inputs
        import tempfile
        temp_files = []
        
        # Create a temporary file for the input audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_input:
            input_audio_path = temp_input.name
            temp_files.append(input_audio_path)
        
        # Save the input audio to the temporary file
        input_waveform = input_audio['waveform'].squeeze(0)
        torchaudio.save(input_audio_path, input_waveform, input_audio['sample_rate'])
        
        # Create a temporary file for the target voice
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_target:
            target_voice_path = temp_target.name
            temp_files.append(target_voice_path)
        
        # Save the target voice to the temporary file
        target_waveform = target_voice['waveform'].squeeze(0)
        torchaudio.save(target_voice_path, target_waveform, target_voice['sample_rate'])
        
        vc_model = None
        pbar = ProgressBar(100) # Simple progress bar for overall process
        try:
            # Load the VC model or reuse if loaded and device matches
            if FL_ChatterboxVCNode._vc_model is not None and FL_ChatterboxVCNode._vc_device == device:
                vc_model = FL_ChatterboxVCNode._vc_model
                message += f"\nReusing loaded VC model on {device}..."
            else:
                if FL_ChatterboxVCNode._vc_model is not None:
                    message += f"\nUnloading previous VC model (device mismatch)..."
                    FL_ChatterboxVCNode._vc_model = None
                    FL_ChatterboxVCNode._vc_device = None

                message += f"\nLoading VC model on {device}..."
                pbar.update_absolute(10) # Indicate model loading started
                vc_model = ChatterboxVC.from_pretrained(device=device)
                pbar.update_absolute(50) # Indicate model loading finished

                if keep_model_loaded:
                    FL_ChatterboxVCNode._vc_model = vc_model
                    FL_ChatterboxVCNode._vc_device = device
                    message += "\nModel will be kept loaded in memory."
                else:
                    message += "\nModel will be unloaded after use."

            # Convert voice
            message += f"\nConverting voice to match target voice"
            
            pbar.update_absolute(60) # Indicate conversion started
            converted_wav = vc_model.generate(
                audio=input_audio_path,
                target_voice_path=target_voice_path,
            )
            pbar.update_absolute(90) # Indicate conversion finished
            
        except RuntimeError as e:
            # Check for CUDA or MPS errors and attempt fallback to CPU
            error_str = str(e)
            fallback_to_cpu = False
            if "CUDA" in error_str and device == "cuda":
                message += "\nCUDA error detected during VC. Falling back to CPU..."
                fallback_to_cpu = True
            elif "MPS" in error_str and device == "mps":
                 message += "\nMPS error detected during VC. Falling back to CPU..."
                 fallback_to_cpu = True

            if fallback_to_cpu:
                device = "cpu"
                # Unload previous model if it exists
                if FL_ChatterboxVCNode._vc_model is not None:
                    message += f"\nUnloading previous VC model for CPU fallback..."
                    FL_ChatterboxVCNode._vc_model = None
                    FL_ChatterboxVCNode._vc_device = None

                message += f"\nLoading VC model on {device}..."
                pbar.update_absolute(10) # Indicate model loading started (fallback)
                vc_model = ChatterboxVC.from_pretrained(device=device)
                pbar.update_absolute(50) # Indicate model loading finished (fallback)
                # Note: keep_model_loaded logic is applied after successful generation
                # to avoid keeping a failed model loaded.

                converted_wav = vc_model.generate(
                    audio=input_audio_path,
                    target_voice_path=target_voice_path,
                )
                pbar.update_absolute(90) # Indicate conversion finished (fallback)
            else:
                # Re-raise if it's not a CUDA/MPS error or we're already on CPU
                message += f"\nError during VC: {str(e)}"
                # Return the original audio
                message += f"\nError: {str(e)}"
                pbar.update_absolute(100) # Ensure progress bar completes on error
                return (input_audio, message)
        except Exception as e:
             message += f"\nAn unexpected error occurred during VC: {str(e)}"
             empty_audio = {"waveform": torch.zeros((1, 2, 1)), "sample_rate": 16000}
             for temp_file in temp_files:
                 if os.path.exists(temp_file):
                     os.unlink(temp_file)
             pbar.update_absolute(100) # Ensure progress bar completes on error
             return (empty_audio, message)
        finally:
            # Clean up all temporary files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            # If keep_model_loaded is False, ensure model is not stored
            # This is done here to ensure model is only kept if generation was successful
            if not keep_model_loaded and FL_ChatterboxVCNode._vc_model is not None:
                 message += "\nUnloading VC model as keep_model_loaded is False."
                 FL_ChatterboxVCNode._vc_model = None
                 FL_ChatterboxVCNode._vc_device = None
                 # Gentle cache cleanup
                 if torch.cuda.is_available():
                     torch.cuda.empty_cache()
                 if torch.backends.mps.is_available():
                     torch.mps.empty_cache()

        # If generation was successful and keep_model_loaded is True, store the model
        if keep_model_loaded and vc_model is not None:
             FL_ChatterboxVCNode._vc_model = vc_model
             FL_ChatterboxVCNode._vc_device = device
             message += "\nModel will be kept loaded in memory."
        elif not keep_model_loaded and FL_ChatterboxVCNode._vc_model is not None:
             # This case handles successful generation when keep_model_loaded was True previously
             # but is now False. Ensure the model is unloaded.
             message += "\nUnloading VC model as keep_model_loaded is now False."
             FL_ChatterboxVCNode._vc_model = None
             FL_ChatterboxVCNode._vc_device = None
             # Gentle cache cleanup
             if torch.cuda.is_available():
                 torch.cuda.empty_cache()
             if torch.backends.mps.is_available():
                 torch.mps.empty_cache()

        # Create audio data structure for the output
        audio_data = {
            "waveform": converted_wav.unsqueeze(0),  # Add batch dimension
            "sample_rate": vc_model.sr if vc_model else 16000 # Use default sample rate if model loading failed
        }
        
        message += f"\nVoice converted successfully"
        pbar.update_absolute(100) # Ensure progress bar completes on success
        
        return (audio_data, message)

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "FL_ChatterboxTTS": FL_ChatterboxTTSNode,
    "FL_ChatterboxVC": FL_ChatterboxVCNode,
}

# Display names for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "FL_ChatterboxTTS": "FL Chatterbox TTS",
    "FL_ChatterboxVC": "FL Chatterbox VC",
}