import split_srt_caption  # Ensure it is in the same directory as this script
from pydub import AudioSegment
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import click
import os
import time
from opencc import OpenCC
import re
import threading
import tqdm
from yt_dlp import YoutubeDL

def download_youtube_video(video_url):
    """
    Downloads a YouTube video using yt-dlp with specific options.

    Args:
        video_url (str): URL of the YouTube video.

    Returns:
        str: The filename of the downloaded video, including the full path.
    """

    # Change to the current directory of the Python script
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    print('Script directory:', script_directory)

    output_directory = os.getcwd()
    print('Output directory:', output_directory)

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': os.path.join(output_directory, '%(title)s.%(ext)s'),  # Output filename template with full path
        'merge_output_format': 'mp4',  # Merge to MP4 format
        'addmetadata': True,  # Add metadata to the output file
        'embedthumbnail': True,  # Embed thumbnail in the video
        'postprocessors': [
            {
                'key': 'FFmpegEmbedSubtitle',
            },
            {
                'key': 'FFmpegMetadata',
            },
            {
                'key': 'EmbedThumbnail',
                'already_have_thumbnail': False,
            },
        ],
    }


    # Download the video and get the filename
    with YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=True)
        filename = ydl.prepare_filename(info_dict)

    print('Downloaded file:', filename)
    return filename
    


def load_model():
    device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        device_map=device,
        attn_implementation="flash_attention_2" if device == "cuda:0" else None
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    # Ensure pad_token is set and different from eos_token
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token_id + 1

    # Set language explicitly
    processor.feature_extractor.language = "zh"  # Set language to Chinese    
    model.generation_config.language = "<|zh|>"  
    model.generation_config.task = "transcribe"
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id

    # Create pipeline without passing generate_kwargs to avoid TypeError
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=16,  # Set based on your device's capacity
        torch_dtype=torch_dtype,
        device_map="auto"
    )
    # Optionally set attributes after pipeline creation
    pipe.model.config.no_speech_threshold = 0.6

    return pipe


def transcribe_audio(pipe, audio_file):
    outputs = pipe(audio_file, return_timestamps=True)
    return outputs


def process_transcription(outputs):
    full_transcription = ""
    for chunk in outputs['chunks']:
        full_transcription += chunk['text'].strip() + "\n"

    full_transcription = convertcc(full_transcription)
    return full_transcription


def save_srt(outputs, audio_file):
    output_srt = ""
    for index, chunk in enumerate(outputs['chunks']):
        start_time_srt = seconds_to_srt_time_format(chunk['timestamp'][0])
        if chunk['timestamp'][1] is None:
            end_time_srt = start_time_srt  # If the end time is None, set it to the start time
        else:
            end_time_srt = seconds_to_srt_time_format(chunk['timestamp'][1])
        output_srt += f"{index + 1}\n"
        output_srt += f"{start_time_srt} --> {end_time_srt}\n"
        output_srt += f"{chunk['text'].strip()}\n\n"

    output_srt = convertcc(output_srt)

    audio_file_name = os.path.splitext(os.path.basename(audio_file))[0]
    srt_filename = f"{audio_file_name}.srt"
    with open(srt_filename, 'w', encoding='UTF-8') as srt_file:
        srt_file.write(output_srt)

    split_srt_caption.process_srt_file(srt_filename)
    return srt_filename


def save_txt(srt_filename):
    txt_filename = os.path.splitext(srt_filename)[0] + ".txt"
    srt_to_text(srt_filename, txt_filename)
    return txt_filename


def seconds_to_srt_time_format(seconds):
    if seconds is None:
        raise ValueError("Seconds cannot be None")
    hours = int(seconds // 3600)
    seconds_remainder = seconds % 3600
    minutes = int(seconds_remainder // 60)
    seconds_remainder = seconds_remainder % 60
    seconds_int = int(seconds_remainder)
    milliseconds = int((seconds_remainder - seconds_int) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds_int:02d},{milliseconds:03d}"


def convertcc(text):
    cc = OpenCC('s2hk')  # Convert Simplified Chinese to Traditional Chinese (Hong Kong)
    converted = cc.convert(text)
    return converted


def srt_to_text(srt_file_name, output_text_file_name):
    try:
        with open(srt_file_name, 'r', encoding='utf-8') as srt_file:
            lines = srt_file.readlines()

        # Extract text lines (skip SRT counter and timecodes)
        text_lines = [line.strip() for line in lines if line.strip() and not line.strip().isdigit() and '-->' not in line]

        # Write the extracted text to a new text file
        with open(output_text_file_name, 'w', encoding='utf-8') as text_file:
            text_file.write("\n".join(text_lines))

        print(f'Successfully extracted text to {output_text_file_name}')
    except Exception as e:
        print(f'An error occurred: {e}')


def get_audio_duration(audio_file_path):
    temp_wav = None
    if not audio_file_path.endswith(('.wav', '.flac', '.ogg', '.aiff')):
        # Convert to WAV using pydub
        audio = AudioSegment.from_file(audio_file_path)
        temp_wav = "temp_audio.wav"
        audio.export(temp_wav, format="wav")
        audio_file_path = temp_wav

    # Use soundfile to get duration
    with sf.SoundFile(audio_file_path) as audio_file:
        frames = len(audio_file)
        samplerate = audio_file.samplerate
        duration = frames / float(samplerate)

    # Clean up temporary file if created
    if temp_wav and os.path.exists(temp_wav):
        os.remove(temp_wav)

    return duration


def update_progress_bar(total_chunks, done_event):
    with tqdm.tqdm(total=total_chunks, desc="Processing audio file", leave=True, ncols=75, mininterval=0.2, colour='green') as progress_bar:
        while not done_event.is_set():
            time.sleep(1)
            progress_bar.update(1)
        progress_bar.close()


@click.command()
@click.argument('video_url')
#@click.argument('audio_file')
def main(video_url):
    print(video_url)
    audio_file=download_youtube_video(video_url)
    # Start timing (after model loading)
    done_event = threading.Event()
    start_time = time.perf_counter()
    click.echo("Loading model...")

    # Load model
    pipe = load_model()
    click.echo("Model loaded.")

    # Estimate audio duration and calculate the number of chunks to process
    audio_duration = get_audio_duration(audio_file)
    chunk_length = 30  # Process 30 seconds of audio at a time
    total_chunks = int(audio_duration // chunk_length) + 1

    # Start a thread to update the progress bar
    done_event.clear()
    progress_thread = threading.Thread(target=update_progress_bar, args=(total_chunks, done_event))
    progress_thread.start()

    # Transcribe audio
    outputs = transcribe_audio(pipe, audio_file)

    done_event.set()  # Stop the progress thread
    progress_thread.join()  # Ensure the progress thread finishes before continuing

    # Process transcription results
    full_transcription = process_transcription(outputs)
    print("Full transcription:\n", full_transcription)

    # Save SRT file
    srt_filename = save_srt(outputs, audio_file)
    # Save TXT file
    save_txt(srt_filename)

    # End timing (after transcription completes)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    click.echo(f"Transcription took {elapsed_time:.2f} seconds.")


if __name__ == "__main__":
    main()
