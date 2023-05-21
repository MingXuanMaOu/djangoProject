import os

from django.shortcuts import render
from django.http import HttpResponse
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests
import time
import cv2
import numpy as np
from django.http import JsonResponse

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)

def hello(request):
    url = "https://cdn.discordapp.com/attachments/1095914193913389161/1109892127166582784/d3e6ff90ab91671c8a06206d9ea224b.png"
    image = Image.open(requests.get(url, stream=True).raw)

    start_time = time.time()

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"得到的时间：{execution_time}")

    return HttpResponse("Hello, World!")


def process_video(request):
    video_url = request.GET.get('video_url', '')
    cap = cv2.VideoCapture(video_url)

    frames_per_second = cap.get(cv2.CAP_PROP_FPS)
    frames_per_second = round(frames_per_second)  # round FPS to the nearest integer
    frame_count = 0

    # Extract the last number from the video URL
    url_number = video_url.split('/')[-1]

    # Create the directory if it doesn't exist
    dir_path = f'/tmp/{url_number}'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    results = []

    while (cap.isOpened()):
        ret, frame = cap.read()

        if ret:
            # Every second process a frame
            if frame_count % frames_per_second == 0:  # change from 3 to 0
                try:  # Add try-except for error handling
                    # Convert the image format from BGR to RGB
                    cv2.imwrite(f'{dir_path}/{frame_count}.jpg', frame)
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(image)

                    inputs = processor(images=image, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    outputs = model(**inputs)

                    target_sizes = torch.tensor([image.size[::-1]])
                    result = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[
                        0]

                    frame_results = []

                    for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
                        box = [round(i, 2) for i in box.tolist()]
                        frame_results.append(
                            {
                                "label": model.config.id2label[label.item()],
                                "confidence": round(score.item(), 3),
                                "location": box
                            }
                        )

                    # Add frame number and timestamp in seconds
                    results.append({
                        "frame": frame_count,
                        "timestamp": frame_count / frames_per_second,
                        "results": frame_results,
                    })
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")

            frame_count += 1
        else:
            break

    cap.release()

    return JsonResponse(results, safe=False)


def det_video(request):
    video_url = request.GET.get('video_url', '')
    cap = cv2.VideoCapture(video_url)

    # Calculate total frames of the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_second = int(cap.get(cv2.CAP_PROP_FPS))

    # Extract the last number from the video URL
    url_number = video_url.split('/')[-1]

    # Create the directory if it doesn't exist
    dir_path = f'/tmp/{url_number}'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    results = []

    for i in range(0, total_frames, frames_per_second):
        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)

        ret, frame = cap.read()

        if ret:
            # Save the frame as a jpg image
            cv2.imwrite(f'{dir_path}/{i}.jpg', frame)

            # Convert the image format from BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)

            target_sizes = torch.tensor([image.size[::-1]])
            result = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

            frame_results = []

            for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                frame_results.append(
                    {
                        "label": model.config.id2label[label.item()],
                        "confidence": round(score.item(), 3),
                        "location": box
                    }
                )

            # Add frame number and timestamp in seconds
            results.append({
                "frame": i,
                "timestamp": i / frames_per_second,
                "results": frame_results,
            })

    cap.release()

    return JsonResponse(results, safe=False)
