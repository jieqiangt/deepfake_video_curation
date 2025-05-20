import cv2
import insightface
import numpy as np
import os
import time
import re
import shutil
import argparse

np.seterr(all='warn')
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def perform_face_swap(source_paths, target_ref_paths, video_path, output_base_name, start_time, end_time, video_number):
    # Load the face swapper model
    model_path = r"C:\Users\QiXuan\Downloads\Video Test\DFL\inswapper_128.onnx"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please check the path.")
    swapper = insightface.model_zoo.get_model(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    # Initialize face analyser
    face_analyser = insightface.app.FaceAnalysis()
    face_analyser.prepare(ctx_id=0, det_size=(512, 512))  # Increased for better detection

    # Load multiple source faces (for swapping)
    source_faces = []
    for source_path in source_paths:
        img = cv2.imread(source_path)
        if img is None:
            raise ValueError(f"Could not load source image: {source_path}")
        faces = face_analyser.get(img)
        if not faces:
            raise ValueError(f"No face detected in {source_path}")
        source_faces.append(faces[0])  # Take the first face detected in each source image

    # Load multiple reference images for the target faces
    target_ref_embeddings = []
    for target_ref_path in target_ref_paths:
        img = cv2.imread(target_ref_path)
        if img is None:
            raise ValueError(f"Could not load reference image: {target_ref_path}")
        faces = face_analyser.get(img)
        if not faces:
            raise ValueError(f"No face detected in {target_ref_path}")
        target_ref_embeddings.append(faces[0].normed_embedding)

    # Ensure the number of source faces matches the number of reference faces
    if len(source_faces) != len(target_ref_embeddings):
        raise ValueError("The number of source faces must match the number of reference faces.")

    # Process video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open target video")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps  # Duration in seconds

    # Validate start and end times
    if start_time < 0 or end_time > video_duration or start_time >= end_time:
        raise ValueError(f"Invalid time range. Start time must be >= 0, end time must be <= {video_duration:.2f}, and start time must be less than end time.")

    # Convert times to frame numbers
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    print(f"Face swapping will be applied from frame {start_frame} to {end_frame} (time {start_time:.2f}s to {end_time:.2f}s)")

    # Set up output directory structure
    base_dir = r"C:\Users\QiXuan\Downloads\Video Test\DFL\vision-deepfake-test-set"
    # Find the highest existing video number to increment
    existing_videos = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('video')]
    if existing_videos:
        max_number = max([int(d.replace('video', '')) for d in existing_videos if d.replace('video', '').isdigit()])
        video_number = f"video{max_number + 1:02d}"  # e.g., video25
    else:
        video_number = "video00"  # Start from video00 if no videos exist

    output_dir = os.path.join(base_dir, video_number)
    fake_faces_dir = os.path.join(output_dir, "fake-faces")
    real_faces_dir = os.path.join(output_dir, "real-faces")

    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fake_faces_dir, exist_ok=True)
    has_multiple_faces = False  # Flag to check if multiple faces are detected

    # Copy the original source video to the output directory
    source_video_name = os.path.basename(video_path)
    source_video_output_path = os.path.join(output_dir, source_video_name)
    counter = 1
    base_name, ext = os.path.splitext(source_video_name)
    while os.path.exists(source_video_output_path):
        source_video_output_path = os.path.join(output_dir, f"{base_name}_{counter}{ext}")
        counter += 1
    shutil.copy2(video_path, source_video_output_path)
    print(f"Original source video copied to: {source_video_output_path}")

    # Set up video writer only when starting the trimmed segment
    out = None
    # Normalize output_base_name to ensure a single .mp4 extension
    output_base_name = os.path.splitext(output_base_name)[0]
    output_base_name = f"{output_base_name}_{video_number}"
    output_video_path = os.path.join(output_dir, f"{output_base_name}.mp4")
    counter = 1
    while os.path.exists(output_video_path):
        output_video_path = os.path.join(output_dir, f"{output_base_name}_{counter}.mp4")
        counter += 1

    # Cosine similarity for face matching
    def cosine_similarity(embedding1, embedding2):
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    # Simple color correction to match lighting
    def adjust_lighting(source_region, target_frame, bbox):
        x1, y1, x2, y2 = bbox
        target_region = target_frame[y1:y2, x1:x2]
        if target_region.size == 0:
            return source_region
        target_mean = np.mean(target_region, axis=(0, 1))
        source_mean = np.mean(source_region, axis=(0, 1))
        adjusted_region = source_region.copy().astype(float)
        for channel in range(3):
            adjusted_region[:, :, channel] = adjusted_region[:, :, channel] * (target_mean[channel] / (source_mean[channel] + 1e-6))
        adjusted_region = np.clip(adjusted_region, 0, 255).astype(np.uint8)
        return adjusted_region

    # Track the last matched faces for consistency
    last_matched_ids = [None] * len(target_ref_embeddings)  # One for each reference face
    last_similarities = [0.0] * len(target_ref_embeddings)
    first_swapped_frame = None

    # Process each frame with progress tracking
    frame_count = 0
    start_time = time.time()
    last_log_time = start_time
    frames_processed_since_last_log = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Save problematic frames if specified
        if frame_count in [123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136]:
            cv2.imwrite(f"debug_frame_{frame_count}.jpg", frame)

        # Initialize video writer when reaching the start frame
        if frame_count == start_frame and out is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # Skip frames before start_frame and after end_frame
        if frame_count < start_frame or frame_count > end_frame:
            if frame_count > end_frame and out is not None:
                break
            frame_count += 1
            frames_processed_since_last_log += 1
            continue

        # Perform face swapping within the specified range
        faces = face_analyser.get(frame)
        if faces:
            if len(faces) > 1:
                has_multiple_faces = True
            swapped_faces = set()  # Track which faces in this frame have been swapped
            for face in faces:
                face_id = hash(str(face.bbox))
                if face_id in swapped_faces:
                    continue  # Skip if this face was already swapped

                # Try matching this face to each reference embedding
                embedding = face.normed_embedding
                best_match_idx = -1
                best_similarity = 0.0
                for idx, ref_embedding in enumerate(target_ref_embeddings):
                    similarity = cosine_similarity(embedding, ref_embedding)
                    # Use temporal consistency
                    similarity_threshold = 0.55
                    if last_matched_ids[idx] is not None and face_id == last_matched_ids[idx]:
                        similarity_threshold = 0.5
                    if similarity > similarity_threshold and similarity > best_similarity:
                        best_match_idx = idx
                        best_similarity = similarity

                # If a match is found, perform the swap
                if best_match_idx != -1:
                    last_matched_ids[best_match_idx] = face_id
                    last_similarities[best_match_idx] = best_similarity

                    # Try swapping with paste_back=True first
                    swapped_frame = swapper.get(frame, face, source_faces[best_match_idx], paste_back=True)
                    if swapped_frame is None or not isinstance(swapped_frame, np.ndarray):
                        print(f"Debug: Face swap failed with paste_back=True at frame {frame_count}, similarity: {best_similarity:.3f}, reason: Invalid swapper output")
                        # Fall back to manual swap with hair
                        swapped_region = swapper.get(frame, face, source_faces[best_match_idx], paste_back=False)
                        if swapped_region is None or not isinstance(swapped_region, np.ndarray):
                            print(f"Debug: Face swap failed with paste_back=False at frame {frame_count}, similarity: {best_similarity:.3f}, reason: Invalid swapper output")
                            continue
                    else:
                        frame = swapped_frame
                        if first_swapped_frame is None:
                            first_swapped_frame = frame.copy()
                        swapped_faces.add(face_id)
                        continue

                    # Extract bounding box and expand to include hair
                    bbox = face.bbox.astype(int)
                    x1, y1, x2, y2 = bbox
                    hair_expansion_factor = 0.5
                    face_width = x2 - x1
                    face_height = y2 - y1
                    x1_exp = max(0, x1 - int(face_width * hair_expansion_factor))
                    y1_exp = max(0, y1 - int(face_height * hair_expansion_factor * 1.5))
                    x2_exp = min(frame.shape[1], x2 + int(face_width * hair_expansion_factor))
                    y2_exp = min(frame.shape[0], y2 + int(face_height * hair_expansion_factor * 0.5))

                    # Validate expanded bounding box
                    if x2_exp <= x1_exp or y2_exp <= y1_exp:
                        print(f"Debug: Invalid expanded bounding box at frame {frame_count}: ({x1_exp}, {y1_exp}, {x2_exp}, {y2_exp}), falling back to original bbox")
                        x1_exp, y1_exp, x2_exp, y2_exp = x1, y1, x2, y2
                        if x2 <= x1 or y2 <= y1:
                            print(f"Debug: Original bounding box also invalid at frame {frame_count}: ({x1}, {y1}, {x2}, {y2})")
                            continue

                    # Resize the swapped region to the bounding box
                    swapped_region = cv2.resize(swapped_region, (x2_exp - x1_exp, y2_exp - y1_exp))

                    # Adjust lighting for better blending
                    swapped_region = adjust_lighting(swapped_region, frame, (x1_exp, y1_exp, x2_exp, y2_exp))

                    # Create a smooth mask for blending
                    mask = np.zeros((y2_exp - y1_exp, x2_exp - x1_exp), dtype=np.uint8)
                    cv2.ellipse(mask, ((x2_exp - x1_exp) // 2, (y2_exp - y1_exp) // 2), 
                                ((x2_exp - x1_exp) // 2, (y2_exp - y1_exp) // 2), 0, 0, 360, 255, -1)
                    mask = cv2.GaussianBlur(mask, (21, 21), 0) / 255.0

                    # Paste the swapped region back with smooth blending
                    frame[y1_exp:y2_exp, x1_exp:x2_exp] = (frame[y1_exp:y2_exp, x1_exp:x2_exp] * (1 - mask[..., None]) + 
                                                           swapped_region * mask[..., None]).astype(np.uint8)

                    if first_swapped_frame is None:
                        first_swapped_frame = frame.copy()
                    swapped_faces.add(face_id)

        out.write(frame)
        frame_count += 1
        frames_processed_since_last_log += 1

        # Log progress every 100 frames
        if frame_count % 100 == 0 or frame_count == end_frame:
            current_time = time.time()
            elapsed_time = current_time - start_time
            time_since_last_log = current_time - last_log_time
            avg_time_per_frame = time_since_last_log / frames_processed_since_last_log if frames_processed_since_last_log > 0 else 0

            # Calculate progress for the trimmed segment
            trimmed_frames = end_frame - start_frame + 1
            segment_progress_percent = ((frame_count - start_frame) / trimmed_frames) * 100 if frame_count >= start_frame else 0
            frames_remaining = max(0, end_frame - frame_count)
            estimated_time_remaining = frames_remaining * avg_time_per_frame

            # Convert times to minutes and seconds
            elapsed_minutes = int(elapsed_time // 60)
            elapsed_seconds = int(elapsed_time % 60)
            remaining_minutes = int(estimated_time_remaining // 60)
            remaining_seconds = int(estimated_time_remaining % 60)

            print(f"Progress: {segment_progress_percent:.2f}% | Frame {frame_count}/{end_frame} | "
                  f"Elapsed: {elapsed_minutes}m {elapsed_seconds}s | "
                  f"ETA: {remaining_minutes}m {remaining_seconds}s")

            # Reset for the next log
            last_log_time = current_time
            frames_processed_since_last_log = 0

    # Save the first swapped frame as fake1.jpg
    if first_swapped_frame is not None:
        fake_face_path = os.path.join(fake_faces_dir, "fake1.jpg")
        cv2.imwrite(fake_face_path, first_swapped_frame)

    # Save a real face if multiple faces detected
    if has_multiple_faces and faces:
        # Use cosine similarity to compare faces against all source faces
        real_face = None
        if len(faces) > 1:
            for face in faces:
                face_id = hash(str(face.bbox))
                is_source_face = False
                for source_face in source_faces:
                    similarity = cosine_similarity(face.normed_embedding, source_face.normed_embedding)
                    if similarity >= 0.55:
                        is_source_face = True
                        break
                if not is_source_face:
                    real_face = face
                    break
        if real_face:
            os.makedirs(real_faces_dir, exist_ok=True)
            x1, y1, x2, y2 = real_face.bbox.astype(int)
            real_face_img = frame[y1:y2, x1:x2]
            real_face_path = os.path.join(real_faces_dir, "real1.jpg")
            cv2.imwrite(real_face_path, real_face_img)

    # Ensure video writer is released
    if out is not None:
        out.release()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Face swapping complete. Output saved as '{output_video_path}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face swapping script for multiple faces without UI.")
    parser.add_argument('--source_paths', type=str, nargs='+', required=True, 
                        help="Paths to the source face images (space-separated).")
    parser.add_argument('--target_ref_paths', type=str, nargs='+', required=True, 
                        help="Paths to the reference face images (space-separated).")
    parser.add_argument('--video_path', type=str, required=True, 
                        help="Path to the input video.")
    parser.add_argument('--output_base_name', type=str, default="output", 
                        help="Base name for the output video (default: output).")
    parser.add_argument('--start_time', type=float, default=0.0, 
                        help="Start time in seconds (default: 0.0).")
    parser.add_argument('--end_time', type=float, default=float('inf'), 
                        help="End time in seconds (default: end of video).")
    parser.add_argument('--video_number', type=str, default=None, 
                        help="Custom video number (e.g., video25), otherwise auto-incremented.")

    args = parser.parse_args()

    try:
        perform_face_swap(
            source_paths=args.source_paths,
            target_ref_paths=args.target_ref_paths,
            video_path=args.video_path,
            output_base_name=args.output_base_name,
            start_time=args.start_time,
            end_time=args.end_time,
            video_number=args.video_number if args.video_number else "video01"
        )
    except Exception as e:
        print(f"An error occurred: {str(e)}")