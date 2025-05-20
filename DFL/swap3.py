import cv2
import insightface
import numpy as np
import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox
import re
import shutil 


np.seterr(all='warn')
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def perform_face_swap(source_path, target_ref_path, video_path, output_base_name, start_time, end_time, video_number):
    model_path = r"C:\Users\QiXuan\Downloads\Video Test\DFL\inswapper_128.onnx"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please check the path.")
    swapper = insightface.model_zoo.get_model(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    face_analyser = insightface.app.FaceAnalysis()
    face_analyser.prepare(ctx_id=0, det_size=(512, 512)) 

    img = cv2.imread(source_path)
    if img is None:
        raise ValueError(f"Could not load source image: {source_path}")
    faces = face_analyser.get(img)
    if not faces:
        raise ValueError(f"No face detected in {source_path}")
    source_face = faces[0]

    img = cv2.imread(target_ref_path)
    if img is None:
        raise ValueError(f"Could not load reference image: {target_ref_path}")
    faces = face_analyser.get(img)
    if not faces:
        raise ValueError(f"No face detected in {target_ref_path}")
    target_ref_embedding = faces[0].normed_embedding

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open target video")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps 

    if start_time < 0 or end_time > video_duration or start_time >= end_time:
        raise ValueError(f"Invalid time range. Start time must be >= 0, end time must be <= {video_duration:.2f}, and start time must be less than end time.")

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    print(f"Face swapping will be applied from frame {start_frame} to {end_frame} (time {start_time:.2f}s to {end_time:.2f}s)")

    base_dir = r"C:\Users\QiXuan\Downloads\Video Test\DFL\vision-deepfake-test-set"
    existing_videos = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('video')]
    if existing_videos:
        max_number = max([int(d.replace('video', '')) for d in existing_videos if d.replace('video', '').isdigit()])
        video_number = f"video{max_number + 1:02d}"
    else:
        video_number = "video00" 

    output_dir = os.path.join(base_dir, video_number)
    fake_faces_dir = os.path.join(output_dir, "fake-faces")
    real_faces_dir = os.path.join(output_dir, "real-faces")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fake_faces_dir, exist_ok=True)
    has_multiple_faces = False

    source_video_name = os.path.basename(video_path)
    source_video_output_path = os.path.join(output_dir, source_video_name)
    counter = 1
    base_name, ext = os.path.splitext(source_video_name)
    while os.path.exists(source_video_output_path):
        source_video_output_path = os.path.join(output_dir, f"{base_name}_{counter}{ext}")
        counter += 1
    shutil.copy2(video_path, source_video_output_path) 
    print(f"Original source video copied to: {source_video_output_path}")

    out = None
    output_base_name = os.path.splitext(output_base_name)[0] 
    output_base_name = f"{output_base_name}_{video_number}" 
    output_video_path = os.path.join(output_dir, f"{output_base_name}.mp4")
    counter = 1
    while os.path.exists(output_video_path): 
        output_video_path = os.path.join(output_dir, f"{output_base_name}_{counter}.mp4")
        counter += 1

    def cosine_similarity(embedding1, embedding2):
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

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

    last_matched_id = None
    last_similarity = 0.0
    first_swapped_frame = None

    frame_count = 0
    start_time = time.time()
    last_log_time = start_time
    frames_processed_since_last_log = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count in [123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136]:
            cv2.imwrite(f"debug_frame_{frame_count}.jpg", frame)

        if frame_count == start_frame and out is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        if frame_count < start_frame or frame_count > end_frame:
            if frame_count > end_frame and out is not None:
                break 
            frame_count += 1
            frames_processed_since_last_log += 1
            continue

        faces = face_analyser.get(frame)
        if faces:
            if len(faces) > 1:
                has_multiple_faces = True
            for face in faces:
                face_id = hash(str(face.bbox))

                embedding = face.normed_embedding
                similarity = cosine_similarity(embedding, target_ref_embedding)

                similarity_threshold = 0.55
                if last_matched_id is not None and face_id == last_matched_id:
                    similarity_threshold = 0.5

                if similarity <= similarity_threshold:
                    print(f"Debug: Face not swapped at frame {frame_count}, similarity: {similarity:.3f}, threshold: {similarity_threshold}, last_matched_id: {last_matched_id == face_id}")
                    continue

                last_matched_id = face_id
                last_similarity = similarity

                # Try swapping with paste_back=True first
                swapped_frame = swapper.get(frame, face, source_face, paste_back=True)
                if swapped_frame is None or not isinstance(swapped_frame, np.ndarray):
                    print(f"Debug: Face swap failed with paste_back=True at frame {frame_count}, similarity: {similarity:.3f}, reason: Invalid swapper output")
                    # Fall back to manual swap with hair
                    swapped_region = swapper.get(frame, face, source_face, paste_back=False)
                    if swapped_region is None or not isinstance(swapped_region, np.ndarray):
                        print(f"Debug: Face swap failed with paste_back=False at frame {frame_count}, similarity: {similarity:.3f}, reason: Invalid swapper output")
                        continue
                else:
                    frame = swapped_frame
                    if first_swapped_frame is None:
                        first_swapped_frame = frame.copy()
                    continue  # Skip manual blending if paste_back works

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

        out.write(frame)
        frame_count += 1
        frames_processed_since_last_log += 1

        if frame_count % 100 == 0 or frame_count == end_frame:
            current_time = time.time()
            elapsed_time = current_time - start_time
            time_since_last_log = current_time - last_log_time
            avg_time_per_frame = time_since_last_log / frames_processed_since_last_log if frames_processed_since_last_log > 0 else 0

            trimmed_frames = end_frame - start_frame + 1
            segment_progress_percent = ((frame_count - start_frame) / trimmed_frames) * 100 if frame_count >= start_frame else 0
            frames_remaining = max(0, end_frame - frame_count)
            estimated_time_remaining = frames_remaining * avg_time_per_frame

            elapsed_minutes = int(elapsed_time // 60)
            elapsed_seconds = int(elapsed_time % 60)
            remaining_minutes = int(estimated_time_remaining // 60)
            remaining_seconds = int(estimated_time_remaining % 60)

            print(f"Progress: {segment_progress_percent:.2f}% | Frame {frame_count}/{end_frame} | "
                  f"Elapsed: {elapsed_minutes}m {elapsed_seconds}s | "
                  f"ETA: {remaining_minutes}m {remaining_seconds}s")

            last_log_time = current_time
            frames_processed_since_last_log = 0

    # Save the first swapped frame as fake1.jpg
    if first_swapped_frame is not None:
        fake_face_path = os.path.join(fake_faces_dir, "fake1.jpg")
        cv2.imwrite(fake_face_path, first_swapped_frame)

    # Save a real face if multiple faces detected
    if has_multiple_faces and faces:
        # Use cosine similarity to compare faces instead of direct comparison
        real_face = None
        if len(faces) > 1:
            similarity = cosine_similarity(faces[0].normed_embedding, source_face.normed_embedding)
            if similarity < 0.55:  # Use the same threshold as elsewhere in the code
                real_face = faces[0]
            else:
                real_face = faces[1] if len(faces) > 1 else None
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

#UI
class FaceSwapApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Swap Tool")
        self.root.geometry("600x400")

        self.source_path = tk.StringVar()
        self.target_ref_path = tk.StringVar()
        self.video_path = tk.StringVar()
        self.output_base_name = tk.StringVar(value="output")
        self.start_time = tk.StringVar()
        self.end_time = tk.StringVar()

        tk.Label(root, text="Source Face Image:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        tk.Button(root, text="Browse", command=self.browse_source).grid(row=0, column=1, padx=5, pady=5)
        tk.Label(root, textvariable=self.source_path).grid(row=0, column=2, padx=5, pady=5, sticky="w")

        # Reference face selection
        tk.Label(root, text="Reference Face Image:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        tk.Button(root, text="Browse", command=self.browse_target_ref).grid(row=1, column=1, padx=5, pady=5)
        tk.Label(root, textvariable=self.target_ref_path).grid(row=1, column=2, padx=5, pady=5, sticky="w")

        # Video selection
        tk.Label(root, text="Input Video:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        tk.Button(root, text="Browse", command=self.browse_video).grid(row=2, column=1, padx=5, pady=5)
        tk.Label(root, textvariable=self.video_path).grid(row=2, column=2, padx=5, pady=5, sticky="w")

        # Output video base name
        tk.Label(root, text="Output Video Base Name:").grid(row=3, column=0, padx=5, pady=5, sticky="e")
        tk.Entry(root, textvariable=self.output_base_name).grid(row=3, column=1, columnspan=2, padx=5, pady=5, sticky="we")

        # Start and end times
        tk.Label(root, text="Start Time (seconds):").grid(row=4, column=0, padx=5, pady=5, sticky="e")
        tk.Entry(root, textvariable=self.start_time).grid(row=4, column=1, columnspan=2, padx=5, pady=5, sticky="we")

        tk.Label(root, text="End Time (seconds):").grid(row=5, column=0, padx=5, pady=5, sticky="e")
        tk.Entry(root, textvariable=self.end_time).grid(row=5, column=1, columnspan=2, padx=5, pady=5, sticky="we")

        # Run button
        tk.Button(root, text="Run Face Swap", command=self.run_face_swap).grid(row=6, column=0, columnspan=3, pady=20)

    def browse_source(self):
        file_path = filedialog.askopenfilename(initialdir=r"C:\Users\QiXuan\Downloads\Video Test\DFL\Extracted", filetypes=[("Image files", "*.jpg *.png")])
        if file_path:
            self.source_path.set(file_path)

    def browse_target_ref(self):
        file_path = filedialog.askopenfilename(initialdir=r"C:\Users\QiXuan\Downloads\Video Test\DFL\Extracted", filetypes=[("Image files", "*.jpg *.png")])
        if file_path:
            self.target_ref_path.set(file_path)

    def browse_video(self):
        file_path = filedialog.askopenfilename(initialdir=r"C:\Users\QiXuan\Downloads\Video Test\DFL\vision-deepfake-test-set", filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if file_path:
            self.video_path.set(file_path)

    def run_face_swap(self):
        if not self.source_path.get():
            messagebox.showerror("Error", "Please select a source face image.")
            return
        if not self.target_ref_path.get():
            messagebox.showerror("Error", "Please select a reference face image.")
            return
        if not self.video_path.get():
            messagebox.showerror("Error", "Please select an input video.")
            return
        if not self.output_base_name.get():
            messagebox.showerror("Error", "Please specify an output video base name.")
            return

        try:
            start_time = float(self.start_time.get())
            end_time = float(self.end_time.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for start and end times.")
            return

        # Allow custom video number or use default/incremented
        custom_video_number = self.output_base_name.get().replace("output", "").strip()
        if custom_video_number and custom_video_number.startswith("video"):
            video_number = custom_video_number
        else:
            # Extract video number from filename or default to increment
            video_number = re.search(r'video(\d+)', os.path.basename(self.video_path.get()), re.IGNORECASE)
            if video_number:
                video_number = video_number.group(0)
            else:
                video_number = "video01"  # Default if no match

        # Compute the output directory path
        base_dir = r"C:\Users\QiXuan\Downloads\Video Test\DFL\vision-deepfake-test-set"
        output_dir = os.path.join(base_dir, video_number)

        try:
            perform_face_swap(self.source_path.get(), self.target_ref_path.get(), self.video_path.get(), self.output_base_name.get(), start_time, end_time, video_number)
            messagebox.showinfo("Success", f"Face swapping completed successfully. Output saved in '{output_dir}'.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Create and run the UI
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceSwapApp(root)
    root.mainloop()