# Realtime Facial Expression Recognition Demo (AAAI 2025)

This repository provides a real-time facial expression recognition demo based on the model proposed in our AAAI 2025 paper:  
[**"Navigating Label Ambiguity for Facial Expression Recognition in the Wild"**](https://arxiv.org/abs/2502.09993)

The model has been fine-tuned and deployed to operate on live webcam input, allowing real-time inference and on-screen visualization of facial expressions.

---

## Description

- Uses the trained NLA (Navigating Label Ambiguity) model
- Captures webcam input and detects faces
- Performs expression classification in real time
- Displays prediction results (e.g., emotion labels or softmax distribution) on the video stream

---

## Demo Video

Watch a short demo of the system in action:

[▶️ Demo Video]
<video src="https://github.com/user-attachments/assets/6746d16f-fdab-46d8-bbe5-3cb256d124b8" controls width="600">
  Your browser does not support the video tag.
</video>

---

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/jungyu0413/realtime_emotion
cd realtime_emotion
python main.py
