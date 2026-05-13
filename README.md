# Visionize

Visionize is a text-to-image generation web application built using Python, Flask, HTML, and Bootstrap. It uses the Hugging Face model `AbsoluteReality_v1.8.1` to generate high-quality AI images from natural language prompts.

The application provides a simple and responsive interface where users can enter a prompt and receive an AI-generated image in real time.

---

## Features

* AI-powered text-to-image generation
* Flask-based backend
* Responsive Bootstrap frontend
* Hugging Face Inference API integration
* PNG image generation and delivery
* Lightweight and easy to deploy

---

## Tech Stack

* Python
* Flask
* Bootstrap
* HTML/CSS
* Hugging Face Hub
* Pillow (PIL)

---

## Project Structure

```text
Visionize/
│
├── app.py
├── main.py
├── templates/
│   └── index.html
├── static/
│   ├── css/
│   └── js/
├── requirements.txt
└── README.md
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/Paraphylus/Visionize.git
cd Visionize
```

### Create a Virtual Environment

```bash
python -m venv venv
```

#### Activate the Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Set the Hugging Face Token

Create an environment variable named `HF_TOKEN`.

Linux/macOS:

```bash
export HF_TOKEN=your_token_here
```

Windows CMD:

```bash
set HF_TOKEN=your_token_here
```

Windows PowerShell:

```powershell
$env:HF_TOKEN="your_token_here"
```

You can generate a token from your Hugging Face account settings.

---

## Running the Application

```bash
python app.py
```

The server will start on:

```text
http://127.0.0.1:8000
```

---

## API Endpoint

### Generate Image

```http
GET /generateimages/<prompt>
```

### Example

```http
/generateimages/cyberpunk%20city%20at%20night
```

Response:

```text
PNG image file
```

---

## Model Used

Visionize uses the Hugging Face model:

```text
digiplay/AbsoluteReality_v1.8.1
```

This model is optimized for realistic AI-generated imagery and detailed visual outputs.

---

## Important Fix

In both `app.py` and `main.py`, replace:

```python
client = InferenceClient("digiplay/AbsoluteReality_v1.8.1", token="HF_TOKEN")
```

with:

```python
client = InferenceClient(
    "digiplay/AbsoluteReality_v1.8.1",
    token=HF_TOKEN
)
```

Using `"HF_TOKEN"` passes the string literal instead of the actual environment variable.

---

## Example Prompts

```text
A futuristic cyberpunk city at night
Astronaut riding a horse on Mars
Photorealistic mountain landscape
A dragon flying above Tokyo
Neon-lit samurai warrior
```

---

## requirements.txt

```text
Flask
flask-frozen
huggingface_hub
Pillow
```

---

## Future Improvements

* Image download functionality
* Prompt history
* User authentication
* Multiple model support
* Gallery view
* Negative prompt support
* Dark mode interface

---

## Contributing

Contributions are welcome.

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Open a pull request

---

## License

This project is licensed under the MIT License.

---

## Author

Developed by Paraphylus

GitHub Repository:
[Visionize](https://github.com/Paraphylus/Visionize?utm_source=chatgpt.com)
