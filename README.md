
# Scene-to-Shop

Turn a photo of a room into shoppable products. The app detects objects in a scene, lets you pick one, and finds matching products online, ranked by visual similarity.

## Pipeline

1. **Detection and segmentation** (`vision/detection`): YOLOE finds objects and cuts out each one.
2. **Quality filtering** (`vision/detection/quality`): drops low-quality crops.
3. **Embeddings** (`vision/embeddings`): CLIP embeddings for each object.
4. **Product search** (`vision/retrieval`): Google Lens search through SerpAPI, then filtering and visual ranking of results.
5. **Verification** (`vision/verification`): checks that candidate products match the selected object.
6. **UI** (`vision/selection/ui`): Streamlit app for choosing an object and browsing results.

## Project structure

~~~
scene-to-shop/
  vision/          detection, embeddings, retrieval, verification, selection UI
  scripts/         pipeline test and debug scripts
  tests/           test scripts
  weights/         model weights (not tracked by git)
  runs/            detection outputs (not tracked by git)
  archive/         old backups (not tracked by git)
  docs/            notes and screenshots (placeholder)
  yoloe-11s-seg-pf.pt   YOLOE weights, loaded from the project root
~~~

## Setup

Requires Python 3.11.
~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
~~~
`requirements.txt` installs the CPU build of PyTorch. For an NVIDIA GPU, install torch from the CUDA index at https://pytorch.org/get-started/locally/ instead.

### API keys

Create a `.env` file in the project root:
~~~
GEMINI_API_KEY=your_key_here
SERPAPI_API_KEY=your_key_here
~~~
Never commit this file. It is listed in `.gitignore`.

### Model weights

- `yoloe-11s-seg-pf.pt` must be in the project root. Ultralytics downloads it automatically on first run if it is missing.
- CLIP weights live in `weights/clip/`.

## Run

~~~powershell
streamlit run vision\selection\ui\app.py
~~~
Then open http://localhost:8501.


