# Image/Text to 3D Model Generator

This project allows users to generate 3D models from either text prompts or images. It uses advanced AI pipelines for 3D model generation and provides a web interface for easy interaction.

---

## Steps to Run

### Option 1: Run Locally

1. **Clone the Repository**  
   Clone the project to your local machine.

2. **Install Dependencies**  
   Install the required Python libraries using the following command:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Ngrok**  
   - Create an account on [Ngrok](https://ngrok.com/) and get your authentication token.
   - Replace `your_ngrok_token` in the code with your actual Ngrok token.

4. **Ensure GPU Availability**  
   - This project requires a GPU with CUDA support for optimal performance.
   - Install the appropriate version of PyTorch with GPU support from [PyTorch's official website](https://pytorch.org/get-started/locally/).

5. **Run the Application**  
   Open the Jupyter Notebook and execute all cells in `img_text_3d.ipynb`. Alternatively, convert it to a Python script and run:
   ```bash
   jupyter nbconvert --to script img_text_3d.ipynb
   python img_text_3d.py
   ```
   The application will start on `http://localhost:5000`. Ngrok will provide a public URL for external access.

6. **Access the Web Interface**  
   Open the Ngrok public URL in your browser. You can upload an image or enter a text prompt to generate a 3D model.

7. **Download the Generated Model**  
   After generation, the 3D model will be downloaded in either `.stl` or `.obj` format based on preference.

---

### Option 2: Run on Google Colab

1. **Open the Notebook on Colab**  
   Upload the `img_text_3d.ipynb` file to your Google Drive and open it in Google Colab.

2. **Enable GPU**  
   - Go to `Runtime > Change runtime type`.
   - Set the hardware accelerator to `GPU`.

3. **Install Dependencies**  
   Run the first cell in the notebook to install all required libraries.

4. **Set Up Ngrok**  
   - Replace `your_ngrok_token` in the notebook with your actual Ngrok token.

5. **Run the Flask App**  
   Execute the notebook cells to start the Flask server. Ngrok will provide a public URL for accessing the app.

6. **Access the Web Interface**  
   Use the Ngrok public URL to interact with the app and generate 3D models.

---

## Libraries Used

- **[Transformers](https://huggingface.co/docs/transformers)**: For AI model pipelines.
- **[Accelerate](https://huggingface.co/docs/accelerate)**: For efficient model execution.
- **[Diffusers](https://github.com/huggingface/diffusers)**: For 3D model generation pipelines.
- **[Flask](https://flask.palletsprojects.com/)**: For creating the web application.
- **[Pyngrok](https://pyngrok.readthedocs.io/)**: For exposing the local server to the internet.
- **[Pillow](https://pillow.readthedocs.io/)**: For image processing.
- **[Numpy](https://numpy.org/)**: For numerical operations.
- **[Trimesh](https://trimsh.org/)**: For 3D mesh processing.
- **[Rembg](https://github.com/danielgatis/rembg)**: For background removal in images.
- **[Torch](https://pytorch.org/)**: For deep learning computations.
- **[Matplotlib](https://matplotlib.org/)**: For 3D visualization.
- **[ONNX Runtime](https://onnxruntime.ai/)**: For running ONNX models.

---

## GPU Requirements

- **CUDA Support**: The project requires a CUDA-enabled GPU for running the AI pipelines efficiently.
- **Minimum GPU Memory**: At least 8GB of GPU memory is recommended for smooth execution.
- **PyTorch with GPU**: Ensure you have installed the GPU-compatible version of PyTorch.

---

## Thought Process

1. **Objective**: The goal was to create a user-friendly interface for generating 3D models from text or images using state-of-the-art AI pipelines.
2. **Pipeline Selection**: Used Hugging Face's `ShapEPipeline` and `ShapEImg2ImgPipeline` for text-to-3D and image-to-3D generation, respectively.
3. **Web Interface**: Designed a simple HTML form with Flask to allow users to upload images or input text prompts.
4. **Ngrok Integration**: Added Ngrok for easy external access to the local Flask server.
5. **Output Formats**: Supported `.stl` and `.obj` formats for compatibility with 3D printing and general 3D applications.
6. **Visualization**: Included a 3D visualization feature using Matplotlib for local inspection of generated models.

Feel free to contribute or raise issues for improvements!