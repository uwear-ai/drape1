from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from PIL import Image
import io
from src.pipeline import load_drape, infer_drape
from transparent_background import Remover
import uvicorn

app = FastAPI(title="Drape1 API")

# Initialize the pipeline and remover
pipeline = load_drape()
remover = Remover()

@app.post("/generate")
async def generate_image(
    prompt: str,
    image_ref: UploadFile = File(...)
):
    # Read and convert the reference image
    image_ref_content = await image_ref.read()
    image_ref_pil = Image.open(io.BytesIO(image_ref_content)).convert("RGB")
    
    # Generate the image
    generated_image = infer_drape(
        pipe=pipeline,
        prompt=prompt,
        image_ref=image_ref_pil,
        remover=remover,
        seed=42
    )
    
    # Save the generated image to a temporary file
    output_path = "generated_image.png"
    generated_image.save(output_path)
    
    # Return the generated image
    return FileResponse(output_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 