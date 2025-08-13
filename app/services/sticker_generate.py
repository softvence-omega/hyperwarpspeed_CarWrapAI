from runwayml import RunwayML
import os
from dotenv import load_dotenv

load_dotenv()

client = RunwayML(api_key=os.getenv("RUNWAYML_API_KEY"))

def generate_image_wrap_sticker(prompt, ref_image_url):
    try:
        if ref_image_url != '':
            task = client.text_to_image.create(
                model='gen4_image',
                ratio='1920:1080',  # keep your original ratio
                prompt_text='use @ref_image to generate other image based on prompt',
                reference_images=[{
                    'uri': ref_image_url,
                    'tag': 'ref_image',
                }],
            ).wait_for_task_output()
        else:
            # Removed trailing comma to fix tuple issue
            prompt_text = (
                f"Generate a realistic, high-resolution car wrap pattern: '{prompt}'. "
                "No car, just the pattern, full coverage, photorealistic, suitable for applying on a vehicle."
            )
            task = client.text_to_image.create(
                model='gen4_image',
                prompt_text=prompt_text,
                ratio='1360:768'  # keep your original ratio
            ).wait_for_task_output()
        print("[INFO] Image generation successful.", task.output[0])
        return task.output[0]

    except Exception as e:
        print(f"[ERROR] Image generation failed for prompt: {prompt}")
        print("Reason:", e)
        return None  # fallback if generation fails

if __name__=="__main__":
    prompt = "Ornamental gold floral and swirl pattern, elegant decorative vector design, applied as a car wrap on the side of a sleek black sports car, photorealistic lighting, shiny reflections, realistic car proportions, high detail"
    res = generate_image_wrap_sticker(prompt, '')
    print(res)
