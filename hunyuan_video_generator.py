# HunyuanVideo-1.5 Inference API Generator
# Generate cinematic videos from text prompts
# Date: Jan 13, 2026

import os
from huggingface_hub import InferenceClient
import time

class HunyuanVideoGenerator:
    """Video generation using HunyuanVideo-1.5 via Inference Providers API"""
    
    def __init__(self, provider="wavespeed"):
        self.hf_token = os.getenv("HF_TOKEN")
        if not self.hf_token:
            raise ValueError(
                "HF_TOKEN environment variable not set!\n"
                "Set it: export HF_TOKEN='your_token_here'\n"
                "Get token: https://huggingface.co/settings/tokens"
            )
        
        self.client = InferenceClient(
            provider=provider,
            api_key=self.hf_token,
        )
        self.model = "tencent/HunyuanVideo-1.5"
        self.provider = provider
        print(f"✅ Initialized with provider: {provider}")
    
    def generate_from_prompt(self, prompt: str, output_path: str = None) -> dict:
        print(f"\n🎬 Generating: {prompt}")
        print(f"⏳ Please wait...")
        
        try:
            result = self.client.text_to_video(prompt, model=self.model)
            print(f"✅ Generated!")
            
            if hasattr(result, 'content') and output_path:
                with open(output_path, 'wb') as f:
                    f.write(result.content)
                print(f"💾 Saved: {output_path}")
                return {"status": "success", "path": output_path}
            return {"status": "success", "data": result}
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return {"status": "error", "error": str(e)}

# Example scenes
SCENES = {
    "urban": "A confident young man walking down a busy city avenue at sunset, cinematic 4K",
    "nature": "A hiker trekking through foggy mountain trail, sunlight breaking through mist, 4K",
    "tech": "A programmer coding at desk with multiple monitors, blue light, cinematic",
    "coffee": "Two friends at a cozy cafe, warm lighting, enjoying coffee",
    "beach": "Person watching ocean sunset on sandy beach, golden hour, 4K cinematic",
}

if __name__ == "__main__":
    print("="*50)
    print("HunyuanVideo-1.5 Generator")
    print("="*50)
    
    try:
        gen = HunyuanVideoGenerator("wavespeed")
        
        # Generate from custom prompt
        result = gen.generate_from_prompt(
            "A dragon flying over misty mountains at sunrise, cinematic, 4K",
            "dragon_video.mp4"
        )
        
        print(f"\n✅ Complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n📌 Setup:")
        print("1. pip install -r requirements.txt")
        print("2. Get token: https://huggingface.co/settings/tokens")
        print("3. export HF_TOKEN='your_token'")
        print("4. Run this script")
