
from smart_contract_assistant.src.ui import build_app
import os
import traceback
from smart_contract_assistant.config import HF_TOKEN

if __name__ == "__main__":
    print("Starting Smart Contract Assistant...")
    print(f"Using HF Token: {HF_TOKEN[:4]}...{HF_TOKEN[-4:]}")
    
    try:
        app = build_app()
        app.launch(show_error=True)
    except Exception:
        traceback.print_exc()
