from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

api.upload_folder(
    folder_path="tourist_customer_prediction01/model_deployment",  
    repo_id="BujjiProjectPrep/tourist_customer_prediction_streamlit_061201",  
    repo_type="space",
    path_in_repo="",  
)
