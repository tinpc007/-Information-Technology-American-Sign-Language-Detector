conda remove -n wlasl_videomae --all -y
conda create -n wlasl_videomae python=3.10 -y
conda activate wlasl_videomae
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

web
pip install gradio
pip install decords
pip install transformers

Activation code : 
Open Window Powershell
wsl
cd /mnt/d
conda activate wlasl_videomae
python wlasl_videomae_gradio_app.py