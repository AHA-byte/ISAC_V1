from ultralytics import YOLO

def main():
    # Load the base YOLO11 Small model
    model = YOLO('yolo11s.pt') 

    # Kick off the training loop
    results = model.train(
        data='data/data.yaml', 
        epochs=250,             # 50 is plenty for a prototype
        imgsz=640,             # Let YOLO auto-letterbox your 16:10 resolution
        batch=16,               # Keeps VRAM usage safe
        device=0,              # Forces execution on your dedicated NVIDIA GPU
        name='isac_run_1'      # The folder where your weights will be saved
    )

if __name__ == '__main__':
    main()