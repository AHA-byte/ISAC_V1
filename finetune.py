from ultralytics import YOLO

def main():
    # THE UPGRADE: Load your existing, trained brain instead of the default YOLO model
    model = YOLO('runs/detect/isac_run_1-7/weights/best.pt') 

    # Kick off the fine-tuning loop
    results = model.train(
        data='path/to/your/NEW_master_dataset/data.yaml', 
        epochs=150,            # It won't need as many epochs to learn now
        imgsz=640,             
        batch=16,              # Keeping your optimized RTX 4050 batch size
        device=0,              
        name='isac_run_2'      # Save it to a new folder so you don't overwrite V1
    )

if __name__ == '__main__':
    main()