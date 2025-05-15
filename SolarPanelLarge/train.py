from ultralytics import YOLO
if __name__ == '__main__':
    # Modeli indir
    model = YOLO('yolo11s.pt')  # Önceden eğitilmiş bir modelle başla

    # Eğitim
    model.train(data='C:/Users/Arda/Desktop/SolarPanelLarge/data.yaml', epochs=50,device="cuda")
