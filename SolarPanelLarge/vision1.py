from roboflow import Roboflow
rf = Roboflow(api_key="nUVA0daHRcbErRH0HCeP")
project = rf.workspace().project("solar-panel-pollution-dataset-vwpax")
model = project.version("2").model

# infer on a local image
model.predict("a.jpg", confidence=40, overlap=30)

# visualize your prediction
# model.predict("your_image.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())