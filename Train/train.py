from ultralytics import YOLO

# Load a YOLOv10 model
model = YOLO("yolov11n.yaml")  # build from scratch
model = YOLO("yolov11n.pt")    # load pretrained model 

# Train the model
model.train(data="coco128.yaml", epochs=3)

# Validate the model
metrics = model.val()

# Run predictions
results = model("https://ultralytics.com/images/bus.jpg")

# Export the model
path = model.export(format="tfjs")
