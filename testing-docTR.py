from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# Load the grocery receipt
doc = DocumentFile.from_images("receipt.jpeg")

# Load the OCR model
model = ocr_predictor(pretrained=True)

# Perform OCR
result = model(doc)

# Display the OCR result
print(result.export())
