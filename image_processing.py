from torchvision import transforms

def preprocess_image(image, image_size=224):
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image_tensor = preprocess(image)
    # Add batch dimension (3, 224, 224) -> (1, 3, 224, 224)
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor