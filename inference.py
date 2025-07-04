import matplotlib.pyplot as plt
import torch
import random
import numpy as np

from utils.download_dataset import download_dataset
from model import vit

device = "cuda" if torch.cuda.is_available() else "cpu"

def predict_and_plot_grid(model,
                          dataset,
                          classes,
                          grid_size=3):
    model.eval()
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(9, 9))
    for i in range(grid_size):
        for j in range(grid_size):
            idx = random.randint(0, len(dataset) - 1)
            img, true_label = dataset[idx]
            input_tensor = img.unsqueeze(dim=0).to(device)
            with torch.inference_mode():
                output = model(input_tensor)
                _, predicted = torch.max(output.data, 1)
            img = img / 2 + 0.5 # Unormalize our images to be able to plot them with matplotlib
            npimg = img.cpu().numpy()
            axes[i, j].imshow(np.transpose(npimg, (1, 2, 0)))
            truth = classes[true_label] == classes[predicted.item()]
            if truth:
                color = "g"
            else:
                color = "r"

            axes[i, j].set_title(f"Truth: {classes[true_label]}\n, Predicted: {classes[predicted.item()]}", fontsize=10, c=color)
            axes[i, j].axis("off")
    plt.tight_layout()
    plt.show()
     
if __name__ == '__main__':
    train_dataloader, test_dataloader = download_dataset(root='./data', isDownload=True, batch_size=vit.ViTConfig.batch_size)

    model = vit.VisionTransformer().to(device)
    model.load_state_dict(torch.load("./checkpoints/vit_epoch_10.pth"))
    model.eval()

    predict_and_plot_grid(model, test_dataloader.dataset, 
                          classes=train_dataloader.dataset.classes, 
                          grid_size=3)
    