# import necessary libraries
import matplotlib.pyplot as plt
import torch
# from data_loader import unnormalize

# def plot_misclassified_images(model, test_loader, device):
#     all_misclassified_images = []

#     model.eval()

#     with torch.no_grad():
#         for data , labels in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             _, pred = torch.max(output, 1)
#             for i in range(len(pred)):
#                 if pred[i] != target[i]:
#                     all_misclassified_images.append({'image': data[i], 'predicted_class': pred[i], 'correct_class': target[i]})

#     fig = plt.figure(figsize=(15, 5))
#     for i in range(10):
#         sub = fig.add_subplot(2, 5, i+1)
#         misclassified_image = all_misclassified_images[i]
#         plt.imshow(misclassified_image['image'].cpu().numpy().squeeze(), cmap='gray', interpolation='none')
#         sub.set_title("Correct class: {}\nPredicted class: {}".format(misclassified_image['correct_class'], misclassified_image['predicted_class']))
#     plt.tight_layout()
#     plt.show()


def compute_accuracy_graph(accuracies):
    plt.plot(accuracies)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title("Epoch vs Accuracy")
    plt.legend()

def compute_loss_graph(losses):
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title("Epoch vs Accuracy")
    plt.legend()
