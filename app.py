import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# # Data Fusion Framework
# # This framework unifies multiple data domains (e.g., text, images, sound) into a single topological space.
# # The goal is to create a shared representation that enables seamless cross-domain understanding and transfer.

# Step 1: Domain-Specific Encoders
class TextEncoder(nn.Module):
    def __init__(self, input_dim, shared_dim):
        super(TextEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, shared_dim)
        )

    def forward(self, x):
        return self.fc(x)


class ImageEncoder(nn.Module):
    def __init__(self, input_dim, shared_dim):
        super(ImageEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, shared_dim)
        )

    def forward(self, x):
        return self.fc(x)


class SoundEncoder(nn.Module):
    def __init__(self, input_dim, shared_dim):
        super(SoundEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, shared_dim)
        )

    def forward(self, x):
        return self.fc(x)


# Step 2: Shared Fusion Space
class FusionModel(nn.Module):
    def __init__(self, shared_dim):
        super(FusionModel, self).__init__()
        self.shared_dim = shared_dim
        self.text_encoder = TextEncoder(input_dim=300, shared_dim=shared_dim)  # Example input_dim for text
        self.image_encoder = ImageEncoder(input_dim=2048, shared_dim=shared_dim)  # Example input_dim for image
        self.sound_encoder = SoundEncoder(input_dim=1024, shared_dim=shared_dim) # Example input_dim for sound

    def forward(self, text=None, image=None, sound=None):
        representations = []
        if text is not None:
            representations.append(self.text_encoder(text))
        if image is not None:
            representations.append(self.image_encoder(image))
        if sound is not None:
            representations.append(self.sound_encoder(sound))

        # Fusion through concatenation
        if len(representations) > 1:
            fused_representation = torch.cat(representations, dim=1)
        elif len(representations) == 1:
            fused_representation = representations[0]
        else:
            raise ValueError("At least one modality input should be provided")

        return fused_representation


# Step 3: Training Loop
class FusionTrainer:
    def __init__(self, model, learning_rate=1e-3):
        self.model = model
        self.criterion = nn.CosineEmbeddingLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def train(self, data_loaders, labels, modality_combinations, epochs=10):
        for epoch in range(epochs):
            epoch_loss = 0.0

            # Create an iterator that cycles through all combinations
            data_loaders_iter = [iter(dl) for dl in data_loaders]

            i = 0
            while i < max([len(loader) for loader in data_loaders]):
                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Get the current modality combination
                current_combination = modality_combinations[i % len(modality_combinations)]

                # Prepare inputs for each modality
                inputs = {}
                modality_names = ['text', 'image', 'sound']
                
                try:
                  # Select the corresponding tensors for this batch
                  for j, modality in enumerate(modality_names):
                      if modality in current_combination:
                          inputs[modality] = next(data_loaders_iter[j])[0]
                except StopIteration:
                   # Break the inner loop if any iterator is exhausted
                   break

                # Forward pass through the FusionModel
                fused_representation = self.model(**inputs)

                # Create target tensor based on the number of modalities used
                if len(inputs) > 1:
                    target = torch.ones(fused_representation.shape[0], device=fused_representation.device)
                    loss = self.criterion(fused_representation, torch.zeros_like(fused_representation), target)
                else:
                    target = torch.ones(fused_representation.shape[0], device=fused_representation.device)
                    loss = self.criterion(fused_representation, torch.zeros_like(fused_representation), target)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                i += 1

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")



# Step 4: Example Usage
if __name__ == "__main__":
    # Example dimensions
    shared_dim = 128
    fusion_model = FusionModel(shared_dim=shared_dim)
    trainer = FusionTrainer(fusion_model)

    # Placeholder data
    text_data = torch.randn(100, 300)  # 100 text samples
    image_data = torch.randn(100, 2048)  # 100 image samples
    sound_data = torch.randn(100, 1024) # 100 sound samples

    # Create TensorDatasets
    text_dataset = TensorDataset(text_data)
    image_dataset = TensorDataset(image_data)
    sound_dataset = TensorDataset(sound_data)

    # Create DataLoaders
    text_loader = DataLoader(text_dataset, batch_size=16, shuffle=True)
    image_loader = DataLoader(image_dataset, batch_size=16, shuffle=True)
    sound_loader = DataLoader(sound_dataset, batch_size=16, shuffle=True)

    # Create a dictionary to hold the data loaders for each combination
    data_loaders_dict = {
        ('text',): text_loader,
        ('image',): image_loader,
        ('sound',): sound_loader,
        ('text', 'image'): zip(text_loader, image_loader),
        ('text', 'sound'): zip(text_loader, sound_loader),
        ('image', 'sound'): zip(image_loader, sound_loader),
        ('text', 'image', 'sound'): zip(text_loader, image_loader, sound_loader),
    }

    # Create lists of data loaders, labels, and modality combinations
    data_loaders = [text_loader, image_loader, sound_loader]
    labels_list = []
    modality_combinations = [
        ('text',),
        ('image',),
        ('sound',),
        ('text', 'image'),
        ('text', 'sound'),
        ('image', 'sound'),
        ('text', 'image', 'sound')
    ]

    # Train the model with different modality combinations
    trainer.train(data_loaders, labels_list, modality_combinations, epochs=5)

    print("Training complete. Fusion model is ready for use.")
