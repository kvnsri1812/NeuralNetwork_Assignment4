import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

# Generator architecture
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x).view(-1, 1, 28, 28)

# Discriminator architecture
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x.view(-1, 784))

# Set device to CPU
device = torch.device("cpu")

# Prepare training data
batch_size = 64
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Optimizers and loss
lr = 0.0002
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=lr)
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)

# Track losses
loss_g_list = []
loss_d_list = []

# Training
epochs = 101
os.makedirs("generated_images", exist_ok=True)

for epoch in range(epochs):
    print(f"Epoch: {epoch}")
    for real_imgs, _ in dataloader:
        real_imgs = real_imgs.to(device)
        b_size = real_imgs.size(0)

        # Generate fake images
        noise = torch.randn(b_size, 100).to(device)
        fake_imgs = generator(noise)

        # Train Discriminator
        real_labels = torch.ones(b_size, 1).to(device)
        fake_labels = torch.zeros(b_size, 1).to(device)

        output_real = discriminator(real_imgs)
        output_fake = discriminator(fake_imgs.detach())

        loss_d_real = criterion(output_real, real_labels)
        loss_d_fake = criterion(output_fake, fake_labels)
        loss_d = (loss_d_real + loss_d_fake) / 2

        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()

        # Train Generator
        output = discriminator(fake_imgs)
        loss_g = criterion(output, real_labels)

        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()

        # Store losses
        loss_g_list.append(loss_g.item())
        loss_d_list.append(loss_d.item())

    if epoch in [0, 50, 100]:
        with torch.no_grad():
            noise = torch.randn(16, 100).to(device)
            sample_imgs = generator(noise).cpu()
            torchvision.utils.save_image(sample_imgs, f"generated_images/generated_epoch_{epoch}.png", normalize=True)

# Plot Losses
plt.figure(figsize=(10, 5))
plt.plot(loss_g_list, label='Generator Loss')
plt.plot(loss_d_list, label='Discriminator Loss')
plt.title('GAN Training Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("generated_images/loss_plot.png")
plt.show()
