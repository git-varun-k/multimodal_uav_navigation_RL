import torch
import torch.nn as nn  # Added this for the loss function
import torch.optim as optim
from torch.utils.data import DataLoader
from drone_dataset import DroneDataset
from model import FusionEncoder

# --- SETTINGS ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OFFICE_PATH = r"E:\data\office\Easy\P000"
FACTORY_PATH = r"E:\data\abandonedfactory\Easy\P004"

#--- INITIALIZE MODEL, OPTIMIZER, CRITERION ---
model = FusionEncoder().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss() # Standard for latent space representation

try:
    office_ds = DroneDataset(OFFICE_PATH)
    factory_ds = DroneDataset(FACTORY_PATH)
    
    train_loader = DataLoader(office_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(factory_ds, batch_size=16, shuffle=False)
    print(f"Data loaded. Office frames: {len(office_ds)}")
    print(f"Training on: {DEVICE}")
except Exception as e:
    print(f"Path Error: {e}")
    exit() # Stop if data isn't found

# --- 3. TRAINING LOOP ---
for epoch in range(10):
    model.train()
    total_train_loss = 0

    for rgb, depth in train_loader:
        rgb, depth = rgb.to(DEVICE), depth.to(DEVICE)

        optimizer.zero_grad()
        latent = model(rgb, depth)

        # We use zeros_like just as a placeholder to see if the latent space converges
        loss = torch.mean(latent ** 2)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() # Fixed typo: was 'los.item()'

    # --- VALIDATION (Sim-to-Real Domain Gap Check) ---
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for rgb, depth in val_loader:
            rgb, depth = rgb.to(DEVICE), depth.to(DEVICE)
            output = model(rgb, depth)
            # Fixed typo: was 'zeroes_like'
            val_loss += criterion(output, torch.zeros_like(output)).item()

    print(f"Epoch {epoch+1} | Train (Office) Loss: {total_train_loss/len(train_loader):.4f} | Val (Factory) Loss: {val_loss/len(val_loader):.4f}")

torch.save(model.state_dict(), "fusion_encoder_v1.pth")
print("Model Saved. Ready for RL integration.")