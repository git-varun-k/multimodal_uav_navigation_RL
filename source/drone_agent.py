import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from PIL import Image
import os
from torchvision import transforms
from model import FusionEncoder

class TartanAirDataset:
    def __init__(self, img_path, depth_path):

        img_files = os.listdir(img_path)
        depth_files = os.listdir(depth_path)

        img_dict = {f.split('_')[0]: f for f in img_files}
        depth_dict = {f.split('_')[0]: f for f in depth_files}

        common_ids = sorted(set(img_dict.keys()) & set(depth_dict.keys()))

        self.img_files = [os.path.join(img_path, img_dict[i]) for i in common_ids]
        self.depth_files = [os.path.join(depth_path, depth_dict[i]) for i in common_ids]

        print(f"Aligned dataset: {len(self.img_files)} frames")

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = Image.open(self.img_files[idx]).convert('RGB')
        img = self.transform(img)

        depth = np.load(self.depth_files[idx])
        depth = Image.fromarray(depth.astype(np.float32))
        depth = self.transform(depth)

        return img, depth

class DroneEnv(gym.Env):
    def __init__(self, dataset, v_brain):
        super().__init__()

        self.dataset = dataset
        self.v_brain = v_brain
        self.device = next(v_brain.parameters()).device

        self.current_idx = 0

        
        self.action_space = spaces.Discrete(2)
        # 0 = TURN
        # 1 = STRAIGHT

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(128,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_idx = np.random.randint(0, len(self.dataset))
        return self._get_obs(), {}

    def _get_obs(self):
        img, depth = self.dataset[self.current_idx]

        img = img.unsqueeze(0).to(self.device)
        depth = depth.unsqueeze(0).to(self.device)

        with torch.no_grad():
            latent = self.v_brain(img, depth)
            latent = torch.nn.functional.normalize(latent, dim=1)

        latent = latent.view(-1)

        if latent.numel() != 128:
            raise RuntimeError(f"Expected 128-dim obs, got {latent.numel()}")

        return latent.cpu().numpy()

    def step(self, action):
        _, depth_tensor = self.dataset[self.current_idx]

        center_depth = depth_tensor[0, 44:84, 44:84]
        valid_depths = center_depth[center_depth < 100]

        if valid_depths.numel() == 0:
            avg_dist = 100.0
        else:
            avg_dist = torch.mean(valid_depths).item()

        collision_threshold = 2.0
        is_blocked = avg_dist < collision_threshold



        # Reward
        if is_blocked:
            reward = 3.0 if action != 1 else -5.0
        else:
            reward = 2.0 if action == 1 else -2.0

        self.current_idx += 1
        done = self.current_idx >= len(self.dataset) - 1

        return self._get_obs(), reward, done, False, {}

if __name__ == "__main__":

    STAGE = "FACTORY"

    office_path = r"E:\data\office\Easy\P000"
    factory_path = r"E:\data\abandonedfactory\Easy\P002"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vision_brain = FusionEncoder().to(device)
    vision_brain.load_state_dict(torch.load("fusion_encoder_v1.pth", map_location=device))
    vision_brain.eval()
    vision_brain.requires_grad_(False)

    print("Vision brain loaded")

    current_path = office_path if STAGE == "OFFICE" else factory_path

    ds = TartanAirDataset(
        os.path.join(current_path, "image_left"),
        os.path.join(current_path, "depth_left")
    )

    env = DroneEnv(ds, vision_brain)

    if STAGE == "OFFICE":
        model = PPO("MlpPolicy", env, verbose=1, device=device, ent_coef=0.02)
        total_steps = 15000
        save_name = "drone_pilot_v1"
    else:
        model = PPO.load("drone_pilot_v1.zip", env=env, device=device)
        model.ent_coef = 0.01
        total_steps = 20000
        save_name = "drone_pilot_adapted_factory"

    print("Training started")
    model.learn(total_timesteps=total_steps)
    model.save(save_name)
    print("Saved:", save_name)