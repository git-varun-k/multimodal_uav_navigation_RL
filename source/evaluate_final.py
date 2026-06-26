import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
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

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = self.transform(Image.open(self.img_files[idx]).convert('RGB'))
        depth = np.load(self.depth_files[idx])
        depth = Image.fromarray(depth.astype(np.float32))
        depth = self.transform(depth)
        return img, depth

class DroneTestEnv(gym.Env):
    def __init__(self, dataset, vision_model):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset
        self.vision_model = vision_model.to(self.device)

        self.current_idx = 0

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(128,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        self.current_idx = 0
        return self._get_obs(), {}

    def _get_obs(self):
        img, depth = self.dataset[self.current_idx]

        img = img.unsqueeze(0).to(self.device)
        depth = depth.unsqueeze(0).to(self.device)

        with torch.no_grad():
            latent = self.vision_model(img, depth)
            latent = torch.nn.functional.normalize(latent, dim=1)

        return latent.cpu().numpy().flatten()

    def step(self, action):
        _, depth_tensor = self.dataset[self.current_idx]

        center_depth = depth_tensor[0, 44:84, 44:84]
        valid_depths = center_depth[center_depth < 100]
        avg_dist = torch.mean(valid_depths).item() if valid_depths.numel() > 0 else 100.0

        is_blocked = avg_dist < 2.0

        if is_blocked:
            reward = -1.0 if action == 1 else 1.0
        else:
            reward = 1.0 if action == 1 else -0.5

        self.current_idx += 1
        done = self.current_idx >= len(self.dataset) - 1

        return self._get_obs(), reward, done, False, {}

if __name__ == "__main__":

    test_path = r"E:\data\abandonedfactory\Easy\P006"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vision_brain = FusionEncoder().to(device)
    vision_brain.load_state_dict(torch.load("fusion_encoder_v1.pth", map_location=device))
    vision_brain.eval()

    ds = TartanAirDataset(
        os.path.join(test_path, "image_left"),
        os.path.join(test_path, "depth_left")
    )

    env = DroneTestEnv(ds, vision_brain)

    def run_eval(model_path, name):

        model = PPO.load(model_path, env=env, device=device)

        obs, _ = env.reset()
        total_rew = 0
        steps = 0
        actions = []
        correct = 0
        blocked = 0

        print(f"\n--- {name} ---")

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, done, _, _ = env.step(action)

            # NOW get the correct frame (after step)
            _, depth_tensor = env.dataset[env.current_idx - 1]
            depth_tensor = depth_tensor.to(device)

            center_depth = depth_tensor[0, 44:84, 44:84]
            valid_depths = center_depth[center_depth < 100]

            avg_dist = torch.mean(valid_depths).item() if valid_depths.numel() > 0 else 100.0
            is_blocked = avg_dist < 2.0

            if is_blocked:
                blocked += 1

            if is_blocked and action != 1:
                correct += 1
            elif not is_blocked and action == 1:
                correct += 1

            total_rew += reward
            actions.append(int(action))
            steps += 1

            if done:
                break

        print("Total Reward:", total_rew)
        print("Avg Reward:", total_rew / max(1, steps))
        print("Blocked frames:", blocked)
        print("Action set:", set(actions))
        print("Accuracy:", correct / steps)


    run_eval("drone_pilot_v1.zip", "Baseline")
    run_eval("drone_pilot_adapted_factory.zip", "Adapted")