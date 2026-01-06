import torch
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset, random_split
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_cosine_schedule_with_warmup
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from typing import Tuple, Optional, List, Any

DATA_DIR = "./01 model train/"
OUTPUT_DIR = "./01 model train/"

TOTAL_STEPS = 500000
BATCH_SIZE = 128
TARGET_SIZE = 64
MOL_MATRIX_SIZE = 64
CROP_SIZE = 36
TRAIN_TEST_SPLIT = 0.8
NUM_TIMESTEPS = 1000
SCHEDULER_STEPS = 50
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500
CHECKPOINT_INTERVAL = 100000

KEY_PROPERTIES = [
    'distillation_cum_fraction_d86_ibpv',
    'distillation_cum_fraction_d86_10v',
    'distillation_cum_fraction_d86_30v',
    'distillation_cum_fraction_d86_50v',
    'distillation_cum_fraction_d86_70v',
    'distillation_cum_fraction_d86_90v',
    'distillation_cum_fraction_d86_fbpv'
]

OPTIONAL_PROPS = [
    'density_288K',
    'density_293K',
    'freeze_point',
    'cetane_number',
    'RON',
    'reid_vapor_pressure',
    'C_weight_percentage',
    'H_weight_percentage',
    'S_weight_percentage',
    'N_weight_percentage',
    'gasoline_P_mass_fraction',
    'gasoline_I_mass_fraction',
    'gasoline_O_mass_fraction',
    'gasoline_N_mass_fraction',
    'gasoline_A_mass_fraction',
    'GCMS_group_P_weight_fraction',
    'GCMS_group_1N_weight_fraction',
    'GCMS_group_2N_weight_fraction',
    'GCMS_group_3N_weight_fraction',
    'GCMS_group_1A_weight_fraction',
    'GCMS_group_2A_weight_fraction',
    'GCMS_group_3A_weight_fraction'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    input_files = {
        'P': 'Fuel_BulkProperty_10w_Diesel_P.csv',
        'N': 'Fuel_BulkProperty_10w_Diesel_N.csv',
        'A': 'Fuel_BulkProperty_10w_Diesel_A.csv',
        'G': 'Fuel_BulkProperty_15w_Gasoline.csv',
        'J': 'Fuel_BulkProperty_4w_Jet.csv'
    }

    output_files = {
        'P': 'Fuel_MoleculeMatrices_10w_Diesel_P.csv',
        'N': 'Fuel_MoleculeMatrices_10w_Diesel_N.csv',
        'A': 'Fuel_MoleculeMatrices_10w_Diesel_A.csv',
        'G': 'Fuel_MoleculeMatrices_15w_Gasoline.csv',
        'J': 'Fuel_MoleculeMatrices_4w_Jet.csv',
    }

    input_dfs = []
    output_dfs = []

    for key in ['P', 'N', 'A', 'G', 'J']:
        input_path = os.path.join(DATA_DIR, input_files[key])
        input_df = pd.read_csv(input_path, dtype=float).head(10)
        input_dfs.append(input_df)

        output_path = os.path.join(DATA_DIR, output_files[key])
        output_df = pd.read_csv(output_path, header=None).head(10)
        output_dfs.append(output_df)

    x = pd.concat(input_dfs, axis=0, ignore_index=True)
    y = pd.concat(output_dfs, axis=0, ignore_index=True)

    return x, y


def preprocess_input(x: pd.DataFrame, scaler: Optional[MinMaxScaler] = None) -> Tuple[torch.Tensor, MinMaxScaler, int]:

    key_data = x[KEY_PROPERTIES].values
    optional_data = x[OPTIONAL_PROPS].values if OPTIONAL_PROPS else np.zeros((x.shape[0], 0))

    key_mask = np.ones_like(key_data)
    optional_mask = np.ones_like(optional_data)

    combined_data = np.hstack([key_data, optional_data, key_mask, optional_mask])

    if scaler is None:
        scaler = MinMaxScaler()
        x_scaled = scaler.fit_transform(combined_data)
    else:
        x_scaled = scaler.transform(combined_data)

    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    padding_size = TARGET_SIZE - x_tensor.size(1)

    if padding_size > 0:
        padding = torch.zeros(x_tensor.size(0), padding_size, dtype=torch.float32)
        x_tensor = torch.cat((x_tensor, padding), dim=1)

    num_optional_props = len(OPTIONAL_PROPS)
    return x_tensor, scaler, num_optional_props


def preprocess_output(y: pd.DataFrame, y_max: Optional[float] = None) -> Tuple[torch.Tensor, float]:
    if y_max is None:
        y_max = y.max().max()
    else:
        y_max = y_max

    y_normalized = y / y_max

    output_tensor = torch.tensor(
        y_normalized.values[:, :36 * 36].reshape(-1, 1, 36, 36),
        dtype=torch.float32
    )

    output_tensor = output_tensor.permute(0, 1, 3, 2)
    output_tensor = F.pad(output_tensor, (14, 14, 14, 14), mode='constant', value=0)

    return output_tensor, y_max


def create_datasets(
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
        original_x: Optional[pd.DataFrame] = None,
        original_y: Optional[pd.DataFrame] = None
) -> Tuple[DataLoader, DataLoader]:
    dataset = TensorDataset(input_tensor, output_tensor)
    train_size = int(TRAIN_TEST_SPLIT * len(dataset))
    test_size = len(dataset) - train_size

    torch.manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    if original_x is not None and original_y is not None:
        train_indices = train_dataset.indices
        test_indices = test_dataset.indices

        split_data_dir = os.path.join(OUTPUT_DIR, "split_data")
        os.makedirs(split_data_dir, exist_ok=True)

        train_x = original_x.iloc[train_indices]
        test_x = original_x.iloc[test_indices]

        train_x.to_csv(os.path.join(split_data_dir, "train_input.csv"), index=False)
        test_x.to_csv(os.path.join(split_data_dir, "test_input.csv"), index=False)

        train_y = original_y.iloc[train_indices]
        test_y = original_y.iloc[test_indices]

        train_y.to_csv(os.path.join(split_data_dir, "train_output.csv"), index=False, header=False)
        test_y.to_csv(os.path.join(split_data_dir, "test_output.csv"), index=False, header=False)

        print(f"Saved {len(train_x)} training samples and {len(test_x)} test samples to {split_data_dir}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_dataloader, test_dataloader


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * (torch.pi / 2)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999).detach().to(torch.float32)


def initialize_model() -> Tuple[UNet2DConditionModel, DDPMScheduler, DDPMPipeline]:
    betas = cosine_beta_schedule(NUM_TIMESTEPS)
    unet = UNet2DConditionModel(
        sample_size=MOL_MATRIX_SIZE,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        cross_attention_dim=TARGET_SIZE
    )

    scheduler = DDPMScheduler(num_train_timesteps=NUM_TIMESTEPS, beta_schedule="custom",
                              trained_betas=betas.clone().detach().numpy())
    pipeline = DDPMPipeline(unet=unet, scheduler=scheduler)
    pipeline.to(device)

    return unet, scheduler, pipeline


def combined_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        weight_center: float = 2.0,
        weight_mse: float = 1.0,
        weight_cos: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, _, height, width = pred.size()

    weights = torch.ones_like(target)
    center_start, center_end = 14, 50
    weights[:, :, center_start:center_end, center_start:center_end] *= weight_center
    mse_loss = torch.mean(weights * (pred - target) ** 2)

    pred_flat = pred.view(batch_size, -1)
    target_flat = target.view(batch_size, -1)
    cosine_sim = F.cosine_similarity(pred_flat, target_flat, dim=-1)
    cos_loss = 1.0 - torch.mean(cosine_sim)

    total_loss = weight_mse * mse_loss + weight_cos * cos_loss

    return total_loss, mse_loss, cos_loss


def train_model(
        unet: UNet2DConditionModel,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        scheduler: DDPMScheduler,
        num_optional_props: int,
        total_steps: int = 500000
) -> tuple[list[Any], list[Any], list[Any], list[Any], list[Any], list[Any], list[Any]]:
    optimizer = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps,
    )
    scaler = GradScaler()
    train_loader_iter = iter(train_dataloader)

    step_losses = {
        "train_total": [],
        "train_mse": [],
        "train_cos": [],
        "test_total": [],
        "test_mse": [],
        "test_cos": [],
        "steps": []
    }

    for step in range(total_steps):
        unet.train()

        try:
            input_batch, target_batch = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_dataloader)
            input_batch, target_batch = next(train_loader_iter)

        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        input_batch = input_batch.unsqueeze(1)

        batch_size = input_batch.size(0)
        key_start = 0
        key_end = len(KEY_PROPERTIES)

        optional_start = key_end
        optional_end = key_end + num_optional_props

        key_mask_start = optional_end
        key_mask_end = key_mask_start + len(KEY_PROPERTIES)

        optional_mask_start = key_mask_end
        optional_mask_end = optional_mask_start + num_optional_props

        drop_rate = 0.3 + 0.5 * torch.rand(1).item()
        drop_mask = torch.rand(batch_size, num_optional_props) < drop_rate

        input_batch[:, :, optional_start:optional_end] = torch.where(
            drop_mask.unsqueeze(1).to(device),
            torch.zeros_like(input_batch[:, :, optional_start:optional_end]),
            input_batch[:, :, optional_start:optional_end]
        )
        input_batch[:, :, optional_mask_start:optional_mask_end] = torch.where(
            drop_mask.unsqueeze(1).to(device),
            torch.zeros_like(input_batch[:, :, optional_mask_start:optional_mask_end]),
            input_batch[:, :, optional_mask_start:optional_mask_end]
        )

        t = torch.randint(0, scheduler.config.num_train_timesteps, (input_batch.size(0),), device=device)

        noise = torch.randn_like(target_batch)
        noisy_input = scheduler.add_noise(target_batch, noise, t)

        optimizer.zero_grad()

        with autocast():
            output = unet(noisy_input, t, input_batch)
            predicted_noise = output.sample

            total_loss, mse_loss, cos_loss = combined_loss(
                predicted_noise,
                noise,
                weight_center=2.0,
                weight_mse=1.0,
                weight_cos=0.5
            )

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()

        if (step + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint_dir = os.path.join(OUTPUT_DIR, "pretraining_checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)

            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"model_step_{step + 1}.pth"
            )

            torch.save({
                'step': step + 1,
                'model_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss.item(),
            }, checkpoint_path)

            print(f"\nCheckpoint saved at step {step + 1} -> {checkpoint_path}")

        if (step+1) % 100 == 0:
            current_step = step + 1
            step_losses["train_total"].append(total_loss.item())
            step_losses["train_mse"].append(mse_loss.item())
            step_losses["train_cos"].append(cos_loss.item())
            step_losses["steps"].append(current_step)

            unet.eval()
            running_test_total = running_test_mse = running_test_cos = 0.0

            with torch.no_grad():
                for input_batch, target_batch in test_dataloader:
                    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
                    input_batch = input_batch.unsqueeze(1)

                    t = torch.randint(0, scheduler.config.num_train_timesteps, (input_batch.size(0),), device=device)
                    noise = torch.randn_like(target_batch)
                    noisy_input = scheduler.add_noise(target_batch, noise, t)

                    output = unet(noisy_input, t, input_batch)
                    predicted_noise = output.sample

                    total_loss, mse_loss, cos_loss = combined_loss(
                        predicted_noise,
                        noise,
                        weight_center=2.0,
                        weight_mse=1.0,
                        weight_cos=0.5
                    )

                    running_test_total += total_loss.item()
                    running_test_mse += mse_loss.item()
                    running_test_cos += cos_loss.item()

            avg_test_total = running_test_total / len(test_dataloader)
            avg_test_mse = running_test_mse / len(test_dataloader)
            avg_test_cos = running_test_cos / len(test_dataloader)

            step_losses["test_total"].append(avg_test_total)
            step_losses["test_mse"].append(avg_test_mse)
            step_losses["test_cos"].append(avg_test_cos)

            print(f"\nStep {current_step}/{total_steps} | "
                  f"Train Total: {step_losses['train_total'][-1]:.6f} "
                  f"(MSE: {step_losses['train_mse'][-1]:.6f}, "
                  f"Cos: {step_losses['train_cos'][-1]:.6f}) | "
                  f"Test Total: {avg_test_total:.6f} "
                  f"(MSE: {avg_test_mse:.5f}, Cos: {avg_test_cos:.6f})")
            print("-" * 120)

    return (
        step_losses["train_total"],
        step_losses["train_mse"],
        step_losses["train_cos"],
        step_losses["test_total"],
        step_losses["test_mse"],
        step_losses["test_cos"],
        step_losses["steps"]
    )


def generate_sample(
        unet: UNet2DConditionModel,
        scheduler: DDPMScheduler,
        input_vector: torch.Tensor,
        y_max: float
) -> np.ndarray:
    unet.eval()
    noise_input = torch.randn(1, 1, MOL_MATRIX_SIZE, MOL_MATRIX_SIZE).to(device)
    sample_input = input_vector.unsqueeze(0).unsqueeze(1).to(device)
    scheduler.set_timesteps(SCHEDULER_STEPS)

    for t in scheduler.timesteps:
        with torch.no_grad():
            noisy_residual = unet(noise_input, t, sample_input).sample
        previous_noisy_sample = scheduler.step(noisy_residual, t, noise_input).prev_sample
        noise_input = previous_noisy_sample

    generated_matrix = noise_input
    cropped_image = generated_matrix[:, :, 14:50, 14:50]
    generated_image = cropped_image.squeeze(0).cpu().numpy().reshape(CROP_SIZE, CROP_SIZE)
    generated_image = generated_image * y_max

    return generated_image


def plot_losses(
        train_total: list,
        test_total: list,
        train_mse: list,
        test_mse: list,
        train_cos: list,
        test_cos: list,
        steps: list,
        save: bool = True
) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(steps, train_total, 'b-', label='Train Total Loss')
    plt.plot(steps, train_mse, 'r--', label='Train MSE Loss')
    plt.plot(steps, train_cos, 'g-.', label='Train Cosine Loss')

    plt.scatter(steps, test_total, c='cyan', marker='o', label='Test Total Loss')
    plt.scatter(steps, test_mse, c='magenta', marker='^', label='Test MSE Loss')
    plt.scatter(steps, test_cos, c='yellow', marker='s', label='Test Cosine Loss')

    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Loss Value", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.yscale('log')
    plt.tight_layout()

    FIGURE_DIR = os.path.join(OUTPUT_DIR, "pretraining_figures_64")
    os.makedirs(FIGURE_DIR, exist_ok=True)

    if save:
        filename = f"training_losses.png"
        save_path = os.path.join(FIGURE_DIR, filename)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss plot saved to: {save_path}")

    plt.show()


def main():
    print("Loading and preprocessing data...")
    x, y = load_data()
    input_tensor, scaler, num_optional_props = preprocess_input(x)
    output_tensor, y_max = preprocess_output(y)

    preprocess_dir = os.path.join(OUTPUT_DIR, "pretraining_preprocess")
    os.makedirs(preprocess_dir, exist_ok=True)

    scaler_path = os.path.join(preprocess_dir, "input_scaler_mask.pkl")
    joblib.dump(scaler, scaler_path)

    y_max_csv_path = os.path.join(preprocess_dir, "y_max_mask.csv")
    pd.DataFrame({"y_max": [float(y_max)]}).to_csv(y_max_csv_path, index=False)

    train_loader, test_loader = create_datasets(input_tensor, output_tensor)

    print("Initializing model...")
    unet, scheduler, pipeline = initialize_model()

    print("Training model...")
    losses = train_model(unet, train_loader, test_loader, scheduler, num_optional_props, total_steps=TOTAL_STEPS)
    train_total, train_mse, train_cos, test_total, test_mse, test_cos, steps = losses

    plot_losses(train_total, test_total, train_mse, test_mse, train_cos, test_cos, steps, save=True)

    print("Generating sample...")
    generated_image = generate_sample(unet, scheduler, input_tensor[45], y_max)

    output_csv_path = os.path.join(OUTPUT_DIR, "conditional_generated_matrix.csv")
    pd.DataFrame(generated_image).to_csv(output_csv_path, index=False, header=False)
    print(f"Generated matrix saved to {output_csv_path}")

    output_matrix_first = output_tensor[0].cpu().numpy()[0, 14:50, 14:50] * y_max
    mae = np.mean(np.abs(generated_image - output_matrix_first))
    mse = np.mean((generated_image - output_matrix_first) ** 2)
    print(f"Generated matrix MAE: {mae}, Generated matrix MSE: {mse}")


if __name__ == "__main__":
    main()