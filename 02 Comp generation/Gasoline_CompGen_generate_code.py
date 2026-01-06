import os
import torch
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DConditionModel
import matplotlib.pyplot as plt
import time
from Bulk_prop_cal_gpu import (
    mixing_rule_density,
    mixing_rule_element_content_wt,
    mixing_rule_GC_MS_type_diesel,
    mixing_rule_distillation_cum_fraction_d86,
    mixing_rule_gasoline_PIONA,
    process_mol_data,
    mixing_rule_cetane_number,
    mixing_rule_freeze_point
)
from Mol_fraction_conversion import fraction_conversion
from typing import Tuple, Optional, List, Dict, Union

BASE_DIR = "./02 Comp generation"

FILE_PATHS = {
    'mol_index': os.path.join(BASE_DIR, "Fuel_mol_index.csv"),
    'mol_info': os.path.join(BASE_DIR, "Fuel_mol_infor_list.csv"),
    'checkpoint': os.path.join(BASE_DIR, "load_model/model_step_500000.pth"),
    'preprocess': os.path.join(BASE_DIR, "load_model"),
    'output': os.path.join(BASE_DIR, "generated_matrices")
}

MOL_COLUMNS = [
    'tb', 'vm_298K', 'C_number', 'H_number', 'S_number',
    'N_number', 'O_number', 'mw', 'aromatic_ring_number',
    'naphthenic_ring_number', 'thiophene_ring_number',
    'total_ring_number', 'gasoline_piona_type'
]

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

MODEL_PARAMS = {
    'batch_size': 64,
    'target_size': 64,
    'mol_matrix_size': 64,
    'crop_size': 36,
    'num_timesteps': 1000,
    'scheduler_steps': 50
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_molecular_data() -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    mol_index_df = pd.read_csv(FILE_PATHS['mol_index'], header=None)
    mol_index_tensor = torch.tensor(mol_index_df.values, dtype=torch.float32)

    mol_info_df = pd.read_csv(FILE_PATHS['mol_info'])
    mol_tensors = {}

    for column in MOL_COLUMNS:
        mol_tensors[column] = torch.tensor(mol_info_df[column].values, dtype=torch.float32)

    return mol_index_tensor, mol_tensors


def preprocess_input(properties_list: pd.Series, scaler: Optional[MinMaxScaler] = None) \
        -> Tuple[torch.Tensor, MinMaxScaler, int]:
    batch_data = []

    for properties in properties_list:
        key_data = np.array([properties.get(prop, 0.0) for prop in KEY_PROPERTIES]).reshape(1, -1)
        key_mask = np.ones_like(key_data)

        optional_data = []
        optional_mask = []

        for prop in OPTIONAL_PROPS:
            value = properties.get(prop, 0)

            if value != 0 and not pd.isna(value):
                optional_data.append(value)
                optional_mask.append(1)
            else:
                optional_data.append(0)
                optional_mask.append(0)

        optional_data = np.array(optional_data).reshape(1, -1)
        optional_mask = np.array(optional_mask).reshape(1, -1)

        combined_data = np.hstack([key_data, optional_data, key_mask, optional_mask])
        batch_data.append(combined_data)

    batch_data = np.vstack(batch_data)

    if scaler is None:
        scaler = MinMaxScaler()
        x_scaled = scaler.fit_transform(batch_data)
    else:
        x_scaled = scaler.transform(batch_data)

    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    padding_size = MODEL_PARAMS['target_size'] - x_tensor.size(1)

    if padding_size > 0:
        padding = torch.zeros(x_tensor.size(0), padding_size, dtype=torch.float32)
        x_tensor = torch.cat((x_tensor, padding), dim=1)

    num_optional_props = len(OPTIONAL_PROPS)

    return x_tensor, scaler, num_optional_props


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * (torch.pi / 2)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999).detach().to(torch.float32)


def load_trained_model() -> Tuple[UNet2DConditionModel, DDPMScheduler, DDPMPipeline]:
    betas = cosine_beta_schedule(MODEL_PARAMS['num_timesteps'])

    unet = UNet2DConditionModel(
        sample_size=MODEL_PARAMS['mol_matrix_size'],
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        cross_attention_dim=MODEL_PARAMS['target_size']
    )

    checkpoint = torch.load(FILE_PATHS['checkpoint'], map_location=device)
    unet.load_state_dict(checkpoint['model_state_dict'])
    unet = unet.to(device)
    unet.eval()

    scheduler = DDPMScheduler(
        num_train_timesteps=MODEL_PARAMS['num_timesteps'],
        beta_schedule="custom",
        trained_betas=betas.clone().detach().numpy())
    pipeline = DDPMPipeline(unet=unet, scheduler=scheduler)
    pipeline.to(device)

    return unet, scheduler, pipeline


def calculate_properties_from_matrix_batch(
        mol_matrix_batch: torch.Tensor,
        mol_index_tensor: torch.Tensor,
        mol_tensors: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    batch_size = mol_matrix_batch.size(0)
    mol_index_tensor = mol_index_tensor.unsqueeze(0).unsqueeze(0)
    mol_index_tensor = mol_index_tensor.repeat(batch_size, 1, 1, 1)

    x_mass = process_mol_data(mol_index_tensor, mol_matrix_batch, device=device)

    x_molar = fraction_conversion(
        mol_tensors['mw'],
        mol_tensors['vm_298K'],
        x_mass, 'mass', 'molar',
        device=device
    )

    x_volume = fraction_conversion(
        mol_tensors['mw'],
        mol_tensors['vm_298K'],
        x_molar, 'molar',
        'volume',
        device=device
    )

    properties = {}

    properties['density_293K'] = mixing_rule_density(
        mol_tensors['mw'],
        mol_tensors['vm_298K'],
        x_volume, T=293.15,
        device=device
    )

    C, H, S, N, O = mixing_rule_element_content_wt(
        mol_tensors['C_number'],
        mol_tensors['H_number'],
        mol_tensors['S_number'],
        mol_tensors['N_number'],
        mol_tensors['O_number'],
        mol_tensors['mw'],
        x_mass,
        device=device
    )

    properties['C_weight_percentage'] = C
    properties['H_weight_percentage'] = H
    properties['S_weight_percentage'] = S
    properties['N_weight_percentage'] = N

    (alkane_weight_fraction,
     one_ring_naphthene_weight_fraction,
     two_ring_naphthene_weight_fraction,
     three_ring_naphthene_weight_fraction,
     four_ring_naphthene_weight_fraction,
     five_ring_naphthene_weight_fraction,
     six_and_more_ring_naphthene_weight_fraction,
     one_ring_aromatic_weight_fraction,
     two_ring_aromatic_weight_fraction,
     three_ring_aromatic_weight_fraction,
     four_ring_aromatic_weight_fraction,
     five_ring_aromatic_weight_fraction,
     six_and_more_ring_aromatic_weight_fraction) = mixing_rule_GC_MS_type_diesel(
        mol_tensors['S_number'],
        mol_tensors['N_number'],
        mol_tensors['O_number'],
        mol_tensors['aromatic_ring_number'],
        mol_tensors['naphthenic_ring_number'],
        x_mass,
        device=device
    )

    properties['GCMS_group_P_weight_fraction'] = alkane_weight_fraction
    properties['GCMS_group_1N_weight_fraction'] = one_ring_naphthene_weight_fraction
    properties['GCMS_group_2N_weight_fraction'] = two_ring_naphthene_weight_fraction
    properties['GCMS_group_3N_weight_fraction'] = three_ring_naphthene_weight_fraction
    properties['GCMS_group_1A_weight_fraction'] = one_ring_aromatic_weight_fraction
    properties['GCMS_group_2A_weight_fraction'] = two_ring_aromatic_weight_fraction
    properties['GCMS_group_3A_weight_fraction'] = three_ring_aromatic_weight_fraction

    distillation_d86 = mixing_rule_distillation_cum_fraction_d86(
        x_volume,
        mol_tensors['tb'],
        device=device
    )

    properties['distillation_cum_fraction_d86_ibpv'] = distillation_d86[0]
    properties['distillation_cum_fraction_d86_10v'] = distillation_d86[2]
    properties['distillation_cum_fraction_d86_30v'] = distillation_d86[6]
    properties['distillation_cum_fraction_d86_50v'] = distillation_d86[10]
    properties['distillation_cum_fraction_d86_70v'] = distillation_d86[14]
    properties['distillation_cum_fraction_d86_90v'] = distillation_d86[18]
    properties['distillation_cum_fraction_d86_fbpv'] = distillation_d86[20]

    (gasoline_P_mass_fraction,
     gasoline_I_mass_fraction,
     gasoline_O_mass_fraction,
     gasoline_N_mass_fraction,
     gasoline_A_mass_fraction) = mixing_rule_gasoline_PIONA(
        x_mass,
        mol_tensors['gasoline_piona_type'],
        device=device
    )

    properties['gasoline_P_mass_fraction'] = gasoline_P_mass_fraction
    properties['gasoline_I_mass_fraction'] = gasoline_I_mass_fraction
    properties['gasoline_O_mass_fraction'] = gasoline_O_mass_fraction
    properties['gasoline_N_mass_fraction'] = gasoline_N_mass_fraction
    properties['gasoline_A_mass_fraction'] = gasoline_A_mass_fraction

    return properties


def generate_composition_from_properties(
        properties_list: Dict[str, float],
        mol_index_tensor: torch.Tensor,
        mol_tensors: Dict[str, torch.Tensor]
) -> np.ndarray:
    os.makedirs(FILE_PATHS['output'], exist_ok=True)

    batch_size = len(properties_list)

    for i, properties in enumerate(properties_list):
        missing_keys = set(KEY_PROPERTIES) - set(properties.keys())
        if missing_keys:
            print(f"Warning: Sample {i + 1} missing required properties: {missing_keys}")

        for prop in OPTIONAL_PROPS:
            if prop not in properties:
                properties[prop] = 0

    scaler_path = os.path.join(FILE_PATHS['preprocess'], "input_scaler_mask.pkl")
    y_max_path = os.path.join(FILE_PATHS['preprocess'], "y_max_mask.csv")

    loaded_scaler = joblib.load(scaler_path)
    y_max_df_loaded = pd.read_csv(y_max_path)
    y_max_loaded = y_max_df_loaded["y_max"].iloc[0]

    prop_series = pd.Series(properties)
    input_tensor, scale, num_optional_props = preprocess_input(properties_list, loaded_scaler)

    torch.manual_seed(42)
    initial_noise = torch.randn(batch_size, 1, MODEL_PARAMS['mol_matrix_size'], MODEL_PARAMS['mol_matrix_size'],
                                device=device)

    print("Loading model...")
    unet, scheduler, pipeline = load_trained_model()

    start_time = time.time()

    print("Generating composition matrix...")
    with torch.no_grad():
        noise_input = initial_noise
        scheduler.set_timesteps(MODEL_PARAMS['scheduler_steps'])
        for t in scheduler.timesteps:
            noisy_residual = unet(noise_input, t, input_tensor.unsqueeze(1).to(device)).sample
            noise_input = scheduler.step(noisy_residual, t, noise_input).prev_sample

        cropped_image = noise_input[:, :, 14:50, 14:50]
        generated_images = cropped_image * y_max_loaded

    calculated_properties = calculate_properties_from_matrix_batch(
        generated_images, mol_index_tensor, mol_tensors
    )

    elapsed = time.time() - start_time
    print(f"Fine-tuning completed in {elapsed:.2f}s.")

    mol_matrices = []
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    properties_df = pd.DataFrame()

    properties_df['sample_id'] = [f"sample_{i + 1}" for i in range(batch_size)]

    for prop_name, prop_values in calculated_properties.items():
        properties_df[prop_name] = prop_values.cpu().numpy().flatten()

    properties_filename = f"calculated_properties_{timestamp}.csv"
    properties_path = os.path.join(FILE_PATHS['output'], properties_filename)
    properties_df.to_csv(properties_path, index=False)
    print(f"Saved calculated properties to: {properties_path}")

    for i in range(batch_size):
        mol_matrix = generated_images[i].squeeze().cpu().numpy()
        mol_matrices.append(mol_matrix)

        filename = f"generated_matrix_{timestamp}_{i + 1}.csv"
        output_path = os.path.join(FILE_PATHS['output'], filename)
        pd.DataFrame(mol_matrix).to_csv(output_path, index=False, header=False)
        print(f"Saved matrix {i + 1} to: {output_path}")

    return mol_matrix


if __name__ == "__main__":
    input_properties_list = [
        {
            'distillation_cum_fraction_d86_ibpv': 312.25,
            'distillation_cum_fraction_d86_10v': 331.25,
            'distillation_cum_fraction_d86_30v': 344.95,
            'distillation_cum_fraction_d86_50v': 372.05,
            'distillation_cum_fraction_d86_70v': 408.55,
            'distillation_cum_fraction_d86_90v': 443.65,
            'distillation_cum_fraction_d86_fbpv': 467.85,

            'density_293K': 0.7447,
            'RON': 92,
            'C_weight_percentage': 87.06,
            'H_weight_percentage': 12.94,
            'S_weight_percentage': 0.00024,
            'N_weight_percentage': 0.000242,
            'GCMS_group_P_weight_fraction': 0.54,
            'GCMS_group_1N_weight_fraction': 0.09,
            'GCMS_group_2N_weight_fraction': 0.000001,
            'GCMS_group_3N_weight_fraction': 0.000001,
            'GCMS_group_1A_weight_fraction': 0.37,
            'GCMS_group_2A_weight_fraction': 0.000001,
            'GCMS_group_3A_weight_fraction': 0.000001,
            'gasoline_P_mass_fraction': 0.03,
            'gasoline_I_mass_fraction': 0.22,
            'gasoline_O_mass_fraction': 0.29,
            'gasoline_N_mass_fraction': 0.09,
            'gasoline_A_mass_fraction': 0.37,
        },
        {
            'distillation_cum_fraction_d86_ibpv': 315.15,
            'distillation_cum_fraction_d86_10v': 329.15,
            'distillation_cum_fraction_d86_30v': 350.15,
            'distillation_cum_fraction_d86_50v': 373.65,
            'distillation_cum_fraction_d86_70v': 416.15,
            'distillation_cum_fraction_d86_90v': 454.65,
            'distillation_cum_fraction_d86_fbpv': 473.15,

            'density_293K': 0.7417,
            'reid_vapor_pressure': 66.5,
            'RON': 92.4,
            'C_weight_percentage': 87.0277,
            'H_weight_percentage': 12.9731,
            'S_weight_percentage': 0.00062,
            'N_weight_percentage': 0.000164,
            'GCMS_group_P_weight_fraction': 0.55,
            'GCMS_group_1N_weight_fraction': 0.09,
            'GCMS_group_2N_weight_fraction': 0.000001,
            'GCMS_group_3N_weight_fraction': 0.000001,
            'GCMS_group_1A_weight_fraction': 0.37,
            'GCMS_group_2A_weight_fraction': 0.000001,
            'GCMS_group_3A_weight_fraction': 0.000001,
            'gasoline_P_mass_fraction': 0.04,
            'gasoline_I_mass_fraction': 0.20,
            'gasoline_O_mass_fraction': 0.31,
            'gasoline_N_mass_fraction': 0.09,
            'gasoline_A_mass_fraction': 0.37,
        },
        {
            'distillation_cum_fraction_d86_ibpv': 308.15,
            'distillation_cum_fraction_d86_10v': 326.65,
            'distillation_cum_fraction_d86_30v': 345.15,
            'distillation_cum_fraction_d86_50v': 374.65,
            'distillation_cum_fraction_d86_70v': 410.15,
            'distillation_cum_fraction_d86_90v': 442.15,
            'distillation_cum_fraction_d86_fbpv': 460.65,

            'density_293K': 0.7327,
            'reid_vapor_pressure': 56.5,
            'RON': 88.7,
            'C_weight_percentage': 86.51,
            'H_weight_percentage': 13.48,
            'S_weight_percentage': 0.0000001,
            'N_weight_percentage': 0.0000039,
            'GCMS_group_P_weight_fraction': 0.62,
            'GCMS_group_1N_weight_fraction': 0.10,
            'GCMS_group_2N_weight_fraction': 0.000001,
            'GCMS_group_3N_weight_fraction': 0.000001,
            'GCMS_group_1A_weight_fraction': 0.29,
            'GCMS_group_2A_weight_fraction': 0.000001,
            'GCMS_group_3A_weight_fraction': 0.000001,
            'gasoline_P_mass_fraction': 0.10,
            'gasoline_I_mass_fraction': 0.23,
            'gasoline_O_mass_fraction': 0.29,
            'gasoline_N_mass_fraction': 0.10,
            'gasoline_A_mass_fraction': 0.29,
        },
        {
            'distillation_cum_fraction_d86_ibpv': 315.15,
            'distillation_cum_fraction_d86_10v': 345.15,
            'distillation_cum_fraction_d86_30v': 364.65,
            'distillation_cum_fraction_d86_50v': 383.15,
            'distillation_cum_fraction_d86_70v': 405.15,
            'distillation_cum_fraction_d86_90v': 434.15,
            'distillation_cum_fraction_d86_fbpv': 464.65,

            'density_293K': 0.7716,
            'reid_vapor_pressure': 18,
            'RON': 91.1,
            'C_weight_percentage': 88.15,
            'H_weight_percentage': 11.84,
            'S_weight_percentage': 0.00003,
            'N_weight_percentage': 0.000018,
            'GCMS_group_P_weight_fraction': 0.34,
            'GCMS_group_1N_weight_fraction': 0.004,
            'GCMS_group_2N_weight_fraction': 0.000001,
            'GCMS_group_3N_weight_fraction': 0.000001,
            'GCMS_group_1A_weight_fraction': 0.62,
            'GCMS_group_2A_weight_fraction': 0.000001,
            'GCMS_group_3A_weight_fraction': 0.000001,
            'gasoline_P_mass_fraction': 0.10,
            'gasoline_I_mass_fraction': 0.24,
            'gasoline_O_mass_fraction': 0.000001,
            'gasoline_N_mass_fraction': 0.04,
            'gasoline_A_mass_fraction': 0.62,
        },
        {
            'distillation_cum_fraction_d86_ibpv': 314.45,
            'distillation_cum_fraction_d86_10v': 343.65,
            'distillation_cum_fraction_d86_30v': 360.25,
            'distillation_cum_fraction_d86_50v': 372.85,
            'distillation_cum_fraction_d86_70v': 384.75,
            'distillation_cum_fraction_d86_90v': 397.95,
            'distillation_cum_fraction_d86_fbpv': 414.85,

            'density_293K': 0.7074,
            'RON': 45,
            'C_weight_percentage': 84.82,
            'H_weight_percentage': 15.18,
            'S_weight_percentage': 0.000018,
            'N_weight_percentage': 0.0000008,
            'GCMS_group_P_weight_fraction': 0.62,
            'GCMS_group_1N_weight_fraction': 0.32,
            'GCMS_group_2N_weight_fraction': 0.000001,
            'GCMS_group_3N_weight_fraction': 0.000001,
            'GCMS_group_1A_weight_fraction': 0.05,
            'GCMS_group_2A_weight_fraction': 0.000001,
            'GCMS_group_3A_weight_fraction': 0.000001,
            'gasoline_P_mass_fraction': 0.40,
            'gasoline_I_mass_fraction': 0.22,
            'gasoline_O_mass_fraction': 0.000001,
            'gasoline_N_mass_fraction': 0.32,
            'gasoline_A_mass_fraction': 0.05,
        },
        {
            'distillation_cum_fraction_d86_ibpv': 330.65,
            'distillation_cum_fraction_d86_10v': 367.85,
            'distillation_cum_fraction_d86_30v': 386.15,
            'distillation_cum_fraction_d86_50v': 400.65,
            'distillation_cum_fraction_d86_70v': 426.35,
            'distillation_cum_fraction_d86_90v': 433.65,
            'distillation_cum_fraction_d86_fbpv': 450.15,

            'density_293K': 0.7251,
            'RON': 39.6,
            'C_weight_percentage': 85.11,
            'H_weight_percentage': 11.49,
            'S_weight_percentage': 0.00002,
            'N_weight_percentage': 0.000001,
            'GCMS_group_P_weight_fraction': 0.67,
            'GCMS_group_1N_weight_fraction': 0.25,
            'GCMS_group_2N_weight_fraction': 0.000001,
            'GCMS_group_3N_weight_fraction': 0.000001,
            'GCMS_group_1A_weight_fraction': 0.008,
            'GCMS_group_2A_weight_fraction': 0.000001,
            'GCMS_group_3A_weight_fraction': 0.000001,
            'gasoline_P_mass_fraction': 0.43,
            'gasoline_I_mass_fraction': 0.24,
            'gasoline_O_mass_fraction': 0.000001,
            'gasoline_N_mass_fraction': 0.25,
            'gasoline_A_mass_fraction': 0.008,
        },
        {
            'distillation_cum_fraction_d86_ibpv': 304.05,
            'distillation_cum_fraction_d86_10v': 324.55,
            'distillation_cum_fraction_d86_30v': 356.75,
            'distillation_cum_fraction_d86_50v': 374.15,
            'distillation_cum_fraction_d86_70v': 396.75,
            'distillation_cum_fraction_d86_90v': 442.65,
            'distillation_cum_fraction_d86_fbpv': 471.75,

            'density_293K': 0.7297,
            'RON': 88.9,
            'C_weight_percentage': 86.20,
            'H_weight_percentage': 13.79,
            'S_weight_percentage': 0.00006,
            'N_weight_percentage': 0.00001,
            'GCMS_group_P_weight_fraction': 0.59,
            'GCMS_group_1N_weight_fraction': 0.16,
            'GCMS_group_2N_weight_fraction': 0.000001,
            'GCMS_group_3N_weight_fraction': 0.000001,
            'GCMS_group_1A_weight_fraction': 0.26,
            'GCMS_group_2A_weight_fraction': 0.000001,
            'GCMS_group_3A_weight_fraction': 0.000001,
            'gasoline_P_mass_fraction': 0.06,
            'gasoline_I_mass_fraction': 0.31,
            'gasoline_O_mass_fraction': 0.22,
            'gasoline_N_mass_fraction': 0.16,
            'gasoline_A_mass_fraction': 0.26,
        },
        {
            'distillation_cum_fraction_d86_ibpv': 307.25,
            'distillation_cum_fraction_d86_10v': 331.15,
            'distillation_cum_fraction_d86_30v': 359.85,
            'distillation_cum_fraction_d86_50v': 376.95,
            'distillation_cum_fraction_d86_70v': 398.25,
            'distillation_cum_fraction_d86_90v': 438.45,
            'distillation_cum_fraction_d86_fbpv': 470.35,

            'density_293K': 0.7351,
            'RON': 85.8,
            'C_weight_percentage': 86.14,
            'H_weight_percentage': 13.85,
            'S_weight_percentage': 0.000052,
            'N_weight_percentage': 0.000007,
            'GCMS_group_P_weight_fraction': 0.55,
            'GCMS_group_1N_weight_fraction': 0.20,
            'GCMS_group_2N_weight_fraction': 0.000001,
            'GCMS_group_3N_weight_fraction': 0.000001,
            'GCMS_group_1A_weight_fraction': 0.25,
            'GCMS_group_2A_weight_fraction': 0.000001,
            'GCMS_group_3A_weight_fraction': 0.000001,
            'gasoline_P_mass_fraction': 0.06,
            'gasoline_I_mass_fraction': 0.32,
            'gasoline_O_mass_fraction': 0.17,
            'gasoline_N_mass_fraction': 0.20,
            'gasoline_A_mass_fraction': 0.25,
        },
        {
            'distillation_cum_fraction_d86_ibpv': 306.45,
            'distillation_cum_fraction_d86_10v': 326.95,
            'distillation_cum_fraction_d86_30v': 356.55,
            'distillation_cum_fraction_d86_50v': 371.75,
            'distillation_cum_fraction_d86_70v': 396.35,
            'distillation_cum_fraction_d86_90v': 442.25,
            'distillation_cum_fraction_d86_fbpv': 472.95,

            'density_293K': 0.7324,
            'C_weight_percentage': 86.29,
            'H_weight_percentage': 13.71,
            'S_weight_percentage': 0.0000047,
            'N_weight_percentage': 0,
            'GCMS_group_P_weight_fraction': 0.55,
            'GCMS_group_1N_weight_fraction': 0.16,
            'GCMS_group_2N_weight_fraction': 0.000001,
            'GCMS_group_3N_weight_fraction': 0.000001,
            'GCMS_group_1A_weight_fraction': 0.28,
            'GCMS_group_2A_weight_fraction': 0.000001,
            'GCMS_group_3A_weight_fraction': 0.000001,
            'gasoline_P_mass_fraction': 0.05,
            'gasoline_I_mass_fraction': 0.32,
            'gasoline_O_mass_fraction': 0.18,
            'gasoline_N_mass_fraction': 0.16,
            'gasoline_A_mass_fraction': 0.28,
        },
        {
            'distillation_cum_fraction_d86_ibpv': 310.35,
            'distillation_cum_fraction_d86_10v': 330.05,
            'distillation_cum_fraction_d86_30v': 355.65,
            'distillation_cum_fraction_d86_50v': 367.15,
            'distillation_cum_fraction_d86_70v': 393.75,
            'distillation_cum_fraction_d86_90v': 439.65,
            'distillation_cum_fraction_d86_fbpv': 469.55,

            'density_293K': 0.7421,
            'C_weight_percentage': 87.15,
            'H_weight_percentage': 12.84,
            'S_weight_percentage': 0.00000643,
            'N_weight_percentage': 0,
            'GCMS_group_P_weight_fraction': 0.43,
            'GCMS_group_1N_weight_fraction': 0.12,
            'GCMS_group_2N_weight_fraction': 0.000001,
            'GCMS_group_3N_weight_fraction': 0.000001,
            'GCMS_group_1A_weight_fraction': 0.45,
            'GCMS_group_2A_weight_fraction': 0.000001,
            'GCMS_group_3A_weight_fraction': 0.000001,
            'gasoline_P_mass_fraction': 0.09,
            'gasoline_I_mass_fraction': 0.23,
            'gasoline_O_mass_fraction': 0.11,
            'gasoline_N_mass_fraction': 0.12,
            'gasoline_A_mass_fraction': 0.45,
        },
        {
            'distillation_cum_fraction_d86_ibpv': 308.15,
            'distillation_cum_fraction_d86_10v': 329.75,
            'distillation_cum_fraction_d86_30v': 344.75,
            'distillation_cum_fraction_d86_50v': 370.75,
            'distillation_cum_fraction_d86_70v': 412.95,
            'distillation_cum_fraction_d86_90v': 440.05,
            'distillation_cum_fraction_d86_fbpv': 474.05,

            'density_293K': 0.7533,
            'C_weight_percentage': 87.42,
            'H_weight_percentage': 12.58,
            'S_weight_percentage': 0.000001,
            'N_weight_percentage': 0.000001,
            'GCMS_group_P_weight_fraction': 0.39,
            'GCMS_group_1N_weight_fraction': 0.15,
            'GCMS_group_2N_weight_fraction': 0.000001,
            'GCMS_group_3N_weight_fraction': 0.000001,
            'GCMS_group_1A_weight_fraction': 0.46,
            'GCMS_group_2A_weight_fraction': 0.000001,
            'GCMS_group_3A_weight_fraction': 0.000001,
            'gasoline_P_mass_fraction': 0.06,
            'gasoline_I_mass_fraction': 0.20,
            'gasoline_O_mass_fraction': 0.13,
            'gasoline_N_mass_fraction': 0.15,
            'gasoline_A_mass_fraction': 0.46,
        },
        {
            'distillation_cum_fraction_d86_ibpv': 312.15,
            'distillation_cum_fraction_d86_10v': 331.05,
            'distillation_cum_fraction_d86_30v': 356.85,
            'distillation_cum_fraction_d86_50v': 378.45,
            'distillation_cum_fraction_d86_70v': 395.45,
            'distillation_cum_fraction_d86_90v': 447.85,
            'distillation_cum_fraction_d86_fbpv': 473.35,

            'density_293K': 0.7393,
            'C_weight_percentage': 86.50,
            'H_weight_percentage': 13.50,
            'S_weight_percentage': 0.000012,
            'N_weight_percentage': 0.000042,
            'GCMS_group_P_weight_fraction': 0.55,
            'GCMS_group_1N_weight_fraction': 0.15,
            'GCMS_group_2N_weight_fraction': 0.000001,
            'GCMS_group_3N_weight_fraction': 0.000001,
            'GCMS_group_1A_weight_fraction': 0.29,
            'GCMS_group_2A_weight_fraction': 0.000001,
            'GCMS_group_3A_weight_fraction': 0.000001,
            'gasoline_P_mass_fraction': 0.04,
            'gasoline_I_mass_fraction': 0.28,
            'gasoline_O_mass_fraction': 0.23,
            'gasoline_N_mass_fraction': 0.15,
            'gasoline_A_mass_fraction': 0.29,
        },
        {
            'distillation_cum_fraction_d86_ibpv': 309.35,
            'distillation_cum_fraction_d86_10v': 328.05,
            'distillation_cum_fraction_d86_30v': 354.05,
            'distillation_cum_fraction_d86_50v': 365.75,
            'distillation_cum_fraction_d86_70v': 392.05,
            'distillation_cum_fraction_d86_90v': 438.15,
            'distillation_cum_fraction_d86_fbpv': 467.95,

            'density_293K': 0.7409,
            'C_weight_percentage': 87.09,
            'H_weight_percentage': 12.91,
            'S_weight_percentage': 0.0000066,
            'N_weight_percentage': 0.000001,
            'GCMS_group_P_weight_fraction': 0.45,
            'GCMS_group_1N_weight_fraction': 0.11,
            'GCMS_group_2N_weight_fraction': 0.000001,
            'GCMS_group_3N_weight_fraction': 0.000001,
            'GCMS_group_1A_weight_fraction': 0.43,
            'GCMS_group_2A_weight_fraction': 0.000001,
            'GCMS_group_3A_weight_fraction': 0.000001,
            'gasoline_P_mass_fraction': 0.09,
            'gasoline_I_mass_fraction': 0.24,
            'gasoline_O_mass_fraction': 0.12,
            'gasoline_N_mass_fraction': 0.11,
            'gasoline_A_mass_fraction': 0.43,
        },
        {
            'distillation_cum_fraction_d86_ibpv': 339.64,
            'distillation_cum_fraction_d86_10v': 354.75,
            'distillation_cum_fraction_d86_30v': 367.34,
            'distillation_cum_fraction_d86_50v': 378.84,
            'distillation_cum_fraction_d86_70v': 393.11,
            'distillation_cum_fraction_d86_90v': 410.85,
            'distillation_cum_fraction_d86_fbpv': 427.65,

            'density_293K': 0.7447,
            'C_weight_percentage': 85.28,
            'H_weight_percentage': 14.71,
            'S_weight_percentage': 0.000001,
            'N_weight_percentage': 0.000001,
            'GCMS_group_P_weight_fraction': 0.5528,
            'GCMS_group_1N_weight_fraction': 0.3378,
            'GCMS_group_2N_weight_fraction': 0.000001,
            'GCMS_group_3N_weight_fraction': 0.000001,
            'GCMS_group_1A_weight_fraction': 0.1095,
            'GCMS_group_2A_weight_fraction': 0.000001,
            'GCMS_group_3A_weight_fraction': 0.000001,
            'gasoline_P_mass_fraction': 0.2529,
            'gasoline_I_mass_fraction': 0.2999,
            'gasoline_O_mass_fraction': 0.00001,
            'gasoline_N_mass_fraction': 0.3378,
            'gasoline_A_mass_fraction': 0.1095,
        }
    ]

    print("Loading molecular data...")
    mol_index_tensor, mol_tensors = load_molecular_data()

    print("Generating composition matrix...")
    pred_matrix = generate_composition_from_properties(
        properties_list=input_properties_list,
        mol_index_tensor=mol_index_tensor,
        mol_tensors=mol_tensors
    )

    print("Generation complete! Matrix dimensions:", pred_matrix.shape)
